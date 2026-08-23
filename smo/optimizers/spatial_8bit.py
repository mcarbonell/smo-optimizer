"""
smo/optimizers/spatial_8bit.py — SMO-8bit (Star Mode)

The ultimate evolution of the SuperMario Optimizer.
Combines:
1. Spatial Compression (SMO): Reduces resolution of states by k_ratio.
2. Block-wise Quantization: Stores the remaining coefficients as 8-bit integers.

Persistent optimizer-state footprint: ~2% of standard AdamW (see
smo/optimizers/spatial.py for the persistent-vs-resident memory distinction).

Only 2D gradients with both dims >= 32 are compressed; other tensors
(e.g. 4D conv weights, 1D biases) fall back to dense Adam moments.

🎮 "It's-a me, ultra-optimizer!" - Star Mode Active
"""

import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from ._spatial_utils import compress_2d, compress_2d_pair, upsample_2d_pair

class SMO8bit(Optimizer):
    """
    Super Mario Optimizer - 8-bit Quantized Variant.
    
    This is the most memory-efficient version of SMO. 
    It compresses the state spatially AND quantizes it to 8 bits.
    
    Args:
        params: model.parameters()
        lr: learning rate (default: 1e-3)
        k_ratio: Spatial compression ratio (default: 0.25)
        block_size: Size of quantization blocks (default: 64)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25, block_size=64):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")
        if block_size <= 0:
            raise ValueError(f"Invalid block_size: {block_size}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k_ratio=k_ratio, block_size=block_size)
        super(SMO8bit, self).__init__(params, defaults)

    @staticmethod
    def _compress_2d(tensor, target_shape):
        return compress_2d(tensor, target_shape)

    def _quantize_blockwise(self, data, block_size):
        """Quantizes a tensor to 8-bit block-wise."""
        orig_shape = data.shape
        flat_data = data.flatten()
        n = flat_data.numel()
        pad_size = (block_size - (n % block_size)) % block_size
        if pad_size > 0:
            flat_data = F.pad(flat_data, (0, pad_size))
        
        blocks = flat_data.view(-1, block_size)
        scales = blocks.abs().max(dim=1, keepdim=True)[0]
        scales = scales.clamp(min=1e-12)
        q_blocks = (blocks / scales * 127).round().to(torch.int8)
        return q_blocks, scales, orig_shape

    def _dequantize_blockwise(self, q_blocks, scales, orig_shape, numel=None, dtype=None):
        """Dequantizes an 8-bit block-wise tensor.

        Args:
            dtype: Target dtype for the output. Defaults to float32; pass
                ``grad.dtype`` so half-precision parameters receive moments
                in a matching dtype (in-place ops require matching dtypes).
        """
        target_dtype = dtype if dtype is not None else torch.float32
        blocks = q_blocks.to(target_dtype) * (scales.to(target_dtype) / 127.0)
        flat_data = blocks.flatten()
        if numel is None:
            numel = math.prod(orig_shape)
        return flat_data[:numel].view(orig_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('SMO8bit does not support sparse gradients.')
                state = self.state[p]
                k_ratio = group['k_ratio']
                block_size = group['block_size']
                # Per-param-group opt-out: {"compress": False} forces dense Adam moments
                use_compression = group.get('compress', True)

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    if use_compression and grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        state['orig_shape'] = grad.shape
                        new_h = max(1, int(grad.shape[0] * k_ratio))
                        new_w = max(1, int(grad.shape[1] * k_ratio))
                        comp_shape = (new_h, new_w)
                        state['comp_numel'] = new_h * new_w
                        
                        # Initialize states as quantized
                        dummy = torch.zeros(comp_shape, dtype=grad.dtype, device=grad.device)
                        m_q, m_s, _ = self._quantize_blockwise(dummy, block_size)
                        v_q, v_s, _ = self._quantize_blockwise(dummy, block_size)
                        
                        state['m_q'], state['m_s'] = m_q, m_s
                        state['v_q'], state['v_s'] = v_q, v_s
                        state['comp_shape'] = comp_shape
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    # 1. Dequantize current state for update (match grad dtype
                    # so the final in-place update on p is dtype-consistent)
                    m = self._dequantize_blockwise(state['m_q'], state['m_s'], state['comp_shape'], state['comp_numel'], dtype=grad.dtype)
                    v = self._dequantize_blockwise(state['v_q'], state['v_s'], state['comp_shape'], state['comp_numel'], dtype=grad.dtype)
                    
                    # 2. Compress gradient
                    g_comp, g_sq_comp = compress_2d_pair(grad, grad.square(), state['comp_shape'])
                    
                    # 3. Update moments (in float32)
                    m.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                    
                    v.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)
                    
                    # 4. Upsample for weight update BEFORE re-quantizing
                    m_rec, v_rec = upsample_2d_pair(m, v, state['orig_shape'])
                    v_rec = torch.clamp(v_rec, min=0.0)
                    
                    # 5. Re-quantize and store
                    state['m_q'], state['m_s'], _ = self._quantize_blockwise(m, block_size)
                    state['v_q'], state['v_s'], _ = self._quantize_blockwise(v, block_size)
                    
                    # 6. Free temporary float32 tensors immediately to avoid VRAM spikes
                    del m, v
                else:
                    # Fallback for 1D/small tensors
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec = state['exp_avg']
                    v_rec = state['exp_avg_sq']

                # Standard Adam update logic
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)

        return loss

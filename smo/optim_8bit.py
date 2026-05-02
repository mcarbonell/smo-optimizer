"""
smo/optim_8bit.py — SMO-8bit (Star Mode)

The ultimate evolution of the SuperMario Optimizer.
Combines:
1. Spatial Compression (SMO): Reduces resolution of states by k_ratio.
2. Block-wise Quantization: Stores the remaining coefficients as 8-bit integers.

Memory footprint: ~2% of standard AdamW.
🎮 "It's-a me, ultra-optimizer!" - Star Mode Active
"""

import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

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

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k_ratio=k_ratio, block_size=block_size)
        super(SMO8bit, self).__init__(params, defaults)

    def _quantize_blockwise(self, data, block_size):
        """Quantizes a tensor to 8-bit block-wise."""
        # Flatten and pad to be divisible by block_size
        orig_shape = data.shape
        flat_data = data.flatten()
        n = flat_data.numel()
        pad_size = (block_size - (n % block_size)) % block_size
        if pad_size > 0:
            flat_data = F.pad(flat_data, (0, pad_size))
        
        # Reshape into blocks
        blocks = flat_data.view(-1, block_size)
        
        # Calculate scales (max absolute value per block)
        scales = blocks.abs().max(dim=1, keepdim=True)[0]
        scales = scales.clamp(min=1e-12)
        
        # Quantize to int8: map [-scale, scale] to [-127, 127]
        q_blocks = (blocks / scales * 127).round().to(torch.int8)
        
        return q_blocks, scales, orig_shape

    def _dequantize_blockwise(self, q_blocks, scales, orig_shape):
        """Dequantizes an 8-bit block-wise tensor."""
        # Dequantize: map [-127, 127] back to [-scale, scale]
        blocks = q_blocks.to(torch.float32) * (scales / 127.0)
        
        # Flatten and restore original shape
        flat_data = blocks.flatten()
        return flat_data[:math.prod(orig_shape)].view(orig_shape)

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
                state = self.state[p]
                k_ratio = group['k_ratio']
                block_size = group['block_size']

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        state['orig_shape'] = grad.shape
                        new_h = max(1, int(grad.shape[0] * k_ratio))
                        new_w = max(1, int(grad.shape[1] * k_ratio))
                        comp_shape = (new_h, new_w)
                        
                        # Initialize states as quantized
                        dummy = torch.zeros(comp_shape, dtype=grad.dtype, device=grad.device)
                        m_q, m_s, _ = self._quantize_blockwise(dummy, block_size)
                        v_q, v_s, _ = self._quantize_blockwise(dummy, block_size)
                        
                        state['m_q'], state['m_s'] = m_q, m_s
                        state['v_q'], state['v_s'] = v_q, v_s
                        state['comp_shape'] = comp_shape
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    # 1. Dequantize current state for update
                    m = self._dequantize_blockwise(state['m_q'], state['m_s'], state['comp_shape'])
                    v = self._dequantize_blockwise(state['v_q'], state['v_s'], state['comp_shape'])
                    
                    # 2. Compress gradient
                    g_view = grad.unsqueeze(0).unsqueeze(0)
                    g_comp = F.adaptive_avg_pool2d(g_view, state['comp_shape']).squeeze(0).squeeze(0)
                    
                    # 3. Update moments (in float32)
                    m.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                    
                    g_sq_comp = F.adaptive_avg_pool2d((grad**2).unsqueeze(0).unsqueeze(0), state['comp_shape']).squeeze(0).squeeze(0)
                    v.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)
                    
                    # 4. Re-quantize and store
                    state['m_q'], state['m_s'], _ = self._quantize_blockwise(m, block_size)
                    state['v_q'], state['v_s'], _ = self._quantize_blockwise(v, block_size)
                    
                    # 5. Upsample for weight update
                    m_rec = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=state['orig_shape'], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = F.interpolate(v.unsqueeze(0).unsqueeze(0), size=state['orig_shape'], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = torch.clamp(v_rec, min=0.0)
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

"""
smo/optim_8bit_triton.py — SMO8bitTriton (Hybrid GPU Architecture)

Combines the best of both worlds:
1. PyTorch CuDNN for perfect 2D Spatial Pooling (Overlapping Bins).
2. Triton for Fused Bilinear Decompression & Weight Update to eliminate
   the massive memory bottleneck of creating full-resolution Momentums.
"""

import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def smo_hybrid_weight_update_kernel(
        weights_ptr,      # Pointer to the model parameters (weights)
        m_comp_ptr,       # Pointer to float32 compressed momentum
        v_comp_ptr,       # Pointer to float32 compressed variance
        
        lr,               # Learning rate
        eps,              # Adam epsilon
        bias_correction1, # 1 - beta1^t
        bias_correction2, # 1 - beta2^t
        
        H, W,             # Original 2D dimensions
        COMP_H, COMP_W,   # Compressed 2D dimensions
        N_ELEMENTS,       # Total parameter elements
        
        BLOCK_SIZE: tl.constexpr # Elements handled per program
    ):
        pid = tl.program_id(axis=0)
        elem_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        elem_mask = elem_offsets < N_ELEMENTS
        
        # 1. Map 1D element offset to 2D original coordinates
        elem_row = elem_offsets // W
        elem_col = elem_offsets % W
        
        # 2. Map original coordinates to compressed grid coordinates (floating point)
        # Force float division to avoid integer truncation issues
        scale_h = (H * 1.0) / COMP_H
        scale_w = (W * 1.0) / COMP_W
        y = (elem_row.to(tl.float32) + 0.5) / scale_h - 0.5
        x = (elem_col.to(tl.float32) + 0.5) / scale_w - 0.5
        
        # Get integer coordinates for the 4 neighbors
        y0 = tl.where(y < 0.0, 0, y.to(tl.int32))
        x0 = tl.where(x < 0.0, 0, x.to(tl.int32))
        
        y1 = y0 + 1
        x1 = x0 + 1
        
        # Clamp to grid bounds
        y0 = tl.where(y0 >= COMP_H, COMP_H - 1, y0)
        y1 = tl.where(y1 >= COMP_H, COMP_H - 1, y1)
        x0 = tl.where(x0 >= COMP_W, COMP_W - 1, x0)
        x1 = tl.where(x1 >= COMP_W, COMP_W - 1, x1)
        
        # Calculate weights for bilinear interpolation
        y_frac = y - y0.to(tl.float32)
        x_frac = x - x0.to(tl.float32)
        y_frac = tl.where(y_frac < 0.0, 0.0, tl.where(y_frac > 1.0, 1.0, y_frac))
        x_frac = tl.where(x_frac < 0.0, 0.0, tl.where(x_frac > 1.0, 1.0, x_frac))
        
        w00 = (1.0 - y_frac) * (1.0 - x_frac)
        w01 = (1.0 - y_frac) * x_frac
        w10 = y_frac * (1.0 - x_frac)
        w11 = y_frac * x_frac
        
        # 3. Calculate 1D indices for the 4 neighbors in compressed tensor
        idx00 = y0 * COMP_W + x0
        idx01 = y0 * COMP_W + x1
        idx10 = y1 * COMP_W + x0
        idx11 = y1 * COMP_W + x1
        
        # Load states for 00
        m_00 = tl.load(m_comp_ptr + idx00, mask=elem_mask, other=0.0)
        v_00 = tl.load(v_comp_ptr + idx00, mask=elem_mask, other=0.0)
        
        # Load states for 01
        m_01 = tl.load(m_comp_ptr + idx01, mask=elem_mask, other=0.0)
        v_01 = tl.load(v_comp_ptr + idx01, mask=elem_mask, other=0.0)
        
        # Load states for 10
        m_10 = tl.load(m_comp_ptr + idx10, mask=elem_mask, other=0.0)
        v_10 = tl.load(v_comp_ptr + idx10, mask=elem_mask, other=0.0)
        
        # Load states for 11
        m_11 = tl.load(m_comp_ptr + idx11, mask=elem_mask, other=0.0)
        v_11 = tl.load(v_comp_ptr + idx11, mask=elem_mask, other=0.0)
        
        # 4. Bilinear Interpolation
        m_interp = w00 * m_00 + w01 * m_01 + w10 * m_10 + w11 * m_11
        v_interp = w00 * v_00 + w01 * v_01 + w10 * v_10 + w11 * v_11
        
        # 5. Weight Update
        w = tl.load(weights_ptr + elem_offsets, mask=elem_mask)
        
        m_hat = m_interp / bias_correction1
        v_hat = v_interp / bias_correction2
        v_hat = tl.where(v_hat < 0.0, 0.0, v_hat)
        
        denom = tl.sqrt(v_hat) + eps
        step_update = (lr * m_hat) / denom
        
        w_new = w - step_update
        tl.store(weights_ptr + elem_offsets, w_new, mask=elem_mask)


class SMO8bitTriton(Optimizer):
    """
    Super Mario Optimizer 8-bit Quantized - Hybrid Triton Version.
    Uses PyTorch for complex overlapping-bin pooling and state logic,
    and Triton for Fused On-the-fly Bilinear Interpolation and Weight Updates.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25, block_size=64):
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed or no NVIDIA GPU detected.")
            
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k_ratio=k_ratio, block_size=block_size)
        super(SMO8bitTriton, self).__init__(params, defaults)

    def _quantize_blockwise(self, data, block_size):
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

    def _dequantize_blockwise(self, q_blocks, scales, orig_shape):
        blocks = q_blocks.to(torch.float32) * (scales / 127.0)
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
                    # 1. Dequantize current state (Fast & Tiny in PyTorch)
                    m = self._dequantize_blockwise(state['m_q'], state['m_s'], state['comp_shape'])
                    v = self._dequantize_blockwise(state['v_q'], state['v_s'], state['comp_shape'])
                    
                    # 2. Compress gradient using PyTorch's robust pooler
                    g_view = grad.unsqueeze(0).unsqueeze(0)
                    g_comp = F.adaptive_avg_pool2d(g_view, state['comp_shape']).squeeze(0).squeeze(0)
                    
                    # 3. Update moments (in float32)
                    m.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                    
                    g_sq_comp = F.adaptive_avg_pool2d((grad**2).unsqueeze(0).unsqueeze(0), state['comp_shape']).squeeze(0).squeeze(0)
                    v.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)
                    
                    # 4. Re-quantize and store (Fast & Tiny)
                    state['m_q'], state['m_s'], _ = self._quantize_blockwise(m, block_size)
                    state['v_q'], state['v_s'], _ = self._quantize_blockwise(v, block_size)
                    
                    # 5. TRITON FUSED UPSAMPLE AND WEIGHT UPDATE
                    # This eliminates the memory spike of creating full-res m_rec and v_rec
                    n_elements = p.numel()
                    H, W = state['orig_shape']
                    COMP_H, COMP_W = state['comp_shape']
                    
                    p_flat = p.view(-1)
                    m_flat = m.view(-1)
                    v_flat = v.view(-1)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    def grid_weight(meta):
                        return (triton.cdiv(n_elements, 1024), )
                        
                    smo_hybrid_weight_update_kernel[grid_weight](
                        p_flat, m_flat, v_flat,
                        group['lr'], group['eps'],
                        bias_correction1, bias_correction2,
                        H, W, COMP_H, COMP_W, n_elements,
                        BLOCK_SIZE=1024
                    )
                    
                else:
                    # Fallback for 1D/small tensors
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] / bias_correction1
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    p.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss

"""
smo/optim_walsh_pure.py — SMOWalshPure: Pure Spectral Optimizer

Instead of Spatial Pooling, this uses a Pure Spectral Compression approach:
1. Pad gradient to nearest power of 2.
2. Apply Fast Walsh-Hadamard Transform (FWHT) to the full gradient.
3. Keep only the low-frequency components (top-left k_ratio crop).
4. Update states on these compressed frequencies.
5. Decompress by padding with zeros and applying Inverse FWHT.
"""

import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

def fwht_1d(x):
    """
    Vectorized Fast Walsh-Hadamard Transform in 1D.
    Optimized to reduce memory allocations and improve GPU throughput.
    """
    N = x.shape[-1]
    if (N & (N - 1)) != 0:
        return x
    
    h = 1
    x = x.clone()
    while h < N:
        # Use a temporary view to perform butterfly operations without 
        # permanently changing the shape of x during the loop iterations.
        x_view = x.view(*x.shape[:-1], N // (2 * h), 2, h)
        
        # Standard butterfly: a, b = a + b, a - b
        x_0 = x_view[..., 0, :].clone()
        x_view[..., 0, :] += x_view[..., 1, :]
        x_view[..., 1, :] = x_0 - x_view[..., 1, :]
        
        h *= 2
    return x

def fwht_2d(x):
    """Fast Walsh-Hadamard Transform in 2D using separable 1D transforms."""
    # Transform rows (last dimension)
    x = fwht_1d(x)
    # Transform columns (transpose, transform last dim, transpose back)
    # Using transpose(-1, -2) is more robust for multi-dimensional tensors
    return fwht_1d(x.transpose(-1, -2)).transpose(-1, -2)

def ifwht_2d(x):
    """Inverse Fast Walsh-Hadamard Transform in 2D."""
    # FWHT is its own inverse up to a scaling factor
    res = fwht_2d(x)
    return res / (x.shape[-2] * x.shape[-1])

class SMOWalshPure(Optimizer):
    """
    Pure Spectral Walsh Optimizer using Fast Butterfly Algorithm.
    Optimized for GPU execution by minimizing host-device synchronization and allocations.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k_ratio=k_ratio)
        super().__init__(params, defaults)

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

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Optimization: Only compress larger 2D matrices
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        orig_h, orig_w = grad.shape
                        state['orig_shape'] = (orig_h, orig_w)
                        
                        # Find next power of 2 for efficient FWHT
                        pad_h = 1 << (orig_h - 1).bit_length()
                        pad_w = 1 << (orig_w - 1).bit_length()
                        state['pad_shape'] = (pad_h, pad_w)
                        
                        # Size of the frequency domain subset to keep
                        comp_h = max(1, int(orig_h * k_ratio))
                        comp_w = max(1, int(orig_w * k_ratio))
                        state['comp_shape'] = (comp_h, comp_w)
                        
                        state['exp_avg'] = torch.zeros(comp_h, comp_w, dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros(comp_h, comp_w, dtype=grad.dtype, device=grad.device)
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    orig_h, orig_w = state['orig_shape']
                    pad_h, pad_w = state['pad_shape']
                    comp_h, comp_w = state['comp_shape']
                    
                    # 1. Fast Padding to power of 2 using F.pad
                    g_padded = F.pad(grad, (0, pad_w - orig_w, 0, pad_h - orig_h))
                    
                    # 2. Pure Spectral Transform (FWHT)
                    # We can compute grad and grad^2 transforms
                    g_freq = fwht_2d(g_padded)
                    # Note: grad^2 padding is the same as padded_grad^2
                    g_sq_freq = fwht_2d(g_padded.pow(2))
                    
                    # 3. Truncate high frequencies (Low-pass filter in spectral domain)
                    g_comp = g_freq[:comp_h, :comp_w]
                    g_sq_comp = g_sq_freq[:comp_h, :comp_w]
                    
                    # 4. Update states in compressed frequency domain
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    exp_avg.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)
                    
                    # 5. Decompress (Spectral Synthesis)
                    # Zero-pad frequency states back to full transform size
                    m_freq_full = F.pad(exp_avg, (0, pad_w - comp_w, 0, pad_h - comp_h))
                    v_freq_full = F.pad(exp_avg_sq, (0, pad_w - comp_w, 0, pad_h - comp_h))
                    
                    # Inverse FWHT
                    m_rec = ifwht_2d(m_freq_full)[:orig_h, :orig_w]
                    v_rec = ifwht_2d(v_freq_full)[:orig_h, :orig_w]
                    
                    # Stability: ensure variance is positive after spectral truncation
                    v_rec.clamp_(min=0.0)
                    
                else:
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec = state['exp_avg']
                    v_rec = state['exp_avg_sq']

                # 6. Apply Adam-style update
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)

        return loss

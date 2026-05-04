"""
smo/optim_dct_pure.py — SMODCTPure: Pure Spectral Optimizer (DCT)

Instead of Spatial Pooling, this uses a Pure Spectral Compression approach:
1. Apply 2D Discrete Cosine Transform (DCT) to the full gradient.
2. Update states in the frequency domain.
3. Apply a smooth spectral mask (low-pass filter) to discarding high-frequency noise.
4. Decompress using Inverse DCT.
"""

import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

def create_dct_matrix(N, dtype, device):
    """Creates a DCT-II transformation matrix."""
    n = torch.arange(N, dtype=dtype, device=device)
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
    dct_mat = torch.cos(math.pi / N * (n + 0.5) * k)
    dct_mat[0] *= 1.0 / math.sqrt(2.0)
    dct_mat *= math.sqrt(2.0 / N)
    return dct_mat

def dct_2d(x, dct_mat_h, dct_mat_w):
    """Fast 2D DCT using separable matrix multiplications. Properly normalized."""
    # Matrix multiplication with orthonormal DCT matrix is self-normalizing
    return torch.matmul(dct_mat_h, torch.matmul(x, dct_mat_w.t()))

def idct_2d(x_dct, dct_mat_h, dct_mat_w):
    """Fast 2D Inverse DCT using separable matrix multiplications. Properly normalized."""
    # Transpose of orthonormal matrix is its inverse
    return torch.matmul(dct_mat_h.t(), torch.matmul(x_dct, dct_mat_w))

class SMODCTPure(Optimizer):
    """
    Pure Spectral DCT Optimizer.
    Optimized for GPU by reducing allocations and using faster padding.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k_ratio=k_ratio)
        super().__init__(params, defaults)
        self._dct_cache = {}
        self._mask_cache = {}

    def _get_dct_matrices(self, h, w, dtype, device):
        key = (h, w, dtype, device)
        if key not in self._dct_cache:
            self._dct_cache[key] = (
                create_dct_matrix(h, dtype, device),
                create_dct_matrix(w, dtype, device)
            )
        return self._dct_cache[key]

    def _get_smooth_mask(self, h, w, comp_h, comp_w, dtype, device):
        key = (h, w, comp_h, comp_w, dtype, device)
        if key not in self._mask_cache:
            row_idx = torch.arange(h, dtype=dtype, device=device).view(-1, 1) / comp_h
            col_idx = torch.arange(w, dtype=dtype, device=device).view(1, -1) / comp_w
            # Fourth-order Butterworth-like filter for smooth spectral roll-off
            dist = torch.sqrt(row_idx**4 + col_idx**4)
            mask = 1.0 / (1.0 + torch.exp(10.0 * (dist - 1.0)))
            self._mask_cache[key] = mask
        return self._mask_cache[key]

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
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        orig_h, orig_w = grad.shape
                        state['orig_shape'] = (orig_h, orig_w)
                        # We maintain spatial states for numerical stability
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Always update states in SPATIAL domain first (Standard Adam EMA)
                # This prevents the spectral feedback loop that causes explosion.
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if state['is_compressed']:
                    orig_h, orig_w = state['orig_shape']
                    comp_h = max(1, int(orig_h * k_ratio))
                    comp_w = max(1, int(orig_w * k_ratio))
                    
                    dct_mat_h, dct_mat_w = self._get_dct_matrices(orig_h, orig_w, grad.dtype, grad.device)
                    mask = self._get_smooth_mask(orig_h, orig_w, comp_h, comp_w, grad.dtype, grad.device)
                    
                    # Apply spectral filtering to the MOMENTUM only
                    # We transform the momentum, mask it, and reconstruct it.
                    # This acts as a spatial denoiser for the update direction.
                    m_freq = dct_2d(exp_avg, dct_mat_h, dct_mat_w)
                    m_rec = idct_2d(m_freq * mask, dct_mat_h, dct_mat_w)
                    
                    # For variance, we use the raw spatial state (more stable)
                    v_rec = exp_avg_sq
                else:
                    m_rec = exp_avg
                    v_rec = exp_avg_sq

                # Apply Adam-style update
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)

        return loss

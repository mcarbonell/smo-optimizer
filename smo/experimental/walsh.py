"""
smo/experimental/walsh.py — SMOWalsh: Super Mario Optimizer (Spectral/Square version)

This variant uses Walsh-Hadamard transform for spectral compression.
The idea is to compress in the frequency domain, keeping only low-frequency 
coefficients, which represents the "smooth" part of the gradient.

This implementation uses precomputed Dense Matrix Multiplications for the 
Hadamard Transform, avoiding slow pure-Python loops, making it as fast as SMODCT.

🎮 "It's-a me, optimizer!" - Spectral Edition
"""

import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def create_hadamard_matrix(N, dtype, device):
    """
    Creates a 1D Walsh-Hadamard transformation matrix of size N x N
    using Sylvester's construction.
    """
    if (N & (N - 1)) != 0:
        raise ValueError("Hadamard matrix size must be a power of 2")
    
    H = torch.tensor([[1.0]], dtype=dtype, device=device)
    n = 1
    while n < N:
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)
        n *= 2
    return H

def fast_walsh_hadamard_2d_matrix(x, H_h, H_w, inverse=False):
    """
    Applies 2D Walsh-Hadamard Transform using separable matrix multiplications.
    H_h @ X @ H_w^T
    """
    res = torch.matmul(H_h, torch.matmul(x, H_w.t()))
    if inverse:
        res /= (H_h.shape[0] * H_w.shape[0])
    return res


class SMOWalsh(Optimizer):
    """
    Super Mario Optimizer - Walsh/Spectral version.
    
    Uses Walsh-Hadamard transform for spectral compression.
    
    Memory savings: 1 - k_ratio²
    
    Args:
        params: model.parameters()
        lr: learning rate (default: 1e-3)
        k_ratio: Fraction of resolution to keep (0.25 = 25% → 93.75% savings)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}. Must be in (0, 1]")
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                    weight_decay=weight_decay, k_ratio=k_ratio)
        super(SMOWalsh, self).__init__(params, defaults)
        
        # Cache for Hadamard matrices to avoid recomputing
        self._hadamard_cache = {}
        
    def _get_hadamard_matrices(self, h, w, dtype, device):
        key = (h, w, dtype, device)
        if key not in self._hadamard_cache:
            self._hadamard_cache[key] = (
                create_hadamard_matrix(h, dtype, device),
                create_hadamard_matrix(w, dtype, device)
            )
        return self._hadamard_cache[key]
    
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
                    raise RuntimeError('SMOWalsh does not support sparse gradients.')
                
                state = self.state[p]
                k_ratio = group['k_ratio']
                
                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    # Only compress large 2D tensors
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        orig_h, orig_w = grad.shape
                        state['orig_shape'] = (orig_h, orig_w)
                        
                        # Use a power of 2 for Walsh transform
                        n_pow2_h = 1 << (max(1, int(orig_h * k_ratio)) - 1).bit_length()
                        n_pow2_w = 1 << (max(1, int(orig_w * k_ratio)) - 1).bit_length()
                        
                        # Limit to avoid giant pads
                        n_pow2_h = min(n_pow2_h, 1 << (orig_h - 1).bit_length())
                        n_pow2_w = min(n_pow2_w, 1 << (orig_w - 1).bit_length())
                        
                        state['comp_h'] = n_pow2_h
                        state['comp_w'] = n_pow2_w
                        
                        state['exp_avg'] = torch.zeros(n_pow2_h, n_pow2_w, dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros(n_pow2_h, n_pow2_w, dtype=grad.dtype, device=grad.device)
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    orig_h, orig_w = state['orig_shape']
                    comp_h, comp_w = state['comp_h'], state['comp_w']
                    
                    # Get cached Hadamard matrices
                    H_h, H_w = self._get_hadamard_matrices(comp_h, comp_w, grad.dtype, grad.device)
                    
                    # 1. Downsample gradient
                    g_view = grad.unsqueeze(0).unsqueeze(0)
                    g_comp = F.adaptive_avg_pool2d(g_view, (comp_h, comp_w)).squeeze(0).squeeze(0)
                    
                    # 2. Walsh transform
                    grad_walsh = fast_walsh_hadamard_2d_matrix(g_comp, H_h, H_w, inverse=False)
                    
                    # 3. Update state
                    exp_avg.mul_(beta1).add_(grad_walsh, alpha=1 - beta1)
                    
                    # Variance
                    g_sq_comp = F.adaptive_avg_pool2d((grad ** 2).unsqueeze(0).unsqueeze(0), (comp_h, comp_w)).squeeze(0).squeeze(0)
                    grad_sq_walsh = fast_walsh_hadamard_2d_matrix(g_sq_comp, H_h, H_w, inverse=False)
                    exp_avg_sq.mul_(beta2).add_(grad_sq_walsh, alpha=1 - beta2)
                    
                    # 4. Decompress
                    m_rec = fast_walsh_hadamard_2d_matrix(exp_avg, H_h, H_w, inverse=True)
                    v_rec = fast_walsh_hadamard_2d_matrix(exp_avg_sq, H_h, H_w, inverse=True)
                    
                    # 5. Upsample
                    m_rec = F.interpolate(m_rec.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = F.interpolate(v_rec.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    
                    v_rec = torch.clamp(v_rec, min=0.0)
                
                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec = exp_avg
                    v_rec = exp_avg_sq
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)
        
        return loss

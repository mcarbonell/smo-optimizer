"""
smo/experimental/dct.py — SMODCT: Super Mario Optimizer (Discrete Cosine Transform)

This experimental variant uses the 2D Discrete Cosine Transform (DCT-II) 
for spectral compression. 

Unlike the Walsh-Hadamard transform (which uses square waves: +1, -1), 
DCT uses continuous cosine functions. This provides a much "smoother" 
frequency representation of the gradients, avoiding the blocky artifacts 
of spatial pooling and the abrupt transitions of Walsh.

The trade-off is computational cost: DCT requires floating-point 
trigonometric multiplications, whereas Walsh relies purely on addition/subtraction.

🎮 "It's-a me, optimizer!" - Smooth Jazz Edition 🎷
"""

import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def create_dct_matrix(N, dtype, device):
    """
    Creates a 1D DCT-II transformation matrix of size N x N.
    For small compressed sizes (e.g., 32x32), matrix multiplication 
    is heavily optimized on CPU/GPU and avoids the need for complex tensors (FFT).
    """
    n = torch.arange(N, dtype=dtype, device=device)
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
    
    # DCT-II formula
    dct_mat = torch.cos(math.pi / N * (n + 0.5) * k)
    
    # Orthogonal normalization
    dct_mat[0] *= 1.0 / math.sqrt(2.0)
    dct_mat *= math.sqrt(2.0 / N)
    
    return dct_mat

def dct_2d(x, dct_mat_h, dct_mat_w):
    """
    Applies 2D DCT using separable matrix multiplications.
    X_dct = D_h @ X @ D_w^T
    """
    # x shape: (H, W)
    return torch.matmul(dct_mat_h, torch.matmul(x, dct_mat_w.t()))

def idct_2d(x_dct, dct_mat_h, dct_mat_w):
    """
    Applies Inverse 2D DCT.
    Since DCT matrix is orthogonal, D^-1 = D^T
    X = D_h^T @ X_dct @ D_w
    """
    return torch.matmul(dct_mat_h.t(), torch.matmul(x_dct, dct_mat_w))


class SMODCT(Optimizer):
    """
    Super Mario Optimizer - DCT (Discrete Cosine Transform) version.
    
    Uses DCT for smooth spectral compression.
    Memory savings: 1 - k_ratio²
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k_ratio=k_ratio)
        super(SMODCT, self).__init__(params, defaults)
        
        # Cache for DCT matrices to avoid recomputing them every step
        self._dct_cache = {}
    
    def _get_dct_matrices(self, h, w, dtype, device):
        key = (h, w, dtype, device)
        if key not in self._dct_cache:
            self._dct_cache[key] = (
                create_dct_matrix(h, dtype, device),
                create_dct_matrix(w, dtype, device)
            )
        return self._dct_cache[key]

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
                    
                    # Compress large 2D tensors
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        orig_h, orig_w = grad.shape
                        state['orig_shape'] = (orig_h, orig_w)
                        
                        comp_h = max(1, int(orig_h * k_ratio))
                        comp_w = max(1, int(orig_w * k_ratio))
                        state['comp_shape'] = (comp_h, comp_w)
                        
                        # Initialize compressed states in the DCT Frequency Domain
                        state['exp_avg'] = torch.zeros(comp_h, comp_w, dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros(comp_h, comp_w, dtype=grad.dtype, device=grad.device)
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    orig_h, orig_w = state['orig_shape']
                    comp_h, comp_w = state['comp_shape']
                    
                    # Retrieve cached DCT matrices
                    dct_mat_h, dct_mat_w = self._get_dct_matrices(comp_h, comp_w, grad.dtype, grad.device)
                    
                    # 1. Spatial Downsample to base resolution
                    g_view = grad.unsqueeze(0).unsqueeze(0)
                    g_comp = F.adaptive_avg_pool2d(g_view, (comp_h, comp_w)).squeeze(0).squeeze(0)
                    
                    # 2. Transform gradient into DCT domain
                    grad_dct = dct_2d(g_comp, dct_mat_h, dct_mat_w)
                    
                    # 3. Update momentum in Frequency Domain
                    exp_avg.mul_(beta1).add_(grad_dct, alpha=1 - beta1)
                    
                    # Compress and transform squared gradient for variance
                    g_sq_comp = F.adaptive_avg_pool2d((grad ** 2).unsqueeze(0).unsqueeze(0), (comp_h, comp_w)).squeeze(0).squeeze(0)
                    grad_sq_dct = dct_2d(g_sq_comp, dct_mat_h, dct_mat_w)
                    exp_avg_sq.mul_(beta2).add_(grad_sq_dct, alpha=1 - beta2)
                    
                    # 4. Inverse DCT back to Spatial Domain
                    m_rec = idct_2d(exp_avg, dct_mat_h, dct_mat_w)
                    v_rec = idct_2d(exp_avg_sq, dct_mat_h, dct_mat_w)
                    
                    # 5. Upsample to original parameter resolution
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

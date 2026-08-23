"""
smo/experimental/dct_pure.py — SMODCTPure: Pure Spectral Optimizer (DCT)

Design: Walsh Hybrid pattern with DCT:
1. Downsample gradient to compressed size (adaptive_avg_pool2d)
2. Apply DCT on the downsampled gradient
3. Update compressed frequency-domain states
4. Inverse DCT back to compressed spatial domain
5. Upsample to original resolution with bilinear interpolation
"""

import math
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


def create_dct_matrix(N, dtype, device):
    """Creates orthonormal DCT-II matrix."""
    n = torch.arange(N, dtype=dtype, device=device)
    k = torch.arange(N, dtype=dtype, device=device).unsqueeze(1)
    dct_mat = torch.cos(math.pi / N * (n + 0.5) * k)
    dct_mat[0] *= 1.0 / math.sqrt(2.0)
    dct_mat *= math.sqrt(2.0 / N)
    return dct_mat


def dct_2d(x, dct_mat_h, dct_mat_w):
    """2D DCT: X_dct = D_h @ X @ D_w^T"""
    return torch.matmul(dct_mat_h, torch.matmul(x, dct_mat_w.t()))


def idct_2d(x_dct, dct_mat_h, dct_mat_w):
    """Inverse 2D DCT: X = D_h^T @ X_dct @ D_w"""
    return torch.matmul(dct_mat_h.t(), torch.matmul(x_dct, dct_mat_w))


class SMODCTPure(Optimizer):
    """
    DCT Spectral Optimizer following Walsh Hybrid pattern.
    Uses downsampling + DCT on compressed resolution for stability.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, k_ratio=k_ratio)
        super().__init__(params, defaults)

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
                beta1, beta2 = group['betas']

                # Initialize
                if len(state) == 0:
                    state['step'] = 0
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        orig_h, orig_w = grad.shape
                        state['orig_shape'] = (orig_h, orig_w)
                        comp_h = max(1, int(orig_h * k_ratio))
                        comp_w = max(1, int(orig_w * k_ratio))
                        state['comp_shape'] = (comp_h, comp_w)
                        state['exp_avg'] = torch.zeros(comp_h, comp_w, dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros(comp_h, comp_w, dtype=grad.dtype, device=grad.device)
                        # Cache DCT matrices for compressed size
                        self._get_dct_matrices(comp_h, comp_w, grad.dtype, grad.device)
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                if state['is_compressed']:
                    orig_h, orig_w = state['orig_shape']
                    comp_h, comp_w = state['comp_shape']

                    # Get cached DCT matrices for compressed resolution
                    dct_mat_h, dct_mat_w = self._get_dct_matrices(comp_h, comp_w, grad.dtype, grad.device)

                    # 1. Downsample gradient to compressed resolution (like Walsh Hybrid)
                    g_view = grad.unsqueeze(0).unsqueeze(0)
                    g_comp = F.adaptive_avg_pool2d(g_view, (comp_h, comp_w)).squeeze(0).squeeze(0)

                    # 2. Transform to DCT frequency domain (on compressed size)
                    g_dct = dct_2d(g_comp, dct_mat_h, dct_mat_w)

                    # 3. Update states in frequency domain
                    exp_avg.mul_(beta1).add_(g_dct, alpha=1 - beta1)

                    # 4. For variance: downsample squared gradient, then DCT
                    g_sq_comp = F.adaptive_avg_pool2d((grad ** 2).unsqueeze(0).unsqueeze(0), (comp_h, comp_w)).squeeze(0).squeeze(0)
                    g_sq_dct = dct_2d(g_sq_comp, dct_mat_h, dct_mat_w)
                    exp_avg_sq.mul_(beta2).add_(g_sq_dct, alpha=1 - beta2)

                    # 5. Inverse DCT back to spatial domain (compressed resolution)
                    m_rec = idct_2d(exp_avg, dct_mat_h, dct_mat_w)
                    v_rec = idct_2d(exp_avg_sq, dct_mat_h, dct_mat_w)

                    # 6. Upsample to original resolution
                    m_rec = F.interpolate(m_rec.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = F.interpolate(v_rec.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

                    v_rec.clamp_(min=0.0)

                else:
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec = exp_avg
                    v_rec = exp_avg_sq

                # Adam update
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)

        return loss
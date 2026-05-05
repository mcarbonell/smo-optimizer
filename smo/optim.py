"""
smo/optim.py — Super Mario Optimizer (SMO)

This optimizer drastically reduces the RAM consumption (up to 93%) by compressing 
the historical gradient states (Momentum 'm' and Variance 'v') using bilinear 
interpolation (a spatial proxy for low-frequency spectral compression).

The "loss of resolution" acts as a denoiser (implicit regularizer), 
filtering out high-frequency stochastic noise from mini-batches.

🎮 "It's-a me, optimizer!"
"""

import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F


class SMO(Optimizer):
    """
    Super Mario Optimizer - Blocky/Spatial version.
    
    Uses adaptive average pooling for compression. The "blocky" approach
    averages local regions, acting as a spatial smoother.
    
    Memory savings: 1 - k_ratio²
    
    Args:
        params: model.parameters()
        lr: learning rate (default: 1e-3)
        k_ratio: Fraction of resolution to keep (0.25 = 25% → 93.75% savings)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25):
        """
        k_ratio: Fraction of the original resolution to keep (e.g. 0.25 = 25% per dimension).
                 For 2D tensors, the RAM savings is 1 - (k_ratio^2) = 93.75%.
        """
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
        super(SMO, self).__init__(params, defaults)

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
                    raise RuntimeError('SMO does not support sparse gradients.')

                state = self.state[p]
                k_ratio = group['k_ratio']

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Only compress large 2D tensors (e.g. weight matrices).
                    # 1D biases or very small embeddings remain full-res to avoid
                    # interpolation instability and because their RAM footprint is negligible.
                    if grad.dim() == 2 and grad.shape[0] >= 32 and grad.shape[1] >= 32:
                        state['is_compressed'] = True
                        new_h = max(1, int(grad.shape[0] * k_ratio))
                        new_w = max(1, int(grad.shape[1] * k_ratio))
                        # Initial compressed states
                        state['exp_avg'] = torch.zeros((new_h, new_w), dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros((new_h, new_w), dtype=grad.dtype, device=grad.device)
                        state['orig_shape'] = grad.shape
                    else:
                        state['is_compressed'] = False
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    # 1. Compress the current gradient (Downsample)
                    g_view = grad.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    
                    # 'area' interpolation (adaptive_avg_pool2d) is mathematically 
                    # more stable for gradients than bilinear downsampling.
                    g_comp = F.adaptive_avg_pool2d(g_view, exp_avg.shape).squeeze(0).squeeze(0)
                    
                    # 2. Update compressed moments
                    exp_avg.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                    
                    # Compress squared gradient for the second moment
                    # Reuse g_comp to avoid computing grad**2 at full resolution
                    g_sq_comp = g_comp ** 2
                    exp_avg_sq.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)

                    # 3. Decompress (Upsample) to apply the update
                    m_view = exp_avg.unsqueeze(0).unsqueeze(0)
                    v_view = exp_avg_sq.unsqueeze(0).unsqueeze(0)
                    
                    m_rec = F.interpolate(m_view, size=state['orig_shape'], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    v_rec = F.interpolate(v_view, size=state['orig_shape'], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                    
                    # Ensure strict positivity in v_rec (mitigates interpolation artifacts)
                    v_rec = torch.clamp(v_rec, min=0.0)

                else:
                    # Standard fallback for 1D / small tensors
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec = exp_avg
                    v_rec = exp_avg_sq

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1

                # Final weight update
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p.addcdiv_(m_rec, denom, value=-step_size)

        return loss

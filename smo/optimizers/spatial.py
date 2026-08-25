"""
smo/optimizers/spatial.py — Super Mario Optimizer (SMO)

This optimizer drastically reduces the persistent optimizer-state memory
(up to 93% of the state_dict/checkpoint footprint) by compressing the
historical gradient states (Momentum 'm' and Variance 'v') using bilinear
interpolation (a spatial proxy for low-frequency spectral compression).

The "loss of resolution" acts as a denoiser (implicit regularizer),
filtering out high-frequency stochastic noise from mini-batches.

MEMORY ACCOUNTING NOTE:
The reported savings refer to the *persistent* optimizer state
(state_dict / checkpoint size). During training, this optimizer keeps a
private buffer pool (`_param_buffers`) that includes full-resolution
reconstruction buffers (`m_rec`, `v_rec`) cached for speed, so the resident
working set is larger than the persistent state. See
benchmarks/METHODOLOGY.md ("State-memory benchmarks") for definitions.

Only tensors with a pooled 2D view of at least 32x32 are compressed — 2D
matrices (linear weights) by default; 4D conv weights only with
``compress_conv=True`` (opt-in: pooling across unrelated input channels
measured negative on CNNs). Everything else falls back to dense Adam moments.

🎮 "It's-a me, optimizer!"
"""

import math
import torch
from torch.optim.optimizer import Optimizer

from ._spatial_utils import compress_2d, compress_2d_pair, compress_2d_pair_into_buffers, compression_view, upsample_2d_pair, upsample_2d_pair_into_buffers


class SMO(Optimizer):
    """
    Super Mario Optimizer - Blocky/Spatial version.
    
    Uses adaptive average pooling for compression. The "blocky" approach
    averages local regions, acting as a spatial smoother.
    
    Memory savings: 1 - k_ratio²
    
    Args:
        params: model.parameters()
        lr: learning rate (default: 1e-3)
        betas: (beta1, beta2) (default: (0.9, 0.999))
        eps: epsilon for numerical stability (default: 1e-8)
        weight_decay: L2 penalty (default: 0)
        k_ratio: Fraction of resolution to keep (0.25 = 25% → 93.75% savings)
        compress_conv: also pool 4D conv weights via the flattened
            (out_c, in_c*kh*kw) row view. Default False: measured negative
            on CNNs (see _spatial_utils.compression_view).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25, compress_conv=False):
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
        self.compress_conv = bool(compress_conv)
        # Private buffer pool: param_id -> {'g_comp', 'g_sq_comp', 'm_rec', 'v_rec'}
        # These are NOT part of the optimizer state (not saved in state_dict).
        self._param_buffers = {}

    @staticmethod
    def _compress_2d(tensor, target_shape):
        return compress_2d(tensor, target_shape)

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
                # Per-param-group opt-out: {"compress": False} forces dense Adam moments
                use_compression = group.get('compress', True)
                if use_compression == "quant_only":
                    raise ValueError("SMO (fp32) has no int8 storage; 'quant_only'"
                                     " groups require SMO8bit")

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    view = compression_view(grad.shape, include_conv=self.compress_conv) if use_compression else None
                    if view is not None and p.is_contiguous():
                        view_shape, param_shape = view
                        state['is_compressed'] = True
                        new_h = max(1, int(view_shape[0] * k_ratio))
                        new_w = max(1, int(view_shape[1] * k_ratio))
                        comp_shape = (new_h, new_w)
                        # Compressed moment states (persistent)
                        state['exp_avg'] = torch.zeros(comp_shape, dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq'] = torch.zeros(comp_shape, dtype=grad.dtype, device=grad.device)
                        # orig_shape is the pooled ROW VIEW the math runs in;
                        # param_shape is the tensor's real shape (e.g. conv 4D).
                        state['orig_shape'] = view_shape
                        state['param_shape'] = param_shape
                        # Allocate reusable buffers (not saved in state_dict)
                        buf_id = id(p)
                        self._param_buffers[buf_id] = {
                            'g_comp': torch.empty(comp_shape, dtype=grad.dtype, device=grad.device),
                            'g_sq_comp': torch.empty(comp_shape, dtype=grad.dtype, device=grad.device),
                            'm_rec': torch.empty(view_shape, dtype=grad.dtype, device=grad.device),
                            'v_rec': torch.empty(view_shape, dtype=grad.dtype, device=grad.device),
                        }
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
                    # Get reusable buffers
                    buf_id = id(p)
                    buffers = self._param_buffers.get(buf_id)
                    if buffers is None:
                        # Should not happen, but allocate fallback
                        buffers = {
                            'g_comp': torch.empty(exp_avg.shape, dtype=grad.dtype, device=grad.device),
                            'g_sq_comp': torch.empty(exp_avg.shape, dtype=grad.dtype, device=grad.device),
                            'm_rec': torch.empty(state['orig_shape'], dtype=grad.dtype, device=grad.device),
                            'v_rec': torch.empty(state['orig_shape'], dtype=grad.dtype, device=grad.device),
                        }
                        self._param_buffers[buf_id] = buffers

                    # 1. Compress gradient into pre-allocated buffers.
                    # The math runs on the pooled row view (identical data for
                    # 2D params; conv 4D weights are flattened row-wise).
                    grad_2d = grad.reshape(state['orig_shape'])
                    # v_rec is reused as scratch space for grad^2 (it is
                    # overwritten by the upsample step below), avoiding a
                    # full-size temporary allocation every step.
                    g_sq_scratch = buffers['v_rec']
                    torch.square(grad_2d, out=g_sq_scratch)
                    g_comp, g_sq_comp = compress_2d_pair_into_buffers(
                        grad_2d, g_sq_scratch,
                        buffers['g_comp'], buffers['g_sq_comp'],
                        target_shape=exp_avg.shape
                    )
                    
                    # 2. Update compressed moments in-place
                    exp_avg.mul_(beta1).add_(g_comp, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)

                    # 3. Upsample into pre-allocated reconstruction buffers
                    m_rec, v_rec = upsample_2d_pair_into_buffers(
                        exp_avg, exp_avg_sq,
                        buffers['m_rec'], buffers['v_rec'],
                        state['orig_shape']
                    )
                    # In-place clamp to ensure positivity (mitigates interpolation artifacts)
                    v_rec.clamp_(min=0.0)

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

                # Final weight update. In the compressed path the math ran on
                # the pooled row view; write through the same view of p
                # (guaranteed contiguous at init), so conv 4D weights are
                # updated in place without any copy.
                if state['is_compressed']:
                    p_target = p.view(state['orig_shape'])
                else:
                    p_target = p
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p_target.addcdiv_(m_rec, denom, value=-step_size)

        return loss

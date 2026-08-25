"""
smo/optimizers/spatial_8bit.py — SMO-8bit (Star Mode)

The ultimate evolution of the SuperMario Optimizer.
Combines:
1. Spatial Compression (SMO): Reduces resolution of states by k_ratio.
2. Block-wise Quantization: Stores the remaining coefficients as 8-bit integers.

Persistent optimizer-state footprint: ~2% of standard AdamW (see
smo/optimizers/spatial.py for the persistent-vs-resident memory distinction).

Only tensors with a pooled 2D view of at least 32x32 are compressed — 2D
matrices (linear weights) by default; 4D conv weights only with
``compress_conv=True`` (opt-in: pooling across unrelated input channels
measured negative on CNNs). Everything else falls back to dense Adam moments.

🎮 "It's-a me, ultra-optimizer!" - Star Mode Active
"""

import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from ._spatial_utils import compress_2d, compress_2d_pair, compression_view, upsample_2d_pair

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
        low_peak: If True, compress and reconstruct in row bands so no
            full-resolution temporary is ever materialized. Reduces peak
            step memory from ~21 to ~9 bytes/param, which is what makes
            "trains where AdamW OOMs" possible. Requires the compressed
            shape to evenly divide the original (exact-pool path); other
            tensors silently fall back to the monolithic step.
        band_mb: Approximate per-band temporary budget in MB for low_peak.
        permute_basis: If True, rows/columns of the gradient are randomly
            permuted (fixed per-parameter, from the global RNG) before
            spatial pooling, and the reconstruction is unpermuted. This
            destroys neighborhood locality while keeping the compression
            ratio — an ablation to test whether smoothing helps because
            adjacent coordinates are correlated or for other reasons.
            Incompatible with low_peak (raises ValueError).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25, block_size=64,
                 low_peak=False, band_mb=64.0, permute_basis=False,
                 compress_conv=False):
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
        self.low_peak = low_peak
        self.band_mb = float(band_mb)
        if low_peak and permute_basis:
            raise ValueError("permute_basis is not supported with low_peak")
        self.permute_basis = permute_basis
        self.compress_conv = bool(compress_conv)

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
        q_blocks = (blocks / scales * 127).round()
        # Dead-zone lift: round-to-nearest silently flushes any entry below
        # half an LSB to EXACT zero. For second moments that means v=0 ->
        # denominator ~eps -> oversized updates -> divergence (observed on
        # CNNs where block scales are dominated by large-magnitude rows).
        # Never encode a non-zero value as zero; costs at most +/-1 LSB of
        # bias and keeps m/v consistently away from the dead zone.
        q_blocks = torch.where((q_blocks == 0) & (blocks != 0),
                               torch.sign(blocks), q_blocks)
        q_blocks = q_blocks.clamp_(-127, 127).to(torch.int8)
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
                # Per-param-group compression contract:
                #   {"compress": True}          -> spatial pooling + int8 (eligible shapes)
                #   {"compress": False}         -> dense fp32 Adam moments
                #   {"compress": "quant_only"}  -> FULL-resolution int8 moments, no spatial
                #                                  pooling. For tensors whose rows have
                #                                  arbitrary neighbors (embedding tables,
                #                                  tied heads): quantization keeps the
                #                                  memory win without mixing tokens.
                raw_compress = group.get('compress', True)
                use_compression = bool(raw_compress)
                quant_only = (raw_compress == "quant_only")

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    if quant_only:
                        state['is_quant_only'] = True
                        state['is_compressed'] = False
                        state['q_shape'] = tuple(grad.shape)
                        state['q_numel'] = grad.numel()
                        # Matrix-like tensors get one int8 scale per row:
                        # token rows differ in scale (same dead-zone lesson
                        # as the conv row-aligned policy).
                        tail = grad.shape[-1] if grad.dim() >= 2 else max(1, grad.numel())
                        state['q_block'] = min(block_size, tail) if grad.dim() >= 2 else block_size
                        dummy = torch.zeros(tuple(grad.shape), dtype=grad.dtype, device=grad.device)
                        state['m_q'], state['m_s'], _ = self._quantize_blockwise(dummy, state['q_block'])
                        state['v_q'], state['v_s'], _ = self._quantize_blockwise(dummy, state['q_block'])
                    else:
                        view = compression_view(grad.shape, include_conv=self.compress_conv) if use_compression else None
                        if view is not None and p.is_contiguous():
                            view_shape, param_shape = view
                            state['is_compressed'] = True
                            # orig_shape is the pooled ROW VIEW the math runs in;
                            # param_shape is the tensor's real shape (e.g. conv 4D).
                            state['orig_shape'] = view_shape
                            state['param_shape'] = param_shape
                            new_h = max(1, int(view_shape[0] * k_ratio))
                            new_w = max(1, int(view_shape[1] * k_ratio))
                            comp_shape = (new_h, new_w)
                            state['comp_numel'] = new_h * new_w
                            # Quantization blocks must not mix rows of the compact
                            # tensor: conv output channels can differ by orders of
                            # magnitude in moment scale, and a shared int8 scale
                            # crushes low-magnitude rows -> underestimated second
                            # moments -> oversized steps -> divergence (observed as
                            # NaN losses on CNNs with block_size=64). Linear
                            # matrices keep the historical flattened-block layout
                            # for campaign comparability; conv-view (4D) params
                            # quantize one full row per block.
                            state['q_block'] = min(block_size, new_w) if len(param_shape) == 4 else block_size

                            # Initialize states as quantized
                            dummy = torch.zeros(comp_shape, dtype=grad.dtype, device=grad.device)
                            m_q, m_s, _ = self._quantize_blockwise(dummy, state['q_block'])
                            v_q, v_s, _ = self._quantize_blockwise(dummy, state['q_block'])

                            state['m_q'], state['m_s'] = m_q, m_s
                            state['v_q'], state['v_s'] = v_q, v_s
                            state['comp_shape'] = comp_shape
                            if self.permute_basis:
                                state['row_perm'] = torch.randperm(view_shape[0], device=grad.device)
                                state['col_perm'] = torch.randperm(view_shape[1], device=grad.device)
                        else:
                            state['is_compressed'] = False
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                beta1, beta2 = group['betas']
                state['step'] += 1

                if state['is_compressed'] and self.low_peak and self._can_band(state):
                    self._step_banded(p, grad, state, group, beta1, beta2)
                    continue

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['is_compressed']:
                    # All math runs on the pooled row view (identical data for
                    # 2D params; conv 4D weights are flattened row-wise).
                    grad_2d = grad.reshape(state['orig_shape'])

                    # 1. Dequantize current state for update (match grad dtype
                    # so the final in-place update on p is dtype-consistent)
                    m = self._dequantize_blockwise(state['m_q'], state['m_s'], state['comp_shape'], state['comp_numel'], dtype=grad.dtype)
                    v = self._dequantize_blockwise(state['v_q'], state['v_s'], state['comp_shape'], state['comp_numel'], dtype=grad.dtype)

                    # 2. Compress gradient (optionally in a permuted basis)
                    if self.permute_basis:
                        rp, cp = state['row_perm'], state['col_perm']
                        g_eff = grad_2d[rp][:, cp]
                        g_comp, g_sq_comp = compress_2d_pair(g_eff, g_eff.square(), state['comp_shape'])
                        del g_eff
                    else:
                        g_comp, g_sq_comp = compress_2d_pair(grad_2d, grad_2d.square(), state['comp_shape'])

                    # 3. Update moments (in float32)
                    m.mul_(beta1).add_(g_comp, alpha=1 - beta1)

                    v.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)

                    # 4. Upsample for weight update BEFORE re-quantizing
                    m_rec, v_rec = upsample_2d_pair(m, v, state['orig_shape'])

                    # Undo the basis permutation so the update lands on the
                    # original coordinates
                    if self.permute_basis:
                        inv_rp = torch.argsort(state['row_perm'])
                        inv_cp = torch.argsort(state['col_perm'])
                        m_rec = m_rec[inv_rp][:, inv_cp]
                        v_rec = v_rec[inv_rp][:, inv_cp]

                    v_rec = torch.clamp(v_rec, min=0.0)

                    # 5. Re-quantize and store
                    state['m_q'], state['m_s'], _ = self._quantize_blockwise(m, state['q_block'])
                    state['v_q'], state['v_s'], _ = self._quantize_blockwise(v, state['q_block'])

                    # 6. Free temporary float32 tensors immediately to avoid VRAM spikes
                    del m, v
                elif state.get('is_quant_only'):
                    # Full-resolution int8 moments: no spatial pooling, so
                    # coordinates with arbitrary neighborhoods (embedding
                    # rows, tied heads) are never mixed. Memory cost is the
                    # quantized storage only (~1 B/param for m+v at fp32
                    # moments), not AdamW's 8 B/param.
                    m = self._dequantize_blockwise(state['m_q'], state['m_s'], state['q_shape'], state['q_numel'], dtype=grad.dtype)
                    v = self._dequantize_blockwise(state['v_q'], state['v_s'], state['q_shape'], state['q_numel'], dtype=grad.dtype)
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    # Keep the fp32 copies: they ARE the reconstruction used
                    # by the update below (re-quantized for storage).
                    state['m_q'], state['m_s'], _ = self._quantize_blockwise(m, state['q_block'])
                    state['v_q'], state['v_s'], _ = self._quantize_blockwise(v, state['q_block'])
                    m_rec, v_rec = m, v
                else:
                    # Fallback for 1D/small tensors
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_rec = state['exp_avg']
                    v_rec = state['exp_avg_sq']

                # Standard Adam update logic. In the compressed path the math
                # ran on the pooled row view; write through the same view of p
                # (guaranteed contiguous at init), so conv 4D weights are
                # updated in place without any copy.
                if state['is_compressed']:
                    p_target = p.view(state['orig_shape'])
                else:
                    p_target = p
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                p_target.addcdiv_(m_rec, denom, value=-step_size)

        return loss

    def _can_band(self, state) -> bool:
        """Banding requires the exact-pool path (orig divisible by comp)."""
        orig_h, orig_w = state['orig_shape']
        comp_h, comp_w = state['comp_shape']
        return orig_h % comp_h == 0 and orig_w % comp_w == 0

    def _step_banded(self, p, grad, state, group, beta1, beta2):
        """Compress, update and reconstruct in row bands (low_peak mode).

        Exact avg-pooling and bilinear interpolation are local along each
        axis, so a row band processed with its neighboring source rows is
        numerically identical to the monolithic computation, while keeping
        every temporary bounded (~self.band_mb). No full-resolution tensor
        is ever allocated: peak step memory drops from ~21 to ~9 bytes/param.
        """
        eps = group['eps']
        q_block = state['q_block']
        wd = group['weight_decay']
        orig_h, orig_w = state['orig_shape']
        comp_h, comp_w = state['comp_shape']
        kh = orig_h // comp_h
        kw = orig_w // comp_w
        elem = grad.element_size()

        # The math runs on the pooled row view; write back through the same
        # view of p (contiguous by init-time eligibility check).
        grad = grad.reshape(orig_h, orig_w)
        p_view = p.view(orig_h, orig_w)

        # Compact (comp-resolution) floats: small by construction
        m = self._dequantize_blockwise(state['m_q'], state['m_s'], state['comp_shape'], state['comp_numel'], dtype=grad.dtype)
        v = self._dequantize_blockwise(state['v_q'], state['v_s'], state['comp_shape'], state['comp_numel'], dtype=grad.dtype)

        # 1. Compress gradient + EMA update, band by source rows
        band_src = max(kh, int(self.band_mb * 1024**2 / (3 * orig_w * elem)))
        r0 = 0
        while r0 < orig_h:
            r1 = min(orig_h, r0 + band_src)
            r1 -= r1 % kh
            if r1 <= r0:
                break
            g_band = grad[r0:r1]
            if wd != 0:
                g_band = g_band.add(p_view[r0:r1], alpha=wd)
            sq_band = torch.square(g_band)
            stacked = torch.stack((g_band, sq_band), dim=0).unsqueeze(0)
            pooled = F.avg_pool2d(stacked, kernel_size=(kh, kw), stride=(kh, kw)).squeeze(0)
            c0, c1 = r0 // kh, r1 // kh
            m[c0:c1].mul_(beta1).add_(pooled[0], alpha=1 - beta1)
            v[c0:c1].mul_(beta2).add_(pooled[1], alpha=1 - beta2)
            r0 = r1

        # 2. Reconstruct + apply update, band by compressed rows (+/- 1 margin)
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] / bias_correction1
        sqrt_bc2 = math.sqrt(bias_correction2)

        band_comp = max(2, int(self.band_mb * 1024**2 / (2 * orig_w * elem * kh)))
        for s0 in range(0, comp_h, band_comp):
            s1 = min(comp_h, s0 + band_comp)
            a, b = max(0, s0 - 1), min(comp_h, s1 + 1)
            stacked = torch.stack((m[a:b], v[a:b]), dim=0).unsqueeze(0)
            up = F.interpolate(stacked, size=((b - a) * kh, orig_w), mode='bilinear', align_corners=False).squeeze(0)
            o0 = (s0 - a) * kh
            o1 = o0 + (s1 - s0) * kh
            m_band = up[0, o0:o1]
            v_band = up[1, o0:o1].clamp_(min=0.0)
            denom = (v_band.sqrt() / sqrt_bc2).add_(eps)
            p_view[s0 * kh:s1 * kh].addcdiv_(m_band, denom, value=-step_size)

        # 3. Re-quantize compact states
        state['m_q'], state['m_s'], _ = self._quantize_blockwise(m, q_block)
        state['v_q'], state['v_s'], _ = self._quantize_blockwise(v, q_block)

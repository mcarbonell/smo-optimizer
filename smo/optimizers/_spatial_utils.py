"""Shared helpers for spatial optimizer variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Tensors whose pooled view has any dimension below this fall back to dense
# Adam moments (pooling too few coordinates per block is not a consensus).
MIN_SPATIAL_DIM = 32


def compression_view(shape, include_conv: bool = False):
    """Return ``(view_2d_shape, param_shape)`` if a tensor is eligible for
    spatial pooling, else ``None``.

    Eligible tensors are treated as a 2D matrix of independent rows:

    - 2D matrices (linear weights) with both dims >= ``MIN_SPATIAL_DIM``
      (historical behavior);
    - 4D conv weights ``(out_c, in_c, kh, kw)``, viewed as
      ``(out_c, in_c * kh * kw)`` when both view dims qualify — **only when
      ``include_conv=True``** (opt-in). Measured verdict at default-off
      (CIFAR-CNN, seed 1234): pooling over the flattened ``in_c*kh*kw`` axis
      costs the SMO family heavily and does NOT recover with training length
      (3ep: 46.9/52.3 vs dense-fallback 58.3/62.0; 10ep: 58.6/65.5 vs
      69.0/71.0; Adam 72.9) — averaging across unrelated input channels
      violates H4's locality prior. Kept available for future basis
      experiments.

    The caller is responsible for ensuring the parameter itself is
    contiguous before writing through the 2D view.
    """
    if len(shape) == 2:
        if shape[0] >= MIN_SPATIAL_DIM and shape[1] >= MIN_SPATIAL_DIM:
            return tuple(shape), tuple(shape)
        return None
    if len(shape) == 4 and include_conv:
        flat_cols = shape[1] * shape[2] * shape[3]
        if shape[0] >= MIN_SPATIAL_DIM and flat_cols >= MIN_SPATIAL_DIM:
            return (shape[0], flat_cols), tuple(shape)
    return None


def _can_use_exact_pool(shape: tuple[int, int], target_shape: tuple[int, int]) -> bool:
    height, width = shape
    target_h, target_w = target_shape
    return (
        target_h > 0
        and target_w > 0
        and height % target_h == 0
        and width % target_w == 0
    )


def compress_2d(tensor: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
    """Compress a 2D tensor, using a faster exact block average when possible."""
    height, width = tensor.shape
    target_h, target_w = target_shape

    if _can_use_exact_pool((height, width), target_shape):
        kernel_h = height // target_h
        kernel_w = width // target_w
        view = tensor.unsqueeze(0).unsqueeze(0)
        return F.avg_pool2d(view, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w)).squeeze(0).squeeze(0)

    view = tensor.unsqueeze(0).unsqueeze(0)
    return F.adaptive_avg_pool2d(view, target_shape).squeeze(0).squeeze(0)


def compress_2d_pair(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    target_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress two 2D tensors in one pooled call when possible."""
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("compress_2d_pair expects matching input shapes")

    height, width = tensor_a.shape
    stacked = torch.stack((tensor_a, tensor_b), dim=0).unsqueeze(0)

    if _can_use_exact_pool((height, width), target_shape):
        target_h, target_w = target_shape
        kernel_h = height // target_h
        kernel_w = width // target_w
        pooled = F.avg_pool2d(stacked, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
    else:
        pooled = F.adaptive_avg_pool2d(stacked, target_shape)

    pooled = pooled.squeeze(0)
    return pooled[0], pooled[1]


def compress_2d_pair_into_buffers(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    target_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress two 2D tensors into pre-allocated output buffers.
    Returns (out_a, out_b) for convenience.
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("compress_2d_pair expects matching input shapes")
    if out_a.shape != target_shape or out_b.shape != target_shape:
        raise ValueError("output buffers must match target_shape")

    height, width = tensor_a.shape
    stacked = torch.stack((tensor_a, tensor_b), dim=0).unsqueeze(0)

    if _can_use_exact_pool((height, width), target_shape):
        target_h, target_w = target_shape
        kernel_h = height // target_h
        kernel_w = width // target_w
        pooled = F.avg_pool2d(stacked, kernel_size=(kernel_h, kernel_w), stride=(kernel_h, kernel_w))
    else:
        pooled = F.adaptive_avg_pool2d(stacked, target_shape)

    pooled = pooled.squeeze(0)
    out_a.copy_(pooled[0])
    out_b.copy_(pooled[1])
    return out_a, out_b


def upsample_2d_pair(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    target_shape: tuple[int, int],
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upsample two 2D tensors in one interpolation call."""
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("upsample_2d_pair expects matching input shapes")

    if tensor_a.shape == target_shape:
        return tensor_a, tensor_b

    stacked = torch.stack((tensor_a, tensor_b), dim=0).unsqueeze(0)
    upsampled = F.interpolate(stacked, size=target_shape, mode=mode, align_corners=align_corners).squeeze(0)
    return upsampled[0], upsampled[1]


def upsample_2d_pair_into_buffers(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    target_shape: tuple[int, int],
    *,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Upsample two 2D tensors into pre-allocated output buffers.
    Returns (out_a, out_b) for convenience.
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("upsample_2d_pair expects matching input shapes")
    if out_a.shape != target_shape or out_b.shape != target_shape:
        raise ValueError("output buffers must match target_shape")

    if tensor_a.shape == target_shape:
        out_a.copy_(tensor_a)
        out_b.copy_(tensor_b)
        return out_a, out_b

    stacked = torch.stack((tensor_a, tensor_b), dim=0).unsqueeze(0)
    upsampled = F.interpolate(stacked, size=target_shape, mode=mode, align_corners=align_corners).squeeze(0)
    out_a.copy_(upsampled[0])
    out_b.copy_(upsampled[1])
    return out_a, out_b

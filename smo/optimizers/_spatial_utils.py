"""Shared helpers for spatial optimizer variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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

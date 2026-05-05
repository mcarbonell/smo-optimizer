"""Backward-compatible wrapper for the Triton spatial optimizer."""

from .optimizers.spatial_triton import HAS_TRITON, SMOTriton

__all__ = ["HAS_TRITON", "SMOTriton"]


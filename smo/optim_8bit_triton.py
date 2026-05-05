"""Backward-compatible wrapper for the Triton 8-bit spatial optimizer."""

from .optimizers.spatial_8bit_triton import HAS_TRITON, SMO8bitTriton

__all__ = ["HAS_TRITON", "SMO8bitTriton"]


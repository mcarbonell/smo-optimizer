"""Backward-compatible wrapper for 8-bit activation compression."""

from .activations.quant8 import SMO8bitLinear, SMO8bitLinearFunction, dequantize_blockwise, quantize_blockwise, wrap_model_activations

__all__ = [
    "SMO8bitLinear",
    "SMO8bitLinearFunction",
    "dequantize_blockwise",
    "quantize_blockwise",
    "wrap_model_activations",
]


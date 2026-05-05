"""Backward-compatible wrapper for delta activation compression."""

from .activations.delta import SMODeltaLinear, SMODeltaLinearFunction, wrap_model_delta

__all__ = ["SMODeltaLinear", "SMODeltaLinearFunction", "wrap_model_delta"]


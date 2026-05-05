"""Backward-compatible wrapper for FP16 activation hooks."""

from .activations.fp16_hooks import SMOActivationSqueezer, smo_squeezer

__all__ = ["SMOActivationSqueezer", "smo_squeezer"]


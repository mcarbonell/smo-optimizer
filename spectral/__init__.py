"""Backward-compatible spectral namespace."""

from .optim_dct import SMODCT
from .optim_dct_pure import SMODCTPure
from .optim_walsh import SMOWalsh
from .optim_walsh_pure import SMOWalshPure

__all__ = ["SMODCT", "SMODCTPure", "SMOWalsh", "SMOWalshPure"]


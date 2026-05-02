"""
SMO: Super Mario Optimizer

A highly compressed, memory-efficient PyTorch optimizer based on spectral/spatial compression.

Versions:
- SMO: Uses adaptive average pooling (spatial compression)
- SMO 8-bit: Spatial compression + 8-bit quantization
"""

from .optim import SMO
from .optim_8bit import SMO8bit
from .optim_8bit_triton import SMO8bitTriton

__all__ = ['SMO', 'SMO8bit', 'SMO8bitTriton']

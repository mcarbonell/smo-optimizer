"""
smo/activations_hooks.py — "Super Mario Hooks" (v5 - Stable FP16)

Uses global autograd hooks to store activations in FP16 (half-precision).
Provides 50% memory savings for activations with near-zero accuracy loss.
"""

import torch

class SMOActivationSqueezer:
    """
    Context manager that 'squeezes' all activations saved during its execution.
    Uses FP16 for 2x compression.
    """
    def __init__(self, enabled=True, min_elements=1024):
        self.enabled = enabled
        self.min_elements = min_elements

    def __enter__(self):
        if self.enabled:
            self.hook = torch.autograd.graph.saved_tensors_hooks(
                self.pack_hook, self.unpack_hook
            )
            self.hook.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.hook.__exit__(*args)

    def pack_hook(self, tensor):
        # 1. Skip if not floating point
        if not torch.is_floating_point(tensor):
            return tensor
            
        # 2. Skip if too small
        if tensor.numel() < self.min_elements:
            return tensor
            
        # 3. Store in float16 for 50% savings
        # We also store the original dtype to restore it later
        return (tensor.to(torch.float16), tensor.dtype)

    def unpack_hook(self, packed):
        if not isinstance(packed, tuple):
            return packed
            
        tensor_fp16, orig_dtype = packed
        return tensor_fp16.to(orig_dtype)

# Simple wrapper for easy use
def smo_squeezer(enabled=True, min_elements=1024):
    return SMOActivationSqueezer(enabled=enabled, min_elements=min_elements)

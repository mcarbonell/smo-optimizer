"""
smo/activations_8bit.py — "Ninja" 8-bit Quantized Activations (v3.1)

Fixed inheritance for Linear layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize_blockwise(data, block_size):
    orig_shape = data.shape
    flat_data = data.flatten()
    n = flat_data.numel()
    pad_size = (block_size - (n % block_size)) % block_size
    if pad_size > 0:
        flat_data = F.pad(flat_data, (0, pad_size))
    
    blocks = flat_data.view(-1, block_size)
    scales = blocks.abs().max(dim=1, keepdim=True)[0]
    scales = scales.clamp(min=1e-12)
    q_blocks = (blocks / scales * 127).round().to(torch.int8)
    return q_blocks, scales, orig_shape

def dequantize_blockwise(q_blocks, scales, orig_shape):
    blocks = q_blocks.to(torch.float32) * (scales / 127.0)
    flat_data = blocks.flatten()
    return flat_data[:math.prod(orig_shape)].view(orig_shape)

class SMO8bitLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, block_size):
        q_input, scales, shape = quantize_blockwise(input, block_size)
        ctx.save_for_backward(q_input, scales, weight, bias)
        ctx.orig_shape = shape
        ctx.block_size = block_size
        with torch.no_grad():
            return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        q_input, scales, weight, bias = ctx.saved_tensors
        input_rec = dequantize_blockwise(q_input, scales, ctx.orig_shape)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input_rec)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None

class SMO8bitLinear(nn.Linear):
    def __init__(self, original_linear, block_size=64):
        # Initialize correctly as a Linear layer
        super().__init__(original_linear.in_features, original_linear.out_features, bias=original_linear.bias is not None)
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.block_size = block_size
        
    def forward(self, input):
        return SMO8bitLinearFunction.apply(input, self.weight, self.bias, self.block_size)

def wrap_model_activations(model, block_size=64):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, SMO8bitLinear(child, block_size))
        else:
            wrap_model_activations(child, block_size)
    return model

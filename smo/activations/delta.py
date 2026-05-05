"""
smo/activations_delta.py — Delta-Encoded Activations (v3.1)

Fixed inheritance for Linear layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SMODeltaLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input.to(torch.float16), weight, bias)
        ctx.orig_dtype = input.dtype
        with torch.no_grad():
            return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input_fp16, weight, bias = ctx.saved_tensors
        input_rec = input_fp16.to(ctx.orig_dtype)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input_rec)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias

class SMODeltaLinear(nn.Linear):
    def __init__(self, original_linear):
        super().__init__(original_linear.in_features, original_linear.out_features, bias=original_linear.bias is not None)
        self.weight = original_linear.weight
        self.bias = original_linear.bias
    def forward(self, input):
        return SMODeltaLinearFunction.apply(input, self.weight, self.bias)

def wrap_model_delta(model):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, SMODeltaLinear(child))
        else:
            wrap_model_delta(child)
    return model

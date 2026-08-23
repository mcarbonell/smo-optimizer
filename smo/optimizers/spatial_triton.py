"""
smo/optimizers/spatial_triton.py — SMOTriton (Experimental Fused Kernel)

This module contains custom Triton kernels to fuse the operations of the
Super Mario Optimizer (Spatial version).

By fusing gradient reading, block-wise pooling (compression), momentum updates, 
and weight updates into a single GPU kernel, we eliminate memory bandwidth 
bottlenecks, making the optimizer mathematically faster than AdamW on GPU.

Note: This requires an NVIDIA GPU and the `triton` package.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def smo_spatial_step_kernel(
        weights_ptr,      # Pointer to the model parameters (weights)
        grads_ptr,        # Pointer to the gradients
        exp_avg_ptr,      # Pointer to compressed momentum state
        exp_avg_sq_ptr,   # Pointer to compressed variance state
        
        lr,               # Learning rate
        beta1,            # Adam beta1
        beta2,            # Adam beta2
        eps,              # Adam epsilon
        bias_correction1, # 1 - beta1^t
        bias_correction2, # 1 - beta2^t
        weight_decay,     # Weight decay
        
        # Dimensions
        N_ELEMENTS,       # Total number of elements in the parameter tensor
        N_COMP_ELEMENTS,  # Total number of compressed state elements
        BLOCK_SIZE: tl.constexpr, # Number of elements in one "compression block"
        
        # Triton metaparameters
        BLOCK_M: tl.constexpr   # Number of compressed blocks handled by one Triton program instance
    ):
        """
        Fused kernel for SMO Spatial (Vectorized V2).
        Handles BLOCK_M blocks of size BLOCK_SIZE in parallel.
        """
        pid = tl.program_id(axis=0)
        
        # 1. Offsets for compressed states
        state_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        state_mask = state_offsets < N_COMP_ELEMENTS
        
        # 2. Offsets for elements (2D grid: BLOCK_M rows, BLOCK_SIZE columns)
        row_offsets = state_offsets[:, None] * BLOCK_SIZE
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
        elem_offsets = row_offsets + col_offsets
        elem_mask = (state_offsets[:, None] < N_COMP_ELEMENTS) & (elem_offsets < N_ELEMENTS)
        
        # 3. Load weights and gradients (2D)
        w = tl.load(weights_ptr + elem_offsets, mask=elem_mask)
        g = tl.load(grads_ptr + elem_offsets, mask=elem_mask)
        
        if weight_decay != 0.0:
            g = g + weight_decay * w
            
        # 4. Pooling (Compress): Average across the BLOCK_SIZE dimension (axis 1)
        g_sum = tl.sum(g, axis=1) # (BLOCK_M,)
        valid_counts = tl.sum(tl.where(elem_mask, 1.0, 0.0), axis=1)
        # Avoid division by zero on fully masked blocks
        g_comp = g_sum / tl.where(valid_counts > 0, valid_counts, 1.0)
        
        # 5. Load and update compressed states
        m = tl.load(exp_avg_ptr + state_offsets, mask=state_mask)
        v = tl.load(exp_avg_sq_ptr + state_offsets, mask=state_mask)
        
        m_new = m * beta1 + g_comp * (1.0 - beta1)
        v_new = v * beta2 + (g_comp * g_comp) * (1.0 - beta2)
        
        tl.store(exp_avg_ptr + state_offsets, m_new, mask=state_mask)
        tl.store(exp_avg_sq_ptr + state_offsets, v_new, mask=state_mask)
        
        # 6. Decompress and apply update
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2
        
        m_hat_2d = m_hat[:, None]
        v_hat_2d = v_hat[:, None]
        
        denom = tl.sqrt(v_hat_2d) + eps
        step_update = (lr * m_hat_2d) / denom
        
        w_new = w - step_update
        tl.store(weights_ptr + elem_offsets, w_new, mask=elem_mask)


class SMOTriton(torch.optim.Optimizer):
    """
    Super Mario Optimizer using Fused Triton Kernels.
    Provides maximum GPU memory bandwidth efficiency.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.5):
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed or no NVIDIA GPU detected.")
            
        # For V1 of the kernel, we simplify k_ratio to a 1D block size.
        # k_ratio = 0.5 in 2D means blocks of 2x2. In 1D, that's 4 elements per block.
        # k_ratio = 0.25 in 2D means blocks of 4x4. In 1D, that's 16 elements per block.
        if k_ratio == 0.5:
            self.block_size = 4
        elif k_ratio == 0.25:
            self.block_size = 16
        else:
            raise ValueError("SMOTriton V1 only supports k_ratio 0.5 or 0.25")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]

                # Initialize states
                if len(state) == 0:
                    state['step'] = 0
                    # The compressed state size is (total_elements + block_size - 1) // block_size
                    comp_elements = (p.numel() + self.block_size - 1) // self.block_size
                    state['exp_avg'] = torch.zeros(comp_elements, dtype=p.dtype, device=p.device)
                    state['exp_avg_sq'] = torch.zeros(comp_elements, dtype=p.dtype, device=p.device)

                state['step'] += 1
                
                beta1, beta2 = group['betas']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Flatten tensors for the 1D Triton kernel
                p_flat = p.view(-1)
                g_flat = grad.view(-1)
                m = state['exp_avg']
                v = state['exp_avg_sq']
                
                n_elements = p.numel()
                comp_elements = m.numel()

                # Launch Triton Kernel
                # BLOCK_M represents how many state elements each Triton program will handle.
                # Let's say each program handles 32 compressed blocks.
                BLOCK_M = 32
                grid = lambda meta: (triton.cdiv(comp_elements, meta['BLOCK_M']), )
                
                smo_spatial_step_kernel[grid](
                    p_flat, g_flat, m, v,
                    group['lr'], beta1, beta2, group['eps'],
                    bias_correction1, bias_correction2, group['weight_decay'],
                    N_ELEMENTS=n_elements,
                    N_COMP_ELEMENTS=comp_elements,
                    BLOCK_SIZE=self.block_size,
                    BLOCK_M=BLOCK_M
                )

        return loss

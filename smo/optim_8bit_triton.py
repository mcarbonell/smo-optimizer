"""
smo/optim_8bit_triton.py — SMO8bitTriton

Fused Triton kernel for the 8-bit quantized Super Mario Optimizer.
Combines spatial compression and 8-bit block-wise quantization 
into a single highly efficient GPU kernel.
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
    def smo_8bit_step_kernel(
        weights_ptr,      # Pointer to the model parameters (weights)
        grads_ptr,        # Pointer to the gradients
        m_q_ptr,          # Pointer to quantized momentum state (int8)
        m_s_ptr,          # Pointer to momentum scales (fp32)
        v_q_ptr,          # Pointer to quantized variance state (int8)
        v_s_ptr,          # Pointer to variance scales (fp32)
        
        lr,               # Learning rate
        beta1,            # Adam beta1
        beta2,            # Adam beta2
        eps,              # Adam epsilon
        bias_correction1, # 1 - beta1^t
        bias_correction2, # 1 - beta2^t
        weight_decay,     # Weight decay
        
        N_ELEMENTS,       # Total number of parameter elements
        N_COMP_ELEMENTS,  # Total number of compressed state elements
        
        SPATIAL_BLOCK_SIZE: tl.constexpr, # Elements per compressed state
        QUANT_BLOCK_SIZE: tl.constexpr    # Elements per quantization block
    ):
        # Each program handles exactly one Quantization Block
        pid = tl.program_id(axis=0)
        
        # 1. State element offsets for this quantization block
        state_offsets = pid * QUANT_BLOCK_SIZE + tl.arange(0, QUANT_BLOCK_SIZE)
        state_mask = state_offsets < N_COMP_ELEMENTS
        
        # 2. Parameter offsets corresponding to these state elements
        row_offsets = state_offsets[:, None] * SPATIAL_BLOCK_SIZE
        col_offsets = tl.arange(0, SPATIAL_BLOCK_SIZE)[None, :]
        elem_offsets = row_offsets + col_offsets
        elem_mask = (state_offsets[:, None] < N_COMP_ELEMENTS) & (elem_offsets < N_ELEMENTS)
        
        # 3. Load quant scales (scalar per pid)
        m_s = tl.load(m_s_ptr + pid)
        v_s = tl.load(v_s_ptr + pid)
        
        # 4. Load quantized states
        m_q = tl.load(m_q_ptr + state_offsets, mask=state_mask)
        v_q = tl.load(v_q_ptr + state_offsets, mask=state_mask)
        
        # Dequantize to float32
        m = m_q.to(tl.float32) * (m_s / 127.0)
        v = v_q.to(tl.float32) * (v_s / 127.0)
        
        # 5. Load weights and gradients
        w = tl.load(weights_ptr + elem_offsets, mask=elem_mask)
        g = tl.load(grads_ptr + elem_offsets, mask=elem_mask)
        
        if weight_decay != 0.0:
            g = g + weight_decay * w
            
        # 6. Spatial pooling (Compress)
        g_sum = tl.sum(g, axis=1) # (QUANT_BLOCK_SIZE,)
        valid_counts = tl.sum(tl.where(elem_mask, 1.0, 0.0), axis=1)
        g_comp = g_sum / tl.where(valid_counts > 0, valid_counts, 1.0)
        
        # 7. Update moments
        m_new = m * beta1 + g_comp * (1.0 - beta1)
        v_new = v * beta2 + (g_comp * g_comp) * (1.0 - beta2)
        
        # 8. Re-quantize moments
        # Find max abs value per block
        m_abs = tl.abs(m_new)
        v_abs = tl.abs(v_new)
        
        m_abs_masked = tl.where(state_mask, m_abs, 0.0)
        v_abs_masked = tl.where(state_mask, v_abs, 0.0)
        
        m_s_new = tl.max(m_abs_masked, axis=0)
        v_s_new = tl.max(v_abs_masked, axis=0)
        
        # Prevent division by zero
        m_s_new = tl.where(m_s_new < 1e-12, 1e-12, m_s_new)
        v_s_new = tl.where(v_s_new < 1e-12, 1e-12, v_s_new)
        
        # Scale back to [-127, 127] and round
        m_scaled = (m_new / m_s_new) * 127.0
        v_scaled = (v_new / v_s_new) * 127.0
        
        m_q_new = tl.where(m_scaled > 0, m_scaled + 0.5, m_scaled - 0.5).to(tl.int8)
        v_q_new = tl.where(v_scaled > 0, v_scaled + 0.5, v_scaled - 0.5).to(tl.int8)
        
        # Store updated states and scales
        tl.store(m_s_ptr + pid, m_s_new)
        tl.store(v_s_ptr + pid, v_s_new)
        tl.store(m_q_ptr + state_offsets, m_q_new, mask=state_mask)
        tl.store(v_q_ptr + state_offsets, v_q_new, mask=state_mask)
        
        # 9. Weight update
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2
        
        # Broadcast decompressed states to parameter shape
        m_hat_2d = m_hat[:, None]
        v_hat_2d = v_hat[:, None]
        
        # Ensure v_hat is non-negative before sqrt
        v_hat_2d = tl.where(v_hat_2d < 0.0, 0.0, v_hat_2d)
        denom = tl.sqrt(v_hat_2d) + eps
        step_update = (lr * m_hat_2d) / denom
        
        w_new = w - step_update
        tl.store(weights_ptr + elem_offsets, w_new, mask=elem_mask)


class SMO8bitTriton(torch.optim.Optimizer):
    """
    Super Mario Optimizer 8-bit Quantized using Fused Triton Kernels.
    Provides maximum GPU memory bandwidth efficiency and lowest memory footprint.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, k_ratio=0.25, block_size=64):
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed or no NVIDIA GPU detected.")
            
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        # V1: map 2D k_ratio to 1D SPATIAL_BLOCK_SIZE
        if k_ratio == 0.5:
            self.spatial_block_size = 4
        elif k_ratio == 0.25:
            self.spatial_block_size = 16
        else:
            raise ValueError("SMO8bitTriton V1 only supports k_ratio 0.5 or 0.25")
            
        # Ensure block_size is a power of 2 for Triton efficiency
        if not (block_size != 0 and ((block_size & (block_size - 1)) == 0)):
            raise ValueError(f"block_size must be a power of 2, got {block_size}")
            
        self.quant_block_size = block_size
            
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
                    
                    n_elements = p.numel()
                    comp_elements = (n_elements + self.spatial_block_size - 1) // self.spatial_block_size
                    num_quant_blocks = (comp_elements + self.quant_block_size - 1) // self.quant_block_size
                    
                    # Quantized states (int8)
                    state['m_q'] = torch.zeros(comp_elements, dtype=torch.int8, device=p.device)
                    state['v_q'] = torch.zeros(comp_elements, dtype=torch.int8, device=p.device)
                    
                    # Scales (fp32) - one per quant block
                    state['m_s'] = torch.zeros(num_quant_blocks, dtype=torch.float32, device=p.device)
                    state['v_s'] = torch.zeros(num_quant_blocks, dtype=torch.float32, device=p.device)

                state['step'] += 1
                
                beta1, beta2 = group['betas']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Flatten tensors
                p_flat = p.view(-1)
                g_flat = grad.view(-1)
                
                n_elements = p.numel()
                comp_elements = state['m_q'].numel()
                num_quant_blocks = state['m_s'].numel()

                # Launch Triton Kernel
                # 1 program = 1 quantization block
                grid = lambda meta: (num_quant_blocks, )
                
                smo_8bit_step_kernel[grid](
                    p_flat, g_flat,
                    state['m_q'], state['m_s'],
                    state['v_q'], state['v_s'],
                    group['lr'], beta1, beta2, group['eps'],
                    bias_correction1, bias_correction2, group['weight_decay'],
                    N_ELEMENTS=n_elements,
                    N_COMP_ELEMENTS=comp_elements,
                    SPATIAL_BLOCK_SIZE=self.spatial_block_size,
                    QUANT_BLOCK_SIZE=self.quant_block_size
                )

        return loss

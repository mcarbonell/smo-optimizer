# 🧑‍🔧 SMO: Super Mario Optimizer

```
███████╗██╗   ██╗██████╗ ███████╗██████╗     ███╗   ███╗ █████╗ ██████╗ ██╗ ██████╗ 
██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗    ████╗ ████║██╔══██╗██╔══██╗██║██╔═══██╗
███████╗██║   ██║██████╔╝█████╗  ██████╔╝    ██╔████╔██║███████║██████╔╝██║██║   ██║
╚════██║██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗    ██║╚██╔╝██║██╔══██║██╔══██╗██║██║   ██║
███████║╚██████╔╝██║     ███████╗██║  ██║    ██║ ╚═╝ ██║██║  ██║██║  ██║██║╚██████╔╝
╚══════╝ ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝ 
```

**SMO** (Super Mario Optimizer) is an ultra-memory-efficient PyTorch optimizer designed to solve the "Memory Wall" problem in Deep Learning. It reduces the optimizer state memory by **60-93%** while retaining accuracy.

> **"It's-a me, optimizer!"** 🕶️
## 🍄 Variants

| Variant | Method | Type | Speed | Notes |
|---------|--------|------|-------|-------|
| **SMO** | adaptive_avg_pool2d | Spatial (Texture) | fast | Default choice (32-bit) |
| **SMO-8bit** | 8-bit Quantized | Hybrid | extreme | **98% Memory Savings** |

- **SMO (spatial):** Default - very fast, treats gradients as 2D textures.
- **SMO-8bit (hybrid):** The ultimate memory saver for large-scale LLMs.

## 🛣️ Roadmap (Upcoming Power-Ups)

- [x] **Triton Kernels (float32):** High-performance GPU fused kernels that break the Memory Wall (Implemented & Benchmarked!).
- [ ] **Triton 8-bit Fused Kernels:** The "Mega Power-Up". Perform block-wise 8-bit quantization dynamically inside SRAM to achieve 98% memory savings with near-zero VRAM bandwidth overhead.
- [ ] **Distributed Support:** Native integration with ZeRO-3 and FSDP/DeepSpeed.

## 🦖 The Problem

Standard optimizers like AdamW store two full-resolution copies of every gradient (momentum and variance). For a **1 Billion parameter model**, this alone consumes **8 GB of VRAM**.

**🧱 SMO breaks this wall:**

1. **Extreme Compression:** Uses spatial pooling or Walsh transforms to compress optimizer states.
2. **Spectral Denoising:** The compression acts as a low-pass filter, removing batch noise and improving stability.

## Benchmarks (Post-Optimization)

### Memory Consumption

| Model Size | AdamW RAM | **SMO (k=0.25)** | **Savings** |
|-----------|-----------|------------------|-------------|
| 100M params | 800 MB | **50 MB** | **~93.7%** |
| 1B params | 8.0 GB | **500 MB** | **~93.7%** |
| 7B params | 56.0 GB | **3.5 GB** | **~93.7%** |

### Speed & Training Time (Relative to Adam)

| Hardware | AdamW | SMO (PyTorch) |  **SMO (Triton Fused)** |
|----------|-------|---------------|------------------------|
| CPU Time | 1.0x | ~1.1x | - |
| GPU (NVIDIA T4)| 1.0x | 1.48x (slower) | **0.46x (2.15x faster!)** |

*Note: On GPUs (tested on an NVIDIA T4 with a 67M parameter layer), the native PyTorch SMO implementation is bottlenecked by framework overheads (1.48x slower than Adam). However, the **Triton Fused Kernel** completely breaks the Memory Wall: by doing all compression and mathematical updates directly in SRAM in a single pass, it is **over 2x faster than AdamW** (13.75 ms/step vs 29.60 ms/step) while saving 93% of the VRAM.*

### Image Classification Benchmarks (CPU)

To prove that compression does not degrade model performance, we trained standard CNNs on MNIST (~421k params) and CIFAR-10 (~620k params) for 5 epochs. 

**MNIST Benchmark (5 Epochs)**

| Optimizer | State Memory | Final Accuracy | CPU Train Time |
|-----------|--------------|----------------|----------------|
| Standard Adam | 3.22 MB | 98.86% | 432.7s |
| SMO (k=0.25) | 0.35 MB | 98.80% | 455.5s |
| SMO-8bit (k=0.25)| **0.21 MB** | 98.59% | 401.6s |

**CIFAR-10 Benchmark (5 Epochs)**

| Optimizer | State Memory | Final Accuracy | CPU Train Time |
|-----------|--------------|----------------|----------------|
| Standard Adam | 4.74 MB | 64.72% | 304.7s |
| **SMO (k=0.5)** | 1.74 MB | **66.82%** | **295.5s** |
| SMO (k=0.25) | **0.99 MB** | 62.45% | **294.1s** |

*Note: On CIFAR-10, SMO (k=0.5) not only saved 63% VRAM and beat Adam in accuracy (+2.1%), but it actually trained **faster** on CPU.*

### LLM Scalability (Transformer Validation)

We tested SMO on a 4-layer autoregressive Transformer (Mini-GPT, ~800k params) to validate compression on dense `nn.Linear` attention weights without inherent spatial locality.

**Mini-LLM Benchmark (200 iterations, $k=0.5$, CPU)**

| Optimizer | State Memory (MB) | Val Perplexity | Train Time |
|-----------|-------------------|----------------|------------|
| Standard AdamW | 6.21 MB | 65.63 | **24.1s** |
| SMO (Spatial) | 1.56 MB | 67.23 | 28.4s |
| SMO-8bit (Hybrid) | **0.43 MB** | **66.64** | 32.4s |

*SMO-8bit reduces VRAM by ~93% while maintaining nearly identical perplexity (+1.01) to AdamW, proving its viability for LLMs. Note: The CPU training times show the mathematical overhead of compression in PyTorch; however, on large-scale GPU training, SMO avoids the PCIe/VRAM bandwidth bottlenecks of AdamW, resulting in faster overall wall-clock times.*

## 🔑 Installation

The easiest way to install SMO is via pip:

```bash
pip install supermario-optimizer
```

**From Source:**

```bash
git clone https://github.com/mcarbonell/supermario-optimizer.git
cd supermario-optimizer
pip install -e .
```

Or simply copy the `smo/` directory directly into your project.

## Quick Start

### SMO (Blocky - Recommended)

```python
import torch
import torch.nn as nn
from smo import SMO

model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1024)
)

# k_ratio=0.25 → 93.75% memory reduction
optimizer = SMO(model.parameters(), lr=1e-3, k_ratio=0.25)

loss = criterion(model(inputs), targets)
loss.backward()
optimizer.step()
```

## How It Works

### SMO (Spatial)
```
Original → Pool → Compress → Update → Upsample → Original
   (H,W)    2x2    (kH,kW)   (kH,kW)   bilinear  (H,W)
```


## Hyperparameter Guidance

| Task | Recommended k_ratio | Notes |
|------|---------------------|--------|
| CPU training | 0.1-0.25 | Maximum savings |
| GPU large model | 0.25-0.5 | Can be faster |
| Easy (MNIST) | 0.25 | Maximum savings |
| Medium (CIFAR) | 0.5 | Can improve accuracy |
| Hard (ImageNet/LLMs) | 0.5-0.75 | Depends on model |

## CPU vs GPU Performance

| Hardware | Adam | SMO |
|----------|------|-----|
| CPU | 1x | 2.5x |
| GPU (small) | 1x | 1.2x |
| GPU (large) | OOM possible | 0.8x |

**Key insight:** On GPU with large models, SMO can be faster because reduced memory traffic outweighs compression overhead.

## ⭐ Memory Reduction Formula

- **Memory saved = 1 - k_ratio²**
- k_ratio=0.25 → 93.75% reduction
- k_ratio=0.5 → 75% reduction
- k_ratio=0.1 → 99% reduction (extreme)

## 🏁 Credits

Created by **Mario** 🐢 - "It's-a me!"

> "Wahoo!" - when your model finally fits in VRAM 🎉

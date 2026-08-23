# SMO — State Memory Optimizer

[![Status: Active Research](https://img.shields.io/badge/status-active_research-green)](docs/ROADMAP.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

SMO is a research repository for memory-efficient optimizer experiments in PyTorch.

The core idea is to reduce optimizer-state memory by compressing first- and second-order moments, with current work spanning:

- **Spatial optimizer-state compression** (SMO-Spatial, SMO-8bit)
- **Spectral optimizer-state compression** (SMO-Walsh, SMO-DCT) — experimental
- **Activation-memory compression** — early experiments
- **Triton kernels** for NVIDIA GPUs — future work

---

## 🎯 Key Results (CPU, seed=1234)

### Memory Savings vs Accuracy Trade-off

| Dataset | Model | Adam (baseline) | SMO k=0.25 | SMO-8bit k=0.25 |
|---------|-------|----------------|------------|-----------------|
| **MNIST** | SimpleCNN | 99.04% / 3.22 MB | 98.90% / 0.35 MB **(89.1% savings)** | 98.97% / **0.21 MB (93.5%)** |
| **CIFAR-10** | CIFAR_CNN | 66.91% / 4.74 MB | 63.59% / 0.99 MB **(79.1%)** | — |
| **MiniGPT** | 4-layer Transformer | PPL 65.31 / 6.21 MB | — | PPL 66.58 / **0.43 MB (93.1%)** |

**Spectral Variants (CIFAR-10, 3 epochs):**
- **Walsh Pure k=0.5**: 64.47% acc / 1.74 MB → **63.3% savings**, **beats Adam by +1.44%**
- Walsh Hybrid k=0.5: 61.24% / 1.74 MB (−1.79%)
- DCT Hybrid k=0.5: 62.74% / 1.74 MB (−0.29%)
- DCT Pure k=0.5: 63.10% / 1.74 MB → **63.3% savings**, (−0.07% vs Adam, now fixed)

> **Takeaway:** Spatial+8bit compression achieves **>90% memory reduction** on simple/medium tasks with minimal accuracy loss. On harder tasks (CIFAR-10) trade-off is larger (−3% at k=0.25) but improves with less aggressive compression (−2% at k=0.5). Surprisingly, Walsh Pure outperforms Adam on CIFAR-10 — worth deeper investigation.

> **⚠️ Memory claims — read the fine print:** savings are measured on the **persistent optimizer state** (what `state_dict()`/a checkpoint stores). During training, SMO keeps full-resolution reconstruction buffers cached for speed, so the **resident working set is larger** than the persistent state. Definitions in `benchmarks/METHODOLOGY.md`.

> **Coverage note:** only 2D parameters with both dims ≥ 32 are compressed. 4D conv weights and 1D biases fall back to dense Adam moments — this is why CIFAR-10 (conv-heavy) saves less than MNIST/MiniGPT (linear-heavy).

---

## 📂 Repository Structure

```
smo_optimizer/
├── smo/
│   ├── optimizers/      # Stable: SMO-Spatial, SMO-Spatial-8bit
│   ├── activations/     # Experimental: activation compression
│   ├── experimental/    # Research: spectral (Walsh, DCT), Triton
│   └── utils/           # Shared utilities
├── benchmarks/
│   ├── suites/          # Canonical benchmark entrypoints
│   │   ├── optimizer_step/   # Microbench: step time, memory
│   │   ├── training/         # End-to-end: MNIST, CIFAR-10, MiniGPT
│   │   ├── activations/      # Activation compression tests
│   │   ├── spectral/         # Spectral variant baselines
│   │   └── comparison/       # Multi-seed, multi-hardware
│   ├── runners/         # Hardware-specific launchers (Modal, DirectML)
│   ├── results/         # JSON outputs (aggregate + per-run)
│   ├── METHODOLOGY.md   # Benchmarking standards
│   └── CATALOG.md       # Inventory of all suites
├── profiles/            # Profiling scripts (torch.profiler)
├── docs/
│   ├── PROJECT_FOUNDATION.md  # Naming, taxonomy, layering
│   └── ROADMAP.md             # Phase-by-phase progress
├── tests/               # Unit tests
├── CHANGELOG.md         # Version history
└── TODO.md             # Current task tracking
```

---

## 📋 Documentation

- **Project Foundation** — naming decisions, variant taxonomy, repository organization  
  → `docs/PROJECT_FOUNDATION.md`
- **Roadmap** — Phase 1–6 progress, current status, next actions  
  → `docs/ROADMAP.md`
- **Benchmark Methodology** — seeding policy, hardware matrix, result format  
  → `benchmarks/METHODOLOGY.md`
- **Benchmark Catalog** — canonical suites, categories, legacy wrappers  
  → `benchmarks/CATALOG.md`
- **Changelog** — versioned releases and unreleased changes  
  → `CHANGELOG.md`
- **Task Tracking** — in-progress work and blockers  
  → `TODO.md`

---

## 🚀 Quick Start

### Running Benchmarks

```bash
# Microbenchmark: optimizer step time (multi-shape, multi-seed)
python -m benchmarks.suites.optimizer_step.benchmark_step_time \
    --shapes 256,512,1024 --seeds 1234,5678,9012 --device cpu

# End-to-end: MNIST (SMO vs Adam)
python -m benchmarks.suites.training.benchmark_mnist --epochs 5 --seed 1234

# End-to-end: CIFAR-10
python -m benchmarks.suites.training.benchmark_cifar10 --epochs 5 --seed 1234

# End-to-end: 8-bit variant (MNIST)
python -m benchmarks.suites.training.benchmark_8bit --epochs 5 --seed 1234

# End-to-end: MiniGPT smoke (200 iters)
python -m benchmarks.suites.training.benchmark_minillm --max_iters 200 --seed 1234

# Spectral variants baseline (CIFAR-10, 3 epochs)
python -m benchmarks.suites.spectral.benchmark_spectral_cpu --epochs 3 --seed 1234

# Single-GPU (T4/Colab/Kaggle): peak memory + quality vs AdamW & bnb-8bit
python -m benchmarks.suites.comparison.t4_memory_benchmark --suite gpt --steps 1000 --amp --seed 1234

# Profiling: step breakdown (CPU)
python profiles/profile_smo_step.py --shape 1024,1024 --steps 100 --seed 1234
```

All results saved to `benchmarks/results/` as JSON bundles.

---

## 🧪 Current Status

**Phase 2 — Correctness & Measurement:** ✅ COMPLETE (2026-05-06)
- Deterministic seeding across 12+ canonical benchmarks
- Multi-seed aggregation (mean±std) in microbenchmarks
- GPU sync infrastructure (CUDA/DirectML)
- Memory accounting corrected (temp buffers excluded)
- Baseline suite executed: MNIST, CIFAR-10, MiniGPT, Spectral variants

**Phase 3 — Bottleneck Analysis:** 🔄 AWAITING GPU
- Profiling suite `profiles/profile_smo_step.py` ready (manual timers + torch.profiler)
- CPU profiling data: compression 41%, upsampling 31%, update 3%
- Blocked: local DirectML GPU >90% utilization

**Phase 4 — Optimization Work:**
- ✅ **4.1 Buffer reuse** — 8–24% CPU speedup, memory accounting fixed
- ⏳ **4.2 Post-profiling** — pending GPU data (pooling kernel eval, upsampling bypass)

---

## 📈 Performance Highlights

**SMO-Spatial (CPU step time, after buffer reuse):**

| Shape | AdamW | SMO (antes) | SMO (después) | Mejora |
|-------|-------|-------------|---------------|--------|
| 256×256 | 1.43 ms | 3.87 ms | 2.93 ms | **−24%** |
| 512×512 | 4.67 ms | 9.97 ms | 8.97 ms | **−10%** |
| 1024×1024 | 20.16 ms | 32.36 ms | 29.86 ms | **−8%** |

**Memory Compression (k_ratio=0.25):**

| Task | Adam State | SMO State | Reduction |
|------|------------|-----------|-----------|
| MNIST (2 linear layers) | 3.22 MB | 0.35 MB | 89.1% |
| CIFAR-10 (2 linear + 3 conv) | 4.74 MB | 0.99 MB | 79.1% |
| MiniGPT (8 linear) | 6.21 MB | 0.43 MB (8bit) | 93.1% |

---

## 🐛 Known Issues

- **Accuracy gap on complex tasks**: CIFAR-10 gap −3.32% at k=0.25; consider k=0.5 for better quality (−2.08% gap)
- **GPU profiling pending**: DirectML machine at high utilization; will run profiling when <50%. Note: manual timers now synchronize on CUDA; DirectML has no sync API, so timings there remain wall-clock approximations
- **MNIST headline numbers pre-date the seeding policy**: the tracked `benchmark_results/` JSONs were recorded 2026-05-05 with `"seed": null` (the script seeds RNGs but did not record it). CIFAR-10, MiniGPT, 8-bit and spectral results are seed=1234. Re-run pending (see TODO.md)
- **Conv layers not compressed**: see coverage note above

---

## 📜 License

MIT — see `LICENSE` for details.

---

## 🙋 Contributing

This is a research project. For issues, see the TODO tracking in `TODO.md`. For questions, open a GitHub issue.

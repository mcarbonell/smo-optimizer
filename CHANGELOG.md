# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Conv-weight pooling experiment (CPU day) — negative result, code kept opt-in**:
  - `compression_view` + `compress_conv` flag on SMO/SMO8bit (and
    `--compress_conv` on the T4 benchmark): pools 4D conv weights via the
    flattened `(out_c, in_c*kh*kw)` row view, incl. `low_peak` banded path
  - Measured verdict (CIFAR-CNN 3ep s1234): conv pooling costs the SMO family
    −12…−17 acc vs dense fallback (46.9/52.3 vs historical 63.6/64.8) — H4's
    locality prior does not hold across unrelated input channels; default stays
    dense-conv
- **Dead-zone lift in SMO8bit quantization**: round-to-nearest used to flush
  sub-half-LSB moment entries to exact zero → v=0 → denominator ~eps → divergent
  updates (reproduced NaN losses on CNNs at block_size=64). Non-zero values are
  never encoded as zero anymore (+/-1 LSB worst-case bias)
- Row-aligned quantization blocks for conv-view tensors (`state['q_block'] =
  min(block_size, comp_w)`); linear matrices keep the historical flattened layout
- **30-epoch convergence-budget table, multi-seed** (`docs/T4_FINDINGS.md`):
  SMO k=0.5 ties bnb-AdamW8bit at matched persistent bytes (80.29±0.55 vs
  80.13±0.79, z≈0.3); SMO-8bit k=0.5 −0.42 with 28% of bnb's state bytes;
  adamw@6e-4 completed to n=3 (73.39±0.77); smo8bit@1.5e-3 n=3 (78.30±0.53).
  Mechanism reading: the k=0.25 long-budget gap was compression capacity,
  not smoothing dynamics
- `t4_lr_sweep --horizon PREFIX`: restrict analysis/report to runs whose
  @horizon starts with the prefix (silences stale rows from other lengths)
- **T4 GPU campaign tooling** (`benchmarks/suites/comparison/`):
  - `t4_memory_benchmark.py` — peak-memory + quality comparison vs AdamW/bnb-8bit/SGD-M;
    OOM-tolerant per optimizer; trajectory logging; per-group `compress` opt-out;
    `--protect_output`, `--low_peak`, `--permute_basis`, `--tag`
  - `t4_lr_sweep.py` — per-optimizer LR fairness sweep, best-tuned ranking, resume-safe
  - `t4_summarize.py` — mean±std aggregation across seed bundles (dedup by seed)
  - `t4_loss_matched.py` — loss-matched generalization analysis from trajectories
  - `runners/colab/t4_benchmark_colab.ipynb` — one-click Colab/Kaggle notebook
- SMO8bit `low_peak=True`: row-banded compress/update with no full-size temporaries
  (step peak ~21 → ~9 B/param; bit-exact vs monolithic in tests)
- MIT `LICENSE`; `pyproject.toml` (`testpaths=tests`)
- Spectral transform unit tests (`tests/test_spectral_transforms.py`, ported from scratch)
- Deterministic seeding across all canonical benchmarks (`--seed` / `--seeds` flags)
- Multi-seed aggregation (mean±std) in microbenchmarks
- GPU synchronization infrastructure (`get_sync_fn` for CUDA/DirectML/CPU)
- Memory reporting for CUDA in optimizer-step microbench
- **Profiling suite** `profiles/profile_smo_step.py`: manual timer decomposition + torch.profiler integration

### Changed
- Persistent-state accounting now walks nested containers (fixes bnb reporting 0 MB)
- Benchmark banner records lr/k_ratio; results carry lr for sweep analysis
- README rewritten around T4 campaign evidence

### Fixed
- SMO8bit dequantization dtype (fp16/bf16 params no longer crash)
- Hadamard recursion bug in legacy spectral scratch tests
- find_packages leaking tests/benchmarks/spectral into site-packages

### Changed
- `benchmarks/suites/optimizer_step/benchmark_step_time.py`: multi-shape (64–4096), device-aware, warmup + memory stats
- `benchmarks/suites/training/benchmark_mnist.py`: now records seed and uses deterministic RNG
- `benchmarks/suites/training/benchmark_cifar10.py`: added argparse, seeding, results JSON output
- `benchmarks/suites/training/benchmark_8bit.py`: added argparse, seeding, results JSON output
- `benchmarks/suites/spectral/benchmark_spectral_cpu.py`: added set_seed, argparse, results recording
- `benchmarks/suites/activations/benchmark_activations.py`: multi-seed aggregation, device-aware
- `benchmarks/suites/activations/test_accuracy_activations.py`: added argparse + seed
- `benchmarks/suites/activations/test_accuracy_delta.py`: added argparse + seed
- `benchmarks/suites/training/benchmark_minillm.py`: added argparse + seed
- `benchmarks/suites/optimizer_step/smoke_spatial_consistency.py`: added argparse + seed + exit code

### Fixed
- SMO-Spatial buffer allocation overhead: pre-allocated intermediate buffers (8-24% speedup on CPU, shape-dependent)
- SMO-Spatial-8bit algorithmic consistency verified (block_size=1 exact match)

### Performance
- **SMO-Spatial CPU step time reduced by 8-24%** via buffer reuse:
  - 256×256: −24.2% (3.865 ms → 2.928 ms)
  - 512×512: −10.0% (9.968 ms → 8.965 ms)
  - 1024×1024: −7.8% (32.363 ms → 29.856 ms)
- **Memory savings baselines**:
  - MNIST (Simple CNN):
    - Adam: 3.22 MB, 99.04% acc
    - SMO k=0.25: 0.35 MB → **89.1% reduction**, 98.90% acc (−0.14% gap)
    - SMO k=0.5: 0.92 MB → 71.5% reduction, 99.13% acc
    - SMO-8bit k=0.25: **0.21 MB** → **93.5% reduction**, 98.97% acc (quality maintained)
  - CIFAR-10 (CIFAR_CNN):
    - Adam: 4.74 MB, 66.91% acc
    - SMO k=0.25: 0.99 MB → **79.1% reduction**, 63.59% acc (−3.32% gap)
    - SMO k=0.5: 1.74 MB → **63.3% reduction**, 64.83% acc (−2.08% gap)
  - MiniGPT (Transformer, 200 iters, smoke):
    - AdamW: 6.21 MB, PPL 65.31
    - SMO k=0.5: 1.56 MB → **74.9% reduction**, PPL 66.58 (+1.27)
    - SMO-8bit k=0.5: **0.43 MB** → **93.1% reduction**, PPL 66.58 (same as SMO)
  - Spectral variants (CIFAR-10, 3 epochs):
    - AdamW: 4.74 MB, 63.03% acc
    - SMOWalsh Pure k=0.5: 1.74 MB → **63.3% reduction**, **64.47% acc** (+1.44% vs Adam)
    - SMOWalsh Hybrid k=0.5: 1.74 MB, 61.24% acc (−1.79% vs Adam)
    - SMODCT Hybrid k=0.5: 1.74 MB, 62.74% acc (−0.29% vs Adam)
    - SMODCT Pure k=0.5: 1.74 MB → **63.3% reduction**, **63.10% acc** (−0.07% vs Adam, now fixed)
- Buffer memory accounting fixed: temporary buffers moved to `_param_buffers` (not counted in optimizer state)

### Documentation
- `docs/ROADMAP.md`: Phase 2 marked complete; Phase 4 buffer reuse documented; profiling suite added
- `docs/PROJECT_FOUNDATION.md`: taxonomy and layering guidance
- `benchmarks/METHODOLOGY.md`: seeding policy (Rule 2: ≥3 seeds for public claims)

## [0.1.0] - 2026-05-05 (initial structured development)

### Added
- Repository consolidation under `smo/`, `benchmarks/suites/`, `benchmarks/runners/`
- Benchmark catalog (`benchmarks/CATALOG.md`) and results utils (`benchmarks/results_utils.py`)
- Unit tests for spatial optimizers (`tests/test_spatial_optimizers.py`)
- Smoke check for SMO vs SMO8bit consistency (`benchmarks/suites/optimizer_step/smoke_spatial_consistency.py`)
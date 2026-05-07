# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Deterministic seeding across all canonical benchmarks (`--seed` / `--seeds` flags)
- Multi-seed aggregation (mean±std) in microbenchmarks
- GPU synchronization infrastructure (`get_sync_fn` for CUDA/DirectML/CPU)
- Memory reporting for CUDA in optimizer-step microbench
- **Profiling suite** `profiles/profile_smo_step.py`: manual timer decomposition + torch.profiler integration

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
    - SMODCT Pure k=0.5: 4.74 MB, 14.28% acc (FAIL — requires investigation)
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
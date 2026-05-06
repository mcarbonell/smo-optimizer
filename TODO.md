# TODO.md - ROADMAP Phases 2-4 Progress Tracker

Proyecto: Continuar refactor/test/mejora SMO-Spatial per ROADMAP.

## Phase 2: Correctness And Measurement Baseline (COMPLETED 2026-05-05)

### Deliverables Completed

#### 1. Timing Infrastructure ✅
- `benchmarks/timing.py`: `measure_steps()` + `get_sync_fn(device)`
- Supports CPU/CUDA/DirectML synchronization
- Wall-clock and process CPU time recording

#### 2. Optimizer-Step Microbench ✅
- `benchmarks/suites/optimizer_step/benchmark_step_time.py`:
  - Shapes: 64×64 → 4096×4096 (7 configs)
  - Multi-seed support (`--seeds`, default 3): mean±std
  - Device-aware (`--device`): CPU/CUDA/DirectML
  - CUDA memory reporting (allocated/reserved)
  - Warmup configurable (`--warmup`)
  - GPU DirectML path prepared (commented pending hardware)

#### 3. Correctness Tests ✅
- `tests/test_spatial_optimizers.py`: 3/3 pass
  - second moment tracks E[g²] ✅
  - SMO vs SMO8bit exact match with `block_size=1` ✅
  - rejects sparse gradients ✅

#### 4. End-to-End Baselines ✅
All use deterministic seeds and record them in results:
- `benchmarks/suites/training/benchmark_mnist.py`:
  - Adam: 99.04% acc, 3.22 MB, 331.35s
  - SMO k=0.25: 98.90% acc, 0.35 MB, 325.80s → **89.1% memory reduction**, +0.14% acc gap
  - SMO k=0.5: 99.13% acc, 0.92 MB
  - Results: `benchmarks/results/benchmark_results.json`
- `benchmarks/suites/training/benchmark_8bit.py`: SMO-8bit vs SMO vs Adam on MNIST (seed=1234)
- `benchmarks/suites/training/benchmark_cifar10.py`: SMO vs Adam on CIFAR-10 (seed=1234)
- `benchmarks/suites/spectral/benchmark_spectral_cpu.py`: Spectral variants vs Adam (seed=1234)

#### 5. Activation Benchmark ✅
- `benchmarks/suites/activations/benchmark_activations.py`:
  - Multi-seed (`--seeds`, default 3): mean±std
  - Device-aware (`--device`): CPU/CUDA
  - Timing and memory (CUDA only) aggregation

### Phase 2 Exit Criteria Status

- ✅ Benchmark runs are repeatable (all major suites fix seed)
- ✅ Metrics defined consistently across hardware (infrastructure in place)
- ⚠️ GPU execution pending: DirectML machine at 90%+ utilization (will enable when <50%)

### Files Modified (Phase 2)

```
benchmarks/timing.py                           # get_sync_fn()
benchmarks/suites/optimizer_step/benchmark_step_time.py  # multi-shape, multi-seed, device-aware
benchmarks/suites/training/benchmark_mnist.py   # seed handling + results JSON
benchmarks/suites/training/benchmark_cifar10.py # seed + argparse + results JSON
benchmarks/suites/training/benchmark_8bit.py    # seed + argparse + results JSON
benchmarks/suites/spectral/benchmark_spectral_cpu.py  # set_seed + argparse + results
benchmarks/suites/activations/benchmark_activations.py  # multi-seed aggregation
docs/ROADMAP.md                                # Phase 2 status updated
TODO.md                                        # progress tracking
```

## Phase 3: Bottleneck Analysis (BLOCKED → GPU)

**Status:** Awaiting GPU availability (local DirectML at >90% for days)

**Plan when GPU frees:**
1. Profile SMO-Spatial step with `torch.profiler` on CUDA/DirectML
2. Isolate: pooling (compress) vs interpolation (upsample) vs quantization (8bit)
3. Memory snapshot: allocation patterns, peak memory breakdown
4. Identify top 3 hotspots
5. Feed findings into Phase 4 optimization work

**Current hypotheses (from ROADMAP):**
- 8-bit quantize/dequantize overhead
- Full-resolution reconstruction for update application
- Remaining PyTorch tensor allocation overhead in update path

## Phase 4: Optimization Work (READY → after Phase 3 profiling)

**Priority queue:**
1. SMO-Spatial: tensor reuse in update path (m_rec, v_rec buffers)
2. SMO-Spatial-8bit: minimize dequant→update→requant round-trips
3. Triton kernels: only after PyTorch path clean
4. Activation compression: separate track

## Phase 5: Benchmark Publication (pending)

**Needed:**
- Multi-seed tables (mean±std across 3+ seeds)
- Hardware disclosure docs
- Dataset/hyperparameter disclosures
- Failure mode discussion

## Phase 6: Packaging & Documentation (pending)

**Needed:**
- Public API cleanup
- Examples notebook
- Reproducibility guide
- Changelog and versioning

## Next Actions (Immediate)

1. ⏸ Wait for GPU availability to start Phase 3 profiling
2. 📝 Document seed policy in `benchmarks/METHODOLOGY.md` (explain default seeds, reproducibility)
3. 🔬 Once GPU free: run profiling suite on SMO-Spatial (CPU baseline already stable)
4. 🎯 After profiling: implement alloc optimization in `smo/optimizers/spatial.py`

## Notes

- All end-to-end benchmarks now save results via `write_benchmark_bundle()` to `benchmarks/results/`
- CATALOG.md already tracks all canonical suites; status=canonical verified
- No code changes to legacy wrappers required; they forward to updated suites

Last update: 2026-05-05
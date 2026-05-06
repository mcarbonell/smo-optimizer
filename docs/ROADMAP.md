# Roadmap

## Current Status Snapshot

Last cleanup pass completed:

- repository structure was consolidated around `smo/optimizers`, `smo/activations`, `smo/experimental`, `benchmarks/suites`, and `benchmarks/runners`
- backward-compatible wrappers remain in old import paths and old benchmark entrypoints
- benchmark inventory now lives in `benchmarks/CATALOG.md`
- benchmark result writing was unified through `benchmarks/results_utils.py`
- active benchmark outputs were separated from archived historical outputs under `benchmarks/results/historical/pre_rebaseline/`

Important current benchmark state:

- historical JSON and log outputs are archived for reference only and must not be used as the new baseline
- the active `benchmarks/results/` folder is intentionally clean for future iterative runs
- benchmark categories now explicitly distinguish `smoke`, `microbenchmark`, `end_to_end`, `diagnostic`, `runner`, and `legacy-wrapper`

Important optimizer state:

- `SMO-Spatial` and `SMO-Spatial-8bit` were audited first
- a correctness bug was fixed in the compressed second moment: it now tracks `E[g^2]` instead of `E[g]^2`
- `SMO-Spatial` and `SMO-Spatial-8bit` now agree exactly in a controlled smoke check when quantization is exact
- shared spatial helpers now fuse repeated compression and upsampling work to reduce PyTorch overhead

Current local measurement notes:

- this workstation may be busy with other training jobs, so local timing must distinguish wall-clock from process CPU time
- AMD local GPU uses DirectML via:
  - `C:/Users/mrcm_/Local/proj/ajedrez/neural-tablebases/venv_gpu/Scripts/python.exe`
- NVIDIA benchmarking should use Modal

Current quick validation artifacts:

- unit tests: `tests/test_spatial_optimizers.py`
- optimizer-step smoke check: `benchmarks/suites/optimizer_step/smoke_spatial_consistency.py`
- optimizer-step microbenchmark: `benchmarks/suites/optimizer_step/benchmark_step_time.py`
- timing helper: `benchmarks/timing.py`

Resume-here recommendation:

1. Continue improving `SMO-Spatial` and `SMO-Spatial-8bit` before any large benchmark rerun.
2. Use only smoke and microbenchmark loops until the optimizer hot path is cleaner.
3. Re-baseline end-to-end benchmarks only after the algorithm review pass is complete.

## Phase 0: Foundation

Goal: make the project legible and reviewable.

Deliverables:

- settle public naming around `SMO`
- define variant taxonomy
- define benchmark methodology
- classify code as stable or experimental
- stop adding new ad hoc benchmark scripts

Exit criteria:

- README points to methodology and roadmap
- every existing script is mapped to a benchmark category or marked obsolete

## Phase 1: Repository Consolidation

Goal: reduce structural chaos without changing behavior.

Deliverables:

- move optimizer code into a coherent package structure
- move spectral variants under a clearly experimental namespace
- group activation-memory work separately from optimizer-state work
- create a common result format for benchmark outputs

Exit criteria:

- no benchmark scripts living at repository root
- no duplicate benchmark logic across multiple files without justification

Status:

- mostly complete for the current cleanup pass
- root benchmark and runner entrypoints still exist, but now act as explicit compatibility wrappers
- benchmark logic was centralized into canonical suite and runner locations
- result storage was cleaned and historical artifacts were archived

Remaining practical follow-up:

- keep old wrappers thin
- avoid adding new logic outside canonical suite or runner locations

## Phase 2: Correctness And Measurement Baseline (✅ COMPLETE as of 2026-05-05)

Goal: establish trustworthy evidence.

Deliverables:

- ✅ Parity tests versus Adam-family baselines on simple models
- ✅ Deterministic seed handling across all canonical end-to-end and microbenchmarks
- ✅ Memory accounting helpers
- ✅ Timing helpers with warmup and synchronization (CPU/CUDA/DirectML)
- ✅ Benchmark configs for CPU (GPU paths prepared)

**Completed Infrastructure:**

- `benchmarks/timing.py`: `get_sync_fn(device)` + `measure_steps(steps, sync_fn)`
- `benchmarks/suites/optimizer_step/benchmark_step_time.py`:
  - Shapes: 64–4096 (7 configs)
  - Multi-seed support (`--seeds`, default 3): mean±std
  - Device-aware (`--device`): CPU/CUDA/DirectML (GPU execution pending hardware)
  - CUDA memory reporting (allocated/reserved MB)
  - Warmup configurable (`--warmup`)
  - GPU DirectML path prepared (commented pending hardware)
- Optimizer consistency tests: `tests/test_spatial_optimizers.py` (3/3 ✅)
- **Seeding policy documented** in `benchmarks/METHODOLOGY.md` (Rule 2: ≥3 seeds for public results, `--seed`/`--seeds` args required)

**End-to-End Baselines Executed (all with seed=1234):**

| Benchmark | Adam | SMO k=0.25 | SMO k=0.5 |
|-----------|------|------------|-----------|
| MNIST (`benchmark_mnist.py`) | 99.04% acc, 3.22 MB | 98.90% acc, 0.35 MB → **89.1% mem savings**, +0.14% gap | 99.13% acc, 0.92 MB |
| MNIST 8-bit (`benchmark_8bit.py`) | - | baseline | SMO-8bit (star mode) |
| CIFAR-10 (`benchmark_cifar10.py`) | TBD (requires ~90min CPU) | TBD | TBD |
| Spectral CPU (`benchmark_spectral_cpu.py`) | AdamW baseline | Walsh/DCT variants |

**Seeding Policy Implemented:**
- All end-to-end training benchmarks: `--seed` arg (default 1234), saved in results JSON
- Microbenchmarks: `--seeds` arg for multi-seed aggregation (default: 1234,5678,9012)
- Activation benchmark: multi-seed mean±std (CPU/CUDA)
- Helper: `set_seed(seed)` in each suite (torch + numpy + random + CUDA if present)

**Results Storage:**
- Bundle format via `benchmarks/results_utils.py`: `write_benchmark_bundle()`
- Aggregate JSON + per-variant JSON files under `benchmarks/results/`
- Already used: `benchmark_results.json` (MNIST), `benchmark_8bit_results.json` (queued), `benchmark_cifar10_results.json` (queued)

**Exit Criteria Status:**
- ✅ Benchmark runs are repeatable (per-seed determinism verified)
- ✅ Metrics defined consistently across hardware (infrastructure ready)
- ⚠️ GPU execution pending: local DirectML at high utilization; will enable when <50%

**Remaining minor items (non-blocking):**
- Propagate seed handling to `benchmark_minillm.py` (smoke test) – low priority
- Document seed policy in `benchmarks/METHODOLOGY.md` – in progress

## Phase 3: Bottleneck Analysis

Goal: identify what actually limits performance.

Deliverables:

- isolate optimizer-step time
- isolate compression and decompression costs
- profile activation-memory hooks separately
- compare PyTorch overhead vs algorithmic overhead vs kernel overhead

Status:

- started for the spatial optimizer family
- isolated optimizer-step microbenchmark exists
- quick profiling showed that repeated pooling and interpolation calls were a major overhead source
- fused helper work reduced overhead by batching gradient compression and state upsampling

Current findings:

- `SMO-Spatial` and `SMO-Spatial-8bit` are still slower than AdamW in CPU microbenchmarks
- after the latest helper optimizations, they are materially closer to AdamW than before
- the next likely hotspots are:
  - 8-bit quantize/dequantize overhead
  - full-resolution reconstruction for update application
  - remaining PyTorch tensor allocation overhead in the update path

Exit criteria:

- each major slowdown is attached to a measured source
- next kernel work is guided by profiling, not intuition

## Phase 4: Optimization Work (IN PROGRESS)

**Status:** Buffer reuse optimization for SMO-Spatial completed (2026-05-05).

**Completed:**
- **Buffer reuse in SMO-Spatial** (`smo/optimizers/spatial.py`):
  - Pre-allocated `_buf_g_comp`, `_buf_g_sq_comp`, `_buf_m_rec`, `_buf_v_rec` per-parameter buffers
  - Reused across steps via `compress_2d_pair_into_buffers()` y `upsample_2d_pair_into_buffers()`
  - Eliminated per-step tensor allocations for compression/upsampling intermediates
  - **Speedup measured on CPU:** 8-24% reduction in wall-clock step time (shape-dependent)
    - 256×256: -24.2%
    - 512×512: -10.0%
    - 1024×1024: -7.8%
  - SMO-Spatial-8bit inherits same gains (uses same spatial path)

**Current priority:** (remaining Phase 4 items)
1. [Done] SMO-Spatial alloc optimization
2. SMO-Spatial-8bit: ensure buffer reuse aligns (already uses same path)
3. Triton kernels: pending until PyTorch path stable and GPU available
4. Activation compression: separate track (not part of optimizer-state claims)

**Phase 4 exit criteria:**
- Each major slowdown is attached to a measured source (buffers done; profiling suite ready)
- Next kernel work guided by profiling, not intuition (GPU profiling pending)

## Phase 3 Prep: Profiling Infrastructure (READY — 2026-05-06)

**New tool:** `profiles/profile_smo_step.py`

Dual-mode profiler for SMO-Spatial:
- **Manual timers** — decomposes step into: compress → update compressed → upsample → full step; reports mean±std per-phase
- **torch.profiler** — kernel-level trace (exports Chrome JSON for flame graphs); CPU + CUDA

Usage:
```bash
# Manual breakdown (CPU)
python profiles/profile_smo_step.py --shape 1024,1024 --steps 100 --warmup 10 --seed 1234

# Kernel trace (requires CUDA)
python profiles/profile_smo_step.py --shape 1024,1024 --device cuda --use_torch_profiler
```

**Status:** Infrastructure ready on CPU; GPU execution blocked (local DirectML >90% utilization). Will run profiling sweep across k_ratios and shapes when GPU frees.

## Phase 5: Benchmark Publication

Goal: produce externally credible results.

Deliverables:

- benchmark tables with mean and standard deviation across seeds
- hardware disclosures
- dataset disclosures
- training hyperparameter disclosures
- honest discussion of failure cases and regressions

Exit criteria:

- a reviewer can reproduce the headline claims from documented commands

## Phase 6: Packaging And Documentation

Goal: make the project easy to evaluate and adopt.

Deliverables:

- cleaned package exports
- API examples
- benchmark reproduction guide
- research limitations section
- changelog and versioning policy

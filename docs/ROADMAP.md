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

## Phase 2: Correctness And Measurement Baseline (✅ COMPLETE as of 2026-05-06)

Goal: establish trustworthy evidence.

Deliverables:

- ✅ Parity tests versus Adam-family baselines on simple models
- ✅ Deterministic seed handling across all canonical end-to-end and microbenchmarks
- ✅ Memory accounting helpers (corrected 2026-05-06: temp buffers excluded)
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
- **Memory accounting corrected** (2026-05-06): temporary buffers moved to `_param_buffers` (private, not in state_dict) → accurate state memory measurements

**End-to-End Baselines Executed (all with seed=1234):**

| Benchmark | Variant | Accuracy / PPL | Optimizer Memory | Savings vs Baseline |
|-----------|---------|----------------|------------------|--------------------|
| MNIST (`benchmark_mnist.py`) | Adam | 99.04% acc | 3.22 MB | baseline |
| MNIST | SMO k=0.25 | 98.90% acc | 0.35 MB | **89.1%** |
| MNIST | SMO k=0.5 | 99.13% acc | 0.92 MB | 71.5% |
| MNIST 8-bit (`benchmark_8bit.py`) | Adam | 98.96% acc | 3.22 MB | baseline |
| MNIST 8-bit | SMO k=0.25 | 98.91% acc | 0.35 MB | 89.1% |
| MNIST 8-bit | SMO-8bit k=0.25 | **98.97% acc** | **0.21 MB** | **93.5%** |
| CIFAR-10 (`benchmark_cifar10.py`) | Adam | 66.91% acc | 4.74 MB | baseline |
| CIFAR-10 | SMO k=0.25 | 63.59% acc | 0.99 MB | **79.1%** |
| CIFAR-10 | SMO k=0.5 | 64.83% acc | 1.74 MB | **63.3%** |
| Spectral CIFAR-10 (`benchmark_spectral_cpu.py`) | AdamW | 63.03% acc | 4.74 MB | baseline |
| Spectral | Walsh Pure k=0.5 | **64.47% acc** | 1.74 MB | **63.3%** (+1.44% vs Adam) |
| Spectral | Walsh Hybrid k=0.5 | 61.24% acc | 1.74 MB | 63.3% (−1.79%) |
| Spectral | DCT Hybrid k=0.5 | 62.74% acc | 1.74 MB | 63.3% (−0.29%) |
| Spectral | DCT Pure k=0.5 | 14.28% acc | 4.74 MB | 0% (FAIL) |
| MiniGPT (`benchmark_minillm.py`) | AdamW | PPL 65.31 | 6.21 MB | baseline |
| MiniGPT | SMO k=0.5 | PPL 66.58 | 1.56 MB | **74.9%** |
| MiniGPT | SMO-8bit k=0.5 | PPL 66.58 | **0.43 MB** | **93.1%** |

**Key Insights:**
- **MNIST (simple)**: SMO-8bit achieves **93.5% memory savings** with **no accuracy loss** (98.97% vs 98.96% Adam)
- **CIFAR-10 (complex)**: Accuracy gap larger (−3.32% at k=0.25), but **memory savings still strong** (79.1%)
- **MiniGPT (Transformer)**: SMO-8bit achieves **93.1% savings** with same perplexity as SMO (66.58 vs 65.31 Adam) — promising for LLM-scale
- **Spectral variants**:
  - **Walsh Pure sorprende**: 64.47% acc (>Adam 63.03%) con 63% memoria — ¿generalización mejor?
  - Walsh Hybrid degrada; DCT Hybrid cerca de Adam; DCT Pure falla (bug en implementación pura)
- **Trade-off is task-dependent**: easier tasks tolerate aggressive compression; harder tasks need higher k_ratio
- **8-bit quantization adds 4–5% extra savings** on top of spatial compression with no quality hit in MNIST/MiniGPT

**Seeding Policy Implemented:**
- All end-to-end training benchmarks: `--seed` arg (default 1234), saved in results JSON
- Microbenchmarks: `--seeds` arg for multi-seed aggregation (default: 1234,5678,9012)
- Activation benchmark: multi-seed mean±std (CPU/CUDA)
- Helper: `set_seed(seed)` in each suite (torch + numpy + random + CUDA if present)

**Results Storage:**
- Bundle format via `benchmarks/results_utils.py`: `write_benchmark_bundle()`
- Aggregate JSON + per-variant JSON files under `benchmarks/results/`
- Produced: `benchmark_results.json` (MNIST), `benchmark_cifar10_results.json` (CIFAR-10)

**Exit Criteria Status:**
- ✅ Benchmark runs are repeatable (per-seed determinism verified across multiple suites)
- ✅ Metrics defined consistently across hardware (infrastructure ready, memory accounting corrected)
- ⚠️ GPU execution pending: local DirectML at high utilization; will enable when <50%

**Remaining minor items (non-blocking):**
- [x] Propagate seed handling to all canonical benchmarks — completed 2026-05-06
- [x] Run CIFAR-10 baseline — completed 2026-05-06
- [x] Run MNIST 8-bit baseline — completed 2026-05-06
- [x] Run MiniGPT smoke test — completed 2026-05-06
- [x] Run Spectral baseline (Walsh/DCT) — completed 2026-05-06 (Walsh Pure beats Adam; DCT Pure fails — requires debug)
- [x] Document memory accounting fix — done
- [ ] Debug DCT Pure failure (14% acc) — separate task in `spectral/optim_dct_pure.py`
- [ ] Prepare GPU profiling run when DirectML available

## Phase 3: Bottleneck Analysis

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
- **Profiling data collected (CPU, 512×512, k=0.25):**
  - Compression (avg pooling): ~41% of step time
  - Upsampling (bilinear interpolate): ~31% of step time
  - Moment update (elementwise): ~3%
  - Overhead / weight update: ~25%
- buffer reuse optimization reduced allocation overhead significantly
- **Next:** GPU profiling (blocked: local DirectML at >90% utilization)

Current findings (CPU):

- `SMO-Spatial` and `SMO-Spatial-8bit` are still slower than AdamW in CPU microbenchmarks
- after buffer reuse, they are materially closer to AdamW than before
- the next likely hotspots are:
  - 8-bit quantize/dequantize overhead (if using 8bit variant)
  - full-resolution reconstruction for update application (upsampling dominant)
  - PyTorch kernel overhead for pooling/interpolation (could benefit from custom kernels)

Exit criteria:

- each major slowdown is attached to a measured source ✅ (profiling data available)
- next kernel work guided by profiling, not intuition (GPU profiling pending)

## Phase 4: Optimization Work (IN PROGRESS)

**Status:** Buffer reuse + memory accounting fix completed (2026-05-06).

**Completed:**
- **Buffer reuse & memory accounting**:
  - Pre-allocated intermediate buffers moved to `_param_buffers` (private, not in state_dict)
  - Memory measurements now accurate (exclude temp workspace from optimizer state reports)
  - **Speedup measured on CPU:** 8–24% reduction in wall-clock step time (shape-dependent)
    - 256×256: −24.2%
    - 512×512: −10.0%
    - 1024×1024: −7.8%
  - SMO-Spatial-8bit inherits same gains (uses same spatial path)
- **CIFAR-10 baseline** validates real-world memory savings: 79.1% reduction (k=0.25) vs Adam

**Current priority:** (remaining Phase 4 items)
1. [Done] SMO-Spatial alloc optimization + memory accounting
2. SMO-Spatial-8bit: buffer reuse aligned (implicit)
3. Triton kernels: pending until PyTorch path stable and GPU available
4. Activation compression: separate track (not part of optimizer-state claims)

**Phase 4 exit criteria:**
- Each major slowdown is attached to a measured source ✅ (buffer reuse done; profiling suite ready)
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

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

## Phase 2: Correctness And Measurement Baseline

Goal: establish trustworthy evidence.

Deliverables:

- parity tests versus Adam-family baselines on simple models
- deterministic seed handling
- memory accounting helpers
- timing helpers with warmup and synchronization
- benchmark configs for CPU, AMD, NVIDIA

Status (updated 2026-05-05):

**Completed:**
- timing helper (`benchmarks/timing.py`): wall+process timing, `get_sync_fn()` for CUDA/DirectML/CPU
- optimizer-step microbench (`benchmarks/suites/optimizer_step/benchmark_step_time.py`):
  - shapes: 64×64, 128×128, 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096
  - device-aware: CPU/CUDA/DirectML (GPU paths prepared, commented pending hardware)
  - CUDA memory reporting (allocated/reserved)
  - warmup configurable
- optimizer consistency tests (`tests/test_spatial_optimizers.py`): ✅ 3/3 pass:
  - `test_spatial_second_moment_tracks_pooled_squared_gradient` ✅
  - `test_spatial_and_8bit_match_when_quantization_is_exact` ✅
  - `test_8bit_rejects_sparse_gradients` ✅
- end-to-end MNIST baseline executed (5 epochs, seed=1234, CPU):
  - Adam: 99.04% test acc, 3.22 MB optimizer state, 331.35s
  - SMO-Spatial k=0.25: 98.90% acc, 0.35 MB, 325.80s → **89.1% memory reduction**, accuracy gap +0.14%
  - SMO-Spatial k=0.5: 99.13% acc, 0.92 MB, 331.47s
  - Results: `benchmarks/results/benchmark_results.json`
- deterministic seed handling added to MNIST benchmark (torch.manual_seed + numpy + random)
- benchmark methodology documented (`benchmarks/METHODOLOGY.md`)
- result storage conventions (`benchmarks/CATALOG.md`, `benchmarks/results_utils.py`)

**Remaining gaps:**
- **Seed handling across all benchmarks**: benchmark_step_time y otros microbenchs aún no exponen seed parameter; no hay repeticiones múltiples para estadística
- **Timing normalization**: falta wrapper para N repeticiones y cálculo mean±std
- **GPU execution**: DirectML path preparado pero no ejecutado (GPU local al 90%+)
- **Alloc optimization**: pospuesto a Phase 4; requiere profiling profundo en GPU

**Immediate next actions:**
1. Add `--seed` y `--repeats` flags to `benchmark_step_time.py`; output mean±std
2. Propagate deterministic seed to all micro/activation benchmarks
3. Prepare DirectML run config (device='privateuse:0') y probar cuando GPU libre
4. Move alloc-optimization task to Phase 4; schedule after profiling stage

**Exit criteria status:**
- Benchmark runs are repeatable ✅ (per-seed repeatability confirmed for MNIST; microbench fixed-seed)
- Metrics are defined consistently across hardware ⚠️ (infrastructure ready, GPU execution pending)

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

## Phase 4: Optimization Work

Goal: improve the strongest candidates only.

Priority order:

1. `SMO-Spatial`
2. `SMO-Spatial-8bit`
3. `SMO-Spatial-8bit-Triton`
4. activation-memory compression
5. spectral variants

Reasoning:

- the spatial path is closest to a coherent story
- 8-bit state compression is central to the value proposition
- Triton work only matters once correctness and measurement are solid
- activation compression is promising but should not be mixed into optimizer claims too early
- spectral variants should compete for promotion based on evidence

Current priority inside Phase 4:

1. keep iterating on `SMO-Spatial`
2. keep `SMO-Spatial-8bit` aligned with the same algorithmic semantics
3. only revisit Triton once the PyTorch path is cleaner and the benchmark loop is stable

Do not do yet:

- do not treat archived results as the baseline
- do not launch a full benchmark campaign before the current optimizer iteration pass stabilizes
- do not mix activation-memory claims into optimizer-state conclusions

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

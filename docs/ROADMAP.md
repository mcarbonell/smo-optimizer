# Roadmap

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

## Phase 2: Correctness And Measurement Baseline

Goal: establish trustworthy evidence.

Deliverables:

- parity tests versus Adam-family baselines on simple models
- deterministic seed handling
- memory accounting helpers
- timing helpers with warmup and synchronization
- benchmark configs for CPU, AMD, NVIDIA

Exit criteria:

- benchmark runs are repeatable
- metrics are defined consistently across hardware

## Phase 3: Bottleneck Analysis

Goal: identify what actually limits performance.

Deliverables:

- isolate optimizer-step time
- isolate compression and decompression costs
- profile activation-memory hooks separately
- compare PyTorch overhead vs algorithmic overhead vs kernel overhead

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

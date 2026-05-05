# SMO

SMO is a research repository for memory-efficient optimizer experiments in PyTorch.

The core idea is to reduce optimizer-state memory by compressing first- and second-order states, with current work spanning:

- Spatial optimizer-state compression
- Spectral optimizer-state compression
- 8-bit compressed optimizer states
- Early activation-memory compression experiments
- Experimental Triton kernels for NVIDIA GPUs

This repository is currently in a consolidation phase. The immediate goal is to make the project professionally reviewable before deeper optimization work:

- stabilize naming and positioning
- unify the variant taxonomy
- standardize benchmark methodology
- separate production-ready code from experiments

## Current Recommendation On Naming

`Super Mario Optimizer` is memorable, but it is not a good long-term public name:

- it is very close to Nintendo's `Super Mario` trademark
- it can make the project feel more playful than rigorous
- it creates unnecessary friction for papers, benchmarks, packaging, and external adoption

Recommendation:

- keep `SMO` as the stable short name
- treat `Super Mario Optimizer` as a temporary internal codename or historical note
- migrate the public expansion of `SMO` toward a neutral technical name such as:
  - `State Memory Optimizer`
  - `State-compressed Memory Optimizer`
  - `Spectral Memory Optimizer`

## Repository Status

Today the repo is organized around four main areas:

- `smo/optimizers/`: main optimizer implementations
- `smo/activations/`: activation-memory experiments
- `smo/experimental/`: spectral and other experimental variants
- `benchmarks/`: benchmark suites, hardware runners, methodology, and stored results

Backward-compatible wrappers are still present at the old import paths to ease the transition.

## Project Documents

- [Project Foundation](/C:/Users/mrcm_/Local/proj/algorithms/supermario_optimizer/docs/PROJECT_FOUNDATION.md)
- [Roadmap](/C:/Users/mrcm_/Local/proj/algorithms/supermario_optimizer/docs/ROADMAP.md)
- [Benchmark Methodology](/C:/Users/mrcm_/Local/proj/algorithms/supermario_optimizer/benchmarks/METHODOLOGY.md)
- [Benchmark Catalog](/C:/Users/mrcm_/Local/proj/algorithms/supermario_optimizer/benchmarks/CATALOG.md)

## Proposed Variant Taxonomy

The current repo names can be normalized like this:

- `SMO-Spatial`: current `smo/optimizers/spatial.py`
- `SMO-Spatial-8bit`: current `smo/optimizers/spatial_8bit.py`
- `SMO-Spatial-Triton`: current `smo/optimizers/spatial_triton.py`
- `SMO-Spatial-8bit-Triton`: current `smo/optimizers/spatial_8bit_triton.py`
- `SMO-Walsh`: current `smo/experimental/walsh.py`
- `SMO-DCT`: current `smo/experimental/dct.py`
- `SMO-Activation-FP16` / `SMO-Activation-8bit` / `SMO-Activation-Delta`: current `smo/activations/`

This preserves continuity while making it much easier to compare variants cleanly.

## Near-Term Priorities

1. Freeze names, scope, and experiment categories.
2. Unify benchmark inputs, metrics, and result formats.
3. Separate stable implementations from research code.
4. Validate claims on CPU, AMD, and NVIDIA with repeatable methodology.
5. Only then refactor kernels and optimize bottlenecks.

## Installation

```bash
pip install -e .
```

## Current Caveat

The repository still contains exploratory code and benchmark scripts with overlapping purposes. The new documentation added in this phase is intended to serve as the contract for the cleanup and consolidation work that follows.

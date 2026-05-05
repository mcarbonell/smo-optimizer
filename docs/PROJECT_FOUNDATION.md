# Project Foundation

## 1. Naming Decision

### Recommendation

Do not keep `Super Mario Optimizer` as the public primary name.

Reasons:

- `Super Mario` is a globally recognized Nintendo trademark.
- Even with a disclaimer, the name introduces avoidable legal and branding risk.
- For research adoption, serious benchmarking, and external packaging, the joke costs more than it helps.

### Practical path

Use this transition:

- public short name: `SMO`
- public long name: choose a neutral technical expansion later
- internal note: "`SMO` was originally nicknamed `Super Mario Optimizer`"

Good candidate expansions:

- `State Memory Optimizer`
- `State-compressed Memory Optimizer`
- `Spectral Memory Optimizer`

Best default for now:

`State Memory Optimizer`

It is broad enough to cover spatial, spectral, 8-bit, and activation-memory work without overcommitting to one mechanism.

## 2. What This Repository Actually Is

Right now the repo mixes three different things:

1. A candidate optimizer package
2. A research lab for experimental variants
3. A benchmark sandbox

Those should be treated as separate layers.

### Recommended layers

- `library`: code we are willing to expose and support
- `research`: experimental variants and ideas under evaluation
- `evaluation`: reproducible benchmark and measurement machinery

## 3. Unified Variant Taxonomy

### Stable family names

- `SMO-Spatial`
- `SMO-Spatial-8bit`
- `SMO-Spatial-Triton`
- `SMO-Spatial-8bit-Triton`
- `SMO-Walsh`
- `SMO-DCT`
- `SMO-Activation-FP16`
- `SMO-Activation-8bit`
- `SMO-Activation-Delta`

### Why this helps

- It separates compression domain from implementation backend.
- It avoids names like `SMOWalsh` and `SMODCT` leaking ad hoc naming decisions.
- It allows tables, plots, and benchmark artifacts to use a consistent scheme.

## 4. Target Repository Organization

Do not execute this full move yet; use it as the target structure for the next refactor phase.

```text
supermario_optimizer/
  smo/
    optimizers/
      spatial.py
      spatial_8bit.py
      spatial_triton.py
      spatial_8bit_triton.py
    activations/
      fp16_hooks.py
      quant8.py
      delta.py
    experimental/
      walsh.py
      dct.py
    utils/
    __init__.py
  benchmarks/
    suites/
      optimizer_state/
      activation_memory/
      end_to_end_training/
      kernels/
    configs/
    results/
    methodology/
  docs/
    PROJECT_FOUNDATION.md
    ROADMAP.md
  examples/
  tests/
```

## 5. Classification Rules

Every implementation should be labeled explicitly with one status:

- `stable`: expected to behave correctly and be benchmarked regularly
- `experimental`: promising but not yet benchmark-grade
- `archived`: useful history, not part of current claims

Recommended status today:

- `SMO-Spatial`: stable candidate
- `SMO-Spatial-8bit`: stable candidate
- `SMO-Spatial-Triton`: experimental
- `SMO-Spatial-8bit-Triton`: experimental
- `SMO-Walsh`: experimental
- `SMO-DCT`: experimental
- activation compression modules: experimental

## 6. Claim Discipline

From now on, claims should be separated into four levels:

- `theoretical`: memory formula or asymptotic argument
- `microbenchmark`: isolated kernel or optimizer-step timing
- `task benchmark`: end-to-end model training on a named task
- `production claim`: only after repeated results across hardware and seeds

This matters because the repo currently mixes these categories in the README, which makes the evidence look stronger than it is.

## 7. Definition Of Success For The Cleanup Phase

This organizational phase is successful when:

- every variant has one canonical name
- every benchmark has a declared purpose
- every result is reproducible from a script plus config
- public claims map to stored evidence
- stable code is clearly separated from research code

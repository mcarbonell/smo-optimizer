# Benchmark Methodology

## Purpose

This document defines how SMO variants should be evaluated so that results are technically credible and easy to compare.

The benchmark system must answer three different questions, and they must not be mixed:

1. Does it reduce memory?
2. Does it preserve training quality?
3. Does it improve or harm wall-clock performance?

## Benchmark Families

### A. State-memory benchmarks

Question:

How much optimizer-state memory is saved relative to AdamW?

Measure:

- theoretical optimizer-state bytes
- realized optimizer-state bytes after state initialization
- parameter count affected by compression

Do not mix with:

- activation memory
- dataloader memory
- CUDA allocator fragmentation

### B. Activation-memory benchmarks

Question:

How much peak memory is saved by activation compression, and what accuracy cost does it introduce?

Measure:

- peak allocated memory
- peak reserved memory
- training throughput
- final metric delta versus no activation compression

### C. Optimizer-step microbenchmarks

Question:

What is the isolated cost of one optimizer step?

Measure:

- forward/backward excluded when possible
- warmup steps
- synchronized timing on GPU
- tensor-shape sweep across representative matrices

### D. End-to-end training benchmarks

Question:

What happens in a real training loop?

Measure:

- final validation metric
- time per epoch or tokens per second
- optimizer-state memory
- peak device memory

## Required Benchmark Metadata

Every result file must record:

- date
- git commit if available
- script name
- benchmark family
- optimizer variant
- hardware
- backend
- framework versions
- dataset
- model
- seed
- batch size
- precision
- number of steps or epochs

## Hardware Matrix

### CPU

Purpose:

- correctness
- baseline overhead
- reproducibility

Preferred measurements:

- wall-clock step time
- total train time
- resident memory when feasible

### AMD GPU

Purpose:

- portability check
- non-CUDA behavior
- whether the non-Triton path remains viable

Backend note:

Current local path appears to be `torch-directml`, which is acceptable for exploratory benchmarking but must be labeled clearly as `DirectML`, not generic `AMD GPU`.

That distinction matters because DirectML results are not the same as ROCm results.

### NVIDIA GPU

Purpose:

- primary performance evaluation
- Triton validation
- peak memory analysis

Preferred environment:

- Modal GPU runners for reproducible CUDA tests

Recommended device tiers:

- one economical device such as `T4` or `L4` for smoke tests
- one stronger device such as `A10G` or better for publication-quality performance measurements

## Benchmark Rules

### Rule 1

Never compare two optimizers with different training recipes unless the benchmark explicitly says it is an ablation.

### Rule 2

Run at least 3 seeds for any result that will be shown publicly.

For stronger claims, use 5 seeds.

### Rule 3

Separate smoke tests from publication benchmarks.

Smoke tests prove the script runs.
Publication benchmarks support claims.

### Rule 4

GPU timings must use synchronization.

Without synchronization, timing numbers are not trustworthy.

For local CPU iteration on a busy machine, record both:

- wall-clock time
- process CPU time

Wall-clock shows user-visible latency.
Process CPU time is more robust to unrelated system load and is often the better signal for optimizer micro-iteration on a shared workstation.

### Rule 5

Report both accuracy-quality metrics and memory metrics together.

A memory win without quality context is incomplete.

### Rule 6

Do not present synthetic-data language-model runs as evidence of task quality.

They are useful for plumbing and scaling checks, not for headline training claims.

## Recommended Benchmark Ladder

### Tier 0: Smoke

- single seed
- very short runs
- confirms no crashes

### Tier 1: Correctness

- small models
- fixed seeds
- compare against AdamW
- verify losses and training curves are sane

Suggested tasks:

- MNIST CNN
- CIFAR-10 small CNN

### Tier 2: Architecture relevance

- transformer-like model
- real dataset if possible
- compare perplexity or validation loss

Suggested tasks:

- small GPT-style model on Tiny Shakespeare or equivalent real text corpus

### Tier 3: Performance

- isolated optimizer step benchmark
- large representative dense layers
- CPU, AMD/DirectML, NVIDIA/CUDA split clearly labeled

### Tier 4: Publication

- repeated seeds
- complete metadata
- stable scripts only
- stored JSON outputs
- one summary table generated from raw outputs

## Recommended Result Format

Use one JSON file per run and one aggregated JSON or CSV per suite.

Each per-run file should contain:

```json
{
  "benchmark_family": "end_to_end_training",
  "variant": "SMO-Spatial-8bit",
  "hardware": "NVIDIA A10G",
  "backend": "CUDA",
  "dataset": "CIFAR-10",
  "model": "CIFAR_CNN",
  "seed": 42,
  "epochs": 5,
  "metrics": {
    "final_accuracy": 0.6535,
    "optimizer_state_mb": 0.80,
    "peak_device_mb": 1234.5,
    "total_time_s": 26.7
  }
}
```

## Immediate Cleanup Recommendations For Existing Benchmarks

- move root benchmark scripts into `benchmarks/` subfolders
- label each script as `smoke`, `microbenchmark`, or `end_to_end`
- retire scripts that duplicate the same question with slightly different code
- stop writing benchmark narratives directly in the README before raw outputs are normalized

## What Counts As Professional Evidence

A result is professional enough to cite publicly when:

- the script is versioned
- the environment is declared
- the metric definitions are explicit
- the run is reproducible
- the claim is scoped correctly

That is the standard this repo should now aim for.

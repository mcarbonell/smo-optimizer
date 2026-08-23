# Benchmark Catalog

This file is the canonical inventory for benchmark entrypoints in the repository.

## Categories

- `smoke`: confirms a code path runs and produces plausible outputs
- `microbenchmark`: isolates a narrow cost such as memory overhead or optimizer-step timing
- `end_to_end`: runs a real training loop and reports task-level outcomes
- `diagnostic`: debugging or ablation script, useful internally but not for headline claims
- `runner`: platform-specific launcher for remote or hardware-specific execution
- `legacy-wrapper`: compatibility shim that forwards to a canonical entrypoint

## Canonical Suites

| Path | Family | Category | Status | Notes |
| --- | --- | --- | --- | --- |
| `benchmarks/suites/training/benchmark_mnist.py` | `end_to_end_training` | `end_to_end` | canonical | CPU correctness baseline on MNIST |
| `benchmarks/suites/training/benchmark_cifar10.py` | `end_to_end_training` | `end_to_end` | canonical | CPU end-to-end comparison on CIFAR-10 |
| `benchmarks/suites/training/benchmark_8bit.py` | `end_to_end_training` | `end_to_end` | canonical | MNIST comparison for `SMO-Spatial-8bit` |
| `benchmarks/suites/training/benchmark_minillm.py` | `end_to_end_training` | `smoke` | canonical | Synthetic-token transformer plumbing check, not a quality benchmark |
| `benchmarks/suites/activations/benchmark_activations.py` | `activation_memory` | `microbenchmark` | canonical | Measures activation-memory and timing overhead |
| `benchmarks/suites/activations/test_accuracy_activations.py` | `activation_memory` | `end_to_end` | canonical | MNIST quality check for activation quantization |
| `benchmarks/suites/activations/test_accuracy_delta.py` | `activation_memory` | `end_to_end` | canonical | MNIST quality check for delta activation compression |
| `benchmarks/suites/activations/debug_accuracy_drop.py` | `activation_memory` | `diagnostic` | canonical | Internal ablation for activation-related regressions |
| `benchmarks/suites/spectral/benchmark_spectral_cpu.py` | `end_to_end_training` | `end_to_end` | canonical | CPU comparison of spectral variants |
| `benchmarks/suites/spectral/benchmark_spectral_gpu_directml.py` | `end_to_end_training` | `end_to_end` | canonical | DirectML exploratory comparison of spectral variants |
| `benchmarks/suites/spectral/benchmark_dct_fix.py` | `end_to_end_training` | `diagnostic` | canonical | Targeted validation of the DCT smoothing fix |
| `benchmarks/suites/comparison/exhaustive_benchmark.py` | `end_to_end_training` | `end_to_end` | canonical | Multi-seed Modal benchmark, still exploratory |
| `benchmarks/suites/comparison/exhaustive_benchmark_local.py` | `end_to_end_training` | `end_to_end` | canonical | Multi-seed local DirectML benchmark, still exploratory |
| `benchmarks/suites/comparison/t4_memory_benchmark.py` | `gpu_memory_training` | `end_to_end` | canonical | Single-GPU (T4/16 GB) peak-memory + quality comparison vs AdamW and bitsandbytes 8-bit; OOM-tolerant per optimizer |
| `benchmarks/suites/comparison/t4_summarize.py` | `gpu_memory_training` | `diagnostic` | canonical | Aggregates t4_* bundles into mean±std across seeds/tags |
| `benchmarks/suites/comparison/t4_loss_matched.py` | `gpu_memory_training` | `diagnostic` | canonical | Loss-matched metric comparison from trajectory histories (generalization at equal optimization progress) |
| `benchmarks/suites/optimizer_step/smoke_spatial_consistency.py` | `optimizer_step` | `smoke` | canonical | Fast consistency check for `SMO-Spatial` vs `SMO-Spatial-8bit` |
| `benchmarks/suites/optimizer_step/benchmark_step_time.py` | `optimizer_step` | `microbenchmark` | canonical | Isolated optimizer-step timing for rapid iteration |

## Hardware Runners

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `benchmarks/runners/modal/test_modal_connection.py` | `runner` | canonical | Smoke test for Modal GPU availability |
| `benchmarks/runners/modal/test_8bit_triton_modal.py` | `runner` | canonical | Smoke test for Triton path on Modal |
| `benchmarks/runners/modal/modal_benchmark_triton.py` | `runner` | canonical | Modal launcher for Triton benchmarking |
| `benchmarks/runners/modal/benchmark_activations_modal.py` | `runner` | canonical | Modal launcher for activation-memory experiments |
| `benchmarks/runners/modal/benchmark_spectral_gpu_modal.py` | `runner` | canonical | Modal runner for spectral CIFAR-10 benchmarks |
| `benchmarks/runners/modal/benchmark_8bit_triton_cifar10_modal.py` | `runner` | canonical | Modal runner for CIFAR-10 8-bit Triton comparison |
| `benchmarks/runners/directml/test_gpu_directml.py` | `runner` | canonical | Local DirectML environment smoke test |
| `benchmarks/runners/colab/t4_benchmark_colab.ipynb` | `runner` | canonical | Colab/Kaggle T4 notebook: setup, quality runs (gpt/vit), killer-demo probe, results table |

## Legacy Compatibility Entry Points

These files should stay thin and contain no benchmark logic.

| Path | Category | Canonical Target |
| --- | --- | --- |
| `benchmarks/benchmark_mnist.py` | `legacy-wrapper` | `benchmarks.suites.training.benchmark_mnist` |
| `benchmarks/benchmark_cifar10.py` | `legacy-wrapper` | `benchmarks.suites.training.benchmark_cifar10` |
| `benchmarks/benchmark_8bit.py` | `legacy-wrapper` | `benchmarks.suites.training.benchmark_8bit` |
| `benchmarks/benchmark_minillm.py` | `legacy-wrapper` | `benchmarks.suites.training.benchmark_minillm` |
| `benchmarks/benchmark_activations.py` | `legacy-wrapper` | `benchmarks.suites.activations.benchmark_activations` |
| `benchmarks/test_accuracy_activations.py` | `legacy-wrapper` | `benchmarks.suites.activations.test_accuracy_activations` |
| `benchmarks/test_accuracy_delta.py` | `legacy-wrapper` | `benchmarks.suites.activations.test_accuracy_delta` |
| `benchmarks/debug_accuracy_drop.py` | `legacy-wrapper` | `benchmarks.suites.activations.debug_accuracy_drop` |
| `benchmarks/benchmark_spectral_cpu.py` | `legacy-wrapper` | `benchmarks.suites.spectral.benchmark_spectral_cpu` |
| `benchmarks/benchmark_spectral_gpu_directml.py` | `legacy-wrapper` | `benchmarks.suites.spectral.benchmark_spectral_gpu_directml` |
| `benchmarks/benchmark_dct_fix.py` | `legacy-wrapper` | `benchmarks.suites.spectral.benchmark_dct_fix` |
| `benchmarks/exhaustive_benchmark.py` | `legacy-wrapper` | `benchmarks.suites.comparison.exhaustive_benchmark` |
| `benchmarks/exhaustive_benchmark_local.py` | `legacy-wrapper` | `benchmarks.suites.comparison.exhaustive_benchmark_local` |
| `benchmark_activations_modal.py` | `legacy-wrapper` | `benchmarks.runners.modal.benchmark_activations_modal` |
| `benchmark_spectral_gpu_modal.py` | `legacy-wrapper` | `benchmarks.runners.modal.benchmark_spectral_gpu_modal` |
| `benchmark_8bit_triton_cifar10_modal.py` | `legacy-wrapper` | `benchmarks.runners.modal.benchmark_8bit_triton_cifar10_modal` |
| `modal_benchmark_triton.py` | `legacy-wrapper` | `benchmarks.runners.modal.modal_benchmark_triton` |
| `test_modal_connection.py` | `legacy-wrapper` | `benchmarks.runners.modal.test_modal_connection` |
| `test_8bit_triton_modal.py` | `legacy-wrapper` | `benchmarks.runners.modal.test_8bit_triton_modal` |
| `test_gpu_directml.py` | `legacy-wrapper` | `benchmarks.runners.directml.test_gpu_directml` |

## Working Rules

- New benchmark logic goes only under `benchmarks/suites/` or `benchmarks/runners/`.
- If a legacy wrapper is still needed, keep it as a pure redirector.
- If a script is only for debugging or ablation, label it `diagnostic` and avoid citing it in public claims.
- Active outputs belong in `benchmarks/results/`; archived pre-cleanup outputs live under `benchmarks/results/historical/pre_rebaseline/`.

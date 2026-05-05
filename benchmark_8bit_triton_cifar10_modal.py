"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.runners.modal.benchmark_8bit_triton_cifar10_modal."""

import runpy


if __name__ == "__main__":
    runpy.run_module("benchmarks.runners.modal.benchmark_8bit_triton_cifar10_modal", run_name="__main__")

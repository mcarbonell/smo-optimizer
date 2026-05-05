"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.runners.modal.modal_benchmark_triton."""

import runpy


if __name__ == "__main__":
    runpy.run_module("benchmarks.runners.modal.modal_benchmark_triton", run_name="__main__")

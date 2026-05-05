"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.runners.modal.benchmark_activations_modal."""

import runpy


if __name__ == "__main__":
    runpy.run_module("benchmarks.runners.modal.benchmark_activations_modal", run_name="__main__")

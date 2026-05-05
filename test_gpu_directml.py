"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.runners.directml.test_gpu_directml."""

import runpy


if __name__ == "__main__":
    runpy.run_module("benchmarks.runners.directml.test_gpu_directml", run_name="__main__")

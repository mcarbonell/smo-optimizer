"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.suites.spectral.benchmark_spectral_gpu_directml."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    runpy.run_module("benchmarks.suites.spectral.benchmark_spectral_gpu_directml", run_name="__main__")

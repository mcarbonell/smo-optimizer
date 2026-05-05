"""Legacy compatibility wrapper. Canonical entrypoint: benchmarks.suites.comparison.exhaustive_benchmark_local."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    runpy.run_module("benchmarks.suites.comparison.exhaustive_benchmark_local", run_name="__main__")

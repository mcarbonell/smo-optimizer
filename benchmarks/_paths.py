"""Shared path helpers for benchmark scripts."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


def add_project_root_to_path() -> Path:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    return PROJECT_ROOT


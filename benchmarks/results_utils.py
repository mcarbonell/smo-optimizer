"""Helpers for writing benchmark results in a consistent format."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks._paths import RESULTS_DIR


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def framework_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import torch  # type: ignore

        versions["torch"] = torch.__version__
        if torch.cuda.is_available():
            versions["cuda"] = torch.version.cuda or "unknown"
    except ImportError:
        pass
    return versions


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "run"


def make_run_record(
    *,
    benchmark_family: str,
    variant: str,
    script_name: str,
    hardware: str,
    backend: str,
    dataset: str,
    model: str,
    metrics: dict[str, Any],
    seed: int | None = None,
    batch_size: int | None = None,
    precision: str | None = None,
    epochs: int | None = None,
    steps: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "date": utc_timestamp(),
        "git_commit": git_commit(),
        "script_name": script_name,
        "benchmark_family": benchmark_family,
        "variant": variant,
        "hardware": hardware,
        "backend": backend,
        "framework_versions": framework_versions(),
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "batch_size": batch_size,
        "precision": precision,
        "epochs": epochs,
        "steps": steps,
        "metrics": metrics,
    }
    if extra:
        record.update(extra)
    return record


def write_benchmark_bundle(
    *,
    aggregate_filename: str,
    suite_name: str,
    benchmark_family: str,
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
) -> tuple[Path, list[Path]]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_path = RESULTS_DIR / aggregate_filename
    bundle = {
        "suite_name": suite_name,
        "benchmark_family": benchmark_family,
        "generated_at": utc_timestamp(),
        "git_commit": git_commit(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "summary": summary,
        "runs": runs,
    }
    aggregate_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    run_dir = RESULTS_DIR / aggregate_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    run_paths: list[Path] = []
    for run in runs:
        run_path = run_dir / f"{_slugify(run['variant'])}.json"
        run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        run_paths.append(run_path)

    return aggregate_path, run_paths

#!/usr/bin/env python3
"""
Aggregate T4 memory-benchmark bundles into a mean±std summary.

Reads benchmarks/results/t4_*_memory_results*.json (one bundle per run,
optionally suffixed with --tag), groups by (suite, variant), and reports
mean±std of the task metric plus persistent state / peak memory / throughput.

Usage:
    python -m benchmarks.suites.comparison.t4_summarize
    python -m benchmarks.suites.comparison.t4_summarize --suite vit
    python -m benchmarks.suites.comparison.t4_summarize --out summary.md
"""

import argparse
import json
import statistics
from collections import defaultdict

from benchmarks._paths import RESULTS_DIR


def fmt(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Summarize t4_memory_benchmark bundles")
    parser.add_argument("--suite", type=str, default=None, help="Filter by suite name (gpt/vit)")
    parser.add_argument("--out", type=str, default=None, help="Also write a markdown report")
    args = parser.parse_args()

    files = sorted(RESULTS_DIR.glob("t4_*_memory_results*.json"))
    if args.suite:
        files = [f for f in files if args.suite in f.name]
    if not files:
        print(f"No bundles matching t4_*_memory_results*.json in {RESULTS_DIR}")
        return

    groups: dict[tuple, list] = defaultdict(list)
    for path in files:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        suite = bundle["summary"]["suite"]
        for run in bundle["runs"]:
            metrics = run["metrics"]
            if metrics.get("status") != "ok":
                continue
            groups[(suite, run["variant"])].append(
                {
                    "seed": run.get("seed"),
                    "metric": metrics.get(run.get("metric_key", "")),
                    "state_mb": metrics.get("persistent_state_mb"),
                    "peak_mb": metrics.get("_peak_alloc_mb"),
                    "throughput": metrics.get("tokens_per_s") or metrics.get("images_per_s"),
                }
            )

    lines = []
    header = (
        f"{'suite':<6} {'optimizer':<18} {'n':>2} {'seeds':<16} "
        f"{'metric (mean±std)':>20} {'state_MB':>12} {'peak_MB':>12} {'thr/s':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for (suite, variant), runs in sorted(groups.items()):
        def agg(key, digits=2):
            vals = [r[key] for r in runs if isinstance(r[key], (int, float))]
            if not vals:
                return "—"
            if len(vals) == 1:
                return f"{vals[0]:.{digits}f}"
            return fmt(statistics.mean(vals), statistics.stdev(vals), digits)

        seeds = ",".join(str(r["seed"]) for r in runs)
        lines.append(
            f"{suite:<6} {variant:<18} {len(runs):>2} {seeds:<16} "
            f"{agg('metric'):>20} {agg('state_mb'):>12} {agg('peak_mb'):>12} "
            f"{agg('throughput', 0):>12}"
        )
    report = "\n".join(lines)
    print(report)
    print(f"\n({len(files)} bundle(s): {', '.join(f.name for f in files)})")

    if args.out:
        out_path = RESULTS_DIR / args.out
        out_path.write_text(f"# T4 benchmark summary\n\n```\n{report}\n```\n", encoding="utf-8")
        print(f"Markdown written to {out_path}")


if __name__ == "__main__":
    main()

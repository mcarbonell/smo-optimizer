#!/usr/bin/env python3
"""
Loss-matched generalization comparison from T4 benchmark bundles.

Reads the per-epoch/per-step training histories stored by t4_memory_benchmark
(metrics.history) and answers: *at equal train loss, which optimizer achieves
the better test metric?* Same seed => identical batch order and LR schedule,
so matching on train loss compares generalization at equal optimization
progress.

For every variant it interpolates its metric on a common train-loss grid
(the overlap region of all trajectories), aggregating mean±std across runs.

Usage:
    python -m benchmarks.suites.comparison.t4_loss_matched --suite vit
    python -m benchmarks.suites.comparison.t4_loss_matched --suite gpt --baseline AdamW
"""

import argparse
import json
import statistics
from collections import defaultdict

from benchmarks._paths import RESULTS_DIR


def interp(xs, ys, x):
    """Linear interpolation with edge clamping; xs must be sorted."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


def load_curves(suite):
    """variant -> list of runs; each run is a sorted [(train_loss, metric)] list."""
    curves = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob(f"t4_{suite}_memory_results*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for run in bundle["runs"]:
            pairs = []
            for h in run["metrics"].get("history") or []:
                loss = h.get("train_loss")
                metric = h.get("test_acc", h.get("val_loss"))
                if isinstance(loss, (int, float)) and isinstance(metric, (int, float)):
                    pairs.append((loss, metric))
            if len(pairs) >= 2:
                pairs.sort(key=lambda p: p[0])
                curves[run["variant"]].append(pairs)
    return curves


def resolve_baseline(curves, requested):
    if requested in curves:
        return requested
    matches = [v for v in curves if v.startswith(requested)]
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Loss-matched comparison from t4 bundles")
    parser.add_argument("--suite", type=str, default="vit", choices=["vit", "gpt"])
    parser.add_argument("--baseline", type=str, default="AdamW",
                        help="Variant-name prefix used as the delta reference")
    parser.add_argument("--points", type=int, default=6)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    curves = load_curves(args.suite)
    if not curves:
        print(f"No histories found under {RESULTS_DIR} for suite '{args.suite}'. "
              "Bundles written before trajectory logging have no history field.")
        return

    baseline = resolve_baseline(curves, args.baseline)
    if baseline is None:
        print(f"Baseline '{args.baseline}' not found. Variants: {sorted(curves)}")
        return

    # Common grid from per-VARIANT pooled loss ranges (per-run interpolation
    # clamps at its own edges, so individual runs may cover less)
    def pooled_range(runs):
        xs = [x for pairs in runs for x, _ in pairs]
        return min(xs), max(xs)

    lo = max(pooled_range(runs)[0] for runs in curves.values())
    hi = min(pooled_range(runs)[1] for runs in curves.values())
    if hi <= lo:
        print("No overlapping train-loss range between variants.")
        return

    grid = [lo + (hi - lo) * i / (args.points - 1) for i in range(args.points)]
    variants = sorted(curves)
    higher_better = args.suite == "vit"  # vit tracks test acc, gpt tracks val loss

    table = {}
    for variant in variants:
        table[variant] = [
            [interp([x for x, _ in pairs], [y for _, y in pairs], g) for pairs in curves[variant]]
            for g in grid
        ]

    def fmt_cell(vals):
        mean = statistics.mean(vals)
        cell = f"{mean:.2f}"
        if len(vals) > 1:
            cell += f"±{statistics.stdev(vals):.2f}"
        return cell

    lines = [
        f"Loss-matched {'test_acc' if higher_better else 'val_loss'} "
        f"| suite={args.suite} | baseline={baseline} | {len(curves)} variant(s)",
        "",
    ]
    header = f"{'train_loss':>10} " + " ".join(f"{v[:24]:>26}" for v in variants)
    lines.append(header)
    lines.append("-" * len(header))
    for i, g in enumerate(grid):
        base_mean = statistics.mean(table[baseline][i])
        cells = []
        for variant in variants:
            cell = fmt_cell(table[variant][i])
            if variant != baseline and len(table[variant][i]) == len(table[baseline][i]):
                delta = statistics.mean(table[variant][i]) - base_mean
                sign = "+" if delta >= 0 else ""
                cell += f" ({sign}{delta:.2f})"
            cells.append(f"{cell:>26}")
        lines.append(f"{g:>10.4f} " + " ".join(cells))

    report = "\n".join(lines)
    print(report)

    if args.out:
        out_path = RESULTS_DIR / args.out
        out_path.write_text(f"# Loss-matched analysis ({args.suite})\n\n```\n{report}\n```\n", encoding="utf-8")
        print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

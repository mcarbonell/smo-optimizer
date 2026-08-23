#!/usr/bin/env python3
"""
Hyperparameter-fairness LR sweep for T4 memory benchmarks.

Launches the canonical benchmark once per (optimizer, lr, seed) with that
single optimizer, then aggregates best-per-optimizer from the recorded
bundles. This kills the "your baseline is mistuned" confound before any
public claim: every optimizer gets the same tuning grid and only its best
configuration is compared.

Only bundles whose runs carry a recorded `lr` (benchmark >= this change)
participate in the report; older bundles are ignored.

Usage (T4):
    python -m benchmarks.suites.comparison.t4_lr_sweep \
        --suite vit --epochs 3 --amp \
        --optimizers adamw,bnb8bit,sgdm,smo,smo8bit \
        --lrs 0.0003,0.001,0.003 --seeds 1234

Local smoke:
    python -m benchmarks.suites.comparison.t4_lr_sweep --suite vit \
        --device cpu --epochs 1 --width 64 --depth 2 --heads 4 \
        --limit_batches 2 --batch 32 --optimizers adamw,smo8bit \
        --lrs 0.001,0.01 --seeds 1234

Resume-friendly: combos whose bundle already exists are skipped unless
--force. Use --analyze_only to rebuild the report without launching.
Unknown extra flags are forwarded verbatim to t4_memory_benchmark.
"""

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict

from benchmarks._paths import RESULTS_DIR

METRIC_KEY = {"vit": "final_test_acc", "gpt": "final_val_loss"}
# Order matters: smo8bit must be tested before smo
FAMILY_RULES = [
    ("SMO-8bit", "smo8bit"),
    ("bnb-AdamW8bit", "bnb8bit"),
    ("SGD-M", "sgdm"),
    ("AdamW", "adamw"),
    ("SMO", "smo"),
]


def slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def bundle_path(suite: str, tag: str):
    return RESULTS_DIR / f"t4_{suite}_memory_results_{tag}.json"


def family_of(variant: str) -> str:
    for prefix, family in FAMILY_RULES:
        if variant.startswith(prefix):
            return family
    return variant.lower().replace(" ", "_")[:12]


def launch(args, lrs, seeds, opts):
    launched = skipped = 0
    for opt in opts:
        for lr in lrs:
            for seed in seeds:
                tag = f"lr{slug(lr)}_{opt}_s{seed}"
                path = bundle_path(args.suite, tag)
                if path.exists() and not args.force:
                    print(f"[skip] {path.name} exists")
                    skipped += 1
                    continue
                cmd = [
                    sys.executable, "-m",
                    "benchmarks.suites.comparison.t4_memory_benchmark",
                    "--suite", args.suite,
                    "--optimizers", opt,
                    "--seed", str(seed),
                    "--lr", repr(lr),
                    "--tag", tag,
                    *args.passthrough,
                ]
                print("[run]", " ".join(cmd), flush=True)
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"[warn] {tag} exited with {result.returncode}")
                launched += 1
    print(f"\nlaunched={launched} skipped={skipped}")


def analyze(args):
    mkey = METRIC_KEY[args.suite]
    groups = defaultdict(list)  # (family, lr) -> list of metric values
    for path in RESULTS_DIR.glob(f"t4_{args.suite}_memory_results*.json"):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for run in bundle.get("runs", []):
            metrics = run.get("metrics", {})
            lr = run.get("lr")
            if metrics.get("status") != "ok" or lr is None:
                continue
            value = metrics.get(mkey)
            if isinstance(value, (int, float)):
                groups[(family_of(run["variant"]), float(lr))].append(value)

    if not groups:
        print("No bundles with recorded lr found — nothing to analyze yet.")
        return

    higher_better = args.suite == "vit"
    lines = [f"LR sweep | suite={args.suite} | metric={mkey} "
             f"({'higher better' if higher_better else 'lower better'})", ""]

    best_per_family = {}
    for family in sorted({fam for fam, _ in groups}):
        rows = sorted((lr, vals) for (fam, lr), vals in groups.items() if fam == family)
        means = {lr: statistics.mean(vals) for lr, vals in rows}
        best_mean = max(means.values()) if higher_better else min(means.values())
        lines.append(f"--- {family} ---")
        for lr, vals in rows:
            cell = f"{means[lr]:.2f}"
            if len(vals) > 1:
                cell += f"±{statistics.stdev(vals):.2f}"
            marker = "  <-- best" if means[lr] == best_mean else ""
            lines.append(f"  lr={lr:<9g} n={len(vals)}  {cell}{marker}")
        best_per_family[family] = (best_mean, min(lr for lr, m in means.items() if m == best_mean))
        lines.append("")

    ranking = sorted(best_per_family.items(), key=lambda kv: kv[1][0], reverse=higher_better)
    lines.append("RANKING by best-tuned configuration:")
    for rank, (family, (mean, lr)) in enumerate(ranking, 1):
        gap = ""
        if rank > 1:
            gap = f" ({mean - ranking[0][1][0]:+.2f})"
        lines.append(f"  {rank}. {family:<10} best@lr={lr:g}: {mean:.2f}{gap}")

    report = "\n".join(lines)
    print(report)
    if args.out:
        out_path = RESULTS_DIR / args.out
        out_path.write_text(f"# LR sweep ({args.suite})\n\n```\n{report}\n```\n", encoding="utf-8")
        print(f"\nWritten to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="LR fairness sweep over t4_memory_benchmark")
    parser.add_argument("--suite", default="vit", choices=["vit", "gpt"])
    parser.add_argument("--optimizers", default="", help="comma-separated optimizer names")
    parser.add_argument("--lrs", default="", help="comma-separated learning rates")
    parser.add_argument("--seeds", default="1234", help="comma-separated seeds")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run even if bundle exists")
    parser.add_argument("--out", default=None, help="markdown filename inside results dir")
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough

    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    opts = [x.strip() for x in args.optimizers.split(",") if x.strip()]

    if not args.analyze_only:
        launch(args, lrs, seeds, opts)
    analyze(args)


if __name__ == "__main__":
    main()

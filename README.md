# SMO — State Memory Optimizer

[![Status: Active Research](https://img.shields.io/badge/status-active_research-green)](docs/T4_FINDINGS.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

SMO is a research optimizer for PyTorch that compresses Adam's moment states
(`exp_avg`, `exp_avg_sq`) with **structured spatial pooling** plus optional
**INT8 block-wise quantization**, cutting persistent optimizer state by
**93–98%** and step-time peak memory (row-banded update), while matching
Adam-class quality at equal budget — and beating every tuned baseline in
short-budget vision training. This repository is also a working example of
adversarial self-evaluation: the campaign's fairness protocol reversed its own
headline result, and that autopsy is documented in full.

---

## 🎯 Headline results (NVIDIA T4, TinyViT/CIFAR-10)

### Quality vs every best-tuned baseline (short-budget regime)

LR-fairness sweep (`benchmarks/suites/comparison/t4_lr_sweep.py`), each optimizer
at its own best learning rate, ViT 3 epochs:

| Rank | Optimizer | best acc | best lr |
|---|---|---|---|
| 1 | **SMO k=0.25** | **59.97** | 3e-3 |
| 2 | **SMO-8bit k=0.25** | **59.53** | 3e-3 |
| 3 | AdamW-fp32 | 57.33 | 6e-4 |
| 4 | bnb-AdamW8bit | 57.29 | 3e-4 |
| 5 | SGD-M (momentum 0.9) | 52.62 | 6e-2 |

At the longer budget (10 epochs, 3 seeds, per-optimizer tuned LRs):

| Optimizer | best-known acc (10 ep) | config |
|---|---|---|
| **bnb-AdamW8bit** | **70.65 ± 0.63** | lr 3e-4 |
| SMO-8bit k=0.25 | 68.66 ± 0.14 | lr 1e-3 |
| SMO k=0.25 | 68.40 ± 0.38 | lr 1e-3 |
| AdamW-fp32 | 65.62 ± 0.19 | lr 6e-4 |

**Honest reading**: under symmetric tuning, bitsandbytes-8bit currently leads long-budget
vision quality; SMO-8bit matches Adam-class quality (±3) with **44× less state**, wins the
short-budget regime, and is the only option that also minimizes step-time peak memory.
A methodological finding from this campaign: **LRs selected on short-horizon proxies do
not transfer to longer budgets — in both directions** (it inflated our own initial +13
claim; see `docs/T4_FINDINGS.md` for the full autopsy).

### Loss-matched generalization frontier

At equal train loss (same seed ⇒ identical batch order), SMO delivers
+3.8…+7.5 test accuracy over AdamW across the shared range — and the margin
widens as fit improves. Combined with faster loss descent, SMO shifts the whole
loss↔generalization frontier rather than trading fit for generalization.

### Memory

| Claim | Evidence |
|---|---|
| Persistent state −93% (SMO) / −98% (SMO-8bit) | state_dict accounting, all runs |
| Trains ~700M params where AdamW OOMs on a T4 | killer demo: AdamW OOM; SMO-8bit ok @ **10.7 GB peak** (lowest of survivors), 94 MB state |
| Usable-LR window shifted upward (~10×) | sweep curves: AdamW collapses ≥3e-3; SMO peaks there |

---

## 🔬 Mechanism (what we established, and how)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Advantage requires correlated neighborhoods (locality prior) | ✅ **Confirmed** | random row/col permutation before pooling collapses SMO-8bit to the noise-only level (55.53±0.34 → 52.13 ≈ bnb's 52.39) |
| "Just less adaptive → SGD-like" | ❌ Falsified | tuned SGD-M frontier is *worse* than AdamW's; SMO sits +14 above SGD-M |
| Generic beneficial noise | ❌ Falsified | pure quantization (bnb) buys only +0.7…+2.8 of the frontier gain |
| Output-layer smoothing harms LM | ✅ Partially | protecting emb/head recovers ~⅓ of the LM gap |
| LR-window shift | ✅ Confirmed | full sweep curves, both sides bracketed |

Working story: **spatial consensus over correlated coordinates low-pass filters
moment estimates while preserving signal geometry** — discarding high-frequency
estimation noise without destroying adaptivity. Full log: `docs/T4_FINDINGS.md`.

---

## ⚠️ Honest limitations

- **Language modeling regresses**: char-LM val loss +0.21…+0.49 vs AdamW;
  `protect_output` recovers ~⅓. SMO's home turf today is vision-from-scratch.
- **Only 2D params ≥32×32 are compressed**; conv weights fall back to dense moments.
- **Scale**: TinyViT (~15M) on CIFAR-10 and a ~40M char-LM are toy regimes; the
  700M killer-demo point validates fit/memory, not quality.
- **Resident ≠ persistent**: SMO-Spatial keeps full-resolution reconstruction
  buffers cached (peak can exceed AdamW); use **SMO-8bit `low_peak=True`** when
  step-time peak matters — its row-banded update allocates nothing full-size.
- Several results are single-seed (flagged per-case in `docs/T4_FINDINGS.md`).

---

## 📂 Repository structure

```
smo_optimizer/
├── smo/
│   ├── optimizers/      # SMO-Spatial, SMO-8bit (low_peak), Triton variants
│   ├── activations/     # Experimental activation compression
│   └── experimental/    # Spectral variants (Walsh, DCT)
├── benchmarks/
│   ├── suites/
│   │   ├── training/        # MNIST, CIFAR-10, MiniGPT end-to-end
│   │   ├── optimizer_step/  # Step-time microbenchmark, consistency smoke
│   │   ├── spectral/        # Walsh/DCT baselines
│   │   ├── activations/     # Activation compression tests
│   │   └── comparison/      # t4_memory_benchmark, t4_lr_sweep,
│   │                        # t4_summarize, t4_loss_matched
│   ├── runners/             # Colab/Kaggle T4 notebook, Modal, DirectML
│   ├── results/             # Versioned JSON bundles + aggregates
│   ├── METHODOLOGY.md       # Persistent-vs-resident definitions, seeding policy
│   └── CATALOG.md           # Canonical inventory of entrypoints
├── profiles/                # torch.profiler step decomposition (device-synced)
├── docs/                    # PROJECT_FOUNDATION, ROADMAP, T4_FINDINGS
├── tests/                   # pytest suite (testpaths configured)
└── spectral/, smo/*.py      # Backward-compat shims (see CATALOG.md)
```

---

## 🚀 Quick start

```bash
# Unit tests
python -m pytest

# Single-GPU comparison vs AdamW & bitsandbytes (Colab/Kaggle T4 ready)
python -m benchmarks.suites.comparison.t4_memory_benchmark \
    --suite vit --epochs 3 --amp --seed 1234

# Killer demo: ~700M params, AdamW OOMs, SMO-8bit low_peak trains
python -m benchmarks.suites.comparison.t4_memory_benchmark --suite gpt \
    --d_model 1280 --layers 36 --block_size 256 --batch 8 --steps 200 \
    --amp --low_peak --seed 1234

# LR fairness sweep (best-tuned comparison)
python -m benchmarks.suites.comparison.t4_lr_sweep --suite vit --epochs 3 --amp \
    --optimizers adamw,bnb8bit,sgdm,smo,smo8bit \
    --lrs 0.0003,0.001,0.003 --seeds 1234

# Aggregation & analysis
python -m benchmarks.suites.comparison.t4_summarize
python -m benchmarks.suites.comparison.t4_loss_matched --suite vit

# One-click notebook (clone, install, run, analyze):
#   benchmarks/runners/colab/t4_benchmark_colab.ipynb
```

All results land in `benchmarks/results/` as JSON bundles with git commit,
framework versions and full training histories.

---

## 📋 Documentation

- **T4 findings & hypothesis log** → `docs/T4_FINDINGS.md`
- **Benchmark methodology** (seeding, persistent-vs-resident memory) → `benchmarks/METHODOLOGY.md`
- **Catalog of entrypoints** → `benchmarks/CATALOG.md`
- **Roadmap & task tracking** → `TODO.md`, `docs/ROADMAP.md`
- **Changelog** → `CHANGELOG.md`

---

## 📜 License

MIT — see `LICENSE`.

---

## 🙋 Contributing

Research project. Task tracking lives in `TODO.md`; benchmark standards in
`benchmarks/METHODOLOGY.md`. For questions, open a GitHub issue.

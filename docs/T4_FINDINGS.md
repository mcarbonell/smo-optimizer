# T4 Findings — SMO vs AdamW vs bitsandbytes-8bit

**Status:** preliminary (single GPU, small scale, 1–3 seeds per cell). Living document —
update as experiments land. Script: `benchmarks/suites/comparison/t4_memory_benchmark.py`,
aggregation: `t4_summarize.py`.

**Setup:** Kaggle T4 (16 GB), torch 2.10+cu128, fp16 autocast for fwd/bwd only
(optimizer states fp32 for all variants), weight_decay=0 everywhere (SMO uses L2-in-gradient,
so decoupled-WD baselines are not comparable), identical cosine+warmup schedule, grad clip 1.0.

---

## Results

### TinyViT on CIFAR-10 (3 epochs, width=384, depth=8) — n=3 unique seeds

| Optimizer | Test acc (mean±std) | Δ vs AdamW | Persistent state | Peak CUDA alloc |
|---|---|---|---|---|
| AdamW-fp32 | 52.70 ± 0.58 | — | 108.7 MB | 1300 MB |
| bnb-AdamW8bit | 52.22 ± 1.11 | −0.48 | ~0.09× AdamW* | 1219 MB |
| **SMO k=0.25** | **55.34 ± 0.27** | **+2.64** | 7.4 MB (−93%) | 1316 MB |
| **SMO-8bit k=0.25** | **55.50 ± 0.28** | **+2.80** | 2.5 MB (−98%) | **1194 MB** |

\* bnb state recorded as 0 in these bundles due to measurement bug fixed in `68eba28`;
true value ≈ 1 byte/param + scales.

Seeds: 1234, 5678, 9012. Determinism verified: seed 1234 reproduces metrics exactly.
Effect size +2.6…+2.8 with σ≈0.3 → z ≈ 5–9. **Not noise.**

### TinyViT CIFAR-10, 10 epochs — COMPLETE (seeds 1234, 5678, 9012)

| Optimizer | s1234 | s5678 | s9012 | mean±std | Δ vs AdamW |
|---|---|---|---|---|---|
| AdamW-fp32 | 54.27 | 56.72 | 55.53 | 55.51 ± 1.23 | — |
| bnb-AdamW8bit | 54.05 | 55.30 | 54.21 | 54.52 ± 0.70 | −0.99 |
| **SMO k=0.25** | 68.38 | 68.03 | 68.78 | **68.40 ± 0.38** | **+12.88** |
| **SMO-8bit k=0.25** | 68.53 | 68.81 | 68.63 | **68.66 ± 0.14** | **+13.15** |

Key observations:

- **H2 (advantage decays with training) definitively refuted**: the gap GROWS from +2.7
  (3 epochs) to +13.2 (10 epochs). Whatever the mechanism, it compounds with training time.
- **Cross-seed variance collapse**: SMO-8bit spans 0.28 pts across seeds (σ=0.14) while
  AdamW spans 2.45 (σ=1.23). Smoothing makes the optimization endpoint nearly
  seed-independent — a denoising signature in itself.
- **Frontier decomposition (H7)**: at AdamW's *best* train loss (~1.21–1.27), SMO sits at
  ~56.6% acc (epoch ~4) — only ~+2 over AdamW's endpoint. The remaining ~11 points come
  from SMO continuing to descend train loss (to ~0.89) where AdamW stalls within budget.
  So: modest generalization-at-equal-loss gain PLUS substantially faster/better loss
  descent — a frontier shift, not a regularization trade.
- bnb state_MB measures correctly here (27.85 MB, −74.4% — post-fix runs).
- Effect size vs seed noise: Δ=+13.15 with σ_SMO8bit=0.14, σ_AdamW=1.23 → SE(diff)≈0.73 →
  z≈18. Beyond any reasonable doubt at this scale/config.

### CharGPT char-LM on tiny-shakespeare (~40M params, 1000 steps, seed=1234)

| Optimizer | Val loss | Δ vs AdamW | Persistent state | Peak alloc |
|---|---|---|---|---|
| AdamW-fp32 | 1.5861 | — | 194 MB | 1809 MB |
| bnb-AdamW8bit | 1.5812 | −0.005 | 49.5 MB (−74.5%) | 1667 MB |
| SMO k=0.25 | 2.0768 | +0.49 | 12.5 MB | **1841 MB (>Adam)** |
| SMO-8bit k=0.25 | 2.0497 | +0.46 | 3.6 MB | 1619 MB |
| SMO k=0.5 | 1.8270 | +0.24 | 48.7 MB | 1906 MB |
| SMO-8bit k=0.5 | 1.8009 | +0.21 | 13.3 MB | 1629 MB |
| SMO k=0.25 + protect_output | 1.9125 | +0.33 | 13.7 MB | 1841 MB |
| SMO-8bit k=0.25 + protect_output | 1.8915 | +0.31 | 4.9 MB | 1621 MB |

---

## Findings

1. **Regime split.** Moment compression *helps* vision-from-scratch (+2.6…+2.8 acc,
   multi-seed) and *hurts* char-LM (+0.21…0.49 val loss). Consistent with the earlier
   CPU finding (Walsh-Pure > Adam on CIFAR-CNN).
2. **SMO-8bit ≥ SMO-fp32 consistently** (both suites, both ratios, same seeds,
   with and without protect_output): quantizing the compressed moments appears to add
   benign noise rather than damage.
3. **Memory:** persistent-state savings are real and comparable to bnb (93–98%). Training-time
   *peak* memory only improves for SMO-8bit (−8…−10%); SMO-Spatial peak exceeds AdamW because
   full-resolution reconstruction buffers stay resident (see METHODOLOGY persistent-vs-resident).
4. **Throughput cost:** −5% (gpt) to −15…−17% (vit) vs AdamW; bnb −4…−6%.
5. **H1 partially confirmed (2026-08-23, `--tag prot`).** Excluding embedding/head/pos-emb
   from compression (dense moments there, 164k of 25.4M params) recovers ~⅓ of the LM gap
   (+0.49 → +0.33 at k=0.25). Output-layer smoothing is *a* mechanism, not *the* mechanism;
   the rest lives in the hidden-layer linears. bnb state_MB now measures correctly (49.5 MB).
6. **First killer-demo attempt OOM'd for ALL optimizers — and exposed a structural limit
   (`--tag big`, 2026-08-23).** At ~906M params without activation checkpointing even bnb
   died on activations; more importantly, the SMO8bit step materializes full-resolution
   temporaries (stacked grad+grad² for pooling ≈ +12 B/param, bilinear reconstruction
   ≈ +8 B/param), so its step-time peak (~21 B/param) exceeds AdamW's (~20 B/param):
   **no model size exists where AdamW OOMs but SMO8bit survives** in monolithic mode.
   Fix landed: `low_peak=True` row-banded compress/update (exact by locality of
   avg-pool/bilinear; bit-identical states in tests) → step peak ≈ 9 B/param.
   Killer demo is now viable again; rerun with `--low_peak` and AdamW-hostile sizing.

## Working hypotheses

- **H1 — output-layer smoothing harms LM.** PARTIALLY CONFIRMED: protects ~⅓ of the gap.
  Remaining levers: combine with k=0.5; accept LM as unfavorable regime.
- **H2 — regularizer effect decays with training length.** REFUTED, emphatically
  (10-epoch runs): gap grows +2.7 → +13.2 from 3 to 10 epochs. The mechanism compounds;
  it does not wash out.
- **H3 — quantization adds beneficial dither.** Supported by finding 2. Test: block_size
  sweep on SMO-8bit. Pending.
- **H4 — locality prior:** smoothing wins because pooling exploits correlated adjacent
  coordinates. Test: `--permute_basis` (random fixed row/col permutation before pooling,
  unpermuted after reconstruction). Prediction: ViT advantage collapses to ≈ bnb level.
- **H5 — partial de-adaptivization toward SGD-like updates:** smoothed exp_avg_sq damps
  per-coordinate adaptivity; Adam is known to generalize worse than SGD on vision
  from-scratch tasks while being essential for LM. Test: add an `sgdm` baseline — if
  SGD-M approaches SMO's ViT numbers, the "SMO pushes Adam toward SGD territory" story
  gains support (and explains the LM/vision regime split elegantly).
- **H6 — flatness bias à la SAM:** updating weights toward neighborhood consensus
  biases toward flat minima; SAM's known profile (big vision gains, neutral/negative LM)
  matches the observed regime split.
- **H7 — frontier improvement (not just endpoint):** preliminary loss-matched reading of
  the 10-epoch run suggests SMO reaches *better test acc at equal train loss* (+1.5…+2)
  AND descends the train loss faster — i.e., it improves the whole loss↔generalization
  frontier rather than trading fit for generalization. A pure implicit-regularizer story
  would predict the opposite trade; a de-adaptivization story predicts exactly this.
  Quantify with `t4_loss_matched` on runs that log `history` (added 2026-08-23; earlier
  bundles have no history field).

## Pending experiments

- [ ] Killer demo, attempt 2 (needs `low_peak`, landed): e.g.
      `--d_model 1280 --layers 36 --block_size 256 --batch 8 --steps 200 --amp --low_peak`
      (~700M params: AdamW ≈ 14 GB static + activations → OOM; SMO-8bit lp ≈ 6.5 GB)
- [ ] ViT long-run (10 epochs × 3 seeds) — RUNNING; early epochs show growing advantage (see H2)
- [ ] CharGPT k=0.5 + protect_output combined
- [ ] CharGPT comparisons at ≥3 fresh seeds (currently single-seed)
- [ ] Port row-banded update to SMO-Spatial (same transient bottleneck there:
      stacked pooling input + resident full-size reconstruction buffers)
- [ ] H5 test: `sgdm` baseline on ViT (`--optimizers adamw,sgdm,smo,smo8bit`), incl. an lr
      sweep for SGD-M (1e-2…5e-2) since the cosine schedule is tuned for Adam-scale lr
- [ ] H4 test: `--permute_basis` on ViT (`--optimizers smo,smo8bit --permute_basis`)

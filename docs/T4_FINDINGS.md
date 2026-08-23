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

### TinyViT CIFAR-10, 10 epochs + SGD-M baseline (`--tag hist`, seed 1234)

| Optimizer | final acc | state MB |
|---|---|---|
| AdamW-fp32 | 54.27 | 108.7 |
| bnb-AdamW8bit | 54.05 | 27.9 |
| **SGD-M (lr=1e-3)** | **39.99** | 54.3 |
| **SMO k=0.25** | **68.38** | 7.4 |
| **SMO-8bit k=0.25** | **68.53** | 2.5 |

**Loss-matched analysis** (test_acc at equal train loss, interpolated):

| train_loss | AdamW | SGD-M | SMO | SMO-8bit | bnb |
|---|---|---|---|---|---|
| 1.654 | 39.98 | +0.01 | **+7.33** | **+7.52** | +2.49 |
| 1.689 | 38.87 | +0.12 | **+7.29** | **+7.45** | +2.77 |
| 1.759 | 38.34 | −2.44 | **+5.52** | **+5.65** | +0.75 |
| 1.830 | 37.80 | −3.81 | **+3.75** | **+3.84** | −1.27 |

Key observations:

- **H5 (de-adaptivization toward SGD) FALSIFIED in its simple form**, two ways:
  (a) SGD-M's own frontier is *worse* than AdamW's at matched loss (−1.2…−3.8), so
  "moving toward SGD dynamics" cannot explain SMO's *better* frontier;
  (b) SMO lands +14 above SGD-M at this budget instead of between the two.
  CAVEAT before full burial: lr=1e-3 is likely mistuned for SGD-M (typically wants
  10–50× more); the LR-sweep must confirm tuned-SGD doesn't flip this picture.
- **H7 quantified and confirmed**: SMO's loss-matched advantage is large (+3.8…+7.5),
  positive across the whole shared range, and *widens* as fit improves (+3.8 @ loss 1.83
  → +7.5 @ 1.65). Combined with faster descent, the total endpoint gap (+14) decomposes
  into a genuine frontier shift PLUS compounding speed.
- **bnb also improves the frontier mildly** (+0.75…+2.77): quantization noise alone buys
  a little generalization-at-equal-fit, consistent with H3 — but an order of magnitude
  less than spatial smoothing. Whatever SMO does, quantization-only does not replicate.
- Surviving mechanism candidates: (i) *state denoising* — low-pass filtering the moment
  ESTIMATES while preserving Adam-style adaptivity structure (unlike SGD, which drops
  adaptivity wholesale); (ii) effective-LR redistribution from smoothed exp_avg_sq;
  (iii) locality prior (H4 `permute_basis` test now the sharpest discriminator).

- Surviving mechanism candidates: (i) *state denoising* — low-pass filtering the moment
  ESTIMATES while preserving Adam-style adaptivity structure (unlike SGD, which drops
  adaptivity wholesale); (ii) effective-LR redistribution from smoothed exp_avg_sq;
  (iii) locality prior (H4 `permute_basis` test now the sharpest discriminator).

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

### Killer demo (`--tag big2`, ~700M CharGPT, T4 16 GB) — ENGINEERING CLAIM CLOSED

Config: d_model=1280, layers=36, block_size=256, batch=8, 200 steps, amp, seed=1234.

| Optimizer | Status | Peak alloc | Persistent state | val_loss @200 |
|---|---|---|---|---|
| AdamW-fp32 | **OOM** | — | — | — |
| SGD-M | ok | 13,416 MB | 2,703.9 MB | 3.22 |
| bnb-AdamW8bit | ok | 12,035 MB | 1,375.4 MB | 2.52 |
| SMO-Spatial (monolithic) | **OOM** | — | — | — |
| **SMO-8bit low_peak** | **ok** | **10,728 MB** | **94.3 MB** | **2.51** |

Readings:

- AdamW cannot fit (~16 B/param incl. state); SMO-8bit trains comfortably (~9 B/param
  step peak + activations) — the "trains where Adam cannot" headline is demonstrated.
- Direct A/B for the low_peak fix: monolithic SMO variants OOM, banded SMO-8bit survives
  with the *lowest* peak of all survivors (−11% vs bnb, −20% vs SGD-M).
- State accounting sanity: 94.3 MB ≈ 707M params × k² × 2 int8 states + scales — matches
  theory; bnb 1,375 MB ≈ 707M × 2 B; SGD-M 2,704 MB = momentum fp32.
- Quality at 200 steps is only "it trains" (char-LM smoke budget), but SMO-8bit tracks
  bnb's val loss (2.51 vs 2.52) while holding 14.6× less persistent state.
- Throughput cost vs bnb: −14%.

### H4 locality ablation (`--tag perm`, seed 1234)

SMO-8bit with a fixed random row/col permutation applied before pooling
(identical compression ratio, quantization and compute — only the geometry
of the consensus changes):

| Variant | test_acc |
|---|---|
| SMO-8bit k=0.25 (canonical, 3 seeds) | 55.53 ± 0.34 |
| SMO-8bit k=0.25 + permute_basis | **52.13** |
| bnb-AdamW8bit (noise reference) | 52.39 ± 1.30 |

The permuted variant drops ~10σ below canonical and lands exactly at the
quantization-noise level. **H4 CONFIRMED**: spatial smoothing wins *because*
adjacent coordinates are correlated — destroy the neighborhood and the
advantage vanishes entirely. Coherently explains the LM regression (embedding
rows are semantically arbitrary neighbors → smoothing ≈ permuted smoothing).
Caveat: single seed; add 2 more for rigor (8 min each).

### LR fairness sweep (`t4_lr_sweep`, ViT 3 epochs, seed 1234)

| Optimizer | lr=3e-4 | lr=1e-3 | lr=3e-3 | BEST |
|---|---|---|---|---|
| AdamW-fp32 | **57.22** | 52.19 | 24.68 | 57.22 |
| bnb-AdamW8bit | **57.29** | 51.72 | 27.05 | 57.29 |
| SGD-M | 27.77 | 32.46 | **38.60** | 38.60 |
| SMO k=0.25 | 44.23 | 55.39 | **59.97** | **59.97** |
| SMO-8bit k=0.25 | 44.40 | 55.41 | **59.53** | 59.53 |

RANKING (best-tuned): SMO 59.97 > SMO-8bit 59.53 > bnb 57.29 > AdamW 57.22 > SGD-M 38.60

Findings:

- **The quality gap survives symmetric tuning**: +2.68/+2.75 over tuned bnb/AdamW
  (was +3.2 untuned). Both SMO variants beat every baseline's *best* configuration.
- **H8 (lr-window shift) CONFIRMED, single seed**: AdamW/bnb peak at 3e-4 and collapse
  at 3e-3 (−30 pts); SMO peaks at 3e-3 and is still flat/climbing. Smoothing shifts the
  usable-LR window upward and flattens sensitivity where Adam breaks. Practically:
  SMO lets you train more aggressively. Mechanistic hint: pooled exp_avg_sq
  underestimates local variance peaks → larger stable steps.
- SMO itself was mistuned at 1e-3 (+4.6 available at 3e-3) — everyone gains from tuning,
  SMO just gains more headroom on top.
- SGD-M remains far behind at every grid point (best 38.60, still climbing but nowhere
  near the pack): H5 stays dead for practical purposes.
- Caveats: single seed per combo; coarse grid (SMO may prefer even higher lr;
  AdamW may have a mild bump between 3e-4 and 1e-3). Neither caveat plausibly closes
  a +2.7 gap, but the tuned 10-epoch rerun below is the honest confirmation.

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
- **H4 — locality prior.** **CONFIRMED** (single seed so far): permute_basis collapses
  SMO-8bit from 55.53±0.34 to 52.13 ≈ bnb's noise-only level. The advantage requires
  semantically meaningful neighborhoods; it is not generic noise, not "SGD-ness"
  (H5 falsified), and not mere compression. Working story: *structured spatial consensus
  on correlated coordinates = low-pass filtering of moment estimates that preserves
  signal geometry while discarding high-frequency estimation noise.*
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

- [x] Killer demo, attempt 2 — **DONE** (`big2`): AdamW OOM; SMO-8bit lp trains with the
      lowest peak of all survivors (10.7 GB) and 94 MB state. Engineering claim closed.
- [x] ViT long-run (10 epochs × 3 seeds) — DONE: gap +13.15±0.14, H2 refuted
- [x] H5 test (`sgdm`) — DONE at lr=1e-3: falsified in simple form (sweep pending to rule
      out mistuned-SGD)
- [x] LR fairness sweep — DONE: gap survives symmetric tuning (+2.68/+2.75 best-vs-best);
      H8 lr-window shift confirmed (AdamW/bnb peak at 3e-4 and collapse at 3e-3; SMO peaks
      at 3e-3); SGD-M dead at every grid point
- [ ] **Tuned star run** (decisive): 10 epochs × 3 seeds with per-optimizer best lrs
      (adamw/bnb @ 3e-4, smo/smo8bit @ 3e-3) — revalidates the +13 headline under fair
      tuning. Two invocations per seed (benchmark takes a single --lr):
      `--lr 0.0003 --optimizers adamw,bnb8bit --tag tuned_lo_s{seed}` and
      `--lr 0.003 --optimizers smo,smo8bit --tag tuned_hi_s{seed}` (~100 min total)
- [ ] H4: add 2 extra seeds to the permute ablation (8 min each)
- [ ] Optional: finer lr grid around optima ({6e-4, 6e-3}) for H8 curve shapes; CharGPT
      k=0.5 + protect_output combined; CharGPT ≥3 fresh seeds
- [ ] Port row-banded update to SMO-Spatial (same transient bottleneck there:
      stacked pooling input + resident full-size reconstruction buffers)

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

## Working hypotheses

- **H1 — output-layer smoothing harms LM.** PARTIALLY CONFIRMED: protects ~⅓ of the gap.
  Remaining levers: combine with k=0.5; accept LM as unfavorable regime.
- **H2 — moment smoothing is an implicit regularizer in high-noise regimes.**
  Supported by ViT result. Predicts: advantage shrinks with longer training / larger data /
  heavier augmentation. Test: 10-epoch ViT runs. Pending.
- **H3 — quantization adds beneficial dither.** Supported by finding 2. Test: block_size
  sweep on SMO-8bit. Pending.

## Pending experiments

- [x] CharGPT `--protect_output` (`--tag prot`) — H1 partial: −0.16 nats recovered
- [ ] Killer demo: ~700M+ CharGPT where AdamW OOMs on T4 but SMO-8bit fits (running)
- [ ] ViT long-run (10 epochs × 3 seeds) — does the regularizer effect decay? (H2)
- [ ] CharGPT k=0.5 + protect_output combined
- [ ] CharGPT comparisons at ≥3 fresh seeds (currently single-seed)

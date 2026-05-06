# TODO.md - ROADMAP Phases 2-4 Progress Tracker

## Current Status (2026-05-06)

- ✅ **Phase 2: Correctness/Measurement** — COMPLETE (2026-05-05)
- 🔄 **Phase 3: Bottleneck Analysis** — PROFILE SCRIPT READY, awaiting GPU (<50% utilization)
- ✅ **Phase 4.1: Buffer Reuse** — COMPLETE (8-24% CPU speedup)

---

## Phase 2 Complete ✅

Infrastructure: seeding policy, multi-seed microbench, timing helpers, end-to-end baselines (MNIST 89.1% mem savings, +0.14% acc gap).

## Phase 3 Prep: Profiling Suite ✅ (2026-05-06)

**Added:** `profiles/profile_smo_step.py`

Features:
- Manual timer breakdown (compress / update_compressed / upsample / full_step)
- `torch.profiler` integration for kernel traces (JSON export)
- CPU + CUDA support
- Configurable shape, k_ratio, steps, warmup, seed

**Sample output** (512×512, k=0.25, CPU):
```
compress            : mean=3.7482 ms
update_compressed   : mean=0.2583 ms
upsample            : mean=2.8164 ms
full_step           : mean=9.0570 ms
```

**Interpretation:**
- Compression (pooling) ≈ 41% of step time
- Upsampling (bilinear interpolate) ≈ 31%
- Elementwise update ≈ 3%
- Overhead / weight update ≈ 25%

**Next:** Run on GPU (DirectML) when device < 50% load to get accurate kernel-level timings and memory breakdown.

## Phase 4.1 Buffer Reuse ✅ (2026-05-05)

Pre-allocated intermediate buffers per-parameter; eliminated per-step allocations.
Speedup: 8-24% depending on tensor shape (larger → smaller % gain).

---

## Completed Script Updates (Seeding)

All canonical benchmarks now expose `--seed` / `--seeds`:
- `benchmark_step_time.py` (microbench) — multi-seed default 3
- `benchmark_mnist.py`, `benchmark_cifar10.py`, `benchmark_8bit.py` — end-to-end, seed=1234 default
- `benchmark_spectral_cpu.py` — end-to-end, seed=1234
- `benchmark_minillm.py` — smoke/end-to-end, seed=1234
- `benchmark_activations.py` — microbench, multi-seed default 3
- `test_accuracy_activations.py`, `test_accuracy_delta.py` — end-to-end, seed=1234
- `smoke_spatial_consistency.py` — smoke, seed=1234

---

## Next Actions

1. **Pending low-priority:** Add `--seed` to diagnostic scripts (`debug_accuracy_drop.py`, `benchmark_dct_fix.py`) — optional
2. **Blocked:** GPU profiling run (Phase 3) when DirectML available
3. **After profiling:** Implement Phase 4.2 (pooling kernel optimization or conditional upsampling bypass)
4. **Documentation:** Update CHANGELOG.md with buffer reuse metrics and profiling tool
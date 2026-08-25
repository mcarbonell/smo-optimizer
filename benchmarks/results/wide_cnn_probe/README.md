# Wide-CNN probe — conv-pooling replication test (2026-08-25, CPU, seed 1234)

Question: was the conv-pooling negative (−12…−17 on the 620k CIFAR_CNN) an
artifact of toy scale?

Model: `WideCNN` = CIFAR_CNN with channels ×2 (~2.5M params, conv-dominated).
Recipe identical to `benchmark_cifar10.py`. Script:
`benchmarks/suites/training/benchmark_cifar10_wide.py`.

## Verdict (10 epochs)

| Config | final test_acc | vs Adam | state |
|---|---|---|---|
| **SMO k=0.5 dense-conv** | **76.11** | **+1.50** | 1.9 MB |
| Adam | 74.61 | — | 19.0 MB |
| SMO k=0.5 conv-pool | 69.79 | −4.82 | 0.5 MB |

Two findings:

1. **The conv-pooling negative replicates at scale**: −6.32 vs its own
   dense-conv fallback (−5.57 on the small CNN at k=0.5). Structural damage
   from averaging across unrelated input channels; not a small-model artifact.
2. **Positive surprise**: dense-conv SMO-k0.5 BEATS Adam on a bigger CNN
   (+1.50). The Phase-2 reading ("CNNs are not SMO's turf", based on the 620k
   model trailing by −2…−4) flips once width doubles — consistent with the
   ViT story that moment smoothing helps as parameter count grows.

Single seed; multi-seed queued before any headline use.

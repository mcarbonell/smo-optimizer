"""Memory Benchmark for 8-bit Activations with deterministic seeding."""

import argparse
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks._paths import add_project_root_to_path
add_project_root_to_path()

from smo.activations_8bit import wrap_model_activations


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(10):
            layers.append(nn.Linear(4096, 4096))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(4096, 1024)

    def forward(self, x):
        return self.head(self.net(x))


def measure_memory_once(model, input_size, iterations=5, device='cpu'):
    model.to(device)
    inputs = torch.randn(input_size, device=device)
    targets = torch.randn(input_size[0], 1024, device=device)

    # Warmup
    for _ in range(2):
        output = model(inputs)
        loss = F.mse_loss(output, targets)
        loss.backward()
        model.zero_grad()

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()
    else:
        start_mem = 0

    start_time = time.time()
    for _ in range(iterations):
        output = model(inputs)
        loss = F.mse_loss(output, targets)
        loss.backward()
        model.zero_grad()
    end_time = time.time()

    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() - start_mem
    else:
        peak_mem = 0  # CPU has no peak memory tracking in PyTorch

    return {
        "peak_mem_mb": peak_mem / (1024**2),
        "avg_time_s": (end_time - start_time) / iterations
    }


def measure_memory(model_fn, input_size, iterations=5, device='cpu', seeds=None):
    """Run measurement across seeds and aggregate."""
    if seeds is None:
        seeds = [1234]

    results = []
    for seed in seeds:
        set_seed(seed)
        model = model_fn()
        res = measure_memory_once(model, input_size, iterations=iterations, device=device)
        results.append(res)

    import numpy as np
    peak_mem_vals = [r['peak_mem_mb'] for r in results]
    time_vals = [r['avg_time_s'] for r in results]

    return {
        'peak_mem_mb_mean': float(np.mean(peak_mem_vals)),
        'peak_mem_mb_std': float(np.std(peak_mem_vals)),
        'avg_time_s_mean': float(np.mean(time_vals)),
        'avg_time_s_std': float(np.std(time_vals)),
        'seeds': seeds
    }


def main():
    parser = argparse.ArgumentParser(description="Activation memory benchmark (SMO 8-bit activations)")
    parser.add_argument('--seeds', type=str, default='1234,5678,9012',
                        help='Comma-separated list of seeds for repetition')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of timed iterations per seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run on')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    batch_size = 32
    input_size = (batch_size, 4096)

    print(f"\n🧪 Activation Memory Benchmark | device={args.device} | iterations={args.iterations} | seeds={seeds}")
    print("="*80)

    # Baseline
    print("\n1️⃣  Baseline (standard float32 activations)")
    base_results = measure_memory(
        model_fn=LargeModel,
        input_size=input_size,
        iterations=args.iterations,
        device=args.device,
        seeds=seeds
    )

    # SMO 8-bit
    print("\n2️⃣  SMO 8-bit Activations (block_size=64)")
    smo_results = measure_memory(
        model_fn=lambda: wrap_model_activations(LargeModel(), block_size=64),
        input_size=input_size,
        iterations=args.iterations,
        device=args.device,
        seeds=seeds
    )

    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"{'Metric':<25} | {'Baseline':<20} | {'SMO 8-bit':<20} | {'Savings/Diff'}")
    print("-" * 90)

    mem_savings = (1 - smo_results['peak_mem_mb_mean'] / base_results['peak_mem_mb_mean']) * 100 if base_results['peak_mem_mb_mean'] > 0 else 0
    time_ratio = smo_results['avg_time_s_mean'] / base_results['avg_time_s_mean'] if base_results['avg_time_s_mean'] > 0 else 0

    print(f"{'Peak Activation RAM':<25} | {base_results['peak_mem_mb_mean']:>8.2f}±{base_results['peak_mem_mb_std']:.2f} MB | {smo_results['peak_mem_mb_mean']:>8.2f}±{smo_results['peak_mem_mb_std']:.2f} MB | {mem_savings:>6.1f}%")
    print(f"{'Avg Step Time':<25} | {base_results['avg_time_s_mean']:>8.4f}±{base_results['avg_time_s_std']:.4f}s | {smo_results['avg_time_s_mean']:>8.4f}±{smo_results['avg_time_s_std']:.4f}s | {time_ratio:>6.2f}x")

    if device != 'cuda':
        print("\n⚠️  Note: Peak memory stats are CUDA-only. On CPU, 'peak_mem_mb' shows 0. Benchmark measures timing overhead only.")


if __name__ == "__main__":
    main()

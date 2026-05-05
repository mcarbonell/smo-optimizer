"""Isolated optimizer-step microbenchmark for rapid iteration."""

import sys
from pathlib import Path
import argparse

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks._paths import add_project_root_to_path

add_project_root_to_path()

from benchmarks.timing import measure_steps, get_sync_fn
from smo import SMO, SMO8bit


# Benchmark classification: family=optimizer_step, category=microbenchmark, status=canonical


def run_optimizer_once(name, optimizer_cls, shape=(2048, 2048), steps=30, warmup=5, device='cpu', seed=1234, **kwargs):
    """Run optimizer once with given seed and return timing."""
    torch.manual_seed(seed)
    param = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32, device=device))
    optimizer = optimizer_cls([param], lr=1e-3, **kwargs)

    def one_step():
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one_step()

    sync_fn = get_sync_fn(device)
    timing = measure_steps(one_step, steps=steps, sync_fn=sync_fn)
    return timing


def run_optimizer(name, optimizer_cls, shape=(2048, 2048), steps=30, warmup=5, device='cpu', seeds=None, **kwargs):
    """
    Run optimizer multiple times with different seeds and return aggregated timing stats.
    If seeds is None, uses a single default seed.
    """
    if seeds is None:
        seeds = [1234]

    timings = []
    for seed in seeds:
        timing = run_optimizer_once(name, optimizer_cls, shape=shape, steps=steps, warmup=warmup, device=device, seed=seed, **kwargs)
        timings.append(timing)

    # Aggregate
    wall_ms = [t.wall_ms_per_step for t in timings]
    process_ms = [t.process_ms_per_step for t in timings]

    import numpy as np
    wall_mean = np.mean(wall_ms)
    wall_std = np.std(wall_ms)
    proc_mean = np.mean(process_ms)
    proc_std = np.std(process_ms)

    mem_info = measure_memory(device)

    print(
        f"{name:20} | shape {str(shape):12} | wall {wall_mean:8.3f}±{wall_std:.3f} ms | "
        f"cpu {proc_mean:8.3f}±{proc_std:.3f} ms | seeds={len(seeds)}{mem_info}"
    )


def measure_memory(device):
    """Returns memory info string for the device."""
    device_type = torch.device(device).type
    if device_type == 'cuda':
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        return f" | mem_alloc {alloc:6.1f} MB | mem_reserved {reserved:6.1f} MB"
    elif device_type == 'privateuse:0':  # DirectML
        return f" | device=DirectML (memory tracking N/A)"
    else:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Microbenchmark optimizer step time across shapes and devices")
    parser.add_argument('--shapes', type=str, default='64,128,256,512,1024,2048,4096',
                        help='Comma-separated list of square sizes (e.g. 64,128,256)')
    parser.add_argument('--steps', type=int, default=30, help='Number of timed steps')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup steps')
    parser.add_argument('--seeds', type=str, default='1234,5678,9012',
                        help='Comma-separated list of seeds for repetition')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'privateuse:0'],
                        help='Device to benchmark')
    args = parser.parse_args()

    shapes = [(int(s), int(s)) for s in args.shapes.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f"\nMICROBENCH: isolated optimizer step | device={args.device} | steps={args.steps} | warmup={args.warmup} | seeds={seeds}")
    print(f"{'='*120}")
    print(f"{'Optimizer':20} | {'Shape':12} | {'Wall ms/step (mean±std)':25} | {'CPU ms/step (mean±std)':25} | {'Memory'}")
    print(f"{'='*120}")

    for shape in shapes:
        run_optimizer("AdamW", torch.optim.AdamW, shape=shape, steps=args.steps, warmup=args.warmup, device=args.device, seeds=seeds)
        run_optimizer("SMO-Spatial", SMO, shape=shape, steps=args.steps, warmup=args.warmup, device=args.device, seeds=seeds, k_ratio=0.25)
        run_optimizer("SMO-Spatial-8bit", SMO8bit, shape=shape, steps=args.steps, warmup=args.warmup, device=args.device, seeds=seeds, k_ratio=0.25, block_size=64)


if __name__ == "__main__":
    main()

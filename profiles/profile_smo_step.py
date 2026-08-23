#!/usr/bin/env python3
"""
Profiling suite for SMO-Spatial optimizer step decomposition.

Measures time breakdown:
- Gradient compression (pooling)
- Moment updates (in compressed space)
- Upsampling (reconstruction)
- Full step

Supports CPU and CUDA/DirectML (with proper synchronization).

Usage:
    python profiles/profile_smo_step.py --shape 1024,1024 --steps 100 --warmup 10 --seed 1234
    python profiles/profile_smo_step.py --shape 1024,1024 --device cuda  # when GPU available
"""

import argparse
import sys
from pathlib import Path
import time
import math
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._paths import add_project_root_to_path
add_project_root_to_path()

from smo import SMO
from smo.optimizers._spatial_utils import compress_2d_pair, upsample_2d_pair
from benchmarks.timing import get_sync_fn


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DummyParam(nn.Module):
    """Wrapper to create a single parameter (bypass ParameterDict)."""
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(*shape))
    
    def parameters(self):
        return [self.weight]


def profile_step_cpu(optimizer, param, steps=100, warmup=10):
    """Profile individual components with detailed per-phase timing.

    Works on any device: on accelerators (CUDA/DirectML) each phase is
    synchronized before reading the clock so timings are not distorted by
    asynchronous execution. On CPU this is a no-op.
    """
    sync_fn = get_sync_fn(param.device)

    # Warmup
    for _ in range(warmup):
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Timers
    times = {
        'compress': [],
        'update_compressed': [],
        'upsample': [],
        'full_step': []
    }

    for _ in range(steps):
        param.grad = torch.randn_like(param)
        
        # We'll manually instrument the step to break it down
        state = optimizer.state[param]
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        grad = param.grad
        beta1, beta2 = optimizer.param_groups[0]['betas']

        # 1. Compress
        if sync_fn is not None:
            sync_fn()
        t0 = time.perf_counter()
        g_comp, g_sq_comp = compress_2d_pair(grad, grad.square(), exp_avg.shape)
        if sync_fn is not None:
            sync_fn()
        t1 = time.perf_counter()

        # 2. Update compressed
        exp_avg.mul_(beta1).add_(g_comp, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).add_(g_sq_comp, alpha=1 - beta2)
        if sync_fn is not None:
            sync_fn()
        t2 = time.perf_counter()

        # 3. Upsample
        m_rec, v_rec = upsample_2d_pair(exp_avg, exp_avg_sq, state['orig_shape'])
        v_rec.clamp_(min=0.0)
        if sync_fn is not None:
            sync_fn()
        t3 = time.perf_counter()

        # 4. Weight update (p.addcdiv_)
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = optimizer.param_groups[0]['lr'] / bias_correction1
        denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(optimizer.param_groups[0]['eps'])
        param.data.addcdiv_(m_rec, denom, value=-step_size)
        if sync_fn is not None:
            sync_fn()
        t4 = time.perf_counter()

        # Record
        times['compress'].append((t1 - t0) * 1000)  # ms
        times['update_compressed'].append((t2 - t1) * 1000)
        times['upsample'].append((t3 - t2) * 1000)
        times['full_step'].append((t4 - t0) * 1000)

        optimizer.zero_grad(set_to_none=True)

    # Statistics
    import numpy as np
    print(f"\n{'='*60}")
    print(f"PROFILE: shape={param.shape}, device={param.device}, steps={steps}")
    print(f"{'='*60}")
    for phase, vals in times.items():
        arr = np.array(vals)
        print(f"{phase:20s}: mean={arr.mean():.4f} ms  std={arr.std():.4f} ms  min={arr.min():.4f} ms  max={arr.max():.4f} ms")
    print(f"{'='*60}")


def profile_with_torch_profiler(optimizer, param, steps=50, warmup=5, device='cpu', output_dir=None):
    """Use torch.profiler for detailed kernel-level breakdown."""
    if output_dir is None:
        output_dir = Path.cwd() / "profiles"
    output_dir.mkdir(exist_ok=True)

    # Warmup
    for _ in range(warmup):
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    activities = [ProfilerActivity.CPU]
    if device == 'cuda' and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # set True for deep call stacks (more overhead)
    ) as prof:
        for step in range(steps):
            param.grad = torch.randn_like(param)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            prof.step()

    # Save trace
    trace_path = output_dir / f"smo_profile_{device}_{param.shape[0]}x{param.shape[1]}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nTrace saved to: {trace_path}")

    # Print top ops
    print(f"\n{'='*60}")
    print(f"TOP 20 OPERATORS (by self time):")
    print(f"{'='*60}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


def main():
    parser = argparse.ArgumentParser(description="Profile SMO-Spatial optimizer step")
    parser.add_argument('--shape', type=str, default='1024,1024', help='Parameter shape as H,W')
    parser.add_argument('--k_ratio', type=float, default=0.25, help='Compression ratio')
    parser.add_argument('--steps', type=int, default=100, help='Number of profiling steps')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup steps')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--use_torch_profiler', action='store_true', help='Use torch.profiler for kernel traces')
    parser.add_argument('--output_dir', type=str, default='profiles', help='Directory for profiler outputs')
    args = parser.parse_args()

    set_seed(args.seed)

    h, w = map(int, args.shape.split(','))
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"SMO-Spatial Profiling")
    print(f"Shape: ({h}, {w})  k_ratio={args.k_ratio}  device={device}")
    print(f"Steps: {args.steps}  Warmup: {args.warmup}  Seed: {args.seed}")
    print(f"{'='*60}")

    # Create param and optimizer
    param = nn.Parameter(torch.randn(h, w, dtype=torch.float32, device=device))
    optimizer = SMO([param], lr=1e-3, k_ratio=args.k_ratio)

    if args.use_torch_profiler:
        profile_with_torch_profiler(optimizer, param, steps=args.steps, warmup=args.warmup, device=args.device, output_dir=Path(args.output_dir))
    else:
        profile_step_cpu(optimizer, param, steps=args.steps, warmup=args.warmup)


if __name__ == "__main__":
    main()
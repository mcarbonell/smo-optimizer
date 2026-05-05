"""Isolated optimizer-step microbenchmark for rapid iteration."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks._paths import add_project_root_to_path

add_project_root_to_path()

from benchmarks.timing import measure_steps
from smo import SMO, SMO8bit


# Benchmark classification: family=optimizer_step, category=microbenchmark, status=canonical


def run_optimizer(name, optimizer_cls, shape=(2048, 2048), steps=30, warmup=5, **kwargs):
    torch.manual_seed(1234)
    param = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32))
    optimizer = optimizer_cls([param], lr=1e-3, **kwargs)

    def one_step():
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one_step()

    timing = measure_steps(one_step, steps=steps)
    print(
        f"{name:18} | wall {timing.wall_ms_per_step:8.3f} ms | "
        f"cpu {timing.process_ms_per_step:8.3f} ms | "
        f"cpu/wall {timing.cpu_to_wall_ratio:5.2f}x"
    )


def main():
    print("MICROBENCH: isolated optimizer step")
    print("shape=(2048, 2048), dtype=float32, device=cpu")
    print("wall = elapsed wall-clock, cpu = process CPU time summed across process threads")
    print("cpu/wall > 1.0x is expected when PyTorch uses multiple CPU threads")
    print("-" * 86)
    run_optimizer("AdamW", torch.optim.AdamW)
    run_optimizer("SMO-Spatial", SMO, k_ratio=0.25)
    run_optimizer("SMO-Spatial-8bit", SMO8bit, k_ratio=0.25, block_size=64)


if __name__ == "__main__":
    main()

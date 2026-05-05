"""Isolated optimizer-step microbenchmark for rapid iteration."""

import time

import torch

from smo import SMO, SMO8bit


# Benchmark classification: family=optimizer_step, category=microbenchmark, status=canonical


def run_optimizer(name, optimizer_cls, shape=(2048, 2048), steps=30, warmup=5, **kwargs):
    torch.manual_seed(1234)
    param = torch.nn.Parameter(torch.randn(*shape, dtype=torch.float32))
    optimizer = optimizer_cls([param], lr=1e-3, **kwargs)

    for _ in range(warmup):
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    start = time.perf_counter()
    for _ in range(steps):
        param.grad = torch.randn_like(param)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - start

    print(f"{name:18} | {elapsed / steps * 1000:9.3f} ms/step")


def main():
    print("MICROBENCH: isolated optimizer step")
    print("shape=(2048, 2048), dtype=float32, device=cpu")
    print("-" * 44)
    run_optimizer("AdamW", torch.optim.AdamW)
    run_optimizer("SMO-Spatial", SMO, k_ratio=0.25)
    run_optimizer("SMO-Spatial-8bit", SMO8bit, k_ratio=0.25, block_size=64)


if __name__ == "__main__":
    main()

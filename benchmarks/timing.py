"""Timing helpers for local and accelerator benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TimingResult:
    wall_s: float
    process_s: float
    steps: int

    @property
    def wall_ms_per_step(self) -> float:
        return self.wall_s / self.steps * 1000.0

    @property
    def process_ms_per_step(self) -> float:
        return self.process_s / self.steps * 1000.0

    @property
    def cpu_to_wall_ratio(self) -> float:
        if self.wall_s == 0:
            return 0.0
        return self.process_s / self.wall_s


def measure_steps(
    step_fn: Callable[[], None],
    *,
    steps: int,
    sync_fn: Callable[[], None] | None = None,
) -> TimingResult:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    if sync_fn is not None:
        sync_fn()

    wall_start = time.perf_counter()
    process_start = time.process_time()

    for _ in range(steps):
        step_fn()

    if sync_fn is not None:
        sync_fn()

    wall_elapsed = time.perf_counter() - wall_start
    process_elapsed = time.process_time() - process_start
    return TimingResult(wall_s=wall_elapsed, process_s=process_elapsed, steps=steps)

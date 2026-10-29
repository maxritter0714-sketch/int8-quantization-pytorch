"""Reusable CPU measurement helpers.

On-disk size is the byte length of ``torch.save(module.state_dict())``,
including the zip container that ``torch.save`` writes. Latency is wall time
on CPU after ``torch.set_num_threads``, a warmup, and ``runs`` timed calls.
Median uses :func:`statistics.median`. p90 is the linear-interpolation
percentile (the same method as NumPy's default).
"""

from __future__ import annotations

import csv
import os
import platform
import statistics
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch

# Flags that matter when reading an INT8-vs-FP32 CPU result.
_CPU_FLAGS = (
    "avx",
    "avx2",
    "avx512f",
    "avx512_vnni",
    "avx_vnni",
    "fma",
    "amx_int8",
    "amx_bf16",
)


def cpu_model() -> str:
    """CPU model string from ``/proc/cpuinfo``, or a platform fallback."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def cpu_features() -> str:
    """Subset of ``/proc/cpuinfo`` flags relevant to numeric kernels."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    return " ".join(name for name in _CPU_FLAGS if name in flags)
    except OSError:
        pass
    return ""


def hardware_info() -> dict[str, str]:
    """Interpreter, torch build, and CPU. Every value is a string."""
    return {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count_logical": str(os.cpu_count() or ""),
        "cpu_features": cpu_features(),
    }


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. ``pct`` is in ``[0, 100]``."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if pct < 0 or pct > 100:
        raise ValueError("pct must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


@contextmanager
def cpu_threads(num_threads: int) -> Iterator[None]:
    """Pin intra-op threads for a block, then restore the previous value."""
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@dataclass(frozen=True)
class LatencyResult:
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    runs: int
    warmup: int
    num_threads: int


def measure_cpu_latency(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    runs: int = 100,
    num_threads: int = 1,
) -> LatencyResult:
    """Time ``fn`` on CPU.

    Warmup calls are not included in the statistics. Callers measuring
    inference should wrap this in ``torch.inference_mode`` so the timed
    region does not build an autograd graph.
    """
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if runs < 1:
        raise ValueError("runs must be >= 1")

    samples: list[float] = []
    with cpu_threads(num_threads):
        for _ in range(warmup):
            fn()
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000.0)

    return LatencyResult(
        median_ms=float(statistics.median(samples)),
        p90_ms=percentile(samples, 90),
        min_ms=min(samples),
        max_ms=max(samples),
        runs=runs,
        warmup=warmup,
        num_threads=num_threads,
    )


def state_dict_size_bytes(module: torch.nn.Module) -> int:
    """On-disk size of ``module.state_dict()`` saved with ``torch.save``."""
    with tempfile.TemporaryDirectory(prefix="state-dict-") as directory:
        path = Path(directory) / "weights.pt"
        torch.save(module.state_dict(), path)
        return path.stat().st_size


def state_dict_size_mb(module: torch.nn.Module) -> float:
    return state_dict_size_bytes(module) / (1024.0 * 1024.0)


def write_csv(path: Path | str, rows: list[dict[str, object]]) -> None:
    """Write ``rows`` to CSV. Every row must use the same keys, in order."""
    if not rows:
        raise ValueError("write_csv requires at least one row")
    fieldnames = list(rows[0].keys())
    for index, row in enumerate(rows):
        if list(row.keys()) != fieldnames:
            raise ValueError(f"row {index} keys do not match the first row")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

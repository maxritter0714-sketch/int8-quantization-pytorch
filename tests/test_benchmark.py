"""Smoke tests for size, latency, and CSV helpers. No Hub downloads."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from benchmark import (
    hardware_info,
    measure_cpu_latency,
    percentile,
    state_dict_size_bytes,
    write_csv,
)


def test_percentile_matches_linear_interpolation():
    assert percentile([1, 2, 3, 4, 5], 0) == 1
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)
    assert percentile([7.5], 90) == 7.5


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_warmup_runs_and_restores_threads():
    torch.set_num_threads(1)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        assert torch.get_num_threads() == 2
        return calls["n"]

    result = measure_cpu_latency(fn, warmup=3, runs=4, num_threads=2)
    assert calls["n"] == 7
    assert result.runs == 4
    assert result.warmup == 3
    assert result.num_threads == 2
    assert result.min_ms <= result.median_ms <= result.max_ms
    assert result.min_ms <= result.p90_ms <= result.max_ms
    assert torch.get_num_threads() == 1


def test_latency_rejects_bad_arguments():
    with pytest.raises(ValueError):
        measure_cpu_latency(lambda: None, warmup=-1, runs=1, num_threads=1)
    with pytest.raises(ValueError):
        measure_cpu_latency(lambda: None, warmup=0, runs=0, num_threads=1)
    with pytest.raises(ValueError):
        measure_cpu_latency(lambda: None, warmup=0, runs=1, num_threads=0)


def test_state_dict_size_grows_with_parameters():
    small = nn.Linear(8, 4)
    large = nn.Linear(128, 128)
    assert state_dict_size_bytes(large) > state_dict_size_bytes(small)
    assert state_dict_size_bytes(small) > 0


def test_write_csv_roundtrip(tmp_path: Path):
    destination = tmp_path / "nested" / "metrics.csv"
    rows = [
        {"method": "fp32", "median_ms": 1.5, "batch_size": 1},
        {"method": "dynamic_int8", "median_ms": 1.25, "batch_size": 32},
    ]
    write_csv(destination, rows)
    text = destination.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "method,median_ms,batch_size"
    assert "dynamic_int8,1.25,32" in text


def test_write_csv_rejects_mismatched_keys(tmp_path: Path):
    with pytest.raises(ValueError):
        write_csv(tmp_path / "bad.csv", [{"a": 1}, {"b": 2}])


def test_hardware_info_records_torch_version():
    info = hardware_info()
    assert info["torch_version"] == torch.__version__
    assert info["python_version"]
    assert info["cpu_model"]
    assert info["cpu_count_logical"].isdigit()

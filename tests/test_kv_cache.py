"""KV-cache smoke test on a random tiny GPT-2. No Hub downloads."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from kv_cache import (
    benchmark_sequence,
    kv_dims_from_config,
    measured_kv_cache_bytes,
    plot_kv_cache,
    prefill_past,
    prompt_from_token,
    run_kv_experiment,
    theoretical_kv_cache_bytes,
)


def _tiny_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def test_theoretical_formula_matches_the_documented_product():
    # batch 1: 2 * layers * heads * head_dim * seq_len * bytes
    assert theoretical_kv_cache_bytes(2, 4, 8, 16, 4) == 2 * 2 * 4 * 8 * 16 * 4
    assert theoretical_kv_cache_bytes(2, 4, 8, 16, 4, batch=3) == 3 * 2 * 2 * 4 * 8 * 16 * 4


def test_tiny_gpt2_cache_is_measured_and_generation_runs(tmp_path: Path):
    model = _tiny_model()
    layers, heads, head_dim = kv_dims_from_config(model.config)
    assert (layers, heads, head_dim) == (2, 4, 8)

    seq_len = 16
    prompt = prompt_from_token(seq_len, token_id=3)
    past = prefill_past(model, prompt)
    measured, shapes = measured_kv_cache_bytes(past)
    theoretical = theoretical_kv_cache_bytes(layers, heads, head_dim, seq_len, 4)
    assert measured > 0
    assert shapes
    # Runtimes sometimes allocate a longer cache than the tokens seen so far.
    assert measured >= theoretical // 2
    assert measured <= theoretical * 8

    timed = benchmark_sequence(
        model,
        prompt,
        new_tokens=2,
        use_cache=True,
        warmup=0,
        runs=1,
        num_threads=1,
    )
    assert timed["new_tokens_min"] == 2
    assert timed["median_tokens_per_sec"] > 0

    rows = run_kv_experiment(
        model,
        {8: prompt_from_token(8, token_id=3)},
        new_tokens=2,
        warmup=0,
        runs=1,
        num_threads=1,
        model_name="tiny-random-gpt2",
    )
    assert len(rows) == 2
    assert {row["use_cache"] for row in rows} == {True, False}
    plot_path = tmp_path / "kv.png"
    plot_kv_cache(rows, plot_path, "tiny-random-gpt2")
    assert plot_path.stat().st_size > 0


def test_generation_without_cache_also_returns_tokens():
    model = _tiny_model()
    prompt = prompt_from_token(8, token_id=4)
    with torch.inference_mode():
        cached = benchmark_sequence(
            model,
            prompt,
            new_tokens=2,
            use_cache=True,
            warmup=0,
            runs=1,
            num_threads=1,
        )
        uncached = benchmark_sequence(
            model,
            prompt,
            new_tokens=2,
            use_cache=False,
            warmup=0,
            runs=1,
            num_threads=1,
        )
    assert cached["new_tokens_max"] == uncached["new_tokens_max"] == 2

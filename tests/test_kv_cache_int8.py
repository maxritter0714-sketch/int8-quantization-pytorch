"""INT8 KV-cache round trip on a random tiny GPT-2. No Hub downloads."""

from __future__ import annotations

import math

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from kv_cache import measured_kv_cache_bytes, prefill_past, prompt_from_token
from kv_cache_int8 import (
    apply_int8_roundtrip,
    int8_storage_bytes,
    perplexity,
    quantize_dequantize_per_head,
    quantize_dequantize_per_tensor,
    stepwise_nll,
)


def _tiny_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
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


def test_per_head_roundtrip_error_is_bounded_by_half_a_bin():
    tensor = torch.tensor(
        [[[[1.0, -2.0, 0.25, 0.0], [0.5, 0.5, -0.5, 1.5]]]],
        dtype=torch.float32,
    )
    # shape (1, 1, 2, 4)
    restored, storage = quantize_dequantize_per_head(tensor)
    scale = tensor.abs().amax() / 127.0
    assert restored.shape == tensor.shape
    assert torch.max(torch.abs(restored - tensor)) <= scale * 0.51
    assert storage == tensor.numel() + 4
    per_tensor, per_tensor_storage = quantize_dequantize_per_tensor(tensor)
    assert per_tensor.shape == tensor.shape
    assert per_tensor_storage == tensor.numel() + 4


def test_cache_roundtrip_shrinks_storage_and_stepwise_nll_is_finite():
    model = _tiny_model()
    prompt = prompt_from_token(8, token_id=3)
    past = prefill_past(model, prompt)
    measured, _ = measured_kv_cache_bytes(past)
    assert int8_storage_bytes(past) < measured
    apply_int8_roundtrip(past)
    fp32_nll, fp32_count = stepwise_nll(model, prompt, quantize_cache=False)
    int8_nll, int8_count = stepwise_nll(model, prompt, quantize_cache=True)
    assert fp32_count == int8_count == 7
    assert math.isfinite(fp32_nll) and math.isfinite(int8_nll)
    assert perplexity(fp32_nll, fp32_count) > 1.0

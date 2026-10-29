"""Helpers that the module split made easier to call on their own."""

from __future__ import annotations

import math

import pytest
import torch

from kv_cache_generate import prompt_from_text
from kv_cache_perplexity import iter_windows, perplexity
from kv_cache_roundtrip import apply_int8_roundtrip
from kv_cache_size import config_int, iter_kv_tensors, kv_dims_from_config, measured_kv_cache_bytes
from sst2_eval import synthetic_batch


class _Config:
    n_layer = 2
    n_head = 4
    n_embd = 32


class _Tokenizer:
    eos_token_id = 7

    def encode(self, text, add_special_tokens=False):
        return [3, 4]


def test_config_int_and_kv_dims():
    assert config_int(_Config(), "missing", "n_layer") == 2
    assert kv_dims_from_config(_Config()) == (2, 4, 8)
    with pytest.raises(ValueError):
        config_int(_Config(), "missing")


def test_legacy_tuple_cache_is_measured_and_round_tripped():
    key = torch.tensor([[[[1.0, -2.0]]]], dtype=torch.float32)
    value = torch.tensor([[[[0.5, 0.25]]]], dtype=torch.float32)
    past = ((key, value),)
    tensors = list(iter_kv_tensors(past))
    assert len(tensors) == 2
    measured, shapes = measured_kv_cache_bytes(past)
    assert measured == key.numel() * 4 + value.numel() * 4
    assert "float32" in shapes
    rebuilt, storage = apply_int8_roundtrip(past)
    assert isinstance(rebuilt, tuple)
    assert storage < measured
    assert rebuilt[0][0].shape == key.shape


def test_prompt_from_text_repeats_token_ids_to_the_requested_length():
    prompt = prompt_from_text(_Tokenizer(), 5, "ignored")
    assert prompt.shape == (1, 5)
    assert prompt.tolist() == [[3, 4, 3, 4, 3]]


def test_synthetic_batch_is_deterministic_and_rejects_empty_shapes():
    first = synthetic_batch(2, 4, 50, seed=3)
    second = synthetic_batch(2, 4, 50, seed=3)
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert first["attention_mask"].shape == (2, 4)
    with pytest.raises(ValueError):
        synthetic_batch(0, 4, 50)


def test_iter_windows_drops_a_short_remainder():
    tokens = torch.arange(5)
    windows = iter_windows(tokens, 2)
    assert len(windows) == 2
    assert windows[0].tolist() == [[0, 1]]
    assert windows[1].tolist() == [[2, 3]]
    with pytest.raises(ValueError):
        iter_windows(tokens, 8)


def test_perplexity_is_exp_of_mean_nll():
    assert perplexity(0.0, 1) == 1.0
    assert perplexity(2.0, 2) == pytest.approx(math.exp(1))
    with pytest.raises(ValueError):
        perplexity(1.0, 0)

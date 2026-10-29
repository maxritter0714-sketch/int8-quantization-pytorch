"""Teacher-forced perplexity with an optional INT8 KV-cache round trip."""

from __future__ import annotations

import math

import torch

from kv_cache_roundtrip import apply_int8_roundtrip, int8_storage_bytes
from kv_cache_size import measured_kv_cache_bytes


@torch.no_grad()
def stepwise_nll(model, input_ids: torch.Tensor, *, quantize_cache: bool) -> tuple[float, int]:
    """Sum of negative log-likelihood of tokens ``1..T-1`` given the prefix.

    When ``quantize_cache`` is set, the cache is round-tripped to per-head
    INT8 before every step after the first.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape (1, T) with T >= 2")
    past = None
    total = 0.0
    count = 0
    for index in range(input_ids.shape[1] - 1):
        if quantize_cache and past is not None:
            apply_int8_roundtrip(past, per_head=True)
        output = model(
            input_ids[:, index : index + 1],
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        log_probs = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
        target = int(input_ids[0, index + 1])
        total += float(-log_probs[0, target])
        count += 1
    return total, count


@torch.no_grad()
def oneshot_nll(model, input_ids: torch.Tensor) -> tuple[float, int]:
    """Teacher-forced NLL from one forward. Same token targets as ``stepwise_nll``."""
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape (batch, T) with T >= 2")
    output = model(input_ids, labels=input_ids)
    count = int(input_ids.shape[0] * (input_ids.shape[1] - 1))
    # ``loss`` is the mean over those tokens.
    total = float(output.loss) * count
    return total, count


def perplexity(total_nll: float, count: int) -> float:
    if count < 1:
        raise ValueError("count must be >= 1")
    return math.exp(total_nll / count)


def iter_windows(token_ids: torch.Tensor, seq_len: int) -> list[torch.Tensor]:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    usable = (token_ids.shape[0] // seq_len) * seq_len
    if usable < seq_len:
        raise ValueError("not enough tokens for one window")
    windows = token_ids[:usable].view(-1, seq_len)
    return [windows[index : index + 1].contiguous() for index in range(windows.shape[0])]


def evaluate_windows(
    model, windows: list[torch.Tensor], *, quantize_cache: bool
) -> dict[str, float | int | str]:
    total_nll = 0.0
    total_count = 0
    oneshot_total = 0.0
    oneshot_count = 0
    fp32_bytes = 0
    int8_bytes = 0
    shapes = ""
    for window in windows:
        nll, count = stepwise_nll(model, window, quantize_cache=quantize_cache)
        total_nll += nll
        total_count += count
        if not quantize_cache:
            shot_nll, shot_count = oneshot_nll(model, window)
            oneshot_total += shot_nll
            oneshot_count += shot_count
        past = model(window, use_cache=True).past_key_values
        fp32_bytes, shapes = measured_kv_cache_bytes(past)
        int8_bytes = int8_storage_bytes(past, per_head=True)
    return {
        "nll": total_nll,
        "tokens": total_count,
        "perplexity": perplexity(total_nll, total_count),
        "oneshot_nll": oneshot_total,
        "oneshot_tokens": oneshot_count,
        "oneshot_perplexity": perplexity(oneshot_total, oneshot_count) if oneshot_count else "",
        "fp32_cache_bytes": fp32_bytes,
        "int8_storage_bytes": int8_bytes,
        "cache_shapes": shapes,
    }

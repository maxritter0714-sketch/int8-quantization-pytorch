"""Prompt construction and timed greedy generation for the KV-cache experiment."""

from __future__ import annotations

import statistics
import time

import torch

from benchmark import cpu_threads, percentile


def prompt_from_token(seq_len: int, token_id: int) -> torch.Tensor:
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    return torch.full((1, seq_len), int(token_id), dtype=torch.long)


def prompt_from_text(tokenizer, seq_len: int, text: str) -> torch.Tensor:
    piece = tokenizer.encode(text, add_special_tokens=False)
    if not piece:
        eos = tokenizer.eos_token_id
        piece = [0 if eos is None else int(eos)]
    token_ids = (piece * (seq_len // len(piece) + 1))[:seq_len]
    if len(token_ids) != seq_len:
        raise RuntimeError("failed to build a prompt of the requested length")
    return torch.tensor([token_ids], dtype=torch.long)


def _as_token_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    sequences = getattr(output, "sequences", None)
    if torch.is_tensor(sequences):
        return sequences
    raise TypeError(f"generate returned {type(output)!r}, expected token ids")


def generate_once(
    model,
    input_ids: torch.Tensor,
    new_tokens: int,
    use_cache: bool,
) -> torch.Tensor:
    """Greedy generation of exactly ``new_tokens`` new tokens when the model allows it."""
    if new_tokens < 1:
        raise ValueError("new_tokens must be >= 1")
    pad = getattr(model.config, "pad_token_id", None)
    if pad is None:
        pad = getattr(model.config, "eos_token_id", None)
    if pad is None:
        pad = 0
    kwargs = {
        "max_new_tokens": new_tokens,
        "min_new_tokens": new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "pad_token_id": int(pad),
    }
    try:
        output = model.generate(input_ids, **kwargs)
    except TypeError:
        kwargs.pop("min_new_tokens")
        output = model.generate(input_ids, **kwargs)
    return _as_token_tensor(output)


@torch.inference_mode()
def prefill_past(model, input_ids: torch.Tensor):
    output = model(input_ids=input_ids, use_cache=True)
    past = getattr(output, "past_key_values", None)
    if past is None:
        raise RuntimeError("model forward did not return past_key_values")
    return past


def parameter_element_size(model: torch.nn.Module) -> int:
    for parameter in model.parameters():
        return int(parameter.element_size())
    raise ValueError("model has no parameters")


def benchmark_sequence(
    model,
    input_ids: torch.Tensor,
    *,
    new_tokens: int,
    use_cache: bool,
    warmup: int,
    runs: int,
    num_threads: int,
) -> dict[str, float | int]:
    """Warm up, then time ``runs`` generations. Returns latency and tokens/sec."""
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be >= 0 and runs must be >= 1")
    samples_ms: list[float] = []
    produced: list[int] = []
    prompt_len = int(input_ids.shape[-1])
    with cpu_threads(num_threads), torch.inference_mode():
        for _ in range(warmup):
            generate_once(model, input_ids, new_tokens, use_cache)
        for _ in range(runs):
            start = time.perf_counter()
            sequences = generate_once(model, input_ids, new_tokens, use_cache)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            new_count = int(sequences.shape[-1] - prompt_len)
            if new_count < 1:
                raise RuntimeError("generation produced no new tokens")
            samples_ms.append(elapsed_ms)
            produced.append(new_count)
    rates = [
        count / (elapsed / 1000.0) for count, elapsed in zip(produced, samples_ms, strict=True)
    ]
    return {
        "median_ms": round(float(statistics.median(samples_ms)), 4),
        "p90_ms": round(percentile(samples_ms, 90), 4),
        "min_ms": round(min(samples_ms), 4),
        "max_ms": round(max(samples_ms), 4),
        "median_tokens_per_sec": round(float(statistics.median(rates)), 4),
        "new_tokens_min": min(produced),
        "new_tokens_max": max(produced),
    }

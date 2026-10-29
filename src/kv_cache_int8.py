"""Experiment: round a GPT-2 KV cache to INT8 and measure perplexity.

This is not a production cache. After each cached decode step the key and
value tensors are replaced by a symmetric per-head INT8 round trip
(scale = max(abs) / 127 over sequence and head dimension, integers in
[-128, 127], then dequantized back to FP32 so the next attention kernel can
run). The model weights stay FP32.

Perplexity is teacher-forced on a prefix of WikiText-2 raw validation,
cut into windows. The FP32 and INT8 numbers use the same stepwise loop.
``int8_storage_bytes`` is the int8 payload plus one FP32 scale per head,
not the size of the dequantized tensor the runtime still holds.

Labeled as an experiment: the slice is small, and a round trip is not the
same as an integer attention kernel.
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark import hardware_info, write_csv
from kv_cache import measured_kv_cache_bytes
from kv_cache_roundtrip import (
    apply_int8_roundtrip,
    int8_storage_bytes,
    quantize_dequantize_per_head,
    quantize_dequantize_per_tensor,
)

LOG = logging.getLogger("kv_cache_int8")

_DATASET_CANDIDATES = (
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext", "wikitext-2-raw-v1"),
)

__all__ = [
    "apply_int8_roundtrip",
    "int8_storage_bytes",
    "quantize_dequantize_per_head",
    "quantize_dequantize_per_tensor",
]


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


def load_wikitext_prefix(tokenizer, max_tokens: int) -> tuple[torch.Tensor, str]:
    """First ``max_tokens`` tokens of WikiText-2 raw validation, skipping blank lines."""
    from datasets import load_dataset

    errors: list[str] = []
    dataset = None
    source = ""
    for name, config in _DATASET_CANDIDATES:
        try:
            dataset = load_dataset(name, config, split="validation")
            source = f"{name}/{config}"
            break
        except Exception as exc:
            errors.append(f"{name}/{config}: {exc}")
    if dataset is None:
        raise RuntimeError("Could not load WikiText-2 validation. " + " | ".join(errors))
    pieces: list[int] = []
    for row in dataset:
        line = str(row["text"]).strip()
        if not line:
            continue
        prefix = "\n" if pieces else ""
        pieces.extend(tokenizer.encode(prefix + line, add_special_tokens=False))
        if len(pieces) >= max_tokens:
            break
    if len(pieces) < max_tokens:
        raise RuntimeError(
            f"WikiText-2 validation tokenized to {len(pieces)} tokens, need {max_tokens}"
        )
    return torch.tensor(pieces[:max_tokens], dtype=torch.long), source


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


def plot_int8_cache(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    perplexities = []
    fp32_mb = []
    int8_mb = []
    for row in rows:
        labels.append(
            "FP32 cache" if row["quantize_cache"] == "none" else "Per-head INT8 round trip"
        )
        perplexities.append(float(row["perplexity"]))
        fp32_mb.append(int(row["fp32_cache_bytes"]) / (1024.0**2))
        int8_mb.append(int(row["int8_storage_bytes"]) / (1024.0**2))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].bar(labels, perplexities, color=["#4c78a8", "#f58518"])
    axes[0].set_ylabel("Perplexity")
    axes[0].set_title("WikiText-2 slice")
    axes[0].grid(True, axis="y", alpha=0.3)
    stored_mb = [fp32_mb[0], int8_mb[-1]]
    axes[1].bar(labels, stored_mb, color=["#4c78a8", "#f58518"])
    axes[1].set_ylabel("Cache storage (MB)")
    axes[1].set_title("FP32 tensor vs INT8 + scales")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("KV-cache INT8 round trip (experiment)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--seq-len", type=_positive_int, default=128)
    parser.add_argument("--max-tokens", type=_positive_int, default=512)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/kv_cache_int8.csv"))
    parser.add_argument("--plot", type=Path, default=Path("results/kv_cache_int8.png"))
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.max_tokens < args.seq_len:
        raise SystemExit("--max-tokens must be >= --seq-len")
    torch.set_num_threads(args.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOG.info("loading %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    model.to("cpu")
    token_ids, source = load_wikitext_prefix(tokenizer, args.max_tokens)
    windows = iter_windows(token_ids, args.seq_len)
    LOG.info("dataset=%s windows=%s seq_len=%s", source, len(windows), args.seq_len)
    hardware = hardware_info()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for quantize_cache, label in ((False, "none"), (True, "per_head_int8")):
        LOG.info("evaluating quantize_cache=%s", label)
        stats = evaluate_windows(model, windows, quantize_cache=quantize_cache)
        LOG.info(
            "%s perplexity=%.4f fp32_bytes=%s int8_bytes=%s",
            label,
            stats["perplexity"],
            stats["fp32_cache_bytes"],
            stats["int8_storage_bytes"],
        )
        rows.append(
            {
                "timestamp_utc": timestamp,
                "experiment": "kv_cache_int8_roundtrip",
                "model": args.model,
                "dataset": source,
                "quantize_cache": label,
                "scale": "per_head_symmetric" if quantize_cache else "none",
                "seq_len": args.seq_len,
                "windows": len(windows),
                "predicted_tokens": stats["tokens"],
                "perplexity": round(float(stats["perplexity"]), 4),
                "oneshot_perplexity": (
                    round(float(stats["oneshot_perplexity"]), 4)
                    if stats["oneshot_perplexity"] != ""
                    else ""
                ),
                "fp32_cache_bytes": stats["fp32_cache_bytes"],
                "int8_storage_bytes": stats["int8_storage_bytes"],
                "cache_measured_at": "full_window_prefill",
                "cache_shapes": stats["cache_shapes"],
                "num_threads": args.threads,
                "note": (
                    "Symmetric per-head INT8 round trip of past_key_values before each "
                    "cached step. int8_storage_bytes counts the int8 payload plus FP32 "
                    "scales. The runtime tensor stays FP32. Slice is a prefix of "
                    "WikiText-2 raw validation, not the full set."
                ),
                **hardware,
            }
        )
    write_csv(args.output, rows)
    plot_int8_cache(rows, args.plot)
    LOG.info("wrote %s", args.output)
    LOG.info("wrote %s", args.plot)


if __name__ == "__main__":
    main()

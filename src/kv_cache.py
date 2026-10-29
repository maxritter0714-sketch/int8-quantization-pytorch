"""KV-cache size and decode throughput for a small GPT-2.

For each prompt length, generate a fixed number of new tokens with
``use_cache=True`` and ``use_cache=False``. Tokens/sec is the median of
per-run rates (new tokens divided by wall time).

Theoretical cache size, batch 1, one value's byte width:

    2 * layers * kv_heads * head_dim * seq_len * bytes_per_value

``kv_heads`` is ``num_key_value_heads`` when the config has it, otherwise
``n_head``. DistilGPT-2 and GPT-2 are multi-head, so the two match.
Measured size walks the key and value tensors in ``past_key_values`` after
a prefill of exactly ``seq_len`` tokens (the extra generated tokens are not
included). A mismatch means the runtime stored a different shape than the
formula, which the CSV ``cache_shapes`` column records.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark import hardware_info, write_csv
from kv_cache_generate import (
    benchmark_sequence,
    parameter_element_size,
    prefill_past,
    prompt_from_text,
    prompt_from_token,
)
from kv_cache_size import (
    iter_kv_tensors,
    kv_dims_from_config,
    measured_kv_cache_bytes,
    theoretical_kv_cache_bytes,
)

LOG = logging.getLogger("kv_cache")

DEFAULT_MODEL = "distilgpt2"

# Re-exported so existing imports from this module keep working.
__all__ = [
    "benchmark_sequence",
    "iter_kv_tensors",
    "kv_dims_from_config",
    "measured_kv_cache_bytes",
    "prefill_past",
    "prompt_from_token",
    "theoretical_kv_cache_bytes",
]


def plot_kv_cache(rows: list[dict[str, object]], path: Path, model_name: str) -> None:
    """One figure: decode tokens/sec, and theoretical vs measured cache size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = sorted({int(row["seq_len"]) for row in rows})

    def tokens_per_sec(use_cache: bool) -> list[float]:
        by_len = {
            int(row["seq_len"]): float(row["median_tokens_per_sec"])
            for row in rows
            if bool(row["use_cache"]) is use_cache
        }
        return [by_len[seq] for seq in seqs]

    by_len = {int(row["seq_len"]): row for row in rows}
    theoretical_mb = [int(by_len[seq]["theoretical_kv_bytes"]) / (1024.0**2) for seq in seqs]
    measured_mb = [int(by_len[seq]["measured_kv_bytes"]) / (1024.0**2) for seq in seqs]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].plot(seqs, tokens_per_sec(True), marker="o", label="use_cache=True")
    axes[0].plot(seqs, tokens_per_sec(False), marker="o", label="use_cache=False")
    axes[0].set_xlabel("Prompt length (tokens)")
    axes[0].set_ylabel("Median tokens/sec")
    axes[0].set_title("Greedy decode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(seqs, theoretical_mb, marker="o", label="Theoretical")
    axes[1].plot(seqs, measured_mb, marker="s", label="Measured past_key_values")
    axes[1].set_xlabel("Sequence length (tokens)")
    axes[1].set_ylabel("KV cache (MB)")
    axes[1].set_title("Cache size at prefill")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(model_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_kv_experiment(
    model,
    prompts: dict[int, torch.Tensor],
    *,
    new_tokens: int,
    warmup: int,
    runs: int,
    num_threads: int,
    model_name: str,
) -> list[dict[str, object]]:
    layers, kv_heads, head_dim = kv_dims_from_config(model.config)
    bytes_per_value = parameter_element_size(model)
    hardware = hardware_info()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, object]] = []

    for seq_len in sorted(prompts):
        input_ids = prompts[seq_len]
        if int(input_ids.shape[-1]) != seq_len or int(input_ids.shape[0]) != 1:
            raise ValueError(f"prompt for seq_len={seq_len} must have shape (1, {seq_len})")
        past = prefill_past(model, input_ids)
        measured_bytes, shapes = measured_kv_cache_bytes(past)
        if measured_bytes <= 0:
            raise RuntimeError(f"measured KV cache at seq_len={seq_len} was empty")
        theoretical = theoretical_kv_cache_bytes(
            layers,
            kv_heads,
            head_dim,
            seq_len,
            bytes_per_value,
            batch=1,
        )
        LOG.info(
            "seq_len=%s theoretical_bytes=%s measured_bytes=%s shapes=%s",
            seq_len,
            theoretical,
            measured_bytes,
            shapes,
        )
        for use_cache in (True, False):
            LOG.info(
                "generate seq_len=%s use_cache=%s new_tokens=%s runs=%s",
                seq_len,
                use_cache,
                new_tokens,
                runs,
            )
            timed = benchmark_sequence(
                model,
                input_ids,
                new_tokens=new_tokens,
                use_cache=use_cache,
                warmup=warmup,
                runs=runs,
                num_threads=num_threads,
            )
            LOG.info(
                "seq_len=%s use_cache=%s median_tokens_per_sec=%s median_ms=%s",
                seq_len,
                use_cache,
                timed["median_tokens_per_sec"],
                timed["median_ms"],
            )
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "model": model_name,
                    "seq_len": seq_len,
                    "use_cache": use_cache,
                    "requested_new_tokens": new_tokens,
                    "new_tokens_min": timed["new_tokens_min"],
                    "new_tokens_max": timed["new_tokens_max"],
                    "warmup": warmup,
                    "runs": runs,
                    "num_threads": num_threads,
                    "median_ms": timed["median_ms"],
                    "p90_ms": timed["p90_ms"],
                    "min_ms": timed["min_ms"],
                    "max_ms": timed["max_ms"],
                    "median_tokens_per_sec": timed["median_tokens_per_sec"],
                    "theoretical_kv_bytes": theoretical,
                    "measured_kv_bytes": measured_bytes,
                    "bytes_per_value": bytes_per_value,
                    "n_layer": layers,
                    "n_kv_heads": kv_heads,
                    "head_dim": head_dim,
                    "cache_shapes": shapes,
                    "cache_measured_at": "prefill",
                    **hardware,
                }
            )
    return rows


def _load_causal_lm(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seq-lens", type=_positive_int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--new-tokens", type=_positive_int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=_positive_int, default=10)
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument(
        "--prompt",
        default="Quantization reduces the cost of neural network inference. ",
    )
    parser.add_argument("--output", type=Path, default=Path("results/kv_cache.csv"))
    parser.add_argument("--plot", type=Path, default=Path("results/kv_cache.png"))
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    LOG.info("loading %s", args.model)
    tokenizer, model = _load_causal_lm(args.model)
    prompts = {
        seq_len: prompt_from_text(tokenizer, seq_len, args.prompt) for seq_len in args.seq_lens
    }
    rows = run_kv_experiment(
        model,
        prompts,
        new_tokens=args.new_tokens,
        warmup=args.warmup,
        runs=args.runs,
        num_threads=args.threads,
        model_name=args.model,
    )
    write_csv(args.output, rows)
    plot_kv_cache(rows, args.plot, args.model)
    LOG.info("wrote %s", args.output)
    LOG.info("wrote %s", args.plot)


if __name__ == "__main__":
    main()

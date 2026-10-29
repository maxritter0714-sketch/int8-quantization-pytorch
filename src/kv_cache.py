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
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark import cpu_threads, hardware_info, percentile, write_csv

LOG = logging.getLogger("kv_cache")

DEFAULT_MODEL = "distilgpt2"


def theoretical_kv_cache_bytes(
    layers: int,
    heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_value: int,
    batch: int = 1,
) -> int:
    """Bytes in a dense KV cache: ``batch * 2 * layers * heads * head_dim * seq_len * bytes``."""
    values = {
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "bytes_per_value": bytes_per_value,
        "batch": batch,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int")
    if batch < 1 or seq_len < 1 or bytes_per_value < 1:
        raise ValueError("batch, seq_len, and bytes_per_value must be >= 1")
    return batch * 2 * layers * heads * head_dim * seq_len * bytes_per_value


def kv_dims_from_config(config) -> tuple[int, int, int]:
    """Return ``(layers, kv_heads, head_dim)`` for a GPT-2-style config."""
    layers = _config_int(config, "n_layer", "num_hidden_layers")
    query_heads = _config_int(config, "n_head", "num_attention_heads")
    hidden = _config_int(config, "n_embd", "hidden_size")
    kv_heads = _config_int(config, "num_key_value_heads", default=query_heads)
    if layers < 1 or query_heads < 1 or kv_heads < 1 or hidden < 1:
        raise ValueError("config dimensions must be positive")
    if hidden % query_heads != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by query heads {query_heads}")
    return layers, kv_heads, hidden // query_heads


def _config_int(config, *names: str, default: int | None = None) -> int:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value
    if default is not None:
        return default
    raise ValueError(f"config is missing an int field among {names}")


def iter_kv_tensors(past_key_values):
    """Yield key and value tensors from a legacy tuple or a Cache object."""
    if past_key_values is None:
        return

    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                for tensor in layer:
                    if torch.is_tensor(tensor):
                        yield tensor
            elif torch.is_tensor(layer):
                yield layer
        return

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if (
        isinstance(key_cache, (list, tuple))
        and key_cache
        and all(tensor is None or torch.is_tensor(tensor) for tensor in key_cache)
    ):
        for tensor in list(key_cache) + list(value_cache or []):
            if torch.is_tensor(tensor):
                yield tensor
        return

    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        found = False
        for layer in layers:
            for attr in ("keys", "values", "key", "value"):
                tensor = getattr(layer, attr, None)
                if torch.is_tensor(tensor):
                    found = True
                    yield tensor
        if found:
            return

    raise TypeError(
        "Unrecognized past_key_values type: "
        f"{type(past_key_values)!r}. Cannot measure the KV cache."
    )


def measured_kv_cache_bytes(past_key_values) -> tuple[int, str]:
    """Logical bytes of unique key/value tensors, plus a shape summary."""
    tensors = []
    seen: set[tuple[int, tuple[int, ...], str]] = set()
    for tensor in iter_kv_tensors(past_key_values):
        identity = (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
        if identity in seen:
            continue
        seen.add(identity)
        tensors.append(tensor)
    total = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    shapes = "|".join(
        f"{tuple(tensor.shape)}:{str(tensor.dtype).replace('torch.', '')}" for tensor in tensors
    )
    return total, shapes


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

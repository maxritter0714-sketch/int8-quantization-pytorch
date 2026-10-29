"""FP32 vs dynamic INT8 for DistilBERT SST-2.

Dynamic PTQ uses ``torch.ao.quantization.quantize_dynamic`` on ``nn.Linear``
only. Embeddings and layer norms stay FP32, so the on-disk model shrinks by
less than 4x. Latency is measured on synthetic token ids of a fixed length
(batch 1 and 32 by default), not on the SST-2 sentences. Accuracy is argmax accuracy on the GLUE SST-2 validation split from
``nyu-mll/glue`` (the bare ``glue`` id is no longer a valid Hub repo id).

``--limit`` keeps the first N labeled validation rows. That prefix is
comparable across runs and is not a random sample of the full split.
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

from benchmark import (
    hardware_info,
    measure_cpu_latency,
    state_dict_size_mb,
    write_csv,
)

LOG = logging.getLogger("quantize_dynamic")

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# GLUE now lives under this namespace. The bare "glue" id is rejected by
# current huggingface_hub ("Repository id must be namespace/name").
DEFAULT_DATASET = "nyu-mll/glue"
_ENGINE_PREFERENCE = ("fbgemm", "x86", "onednn", "qnnpack")


def configure_quantized_engine() -> str:
    """Pick a CPU quantized engine and make it the process default."""
    supported = list(torch.backends.quantized.supported_engines)
    for name in _ENGINE_PREFERENCE:
        if name in supported:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(
        f"No CPU quantized engine is available for dynamic INT8. supported_engines={supported}."
    )


def quantize_linear_dynamic(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Quantize ``nn.Linear`` modules to dynamic INT8.

    With ``inplace=False`` (the default) the input module is copied first.
    The eager ``quantize_dynamic`` API is deprecated upstream in favor of
    torchao; this project calls it because that is the comparison we report.
    """
    from torch.ao.quantization import quantize_dynamic

    configure_quantized_engine()
    target = model if inplace else copy.deepcopy(model)
    target.eval()
    quantized = quantize_dynamic(
        target,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )
    quantized.eval()
    return quantized


def count_linears(model: nn.Module) -> tuple[int, int]:
    """Return ``(fp32_linear, dynamic_quantized_linear)`` module counts."""
    fp32 = 0
    quantized = 0
    for module in model.modules():
        if type(module).__name__ != "Linear":
            continue
        if "quantized" in type(module).__module__:
            quantized += 1
        elif isinstance(module, nn.Linear):
            fp32 += 1
    return fp32, quantized


def classification_accuracy(
    model: nn.Module,
    tokenizer,
    rows,
    *,
    batch_size: int,
    max_length: int,
    text_key: str = "sentence",
    label_key: str = "label",
) -> tuple[float, int, int]:
    """Argmax accuracy on labeled rows.

    Returns ``(accuracy, correct, seen)``. ``rows`` may be a list of dicts
    or a dataset that slices into a column dict (Hugging Face ``datasets``).
    Labels ``< 0`` are skipped.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    total = len(rows)
    correct = 0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            if start == 0 or start % (batch_size * 10) == 0:
                LOG.info("accuracy progress %s/%s", start, total)
            batch = rows[start : start + batch_size]
            if isinstance(batch, dict):
                texts = list(batch[text_key])
                labels = [int(label) for label in batch[label_key]]
            else:
                texts = [example[text_key] for example in batch]
                labels = [int(example[label_key]) for example in batch]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            predictions = logits.argmax(dim=-1).tolist()
            for prediction, label in zip(predictions, labels):
                if label < 0:
                    continue
                seen += 1
                correct += int(prediction == label)
    if seen == 0:
        raise ValueError("no labeled examples to score")
    return correct / seen, correct, seen


def synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Fixed synthetic inputs so FP32 and INT8 latency see the same tokens."""
    if batch_size < 1 or seq_len < 1 or vocab_size < 1:
        raise ValueError("batch_size, seq_len, and vocab_size must be >= 1")
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
    )
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _load_classifier(model_name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to("cpu")
    return tokenizer, model


def _load_split(dataset_name: str, split: str, limit: int | None):
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, "sst2", split=split)
    if limit is None:
        return dataset, "full"
    if limit < 1:
        raise ValueError("--limit must be >= 1")
    kept = min(limit, len(dataset))
    if kept < limit:
        LOG.warning("requested limit %s but split only has %s rows", limit, kept)
    return dataset.select(range(kept)), str(limit)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Score only the first N validation rows. Omit for the full split.",
    )
    parser.add_argument("--batch-sizes", type=_positive_int, nargs="+", default=[1, 32])
    parser.add_argument("--seq-len", type=_positive_int, default=128)
    parser.add_argument("--accuracy-batch-size", type=_positive_int, default=16)
    parser.add_argument("--max-length", type=_positive_int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=_positive_int, default=50)
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/distilbert_sst2_dynamic_int8.csv"),
    )
    return parser


def _measure_model(
    model: nn.Module,
    batches: dict[int, dict[str, torch.Tensor]],
    *,
    warmup: int,
    runs: int,
    num_threads: int,
) -> dict[int, object]:
    results = {}
    with torch.inference_mode():
        for batch_size, inputs in batches.items():
            LOG.info("latency batch_size=%s warmup=%s runs=%s", batch_size, warmup, runs)
            results[batch_size] = measure_cpu_latency(
                lambda inputs=inputs: model(**inputs),
                warmup=warmup,
                runs=runs,
                num_threads=num_threads,
            )
    return results


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    engine = configure_quantized_engine()
    LOG.info("quantized engine: %s", engine)
    LOG.info("loading %s", args.model)
    tokenizer, fp32 = _load_classifier(args.model)
    dataset, limit_label = _load_split(args.dataset, args.split, args.limit)
    LOG.info("accuracy split=%s rows=%s limit=%s", args.split, len(dataset), limit_label)

    vocab_size = int(tokenizer.vocab_size)
    batches = {
        batch_size: synthetic_batch(batch_size, args.seq_len, vocab_size)
        for batch_size in args.batch_sizes
    }
    hardware = hardware_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, object]] = []

    def record(model: nn.Module, method: str, accuracy: float, correct: int, seen: int) -> None:
        fp32_linears, quantized_linears = count_linears(model)
        size_mb = round(state_dict_size_mb(model), 4)
        LOG.info(
            "%s size_mb=%s accuracy=%.4f (%s/%s) linears fp32=%s int8=%s",
            method,
            size_mb,
            accuracy,
            correct,
            seen,
            fp32_linears,
            quantized_linears,
        )
        latency = _measure_model(
            model,
            batches,
            warmup=args.warmup,
            runs=args.runs,
            num_threads=args.threads,
        )
        for batch_size in args.batch_sizes:
            result = latency[batch_size]
            LOG.info(
                "%s batch=%s median_ms=%.3f p90_ms=%.3f",
                method,
                batch_size,
                result.median_ms,
                result.p90_ms,
            )
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "model": args.model,
                    "dataset": args.dataset,
                    "method": method,
                    "quant_engine": engine,
                    "fp32_linears": fp32_linears,
                    "quantized_linears": quantized_linears,
                    "size_mb": size_mb,
                    "accuracy": round(accuracy, 6),
                    "accuracy_correct": correct,
                    "accuracy_samples": seen,
                    "split": args.split,
                    "requested_limit": limit_label,
                    "batch_size": batch_size,
                    "seq_len": args.seq_len,
                    "latency_input": "synthetic_token_ids",
                    "warmup": result.warmup,
                    "runs": result.runs,
                    "num_threads": result.num_threads,
                    "median_ms": round(result.median_ms, 4),
                    "p90_ms": round(result.p90_ms, 4),
                    "min_ms": round(result.min_ms, 4),
                    "max_ms": round(result.max_ms, 4),
                    **hardware,
                }
            )

    LOG.info("scoring and timing FP32")
    fp32_accuracy, fp32_correct, fp32_seen = classification_accuracy(
        fp32,
        tokenizer,
        dataset,
        batch_size=args.accuracy_batch_size,
        max_length=args.max_length,
    )
    record(fp32, "fp32", fp32_accuracy, fp32_correct, fp32_seen)
    del fp32
    gc.collect()

    LOG.info("quantizing a fresh FP32 copy (dynamic INT8, nn.Linear)")
    _, fp32_for_quant = _load_classifier(args.model)
    int8 = quantize_linear_dynamic(fp32_for_quant, inplace=True)
    del fp32_for_quant
    gc.collect()

    LOG.info("scoring and timing dynamic INT8")
    int8_accuracy, int8_correct, int8_seen = classification_accuracy(
        int8,
        tokenizer,
        dataset,
        batch_size=args.accuracy_batch_size,
        max_length=args.max_length,
    )
    record(int8, "dynamic_int8", int8_accuracy, int8_correct, int8_seen)

    write_csv(args.output, rows)
    LOG.info("wrote %s", args.output)


if __name__ == "__main__":
    main()

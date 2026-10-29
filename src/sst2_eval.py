"""SST-2 accuracy and the synthetic latency batches used by dynamic INT8."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

# Same logger as the DistilBERT script so progress lines stay unchanged.
LOG = logging.getLogger("quantize_dynamic")


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
            for prediction, label in zip(predictions, labels, strict=True):
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

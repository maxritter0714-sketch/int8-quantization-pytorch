"""Dynamic INT8 smoke test on a tiny local module. No Hub downloads."""

from __future__ import annotations

import torch
import torch.nn as nn

from benchmark import state_dict_size_bytes
from quantize_dynamic import (
    classification_accuracy,
    configure_quantized_engine,
    count_linears,
    quantize_linear_dynamic,
)


class _Tower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _AlwaysPositive(nn.Module):
    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        logits = torch.zeros(batch, 2)
        logits[:, 1] = 1
        return type("Output", (), {"logits": logits})()


class _FakeTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        batch = len(texts)
        return {
            "input_ids": torch.ones(batch, 4, dtype=torch.long),
            "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        }


class _ColumnDataset:
    """Mimics a Hugging Face dataset slice: ``ds[i:j]`` is a column dict."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, slice):
            chunk = self.rows[item]
            return {
                "sentence": [row["sentence"] for row in chunk],
                "label": [row["label"] for row in chunk],
            }
        return self.rows[item]


def test_dynamic_int8_shrinks_linear_weights_and_still_runs():
    engine = configure_quantized_engine()
    assert engine
    fp32 = _Tower().eval()
    int8 = quantize_linear_dynamic(fp32)
    fp32_linears, fp32_quantized = count_linears(fp32)
    int8_linears, int8_quantized = count_linears(int8)
    assert (fp32_linears, fp32_quantized) == (2, 0)
    assert int8_quantized == 2
    assert int8_linears == 0
    assert state_dict_size_bytes(int8) < state_dict_size_bytes(fp32)
    with torch.inference_mode():
        output = int8(torch.randn(2, 256))
    assert output.shape == (2, 64)


def test_classification_accuracy_skips_unlabeled_rows():
    rows = [
        {"sentence": "good", "label": 1},
        {"sentence": "bad", "label": 0},
        {"sentence": "mystery", "label": -1},
        {"sentence": "also good", "label": 1},
    ]
    accuracy, correct, seen = classification_accuracy(
        _AlwaysPositive(),
        _FakeTokenizer(),
        rows,
        batch_size=2,
        max_length=16,
    )
    assert seen == 3
    assert correct == 2
    assert accuracy == 2 / 3

    column_accuracy, column_correct, column_seen = classification_accuracy(
        _AlwaysPositive(),
        _FakeTokenizer(),
        _ColumnDataset(rows),
        batch_size=3,
        max_length=16,
    )
    assert (column_accuracy, column_correct, column_seen) == (accuracy, correct, seen)

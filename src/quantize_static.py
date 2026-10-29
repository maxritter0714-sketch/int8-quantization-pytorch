"""Static PTQ for a trained SmallCNN checkpoint.

Loads FP32 weights, calibrates ``prepare_fx`` on a prefix of the CIFAR-10
training set, and ``convert_fx`` to INT8. Prints size and test accuracy.
The full FP32 vs static vs QAT table is produced by ``src/train_cnn.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from cifar10 import load_cifar10
from cnn_quantize import classification_accuracy, configure_static_backend, load_fp32, static_ptq
from benchmark import state_dict_size_mb

LOG = logging.getLogger("quantize_static")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cnn_fp32.pt"))
    parser.add_argument("--calib-samples", type=_positive_int, default=256)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--backend", default="x86", choices=("x86", "fbgemm"))
    parser.add_argument("--threads", type=_positive_int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}. Train with src/train_cnn.py first.")
    torch.set_num_threads(args.threads)
    backend = configure_static_backend(args.backend)
    LOG.info("backend=%s checkpoint=%s", backend, args.checkpoint)
    train_x, _, test_x, test_y = load_cifar10()
    model = load_fp32(args.checkpoint)
    quantized = static_ptq(
        model,
        train_x[: args.calib_samples],
        backend=backend,
        batch_size=args.batch_size,
    )
    accuracy, correct, seen = classification_accuracy(quantized, test_x, test_y)
    LOG.info(
        "static_ptq size_mb=%.4f accuracy=%.4f (%s/%s) fp32_size_mb=%.4f",
        state_dict_size_mb(quantized),
        accuracy,
        correct,
        seen,
        state_dict_size_mb(model),
    )


if __name__ == "__main__":
    main()

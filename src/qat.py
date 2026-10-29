"""QAT fine-tune for a trained SmallCNN checkpoint.

``prepare_qat_fx`` inserts fake-quant observers, the copy is fine-tuned for
``--epochs`` passes over CIFAR-10 train, then ``convert_fx`` produces INT8.
The full comparison table is produced by ``src/train_cnn.py``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from benchmark import state_dict_size_mb
from cifar10 import load_cifar10
from cnn_quantize import classification_accuracy, configure_static_backend, load_fp32, qat_finetune

LOG = logging.getLogger("qat")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cnn_fp32.pt"))
    parser.add_argument("--epochs", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", default="x86", choices=("x86", "fbgemm"))
    parser.add_argument("--threads", type=_positive_int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}. Train with src/train_cnn.py first.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    backend = configure_static_backend(args.backend)
    LOG.info("backend=%s checkpoint=%s", backend, args.checkpoint)
    train_x, train_y, test_x, test_y = load_cifar10()
    model = load_fp32(args.checkpoint)
    quantized = qat_finetune(
        model,
        train_x,
        train_y,
        backend=backend,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    accuracy, correct, seen = classification_accuracy(quantized, test_x, test_y)
    LOG.info(
        "qat size_mb=%.4f accuracy=%.4f (%s/%s)",
        state_dict_size_mb(quantized),
        accuracy,
        correct,
        seen,
    )


if __name__ == "__main__":
    main()

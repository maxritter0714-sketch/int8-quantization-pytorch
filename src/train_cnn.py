"""Train SmallCNN on CIFAR-10, then measure FP32, static PTQ, and QAT.

One command writes ``results/cnn_quantization.csv`` and
``results/cnn_quantization.png``. The FP32 state dict is saved next to them
(gitignored) so the static and QAT entry points can reload it.

Static PTQ calibrates on the first ``--calib-samples`` training images
(a prefix, 100 to 500 is the intended range). QAT fine-tunes a copy of the
trained FP32 weights. Test accuracy is the full 10,000-image test split.
Latency uses the first 1 and 32 test images.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from cifar10 import load_cifar10
from cnn_quantize import (
    build_rows,
    configure_static_backend,
    plot_cnn,
    qat_finetune,
    save_state_dict,
    static_ptq,
    train_fp32,
)
from benchmark import write_csv
from models.cnn import SmallCNN

LOG = logging.getLogger("train_cnn")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=_positive_int, default=4)
    parser.add_argument("--qat-epochs", type=_positive_int, default=1)
    parser.add_argument("--calib-samples", type=_positive_int, default=256)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="x86", choices=("x86", "fbgemm"))
    parser.add_argument("--batch-sizes", type=_positive_int, nargs="+", default=[1, 32])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=_positive_int, default=40)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/cnn_fp32.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/cnn_quantization.csv"))
    parser.add_argument("--plot", type=Path, default=Path("results/cnn_quantization.png"))
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.calib_samples < 100 or args.calib_samples > 500:
        LOG.info(
            "calibration prefix is %s; the intended range for this experiment is 100 to 500",
            args.calib_samples,
        )

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    backend = configure_static_backend(args.backend)
    LOG.info("static quantization backend: %s", backend)
    LOG.info("loading CIFAR-10")
    train_x, train_y, test_x, test_y = load_cifar10()
    LOG.info("train=%s test=%s", tuple(train_x.shape), tuple(test_x.shape))
    if args.calib_samples > train_x.shape[0]:
        raise SystemExit("--calib-samples exceeds the training split")

    model = SmallCNN()
    LOG.info("training FP32 for %s epochs", args.epochs)
    train_fp32(
        model,
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        test_images=test_x,
        test_labels=test_y,
    )
    save_state_dict(model, args.checkpoint)
    LOG.info("wrote %s", args.checkpoint)

    LOG.info("static PTQ, calibration prefix=%s", args.calib_samples)
    static_model = static_ptq(
        model,
        train_x[: args.calib_samples],
        backend=backend,
        batch_size=args.batch_size,
    )
    LOG.info("QAT fine-tune for %s epoch(s)", args.qat_epochs)
    qat_model = qat_finetune(
        model,
        train_x,
        train_y,
        backend=backend,
        epochs=args.qat_epochs,
        batch_size=args.batch_size,
        lr=args.qat_lr,
        seed=args.seed + 1,
    )
    rows = build_rows(
        {"fp32": model, "static_ptq": static_model, "qat": qat_model},
        test_x,
        test_y,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        runs=args.runs,
        num_threads=args.threads,
        backend=backend,
        epochs=args.epochs,
        qat_epochs=args.qat_epochs,
        calib_samples=args.calib_samples,
        seed=args.seed,
        lr=args.lr,
        qat_lr=args.qat_lr,
        train_batch_size=args.batch_size,
    )
    write_csv(args.output, rows)
    plot_cnn(rows, args.plot)
    LOG.info("wrote %s", args.output)
    LOG.info("wrote %s", args.plot)


if __name__ == "__main__":
    main()

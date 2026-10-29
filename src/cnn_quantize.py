"""FX static PTQ and QAT for :class:`models.cnn.SmallCNN`.

Static PTQ: ``prepare_fx`` / ``convert_fx`` with the ``x86`` qconfig
(``fbgemm`` if ``x86`` is missing), calibrated on a prefix of the CIFAR-10
training set. QAT: ``prepare_qat_fx``, a short fine-tune, then ``convert_fx``.

Calibration uses ``torch.no_grad`` rather than ``inference_mode``. Observers
update activations in place, and inference mode can block those writes.
Latency measurement still runs under ``inference_mode``.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark import hardware_info, measure_cpu_latency, state_dict_size_mb, write_csv
from cifar10 import maybe_flip_horizontal, minibatches
from models.cnn import SmallCNN, count_parameters

LOG = logging.getLogger("cnn")

_BACKEND_PREFERENCE = ("x86", "fbgemm")


def configure_static_backend(preferred: str = "x86") -> str:
    """Select a CPU static-quantization engine and return its name."""
    supported = list(torch.backends.quantized.supported_engines)
    order = (preferred,) + tuple(name for name in _BACKEND_PREFERENCE if name != preferred)
    for name in order:
        if name in supported:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(
        "No x86/fbgemm quantized engine is available for static PTQ. "
        f"supported_engines={supported}."
    )


def _qconfig_mapping(backend: str, *, qat: bool):
    from torch.ao.quantization import get_default_qat_qconfig_mapping, get_default_qconfig_mapping

    if qat:
        return get_default_qat_qconfig_mapping(backend)
    return get_default_qconfig_mapping(backend)


def _example_inputs(in_channels: int = 3) -> tuple[torch.Tensor, ...]:
    return (torch.randn(1, in_channels, 32, 32),)


def static_ptq(
    model: nn.Module,
    calibration_images: torch.Tensor,
    *,
    backend: str,
    batch_size: int = 64,
) -> nn.Module:
    """Calibrate a copy of ``model`` and return an INT8 graph module.

    ``calibration_images`` is used in order, from the start. The input module
    is not modified.
    """
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    if calibration_images.shape[0] < 1:
        raise ValueError("calibration requires at least one image")
    backend = configure_static_backend(backend)
    prepared = prepare_fx(
        copy.deepcopy(model).eval(),
        _qconfig_mapping(backend, qat=False),
        _example_inputs(calibration_images.shape[1]),
    )
    prepared.eval()
    seen = 0
    target = calibration_images.shape[0]
    with torch.no_grad():
        for start in range(0, target, batch_size):
            prepared(calibration_images[start : start + batch_size])
            seen += min(batch_size, target - start)
    if seen < 1:
        raise RuntimeError("calibration saw no examples")
    converted = convert_fx(prepared)
    converted.eval()
    return converted


def qat_finetune(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    backend: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> nn.Module:
    """Fine-tune a copy with fake quantization and return a converted INT8 model."""
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

    if epochs < 1:
        raise ValueError("QAT epochs must be >= 1")
    backend = configure_static_backend(backend)
    prepared = prepare_qat_fx(
        copy.deepcopy(model).train(),
        _qconfig_mapping(backend, qat=True),
        _example_inputs(images.shape[1]),
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in prepared.parameters() if parameter.requires_grad),
        lr=lr,
    )
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        prepared.train()
        total_loss = 0.0
        seen = 0
        for batch_images, batch_labels in minibatches(
            images,
            labels,
            batch_size,
            shuffle=True,
            generator=generator,
        ):
            batch_images = maybe_flip_horizontal(batch_images, generator)
            optimizer.zero_grad(set_to_none=True)
            logits = prepared(batch_images)
            loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_labels.shape[0]
            seen += int(batch_labels.shape[0])
        LOG.info(
            "qat epoch %s/%s train_loss=%.4f",
            epoch + 1,
            epochs,
            total_loss / max(seen, 1),
        )
    prepared.eval()
    converted = convert_fx(prepared)
    converted.eval()
    return converted


def classification_accuracy(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 256,
) -> tuple[float, int, int]:
    """Argmax accuracy. Returns ``(accuracy, correct, seen)``."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    correct = 0
    seen = 0
    model.eval()
    with torch.inference_mode():
        for batch_images, batch_labels in minibatches(
            images,
            labels,
            batch_size,
            shuffle=False,
        ):
            predictions = model(batch_images).argmax(dim=-1)
            correct += int((predictions == batch_labels).sum())
            seen += int(batch_labels.shape[0])
    if seen == 0:
        raise ValueError("no examples to score")
    return correct / seen, correct, seen


def train_fp32(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    test_images: torch.Tensor | None = None,
    test_labels: torch.Tensor | None = None,
) -> None:
    """Train ``model`` in place with Adam and random horizontal flips."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_images, batch_labels in minibatches(
            images,
            labels,
            batch_size,
            shuffle=True,
            generator=generator,
        ):
            batch_images = maybe_flip_horizontal(batch_images, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_labels.shape[0]
            seen += int(batch_labels.shape[0])
        message = f"fp32 epoch {epoch + 1}/{epochs} train_loss={total_loss / max(seen, 1):.4f}"
        if test_images is not None and test_labels is not None:
            accuracy, correct, count = classification_accuracy(model, test_images, test_labels)
            message += f" test_acc={accuracy:.4f} ({correct}/{count})"
        LOG.info(message)
    model.eval()


def measure_latency(
    model: nn.Module,
    batches: dict[int, torch.Tensor],
    *,
    warmup: int,
    runs: int,
    num_threads: int,
) -> dict[int, object]:
    model.eval()
    results = {}
    with torch.inference_mode():
        for batch_size, inputs in batches.items():
            LOG.info("latency batch_size=%s warmup=%s runs=%s", batch_size, warmup, runs)
            results[batch_size] = measure_cpu_latency(
                lambda inputs=inputs: model(inputs),
                warmup=warmup,
                runs=runs,
                num_threads=num_threads,
            )
    return results


def latency_inputs(test_images: torch.Tensor, batch_sizes: list[int]) -> dict[int, torch.Tensor]:
    batches = {}
    for batch_size in batch_sizes:
        if test_images.shape[0] < batch_size:
            raise ValueError(f"need at least {batch_size} test images for latency")
        batches[batch_size] = test_images[:batch_size].contiguous()
    return batches


def build_rows(
    models: dict[str, nn.Module],
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    batch_sizes: list[int],
    warmup: int,
    runs: int,
    num_threads: int,
    backend: str,
    epochs: int,
    qat_epochs: int,
    calib_samples: int,
    seed: int,
    lr: float,
    qat_lr: float,
    train_batch_size: int,
) -> list[dict[str, object]]:
    hardware = hardware_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    batches = latency_inputs(test_images, batch_sizes)
    parameters = count_parameters(SmallCNN())
    rows: list[dict[str, object]] = []
    for method, model in models.items():
        accuracy, correct, seen = classification_accuracy(model, test_images, test_labels)
        size_mb = round(state_dict_size_mb(model), 4)
        LOG.info("%s size_mb=%s accuracy=%.4f (%s/%s)", method, size_mb, accuracy, correct, seen)
        latency = measure_latency(
            model,
            batches,
            warmup=warmup,
            runs=runs,
            num_threads=num_threads,
        )
        for batch_size in batch_sizes:
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
                    "model": "SmallCNN",
                    "dataset": "uoft-cs/cifar10",
                    "method": method,
                    "backend": backend,
                    "parameters": parameters,
                    "train_epochs": epochs,
                    "qat_epochs": qat_epochs,
                    "calib_samples": calib_samples,
                    "seed": seed,
                    "lr": lr,
                    "qat_lr": qat_lr,
                    "train_batch_size": train_batch_size,
                    "size_mb": size_mb,
                    "accuracy": round(accuracy, 6),
                    "accuracy_correct": correct,
                    "accuracy_samples": seen,
                    "split": "test",
                    "batch_size": batch_size,
                    "latency_input": "cifar10_test_prefix",
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
    return rows


def plot_cnn(rows: list[dict[str, object]], path: Path) -> None:
    """Accuracy bars and median-latency bars for each method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = []
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            methods.append(method)
    by_method = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), {})[int(row["batch_size"])] = row

    accuracies = [
        float(by_method[method][next(iter(by_method[method]))]["accuracy"]) for method in methods
    ]
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].bar(methods, accuracies, color=["#4c78a8", "#f58518", "#54a24b"][: len(methods)])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("CIFAR-10 test accuracy")
    axes[0].set_title("Accuracy")
    axes[0].grid(True, axis="y", alpha=0.3)

    width = 0.35 if len(batch_sizes) > 1 else 0.5
    positions = list(range(len(methods)))
    for offset, batch_size in enumerate(batch_sizes):
        shift = (offset - (len(batch_sizes) - 1) / 2) * width
        values = [float(by_method[method][batch_size]["median_ms"]) for method in methods]
        axes[1].bar(
            [position + shift for position in positions],
            values,
            width=width,
            label=f"batch {batch_size}",
        )
    axes[1].set_xticks(positions, methods)
    axes[1].set_ylabel("Median latency (ms)")
    axes[1].set_title("CPU latency")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("SmallCNN CIFAR-10: FP32 vs static PTQ vs QAT")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_state_dict(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_fp32(path: Path) -> SmallCNN:
    model = SmallCNN()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

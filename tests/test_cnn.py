"""CNN, static PTQ, and QAT smoke tests. No CIFAR-10 download."""

from __future__ import annotations

import torch

from benchmark import state_dict_size_bytes
from cifar10 import normalize_cifar
from cnn_quantize import classification_accuracy, qat_finetune, static_ptq
from models.cnn import SmallCNN, count_parameters


def test_small_cnn_forward_shape_and_parameter_count():
    model = SmallCNN()
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)
    assert count_parameters(model) > 10_000


def test_normalize_cifar_centers_the_training_mean():
    images = torch.zeros(1, 3, 32, 32)
    images[:, 0] = 0.4914
    images[:, 1] = 0.4822
    images[:, 2] = 0.4465
    normalized = normalize_cifar(images)
    assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-5)


def test_static_ptq_runs_and_is_smaller_than_fp32():
    torch.manual_seed(0)
    model = SmallCNN().eval()
    calibration = torch.randn(8, 3, 32, 32)
    quantized = static_ptq(model, calibration, backend="x86", batch_size=4)
    with torch.inference_mode():
        output = quantized(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)
    assert state_dict_size_bytes(quantized) < state_dict_size_bytes(model)
    # The FP32 module is left trainable and unconverted.
    assert type(model.conv1).__module__.startswith("torch.nn")


def test_qat_one_step_converts_and_scores():
    torch.manual_seed(0)
    model = SmallCNN()
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    quantized = qat_finetune(
        model,
        images,
        labels,
        backend="x86",
        epochs=1,
        batch_size=4,
        lr=1e-4,
        seed=0,
    )
    accuracy, correct, seen = classification_accuracy(quantized, images, labels, batch_size=4)
    assert seen == 8
    assert 0 <= correct <= 8
    assert 0.0 <= accuracy <= 1.0
    with torch.inference_mode():
        assert quantized(images[:2]).shape == (2, 10)

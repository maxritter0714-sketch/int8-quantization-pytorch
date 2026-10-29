"""CIFAR-10 tensors for the small CNN.

Images come from ``uoft-cs/cifar10`` on the Hugging Face Hub (50,000 train
and 10,000 test, 32x32 RGB, 10 classes). That is the same split as the
Toronto tarball. Values are scaled to ``[0, 1]`` and normalized with the
usual CIFAR-10 mean and standard deviation.
"""

from __future__ import annotations

import torch

DATASET_ID = "uoft-cs/cifar10"
# Mean and std of the CIFAR-10 training set after scaling pixels to [0, 1].
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def normalize_cifar(images: torch.Tensor) -> torch.Tensor:
    """Normalize NCHW images that are already scaled to ``[0, 1]``."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected NCHW with 3 channels, got {tuple(images.shape)}")
    mean = images.new_tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def load_cifar10() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(train_x, train_y, test_x, test_y)`` on CPU."""
    train_x, train_y = _load_split("train")
    test_x, test_y = _load_split("test")
    return train_x, train_y, test_x, test_y


def _load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split=split)
    packed = np.stack([np.asarray(image, dtype=np.uint8) for image in dataset["img"]])
    if packed.ndim != 4 or packed.shape[1:] != (32, 32, 3):
        raise RuntimeError(f"unexpected CIFAR-10 image shape {packed.shape} for split {split}")
    images = torch.from_numpy(packed).permute(0, 3, 1, 2).contiguous()
    images = images.to(dtype=torch.float32).div_(255.0)
    images = normalize_cifar(images)
    labels = torch.tensor([int(label) for label in dataset["label"]], dtype=torch.long)
    if labels.shape[0] != images.shape[0]:
        raise RuntimeError("CIFAR-10 image and label counts differ")
    return images, labels


def minibatches(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
):
    """Yield ``(images, labels)`` views. Training flips are applied by the caller."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    count = images.shape[0]
    if shuffle:
        order = torch.randperm(count, generator=generator)
    else:
        order = torch.arange(count)
    for start in range(0, count, batch_size):
        index = order[start : start + batch_size]
        yield images[index], labels[index]


def maybe_flip_horizontal(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Return a batch with each image flipped left-right with probability 1/2."""
    flipped = images.clone()
    mask = torch.rand(flipped.shape[0], generator=generator) < 0.5
    flipped[mask] = flipped[mask].flip(-1)
    return flipped

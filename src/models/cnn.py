"""Small CNN for MNIST or CIFAR-10.

TODO (week 2): a few-minute CPU model for static post-training quantization
and quantization-aware training. Not used by the week-1 scripts.
"""

from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    """Placeholder. The forward pass is intentionally unimplemented."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError(
            "SmallCNN is a week-2 stub for static PTQ and QAT. "
            "The architecture and training loop are not in this revision."
        )

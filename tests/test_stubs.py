"""Week-2 entry points exist and refuse to pretend they are done."""

from __future__ import annotations

import pytest

import qat
import quantize_static
import train_cnn
from models.cnn import SmallCNN


def test_cnn_forward_is_not_implemented():
    model = SmallCNN()
    with pytest.raises(NotImplementedError):
        model(None)


@pytest.mark.parametrize("module", [train_cnn, quantize_static, qat])
def test_week_two_mains_are_not_implemented(module):
    with pytest.raises(NotImplementedError):
        module.main()

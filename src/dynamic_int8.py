"""Dynamic INT8 quantization of ``nn.Linear`` modules.

Embeddings and other non-linear modules stay FP32. The eager
``quantize_dynamic`` API is deprecated upstream in favor of torchao; this
project calls it because that is the comparison we report.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

_ENGINE_PREFERENCE = ("fbgemm", "x86", "onednn", "qnnpack")


def configure_quantized_engine() -> str:
    """Pick a CPU quantized engine and make it the process default."""
    supported = list(torch.backends.quantized.supported_engines)
    for name in _ENGINE_PREFERENCE:
        if name in supported:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(
        f"No CPU quantized engine is available for dynamic INT8. supported_engines={supported}."
    )


def quantize_linear_dynamic(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Quantize ``nn.Linear`` modules to dynamic INT8.

    With ``inplace=False`` (the default) the input module is copied first.
    """
    from torch.ao.quantization import quantize_dynamic

    configure_quantized_engine()
    target = model if inplace else copy.deepcopy(model)
    target.eval()
    quantized = quantize_dynamic(
        target,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )
    quantized.eval()
    return quantized


def count_linears(model: nn.Module) -> tuple[int, int]:
    """Return ``(fp32_linear, dynamic_quantized_linear)`` module counts."""
    fp32 = 0
    quantized = 0
    for module in model.modules():
        if type(module).__name__ != "Linear":
            continue
        if "quantized" in type(module).__module__:
            quantized += 1
        elif isinstance(module, nn.Linear):
            fp32 += 1
    return fp32, quantized

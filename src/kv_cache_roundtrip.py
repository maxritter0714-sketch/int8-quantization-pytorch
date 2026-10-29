"""Symmetric INT8 round trip for KV-cache tensors.

This is not an integer attention kernel. Values are quantized and immediately
dequantized so the next matmul still runs in FP32. Storage byte counts are
the int8 payload plus FP32 scales, not the dequantized tensor.
"""

from __future__ import annotations

import torch

from kv_cache_size import iter_kv_tensors


def quantize_dequantize_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Symmetric per-tensor INT8 round trip. Returns values and storage bytes."""
    maximum = tensor.detach().abs().amax().clamp(min=1e-8)
    scale = maximum / 127.0
    quantized = torch.round(tensor.detach() / scale).clamp(-128, 127)
    restored = (quantized * scale).to(dtype=tensor.dtype)
    return restored, int(tensor.numel()) + 4


def quantize_dequantize_per_head(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Symmetric per-head INT8 round trip for ``(batch, heads, seq, dim)``.

    Other ranks fall back to a per-tensor scale. Storage is ``numel`` int8
    bytes plus 4 bytes per scale.
    """
    if tensor.ndim != 4:
        return quantize_dequantize_per_tensor(tensor)
    maximum = tensor.detach().abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = maximum / 127.0
    quantized = torch.round(tensor.detach() / scale).clamp(-128, 127)
    restored = (quantized * scale).to(dtype=tensor.dtype)
    return restored, int(tensor.numel()) + int(scale.numel()) * 4


def int8_storage_bytes(past_key_values, *, per_head: bool = True) -> int:
    """Hypothetical INT8 cache size. Does not modify ``past_key_values``."""
    quantize = quantize_dequantize_per_head if per_head else quantize_dequantize_per_tensor
    seen: set[int] = set()
    total = 0
    for tensor in iter_kv_tensors(past_key_values):
        pointer = tensor.data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        _, nbytes = quantize(tensor)
        total += nbytes
    return total


def apply_int8_roundtrip(past_key_values, *, per_head: bool = True):
    """Write an INT8 round trip back into the cache. Returns ``(cache, storage_bytes)``.

    Normal tensors are updated in place. Inference tensors reject in-place
    writes, so those are replaced on the owning layer instead.
    """
    quantize = quantize_dequantize_per_head if per_head else quantize_dequantize_per_tensor
    seen: set[int] = set()
    total = 0

    def write(owner, attr: str, tensor: torch.Tensor) -> None:
        nonlocal total
        pointer = tensor.data_ptr()
        if pointer in seen:
            return
        seen.add(pointer)
        restored, nbytes = quantize(tensor)
        total += nbytes
        try:
            tensor.copy_(restored)
        except RuntimeError:
            setattr(owner, attr, restored)

    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        wrote = False
        for layer in layers:
            for attr in ("keys", "values", "key", "value"):
                tensor = getattr(layer, attr, None)
                if torch.is_tensor(tensor):
                    write(layer, attr, tensor)
                    wrote = True
        if wrote:
            return past_key_values, total

    if isinstance(past_key_values, (tuple, list)):
        rebuilt = []
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                new_layer = []
                for item in layer:
                    if torch.is_tensor(item):
                        pointer = item.data_ptr()
                        if pointer not in seen:
                            seen.add(pointer)
                            restored, nbytes = quantize(item)
                            total += nbytes
                            new_layer.append(restored)
                        else:
                            new_layer.append(item)
                    else:
                        new_layer.append(item)
                rebuilt.append(tuple(new_layer))
            else:
                rebuilt.append(layer)
        return tuple(rebuilt), total

    raise TypeError(
        f"Unrecognized past_key_values type: {type(past_key_values)!r}. "
        "Cannot round-trip the KV cache."
    )

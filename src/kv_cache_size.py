"""Theoretical and measured KV-cache size.

Theoretical cache size, batch 1, one value's byte width:

    2 * layers * kv_heads * head_dim * seq_len * bytes_per_value

``kv_heads`` is ``num_key_value_heads`` when the config has it, otherwise
``n_head``. DistilGPT-2 and GPT-2 are multi-head, so the two match.
Measured size walks the key and value tensors in ``past_key_values``.
"""

from __future__ import annotations

import torch


def theoretical_kv_cache_bytes(
    layers: int,
    heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_value: int,
    batch: int = 1,
) -> int:
    """Bytes in a dense KV cache: ``batch * 2 * layers * heads * head_dim * seq_len * bytes``."""
    values = {
        "layers": layers,
        "heads": heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "bytes_per_value": bytes_per_value,
        "batch": batch,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int")
    if batch < 1 or seq_len < 1 or bytes_per_value < 1:
        raise ValueError("batch, seq_len, and bytes_per_value must be >= 1")
    return batch * 2 * layers * heads * head_dim * seq_len * bytes_per_value


def config_int(config, *names: str, default: int | None = None) -> int:
    """First integer attribute among ``names``, or ``default``."""
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value
    if default is not None:
        return default
    raise ValueError(f"config is missing an int field among {names}")


def kv_dims_from_config(config) -> tuple[int, int, int]:
    """Return ``(layers, kv_heads, head_dim)`` for a GPT-2-style config."""
    layers = config_int(config, "n_layer", "num_hidden_layers")
    query_heads = config_int(config, "n_head", "num_attention_heads")
    hidden = config_int(config, "n_embd", "hidden_size")
    kv_heads = config_int(config, "num_key_value_heads", default=query_heads)
    if layers < 1 or query_heads < 1 or kv_heads < 1 or hidden < 1:
        raise ValueError("config dimensions must be positive")
    if hidden % query_heads != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by query heads {query_heads}")
    return layers, kv_heads, hidden // query_heads


def iter_kv_tensors(past_key_values):
    """Yield key and value tensors from a legacy tuple or a Cache object."""
    if past_key_values is None:
        return

    if isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                for tensor in layer:
                    if torch.is_tensor(tensor):
                        yield tensor
            elif torch.is_tensor(layer):
                yield layer
        return

    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if (
        isinstance(key_cache, (list, tuple))
        and key_cache
        and all(tensor is None or torch.is_tensor(tensor) for tensor in key_cache)
    ):
        for tensor in list(key_cache) + list(value_cache or []):
            if torch.is_tensor(tensor):
                yield tensor
        return

    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)) and layers:
        found = False
        for layer in layers:
            for attr in ("keys", "values", "key", "value"):
                tensor = getattr(layer, attr, None)
                if torch.is_tensor(tensor):
                    found = True
                    yield tensor
        if found:
            return

    raise TypeError(
        "Unrecognized past_key_values type: "
        f"{type(past_key_values)!r}. Cannot measure the KV cache."
    )


def measured_kv_cache_bytes(past_key_values) -> tuple[int, str]:
    """Logical bytes of unique key/value tensors, plus a shape summary."""
    tensors = []
    seen: set[tuple[int, tuple[int, ...], str]] = set()
    for tensor in iter_kv_tensors(past_key_values):
        identity = (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
        if identity in seen:
            continue
        seen.add(identity)
        tensors.append(tensor)
    total = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    shapes = "|".join(
        f"{tuple(tensor.shape)}:{str(tensor.dtype).replace('torch.', '')}" for tensor in tensors
    )
    return total, shapes

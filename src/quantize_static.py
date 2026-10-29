"""Static post-training quantization for the small CNN.

TODO (week 2): FX graph-mode static PTQ (``prepare_fx`` / ``convert_fx``,
or the torchao PT2E path if eager FX quantization has been removed),
calibration on about 100 to 500 examples, backend ``fbgemm`` or ``x86``.
Compare on-disk size, CPU latency, and accuracy with the FP32 checkpoint.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Static PTQ is scheduled for week 2. See the module docstring."
    )


if __name__ == "__main__":
    main()

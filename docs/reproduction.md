# Reproduction

Install from the repository root. `requirements.txt` selects the CPU torch wheel. `pyproject.toml` holds project metadata and the ruff and pytest configuration. It is not a second install path for torch.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The recorded runs used `--threads 4`, matching the four logical CPUs on the measurement host. The script defaults are not all 4. Pass the flags below if you want the same protocol. A different machine will not reproduce the same milliseconds.

## Commands that wrote the committed artifacts

```bash
# results/distilbert_sst2_dynamic_int8.csv
# Full SST-2 validation (872 rows). --limit N scores a prefix instead.
python src/quantize_dynamic.py --threads 4 --warmup 10 --runs 40

# results/kv_cache.csv and results/kv_cache.png
# Prompt lengths 32, 128, 512. 16 new greedy tokens. Warmup 2, 10 timed runs.
python src/kv_cache.py --threads 4

# results/cnn_quantization.csv and results/cnn_quantization.png
# Also writes results/cnn_fp32.pt, which is gitignored.
python src/train_cnn.py --epochs 4 --qat-epochs 1 --calib-samples 256 \
  --threads 4 --warmup 10 --runs 40

# results/kv_cache_int8.csv and results/kv_cache_int8.png
# First 512 tokens of WikiText-2 raw validation, windows of 128.
python src/kv_cache_int8.py --threads 4 --seq-len 128 --max-tokens 512
```

After `train_cnn.py` has written the checkpoint, these repeat one conversion and print size and test accuracy. They do not rewrite the combined CSV:

```bash
python src/quantize_static.py --checkpoint results/cnn_fp32.pt --calib-samples 256 --threads 4
python src/qat.py --checkpoint results/cnn_fp32.pt --epochs 1 --threads 4
```

## Tests and lint

Smoke tests build tiny local modules, including a random GPT-2 config. They do not download DistilBERT, DistilGPT-2, CIFAR-10, or WikiText-2. `tests/conftest.py` sets the Hub offline.

```bash
pytest
ruff check src tests
ruff format --check src tests
```

GitHub Actions runs pytest on the CPU wheel and runs the two ruff commands. The pytest job installs `requirements.txt`. The ruff job installs `ruff==0.17.3` only.

## What a rerun will and will not match

A rerun with the same flags, the same library pins, and a quiet machine should land near these medians. It will not match them digit for digit: the guest is shared, and QAT and FP32 training use dropout-free but still data-order-dependent flips under a fixed seed. The committed CSVs and plots are the run from 2026-10-29. Do not replace them with numbers from another host without saying so.

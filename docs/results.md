# Results

Rounded from the CSVs in `results/`. The exact fields, including minimum and maximum latency, are in those files. Hardware and the timing protocol are in [methodology.md](methodology.md).

## Combined table

| Experiment | Method | Footprint | Quality | Batch 1 median (p90) | Batch 32 median (p90) |
| --- | --- | --- | --- | --- | --- |
| DistilBERT SST-2 | FP32 | 255.45 MB | 0.9106 acc (794/872) | 19.49 ms (21.63) | 583.29 ms (590.37) |
| DistilBERT SST-2 | Dynamic INT8, `nn.Linear` | 132.29 MB | 0.9002 (785/872) | 11.47 ms (12.89) | 332.18 ms (339.17) |
| SmallCNN CIFAR-10 | FP32, 4 epochs | 0.364 MB | 0.5735 (5735/10000) | 0.21 ms (0.25) | 2.71 ms (2.89) |
| SmallCNN CIFAR-10 | Static PTQ, 256 calib | 0.102 MB | 0.5730 (5730/10000) | 0.22 ms (0.26) | 0.43 ms (0.51) |
| SmallCNN CIFAR-10 | QAT, plus 1 epoch | 0.102 MB | 0.6051 (6051/10000) | 0.22 ms (0.24) | 0.42 ms (0.46) |
| distilgpt2, prompt 512 | `use_cache=True` | 18.0 MB KV | — | 76.22 tokens/s | — |
| distilgpt2, prompt 512 | `use_cache=False` | 18.0 MB KV | — | 10.89 tokens/s | — |
| distilgpt2, WikiText-2 prefix | FP32 cache | 4.50 MB at seq 128 | ppl 57.02 (508 tokens) | — | — |
| distilgpt2, WikiText-2 prefix | Per-head INT8 round trip | 1.125 MB stored | ppl 57.35 | — | — |

## DistilBERT, dynamic INT8

Source: `results/distilbert_sst2_dynamic_int8.csv`. Model `distilbert-base-uncased-finetuned-sst-2-english`. All 38 `nn.Linear` modules were converted. Embeddings stayed FP32.

The checkpoint went from 255.45 MB to 132.29 MB (1.93×), not 4×. Accuracy fell by 9 validation items (1.03 percentage points), 794/872 to 785/872. On this CPU, median latency improved 1.70× at batch 1 and 1.76× at batch 32. That speedup is tied to a host that advertises AVX-512 VNNI and AMX-INT8. It is not a property of dynamic INT8 on every CPU. Activation quantization has overhead, and on a machine without a fast integer GEMM the same conversion can shrink the file and still not get faster.

## SmallCNN, static PTQ and QAT

Source: `results/cnn_quantization.csv` and `results/cnn_quantization.png`.

The network is three conv-relu-pool stages (32, 64, 128 channels), global average pooling, and a linear head: 94,538 parameters, no batch norm. CIFAR-10 test accuracy after four epochs is 0.5735. That is well above chance (0.10) and far from a trained vision model. The point of the run is the FP32-to-INT8 gap, not the absolute accuracy.

Static PTQ (`prepare_fx` / `convert_fx`, `x86`, first 256 training images) cut the state dict from 0.364 MB to 0.102 MB (3.56×) and lost 5 test images (5735 to 5730).

**Batch 1 did not get faster.** Median latency went from 0.21 ms to 0.22 ms (about 4% slower). The integer conv did not pay for its overhead on a single 32×32 image.

**Batch 32 did.** Median latency fell from 2.71 ms to 0.43 ms (6.4×). p90 fell from 2.89 ms to 0.51 ms, so the speedup is not an artifact of one lucky median.

QAT reached the same 0.102 MB INT8 state dict and the same batch-32 speed (median 0.42 ms, p90 0.46 ms) with test accuracy 0.6051 (6051/10000). That is 316 more correct images than the 4-epoch FP32 model. **The QAT run is an extra fine-tune epoch.** This table has no fifth FP32 epoch, so the accuracy lift is not evidence that quantization itself improved the model. It is evidence that one epoch of fake-quant fine-tuning, starting from those FP32 weights, landed at a higher test accuracy and at the INT8 size and speed of static PTQ.

## KV cache, FP32

Source: `results/kv_cache.csv` and `results/kv_cache.png`. `distilgpt2`, 16 new greedy tokens, warmup 2, 10 timed runs, batch 1.

| Prompt length | use_cache | Median tokens/s | Median ms | KV bytes |
| --- | --- | --- | --- | --- |
| 32 | true | 147.11 | 108.79 | 1,179,648 |
| 32 | false | 78.39 | 204.11 | 1,179,648 |
| 128 | true | 130.07 | 123.01 | 4,718,592 |
| 128 | false | 40.53 | 394.76 | 4,718,592 |
| 512 | true | 76.22 | 209.93 | 18,874,368 |
| 512 | false | 10.89 | 1469.22 | 18,874,368 |

Measured bytes matched `2 * 6 * 12 * 64 * seq_len * 4` at every length. The cache grows linearly: 32 → 128 → 512 is ×4 in length and ×4 in bytes (1.125 MB, 4.5 MB, 18.0 MB).

At 512 tokens, cached decode was 7.0× the tokens/s of recomputing the prefix (76.22 vs 10.89). Cached throughput still falls as the prompt grows, because each new token attends over the stored keys and values. The uncached path falls faster.

## KV cache, INT8 round trip

Source: `results/kv_cache_int8.csv` and `results/kv_cache_int8.png`. This is an experiment, not a kernel.

On a 508-token prefix of WikiText-2 raw validation (four windows of 128), stepwise FP32 perplexity was 57.02 and matched a one-shot forward at the same 57.02. A symmetric per-head INT8 round trip of `past_key_values` before each cached step moved perplexity to 57.35. That is a small increase on a small slice, not a claim about full WikiText-2.

The stored form of that cache would be 1,180,224 bytes (1.125 MB) instead of 4,718,592 bytes (4.50 MB): the int8 payload plus one FP32 scale per head. The runtime tensor after the round trip is still FP32, so this number is a storage estimate, not a measured RSS drop.

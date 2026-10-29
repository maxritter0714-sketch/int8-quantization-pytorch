# Measurement methodology

Every number in `results/` comes from one command on one CPU guest. The CSVs are the source of truth. Tables in the README and in [results.md](results.md) are rounded from those files.

## Hardware

Recorded on 2026-10-29.

| Field | Value |
| --- | --- |
| CPU string | Intel(R) Xeon(R) Processor (the guest does not expose a SKU) |
| Logical CPUs | 4 |
| Flags | `avx avx2 avx512f avx512_vnni avx_vnni fma amx_int8 amx_bf16` |
| OS | Linux 6.12.94+ |
| Torch | `2.15.0+cpu` |
| Python | 3.11.18 |
| Threads | `torch.set_num_threads(4)` for every recorded run |

Dynamic INT8 selected the first available engine in `fbgemm`, `x86`, `onednn`, `qnnpack`. On this host that was `fbgemm`. CNN static PTQ and QAT requested the `x86` qconfig, which is also the process default engine on this build.

## On-disk size

Size is the byte length of `torch.save(module.state_dict())`, including the zip container PyTorch writes. It is not a count of parameter bytes alone. Quantized weights are stored packed, so an INT8 state dict is smaller even when some tensors stay FP32.

For the KV cache, "measured" bytes are the unique key and value tensors in `past_key_values` (`numel * element_size`). The INT8 experiment additionally reports a hypothetical storage size: one byte per value plus one FP32 scale per head. That count is not the size of the tensor the process holds after the round trip, which is still FP32.

## CPU latency

Latency is wall time from `time.perf_counter` around the model call.

1. Set `torch.set_num_threads` for the block, then restore the previous value.
2. Run the warmup calls and discard them.
3. Time `runs` calls.
4. Report the median and the linear-interpolation 90th percentile (the same method as NumPy's default percentile).

Inference calls run under `torch.inference_mode`. There is no CUDA synchronization because these runs are CPU-only.

What is inside the timed region:

| Experiment | Timed call | Input |
| --- | --- | --- |
| DistilBERT | `model(**batch)` | Synthetic token ids, sequence length 128, batch 1 and 32. Not SST-2 sentences. |
| SmallCNN | `model(images)` | The first 1 and the first 32 images of the CIFAR-10 test set, already normalized. |
| DistilGPT-2 decode | `model.generate` for 16 new greedy tokens | A repeated text prompt of length 32, 128, or 512. |

The recorded DistilBERT and CNN runs used warmup 10 and 40 timed calls. The KV decode run used warmup 2 and 10 timed calls. Tokens per second is the median of `new_tokens / seconds` across those timed calls.

## Accuracy and perplexity

- **SST-2.** Argmax over the two logits on the GLUE validation split from `nyu-mll/glue` (872 labeled rows). Labels below 0 are skipped. The recorded run did not use `--limit`.
- **CIFAR-10.** Argmax over 10 logits on the full 10,000-image test split from `uoft-cs/cifar10`. Pixels are scaled to `[0, 1]` and normalized with mean `(0.4914, 0.4822, 0.4465)` and std `(0.2470, 0.2435, 0.2616)`. Training uses a random horizontal flip. The test set does not.
- **WikiText-2 perplexity.** Teacher-forced negative log-likelihood of each next token, then `exp(mean NLL)`. The slice is the first 512 tokens of non-empty WikiText-2 raw validation lines, cut into four windows of 128. Each window scores 127 tokens (508 predictions total). The FP32 stepwise loop matched a single forward with `labels=` at perplexity 57.02, which is the check that the cache path is scoring the same tokens.

## Calibration and training

Static PTQ watches the first 256 training images, in dataset order, under `torch.no_grad` (not `inference_mode`, so observers can update). That prefix is inside the 100-to-500 range and is not a random sample.

FP32 training is Adam at learning rate `1e-3` for 4 epochs, batch 128, seed 0. QAT is a copy of those weights, `prepare_qat_fx`, one more epoch at learning rate `1e-4`, then `convert_fx`. The extra epoch is part of the QAT number. There is no fifth FP32 epoch in the table.

## Noise

A shared virtual machine moves. Median and p90 reduce that; they do not remove it. On the recorded runs, batch-32 DistilBERT p90 sat within a few percent of the median. Batch-1 DistilBERT p90 sat about 11–12% above the median. CNN static PTQ at batch 32 had a wider tail (median 0.43 ms, p90 0.51 ms) and the speedup versus FP32 was still several times at both statistics. Do not read one run as a confidence interval.

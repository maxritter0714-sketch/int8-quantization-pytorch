# Design decisions

## Dynamic PTQ for DistilBERT, static PTQ and QAT for the CNN

DistilBERT's heavy parameters are `nn.Linear` (attention and the feed-forward). Dynamic INT8 (`quantize_dynamic` on `nn.Linear` only) does not need a calibration set: weights are stored as integers and activations are quantized when the linear runs. Embeddings and layer norms stay FP32, which is why the file shrinks by less than 4×. That is the simplest honest comparison for a transformer that is already fine-tuned.

A CNN is a better fit for static post-training quantization. Convolution activations have a range you can observe, and FX graph mode (`prepare_fx` / `convert_fx`) can fuse conv and ReLU and then quantize both weights and activations. QAT (`prepare_qat_fx`, a short fine-tune, `convert_fx`) is the same graph with the ranges learned instead of only observed. The CNN has no batch norm, so the comparison is not mixed up with batch-norm folding.

The three methods answer different questions:

- Dynamic PTQ: how much do you get if you only touch `nn.Linear` and do not calibrate?
- Static PTQ: how much do you get if you calibrate once and convert, with no extra training?
- QAT: how much do you get if you spend one more epoch letting the network adapt to fake quantization?

## Why the CNN backend is `x86`

`torch.backends.quantized.supported_engines` on this build lists `qnnpack`, `onednn`, `x86`, and `fbgemm`. The process default is `x86`. Static PTQ and QAT ask for the `x86` qconfig mapping, which matches that default. `fbgemm` is still present; dynamic INT8 prefers it and used it for DistilBERT. Both names are the x86 integer path. The CSV records which one each script selected.

PyTorch marks `torch.ao.quantization` deprecated and points new work at torchao. These scripts still call `quantize_dynamic`, `prepare_fx`, `convert_fx`, and `prepare_qat_fx`, because those are the measurements this repository is about.

## Calibration size

Static PTQ uses the first 256 training images. The intended range for this experiment is about 100 to 500 examples. 256 is inside that range, it is a prefix of the training split (so two runs see the same images), and it is not a random sample and not the test set. Accuracy is always reported on the full 10,000-image test split.

## CIFAR-10 instead of MNIST

The recorded `train_cnn.py` command (4 FP32 epochs, static PTQ, 1 QAT epoch, latency) finished in 62 seconds on this host after CIFAR-10 was already in the Hugging Face cache. MNIST would have been faster still and would have said less about a color image model. The network stays small on purpose: 94,538 parameters, so a few epochs are enough to leave chance behind and the quantization gap is the result.

## KV INT8 is a round trip

An integer attention kernel would keep keys and values in INT8 inside the matmul. This experiment does not do that. After each cached step it replaces each key and value tensor with a symmetric per-head quantize/dequantize (`scale = max(abs) / 127`, integers in `[-128, 127]`, then back to FP32) so the next PyTorch attention kernel can run unchanged.

That isolates two things:

- numeric damage, measured as perplexity on a small WikiText-2 prefix
- the byte count of the packed form (int8 values plus FP32 scales)

It does not measure latency of INT8 attention, and it does not reduce the resident FP32 tensor. Per-head scales (over sequence and head dimension) match the `(batch, heads, seq, dim)` layout DistilGPT-2 actually stores. A non-4D tensor falls back to one scale for the whole tensor.

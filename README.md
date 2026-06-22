# ML-workflow-image

[![Docker Image CI](https://github.com/JuiceB0xC0de/ML-workflow-image/actions/workflows/docker.yml/badge.svg)](https://github.com/JuiceB0xC0de/ML-workflow-image/actions/workflows/docker.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/juiceboxdocks/ml-workflow-image)](https://hub.docker.com/r/juiceboxdocks/ml-workflow-image)
[![HF Wheel](https://img.shields.io/badge/HF%20Wheel-juiceb0xc0de%2Fllama--cpp--cu128--wheel-blue)](https://huggingface.co/juiceb0xc0de/llama-cpp-cu128-wheel)

A pre-built CUDA 12.8 ML GPU image for RunPod and any Docker host with an NVIDIA GPU. Pull it, mount your code/data, and train/fine-tune/evaluate/serve without waiting for conda or compiling llama.cpp from source.

```bash
docker pull juiceboxdocks/ml-workflow-image:latest

# Run an interactive shell with all GPUs
docker run --rm -it --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  juiceboxdocks/ml-workflow-image:latest

# One-shot smoke test
docker run --rm --gpus all juiceboxdocks/ml-workflow-image:latest \
  bash -c "llama-cli --version && python -c 'import torch, flash_attn; print(torch.__version__, torch.cuda.is_available())'"
```

## What it's for

- Fine-tuning LLMs: full FT, LoRA/QLoRA (`peft` + `bitsandbytes`)
- RLHF / DPO / PPO (`trl`)
- Mechanistic interpretability (`transformer-lens`, `sae-lens`)
- SAE training and feature analysis
- Local GGUF inference and quantization (`llama.cpp` binaries)
- Dataset wrangling, W&B logging, Jupyter exploration, pytest CI

## Image tags

| Tag | Use |
|---|---|
| `juiceboxdocks/ml-workflow-image:latest` | Latest build |
| `juiceboxdocks/ml-workflow-image:cu128` | Pinned CUDA 12.8 build |

## Anchor & compatibility

The image is **anchored on a prebuilt flash-attn wheel** so nothing slow has to compile, which locks the stack:

| | |
|---|---|
| flash-attn | `2.8.3.post1 +cu12 torch2.4 cxx11abiFALSE cp310` (prebuilt) |
| Python | 3.10 |
| torch | 2.4.0 (pip wheel, cu121) |
| CUDA runtime | 12.8.1 (host driver must be CUDA 12.0+) |
| Base image | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` |

**GPU coverage:** Ampere `sm_80` (A100) · Ada `sm_89` (RTX 6000 Ada) · Hopper `sm_90` (H100).
**Not covered:** Blackwell `sm_100` / `sm_120` — would need a cu130 build.

## What's installed

torch/torchvision/torchaudio · xformers · flash-attn · transformers · accelerate
· datasets · peft · trl · bitsandbytes (LoRA/QLoRA) · transformer-lens · sae-lens
· huggingface_hub + hf_transfer · wandb · tensorboard · boto3 · rclone · llama.cpp
binaries (`llama-cli`/`llama-server`/`llama-quantize`/`llama-embedding`) compiled
for CUDA 12.8 + sm_80 · Jupyter · pytest · pandas · scipy · scikit-learn ·
matplotlib · seaborn · umap-learn · einops · rich · tqdm.

**Intentionally excluded:** DeepSpeed, autoawq, gptqmodel, unsloth, llama-cpp-python.

The image ships the official `llama.cpp` binaries directly because building
`llama-cpp-python` from source in GitHub Actions is slow and unreliable. If you
need the Python binding, install it at runtime.

## Build

GitHub Actions builds on push to `main` (or via **Run workflow**) and pushes:
- `juiceboxdocks/ml-workflow-image:cu128`
- `juiceboxdocks/ml-workflow-image:latest`

### Required repo secrets (Settings → Secrets → Actions)
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## RunPod

- **Image:** `juiceboxdocks/ml-workflow-image:latest`
- **GPU:** A100 / H100 / RTX 6000 Ada
- **Env:** `HF_TOKEN`, `WANDB_API_KEY`, plus `B2_ACCOUNT` / `B2_KEY` / `B2_BUCKET` for `scripts/sync_pools.sh`
- **Network volume:** optional; keep hot data local and archive pool batches to B2.
- **Start command:** leave the template's container start command blank. The image's own `CMD ["/workspace/scripts/start.sh"]` keeps the container alive; RunPod's init injects SSH and Jupyter.
- If you previously used the `cxx11abiFALSE` wheel, note that the image now uses the `cxx11abiTRUE` wheel to match PyPI torch's ABI.

## Smoke test

```bash
pip check && python -c "import torch, flash_attn, transformers, peft, trl, \
transformer_lens, sae_lens, bitsandbytes; print('imports OK', torch.__version__)"
llama-cli --version
```

## GPU reference

See below for architecture specs. This image is built for **Ampere / Ada / Hopper** (`sm_80`, `sm_89`, `sm_90`). Blackwell is not supported by this tag.

### Ampere Architecture (2020–2022)

3rd-gen Tensor Cores · TF32/BF16/INT8 · NVLink 3.0 (600 GB/s) · MIG on A100/A30 · Structural 2:4 sparsity

| GPU | VRAM | Mem Type | Mem BW | CUDA Cores | Tensor Cores | TDP | FP16 TFLOPS | FP8 TFLOPS | NVLink BW | MIG |
|---|---|---|---|---|---|---|---|---|---|---|
| A100 SXM 80GB | 80 GB | HBM2e | 2,039 GB/s | 6,912 | 432 | 400W | 77.6 | — | 600 GB/s | ✅ Up to 7 |
| A100 PCIe 80GB | 80 GB | HBM2e | 1,935 GB/s | 6,912 | 432 | 300W | 77.6 | — | — | ✅ Up to 7 |
| A100 40GB | 40 GB | HBM2e | 1,555 GB/s | 6,912 | 432 | 300W | 77.6 | — | 600 GB/s | ✅ Up to 7 |
| A40 | 48 GB | GDDR6 ECC | 696 GB/s | 10,752 | 336 | 300W | 37.4 | — | — | ❌ |
| A30 | 24 GB | HBM2 | 933 GB/s | 3,584 | 224 | 165W | 165 | — | 200 GB/s | ✅ Up to 4 |
| A10 / A10G | 24 GB | GDDR6 ECC | 600 GB/s | 9,216 | 288 | 150–300W | 31.2 | — | — | ❌ |
| RTX A5000 | 24 GB | GDDR6 ECC | 768 GB/s | 8,192 | 256 | 230W | 27.8 | — | — | ❌ |
| RTX A6000 | 48 GB | GDDR6 ECC | 768 GB/s | 10,752 | 336 | 300W | 38.7 | — | — | ❌ |
| RTX A4000 | 16 GB | GDDR6 ECC | 448 GB/s | 6,144 | 192 | 140W | 19.2 | — | — | ❌ |
| RTX A4500 | 20 GB | GDDR6 ECC | 640 GB/s | 7,168 | 224 | 200W | 23.7 | — | — | ❌ |
| RTX 3090 | 24 GB | GDDR6X | 936 GB/s | 10,496 | 328 | 350W | 35.6 | — | — | ❌ |
| RTX 3080 | 10 GB | GDDR6X | 760 GB/s | 8,704 | 272 | 320W | 29.8 | — | — | ❌ |

### Ada Lovelace Architecture (2022–2023)

4th-gen Tensor Cores · Native FP8 (L-series pro) · DLSS 3 · 3rd-gen RT Cores · AV1 encode/decode

| GPU | VRAM | Mem Type | Mem BW | CUDA Cores | Tensor Cores | TDP | FP16 TFLOPS | FP8 TFLOPS | MIG |
|---|---|---|---|---|---|---|---|---|---|
| L40S | 48 GB | GDDR6 ECC | 864 GB/s | 18,176 | 568 | 350W | 91.6 | 183 | ❌ |
| L40 | 48 GB | GDDR6 ECC | 864 GB/s | 18,176 | 568 | 300W | 90.5 | — | ❌ |
| L4 | 24 GB | GDDR6 ECC | 300 GB/s | 7,680 | 240 | 72W | 30.3 | 60.6 | ❌ |
| RTX 6000 Ada | 48 GB | GDDR6 ECC | 960 GB/s | 18,176 | 568 | 300W | 91.1 | — | ❌ |
| RTX 5000 Ada | 32 GB | GDDR6 ECC | 576 GB/s | 12,800 | 400 | 250W | 57.7 | — | ❌ |
| RTX 4500 Ada | 24 GB | GDDR6 ECC | 432 GB/s | 7,680 | 240 | 210W | 29.6 | — | ❌ |
| RTX 4000 Ada | 20 GB | GDDR6 ECC | 360 GB/s | 6,144 | 192 | 100W | 26.7 | — | ❌ |
| RTX 2000 Ada | 16 GB | GDDR6 ECC | 224 GB/s | 3,072 | 96 | 70W | 12.0 | — | ❌ |
| RTX 4090 | 24 GB | GDDR6X | 1,008 GB/s | 16,384 | 512 | 450W | 82.6 | — | ❌ |
| RTX 4080 | 16 GB | GDDR6X | 717 GB/s | 9,728 | 304 | 320W | 48.7 | — | ❌ |
| RTX 4070 | 12 GB | GDDR6X | 504 GB/s | 5,888 | 184 | 200W | 29.1 | — | ❌ |
| RTX PRO 4000 | 24 GB | GDDR7 | ~576 GB/s | 7,680 | 240 | 130W | ~45 | — | ❌ |
| RTX PRO 4500 | 32 GB | GDDR7 | ~640 GB/s | 9,728 | 304 | 250W | ~57 | — | ❌ |
| RTX PRO 6000 | 96 GB | GDDR7 | ~960 GB/s | 18,176 | 568 | 300W | ~125 | — | ❌ |
| RTX PRO 6000 WK | 96 GB | GDDR7 | ~960 GB/s | 18,176 | 568 | 300W | ~125 | — | ❌ |

> ⚠️ Ada PRO GDDR7 specs are partially pre-release estimates. Verify with NVIDIA datacenter docs before production use.

### Hopper Architecture (2022–2023)

Transformer Engine (auto FP8↔FP16) · 4th-gen Tensor Cores · NVLink 4.0 (900 GB/s) · HBM3/HBM3e · MIG

| GPU | VRAM | Mem Type | Mem BW | CUDA Cores | Tensor Cores | TDP | FP16 TFLOPS | FP8 TFLOPS | NVLink BW | MIG |
|---|---|---|---|---|---|---|---|---|---|---|
| H100 SXM | 80 GB | HBM3 | 3,350 GB/s | 16,896 | 528 | 700W | 989 | 1,979 | 900 GB/s | ✅ Up to 7 |
| H100 PCIe | 80 GB | HBM3 | 2,000 GB/s | 14,592 | 456 | 350W | 800 | 1,600 | — | ✅ Up to 7 |
| H100 NVL | 94 GB | HBM3 | 3,350 GB/s | 16,896 | 528 | 700W | 989 | 1,979 | 900 GB/s | ✅ Up to 7 |
| H200 SXM | 141 GB | HBM3e | 4,800 GB/s | 16,896 | 528 | 700W | 989 | 1,979 | 900 GB/s | ✅ Up to 7 |
| H200 NVL | 143 GB | HBM3e | 4,800 GB/s | 16,896 | 528 | 700W | 989 | 1,979 | 900 GB/s | ✅ Up to 7 |

### Blackwell Architecture (2024–2025)

5th-gen Tensor Cores · FP4 support · NVLink 5.0 · HBM3e · GB200 NVL72 rack-scale design

| GPU | VRAM | Mem Type | Mem BW | CUDA Cores | TDP | FP16 TFLOPS | FP8 TFLOPS | FP4 TFLOPS | MIG |
|---|---|---|---|---|---|---|---|---|---|
| B200 | 180 GB | HBM3e | 8,000 GB/s | 20,480 | 1,000W | 2,250 | 4,500 | 9,000 | ✅ |
| B300 | 288 GB | HBM3e | 16,000 GB/s | 20,480 | 1,200W | 2,500 | 5,000 | 10,000 | ✅ |

> ⚠️ Blackwell specs are partially pre-release estimates. This image does **not** support Blackwell. Build a cu130 variant if you need it.

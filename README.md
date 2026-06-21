# ML-workflow-image

One-size-fits-all ML GPU Docker image for RunPod — training, fine-tuning,
LoRA/QLoRA quantization, interpretability/SAE work, data science, and local
GGUF inference via llama.cpp. Built by GitHub Actions, pushed to Docker Hub.

## Anchor & compatibility

The image is **anchored on a prebuilt flash-attn wheel** so nothing slow has to
compile, which locks the stack:

| | |
|---|---|
| flash-attn | `2.8.3.post1 +cu12 torch2.4 cxx11abiFALSE cp310` (prebuilt) |
| Python | 3.10 |
| torch | 2.4.0 (pip wheel, cu121) |
| CUDA runtime | 12.8.1 (host driver must be CUDA 12.0+) |

**GPU coverage:** Ampere `sm_80` (A100) · Ada `sm_89` (RTX 6000 Ada) · Hopper
`sm_90` (H100).
**Not covered:** Blackwell — B200 `sm_100`, RTX Pro 6000 Blackwell `sm_120`.

Base is `nvidia/cuda:12.8.1-cudnn9-runtime-ubuntu22.04` plus pip torch (NOT an
NGC image) because the `cxx11abiFALSE` wheel matches PyPI torch's ABI, not
NGC's.

## What's installed

torch/torchvision/torchaudio · xformers · flash-attn · transformers · accelerate
· datasets · peft · trl · bitsandbytes (LoRA/QLoRA) · transformer-lens · sae-lens
· huggingface_hub + hf_transfer · wandb · tensorboard · boto3 · llama.cpp binaries
(`llama-cli`/`llama-server`/`llama-quantize`/`llama-embedding`) compiled for CUDA
12.8 + sm_80 · the usual data-science + Jupyter + pytest stack.

**Intentionally excluded:** DeepSpeed, autoawq, gptqmodel, unsloth.

## Build

GitHub Actions builds on push to `main` (or via **Run workflow**) and pushes:
- `juiceboxdocks/ml-workflow-image:cu128`
- `juiceboxdocks/ml-workflow-image:latest`

### Required repo secrets (Settings → Secrets → Actions)
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (Docker Hub → Account → Security → New Access Token)

## RunPod

- **Image:** `juiceboxdocks/ml-workflow-image:latest`
- **GPU:** A100 / H100 / RTX 6000 Ada
- **Env:** `HF_TOKEN`, `WANDB_API_KEY`, plus `B2_ACCOUNT` / `B2_KEY` / `B2_BUCKET`
  for `sync_pools`.

## Smoke test

```bash
pip check && python -c "import torch, flash_attn, transformers, peft, trl, \
transformer_lens, sae_lens, bitsandbytes; print('imports OK', torch.__version__)"
llama-cli --version
```

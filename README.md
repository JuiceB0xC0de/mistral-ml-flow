# ML-workflow-image

[![Docker Image CI](https://github.com/JuiceB0xC0de/ML-workflow-image/actions/workflows/docker.yml/badge.svg)](https://github.com/JuiceB0xC0de/ML-workflow-image/actions/workflows/docker.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/juiceboxdocks/ml-workflow-image)](https://hub.docker.com/r/juiceboxdocks/ml-workflow-image)

Single-purpose NVIDIA GPU image for running the GWIQ atlas CLI from a pod or Docker host.

This image is no longer an "install every ML tool" workstation. It gives you a clean CUDA/Python runtime with PyTorch, Transformers, a prebuilt FlashAttention wheel, and a source-built xIELU extension. The atlas app itself is cloned into the container at runtime.

## Quick Start

```bash
docker pull juiceboxdocks/ml-workflow-image:latest

docker run --rm -it --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/prompts:/workspace/prompts" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/atlas:/workspace/atlas" \
  juiceboxdocks/ml-workflow-image:latest bash
```

Inside the container:

```bash
atlas-clone
atlas-run-vibethinker
```

The default scan runs:

```bash
python /workspace/atlasing/app.py \
  --model WeiboAI/VibeThinker-3B \
  --corpus /workspace/prompts/prompts.jsonl \
  --outdir /workspace/outputs/vibethinker-3b-census \
  --atlas /workspace/atlas/vibethinker-3b \
  --batch-size 8 \
  --max-length 128 \
  --components mlp,gate,up
```

Tune from there:

```bash
ATLAS_BATCH_SIZE=12 atlas-run-vibethinker --layers 0
ATLAS_BATCH_SIZE=12 atlas-run-vibethinker
```

## Runtime Layout

| Path | Purpose |
|---|---|
| `/workspace/atlasing` | Cloned GWIQ atlas app |
| `/workspace/prompts` | Prompt JSONL files |
| `/workspace/outputs` | Census and analysis outputs |
| `/workspace/atlas` | Final atlas directory |
| `/root/.cache/huggingface` | Hugging Face model cache |

`atlas-clone` defaults to:

```bash
git clone https://huggingface.co/spaces/juiceb0xc0de/atlasing /workspace/atlasing
```

Pass a different source or target if needed:

```bash
atlas-clone https://huggingface.co/spaces/juiceb0xc0de/atlasing /workspace/atlasing
```

## FlashAttention Anchor

The intentionally difficult packages are FlashAttention and xIELU. FlashAttention is still installed from a prebuilt wheel. xIELU is built from source during the image build, but only after the image has a real CUDA toolchain and C++ build chain available.

Installed stack:

| Package | Version |
|---|---|
| Base image | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| Python | Ubuntu 22.04 Python 3.10 |
| CUDA tooling | nvcc + headers from the CUDA devel base image |
| CMake | `cmake>=3.30` from pip |
| PyTorch | `torch==2.4.0` from CUDA 12.1 wheel index |
| xformers | `0.0.27.post2` |
| FlashAttention | prebuilt `2.8.3.post1+cu12torch2.4cxx11abiTRUE` wheel |
| xIELU | source build from `nickjbrowning/XIELU` |
| Transformers | `>=4.54` for `WeiboAI/VibeThinker-3B` |

FlashAttention-2 supports Ampere, Ada, and Hopper GPUs. The first target is Ada Lovelace, especially RTX 4090 class cards.

Note: the atlas extractor captures activations through model hooks. FlashAttention is included so the environment is ready for compatible training/inference paths, but a particular atlas scan only benefits directly if the model path dispatches through FlashAttention-compatible attention.

The xIELU build uses the same torch 2.4 / CUDA 12.8 anchor and is compiled with the C++11 ABI flag aligned to torch. If you need a strict source pin, set the `XIELU_REF` build arg in the Docker build.

## What's Installed

- PyTorch / torchvision / torchaudio
- xformers
- FlashAttention
- xIELU
- Transformers
- Accelerate
- Hugging Face Hub + `hf_transfer`
- NumPy, SciPy, pandas, scikit-learn
- matplotlib, seaborn
- orjson, tqdm, tokenizers, sentencepiece, protobuf

## What's Excluded

These are intentionally not installed:

- `sae-lens`
- `transformer-lens`
- `trl`
- `peft`
- `bitsandbytes`
- `llama.cpp`
- `llama-cpp-python`
- Jupyter/notebook tooling
- DeepSpeed, AutoAWQ, GPTQModel, Unsloth

Install extras later only after the atlas image is proven.

## RunPod

- **Image:** `juiceboxdocks/ml-workflow-image:latest`
- **GPU:** RTX 4090 / RTX 6000 Ada / A100 / H100
- **Env:** set `HF_TOKEN` for gated/private models or Space clones
- **Start command:** leave blank. The image starts `/usr/local/bin/start.sh`, prints runtime versions, and stays alive.
- **Workflow:** SSH into the pod, run `atlas-clone`, confirm prompts exist under `/workspace/prompts`, then run `atlas-run-vibethinker`.

Jupyter may be supplied by the RunPod template, but this image does not install it. The intended atlas workflow is CLI-first.

## Build

GitHub Actions builds on push to `main` and publishes:

- `juiceboxdocks/ml-workflow-image:latest`
- `juiceboxdocks/ml-workflow-image:cu128`

Required repo secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Local build:

```bash
docker buildx build --load -t gwiq-atlas-image:local .
```

Smoke import:

```bash
docker run --rm --gpus all gwiq-atlas-image:local bash -lc \
  "python -c 'import torch, flash_attn, transformers, xielu; print(torch.__version__, flash_attn.__version__, transformers.__version__, getattr(xielu, \"__version__\", \"source-build\"))'"
```

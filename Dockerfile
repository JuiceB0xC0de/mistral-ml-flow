# syntax=docker/dockerfile:1
###############################################################################
# GWIQ atlas GPU image
#
# Single-purpose runtime for cloning and running the atlas CLI on NVIDIA GPUs.
# Anchor: Python 3.10, torch 2.4.0 cu121, xformers 0.0.27.post2,
#         prebuilt flash-attn 2.8.3.post1 CUDA 12 wheel.
###############################################################################

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip \
      git curl ca-certificates build-essential ninja-build \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --upgrade pip setuptools wheel packaging psutil \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && ln -sf /usr/bin/python3 /usr/local/bin/python3 \
 && printf '#!/bin/sh\nexec /usr/bin/python3 -m pip "$@"\n' > /usr/local/bin/pip \
 && chmod +x /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

WORKDIR /workspace

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
      --index-url https://download.pytorch.org/whl/cu121

RUN pip install xformers==0.0.27.post2 \
      --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-build-isolation \
      "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.12/flash_attn-2.8.0+cu128torch2.4-cp310-cp310-linux_x86_64.whl"
RUN pip install \
      "numpy<2" \
      "transformers>=4.56" \
      "accelerate>=0.34" \
      "huggingface_hub>=0.19.0,<1.0.0" \
      "hf_transfer>=0.1.9" \
      "orjson>=3.10" \
      "pandas>=2.2" \
      "scikit-learn>=1.5" \
      "scipy>=1.13" \
      "matplotlib>=3.8" \
      "seaborn>=0.13" \
      "tqdm>=4.66" \
      "sentencepiece>=0.2.0" \
      "protobuf>=3.20.0" \
      "tokenizers>=0.15"

COPY scripts/ /usr/local/bin/
RUN chmod +x /usr/local/bin/start.sh \
      /usr/local/bin/atlas-clone \
      /usr/local/bin/atlas-run-vibethinker

RUN python -c "import torch, flash_attn, transformers, numpy, orjson, sklearn; \
      print('torch', torch.__version__); \
      print('flash-attn', flash_attn.__version__); \
      print('transformers', transformers.__version__); \
      print('cuda', torch.version.cuda, torch.cuda.is_available()); \
      assert torch.__version__.startswith('2.4'), torch.__version__"

CMD ["/usr/local/bin/start.sh"]

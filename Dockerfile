# syntax=docker/dockerfile:1
###############################################################################
# GWIQ atlas GPU image
#
# Single-purpose runtime for cloning and running the atlas CLI on NVIDIA GPUs.
# Anchor: Python 3.13, torch 2.12.1 cu130,
#         prebuilt flash-attn 2.8.3 CUDA 13 wheel,
#         xIELU built from source against the same torch/CUDA stack.
###############################################################################

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ARG XIELU_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.13 python3.13-dev \
      git curl ca-certificates build-essential ninja-build \
 && rm -rf /var/lib/apt/lists/* \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
 && python3.13 -m pip install --upgrade pip setuptools wheel packaging psutil "cmake>=3.30" nvtx \
 && ln -sf /usr/bin/python3.13 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.13 /usr/local/bin/python3 \
 && printf '#!/bin/sh\nexec python3.13 -m pip "$@"\n' > /usr/local/bin/pip \
 && chmod +x /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

WORKDIR /workspace

RUN pip install torch==2.12.1 \
      --index-url https://download.pytorch.org/whl/cu130

RUN pip install --no-build-isolation \
      "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp313-cp313-linux_x86_64.whl"

RUN CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
    CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    pip install --no-cache-dir --no-build-isolation --no-deps \
      "git+https://github.com/nickjbrowning/XIELU@${XIELU_REF}"

RUN pip install \
      "numpy==1.26.4" \
      "transformers==5.2.0" \
      "mistral-common==1.11.4" \
      "accelerate==1.14.0" \
      "huggingface_hub==1.16.1" \
      "hf_transfer==0.1.9" \
      "safetensors==0.8.0" \
      "orjson==3.11.9" \
      "pandas==3.0.3" \
      "scikit-learn==1.9.0" \
      "scipy==1.17.1" \
      "matplotlib==3.11.0" \
      "seaborn==0.13.2" \
      "tqdm==4.68.3" \
      "sentencepiece==0.2.1" \
      "protobuf==3.20.3" \
      "tokenizers==0.22.2"

COPY scripts/ /usr/local/bin/
RUN chmod +x /usr/local/bin/start.sh \
      /usr/local/bin/atlas-clone \
      /usr/local/bin/atlas-run-ministral \
      /usr/local/bin/atlas-run-vibethinker

RUN python -c "import torch, flash_attn, transformers, numpy, orjson, sklearn, xielu, mistral_common; \
      from transformers import AutoConfig; \
      AutoConfig.for_model('qwen3_5'); AutoConfig.for_model('mistral3'); \
      print('torch', torch.__version__); \
      print('flash-attn', flash_attn.__version__); \
      print('transformers', transformers.__version__); \
      print('mistral-common', mistral_common.__version__); \
      print('xielu', getattr(xielu, '__version__', 'source-build')); \
      print('cuda', torch.version.cuda, torch.cuda.is_available()); \
      assert torch.__version__.startswith('2.12'), torch.__version__"

CMD ["/usr/local/bin/start.sh"]

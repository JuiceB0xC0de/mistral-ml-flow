# syntax=docker/dockerfile:1
###############################################################################
# One-size ML GPU image — ANCHORED on the prebuilt flash-attn wheel.
# Anchor => Python 3.10 · torch 2.4.0 (pip/abiFALSE) · CUDA 12.x
# Covers: Ampere sm_80 · Ada sm_89 · Hopper sm_90
# Does NOT cover Blackwell: B200 sm_100, RTX Pro 6000 sm_120
###############################################################################

# Stage 1: prebuilt llama.cpp CUDA binaries compiled on CUDA 12.8 for sm_80 (A100)
# Built in HF Space juiceb0xc0de/llama.cpp-cu12.8-sm_80 and stored in HF model repo.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS llamacpp
ARG HF_WHEEL_REPO="juiceb0xc0de/llama-cpp-cu128-wheel"

# install only what's needed to fetch the binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/llama.cpp

# Pull every binary + shared lib from the HF model repo.
# (symlinks were flattened on upload; the versioned .so names are the real files).
RUN curl -fsSL -o libggml-base.so.0.15.2           "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libggml-base.so.0.15.2" && \
    curl -fsSL -o libggml-cpu.so.0.15.2            "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libggml-cpu.so.0.15.2" && \
    curl -fsSL -o libggml-cuda.so.0.15.2           "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libggml-cuda.so.0.15.2" && \
    curl -fsSL -o libggml.so.0.15.2                "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libggml.so.0.15.2" && \
    curl -fsSL -o libllama-cli-impl.so             "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libllama-cli-impl.so" && \
    curl -fsSL -o libllama-common.so.0.0.1         "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libllama-common.so.0.0.1" && \
    curl -fsSL -o libllama-quantize-impl.so        "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libllama-quantize-impl.so" && \
    curl -fsSL -o libllama-server-impl.so          "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libllama-server-impl.so" && \
    curl -fsSL -o libllama.so.0.0.1                "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libllama.so.0.0.1" && \
    curl -fsSL -o libmtmd.so.0.0.1                 "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/libmtmd.so.0.0.1" && \
    curl -fsSL -o llama-cli                        "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/llama-cli" && \
    curl -fsSL -o llama-embedding                  "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/llama-embedding" && \
    curl -fsSL -o llama-quantize                   "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/llama-quantize" && \
    curl -fsSL -o llama-server                     "https://huggingface.co/${HF_WHEEL_REPO}/resolve/main/bin/llama-server" && \
    chmod +x llama-cli llama-embedding llama-quantize llama-server

# Stage 2: main image
# CUDA 12.8.1 runtime matches the precompiled llama.cpp binaries (built on 12.8).
# torch 2.4 / the flash-attn wheel use torch's own bundled CUDA (cu121), which is fine on a 12.8 host.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_NO_CACHE_DIR=1

# ubuntu22.04 already ships Python 3.10 — no deadsnakes, no apt PPA dance
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip \
      git curl ca-certificates build-essential ninja-build cmake unzip \
 && curl https://rclone.org/install.sh | bash \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --upgrade pip setuptools wheel packaging

# llama.cpp: copy precompiled binaries + shared libs; expose on PATH and LD_LIBRARY_PATH.
COPY --from=llamacpp /opt/llama.cpp /opt/llama.cpp
ENV PATH="/opt/llama.cpp:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama.cpp:${LD_LIBRARY_PATH}"

WORKDIR /workspace

# ── ABI-critical trio: torch + matched xformers, THEN the flash-attn wheel. ──
#    These three are the ONLY hard-coupled versions. Everything below floats.
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
      --index-url https://download.pytorch.org/whl/cu121
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121

# the anchor — prebuilt, no compile, ~seconds
RUN pip install --no-build-isolation \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ── Core ML stack (torch-2.4 / py3.10 era) ──
RUN pip install \
      "numpy<2" \
      "transformers==4.46.3" \
      "accelerate>=0.34,<1.1" \
      "datasets>=2.20,<3.2" \
      tokenizers "sentencepiece>=0.2.0" tiktoken "protobuf>=3.20.0" \
      "peft>=0.12,<0.14" "trl>=0.11,<0.13"

# ── Quantization (LoRA/QLoRA) ──
RUN pip install "bitsandbytes>=0.44"

# ── Interpretability / SAE ──
RUN pip install transformer-lens sae-lens

# ── Hub / logging / storage / data-science / utils ──
RUN pip install \
      "huggingface_hub>=0.24" "hf_transfer>=0.1.9" boto3 \
      "wandb>=0.17" tensorboard \
      scipy pandas scikit-learn matplotlib seaborn umap-learn einops \
      tqdm rich jaxtyping jupyter ipywidgets pytest pytest-cov

# ── llama-cpp-python (CUDA; Ampere/Ada/Hopper only) ──
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;86;89;90" FORCE_CMAKE=1 \
      pip install llama-cpp-python

# DeepSpeed / autoawq / gptqmodel / unsloth: intentionally NOT installed.

# project code last (most-churned layer)
COPY . .

CMD ["bash"]

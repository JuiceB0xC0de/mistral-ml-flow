# syntax=docker/dockerfile:1
###############################################################################
# One-size ML GPU image — ANCHORED on the prebuilt flash-attn wheel.
# Anchor => Python 3.10 · torch 2.4.0 (pip/abiFALSE) · CUDA 12.x
# Covers: Ampere sm_80 · Ada sm_89 · Hopper sm_90
# Does NOT cover Blackwell: B200 sm_100, RTX Pro 6000 sm_120
###############################################################################

# Stage 1: prebuilt llama.cpp CUDA binaries (swap in your own HF-compiled ones if you prefer)
FROM ghcr.io/ggml-org/llama.cpp:full-cuda AS llamacpp

# Stage 2: main image
# CUDA 12.8.1 to match the ggml-org llama.cpp:full-cuda binaries (built on 12.8.1);
# still cu12 — torch 2.4 / the flash-attn wheel use torch's own bundled CUDA.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

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

# llama.cpp: ggml-org's prebuilt CUDA binaries + their .so libs both live in /app.
# Copy the whole dir; expose on PATH (binaries) and LD_LIBRARY_PATH (libllama/libggml).
COPY --from=llamacpp /app /opt/llama.cpp
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

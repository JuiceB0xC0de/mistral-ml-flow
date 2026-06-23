# GWIQ Atlas Docker Image Design

Date: 2026-06-22

## Goal

Replace the broad ML workstation image with a single-purpose NVIDIA GPU image for running the GWIQ atlas pipeline from the command line. The first target scan is `WeiboAI/VibeThinker-3B` on an Ada Lovelace GPU such as an RTX 4090.

## Non-Goals

- Do not build the old "ultimate ML image".
- Do not install `sae-lens`, `transformer-lens`, `trl`, `peft`, `bitsandbytes`, `llama.cpp`, Jupyter, or notebook tooling.
- Do not vendor or fork the original GWIQ code into this image repository.
- Do not compile FlashAttention from source during the Docker build.

## Source Code Boundary

The image provides the Python/CUDA runtime and helper scripts only. The atlas app code is cloned at runtime from the Hugging Face Space or other chosen source repo into `/workspace/atlasing`.

This keeps `ML-workflow-image` as the container recipe and leaves GWIQ/atlasing as its own source of truth.

## Base Image And GPU Stack

Use `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` with Ubuntu's Python 3.10.

Install:

- `torch==2.4.0`, `torchvision==0.19.0`, `torchaudio==2.4.0` from the PyTorch CUDA 12.1 wheel index.
- `xformers==0.0.27.post2` from the same PyTorch wheel index.
- Prebuilt `flash_attn==2.8.3.post1` wheel URL for CUDA 12, torch 2.4, Python 3.10, Linux x86_64, `cxx11abiTRUE`.
- nvcc and CUDA headers from the devel base image, plus `cmake>=3.30` from pip, so the xIELU extension can be built from source without fighting apt.
- `xielu` from the upstream GitHub source repo, compiled against the same torch 2.4 + CUDA 12.8 stack with the torch ABI flag aligned. Install it with `--no-deps`; otherwise its loose `torch>=2.0` dependency may replace the pinned torch/FlashAttention pair.

FlashAttention is intentionally kept as a prebuilt wheel. xIELU is the one source-built dependency and should compile inside the image, not at runtime.

## Atlas Runtime Dependencies

Install only the direct runtime dependencies needed by the atlas CLI:

- `transformers`
- `accelerate`
- `huggingface_hub`
- `hf_transfer`
- `numpy<2`
- `orjson`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `tokenizers`
- `sentencepiece`
- `protobuf`

`WeiboAI/VibeThinker-3B` currently states `transformers>=4.54.0`, so the implementation must test whether that version works with the torch 2.4 FlashAttention anchor. If it does not, prefer keeping the FlashAttention anchor and choose the newest compatible `transformers` version only after a failing build proves the conflict.

## Runtime Layout

- `/workspace` is the working directory.
- `/workspace/atlasing` is where the app is cloned.
- `/workspace/prompts` is for mounted or copied prompt JSONL files.
- `/workspace/outputs` is for census and analysis outputs.
- `/workspace/atlas` is for the final atlas.
- Hugging Face cache should be mountable at `/root/.cache/huggingface`.

## Helper Commands

Provide small shell helpers:

- `atlas-clone`: clone or refresh the atlas app source into `/workspace/atlasing`.
- `atlas-run-vibethinker`: run the default VibeThinker-3B atlas command once the app and prompts are present.
- `scripts/start.sh`: RunPod stay-alive script that prints Python, torch, CUDA, GPU visibility, FlashAttention version, and the canonical CLI command.

The default atlas run should be conservative:

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

Batch size is expected to be tuned upward on the actual GPU after a smoke run.

## Verification

Build-time smoke checks:

- Import `torch`, `flash_attn`, `transformers`, `numpy`, `orjson`, `sklearn`.
- Import `xielu` and confirm the CUDA toolchain is present (`cmake --version`, `nvcc --version`).
- Assert torch is still `2.4.x`.
- Print CUDA build version and `torch.cuda.is_available()`.
- Verify the helper scripts are executable.

Runtime smoke checks:

- `atlas-clone` succeeds in a container with network access.
- `python /workspace/atlasing/app.py --help` succeeds.
- On GPU hardware, FlashAttention imports and torch sees the GPU.
- A tiny one-layer atlas extraction on a small public model succeeds before running VibeThinker-3B.

## Open Risks

- `transformers>=4.54.0` may not support the torch 2.4/xformers/flash-attn anchor cleanly.
- VibeThinker-3B is Qwen2.5-family; the existing atlas layer inspection must recognize its module names.
- The atlas extractor captures activations via model hooks; FlashAttention may speed training and compatible attention paths, but the atlas scan itself may not directly use FlashAttention unless the model implementation dispatches to it.
- The first real VibeThinker run may need lower batch size on a 24 GB RTX 4090.

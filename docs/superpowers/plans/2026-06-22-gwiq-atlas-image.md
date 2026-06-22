# GWIQ Atlas Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broad ML Docker image with a focused GPU runtime for cloning and running the GWIQ atlas CLI against `WeiboAI/VibeThinker-3B`.

**Architecture:** The Docker repo owns only the CUDA/Python runtime and tiny helper commands. The atlas source remains in `~/atlasing` and is cloned/refreshed into `/workspace/atlasing` at runtime. A small companion update to `~/atlasing` removes dead SAE dependencies and documents the VibeThinker default run.

**Tech Stack:** Docker, NVIDIA CUDA 12.8 runtime, Python 3.10, PyTorch 2.4 CUDA 12.1 wheels, xformers 0.0.27.post2, prebuilt FlashAttention 2.8.3.post1 wheel, Transformers, Hugging Face Hub.

## Global Constraints

- Use `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`.
- Install FlashAttention from a prebuilt wheel; do not compile it.
- Exclude `sae-lens`, `transformer-lens`, `trl`, `peft`, `bitsandbytes`, `llama.cpp`, Jupyter, and notebook tooling.
- Do not vendor or fork GWIQ/atlasing into `ML-workflow-image`.
- Runtime atlas source path is `/workspace/atlasing`.
- Default model target is `WeiboAI/VibeThinker-3B`.
- Preserve unrelated dirty files such as `.DS_Store` and `~/atlasing/prompts/prompts.jsonl`.

---

## File Structure

- `Dockerfile`: Replace old workstation image with a minimal atlas GPU runtime.
- `scripts/start.sh`: Replace old startup logging with atlas-specific RunPod stay-alive guidance.
- `scripts/atlas-clone`: New helper to clone or refresh the atlas source into `/workspace/atlasing`.
- `scripts/atlas-run-vibethinker`: New helper to run the default VibeThinker atlas command.
- `README.md`: Rewrite for the atlas image, RunPod usage, FlashAttention anchor, and CLI workflow.
- `.github/workflows/docker.yml`: Keep existing DockerHub build flow, but update paths if new helper scripts are added.
- `/Users/chiggy/atlasing/requirements.txt`: Remove `sae-lens`, keep only direct atlas deps, and choose a Transformers pin compatible with VibeThinker testing.
- `/Users/chiggy/atlasing/README.md`: Add the VibeThinker default CLI command and note that Docker image users clone this app at runtime.

## Task 1: Update Atlasing App Dependencies And Usage

**Files:**
- Modify: `/Users/chiggy/atlasing/requirements.txt`
- Modify: `/Users/chiggy/atlasing/README.md`

**Interfaces:**
- Consumes: Existing `python app.py` runner in `/Users/chiggy/atlasing/app.py`.
- Produces: A Space-ready app dependency file without `sae-lens`, plus VibeThinker CLI docs for Rick to push to HF.

- [ ] **Step 1: Inspect dependency imports**

Run:

```bash
rg -n "sae_lens|transformer_lens|sentence_transformers|datasets|AutoModelForCausalLM|AutoTokenizer" /Users/chiggy/atlasing --glob '*.py'
```

Expected: no `sae_lens` import; if `sentence_transformers` and `datasets` are not imported, they can be removed from requirements too.

- [ ] **Step 2: Edit requirements**

Replace `/Users/chiggy/atlasing/requirements.txt` with:

```txt
numpy<2
orjson>=3.10
pandas>=2.2
scikit-learn>=1.5
scipy>=1.13
matplotlib>=3.8
seaborn>=0.13
accelerate>=0.34
torch>=2.4
transformers>=4.54
tqdm>=4.66
huggingface_hub>=0.24
hf_transfer>=0.1.9
sentencepiece>=0.2.0
protobuf>=3.20.0
tokenizers>=0.15
```

- [ ] **Step 3: Update README run section**

Add a VibeThinker command:

```bash
python app.py \
    --model WeiboAI/VibeThinker-3B \
    --corpus prompts/prompts.jsonl \
    --outdir outputs/vibethinker-3b-census \
    --atlas atlas/vibethinker-3b \
    --batch-size 8 \
    --max-length 128 \
    --components mlp,gate,up
```

Also add one sentence: "The Docker image does not vendor this app; clone or upload this Space repo into `/workspace/atlasing` inside the container."

- [ ] **Step 4: Validate docs and requirements**

Run:

```bash
rg -n "sae-lens|sae_lens|VibeThinker|/workspace/atlasing" /Users/chiggy/atlasing/requirements.txt /Users/chiggy/atlasing/README.md
```

Expected: `sae-lens` is absent from requirements; README contains `VibeThinker` and `/workspace/atlasing`.

## Task 2: Replace Dockerfile With Focused Atlas Runtime

**Files:**
- Modify: `/Users/chiggy/ML-workflow-image/Dockerfile`

**Interfaces:**
- Consumes: helper scripts from Task 3 copied via `COPY scripts/ /usr/local/bin/`.
- Produces: image with `python`, `torch`, `xformers`, `flash_attn`, and atlas runtime Python deps.

- [ ] **Step 1: Replace Dockerfile content**

Use a single-stage Dockerfile with these logical sections:

```dockerfile
# syntax=docker/dockerfile:1
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
 && printf '#!/bin/sh\nexec /usr/bin/python3 -m pip "$@"\n' > /usr/local/bin/pip \
 && chmod +x /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip /usr/local/bin/pip3

WORKDIR /workspace

RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
      --index-url https://download.pytorch.org/whl/cu121
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-build-isolation \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

RUN pip install \
      "numpy<2" \
      "transformers>=4.54" \
      "accelerate>=0.34" \
      "huggingface_hub>=0.24" \
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
RUN chmod +x /usr/local/bin/start.sh /usr/local/bin/atlas-clone /usr/local/bin/atlas-run-vibethinker

RUN python -c "import torch, flash_attn, transformers, numpy, orjson, sklearn; \
      print('torch', torch.__version__); \
      print('flash-attn', flash_attn.__version__); \
      print('transformers', transformers.__version__); \
      print('cuda', torch.version.cuda, torch.cuda.is_available()); \
      assert torch.__version__.startswith('2.4'), torch.__version__"

CMD ["/usr/local/bin/start.sh"]
```

- [ ] **Step 2: Syntax check Dockerfile**

Run:

```bash
docker buildx build --load -t gwiq-atlas-image:local .
```

Expected: build reaches the FlashAttention wheel install or completes. If `transformers>=4.54` conflicts with torch 2.4, capture the exact resolver error and decide whether to pin the newest compatible `transformers`.

## Task 3: Add Runtime Helper Scripts

**Files:**
- Modify: `/Users/chiggy/ML-workflow-image/scripts/start.sh`
- Create: `/Users/chiggy/ML-workflow-image/scripts/atlas-clone`
- Create: `/Users/chiggy/ML-workflow-image/scripts/atlas-run-vibethinker`

**Interfaces:**
- Produces: `atlas-clone [source] [target]` and `atlas-run-vibethinker [extra app.py args...]`.

- [ ] **Step 1: Write `atlas-clone`**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-https://huggingface.co/spaces/juiceb0xc0de/atlasing}"
TARGET="${2:-/workspace/atlasing}"

if [ -d "${TARGET}/.git" ]; then
  echo "[atlas-clone] refreshing ${TARGET}"
  git -C "${TARGET}" pull --ff-only
else
  echo "[atlas-clone] cloning ${SOURCE} -> ${TARGET}"
  rm -rf "${TARGET}"
  git clone "${SOURCE}" "${TARGET}"
fi
```

- [ ] **Step 2: Write `atlas-run-vibethinker`**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${ATLAS_APP_DIR:-/workspace/atlasing}"

if [ ! -f "${APP_DIR}/app.py" ]; then
  echo "[atlas-run-vibethinker] ${APP_DIR}/app.py not found. Run atlas-clone first." >&2
  exit 1
fi

exec python "${APP_DIR}/app.py" \
  --model WeiboAI/VibeThinker-3B \
  --corpus /workspace/prompts/prompts.jsonl \
  --outdir /workspace/outputs/vibethinker-3b-census \
  --atlas /workspace/atlas/vibethinker-3b \
  --batch-size "${ATLAS_BATCH_SIZE:-8}" \
  --max-length "${ATLAS_MAX_LENGTH:-128}" \
  --components "${ATLAS_COMPONENTS:-mlp,gate,up}" \
  "$@"
```

- [ ] **Step 3: Replace `start.sh`**

Use:

```bash
#!/usr/bin/env bash
set -e

log() { echo "[gwiq-atlas-image] $*"; }

log "stay-alive entrypoint started (PID $$)"
log "python: $(command -v python)"
log "python version: $(python --version 2>&1)"
log "torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
log "flash-attn: $(python -c 'import flash_attn; print(flash_attn.__version__)' 2>&1)"
log "cuda visible: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>&1)"
log "clone atlas app with: atlas-clone"
log "run default scan with: atlas-run-vibethinker"

exec tail -f /dev/null
```

- [ ] **Step 4: Validate shell syntax**

Run:

```bash
bash -n scripts/start.sh
bash -n scripts/atlas-clone
bash -n scripts/atlas-run-vibethinker
```

Expected: no output and exit code 0.

## Task 4: Rewrite Docker Repo README

**Files:**
- Modify: `/Users/chiggy/ML-workflow-image/README.md`

**Interfaces:**
- Consumes: final script names and Docker image name.
- Produces: user-facing usage docs for RunPod and local Docker.

- [ ] **Step 1: Replace README**

Include these sections:

- Title: `# ML-workflow-image`
- Purpose: single-purpose GWIQ atlas GPU image.
- Quick start with `docker run --gpus all`.
- Runtime flow: `atlas-clone`, mount prompts/cache, run `atlas-run-vibethinker`.
- FlashAttention anchor: CUDA 12.8 runtime, torch 2.4, prebuilt FA 2.8.3.post1, Ada/Ampere/Hopper support.
- Excluded packages list.
- RunPod notes: leave start command blank, use CLI, Jupyter is not installed by the image.

- [ ] **Step 2: Check old claims are gone**

Run:

```bash
rg -n "llama.cpp|transformer-lens|sae-lens|trl|peft|bitsandbytes|ultimate|workstation" README.md Dockerfile
```

Expected: matches only appear in the excluded/non-goals context, not as installed packages.

## Task 5: Final Verification

**Files:**
- Verify all changed files.

**Interfaces:**
- Produces: final confidence before user pushes.

- [ ] **Step 1: Check git status**

Run:

```bash
git -C /Users/chiggy/ML-workflow-image status --short
git -C /Users/chiggy/atlasing status --short
```

Expected: `ML-workflow-image` shows Docker/docs/script changes plus pre-existing `.DS_Store`; `atlasing` shows requirements/README plus pre-existing `prompts/prompts.jsonl`.

- [ ] **Step 2: Build image if Docker is available**

Run:

```bash
docker buildx build --load -t gwiq-atlas-image:local /Users/chiggy/ML-workflow-image
```

Expected: image builds. If network/build fails due sandbox or Docker daemon, rerun with approval if needed and report exact blocker.

- [ ] **Step 3: Run image smoke test if build succeeds**

Run:

```bash
docker run --rm --gpus all gwiq-atlas-image:local bash -lc \
  "python -c 'import torch, flash_attn, transformers; print(torch.__version__, flash_attn.__version__, transformers.__version__)' && atlas-run-vibethinker --help || true"
```

Expected: imports succeed. `atlas-run-vibethinker --help` may fail before clone with the intended "Run atlas-clone first" message.

- [ ] **Step 4: Do not commit unless Rick asks**

Leave changes staged/unstaged as appropriate for Rick to inspect and push. Preserve `.DS_Store` dirty state.

# mistral-ml-flow

Single-purpose NVIDIA GPU image for running the GWIQ **Mistral atlas** CLI from a
RunPod pod or any Docker host.

This is a CLI-first CUDA 13 runtime, not an "install every ML tool" workstation.
It ships a clean Python/PyTorch stack with a prebuilt FlashAttention wheel and a
source-built xIELU extension, plus `mistral-common` for Mistral/Ministral models.
The atlas app itself is cloned into the container at runtime from a Hugging Face
Space — it is not baked into the image.

- **Image:** `juiceboxdocks/mistral-ml-flow`
- **Tags:** `latest`, `cu130` (current), `cu128` (legacy alias → same CUDA 13 image)
- **Default model:** `mistralai/Ministral-3-3B-Base-2512`

## Quick Start

```bash
docker pull juiceboxdocks/mistral-ml-flow:latest

docker run --rm -it --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/prompts:/workspace/prompts" \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/atlas:/workspace/atlas" \
  juiceboxdocks/mistral-ml-flow:latest bash
```

Inside the container:

```bash
atlas-clone            # pulls the atlas app into /workspace/mistral-atlasing
atlas-run-ministral    # default scan: mistralai/Ministral-3-3B-Base-2512
```

The default `atlas-run-ministral` runs:

```bash
python /workspace/mistral-atlasing/app.py \
  --model mistralai/Ministral-3-3B-Base-2512 \
  --corpus /workspace/mistral-atlasing/prompts/prompts.jsonl \
  --outdir /workspace/outputs/ministral-3-3b-census \
  --atlas /workspace/atlas/ministral-3-3b \
  --batch-size 8 \
  --max-length 128 \
  --components mlp,gate,up
```

Everything is overridable by environment variable, e.g.:

```bash
ATLAS_BATCH_SIZE=12 ATLAS_MODEL_ID=mistralai/Ministral-3-3B-Base-2512 \
  atlas-run-ministral --layers 0
```

| Env var | Default |
|---|---|
| `ATLAS_MODEL_ID` | `mistralai/Ministral-3-3B-Base-2512` |
| `ATLAS_APP_DIR` | `/workspace/mistral-atlasing` |
| `ATLAS_CORPUS` | `<APP_DIR>/prompts/prompts.jsonl` |
| `ATLAS_OUTDIR` | `/workspace/outputs/ministral-3-3b-census` |
| `ATLAS_DB_DIR` | `/workspace/atlas/ministral-3-3b` |
| `ATLAS_BATCH_SIZE` | `8` |
| `ATLAS_MAX_LENGTH` | `128` |
| `ATLAS_COMPONENTS` | `mlp,gate,up` |

## Scripts

All live on `PATH` inside the container.

| Command | What it does |
|---|---|
| `start.sh` | Default entrypoint. Prints runtime versions (python, torch, flash-attn, xielu, CUDA) and stays alive (`tail -f /dev/null`). |
| `atlas-clone [src] [dst]` | Clones/refreshes the atlas app. Defaults: `https://huggingface.co/spaces/juiceb0xc0de/mistral-atlasing` → `/workspace/mistral-atlasing`. Uses `HF_TOKEN` for private/gated Spaces. |
| `atlas-run-ministral [args]` | Default Ministral-3-3B scan (see above). Extra args pass through to `app.py`. |
| `atlas-run-vibethinker [args]` | Same harness against `WeiboAI/VibeThinker-3B`. |
| `sync_pools push\|pull <layer> [bucket]` | Backblaze B2 sync of SAE pool batches (layer-1, layer, layer+1). Requires `rclone` plus `B2_ACCOUNT` / `B2_KEY` / `B2_BUCKET`. **`rclone` is not in the image — install it on the pod before using.** |

## Runtime Layout

| Path | Purpose |
|---|---|
| `/workspace/mistral-atlasing` | Cloned atlas app (`app.py`) |
| `/workspace/prompts` | Prompt JSONL files |
| `/workspace/outputs` | Census / analysis outputs |
| `/workspace/atlas` | Final atlas directories |
| `/root/.cache/huggingface` | Hugging Face model cache |

## Installed Stack

| Component | Version |
|---|---|
| Base image | `nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04` |
| Python | Ubuntu 22.04 Python 3.10 |
| CUDA tooling | nvcc + headers from the CUDA 13.0 devel base |
| PyTorch | `torch==2.11.0` (+ `torchvision==0.26.0`, `torchaudio==2.11.0`), cu130 wheel index |
| FlashAttention | prebuilt `flash_attn-2.8.3+cu130torch2.11` cp310 wheel |
| xIELU | source build from `nickjbrowning/XIELU` (`--no-deps`, C++11-ABI=0, arch `8.0;8.6;8.9;9.0`) |
| Transformers | `5.0.0rc0` |
| mistral-common | `>=1.8.6` |
| Accelerate | `>=0.34` |
| Hugging Face Hub | `>=1.0.0,<2.0.0` (+ `hf_transfer`) |
| Scientific | `numpy<2`, scipy, pandas, scikit-learn, matplotlib, seaborn |
| Serialization / text | orjson, tqdm, tokenizers, sentencepiece, protobuf |

FlashAttention-2 supports Ampere, Ada, and Hopper GPUs. The xIELU build is
arch-targeted at `8.0;8.6;8.9;9.0` (A100, A10/30xx, Ada/4090/6000 Ada, H100) and
installed with `--no-deps` so its loose `torch>=2.0` requirement cannot upgrade the
pinned torch/FlashAttention stack. Set the `XIELU_REF` build arg to pin a source rev.

> Note: the atlas extractor captures activations through model hooks.
> FlashAttention is present so the environment is ready for compatible
> training/inference paths, but a given scan only benefits if the model path
> dispatches through FlashAttention-compatible attention.

## What's Excluded

Intentionally not installed (no xformers, no notebook tooling, no quantization/
training extras): `xformers`, `sae-lens`, `transformer-lens`, `trl`, `peft`,
`bitsandbytes`, `llama.cpp`, `llama-cpp-python`, Jupyter/notebook, DeepSpeed,
AutoAWQ, GPTQModel, Unsloth. Add extras only after the image is proven.

## RunPod

- **Image:** `juiceboxdocks/mistral-ml-flow:latest`
- **Template:** `mlworkflowimage` (`n5y733j8pc`) — container disk 75 GB, volume 750 GB at `/workspace`
- **Ports:** `8080/http`, `8888/http`, `8000/http`, `22/tcp`, `22/udp`
- **GPU:** RTX 4090 / RTX 6000 Ada / A100 / H100
- **Env:** set `HF_TOKEN` for gated/private models or Space clones
- **Start command:** leave blank — the image runs `start.sh`, prints versions, and stays alive
- **Workflow:** SSH in → `atlas-clone` → confirm prompts under `/workspace/prompts` → `atlas-run-ministral`

### Pulling a private image

If `juiceboxdocks/mistral-ml-flow` is private on Docker Hub, RunPod needs
registry credentials or the pull fails with
`unauthorized: repository is private or does not exist`:

1. **Settings → Container Registry Auth → Add** — name it (e.g. `dockerhub`),
   username = your Docker Hub user, password = a Docker Hub **access token**.
2. Attach that auth to this template (or the pod) so `containerRegistryAuthId`
   is set, then start the pod.

Alternatively, make the Docker Hub repo public and no auth is needed.

> Make sure the pod/template reference the canonical name
> `juiceboxdocks/mistral-ml-flow` — not `mistral-ml-workflow` or
> `ml-workflow-image`, which are stale and will 404 on pull.

## Build

GitHub Actions builds on push to `main` (changes to `Dockerfile`, `scripts/**`,
or the workflow) and on manual dispatch, publishing:

- `juiceboxdocks/mistral-ml-flow:latest`
- `juiceboxdocks/mistral-ml-flow:cu130`
- `juiceboxdocks/mistral-ml-flow:cu128` (legacy alias)

Required repo secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (Docker Hub access token with push rights)

Local build + smoke import:

```bash
docker buildx build --load -t mistral-ml-flow:local .

docker run --rm --gpus all mistral-ml-flow:local bash -lc \
  "python -c 'import torch, flash_attn, transformers, xielu, mistral_common; \
   print(torch.__version__, flash_attn.__version__, transformers.__version__, \
   mistral_common.__version__, getattr(xielu, \"__version__\", \"source-build\"))'"
```

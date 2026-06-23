#!/usr/bin/env bash
set -e

log() { echo "[mistral-atlas-image] $*"; }

log "stay-alive entrypoint started (PID $$)"
log "python: $(command -v python)"
log "python version: $(python --version 2>&1)"
if command -v cmake >/dev/null 2>&1; then
  log "cmake: $(cmake --version 2>/dev/null | head -n1)"
else
  log "cmake: not found"
fi
if command -v nvcc >/dev/null 2>&1; then
  log "nvcc: $(nvcc --version 2>/dev/null | tail -n1)"
else
  log "nvcc: not found"
fi
log "CUDA_HOME: ${CUDA_HOME:-unset}"
log "torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
log "flash-attn: $(python -c 'import flash_attn; print(flash_attn.__version__)' 2>&1)"
if python -c 'import xielu' >/dev/null 2>&1; then
  log "xielu: $(python -c 'import xielu; print(getattr(xielu, \"__version__\", \"source-build\"))' 2>&1)"
else
  log "xielu: not importable"
fi
log "cuda visible: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>&1)"
log "clone atlas app with: atlas-clone"
log "run default scan with: atlas-run-ministral"

exec tail -f /dev/null

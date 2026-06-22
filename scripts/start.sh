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

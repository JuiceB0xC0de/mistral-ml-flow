#!/usr/bin/env bash
# Keep the RunPod container alive. RunPod's own init handles SSH/Jupyter injection.
set -e

log() { echo "[ml-workflow-image] $*"; }

log "stay-alive entrypoint started (PID $$)"
log "python: $(command -v python)"
log "python version: $(python --version 2>&1)"
log "torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
log "flash-attn: $(python -c 'import flash_attn; print(flash_attn.__version__)' 2>&1)"

# RunPod's init will start sshd/jupyter. We just keep the container alive.
exec tail -f /dev/null

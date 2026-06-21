#!/usr/bin/env bash
# Usage:
#   sync_pools push <layer> [bucket]   — upload layer-1, layer, layer+1 to B2
#   sync_pools pull <layer> [bucket]   — download them back on pod resume
#
# Env vars required on pod:
#   B2_ACCOUNT   — Backblaze B2 account ID (application key ID)
#   B2_KEY       — Backblaze B2 application key
#   B2_BUCKET    — bucket name (or pass as 3rd arg; default: sae-pool-batches)
set -euo pipefail

ACTION=${1:?Usage: sync_pools push|pull <layer> [bucket]}
LAYER=${2:?Layer index required}
BUCKET=${3:-${B2_BUCKET:-sae-pool-batches}}
POOL_DIR=${SAE_SCRATCH_DIR:-/workspace/rollcache}

export RCLONE_CONFIG_B2_TYPE=b2
export RCLONE_CONFIG_B2_ACCOUNT="${B2_ACCOUNT:?B2_ACCOUNT not set}"
export RCLONE_CONFIG_B2_KEY="${B2_KEY:?B2_KEY not set}"

for L in $((LAYER - 1)) "$LAYER" $((LAYER + 1)); do
    [ "$L" -lt 0 ] && continue
    LOCAL="$POOL_DIR/layer_$L"
    REMOTE="b2:$BUCKET/layer_$L"
    if [ "$ACTION" = "push" ]; then
        echo "Syncing $LOCAL → $REMOTE"
        rclone sync "$LOCAL" "$REMOTE" --progress --transfers 8
    elif [ "$ACTION" = "pull" ]; then
        echo "Syncing $REMOTE → $LOCAL"
        mkdir -p "$LOCAL"
        rclone sync "$REMOTE" "$LOCAL" --progress --transfers 8
    else
        echo "Unknown action: $ACTION (use push or pull)" >&2
        exit 1
    fi
done
echo "Done."

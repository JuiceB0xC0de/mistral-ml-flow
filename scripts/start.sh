#!/usr/bin/env bash
# Keep the RunPod container alive. RunPod's own init handles SSH/Jupyter injection.
set -e
echo "ML-workflow-image ready — container stay-alive started"
exec tail -f /dev/null

#!/usr/bin/env bash
# Project-local HF credentials on SSD; large caches on the CellWeft NAS.
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
resource_root="${CELLWEFT_DATA_ROOT:-/mnt/NAS_21T/ProjectData/CellWeft}"
export HF_HOME="$project_root/.cache/huggingface"
export HF_TOKEN_PATH="$HF_HOME/token"
export HF_HUB_CACHE="$resource_root/cache/huggingface/hub"
export HF_XET_CACHE="$resource_root/cache/huggingface/xet"
export HF_ASSETS_CACHE="$resource_root/cache/huggingface/assets"
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_TELEMETRY=1
umask 077
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_ASSETS_CACHE"
chmod 700 "$HF_HOME"
if [[ -f "$HF_HOME/network.env" ]]; then
    source "$HF_HOME/network.env"
fi
exec "$project_root/.venv-hf/bin/hf" "$@"

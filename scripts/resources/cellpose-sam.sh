#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${CELLWEFT_DATA_ROOT:-/mnt/NAS_21T/ProjectData/CellWeft}"

export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-${data_root}/resources/pretrained/cellpose}"
export CELLPOSE_LOCAL_CACHE="${CELLPOSE_LOCAL_CACHE:-${data_root}/cache/cellpose}"
export CELLPOSE_SAM_MODEL="${CELLPOSE_SAM_MODEL:-${data_root}/resources/pretrained/cellpose/cpsam_v2/checkpoint}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${data_root}/cache/matplotlib}"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$CELLPOSE_LOCAL_CACHE" "$MPLCONFIGDIR"

exec "$project_root/.venv-cellpose-sam/bin/python" "$@"

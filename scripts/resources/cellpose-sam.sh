#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${CELLWEFT_DATA_ROOT:-/mnt/NAS_21T/ProjectData/CellWeft}"
scratch_root="${CELLWEFT_SCRATCH_ROOT:-${project_root}/.cache/cellweft/runtime}"

export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-${data_root}/raw/shared/pretrained/cellpose}"
export CELLPOSE_LOCAL_CACHE="${CELLPOSE_LOCAL_CACHE:-${scratch_root}/cellpose}"
export CELLPOSE_SAM_MODEL="${CELLPOSE_SAM_MODEL:-${data_root}/raw/shared/pretrained/cellpose/cpsam_v2/checkpoint}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${scratch_root}/matplotlib}"
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH" "$CELLPOSE_LOCAL_CACHE" "$MPLCONFIGDIR"

exec "$project_root/.venv-cellpose-sam/bin/python" "$@"

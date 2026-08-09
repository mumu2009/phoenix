#!/usr/bin/env bash
set -euo pipefail

root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="$root/runtime_store/rdk_x5_launcher.json"
: "${HOST_LLAMA_SERVER_URL:?set HOST_LLAMA_SERVER_URL to the host llama-server URL}"
: "${JEPA_IMAGE_HORIZON_MODEL:=$root/runtime_store/models/ijepa/ijepa_vith14_1k/model.bin}"
: "${JEPA_IMAGE_INPUT_COLOR:=bgr}"

if [[ ! -x "$root/phoenix_main" ]]; then
    printf 'missing executable: run tools/build_rdk_x5.sh on the RDK X5 first\n' >&2
    exit 1
fi
if [[ ! -f "$JEPA_IMAGE_HORIZON_MODEL" ]]; then
    printf 'missing compiled Horizon JEPA model: %s\n' "$JEPA_IMAGE_HORIZON_MODEL" >&2
    exit 1
fi
if [[ ! -e /dev/bpu && ! -e /dev/bpu_core0 ]]; then
    printf 'RDK X5 BPU device nodes are unavailable\n' >&2
    exit 1
fi

export AI_CONFIG_FILE="$config"
export AI_LLAMACPP_BASE_URL="$HOST_LLAMA_SERVER_URL"
export JEPA_IMAGE_BACKEND=horizon-hbdnn
export JEPA_IMAGE_HORIZON_MODEL
export JEPA_IMAGE_INPUT_COLOR
export JEPA_IMAGE_VARIANT=ijepa_vith14_1k
export JEPA_IMAGE_CONCEPT_DIM=128
export JEPA_CAMERA_DEVICE="${JEPA_CAMERA_DEVICE:-/dev/video0}"
export JEPA_CAMERA_WIDTH="${JEPA_CAMERA_WIDTH:-1920}"
export JEPA_CAMERA_HEIGHT="${JEPA_CAMERA_HEIGHT:-1080}"
export JEPA_CAMERA_FPS="${JEPA_CAMERA_FPS:-30}"
exec "$root/phoenix_main"

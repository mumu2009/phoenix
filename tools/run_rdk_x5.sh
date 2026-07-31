#!/usr/bin/env bash
set -euo pipefail

root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
config="$root/runtime_store/rdk_x5_launcher.json"
: "${HOST_LLAMA_SERVER_URL:?set HOST_LLAMA_SERVER_URL to the host llama-server URL}"
: "${JPEA_IMAGE_HORIZON_MODEL:=$root/runtime_store/models/ijepa/ijepa_vith14_1k/model.bin}"
: "${JPEA_IMAGE_INPUT_COLOR:=bgr}"

if [[ ! -x "$root/phoenix_main" ]]; then
    printf 'missing executable: run tools/build_rdk_x5.sh on the RDK X5 first\n' >&2
    exit 1
fi
if [[ ! -f "$JPEA_IMAGE_HORIZON_MODEL" ]]; then
    printf 'missing compiled Horizon JPEA model: %s\n' "$JPEA_IMAGE_HORIZON_MODEL" >&2
    exit 1
fi
if [[ ! -e /dev/bpu && ! -e /dev/bpu_core0 ]]; then
    printf 'RDK X5 BPU device nodes are unavailable\n' >&2
    exit 1
fi

export AI_CONFIG_FILE="$config"
export AI_LLAMACPP_BASE_URL="$HOST_LLAMA_SERVER_URL"
export JPEA_IMAGE_BACKEND=horizon-hbdnn
export JPEA_IMAGE_HORIZON_MODEL
export JPEA_IMAGE_INPUT_COLOR
export JPEA_IMAGE_VARIANT=ijepa_vith14_1k
export JPEA_IMAGE_CONCEPT_DIM=128
export JPEA_CAMERA_DEVICE="${JPEA_CAMERA_DEVICE:-/dev/video0}"
export JPEA_CAMERA_WIDTH="${JPEA_CAMERA_WIDTH:-1920}"
export JPEA_CAMERA_HEIGHT="${JPEA_CAMERA_HEIGHT:-1080}"
export JPEA_CAMERA_FPS="${JPEA_CAMERA_FPS:-30}"
exec "$root/phoenix_main"

#!/usr/bin/env bash
# Run compile_bpu.sh inside the OpenExplorer Docker container.
#
# This is the default compile driver for the additive residual BPU evolution
# controller on Kali (where hb_mapper is only available inside Docker).
#
# Usage:
#   bash compile_bpu_docker.sh \
#       --model-name speech_decoder \
#       --onnx /home/kali/phoenix/additive/.../model.onnx \
#       --calib-dir /home/kali/phoenix/additive/.../calibration \
#       --input-name concept \
#       --input-shape 1x128x1x1 \
#       --out-dir /home/kali/phoenix/additive/.../bpu \
#       --per-channel False \
#       --calib-type max

set -euo pipefail

DOCKER_IMAGE="${DOCKER_IMAGE:-openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310}"
HOST_WORK="${HOST_WORK:-/home/kali/phoenix}"
CONTAINER_WORK="/workspace"

# Rewrite all absolute paths under HOST_WORK to the container mount.
args=()
for a in "$@"; do
    if [[ "$a" == "$HOST_WORK"* ]]; then
        a="${a/#$HOST_WORK/$CONTAINER_WORK}"
    fi
    args+=("$a")
done

# Force the in-container run_hb_mapper path.
args+=("--run-hb-mapper" "$CONTAINER_WORK/tools/run_hb_mapper.py")

exec docker run --rm -v "$HOST_WORK:$CONTAINER_WORK" "$DOCKER_IMAGE" \
    bash "$CONTAINER_WORK/tools/compile_bpu.sh" "${args[@]}"

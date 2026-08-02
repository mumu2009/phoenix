#!/usr/bin/env bash
# Compile the small speech autoencoder ONNX to Horizon .bin.
# Designed to run inside the OpenExplorer Docker container with
# /workspace mounted to the project/output directory.
#
# Example (run inside Docker):
#   bash /workspace/tools/compile_bpu_speech_small.sh \
#        /workspace/speech_onnx /workspace/speech_bin

set -uo pipefail

ONNX_DIR="${1:-/workspace/speech_onnx}"
OUT_DIR="${2:-/workspace/speech_bin}"
MARCH="${MARCH:-bayes-e}"

mkdir -p "$OUT_DIR" "$OUT_DIR/mapper_work"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

fix_onnx_ir() {
    local onnx_path="$1"
    # hb_mapper 1.24.3 only supports IR version up to 9.
    python3 - <<PY
import onnx
m = onnx.load('$onnx_path')
m.ir_version = 8
onnx.save(m, '$onnx_path')
PY
    log "Downgraded IR version for $onnx_path"
}

compile_onnx() {
    local name="$1"
    local onnx_path="$2"
    local calib_dir="$3"
    local in_name="$4"
    local in_shape="$5"
    local per_channel="$6"
    local calib_type="${7:-max}"

    local config_file="$OUT_DIR/${name}_config.yaml"
    local mapper_out="$OUT_DIR/mapper_work/${name}"
    mkdir -p "$mapper_out"

    cat > "$config_file" <<EOF
model_parameters:
  onnx_model: '$onnx_path'
  march: '$MARCH'
  output_model_file_prefix: '${name}_${MARCH}'
  working_dir: '$mapper_out'
  layer_out_dump: False

input_parameters:
  input_name: '$in_name'
  input_shape: '$in_shape'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  input_type_rt: 'featuremap'
  input_layout_rt: 'NHWC'
  norm_type: 'no_preprocess'

calibration_parameters:
  cal_data_dir: '$calib_dir'
  cal_data_type: 'float32'
  calibration_type: '$calib_type'
  per_channel: $per_channel
EOF

    log "Compiling $name ($onnx_path) with march=$MARCH"
    python3 /workspace/tools/run_hb_mapper.py makertbin \
        --config "$config_file" --model-type onnx

    local bin_file
    bin_file=$(find "$mapper_out" -maxdepth 2 -name '*.bin' | head -n 1)
    if [[ ! -f "$bin_file" ]]; then
        die "No .bin produced for $name in $mapper_out"
    fi
    cp "$bin_file" "$OUT_DIR/${name}.bin"
    cp "$onnx_path" "$OUT_DIR/${name}.onnx"
    log "Wrote $OUT_DIR/${name}.bin"
}

[[ -f "$ONNX_DIR/model_encoder.onnx" ]] || die "Encoder ONNX not found: $ONNX_DIR/model_encoder.onnx"
[[ -f "$ONNX_DIR/model_decoder.onnx" ]] || die "Decoder ONNX not found: $ONNX_DIR/model_decoder.onnx"
[[ -d "$ONNX_DIR/calibration_encoder" ]] || die "Encoder calibration not found"
[[ -d "$ONNX_DIR/calibration_decoder" ]] || die "Decoder calibration not found"

fix_onnx_ir "$ONNX_DIR/model_encoder.onnx"
fix_onnx_ir "$ONNX_DIR/model_decoder.onnx"

compile_onnx "model_encoder" "$ONNX_DIR/model_encoder.onnx" "$ONNX_DIR/calibration_encoder" "waveform" "1x1x1x16000" "True" "kl"
compile_onnx "model_decoder" "$ONNX_DIR/model_decoder.onnx" "$ONNX_DIR/calibration_decoder" "concept" "1x32x1x1" "False"

log "Done: $OUT_DIR/model_encoder.bin $OUT_DIR/model_decoder.bin"

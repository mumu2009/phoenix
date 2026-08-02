#!/usr/bin/env bash
# Compile a JPEA-v2 speech or vision autoencoder ONNX pair to Horizon BPU .bin.
#
# Designed to run inside the OpenExplorer Docker container with the project
# mounted at /workspace, but can also be run from the Kali host if hb_mapper
# is in PATH and /workspace/tools/run_hb_mapper.py exists.
#
# Examples (inside Docker):
#   bash /workspace/tools/compile_bpu_jepa_v2.sh \
#        --modality speech --onnx-dir /workspace/speech_onnx \
#        --out-dir /workspace/speech_bin
#
#   bash /workspace/tools/compile_bpu_jepa_v2.sh \
#        --modality image --onnx-dir /workspace/image_onnx \
#        --out-dir /workspace/image_bin \
#        --concept 128 --resolution 224
#
# Generic single-model mode (new) for the additive residual BPU framework:
#   bash /workspace/tools/compile_bpu_jepa_v2.sh \
#        --model-name speech_encoder \
#        --onnx /workspace/additive/speech_encoder/model.onnx \
#        --calib-dir /workspace/additive/speech_encoder/calibration \
#        --input-name waveform \
#        --input-shape 1x1x1x16000 \
#        --out-dir /workspace/additive/speech_encoder/bin

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODALITY="speech"
ONNX_DIR=""
OUT_DIR=""
CONCEPT=128
RESOLUTION=224
CHUNK=16000
DECODER_OUTPUT=15872
MARCH="${MARCH:-bayes-e}"
RUN_HB_MAPPER="${RUN_HB_MAPPER:-/workspace/tools/run_hb_mapper.py}"

# Generic single-model parameters
MODEL_NAME=""
ONNX=""
CALIB_DIR=""
INPUT_NAME=""
INPUT_SHAPE=""
PER_CHANNEL="True"
CALIB_TYPE="kl"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --modality)
      MODALITY="$2"; shift 2 ;;
    --onnx-dir)
      ONNX_DIR="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --concept)
      CONCEPT="$2"; shift 2 ;;
    --resolution)
      RESOLUTION="$2"; shift 2 ;;
    --march)
      MARCH="$2"; shift 2 ;;
    --run-hb-mapper)
      RUN_HB_MAPPER="$2"; shift 2 ;;
    # Generic mode arguments
    --model-name)
      MODEL_NAME="$2"; shift 2 ;;
    --onnx)
      ONNX="$2"; shift 2 ;;
    --calib-dir)
      CALIB_DIR="$2"; shift 2 ;;
    --input-name)
      INPUT_NAME="$2"; shift 2 ;;
    --input-shape)
      INPUT_SHAPE="$2"; shift 2 ;;
    --per-channel)
      PER_CHANNEL="$2"; shift 2 ;;
    --calib-type)
      CALIB_TYPE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,45p' "$0"; exit 0 ;;
    -*)
      echo "Unknown option: $1" >&2; exit 1 ;;
    *)
      POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

[[ -n "$OUT_DIR" ]] && mkdir -p "$OUT_DIR" "$OUT_DIR/mapper_work"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

fix_onnx_ir() {
    local onnx_path="$1"
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

    local config_file="$OUT_DIR/${name}_hb_mapper_config.yaml"
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
    if [[ -f "$RUN_HB_MAPPER" ]]; then
        python3 "$RUN_HB_MAPPER" makertbin --config "$config_file" --model-type onnx
    elif command -v hb_mapper >/dev/null 2>&1; then
        hb_mapper makertbin --config "$config_file" --model-type onnx
    else
        die "Cannot find $RUN_HB_MAPPER or hb_mapper. Are you inside the OpenExplorer Docker?"
    fi

    local bin_file
    bin_file=$(find "$mapper_out" -maxdepth 2 -name '*.bin' | head -n 1)
    if [[ ! -f "$bin_file" ]]; then
        die "No .bin produced for $name in $mapper_out"
    fi
    cp "$bin_file" "$OUT_DIR/${name}.bin"
    cp "$onnx_path" "$OUT_DIR/${name}.onnx"
    # config_file is already created inside OUT_DIR, no need to copy.
    log "Wrote $OUT_DIR/${name}.bin"
}

# ---------------------------------------------------------------------------
# Generic single-model mode
# ---------------------------------------------------------------------------
if [[ -n "$MODEL_NAME" && -n "$ONNX" ]]; then
    [[ -f "$ONNX" ]] || die "ONNX not found: $ONNX"
    [[ -n "$OUT_DIR" ]] || die "--out-dir is required"
    [[ -d "$CALIB_DIR" ]] || die "Calibration dir not found: $CALIB_DIR"
    [[ -n "$INPUT_NAME" ]] || die "--input-name is required in generic mode"
    [[ -n "$INPUT_SHAPE" ]] || die "--input-shape is required in generic mode"

    mkdir -p "$OUT_DIR" "$OUT_DIR/mapper_work"
    fix_onnx_ir "$ONNX"
    compile_onnx "$MODEL_NAME" "$ONNX" "$CALIB_DIR" "$INPUT_NAME" "$INPUT_SHAPE" "$PER_CHANNEL" "$CALIB_TYPE"

    # Copy the manifest and any sidecars from the ONNX directory.
    onnx_dir="$(dirname "$ONNX")"
    cp "$onnx_dir/model.manifest.json" "$OUT_DIR/" 2>/dev/null || true
    cp "$onnx_dir/encoder_head.json" "$OUT_DIR/" 2>/dev/null || true
    cp "$onnx_dir/decoder_inverse_matrix.json" "$OUT_DIR/" 2>/dev/null || true
    cp "$onnx_dir/model_encoder_head.onnx" "$OUT_DIR/" 2>/dev/null || true

    log "Done: $OUT_DIR/${MODEL_NAME}.bin"
    exit 0
fi

# ---------------------------------------------------------------------------
# Legacy speech/image pair mode
# ---------------------------------------------------------------------------
if [[ -z "$ONNX_DIR" || -z "$OUT_DIR" ]]; then
  echo "Usage: $0 --modality speech|image --onnx-dir <dir> --out-dir <dir> [--concept N] [--resolution N]" >&2
  echo "   or: $0 --model-name <name> --onnx <file> --calib-dir <dir> --input-name <name> --input-shape <shape> --out-dir <dir>" >&2
  exit 1
fi

[[ -f "$ONNX_DIR/model_encoder.onnx" ]] || die "Encoder ONNX not found: $ONNX_DIR/model_encoder.onnx"
[[ -f "$ONNX_DIR/model_decoder.onnx" ]] || die "Decoder ONNX not found: $ONNX_DIR/model_decoder.onnx"
[[ -d "$ONNX_DIR/calibration_encoder" ]] || die "Encoder calibration not found in $ONNX_DIR"
[[ -d "$ONNX_DIR/calibration_decoder" ]] || die "Decoder calibration not found in $ONNX_DIR"

mkdir -p "$OUT_DIR" "$OUT_DIR/mapper_work"
fix_onnx_ir "$ONNX_DIR/model_encoder.onnx"
fix_onnx_ir "$ONNX_DIR/model_decoder.onnx"

if [[ "$MODALITY" == "speech" ]]; then
    compile_onnx "model_encoder" "$ONNX_DIR/model_encoder.onnx" "$ONNX_DIR/calibration_encoder" \
                 "waveform" "1x1x1x${CHUNK}" "True" "kl"
    compile_onnx "model_decoder" "$ONNX_DIR/model_decoder.onnx" "$ONNX_DIR/calibration_decoder" \
                 "concept" "1x${CONCEPT}x1x1" "False" "max"
elif [[ "$MODALITY" == "image" ]]; then
    compile_onnx "model_encoder" "$ONNX_DIR/model_encoder.onnx" "$ONNX_DIR/calibration_encoder" \
                 "pixel_values" "1x3x${RESOLUTION}x${RESOLUTION}" "True" "kl"
    compile_onnx "model_decoder" "$ONNX_DIR/model_decoder.onnx" "$ONNX_DIR/calibration_decoder" \
                 "concept" "1x${CONCEPT}x1x1" "False" "max"
else
    die "Unsupported modality: $MODALITY (use speech or image)"
fi

cp "$ONNX_DIR/model.manifest.json" "$OUT_DIR/" 2>/dev/null || true
# Copy optional CPU/JSON sidecars for the image head.
cp "$ONNX_DIR/encoder_head.json" "$OUT_DIR/" 2>/dev/null || true
cp "$ONNX_DIR/decoder_inverse_matrix.json" "$OUT_DIR/" 2>/dev/null || true
cp "$ONNX_DIR/model_encoder_head.onnx" "$OUT_DIR/" 2>/dev/null || true

log "Done: $OUT_DIR/model_encoder.bin $OUT_DIR/model_decoder.bin"

#!/usr/bin/env python3
"""One-shot ONNX runner for Phoenix local x86_64 / CPU / GPU backends.

Loads a single ONNX model, reads a float32 input binary, runs inference, and
writes the float32 output binary.  Communication with the C++ side is through
stdout JSON plus temporary binary files.

Example:
    python tools/local_onnx_runner.py \
        --model runtime_store/models/additive_jpea/vision_encoder/best.onnx \
        --input /tmp/in.tensor --input-name pixel_values --input-shape 1x3x224x224 \
        --output /tmp/out.tensor --output-name concept --output-shape 1x128x1x1
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:  # pragma: no cover - runner requires onnxruntime
    print(json.dumps({"ok": False, "error": f"onnxruntime not available: {e}"}))
    sys.exit(1)


def parse_shape(s: str):
    return tuple(int(x) for x in s.split("x"))


def run(args) -> dict:
    model_path = Path(args.model)
    if not model_path.is_file():
        return {"ok": False, "error": f"model not found: {args.model}"}

    input_path = Path(args.input)
    output_path = Path(args.output)

    in_arr = np.fromfile(input_path, dtype=np.float32)
    in_shape = parse_shape(args.input_shape)
    if in_arr.size != np.prod(in_shape):
        return {
            "ok": False,
            "error": (
                f"input size mismatch: expected {np.prod(in_shape)} "
                f"floats, got {in_arr.size}"
            ),
        }
    in_arr = in_arr.reshape(in_shape)

    providers = ["CPUExecutionProvider"]
    if args.gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        return {"ok": False, "error": f"failed to load ONNX: {e}"}

    # Resolve input/output names if not supplied.
    input_name = args.input_name
    output_name = args.output_name
    if not input_name:
        for inp in sess.get_inputs():
            input_name = inp.name
            break
    if not output_name:
        for out in sess.get_outputs():
            output_name = out.name
            break

    try:
        outputs = sess.run([output_name], {input_name: in_arr})
    except Exception as e:
        return {"ok": False, "error": f"inference failed: {e}"}

    out_arr = np.asarray(outputs[0], dtype=np.float32)
    out_shape = parse_shape(args.output_shape)
    if out_arr.size != np.prod(out_shape):
        return {
            "ok": False,
            "error": (
                f"output size mismatch: expected {np.prod(out_shape)} "
                f"floats, got {out_arr.size}"
            ),
        }
    out_arr = out_arr.reshape(out_shape)
    out_arr.tofile(output_path)

    return {
        "ok": True,
        "inputShape": list(in_shape),
        "outputShape": list(out_shape),
        "outputName": output_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot local ONNX runner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-name", default="")
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-name", default="")
    parser.add_argument("--output-shape", required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

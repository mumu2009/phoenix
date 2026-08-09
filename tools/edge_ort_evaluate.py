#!/usr/bin/env python3
"""Generic ONNX Runtime evaluator for non-Horizon edge devices.

Runs on RK3588, Jetson Nano, or any Linux device with an ONNX model.
If the accelerated backend (RKNN/TensorRT) is not available, it falls back
 to CPU ONNX Runtime.

Usage:
    python3 edge_ort_evaluate.py \
        --bin-dir /path/to/bins \
        --inputs inputs.bin --targets targets.bin \
        --input-shape 1x1x1x16000 --target-shape 1x1x1x15872 \
        --format bin --out /path/to/losses.json
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from edge_evaluate_common import evaluate_batch_mse, find_model_variant, load_batch, parse_shape

try:
    import onnxruntime as ort
except Exception as e:
    print(f"[ERROR] cannot import onnxruntime: {e}", file=sys.stderr)
    sys.exit(1)


def find_model(model_path: Path):
    """Pick the best available model: .rknn, .trt, .onnx, or .bin."""
    return find_model_variant(model_path, (".rknn", ".trt", ".onnx", ".bin"))


def evaluate_onnx(model_path: str, inputs: np.ndarray, targets: np.ndarray) -> float:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    return evaluate_batch_mse(inputs, targets, lambda inp: sess.run([out_name], {in_name: inp})[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX Runtime edge evaluator")
    parser.add_argument("--bin-dir", help="Directory with models")
    parser.add_argument("--bin", help="Single model to evaluate")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", default="/tmp/edge_ort_evaluate.json")
    parser.add_argument("--pattern", default="*", help="glob pattern")
    parser.add_argument("--input-shape", default=None)
    parser.add_argument("--target-shape", default=None)
    parser.add_argument("--format", choices=["auto", "npy", "bin"], default="auto")
    args = parser.parse_args()

    input_shape = parse_shape(args.input_shape)
    target_shape = parse_shape(args.target_shape)
    try:
        inputs = load_batch(args.inputs, input_shape, args.format)
        targets = load_batch(args.targets, target_shape, args.format)
    except Exception as e:
        print(f"[ERROR] failed to load batch: {e}", file=sys.stderr)
        return 1

    if args.bin:
        bin_paths = [args.bin]
    else:
        bin_dir = Path(args.bin_dir)
        if not bin_dir.is_dir():
            print(f"[ERROR] {bin_dir} is not a directory", file=sys.stderr)
            return 1
        bin_paths = sorted(glob.glob(str(bin_dir / args.pattern)))
        if not bin_paths:
            print(f"[ERROR] no models in {bin_dir}", file=sys.stderr)
            return 1

    results = {}
    for bin_path in bin_paths:
        p = Path(bin_path)
        name = p.name
        model = find_model(p) if p.is_dir() else p
        if model is None:
            model = find_model(p.with_suffix(""))
        if model is None or not model.is_file():
            results[name] = {"loss": float("inf"), "ok": False, "error": "no model found"}
            print(f"{name}: no model found")
            continue
        try:
            loss = evaluate_onnx(str(model), inputs, targets)
            results[name] = {"loss": loss, "ok": True}
            print(f"{name}: loss={loss:.6f}")
        except Exception as e:
            results[name] = {"loss": float("inf"), "ok": False, "error": str(e)}
            print(f"{name}: ERROR {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[DONE] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

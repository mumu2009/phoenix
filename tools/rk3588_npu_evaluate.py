#!/usr/bin/env python3
"""RK3588 NPU batch evaluator.

Uses RKNN Lite on the device for .rknn models and falls back to ONNX Runtime
if no .rknn is available. The loss format is identical to x5_bpu_evaluate.py so
bpu_evolve can consume it directly.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from edge_evaluate_common import evaluate_batch_mse, find_model_variant, load_batch, parse_shape


def _load_rknn_runtime():
    try:
        from rknnlite.api import RKNNLite
        return RKNNLite
    except Exception:
        try:
            from rknn.api import RKNN
            return RKNN
        except Exception as e:
            raise RuntimeError("Neither rknnlite nor rknn is installed") from e


def evaluate_rknn(model_path: str, inputs: np.ndarray, targets: np.ndarray):
    Runtime = _load_rknn_runtime()
    rknn = Runtime()
    rknn.load_rknn(model_path)
    try:
        from rknnlite.api import RKNNLite
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    except Exception:
        rknn.init_runtime()

    try:
        return evaluate_batch_mse(inputs, targets, lambda inp: rknn.inference(inputs=[inp])[0])
    finally:
        rknn.release()


def evaluate_onnx(model_path: str, inputs: np.ndarray, targets: np.ndarray):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    return evaluate_batch_mse(inputs, targets, lambda inp: sess.run([out_name], {in_name: inp})[0])


def find_model(model_dir: Path, model_path: Path):
    return find_model_variant(model_path, (".rknn", ".onnx"))


def main() -> int:
    parser = argparse.ArgumentParser(description="RK3588 NPU batch evaluator")
    parser.add_argument("--bin-dir", help="Directory with .rknn/.onnx models")
    parser.add_argument("--bin", help="Single .rknn/.onnx model")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", default="/tmp/rk3588_evaluate.json")
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
        model_paths = [Path(args.bin)]
    else:
        bin_dir = Path(args.bin_dir)
        if not bin_dir.is_dir():
            print(f"[ERROR] {bin_dir} is not a directory", file=sys.stderr)
            return 1
        model_paths = sorted([Path(p) for p in glob.glob(str(bin_dir / args.pattern))])
        if not model_paths:
            print(f"[ERROR] no models in {bin_dir}", file=sys.stderr)
            return 1

    results = {}
    for model_path in model_paths:
        name = model_path.name
        real_model = find_model(model_path.parent, model_path) if not model_path.is_file() else model_path
        if real_model is None or not real_model.is_file():
            results[name] = {"loss": float("inf"), "ok": False, "error": "no model"}
            print(f"{name}: no model found")
            continue
        try:
            if real_model.suffix == ".rknn":
                loss = evaluate_rknn(str(real_model), inputs, targets)
            else:
                loss = evaluate_onnx(str(real_model), inputs, targets)
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

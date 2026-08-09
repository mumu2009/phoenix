#!/usr/bin/env python3
"""Jetson TensorRT / ONNX Runtime batch evaluator.

Tries to use a pre-built .trt engine; if none exists, falls back to
onnxruntime with the TensorRT / CUDA execution providers on the Jetson.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

from edge_evaluate_common import evaluate_batch_mse, find_model_variant, load_batch, parse_shape


def _load_engine(trt_path: str):
    try:
        import tensorrt as trt
    except Exception as e:
        raise RuntimeError("tensorrt not installed") from e
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, "rb") as f:
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(f.read())


def _trt_infer(engine, inp: np.ndarray):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
    except Exception as e:
        raise RuntimeError("pycuda not installed; cannot run .trt engine") from e

    context = engine.create_execution_context()
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)

    # Set input shape for dynamic shapes.
    context.set_input_shape(in_name, tuple(inp.shape))
    in_size = trt.volume(inp.shape) * np.dtype(np.float32).itemsize
    out_shape = context.get_tensor_shape(out_name)
    out_size = trt.volume(out_shape) * np.dtype(np.float32).itemsize

    d_in = cuda.mem_alloc(in_size)
    d_out = cuda.mem_alloc(out_size)
    cuda.memcpy_htod(d_in, inp.astype(np.float32).tobytes())
    context.set_tensor_address(in_name, int(d_in))
    context.set_tensor_address(out_name, int(d_out))
    context.execute_v2(bindings=[int(d_in), int(d_out)])
    out = np.empty(out_shape, dtype=np.float32)
    cuda.memcpy_dtoh(out, d_out)
    d_in.free()
    d_out.free()
    return out


def evaluate_trt(model_path: str, inputs: np.ndarray, targets: np.ndarray):
    engine = _load_engine(model_path)
    return evaluate_batch_mse(inputs, targets, lambda inp: _trt_infer(engine, inp))


def evaluate_ort(model_path: str, inputs: np.ndarray, targets: np.ndarray):
    import onnxruntime as ort
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    return evaluate_batch_mse(inputs, targets, lambda inp: sess.run([out_name], {in_name: inp})[0])


def find_model(model_dir: Path, model_path: Path):
    return find_model_variant(model_path, (".trt", ".onnx"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Jetson TensorRT / ONNX evaluator")
    parser.add_argument("--bin-dir", help="Directory with .trt/.onnx models")
    parser.add_argument("--bin", help="Single .trt/.onnx model")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", default="/tmp/jetson_evaluate.json")
    parser.add_argument("--pattern", default="*")
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
            if real_model.suffix == ".trt":
                loss = evaluate_trt(str(real_model), inputs, targets)
            else:
                loss = evaluate_ort(str(real_model), inputs, targets)
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

#!/usr/bin/env python3
"""Jetson TensorRT / ONNX Runtime batch evaluator.

Tries to use a pre-built .trt engine; if none exists, falls back to
onnxruntime with the TensorRT / CUDA execution providers on the Jetson.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


def parse_shape(s: str):
    if not s:
        return None
    parts = re.split(r"[x,]", s)
    return tuple(int(p.strip()) for p in parts if p.strip())


def load_batch(path: str, shape=None, format_hint=None):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"batch file not found: {path}")
    ext = p.suffix.lower()
    if format_hint == "npy" or (format_hint is None and ext == ".npy"):
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.ndarray):
            return arr
        return np.stack([np.asarray(a) for a in arr])
    if format_hint == "bin" or (format_hint is None and ext == ".bin"):
        if shape is None:
            raise ValueError(f"--input-shape/--target-shape required for .bin batch: {path}")
        arr = np.fromfile(path, dtype=np.float32)
        per = int(np.prod(shape))
        if per == 0:
            raise ValueError(f"invalid shape {shape}")
        if arr.size % per != 0:
            raise ValueError(f"bin file size {arr.size} not divisible by sample size {per}")
        n = arr.size // per
        return arr.reshape((n, *shape))
    raise ValueError(f"unsupported batch format: {path}")


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
    total = 0.0
    count = 0
    for inp, tgt in zip(inputs, targets):
        inp = np.asarray(inp, dtype=np.float32)
        tgt = np.asarray(tgt, dtype=np.float32)
        out = _trt_infer(engine, inp)
        out = out.reshape(-1)
        tgt = tgt.reshape(-1)
        min_len = min(out.size, tgt.size)
        out = out[:min_len]
        tgt = tgt[:min_len]
        mse = float(np.mean((out - tgt) ** 2))
        total += mse
        count += 1
    return total / count if count else float("inf")


def evaluate_ort(model_path: str, inputs: np.ndarray, targets: np.ndarray):
    import onnxruntime as ort
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    total = 0.0
    count = 0
    for inp, tgt in zip(inputs, targets):
        inp = np.asarray(inp, dtype=np.float32)
        tgt = np.asarray(tgt, dtype=np.float32)
        out = sess.run([out_name], {in_name: inp})[0]
        out = out.reshape(-1)
        tgt = tgt.reshape(-1)
        min_len = min(out.size, tgt.size)
        out = out[:min_len]
        tgt = tgt[:min_len]
        mse = float(np.mean((out - tgt) ** 2))
        total += mse
        count += 1
    return total / count if count else float("inf")


def find_model(model_dir: Path, model_path: Path):
    if model_path.is_file() and model_path.suffix in (".trt", ".onnx"):
        return model_path
    candidates = [model_dir / f"{model_path.stem}.trt", model_dir / f"{model_path.stem}.onnx"]
    for c in candidates:
        if c.is_file():
            return c
    return None


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

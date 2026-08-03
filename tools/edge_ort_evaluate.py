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
import re
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    print(f"[ERROR] cannot import onnxruntime: {e}", file=sys.stderr)
    sys.exit(1)


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


def find_model(model_path: Path):
    """Pick the best available model: .rknn, .trt, .onnx, or .bin."""
    for suffix in [".rknn", ".trt", ".onnx", ".bin"]:
        candidate = model_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def evaluate_onnx(model_path: str, inputs: np.ndarray, targets: np.ndarray) -> float:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
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

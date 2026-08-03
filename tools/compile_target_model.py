#!/usr/bin/env python3
"""Compile an ONNX model to a target edge-device format.

Backends:
  - horizon_bpu (rdk_x5, rdk_s100): uses tools/compile_bpu_docker.sh (hb_mapper)
  - rockchip_rknn (rk3588): placeholder - needs rknn-toolkit / NPU toolchain
  - nvidia_tensorrt (jetson_nano): placeholder - needs TensorRT / jetson toolchain

Usage is identical to compile_bpu_docker.sh:
    python tools/compile_target_model.py \
        --backend horizon_bpu \
        --model-name speech_decoder \
        --onnx .../model.onnx \
        --calib-dir .../calibration \
        --input-name concept \
        --input-shape 1x128x1x1 \
        --out-dir .../bpu
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import shutil

TOOLS_DIR = Path(__file__).resolve().parent


def compile_horizon(args) -> int:
    script = TOOLS_DIR / "compile_bpu_docker.sh"
    if not script.is_file():
        print(f"[error] {script} not found", file=sys.stderr)
        return 1
    cmd = [
        "bash", str(script),
        "--model-name", args.model_name,
        "--onnx", args.onnx,
        "--calib-dir", args.calib_dir,
        "--input-name", args.input_name,
        "--input-shape", args.input_shape,
        "--out-dir", args.out_dir,
        "--per-channel", str(args.per_channel),
        "--calib-type", args.calib_type,
    ]
    if args.march:
        cmd += ["--march", args.march]
    env = os.environ.copy()
    if args.run_hb_mapper:
        env["RUN_HB_MAPPER"] = args.run_hb_mapper
    return subprocess.run(cmd, env=env, timeout=args.timeout).returncode


def compile_rockchip(args) -> int:
    print(f"[rockchip_rknn] {args.model_name}: RKNN conversion is not yet implemented.", file=sys.stderr)
    print("              Install rknn-toolkit on a Linux x86/aarch64 host and convert the ONNX to .rknn", file=sys.stderr)
    print("              Falling back to ONNX Runtime evaluation (CPU on the device).", file=sys.stderr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_src = Path(args.onnx)
    onnx_dst = out_dir / f"{args.model_name}.onnx"
    if onnx_src.is_file():
        shutil.copy(onnx_src, onnx_dst)
    else:
        (out_dir / f"{args.model_name}.bin").touch()
    return 0


def compile_tensorrt(args) -> int:
    print(f"[nvidia_tensorrt] {args.model_name}: TensorRT conversion is not yet implemented.", file=sys.stderr)
    print("                Install TensorRT on the Jetson and build an engine from the ONNX.", file=sys.stderr)
    print("                Falling back to ONNX Runtime evaluation (CPU on the device).", file=sys.stderr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_src = Path(args.onnx)
    onnx_dst = out_dir / f"{args.model_name}.onnx"
    if onnx_src.is_file():
        shutil.copy(onnx_src, onnx_dst)
    else:
        (out_dir / f"{args.model_name}.bin").touch()
    return 0


BACKENDS = {
    "horizon_bpu": compile_horizon,
    "rdk_x5": compile_horizon,
    "rdk_s100": compile_horizon,
    "rockchip_rknn": compile_rockchip,
    "rk3588": compile_rockchip,
    "nvidia_tensorrt": compile_tensorrt,
    "jetson_nano": compile_tensorrt,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True,
                        help="Target backend (horizon_bpu, rockchip_rknn, nvidia_tensorrt)")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--input-name", required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-channel", required=True)
    parser.add_argument("--calib-type", required=True)
    parser.add_argument("--march", default=None)
    parser.add_argument("--run-hb-mapper", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    backend = BACKENDS.get(args.backend)
    if backend is None:
        print(f"[error] unknown backend: {args.backend}", file=sys.stderr)
        print(f"        supported: {list(BACKENDS)}", file=sys.stderr)
        return 1
    return backend(args)


if __name__ == "__main__":
    raise SystemExit(main())

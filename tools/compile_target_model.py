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
import glob
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

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


def _parse_shape(s: str):
    parts = re.split(r"[x,]", s)
    return [int(p.strip()) for p in parts if p.strip()]


def _calib_bin_to_npy(calibration_dir: Path, out_dir: Path, input_shape):
    """Convert .bin float32 calibration files to .npy files for RKNN."""
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / "dataset.txt"
    bins = sorted(glob.glob(str(calibration_dir / "*.bin")))
    if not bins:
        return None
    with open(txt, "w", encoding="utf-8") as f:
        for b in bins:
            arr = Path(b)
            npy = out_dir / arr.with_suffix(".npy").name
            data = np.fromfile(arr, dtype=np.float32)
            data = data.reshape(input_shape)
            np.save(npy, data)
            f.write(f"{npy}\n")
    return txt


def compile_rockchip(args) -> int:
    """Try rknn-toolkit2 conversion; fall back to ONNX if unavailable or fails."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_src = Path(args.onnx)
    if not onnx_src.is_file():
        print(f"[rockchip_rknn] {args.model_name}: ONNX not found: {args.onnx}", file=sys.stderr)
        return 1

    # Try rknn-toolkit2 first.
    try:
        import numpy as np
        from rknn.api import RKNN
    except Exception as e:
        print(f"[rockchip_rknn] {args.model_name}: rknn-toolkit2 not installed ({e}).", file=sys.stderr)
        print("              Falling back to ONNX Runtime evaluation (CPU on the device).", file=sys.stderr)
        shutil.copy(onnx_src, out_dir / f"{args.model_name}.onnx")
        return 0

    input_shape = _parse_shape(args.input_shape)
    calib_dir = Path(args.calib_dir)
    dataset_txt = None
    if calib_dir.is_dir():
        dataset_dir = out_dir / "rknn_calib"
        dataset_txt = _calib_bin_to_npy(calib_dir, dataset_dir, input_shape)

    rknn_path = out_dir / f"{args.model_name}.rknn"
    try:
        rknn = RKNN(verbose=False)
        # Single input. mean/std = 0/1 because the model already consumes raw floats.
        rknn.config(
            mean_values=[[0] * input_shape[1]],
            std_values=[[1] * input_shape[1]],
            target_platform="rk3588",
        )
        rknn.load_onnx(
            model=str(onnx_src),
            inputs=[args.input_name],
            input_size_list=[input_shape],
        )
        rknn.build(do_quantization=dataset_txt is not None, dataset=str(dataset_txt) if dataset_txt else None)
        rknn.export_rknn(str(rknn_path))
        print(f"[rockchip_rknn] {args.model_name}: exported {rknn_path}")
        return 0
    except Exception as e:
        print(f"[rockchip_rknn] {args.model_name}: conversion failed ({e}).", file=sys.stderr)
        print("              Falling back to ONNX Runtime evaluation.", file=sys.stderr)
        # Ensure a usable artifact remains.
        shutil.copy(onnx_src, out_dir / f"{args.model_name}.onnx")
        if rknn_path.is_file():
            rknn_path.unlink()
        return 0


def _has_trt():
    try:
        import tensorrt
        return True
    except Exception:
        return False


def compile_tensorrt(args) -> int:
    """Build a TensorRT engine from ONNX. If the toolchain is missing or the host
    is x86 and cannot produce a Jetson-compatible engine, fall back to ONNX."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_src = Path(args.onnx)
    if not onnx_src.is_file():
        print(f"[nvidia_tensorrt] {args.model_name}: ONNX not found: {args.onnx}", file=sys.stderr)
        return 1

    trt_path = out_dir / f"{args.model_name}.trt"
    arch = platform.machine().lower()
    is_jetson = arch in ("aarch64", "arm64") and _has_trt()

    if not is_jetson:
        print(f"[nvidia_tensorrt] {args.model_name}: x86 host cannot produce a Jetson-compatible .trt engine.", file=sys.stderr)
        print("                ONNX will be deployed and the Jetson can build the engine on-device.", file=sys.stderr)
        shutil.copy(onnx_src, out_dir / f"{args.model_name}.onnx")
        return 0

    # 1) Try trtexec (TensorRT >= 7.0) on an aarch64 host.
    trtexec = shutil.which("trtexec")
    if trtexec:
        input_shape = _parse_shape(args.input_shape)
        shape_arg = "x".join(str(s) for s in input_shape)
        cmd = [
            trtexec,
            f"--onnx={onnx_src}",
            f"--saveEngine={trt_path}",
            f"--minShapes={args.input_name}:{shape_arg}",
            f"--optShapes={args.input_name}:{shape_arg}",
            f"--maxShapes={args.input_name}:{shape_arg}",
            "--explicitBatch",
        ]
        try:
            rc = subprocess.run(cmd, timeout=args.timeout).returncode
            if rc == 0 and trt_path.is_file():
                print(f"[nvidia_tensorrt] {args.model_name}: built {trt_path} with trtexec")
                return 0
        except Exception as e:
            print(f"[nvidia_tensorrt] {args.model_name}: trtexec failed ({e}).", file=sys.stderr)

    # 2) Try Python TensorRT API on an aarch64 host.
    try:
        import tensorrt as trt
    except Exception as e:
        print(f"[nvidia_tensorrt] {args.model_name}: TensorRT not installed ({e}).", file=sys.stderr)
        print("                Falling back to ONNX Runtime evaluation.", file=sys.stderr)
        shutil.copy(onnx_src, out_dir / f"{args.model_name}.onnx")
        return 0

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_src, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            return 1

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = 1 << 30

    input_shape = _parse_shape(args.input_shape)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        args.input_name,
        tuple(input_shape),
        tuple(input_shape),
        tuple(input_shape),
    )
    config.add_optimization_profile(profile)

    try:
        if hasattr(builder, "build_serialized_network"):
            # TensorRT 8.5+ / 10
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise RuntimeError("build_serialized_network returned None")
            with open(trt_path, "wb") as f:
                f.write(engine)
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("build_engine returned None")
            with open(trt_path, "wb") as f:
                f.write(engine.serialize())
        print(f"[nvidia_tensorrt] {args.model_name}: built {trt_path} with Python API")
        return 0
    except Exception as e:
        print(f"[nvidia_tensorrt] {args.model_name}: build failed ({e}).", file=sys.stderr)
        print("                Falling back to ONNX Runtime evaluation.", file=sys.stderr)
        shutil.copy(onnx_src, out_dir / f"{args.model_name}.onnx")
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

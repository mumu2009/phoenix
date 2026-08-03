#!/usr/bin/env python3
"""Deploy the four trained real-data models to an edge device runtime store.

Supports any device registered in config/edge_devices.json (rdk_x5, rdk_s100,
rk3588, jetson_nano).  The best artifact (.bin for BPU, .onnx for ORT) is copied
under runtime_store/models/additive_jpea/<model>/.

Usage (from Kali, through reverse tunnel):
    python tools/deploy_real_models_to_x5.py --edge-device lab_x5

Usage (from Windows, direct to X5):
    python tools/deploy_real_models_to_x5.py \
        --edge-device-config config/edge_devices.json --edge-device lab_x5_direct
"""
import argparse
import os
import sys
from pathlib import Path, PurePosixPath


def mkdir_p(sftp, path: str):
    parts = []
    for part in path.strip("/").split("/"):
        parts.append(part)
        current = "/" + "/".join(parts)
        try:
            sftp.mkdir(current)
        except IOError:
            pass


def _best_artifact(src: Path, compile_backend: str) -> Path:
    """Pick the best compiled artifact for the target backend."""
    if compile_backend in ("horizon_bpu", "rdk_x5", "rdk_s100"):
        candidates = ["best.bin", "best.onnx"]
    elif compile_backend in ("rockchip_rknn", "rk3588"):
        candidates = ["best.rknn", "best.onnx"]
    elif compile_backend in ("nvidia_tensorrt", "jetson_nano"):
        candidates = ["best.trt", "best.onnx"]
    else:
        candidates = ["best.bin", "best.onnx"]
    for c in candidates:
        p = src / c
        if p.is_file():
            return p
    raise FileNotFoundError(f"No best artifact in {src}; run training first")


def deploy_one(sftp, src_dir: Path, x5_root: PurePosixPath, name: str, compile_backend: str):
    """Copy the best compiled artifact and manifest to the edge runtime_store path."""
    src = src_dir / name / name
    manifest = src / "model.manifest.json"
    best = _best_artifact(src, compile_backend)

    dst_dir = x5_root / "models" / "additive_jpea" / name
    mkdir_p(sftp, str(dst_dir.parent))
    mkdir_p(sftp, str(dst_dir))

    sftp.put(str(best), str(dst_dir / best.name))
    if manifest.exists():
        sftp.put(str(manifest), str(dst_dir / "model.manifest.json"))

    # C++ factory tries model_encoder/decoder with the same extension.
    ext = best.suffix
    if "encoder" in name:
        sftp.put(str(best), str(dst_dir / f"model_encoder{ext}"))
    else:
        sftp.put(str(best), str(dst_dir / f"model_decoder{ext}"))

    print(f"[deploy] {name} -> {dst_dir} ({best.name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-device", default=None,
                        help="Device name from config/edge_devices.json")
    parser.add_argument("--edge-device-config", default=None)
    parser.add_argument("--work-dir", default="/home/kali/phoenix/additive_work/real")
    parser.add_argument("--models",
                        default="speech_encoder,speech_decoder,vision_encoder,vision_decoder")
    # Legacy args for direct use without a config.
    parser.add_argument("--x5-host", default=None)
    parser.add_argument("--x5-user", default=None)
    parser.add_argument("--x5-pass", default=None)
    parser.add_argument("--x5-port", type=int, default=22)
    parser.add_argument("--x5-runtime", default=None)
    args = parser.parse_args()

    # Prefer the new edge-device manager; fall back to legacy direct args.
    if args.edge_device:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.edge_device_manager import load_edge_device
        dev = load_edge_device(args.edge_device, config_path=args.edge_device_config)
        x5_host = dev.host
        x5_user = dev.user
        x5_pass = dev.password
        x5_port = dev.port
        x5_runtime = args.x5_runtime or dev.cfg.get("runtime_store", "/root/phoenix/runtime_store")
        compile_backend = dev.cfg.get("compile_backend", "horizon_bpu")
    else:
        if not all([args.x5_host, args.x5_user, args.x5_pass, args.x5_runtime]):
            raise SystemExit("--edge-device or all --x5-* arguments are required")
        x5_host = args.x5_host
        x5_user = args.x5_user
        x5_pass = args.x5_pass
        x5_port = args.x5_port
        x5_runtime = args.x5_runtime
        compile_backend = "horizon_bpu"

    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(x5_host, port=x5_port, username=x5_user, password=x5_pass, timeout=30)
    sftp = client.open_sftp()

    work = Path(args.work_dir)
    x5_root = PurePosixPath(x5_runtime)
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        deploy_one(sftp, work, x5_root, name, compile_backend)

    sftp.close()
    client.close()
    print("[done] deployed to edge device")


if __name__ == "__main__":
    raise SystemExit(main())

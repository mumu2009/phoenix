#!/usr/bin/env python3
"""Deploy the four trained real-data BPU models to an RDK X5.

Usage:
    python tools/deploy_real_models_to_x5.py \
        --x5-host 192.168.0.107 --x5-user sunrise --x5-pass sunrise \
        --work-dir /home/kali/phoenix/additive_work/real \
        --x5-runtime /home/sunrise/phoenix/runtime_store
"""
import argparse
import paramiko
from pathlib import Path, PurePosixPath


def deploy_one(sftp, src_dir: Path, x5_root: PurePosixPath, name: str, kind: str):
    """Copy best.bin + manifest to the X5 runtime_store path."""
    src = src_dir / name / name
    bin_file = src / "best.bin"
    manifest = src / "model.manifest.json"
    if not bin_file.exists():
        raise FileNotFoundError(f"{bin_file} not found; run training first")

    # C++ factory tries runtime_store/models/additive_jpea/<name>/best.bin
    dst_dir = x5_root / "models" / "additive_jpea" / name
    sftp.mkdir(str(dst_dir.parent), ignore_existing=True)
    sftp.mkdir(str(dst_dir), ignore_existing=True)

    sftp.put(str(bin_file), str(dst_dir / "best.bin"))
    if manifest.exists():
        sftp.put(str(manifest), str(dst_dir / "model.manifest.json"))

    # Also create a model_encoder/decoder alias for the config paths used by
    # jpea_v2_image_world_model.cpp and jpea_v2_speech_world_model.cpp.
    if "encoder" in name:
        sftp.put(str(bin_file), str(dst_dir / "model_encoder.bin"))
    else:
        sftp.put(str(bin_file), str(dst_dir / "model_decoder.bin"))

    print(f"[deploy] {name} -> {dst_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x5-host", default="192.168.0.107")
    parser.add_argument("--x5-user", default="sunrise")
    parser.add_argument("--x5-pass", default="sunrise")
    parser.add_argument("--work-dir", default="/home/kali/phoenix/additive_work/real")
    parser.add_argument("--x5-runtime", default="/home/sunrise/phoenix/runtime_store")
    parser.add_argument(
        "--models",
        default="speech_encoder,speech_decoder,vision_encoder,vision_decoder",
        help="Comma-separated models to deploy",
    )
    args = parser.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.x5_host, username=args.x5_user, password=args.x5_pass, timeout=30)
    sftp = client.open_sftp()

    work = Path(args.work_dir)
    x5_root = PurePosixPath(args.x5_runtime)
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        deploy_one(sftp, work, x5_root, name, "encoder" if "encoder" in name else "decoder")

    sftp.close()
    client.close()
    print("[done] deployed to X5")


if __name__ == "__main__":
    raise SystemExit(main())

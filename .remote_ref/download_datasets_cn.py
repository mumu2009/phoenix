#!/usr/bin/env python3
"""One-click Chinese-mirror dataset downloader for Phoenix remote training.

This script avoids github.com / huggingface.co / openslr.org and uses domestic
mirrors instead:
  - Images (Tiny-ImageNet-200): HuggingFace China mirror (hf-mirror.com)
    via the `datasets` library (`HF_ENDPOINT=https://hf-mirror.com`).
  - Audio (MUSAN 16 kHz): ModelScope (modelscope.cn) via
    `tools/download_modelscope_audio.py`.

Usage:
    python tools/download_datasets_cn.py --all --data-root ../remote_training_data
    python tools/download_datasets_cn.py --image --data-root ../remote_training_data
    python tools/download_datasets_cn.py --audio --data-root ../remote_training_data --max-audio-samples 10000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


def set_cn_mirror_env():
    """Force HF downloads through the Chinese mirror."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def download_tiny_imagenet(out_dir: Path, max_samples: Optional[int] = None) -> int:
    """Download Tiny-ImageNet-200 from hf-mirror.com and save as flat JPEGs."""
    try:
        from datasets import load_dataset
        from PIL import Image
    except Exception as e:
        print(f"[ERROR] missing dependencies (datasets, Pillow): {e}", file=sys.stderr)
        return 1

    out_dir = Path(out_dir) / "tiny-imagenet-200"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try a few dataset ids on the Chinese HF mirror.
    candidates = [
        "torch-uncertainty/tiny-imagenet-200",
        "SteveZeyuZhang/Tiny-ImageNet-200",
        "shuqike/tiny-imagenet-200-clean",
    ]
    ds_train = None
    ds_val = None
    for ds_id in candidates:
        try:
            print(f"[image] trying {ds_id} via HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
            ds = load_dataset(ds_id)
            ds_train = ds["train"]
            ds_val = ds["valid"] if "valid" in ds else ds.get("validation")
            print(f"[image] loaded {ds_id}: train={len(ds_train)}, val={len(ds_val) if ds_val is not None else 0}")
            break
        except Exception as e:
            print(f"[image] {ds_id} failed: {e}", file=sys.stderr)

    if ds_train is None:
        print("[image] all Tiny-ImageNet-200 dataset mirrors failed", file=sys.stderr)
        return 1

    saved = 0
    for split_name, split in [("train", ds_train), ("val", ds_val)]:
        if split is None:
            continue
        for i, item in enumerate(split):
            if max_samples is not None and i >= max_samples:
                break
            try:
                img = item.get("image")
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                label = str(item.get("label", f"label_{i}"))
                split_dir = out_dir / split_name
                if split_name == "train":
                    # ImageFolder-style: class/images/*.JPEG
                    cls_dir = split_dir / label / "images"
                else:
                    # Keep val flat; ImageGlobDataset will find them anyway.
                    cls_dir = split_dir / "images"
                cls_dir.mkdir(parents=True, exist_ok=True)
                out_path = cls_dir / f"{split_name}_{i:06d}.JPEG"
                img.convert("RGB").save(out_path, "JPEG", quality=95)
                saved += 1
                if saved % 1000 == 0:
                    print(f"[image] saved {saved} images ...")
            except Exception as e:
                print(f"[image] skip sample {i}: {e}", file=sys.stderr)

    print(f"[image] done. {saved} images saved to {out_dir}")
    return 0


def download_musan(out_dir: Path, max_samples: int = -1) -> int:
    """Download MUSAN 16 kHz WAVs from ModelScope."""
    script = ROOT / "tools" / "download_modelscope_audio.py"
    if not script.is_file():
        print(f"[ERROR] missing {script}", file=sys.stderr)
        return 1

    target = out_dir / "musan_16k"
    target.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--out-dir", str(target),
        "--target-sr", "16000",
    ]
    if max_samples > 0:
        cmd += ["--max-samples", str(max_samples)]

    print("[audio] running:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description="Download training datasets from Chinese mirrors")
    p.add_argument("--data-root", required=True, help="Parent directory for downloaded datasets")
    p.add_argument("--all", action="store_true", help="Download both image and audio")
    p.add_argument("--image", action="store_true", help="Download Tiny-ImageNet-200")
    p.add_argument("--audio", action="store_true", help="Download MUSAN")
    p.add_argument("--max-image-samples", type=int, default=None, help="Limit image samples for quick tests")
    p.add_argument("--max-audio-samples", type=int, default=-1, help="Limit MUSAN samples")
    p.add_argument("--hf-endpoint", default="https://hf-mirror.com", help="Chinese HF mirror")
    args = p.parse_args()

    set_cn_mirror_env()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    rc = 0
    if args.all or args.image:
        rc |= download_tiny_imagenet(root, args.max_image_samples)
    if args.all or args.audio:
        rc |= download_musan(root, args.max_audio_samples)

    if not (args.all or args.image or args.audio):
        p.print_help()
        return 0

    print(f"\n[datasets] outputs under {root}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

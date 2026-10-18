#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the four official V-JEPA 2.1 checkpoints from Meta.
Supports resume and simple progress reporting.
"""
import os
import sys
import time
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "models" / "vjepa2_1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "vitb": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
    "vitl": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
    "vitg": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt",
    "vitG": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt",
}

# Windows filenames are case-insensitive: vitg and vitG would collide.
LOCAL_FILENAME = {
    "vitb": "vjepa2_1_vitb_384.pt",
    "vitl": "vjepa2_1_vitl_384.pt",
    "vitg": "vjepa2_1_vitg_384.pt",
    "vitG": "vjepa2_1_vitG_2b_384.pt",
}

CHUNK_SIZE = 1024 * 1024  # 1 MB


def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def download(name: str, url: str, dest: Path) -> None:
    print(f"\n[{name}] {url}")
    print(f"Destination: {dest}")
    start_offset = dest.stat().st_size if dest.exists() else 0
    if start_offset > 0:
        print(f"Resuming from {format_bytes(start_offset)}")

    req = urllib.request.Request(url)
    if start_offset > 0:
        req.add_header("Range", f"bytes={start_offset}-")

    with urllib.request.urlopen(req, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0)) + start_offset
        mode = "ab" if start_offset > 0 else "wb"
        downloaded = start_offset
        t0 = time.time()
        with open(dest, mode) as f:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * CHUNK_SIZE) < CHUNK_SIZE:
                    pct = downloaded * 100.0 / total
                    rate = downloaded / (time.time() - t0 + 1e-6)
                    print(
                        f"  {format_bytes(downloaded)} / {format_bytes(total)} ({pct:.1f}%)  @ {format_bytes(rate)}/s",
                        flush=True,
                    )
    final_size = dest.stat().st_size
    print(f"Done: {dest.name} -> {format_bytes(final_size)}")


def main() -> int:
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(CHECKPOINTS.keys())

    for name in names:
        if name not in CHECKPOINTS:
            print(f"Unknown checkpoint: {name}")
            continue
        url = CHECKPOINTS[name]
        dest = SAVE_DIR / LOCAL_FILENAME[name]
        try:
            download(name, url, dest)
        except Exception as e:
            print(f"Failed to download {name}: {e}", file=sys.stderr)
            return 1
    print("\nAll requested checkpoints saved to:", SAVE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

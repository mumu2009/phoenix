#!/usr/bin/env python3
"""Download the four official I-JEPA / video world model weights.

Models are placed under runtime_store/models/ijepa/<variant>/, mirroring the
VideoModelConfig ids in video_model.hpp.

Requires an internet connection.  Uses huggingface_hub if available, otherwise
falls back to urllib.  To use the original research .pth.tar checkpoints, pass
--format pth_tar.

Mirror support:
- --mirror hf-mirror  uses https://hf-mirror.com (HuggingFace 清华镜像)
- --mirror modelscope uses https://www.modelscope.cn (魔搭社区)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = REPO_ROOT / "runtime_store" / "models" / "ijepa"

# HuggingFace-compatible and ModelScope endpoints
HF_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"

# Local variant id -> HuggingFace repo id.
# Note: the 448-resolution variant lives under facebook/ijepa_vith16_1k on HF.
VARIANTS = {
    "ijepa_vith14_1k": "facebook/ijepa_vith14_1k",
    "ijepa_vith16_448": "facebook/ijepa_vith16_1k",
    "ijepa_vith14_22k": "facebook/ijepa_vith14_22k",
    "ijepa_vitg16_22k": "facebook/ijepa_vitg16_22k",
}

# Original Meta research .pth.tar checkpoints (optional)
PTH_TAR_URLS = {
    "ijepa_vith14_1k": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
    "ijepa_vith16_448": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-300e.pth.tar",
    "ijepa_vith14_22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar",
    "ijepa_vitg16_22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16.224-900e.pth.tar",
}


def variant_dir(variant: str) -> Path:
    """Return the local directory for a variant id."""
    return DEST_DIR / variant


def download_with_hf(variant: str, repo_id: str, dest: Path, endpoint: str = HF_ENDPOINT) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed") from exc
    # Respect a user-provided HF_ENDPOINT, otherwise use the selected mirror.
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    local_dir = dest / variant
    print(f"Downloading {repo_id} via huggingface_hub ({endpoint}) to {local_dir} ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            endpoint=endpoint,
        )
    except TypeError:
        # Older versions of huggingface_hub do not accept `endpoint`.
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
    print(f"  -> {local_dir}")


def download_file(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Simple urllib-based download with progress."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "phoenix-ijepa-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded}/{total} bytes)", end="")
    print()


def download_safetensors_fallback(
    variant: str, repo_id: str, dest: Path, endpoint: str = HF_ENDPOINT
) -> None:
    """Download config.json, preprocessor_config.json and model.safetensors via urllib."""
    local_dir = dest / variant
    base = f"{endpoint.rstrip('/')}/{repo_id}/resolve/main"
    for filename in ["config.json", "preprocessor_config.json", "model.safetensors"]:
        url = f"{base}/{filename}?download=true"
        download_file(url, local_dir / filename)


def download_pth_tar(variant: str, dest: Path) -> None:
    url = PTH_TAR_URLS[variant]
    filename = url.split("/")[-1]
    download_file(url, dest / variant / filename)


def _modelscope_id(repo_id: str, override: Optional[str]) -> str:
    """Return the ModelScope model id to use."""
    return override or repo_id


def download_with_modelscope(
    variant: str,
    repo_id: str,
    dest: Path,
    modelscope_id_override: Optional[str] = None,
) -> None:
    try:
        from modelscope.hub.snapshot_download import (  # type: ignore
            snapshot_download,
        )
    except ImportError as exc:
        raise RuntimeError("modelscope is not installed (`pip install modelscope`)") from exc
    ms_id = _modelscope_id(repo_id, modelscope_id_override)
    local_dir = dest / variant
    print(f"Downloading {ms_id} via modelscope to {local_dir} ...")
    snapshot_download(ms_id, local_dir=str(local_dir), revision="master")
    print(f"  -> {local_dir}")


def download_safetensors_modelscope(
    variant: str,
    repo_id: str,
    dest: Path,
    modelscope_id_override: Optional[str] = None,
) -> None:
    """Direct urllib fallback for ModelScope."""
    ms_id = _modelscope_id(repo_id, modelscope_id_override)
    local_dir = dest / variant
    base = f"{MODELSCOPE_ENDPOINT}/models/{ms_id}/resolve/master"
    print(f"Downloading {ms_id} from modelscope via urllib to {local_dir} ...")
    for filename in ["config.json", "preprocessor_config.json", "model.safetensors"]:
        url = f"{base}/{filename}"
        download_file(url, local_dir / filename)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download I-JEPA / video model weights")
    parser.add_argument(
        "--format",
        choices=["safetensors", "pth_tar"],
        default="safetensors",
        help="Weight format to download (default: safetensors)",
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        help="Download only a single variant",
    )
    parser.add_argument(
        "--use-urllib",
        action="store_true",
        help="Force urllib fallback instead of huggingface_hub / modelscope SDK",
    )
    parser.add_argument(
        "--mirror",
        choices=["huggingface", "hf-mirror", "modelscope"],
        default="huggingface",
        help="Download source mirror (default: huggingface)",
    )
    parser.add_argument(
        "--modelscope-id",
        help="Override ModelScope model id (default: same as HF repo_id)",
    )
    parser.add_argument(
        "--endpoint",
        help="Custom HuggingFace-compatible endpoint, e.g. https://hf-mirror.com",
    )
    args = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if args.variant:
        items = [(args.variant, VARIANTS[args.variant])]
    else:
        items = list(VARIANTS.items())

    if args.endpoint:
        endpoint = args.endpoint
    elif args.mirror == "hf-mirror":
        endpoint = HF_MIRROR_ENDPOINT
    else:
        endpoint = HF_ENDPOINT

    for variant, repo_id in items:
        try:
            if args.format == "pth_tar":
                download_pth_tar(variant, DEST_DIR)
            elif args.mirror == "modelscope":
                if not args.use_urllib:
                    try:
                        download_with_modelscope(variant, repo_id, DEST_DIR, args.modelscope_id)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        print(f"modelscope SDK failed ({exc}), falling back to urllib.")
                download_safetensors_modelscope(variant, repo_id, DEST_DIR, args.modelscope_id)
            elif not args.use_urllib:
                try:
                    download_with_hf(variant, repo_id, DEST_DIR, endpoint)
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"huggingface_hub failed ({exc}), falling back to urllib.")
                    download_safetensors_fallback(variant, repo_id, DEST_DIR, endpoint)
            else:
                download_safetensors_fallback(variant, repo_id, DEST_DIR, endpoint)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR downloading {variant}: {exc}", file=sys.stderr)
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

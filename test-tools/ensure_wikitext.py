"""Ensure robots/wikitext-103-all.txt exists; download and build it if missing."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIKITEXT_PATH = ROOT / "robots" / "wikitext-103-all.txt"
WIKITEXT_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"


def log(msg: str) -> None:
    print(f"[ensure_wikitext] {msg}", flush=True)


def has_existing_wikitext() -> bool:
    if not WIKITEXT_PATH.exists():
        return False
    size = WIKITEXT_PATH.stat().st_size
    return size > 10 * 1024 * 1024  # at least 10 MB


def build_wikitext_from_zip(extract_dir: Path) -> None:
    parts = []
    for name in ("wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens"):
        p = extract_dir / "wikitext-103-raw" / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
    if not parts:
        raise RuntimeError("No wiki.*.tokens files found in downloaded archive")
    WIKITEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    WIKITEXT_PATH.write_text("\n".join(parts), encoding="utf-8", errors="replace")


def download_wikitext() -> None:
    log(f"wikitext missing, downloading from {WIKITEXT_URL}")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        zip_path = td_path / "wikitext-103-raw-v1.zip"
        try:
            urllib.request.urlretrieve(WIKITEXT_URL, zip_path)
            log(f"downloaded {zip_path.stat().st_size} bytes")
        except Exception as e:
            raise RuntimeError(f"Failed to download wikitext: {e}") from e
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(td_path)
        build_wikitext_from_zip(td_path)


def ensure_wikitext() -> None:
    if has_existing_wikitext():
        log(f"wikitext already exists ({WIKITEXT_PATH.stat().st_size} bytes)")
        return
    download_wikitext()
    log(f"wikitext built at {WIKITEXT_PATH} ({WIKITEXT_PATH.stat().st_size} bytes)")


def main() -> int:
    try:
        ensure_wikitext()
        return 0
    except Exception as e:
        log(f"error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""vboxsf-safe file I/O helpers.

VirtualBox shared folders (vboxsf) can return an incorrect number of bytes
from a single raw write() when the buffer is >= 64 KiB.  These helpers write
and copy files in small chunks to avoid that kernel bug.
"""
import shutil
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]
SAFE_CHUNK = 32768  # 32 KiB, comfortably below the vboxsf 64 KiB threshold


def safe_write_bytes(path: PathLike, data: bytes, chunk: int = SAFE_CHUNK) -> int:
    """Write *data* to *path* in small chunks and return bytes written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "wb") as f:
        for i in range(0, len(data), chunk):
            n = f.write(data[i:i + chunk])
            if n is None:
                n = chunk
            written += n
    return written


def safe_tofile(arr: np.ndarray, path: PathLike, chunk: int = SAFE_CHUNK) -> None:
    """Equivalent to ``arr.astype(np.float32).tofile(path)`` but vboxsf-safe."""
    data = arr.astype(np.float32, copy=False).tobytes()
    safe_write_bytes(path, data, chunk=chunk)


def safe_copy(src: PathLike, dst: PathLike, chunk: int = SAFE_CHUNK) -> Path:
    """shutil.copy replacement that copies file contents in small chunks.

    Preserves the destination file name like ``shutil.copy(src, dst)``.
    """
    src = Path(src)
    dst = Path(dst)
    if dst.is_dir():
        dst = dst / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            buf = fsrc.read(chunk)
            if not buf:
                break
            fdst.write(buf)
    shutil.copymode(src, dst)
    return dst

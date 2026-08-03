#!/usr/bin/env python3
"""vboxsf-safe file I/O helpers.

VirtualBox shared folders (vboxsf) can return an incorrect number of bytes
from a single raw write() when the buffer is >= 64 KiB, and in some
configurations even smaller writes (8 KiB - 32 KiB) return bogus counts.
These helpers write and copy files in small chunks and, when the guest
kernel reports a count larger than requested, clamp it to the requested
size (the data cannot physically be larger than what we passed down).
"""
import errno
import os
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]
SAFE_CHUNK = 8192  # 8 KiB; small enough to dodge vboxsf write-count glitches
_MAX_RETRIES = 8
_MIN_CHUNK = 512


def _write_all(fd: int, data: bytes, chunk: int) -> int:
    """Write *data* to open fd in small pieces, retrying/halving on errors."""
    offset = 0
    current_chunk = max(_MIN_CHUNK, chunk)
    while offset < len(data):
        piece = data[offset:offset + current_chunk]
        if not piece:
            break
        try:
            n = os.write(fd, piece)
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            # On vboxsf/odd filesystems, EIO/EINVAL/EFAULT can be transient;
            # try a smaller write before giving up.
            if exc.errno in (errno.EIO, errno.EINVAL, errno.EFAULT, errno.EBUSY) and current_chunk > _MIN_CHUNK:
                current_chunk = max(_MIN_CHUNK, current_chunk // 2)
                continue
            raise
        if n == 0:
            # Some filesystems return 0 for a transient busy condition; retry once.
            if current_chunk > _MIN_CHUNK:
                current_chunk = max(_MIN_CHUNK, current_chunk // 2)
                continue
            raise OSError("os.write returned 0")
        if n > len(piece):
            # vboxsf bug: kernel returned a count larger than the request.
            # Clamp it; we know the OS cannot write more bytes than we gave it.
            n = len(piece)
        offset += n
        current_chunk = min(chunk, current_chunk)
    return offset


def _read_all(fd: int, chunk: int) -> bytes:
    """Read up to *chunk* bytes, retrying on transient errors."""
    while True:
        try:
            return os.read(fd, chunk)
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            raise


def safe_write_bytes(path: PathLike, data: bytes, chunk: int = SAFE_CHUNK) -> int:
    """Write *data* to *path* in small chunks and return bytes written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    try:
        total = _write_all(fd, data, chunk)
    finally:
        os.close(fd)
    return total


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
    read_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    src_fd = os.open(src, read_flags)
    dst_fd = os.open(dst, write_flags, 0o644)
    try:
        while True:
            piece = _read_all(src_fd, chunk)
            if not piece:
                break
            _write_all(dst_fd, piece, chunk)
    finally:
        os.close(src_fd)
        os.close(dst_fd)
    shutil.copymode(src, dst)
    return dst

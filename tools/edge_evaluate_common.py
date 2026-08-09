#!/usr/bin/env python3
"""Shared helpers for the per-device edge model evaluators.

``x5_bpu_evaluate.py``, ``rk3588_npu_evaluate.py``, ``jetson_trt_evaluate.py``
and ``edge_ort_evaluate.py`` each evaluate a batch of models against the same
input/target batch and report per-model MSE in an identical JSON shape, so
``bpu_evolve_additive.py`` can consume any of them interchangeably. Historically
each script duplicated ``parse_shape``/``load_batch``/the per-sample MSE loop
and the "find the actual model file for this candidate" lookup; this module
factors that shared logic out so it lives in one place. Every caller keeps its
own CLI parsing untouched -- only the internal implementation is shared.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


def parse_shape(s: Optional[str]):
    """Parse '1x1x1x16000' or '1,1,1,16000' into a tuple of ints."""
    if not s:
        return None
    parts = re.split(r"[x,]", s)
    return tuple(int(p.strip()) for p in parts if p.strip())


def load_batch(path: str, shape=None, format_hint: Optional[str] = None) -> np.ndarray:
    """Load an input or target batch.

    For ``.npy`` files, the first dimension is the batch. For ``.bin`` files,
    ``shape`` is the per-sample shape (including batch=1); the total file
    length determines the actual batch size.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"batch file not found: {path}")

    ext = p.suffix.lower()
    if format_hint == "npy" or (format_hint is None and ext == ".npy"):
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.ndarray):
            return arr
        # list of arrays
        return np.stack([np.asarray(a) for a in arr])

    if format_hint == "bin" or (format_hint is None and ext == ".bin"):
        if shape is None:
            raise ValueError(f"--input-shape/--target-shape required for .bin batch: {path}")
        arr = np.fromfile(path, dtype=np.float32)
        per = int(np.prod(shape))
        if per == 0:
            raise ValueError(f"invalid shape {shape}")
        if arr.size % per != 0:
            raise ValueError(
                f"bin file size {arr.size} not divisible by sample size {per} ({shape})"
            )
        n = arr.size // per
        return arr.reshape((n, *shape))

    raise ValueError(f"unsupported batch format: {path}")


def compute_mse(out: np.ndarray, tgt: np.ndarray, mismatch_context: Optional[str] = None) -> float:
    """Flatten ``out``/``tgt`` and return the MSE over their common prefix.

    Some backends return output with a trailing length that doesn't exactly
    match the target dimensions, so both are flattened and truncated to the
    shorter length before comparing. If ``mismatch_context`` is given, a
    diagnostic line is printed on a length mismatch (matching the behavior
    ``x5_bpu_evaluate.py`` has always had; other callers pass ``None`` to stay
    silent, matching their existing behavior).
    """
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    tgt = np.asarray(tgt, dtype=np.float32).reshape(-1)
    min_len = min(out.size, tgt.size)
    if mismatch_context and out.size != tgt.size:
        print(f"  [shape] {mismatch_context} out={out.size} tgt={tgt.size} using first {min_len}")
    out = out[:min_len]
    tgt = tgt[:min_len]
    return float(np.mean((out - tgt) ** 2))


def evaluate_batch_mse(
    inputs: np.ndarray,
    targets: np.ndarray,
    infer_fn: Callable[[np.ndarray], np.ndarray],
    mismatch_context: Optional[str] = None,
) -> float:
    """Run ``infer_fn`` over every (input, target) pair and average the MSE.

    ``infer_fn`` takes a single float32 input sample and returns the model's
    raw output for it; this factors out the per-sample cast/MSE bookkeeping
    that was previously duplicated in each backend's ``evaluate_*`` function.
    """
    total = 0.0
    count = 0
    for inp, tgt in zip(inputs, targets):
        inp = np.asarray(inp, dtype=np.float32)
        tgt = np.asarray(tgt, dtype=np.float32)
        out = infer_fn(inp)
        total += compute_mse(out, tgt, mismatch_context)
        count += 1
    if count == 0:
        return float("inf")
    return total / count


def find_model_variant(model_path: Path, suffixes: Iterable[str]) -> Optional[Path]:
    """Return the first existing path among ``model_path`` with each of
    ``suffixes`` swapped in (in order), or ``model_path`` itself if it
    already exists with one of those suffixes.
    """
    suffixes = tuple(suffixes)
    if model_path.is_file() and model_path.suffix in suffixes:
        return model_path
    for suf in suffixes:
        candidate = model_path.with_suffix(suf)
        if candidate.is_file():
            return candidate
    return None

#!/usr/bin/env python3
"""llama_prune_analyzer.py - GGUF weight-pruning analyzer (magnitude-based)

Scientific basis: magnitude pruning (Han et al. 2015, "Learning both Weights
and Connections for Efficient Neural Networks") and the Wanda family
(Sun et al. 2023, "A Simple and Effective Pruning Approach for LLMs").

The importance of a weight is judged from the WEIGHT MATRICES THEMSELVES
(per-tensor magnitude statistics, including a dedicated report for the last
layer's matrix) - NOT from model outputs, so no forward passes are required
(this matches the "看最后一层矩阵而不是输出" requirement; the per-layer
threshold adapts to each tensor's own magnitude distribution).

Steps:
  1. Backup: SHA-256 of the original file (+ optional copy) -> manifest.json.
  2. Analyze: per-tensor magnitude statistics (mean|W|, percentiles) + a
     dedicated LAST-LAYER matrix report.
  3. Mask:   keep-mask (1 = keep, 0 = zero) saved as .npz, per-tensor threshold
             = the tensor's own (100 * sparsity)-th magnitude percentile.
  4. Prune:  best-effort pruned GGUF (same metadata, pruned entries zeroed).
             Zeroed entries enable the block-sparse matmul skip (see
             phoenix/sparse_block_matmul.hpp). Verify this step against the
             installed gguf-py API before trusting it.

Requires: pip install gguf numpy
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np


def load_gguf():
    try:
        from gguf import GGUFReader, GGUFWriter  # noqa: F401
        return GGUFReader, GGUFWriter
    except ImportError:
        sys.exit("missing dependency: pip install gguf")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_stats(name, data, sparsity):
    flat = np.abs(np.asarray(data, dtype=np.float32).ravel())
    n = flat.size
    thr = float(np.percentile(flat, 100.0 * sparsity)) if n else 0.0
    return {
        "tensor": name,
        "count": int(n),
        "meanAbs": float(flat.mean()) if n else 0.0,
        "maxAbs": float(flat.max()) if n else 0.0,
        "p50": float(np.percentile(flat, 50)) if n else 0.0,
        "p90": float(np.percentile(flat, 90)) if n else 0.0,
        "p99": float(np.percentile(flat, 99)) if n else 0.0,
        "threshold": thr,
        "zeroed": int(np.sum(flat < thr)) if n else 0,
    }


def write_pruned_gguf(reader, masks, out_path):
    """Best-effort GGUF rewrite.  Verify against your gguf-py version."""
    _, GGUFWriter = load_gguf()
    arch = None
    for k, v in reader.fields.items():
        if k == "general.architecture":
            arch = v
    if arch is None:
        sys.exit("general.architecture not found in GGUF metadata")
    writer = GGUFWriter(out_path, arch=arch)
    # Generic metadata copy: dispatch on the field value type.
    from gguf.constants import GGUFValueType
    type_names = {
        GGUFValueType.UINT8: "add_uint8",
        GGUFValueType.INT8: "add_int8",
        GGUFValueType.UINT16: "add_uint16",
        GGUFValueType.INT16: "add_int16",
        GGUFValueType.UINT32: "add_uint32",
        GGUFValueType.INT32: "add_int32",
        GGUFValueType.FLOAT32: "add_float32",
        GGUFValueType.BOOL: "add_bool",
        GGUFValueType.STRING: "add_string",
        GGUFValueType.ARRAY: "add_array",
        GGUFValueType.UINT64: "add_uint64",
        GGUFValueType.INT64: "add_int64",
        GGUFValueType.FLOAT64: "add_float64",
    }
    skipped = []
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue
        parts = field.types
        add_name = type_names.get(parts[-1])
        if add_name is None:
            skipped.append(key)
            continue
        try:
            getattr(writer, add_name)(key, field.parts[-1])
        except Exception:
            skipped.append(key)
    for tensor in reader.tensors:
        data = np.asarray(tensor.data, dtype=np.float32)
        keep = masks.get(tensor.name)
        pruned = np.where(keep, data, np.float32(0.0))
        writer.add_tensor(tensor.name, pruned)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    return skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="input .gguf path")
    ap.add_argument("--sparsity", type=float, default=0.3,
                    help="fraction of weights to zero per tensor (default 0.3)")
    ap.add_argument("--backup-dir", default=None,
                    help="directory to store the original copy + manifest.json")
    ap.add_argument("--mask", default=None, help="save keep-masks to this .npz")
    ap.add_argument("--output", default=None,
                    help="best-effort pruned .gguf output path")
    ap.add_argument("--report", default=None, help="JSON report path")
    args = ap.parse_args()

    GGUFReader, _ = load_gguf()
    reader = GGUFReader(args.model)
    sparsity = max(0.0, min(0.9, args.sparsity))

    manifest = {
        "model": os.path.abspath(args.model),
        "sha256": sha256_file(args.model),
        "sparsity": sparsity,
    }
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)
        backup_path = os.path.join(args.backup_dir, "original.gguf")
        if not os.path.exists(backup_path):
            shutil.copy2(args.model, backup_path)
        manifest["backupCopy"] = os.path.abspath(backup_path)
        with open(os.path.join(args.backup_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    reports = []
    masks = {}
    last_tensor = list(reader.tensors.keys())[-1] if reader.tensors else None
    for name, tensor in reader.tensors.items():
        data = np.asarray(tensor.data, dtype=np.float32)
        flat = np.abs(data.ravel())
        thr = float(np.percentile(flat, 100.0 * sparsity)) if flat.size else 0.0
        masks[name] = (flat >= thr).reshape(data.shape)
        rep = tensor_stats(name, data, sparsity)
        rep["isLastLayerMatrix"] = (name == last_tensor)
        reports.append(rep)

    # Dedicated LAST-LAYER matrix report (per the requirement: judge from the
    # last layer's matrix, not from model outputs).
    if last_tensor:
        print("=== LAST-LAYER MATRIX: %s ===" % last_tensor)
        r = tensor_stats(last_tensor, reader.tensors[last_tensor].data, sparsity)
        for k, v in r.items():
            print("  %-12s %s" % (k, v))

    print("=== ALL TENSORS ===")
    for r in reports:
        print("  %-46s count=%-9d meanAbs=%.4g zeroed=%d" %
              (r["tensor"], r["count"], r["meanAbs"], r["zeroed"]))

    if args.mask:
        np.savez_compressed(args.mask,
                            **{k: v.astype(np.uint8) for k, v in masks.items()})
        print("masks ->", args.mask)

    if args.output:
        skipped = write_pruned_gguf(reader, masks, args.output)
        print("pruned GGUF ->", args.output)
        if skipped:
            print("WARN skipped metadata fields:", skipped)

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"manifest": manifest, "tensors": reports}, f, indent=2)
        print("report ->", args.report)


if __name__ == "__main__":
    main()

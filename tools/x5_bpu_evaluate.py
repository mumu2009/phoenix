#!/usr/bin/env python3
"""X5 BPU batch evaluator.

Evaluates one or more Horizon BPU ``.bin`` models against a batch of inputs
and reports the average MSE against the target tensor.

Supports:
  - ``.npy`` input/target batches (legacy / local ORT debugging)
  - raw float32 ``.bin`` batches with an explicit per-sample ``--input-shape``
    and ``--target-shape``
  - single ``--bin`` evaluation
  - directory ``--bin-dir`` evaluation (backward compatible)

Usage (single .bin with .bin batch):
    python3 x5_bpu_evaluate.py \
        --bin /home/sunrise/phoenix/evolve/speech_encoder/round_0000/candidate_0000.bin \
        --inputs /home/sunrise/phoenix/evolve/speech_encoder/round_0000_inputs.bin \
        --targets /home/sunrise/phoenix/evolve/speech_encoder/round_0000_targets.bin \
        --input-shape 1x1x1x16000 \
        --target-shape 1x1x1x15872 \
        --format bin \
        --out /home/sunrise/phoenix/evolve/losses.json

Usage (directory mode, .npy):
    python3 x5_bpu_evaluate.py \
        --bin-dir /home/sunrise/phoenix/evolve/speech_gen_0/bins \
        --inputs /home/sunrise/phoenix/evolve/speech_inputs.npy \
        --targets /home/sunrise/phoenix/evolve/speech_targets.npy \
        --out /home/sunrise/phoenix/evolve/speech_gen_0_losses.json
"""

import argparse
import glob
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from hobot_dnn import pyeasy_dnn as dnn
except Exception as e:
    print(f"[ERROR] cannot import hobot_dnn/pyeasy_dnn: {e}", file=sys.stderr)
    sys.exit(1)


def parse_shape(s: str):
    """Parse '1x1x1x16000' or '1,1,1,16000' into a tuple of ints."""
    if not s:
        return None
    parts = re.split(r"[x,]", s)
    return tuple(int(p.strip()) for p in parts if p.strip())


def load_batch(path: str, shape=None, format_hint=None):
    """Load an input or target batch.

    For ``.npy`` files, the first dimension is the batch.
    For ``.bin`` files, ``shape`` is the per-sample shape (including batch=1).
    The total file length determines the actual batch size.
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


def evaluate_bin(bin_path: str, inputs: np.ndarray, targets: np.ndarray) -> float:
    """Return average MSE of a single .bin over all input/target pairs."""
    models = dnn.load(bin_path)
    if not models:
        raise RuntimeError(f"failed to load {bin_path}")
    model = models[0]

    total = 0.0
    count = 0
    for inp, tgt in zip(inputs, targets):
        inp = np.asarray(inp, dtype=np.float32)
        tgt = np.asarray(tgt, dtype=np.float32)

        out_tensors = model.forward([inp])
        if not out_tensors:
            raise RuntimeError(f"{bin_path} produced no output")
        out = np.asarray(out_tensors[0].buffer, dtype=np.float32)

        # Match target length / shape robustly.  Some BPU models return the
        # output with a trailing length that may not exactly match the target
        # dimensions.  Flatten both and compute MSE over the common prefix.
        out = out.reshape(-1)
        tgt = tgt.reshape(-1)
        min_len = min(out.size, tgt.size)
        if out.size != tgt.size:
            print(f"  [shape] {bin_path} out={out.size} tgt={tgt.size} using first {min_len}")
        out = out[:min_len]
        tgt = tgt[:min_len]

        mse = float(np.mean((out - tgt) ** 2))
        total += mse
        count += 1

    if count == 0:
        return float("inf")
    return total / count


def main() -> int:
    parser = argparse.ArgumentParser(description="X5 BPU batch evaluator")
    parser.add_argument("--bin-dir", help="Directory with .bin models")
    parser.add_argument("--bin", help="Single .bin model to evaluate")
    parser.add_argument("--inputs", required=True, help=".npy or .bin batch inputs")
    parser.add_argument("--targets", required=True, help=".npy or .bin batch targets")
    parser.add_argument("--out", default="/tmp/bpu_evaluate.json", help="JSON output")
    parser.add_argument("--pattern", default="*.bin", help=".bin glob pattern")
    parser.add_argument("--input-shape", default=None, help="Per-sample input shape, e.g. 1x1x1x16000")
    parser.add_argument("--target-shape", default=None, help="Per-sample target shape, e.g. 1x1x1x15872")
    parser.add_argument("--format", choices=["auto", "npy", "bin"], default="auto",
                        help="Input/target batch format")
    args = parser.parse_args()

    if args.bin_dir and args.bin:
        print("[ERROR] use either --bin-dir or --bin, not both", file=sys.stderr)
        return 1

    input_shape = parse_shape(args.input_shape)
    target_shape = parse_shape(args.target_shape)

    try:
        inputs = load_batch(args.inputs, input_shape, args.format)
        targets = load_batch(args.targets, target_shape, args.format)
    except Exception as e:
        print(f"[ERROR] failed to load batch: {e}", file=sys.stderr)
        return 1

    if not args.bin_dir and not args.bin:
        print("[ERROR] either --bin-dir or --bin is required", file=sys.stderr)
        return 1

    if args.bin:
        bin_paths = [args.bin]
    else:
        bin_dir = Path(args.bin_dir)
        if not bin_dir.is_dir():
            print(f"[ERROR] {bin_dir} is not a directory", file=sys.stderr)
            return 1
        bin_paths = sorted(glob.glob(str(bin_dir / args.pattern)))
        if not bin_paths:
            print(f"[ERROR] no .bin files in {bin_dir}", file=sys.stderr)
            return 1

    # Evaluate each .bin in its own child process so a BPU runtime crash/segfault
    # in one model does not abort the whole round.
    results = {}
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for bin_path in bin_paths:
        name = os.path.basename(bin_path)
        tmp_out = out_dir / f".tmp_{Path(name).stem}_eval.json"
        cmd = [
            sys.executable, __file__,
            "--bin", bin_path,
            "--inputs", args.inputs,
            "--targets", args.targets,
            "--out", str(tmp_out),
            "--input-shape", args.input_shape or "",
            "--target-shape", args.target_shape or "",
            "--format", args.format,
        ]
        try:
            rc = subprocess.run(cmd, timeout=600, check=False).returncode
        except Exception as e:
            rc = -1
            print(f"{name}: subprocess exception {e}")
        if rc == 0 and tmp_out.is_file():
            try:
                with open(tmp_out, "r", encoding="utf-8") as f:
                    one_result = json.load(f)
                results.update(one_result)
            except Exception as e:
                results[name] = {"loss": float("inf"), "ok": False, "error": f"bad json: {e}"}
                print(f"{name}: ERROR reading json: {e}")
            tmp_out.unlink(missing_ok=True)
        else:
            results[name] = {"loss": float("inf"), "ok": False, "error": f"subprocess rc={rc}"}
            print(f"{name}: ERROR subprocess rc={rc}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[DONE] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

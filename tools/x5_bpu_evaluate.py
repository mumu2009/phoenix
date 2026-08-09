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
import subprocess
import sys
from pathlib import Path

import numpy as np

from edge_evaluate_common import evaluate_batch_mse, load_batch, parse_shape

try:
    from hobot_dnn import pyeasy_dnn as dnn
except Exception as e:
    print(f"[ERROR] cannot import hobot_dnn/pyeasy_dnn: {e}", file=sys.stderr)
    sys.exit(1)


def evaluate_bin(bin_path: str, inputs: np.ndarray, targets: np.ndarray) -> float:
    """Return average MSE of a single .bin over all input/target pairs."""
    models = dnn.load(bin_path)
    if not models:
        raise RuntimeError(f"failed to load {bin_path}")
    model = models[0]

    def infer(inp: np.ndarray) -> np.ndarray:
        out_tensors = model.forward([inp])
        if not out_tensors:
            raise RuntimeError(f"{bin_path} produced no output")
        return np.asarray(out_tensors[0].buffer, dtype=np.float32)

    return evaluate_batch_mse(inputs, targets, infer, mismatch_context=bin_path)


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
        # Single-file mode: evaluate directly in this process (no fork).
        # This is the leaf invocation spawned by --bin-dir mode below.
        # It MUST NOT subprocess itself again or it creates an infinite fork
        # recursion that eats all RAM on the X5 until OOM-killer fires.
        name = os.path.basename(args.bin)
        results = {}
        try:
            loss = evaluate_bin(args.bin, inputs, targets)
            results[name] = {"loss": loss, "ok": True}
            print(f"{name}: loss={loss:.6f}")
        except Exception as e:
            results[name] = {"loss": float("inf"), "ok": False, "error": str(e)}
            print(f"{name}: ERROR {e}")

        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return 0

    # --bin-dir mode: evaluate each .bin in its own child process so a BPU
    # runtime crash/segfault in one model does not abort the whole round.
    bin_dir = Path(args.bin_dir)
    if not bin_dir.is_dir():
        print(f"[ERROR] {bin_dir} is not a directory", file=sys.stderr)
        return 1
    bin_paths = sorted(glob.glob(str(bin_dir / args.pattern)))
    if not bin_paths:
        print(f"[ERROR] no .bin files in {bin_dir}", file=sys.stderr)
        return 1

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
            # subprocess.run with timeout kills the child on TimeoutExpired,
            # then raises.  We catch it and record failure cleanly.
            rc = subprocess.run(cmd, timeout=600, check=False).returncode
        except subprocess.TimeoutExpired:
            rc = -1
            print(f"{name}: TIMEOUT after 600s (child killed)")
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

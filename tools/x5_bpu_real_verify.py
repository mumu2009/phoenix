#!/usr/bin/env python3
"""Real-input BPU verification on RDK X5.

Loads the four additive-JEPA .bin models, feeds synthetic but non-constant
image/audio tensors, and checks that the outputs are:
  - finite
  - not all zero
  - have the expected shape
  - have non-trivial variance (so the model is not a no-op)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from hobot_dnn import pyeasy_dnn as dnn
except Exception as e:
    print(f"[ERROR] cannot import hobot_dnn/pyeasy_dnn: {e}", file=sys.stderr)
    sys.exit(1)


def tensor_stats(arr: np.ndarray):
    return {
        "shape": list(arr.shape),
        "size": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "finite": bool(np.all(np.isfinite(arr))),
        "nonzero": bool(np.any(np.abs(arr) > 1e-7)),
    }


def run_model(bin_path: Path, inp: np.ndarray):
    models = dnn.load(str(bin_path))
    if not models:
        raise RuntimeError(f"failed to load {bin_path}")
    model = models[0]
    # Ensure the input is a plain numpy array (not a hobot_dnn Tensor).
    if not isinstance(inp, np.ndarray):
        inp = np.asarray(inp)
    outs = model.forward([inp])
    if not outs:
        raise RuntimeError(f"{bin_path}: forward returned no outputs")
    return np.asarray(outs[0].buffer, dtype=np.float32)


def make_vision_input(res: int = 224):
    """NHWC float32 RGB image with a gradient."""
    img = np.zeros((1, res, res, 3), dtype=np.float32)
    for y in range(res):
        for x in range(res):
            img[0, y, x, 0] = x / res
            img[0, y, x, 1] = y / res
            img[0, y, x, 2] = 0.5
    return img


def make_audio_input(samples: int = 16000):
    """1-D float32 waveform with a 1 kHz sine at 16 kHz."""
    t = np.arange(samples, dtype=np.float32) / 16000.0
    wave = np.sin(2 * math.pi * 1000.0 * t).astype(np.float32)
    # Model expects [1,1,16000,1] from the x5_bpu_smoke printout.
    return wave.reshape(1, 1, samples, 1)


def make_concept(concept: int = 128):
    c = np.random.randn(1, 1, 1, concept).astype(np.float32) * 0.5
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/root/x5_smoke/models", help="dir with additive_jepa .bin variants")
    p.add_argument("--out", default="/root/x5_smoke/x5_bpu_real_verify_report.json")
    args = p.parse_args()

    root = Path(args.model_dir)
    report = {}
    checks = [
        ("vision_encoder", root / "vision_encoder" / "best.bin", make_vision_input(), [1, 128, 1, 1]),
        ("vision_decoder", root / "vision_decoder" / "best.bin", make_concept(), [1, 3, 224, 224]),
        ("speech_encoder", root / "speech_encoder" / "best.bin", make_audio_input(), [1, 128, 1, 1]),
        ("speech_decoder", root / "speech_decoder" / "best.bin", make_concept(), [1, 1, 1, 15872]),
    ]

    all_ok = True
    for name, bin_path, inp, expected_shape in checks:
        entry = {"bin": str(bin_path)}
        if not bin_path.is_file():
            entry["ok"] = False
            entry["error"] = f"missing {bin_path}"
            report[name] = entry
            all_ok = False
            continue
        try:
            out = run_model(bin_path, inp)
            stats = tensor_stats(out)
            entry["output"] = stats
            entry["expected_shape"] = expected_shape
            ok = (list(out.shape) == expected_shape and stats["finite"] and
                  stats["nonzero"] and stats["std"] > 1e-7)
            entry["ok"] = ok
            if not ok:
                all_ok = False
        except Exception as e:
            entry["ok"] = False
            entry["error"] = str(e)
            all_ok = False
        report[name] = entry

    report["all_ok"] = all_ok
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

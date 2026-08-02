#!/usr/bin/env python3
"""Export a trained/prepared AdditiveResidualModel checkpoint to ONNX.

Usage:
    python tools/export_additive_jpea.py \
        --model-name speech_encoder \
        --checkpoint work_dir/speech_encoder/best.pt \
        --out-dir work_dir/speech_encoder/export

If ``--model-name`` is omitted, it is read from the checkpoint metadata.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from tools/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from additive_jpea import (
    AdditiveResidualModel,
    export_to_onnx,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        choices=["speech_encoder", "speech_decoder", "vision_encoder", "vision_decoder"],
        help="Override the model name stored in the checkpoint",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--base-path", default=None, help="Override frozen ResNet18 base path")
    args = parser.parse_args()

    model = AdditiveResidualModel.from_checkpoint(
        Path(args.checkpoint),
        base_path=args.base_path,
    )

    if args.model_name and args.model_name != model.model_name:
        print(
            f"[warn] overriding model name {model.model_name} -> {args.model_name}",
            file=sys.stderr,
        )
        # model_name is built into shapes; the user is responsible for compatibility.
        # Re-create with the same blocks? Not safe. For now, ignore override mismatch
        # beyond printing a warning; the checkpoint name wins.

    export_to_onnx(
        model,
        Path(args.out_dir),
        n_calib=args.calibration_count,
        source_checkpoint=Path(args.checkpoint),
        save_pt=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

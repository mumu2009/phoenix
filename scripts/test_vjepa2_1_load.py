#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a V-JEPA 2.1 checkpoint from the local models/vjepa2_1/ directory
and run a dummy forward pass to verify the host environment.

The official torch.hub loader tries to download weights from a hardcoded
localhost test URL in the current upstream main branch, so this script
instantiates the model with pretrained=False and loads our locally
downloaded .pt weights manually.
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "models" / "vjepa2_1"

VARIANT_TO_HUB_NAME = {
    "vitb": "vjepa2_1_vit_base_384",
    "vitl": "vjepa2_1_vit_large_384",
    "vitg": "vjepa2_1_vit_giant_384",
    "vitG": "vjepa2_1_vit_gigantic_384",
}

# Windows filenames are case-insensitive: vitg and vitG would collide.
LOCAL_FILENAME = {
    "vitb": "vjepa2_1_vitb_384.pt",
    "vitl": "vjepa2_1_vitl_384.pt",
    "vitg": "vjepa2_1_vitg_384.pt",
    "vitG": "vjepa2_1_vitG_2b_384.pt",
}

# Use the cached torch.hub copy of facebookresearch/vjepa2 to avoid GitHub timeouts.
LOCAL_HUB_DIR = Path(torch.hub.get_dir()) / "facebookresearch_vjepa2_main"


def clean_state_dict(state_dict):
    """Remove module/backbone prefixes that come from DataParallel wrapping."""
    out = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "").replace("backbone.", "")
        out[new_k] = v
    return out


def load_local_checkpoint(variant: str):
    hub_name = VARIANT_TO_HUB_NAME[variant]
    print(f"Instantiating model architecture via torch.hub: {hub_name}")
    print(f"Local hub source: {LOCAL_HUB_DIR}")
    if not LOCAL_HUB_DIR.exists():
        raise FileNotFoundError(
            f"Cached V-JEPA repo not found at {LOCAL_HUB_DIR}. "
            "Run once with network to let torch.hub download it, or clone manually."
        )
    # pretrained=False avoids the broken localhost URL in upstream backbones.py.
    # source='local' avoids GitHub API calls.
    encoder, predictor = torch.hub.load(
        str(LOCAL_HUB_DIR),
        hub_name,
        pretrained=False,
        source="local",
        trust_repo=True,
        force_reload=False,
    )
    print("Architecture instantiated.")

    pt_file = SAVE_DIR / LOCAL_FILENAME[variant]
    if not pt_file.exists():
        raise FileNotFoundError(f"Local checkpoint not found: {pt_file}")

    print(f"Loading local checkpoint: {pt_file} ({pt_file.stat().st_size / (1024**3):.2f} GB)")
    ckpt = torch.load(pt_file, map_location="cpu", weights_only=False)
    print("Checkpoint top-level keys:", list(ckpt.keys()))

    # Distilled V-JEPA 2.1 variants (ViT-B/L) use ema_encoder; g/G use target_encoder.
    encoder_key = None
    for key in ("ema_encoder", "target_encoder", "encoder"):
        if key in ckpt:
            encoder_key = key
            break
    if encoder_key is None:
        raise KeyError("No suitable encoder key found in checkpoint")
    print(f"Using checkpoint key for encoder: {encoder_key}")

    encoder_state = clean_state_dict(ckpt[encoder_key])
    encoder.load_state_dict(encoder_state, strict=False)
    print("Encoder weights loaded.")

    if "predictor" in ckpt:
        predictor_state = clean_state_dict(ckpt["predictor"])
        predictor.load_state_dict(predictor_state, strict=False)
        print("Predictor weights loaded.")

    return encoder, predictor


def run_dummy_forward(model, device="cpu", num_frames=8, img_size=384):
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, num_frames, img_size, img_size, device=device)
    print(f"Dummy input shape: {dummy.shape}")
    with torch.no_grad():
        t0 = time.time()
        out = model(dummy)
        t1 = time.time()
    if isinstance(out, tuple):
        out = out[0]
    print(f"Encoder output shape: {out.shape}")
    print(f"Inference time: {t1 - t0:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_TO_HUB_NAME.keys()),
        default="vitg",
        help="V-JEPA 2.1 variant to load",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()

    try:
        encoder, _ = load_local_checkpoint(args.variant)
        run_dummy_forward(encoder, device=args.device, num_frames=args.num_frames)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

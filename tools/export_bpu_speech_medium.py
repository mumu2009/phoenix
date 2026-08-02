#!/usr/bin/env python3
"""Export a medium, BPU-friendly 1D speech autoencoder to ONNX.

Architecture is larger than the small 70k autoencoder while still
being suitable for black-box / evolutionary search on the RDK X5 BPU
(march=bayes-e).

Input:  [N, 1, 1, 16000]  (16 kHz mono waveform)
Output: [N, 1, 1, 16000]  (reconstructed waveform, linear / no tanh)
Bottleneck concept: [N, 64, 1, 1]
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CHUNK = 16000
CONCEPT = 64


class MediumSpeechEncoder(nn.Module):
    """1x1x1x16000 -> 1x64x1x1 concept."""
    def __init__(self, concept: int = CONCEPT):
        super().__init__()
        self.concept = concept
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, concept)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,1,1,16000]
        x = self.conv(x)          # [N,64,1,62]
        x = self.gap(x)           # [N,64,1,1]
        x = x.view(-1, 64)        # [N,64]  (dynamic batch)
        x = self.fc(x)            # [N,concept]
        # BPU-friendly 4D concept tensor.
        return x.view(-1, self.concept, 1, 1)


class MediumSpeechDecoder(nn.Module):
    """1xconceptx1x1 concept -> 1x1x1x16000 waveform, linear output."""
    def __init__(self, concept: int = CONCEPT):
        super().__init__()
        self.concept = concept
        # concept -> 8*1*250 = 2000, then 3 ConvTranspose of stride 4 to reach 16000.
        self.fc = nn.Linear(concept, 8 * 1 * 250)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(1, 4), stride=(1, 4)),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(1, 4), stride=(1, 4)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, concept, 1, 1]
        x = x.view(-1, self.concept)
        x = self.fc(x)                    # [N,2000]
        x = x.view(-1, 8, 1, 250)         # [N,8,1,250]
        x = self.deconv(x)                # [N,1,1,16000]
        return x


class MediumSpeechAutoencoder(nn.Module):
    def __init__(self, concept: int = CONCEPT):
        super().__init__()
        self.encoder = MediumSpeechEncoder(concept)
        self.decoder = MediumSpeechDecoder(concept)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def export_onnx(model: nn.Module, out_dir: Path, concept: int = CONCEPT):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        dummy_enc = torch.randn(1, 1, 1, CHUNK)
        dummy_dec = torch.randn(1, concept, 1, 1)
        torch.onnx.export(
            model.encoder, dummy_enc, out_dir / "model_encoder.onnx",
            input_names=["waveform"], output_names=["concept"],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={"waveform": {0: "batch"}, "concept": {0: "batch"}},
        )
        torch.onnx.export(
            model.decoder, dummy_dec, out_dir / "model_decoder.onnx",
            input_names=["concept"], output_names=["reconstruction"],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={"concept": {0: "batch", 2: "height", 3: "width"},
                          "reconstruction": {0: "batch", 2: "height", 3: "width"}},
        )
    print(f"[export] wrote {out_dir}/model_encoder.onnx and model_decoder.onnx")


def write_calibration(out_dir: Path, name: str, shape: tuple, count: int = 10):
    calib_dir = out_dir / f"calibration_{name}"
    calib_dir.mkdir(parents=True, exist_ok=True)
    numel = int(np.prod(shape))
    rng = np.random.default_rng(42)
    for i in range(count):
        arr = rng.normal(0.0, 0.5, size=shape).astype(np.float32)
        with open(calib_dir / f"cal_{i:04d}.bin", "wb") as f:
            f.write(struct.pack(f"<{numel}f", *arr.flatten()))
    print(f"[calibration] wrote {count} samples to {calib_dir}")
    return calib_dir


def write_manifest(out_dir: Path, concept: int, chunk: int):
    manifest = {
        "name": "speech_medium_16k",
        "concept_dim": concept,
        "chunk_size": chunk,
        "encoder_onnx": str(out_dir / "model_encoder.onnx"),
        "decoder_onnx": str(out_dir / "model_decoder.onnx"),
    }
    with open(out_dir / "model.manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {out_dir}/model.manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", "--output-dir", dest="work_dir",
                        default="/media/sf_phoenix/speech_evolve/init5")
    parser.add_argument("--concept", type=int, default=CONCEPT)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = MediumSpeechAutoencoder(args.concept)
    export_onnx(model, out_dir, args.concept)

    # Save PyTorch state dict for later training / transfer.
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"[state] wrote {out_dir}/model.pt")

    # Calibration data for hb_mapper.
    write_calibration(out_dir, "encoder", (1, 1, 1, CHUNK), args.calibration_count)
    write_calibration(out_dir, "decoder", (1, args.concept, 1, 1), args.calibration_count)
    write_manifest(out_dir, args.concept, CHUNK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

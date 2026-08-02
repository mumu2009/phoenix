#!/usr/bin/env python3
"""Export a 1D JPEA-v2 speech autoencoder to BPU-friendly ONNX.

The model treats a 1-second 16 kHz waveform as a 2D tensor with height=1,
which allows the same Conv2d/ConvTranspose2d ops used by the image pipeline.
Encoder: 1x1x1x16000 -> 1x128
Decoder: 1x128x1x1  -> 1x1x1x15872  (C++ pads the final 128 samples to 16000)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 400
STRIDE_SAMPLES = 160
CLIP_SECONDS = 1
LENGTH = SAMPLE_RATE * CLIP_SECONDS  # 16000
ENCODER_OUT_LEN = 62


class SpeechEncoder(nn.Module):
    def __init__(self, concept_dim=128):
        super().__init__()
        # Treat waveform as NCHW with H=1, W=LENGTH.
        # k=4, s=4, p=0: 16000 -> 4000 -> 1000 -> 250 -> 62
        self.conv0 = nn.Conv2d(1, 32, (1, 4), stride=(1, 4), bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (1, 4), stride=(1, 4), bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, (1, 4), stride=(1, 4), bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, (1, 4), stride=(1, 4), bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, concept_dim)

    def forward(self, x):
        x = torch.relu(self.bn0(self.conv0(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SpeechDecoder(nn.Module):
    def __init__(self, concept_dim=128):
        super().__init__()
        # Project concept to an initial temporal feature map (B,256,1,62).
        self.fc = nn.Linear(concept_dim, 256 * ENCODER_OUT_LEN)
        # k=4, s=4, p=0: 62 -> 248 -> 992 -> 3968 -> 15872
        self.conv0 = nn.ConvTranspose2d(256, 128, (1, 4), stride=(1, 4), bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, (1, 4), stride=(1, 4), bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, (1, 4), stride=(1, 4), bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 1, (1, 4), stride=(1, 4), bias=True)

    def forward(self, x):
        # x: (B, 128, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 1, ENCODER_OUT_LEN)
        x = torch.relu(self.bn0(self.conv0(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.conv6(x)  # (B, 1, 1, 15872)
        return x


def save_floats(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.astype(np.float32).tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--concept-dim', type=int, default=128)
    parser.add_argument('--calibration-count', type=int, default=100)
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    enc = SpeechEncoder(args.concept_dim)
    dec = SpeechDecoder(args.concept_dim)
    enc.eval()
    dec.eval()

    dummy_enc = torch.zeros(1, 1, 1, LENGTH)
    dummy_dec = torch.zeros(1, args.concept_dim, 1, 1)

    with torch.no_grad():
        torch.onnx.export(enc, dummy_enc, os.path.join(out_dir, 'model_encoder.onnx'),
                          input_names=['waveform'], output_names=['concept'],
                          opset_version=11, dynamo=False)
        torch.onnx.export(dec, dummy_dec, os.path.join(out_dir, 'model_decoder.onnx'),
                          input_names=['concept'], output_names=['reconstruction'],
                          opset_version=11, dynamo=False)

    # Generate synthetic calibration data: 100 random waveforms and concepts.
    calib_enc_dir = os.path.join(out_dir, 'calibration_encoder')
    calib_dec_dir = os.path.join(out_dir, 'calibration_decoder')
    os.makedirs(calib_enc_dir, exist_ok=True)
    os.makedirs(calib_dec_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    for i in range(args.calibration_count):
        wave = rng.normal(0, 0.5, (1, 1, 1, LENGTH)).astype(np.float32)
        concept = rng.normal(0, 1.0, (1, args.concept_dim, 1, 1)).astype(np.float32)
        save_floats(wave, os.path.join(calib_enc_dir, f'{i:04d}.bin'))
        save_floats(concept, os.path.join(calib_dec_dir, f'{i:04d}.bin'))

    print('Saved ONNX and calibration to', out_dir)


if __name__ == '__main__':
    main()

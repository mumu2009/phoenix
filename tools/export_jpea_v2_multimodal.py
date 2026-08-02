"""Unified ONNX export for trained JPEA-v2 speech or vision checkpoints.

Usage:
    python tools/export_jpea_v2_multimodal.py \
        --modality speech|image \
        --checkpoint checkpoints/jpea_v2_speech/best.pt \
        --out-dir runtime_store/models/ijepa/my_variant

The script reads the architecture metadata saved by the training scripts,
reconstructs the encoder/decoder, and writes:
  - model_encoder.onnx
  - model_decoder.onnx
  - calibration_encoder/   (bin float32 files)
  - calibration_decoder/
  - model.manifest.json
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch

from train_jpea_v2_speech import (
    JpeaV2SpeechAutoencoder, CHUNK as SPEECH_CHUNK, TARGET as SPEECH_TARGET,
)
from train_jpea_v2_vision import (
    JpeaV2ImageAutoencoder, RESOLUTION as IMAGE_RES,
    IMAGENET_MEAN, IMAGENET_STD,
)


def write_calibration(out_dir: Path, name: str, shape: tuple, count: int = 10, scale=0.5, mean=0.0):
    calib_dir = out_dir / f'calibration_{name}'
    calib_dir.mkdir(parents=True, exist_ok=True)
    numel = int(np.prod(shape))
    rng = np.random.default_rng(42)
    for i in range(count):
        arr = rng.normal(mean, scale, size=shape).astype(np.float32)
        with open(calib_dir / f'cal_{i:04d}.bin', 'wb') as f:
            f.write(struct.pack(f'<{numel}f', *arr.flatten()))
    print(f'[calibration] wrote {count} samples to {calib_dir}')
    return calib_dir


def export_speech(ckpt, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    concept = ckpt.get('concept', 128)
    enc_channels = ckpt['enc_channels']
    dec_channels = ckpt['dec_channels']
    blocks = ckpt.get('blocks', 0)

    model = JpeaV2SpeechAutoencoder(enc_channels, dec_channels, concept=concept, blocks=blocks)
    model.load_state_dict(ckpt['model'])
    model.eval().cpu()

    with torch.no_grad():
        dummy_enc = torch.randn(1, 1, 1, SPEECH_CHUNK)
        torch.onnx.export(
            model.encoder,
            dummy_enc,
            out_dir / 'model_encoder.onnx',
            input_names=['waveform'],
            output_names=['concept'],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={'waveform': {0: 'batch'}, 'concept': {0: 'batch'}},
        )
        dummy_dec = torch.randn(1, concept, 1, 1)
        torch.onnx.export(
            model.decoder,
            dummy_dec,
            out_dir / 'model_decoder.onnx',
            input_names=['concept'],
            output_names=['reconstruction'],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={'concept': {0: 'batch'}, 'reconstruction': {0: 'batch'}},
        )

    write_calibration(out_dir, 'encoder', (1, 1, 1, SPEECH_CHUNK), 10, scale=0.5)
    write_calibration(out_dir, 'decoder', (1, concept, 1, 1), 10, scale=0.5)

    with open(out_dir / 'model.manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'name': 'jpea_v2_speech_16k',
            'modality': 'speech',
            'concept_dim': concept,
            'chunk_size': SPEECH_CHUNK,
            'decoder_output_samples': SPEECH_TARGET,
            'enc_channels': enc_channels,
            'dec_channels': dec_channels,
            'blocks': blocks,
            'checkpoint': str(args.checkpoint),
        }, f, indent=2)
    print(f'[export] speech ONNX -> {out_dir}')


def export_image(ckpt, out_dir: Path, calib_images=None, calib_count=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    concept = ckpt.get('concept', 128)
    resnet = ckpt.get('resnet', 'resnet18')
    dec_channels = ckpt.get('dec_channels')
    dec_depth = ckpt.get('dec_depth', 0)
    resolution = ckpt.get('resolution', IMAGE_RES)

    model = JpeaV2ImageAutoencoder(
        resnet_name=resnet,
        concept=concept,
        pretrained=False,  # weights come from checkpoint
        unfreeze=False,
        dec_width=1.0,
        dec_depth=dec_depth,
        dec_channels=dec_channels,
    )
    model.load_state_dict(ckpt['model'])
    model.eval().cpu()

    base = model.encoder.resnet
    head = model.encoder.head
    feature_dim = model.encoder.feature_dim

    with torch.no_grad():
        dummy_enc = torch.randn(1, 3, resolution, resolution)
        torch.onnx.export(
            base,
            dummy_enc,
            out_dir / 'model_encoder.onnx',
            input_names=['pixel_values'],
            output_names=['embedding'],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={'pixel_values': {0: 'batch'}, 'embedding': {0: 'batch'}},
        )
        dummy_head = torch.randn(1, feature_dim)
        torch.onnx.export(
            head,
            dummy_head,
            out_dir / 'model_encoder_head.onnx',
            input_names=['embedding'],
            output_names=['concept'],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={'embedding': {0: 'batch'}, 'concept': {0: 'batch'}},
        )
        dummy_dec = torch.randn(1, concept, 1, 1)
        torch.onnx.export(
            model.decoder,
            dummy_dec,
            out_dir / 'model_decoder.onnx',
            input_names=['concept'],
            output_names=['reconstruction'],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={'concept': {0: 'batch'}, 'reconstruction': {0: 'batch'}},
        )

    # Save C++-readable head weights (outDim x inDim, row-major).
    head_w = head.weight.detach().cpu().float().contiguous()
    head_b = head.bias.detach().cpu().float().contiguous() if head.bias is not None else torch.zeros(concept)
    with open(out_dir / 'encoder_head.json', 'w', encoding='utf-8') as f:
        json.dump({
            'inDim': feature_dim,
            'outDim': concept,
            'W': head_w.view(-1).tolist(),
            'b': head_b.tolist(),
        }, f, indent=2)

    # Synthetic calibration for the encoder. If real images are later required,
    # replace this with real NCHW normalized frames.
    write_calibration(out_dir, 'encoder', (1, 3, resolution, resolution), calib_count,
                      scale=0.5, mean=0.0)
    # For the decoder, concepts are zero-mean with small std.
    write_calibration(out_dir, 'decoder', (1, concept, 1, 1), calib_count,
                      scale=0.5, mean=0.0)

    with open(out_dir / 'model.manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'name': 'jpea_v2_image_224',
            'modality': 'image',
            'concept_dim': concept,
            'feature_dim': feature_dim,
            'resolution': resolution,
            'resnet': resnet,
            'dec_channels': dec_channels,
            'dec_depth': dec_depth,
            'checkpoint': str(args.checkpoint),
        }, f, indent=2)
    print(f'[export] image ONNX -> {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', required=True, choices=['speech', 'image'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--calib-count', type=int, default=10)
    global args
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' not in ckpt:
        raise ValueError(f'Checkpoint {args.checkpoint} does not contain "model" key')

    if args.modality == 'speech':
        export_speech(ckpt, out_dir)
    elif args.modality == 'image':
        export_image(ckpt, out_dir, calib_count=args.calib_count)
    else:
        raise ValueError(f'Unknown modality {args.modality}')

    print(f'[export] done. Artifacts in {out_dir}')


if __name__ == '__main__':
    main()

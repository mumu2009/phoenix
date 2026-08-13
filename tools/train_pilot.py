"""Quick end-to-end pilot for the audio pipeline.

Runs a tiny (~1 M param) 1-D autoencoder for a few epochs to verify
shapes and then exports ONNX.  Intended for Kali CPU or a small remote GPU.

Example:
    python tools/train_pilot.py \
        --data-dir /home/kali/phoenix/datasets/musan_16k \
        --out-dir /tmp/audio_pilot

After training, continue with:
    python tools/export_multimodal.py --modality speech \
        --checkpoint /tmp/audio_pilot/best.pt \
        --out-dir /tmp/audio_pilot/onnx
    bash tools/compile_bpu.sh --modality speech \
        --onnx-dir /tmp/audio_pilot/onnx \
        --out-dir /tmp/audio_pilot/bin
"""

import argparse
import sys
from pathlib import Path

from train_audio import main as train_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', default='checkpoints/audio_pilot')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-files', type=int, default=10,
                        help='limit files for a quick smoke test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--compile', action='store_true', help='try torch.compile')
    args = parser.parse_args()

    # Hand off to train_audio.py with tiny scale + safe defaults.
    sys.argv = [
        'train_audio.py',
        '--data-dir', args.data_dir,
        '--out-dir', args.out_dir,
        '--scale', 'tiny',
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', '1e-3',
        '--grad-accum', '1',
        '--device', args.device,
        '--num-workers', '0',
        '--no-pin-memory',
        '--save-every', '0',
    ]
    if args.compile:
        sys.argv.append('--compile')
    if args.max_files:
        sys.argv += ['--max-files', str(args.max_files)]

    train_main()

    print('\n[pilot] training complete.')
    print(f'[pilot] next: export  -> python tools/export_multimodal.py --modality speech --checkpoint {args.out_dir}/best.pt --out-dir {args.out_dir}/onnx')
    print(f'[pilot] next: compile -> bash tools/compile_bpu.sh --modality speech --onnx-dir {args.out_dir}/onnx --out-dir {args.out_dir}/bin')
    print(f'[pilot] next: copy {args.out_dir}/bin/model_*.bin to the X5 under runtime_store/models/ijepa/speech_16k/')


if __name__ == '__main__':
    main()

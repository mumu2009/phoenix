#!/usr/bin/env python3
"""Legacy speech-only export wrapper.

For new work use `export_multimodal.py --modality speech`.
This script exists to keep old commands working with the scalable speech
autoencoder.
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    cmd = [
        sys.executable, 'tools/export_multimodal.py',
        '--modality', 'speech',
        '--checkpoint', args.checkpoint,
        '--out-dir', args.out_dir,
    ]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

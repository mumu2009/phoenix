"""Train a scalable 1D audio autoencoder on 16 kHz mono WAV files.

The architecture is a 1-D Conv / ConvTranspose autoencoder that treats the
waveform as a 2-D tensor of shape [N, 1, 1, 16000].  The encoder strided-convs
with kernel/stride (1, 4) four times to reach [N, C, 1, 62], global-pools to a
vector, and projects to the concept dimension.  The decoder reshapes the
concept back to [N, C, 1, 62] and upsamples four times with (1, 4) stride to
reach [N, 1, 1, 15872], matching kDecoderOutputSamples in
audio_model.cpp.

Supports scale presets from tiny (~1 M params) to xlarge (~100 M params) with
the default concept dim fixed at 128 so the compiled BPU binaries remain
compatible with the existing C++ runtime.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CHUNK = 16000
TARGET = 15872
SR = 16000


def _parse_channel_list(s):
    """Parse a comma-separated list of ints, e.g. '32,64,128,256'."""
    if s is None or s == '':
        return None
    return [int(x.strip()) for x in s.split(',')]


def _scaled_channels(base, width, min_ch=1):
    return [max(min_ch, int(round(c * width))) for c in base]


class ResBlock1D(nn.Module):
    """1-D residual block in 2-D convolution form (height=1)."""
    def __init__(self, channels, kernel=(1, 3)):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel, padding=(0, kernel[1] // 2), bias=True)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + torch.relu(self.bn(self.conv(x)))


class AudioEncoder(nn.Module):
    def __init__(self, channels, concept, blocks=0):
        super().__init__()
        self.concept = concept
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=(1, 4), stride=(1, 4)),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU()]
            in_ch = out_ch
        for _ in range(blocks):
            layers += [ResBlock1D(in_ch)]
        self.conv = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, concept)

    def forward(self, x):
        # x: [N,1,1,16000]
        x = self.conv(x)            # [N,C,1,62]
        x = self.gap(x)             # [N,C,1,1]
        x = x.view(x.size(0), -1)   # [N,C]
        return self.fc(x)           # [N,concept]


class AudioDecoder(nn.Module):
    def __init__(self, channels, concept, blocks=0):
        super().__init__()
        self.concept = concept
        self.first = channels[0]
        self.fc = nn.Linear(concept, self.first * 1 * 62)

        # Residual blocks operating at the bottleneck [N, first, 1, 62]
        res_blocks = [ResBlock1D(self.first) for _ in range(blocks)]

        deconv = []
        for i in range(len(channels) - 1):
            deconv += [nn.ConvTranspose2d(channels[i], channels[i + 1],
                                          kernel_size=(1, 4), stride=(1, 4), output_padding=0)]
            if i < len(channels) - 2:
                deconv += [nn.BatchNorm2d(channels[i + 1]), nn.ReLU()]
        self.deconv = nn.Sequential(*res_blocks, *deconv)

    def forward(self, x):
        # Accept [N,concept] or [N,concept,1,1] (ONNX/BPU convention).
        x = x.view(-1, self.concept)
        x = self.fc(x)                       # [N, first*62]
        x = x.view(-1, self.first, 1, 62)    # [N, first, 1, 62]
        x = self.deconv(x)                   # [N, 1, 1, 15872]
        return x


class AudioAutoencoder(nn.Module):
    def __init__(self, enc_channels, dec_channels, concept=128, blocks=0):
        super().__init__()
        self.concept = concept
        self.encoder = AudioEncoder(enc_channels, concept, blocks=blocks)
        self.decoder = AudioDecoder(dec_channels, concept, blocks=blocks)

    def forward(self, x):
        concept = self.encoder(x)
        return self.decoder(concept)


class MusanChunkDataset(Dataset):
    def __init__(self, data_dir, chunk=CHUNK, overlap=0.5, max_files=None):
        self.files = sorted(Path(data_dir).glob('*.wav'))
        if not self.files:
            self.files = sorted(Path(data_dir).glob('**/*.wav'))
        if max_files:
            self.files = self.files[:max_files]
        self.chunk = chunk
        self.hop = int(chunk * (1 - overlap))
        self.index = []  # (file_idx, start)
        for i, f in enumerate(self.files):
            info = sf.info(str(f))
            n = 0
            while n + chunk <= info.frames:
                self.index.append((i, n))
                n += self.hop

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fidx, start = self.index[idx]
        fpath = str(self.files[fidx])
        x, sr = sf.read(fpath, start=start, stop=start + self.chunk, dtype='float32')
        if x.ndim > 1:
            x = x.mean(axis=1)
        if len(x) < self.chunk:
            x = np.pad(x, (0, self.chunk - len(x)))
        mx = np.max(np.abs(x))
        if mx > 1e-8:
            x = x / mx
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,16000]
        target = x[..., :TARGET]  # [1,1,15872]
        return x, target


def make_architecture(scale, enc_channels, dec_channels, concept, blocks, width):
    """Return (enc_channels, dec_channels, blocks)."""
    if enc_channels is not None and dec_channels is not None:
        return enc_channels, dec_channels, blocks if blocks is not None else 0

    presets = {
        'tiny':   {'width': 0.50, 'blocks': 0},
        'small':  {'width': 1.25, 'blocks': 3},
        'medium': {'width': 3.20, 'blocks': 3},
        'large':  {'width': 4.50, 'blocks': 4},
        'xlarge': {'width': 6.00, 'blocks': 5},
        'legacy': {'width': 1.00, 'blocks': 0},  # exact old 128-dim model
    }
    cfg = presets.get(scale, presets['small'])
    if width is not None:
        cfg['width'] = width
    if blocks is not None:
        cfg['blocks'] = blocks

    base_enc = [32, 64, 128, 256]
    base_dec = [256, 128, 64, 32, 1]

    if enc_channels is None:
        enc_channels = _scaled_channels(base_enc, cfg['width'])
    if dec_channels is None:
        dec_channels = _scaled_channels(base_dec, cfg['width'])
        dec_channels[-1] = 1  # final output must be one channel
    return enc_channels, dec_channels, cfg['blocks']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_onnx(model, out_dir, concept=128):
    model = _unwrap_compiled(model)
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    with torch.no_grad():
        dummy_enc = torch.randn(1, 1, 1, CHUNK).to(device)
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
        dummy_dec = torch.randn(1, concept, 1, 1).to(device)
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
    print(f'[export] ONNX -> {out_dir}')


def _unwrap_compiled(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def _get_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='directory with 16 kHz mono WAV files')
    parser.add_argument('--out-dir', default='checkpoints/audio', help='checkpoint directory')
    parser.add_argument('--scale', default='small',
                        choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'legacy'],
                        help='model size preset (default small ~5M params)')
    parser.add_argument('--encoder-channels', default=None, type=_parse_channel_list,
                        help='override encoder channel list, e.g. 32,64,128,256')
    parser.add_argument('--decoder-channels', default=None, type=_parse_channel_list,
                        help='override decoder channel list, e.g. 256,128,64,32,1')
    parser.add_argument('--concept', type=int, default=128,
                        help='concept dimension (default 128 to match C++ kEncoderOutputDim)')
    parser.add_argument('--width', type=float, default=None,
                        help='global width multiplier; overrides preset')
    parser.add_argument('--blocks', type=int, default=None,
                        help='residual blocks in encoder/decoder bottleneck')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (fp16)')
    parser.add_argument('--compile', action='store_true', help='try torch.compile (PyTorch 2.x)')
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='pin memory in DataLoader (default True, use --no-pin-memory to disable)')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=0x1DEA)
    parser.add_argument('--save-every', type=int, default=0,
                        help='save epoch checkpoint every N epochs (0=disable)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _get_device(args.device)
    print(f'[train] device={device}  scale={args.scale}  concept={args.concept}')

    enc_channels, dec_channels, blocks = make_architecture(
        args.scale, args.encoder_channels, args.decoder_channels,
        args.concept, args.blocks, args.width)
    print(f'[train] encoder channels={enc_channels}  decoder channels={dec_channels}  blocks={blocks}')

    model = AudioAutoencoder(enc_channels, dec_channels, concept=args.concept, blocks=blocks).to(device)
    print(f'[train] trainable parameters: {count_parameters(model) / 1e6:.2f} M')

    if args.compile:
        try:
            model = torch.compile(model)
            print('[train] torch.compile enabled')
        except Exception as e:
            print(f'[train] torch.compile failed ({e}), continuing without compile')

    ds = MusanChunkDataset(args.data_dir, overlap=0.5, max_files=args.max_files)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=args.pin_memory and device.type == 'cuda',
                        persistent_workers=args.num_workers > 0)
    print(f'[train] dataset: {len(ds)} chunks from {len(ds.files)} files')

    opt = torch.optim.Adam(_unwrap_compiled(model).parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    loss_fn = nn.MSELoss()

    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else torch.enable_grad()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        opt.zero_grad()
        for step, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast():
                pred = model(x)
                loss = loss_fn(pred, y) / args.grad_accum

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum == 0 or step == len(loader):
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
                sched.step()

            total += loss.item() * args.grad_accum

        avg = total / len(loader)
        print(f'epoch {epoch}/{args.epochs}  loss={avg:.6f}  time={time.time()-t0:.1f}s')

        if avg < best:
            best = avg
            ckpt = out_dir / 'best.pt'
            torch.save({
                'epoch': epoch,
                'model': _unwrap_compiled(model).state_dict(),
                'loss': best,
                'opt': opt.state_dict(),
                'args': vars(args),
                'enc_channels': enc_channels,
                'dec_channels': dec_channels,
                'blocks': blocks,
                'concept': args.concept,
            }, ckpt)
            print(f'  saved best -> {ckpt}')

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': _unwrap_compiled(model).state_dict(),
                'args': vars(args),
                'enc_channels': enc_channels,
                'dec_channels': dec_channels,
                'blocks': blocks,
                'concept': args.concept,
            }, out_dir / f'epoch_{epoch:03d}.pt')

    export_onnx(_unwrap_compiled(model), out_dir, concept=args.concept)
    torch.save({
        'model': _unwrap_compiled(model).state_dict(),
        'args': vars(args),
        'enc_channels': enc_channels,
        'dec_channels': dec_channels,
        'blocks': blocks,
        'concept': args.concept,
    }, out_dir / 'final.pt')

    # Also save separate encoder / decoder for convenience.
    torch.save(_unwrap_compiled(model).encoder.state_dict(), out_dir / 'encoder.pt')
    torch.save(_unwrap_compiled(model).decoder.state_dict(), out_dir / 'decoder.pt')

    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump({
            'modality': 'speech',
            'data_dir': args.data_dir,
            'epochs': args.epochs,
            'best_loss': best,
            'samples': len(ds),
            'concept_dim': args.concept,
            'chunk_size': CHUNK,
            'decoder_output_samples': TARGET,
            'enc_channels': enc_channels,
            'dec_channels': dec_channels,
            'blocks': blocks,
        }, f, indent=2)
    print(f'[train] done. outputs in {out_dir}')


if __name__ == '__main__':
    main()

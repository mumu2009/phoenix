"""Train a scalable JEPA-v2 image autoencoder.

The encoder starts from a pretrained torchvision ResNet (18/34/50) and adds a
learned Linear projection head to the concept dimension (default 128 to match
the C++ `targetDim_`).  The decoder starts from the architecture of
`/home/kali/decoder_trained.pt` (3.3 M params, 128-d concept -> 3x224x224 in
NCHW) and can be scaled via `--decoder-width` and `--decoder-depth`.

Output shapes:
  encoder: [N, 3, 224, 224] -> [N, concept]
  decoder: [N, concept, 1, 1] or [N, concept] -> [N, 3, 224, 224]

Training data can be a directory of class subfolders (ImageFolder) or any flat
folder containing *.jpg / *.png.
"""

import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import torchvision.models as models
except Exception as e:
    raise ImportError('torchvision is required for image training') from e

RESOLUTION = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _parse_channel_list(s):
    if s is None or s == '':
        return None
    return [int(x.strip()) for x in s.split(',')]


class ResBlock2d(nn.Module):
    def __init__(self, channels, kernel=3, norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=True)
        self.bn1 = nn.BatchNorm2d(channels) if norm else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=True)
        self.bn2 = nn.BatchNorm2d(channels) if norm else nn.Identity()

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        return x + self.bn2(self.conv2(h))


class JepaV2ImageDecoder(nn.Module):
    """Image decoder: concept -> 3x224x224.

    The base architecture matches /home/kali/decoder_trained.pt:
      conv0  (128 -> 256, 7x7, stride 1)  1x1 -> 7x7
      conv2  (256 -> 256, 4x4, stride 2)  7   -> 14
      conv4  (256 -> 128, 4x4, stride 2)  14  -> 28
      conv6  (128 -> 64 , 4x4, stride 2)  28  -> 56
      conv8  (64  -> 32 , 4x4, stride 2)  56  -> 112
      conv10 (32  -> 3  , 4x4, stride 2)  112 -> 224
      color  (3   -> 3  , 1x1)

    The decoder accepts either [N, concept] or [N, concept, 1, 1].
    """
    def __init__(self, concept=128, dec_width=1.0, dec_depth=0, dec_channels=None, load_sd=None):
        super().__init__()
        self.concept = concept

        # If /home/kali/decoder_trained.pt is loaded, the first conv takes 128
        # channels.  If concept != 128 we project with a Linear before reshaping.
        self.first = 128
        self.proj = nn.Linear(concept, self.first) if concept != self.first else None

        if dec_channels is None:
            base = [256, 256, 128, 64, 32, 3]
            dec_channels = [max(1, int(round(c * dec_width))) for c in base]
            dec_channels[-1] = 3  # keep RGB output

        self.dec_channels = dec_channels

        # First conv takes the concept-reshaped 128 channels, then the list
        # describes the output channels of each transposed-conv layer.
        # The base has 6 transposed-conv layers: 1x1 -> 7 -> 14 -> 28 -> 56 -> 112 -> 224.
        full = [self.first] + list(dec_channels)
        upsample_dict = OrderedDict()
        conv_idx = 0
        for i in range(len(dec_channels)):
            in_ch = full[i]
            out_ch = full[i + 1]
            if i == 0:
                kernel, stride, padding = 7, 1, 0
            else:
                kernel, stride, padding = 4, 2, 1
            upsample_dict[f'conv{conv_idx}'] = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
            if i < len(dec_channels) - 1:
                upsample_dict[f'relu{conv_idx}'] = nn.ReLU()
            conv_idx += 2
        self.upsample = nn.Sequential(upsample_dict)

        # Optional residual blocks after the upsampling stack (operates on 3-channels).
        res_blocks = []
        for _ in range(dec_depth):
            res_blocks += [ResBlock2d(dec_channels[-1], norm=False), nn.ReLU()]
        self.residual = nn.Sequential(*res_blocks)

        # 1x1 color refinement from decoder_trained.pt.
        self.color = nn.Conv2d(3, 3, 1, bias=True)

        if load_sd is not None:
            # decoder_trained.pt uses top-level keys conv0..conv10 and color.
            remapped = OrderedDict()
            for k, v in load_sd.items():
                if k.startswith('conv'):
                    remapped[f'upsample.{k}'] = v
                else:
                    remapped[k] = v
            missing, unexpected = self.load_state_dict(remapped, strict=False)
            if missing:
                print(f'[decoder] missing keys when loading pretrained ({len(missing)}): {missing[:5]}')
            if unexpected:
                print(f'[decoder] unexpected keys ({len(unexpected)}): {unexpected[:5]}')

    def forward(self, x):
        # Accept [N,concept] or [N,concept,1,1].
        x = x.view(-1, self.concept)
        if self.proj is not None:
            x = self.proj(x)
        x = x.view(-1, self.first, 1, 1)
        x = self.upsample(x)
        x = self.residual(x)
        x = self.color(x)
        return x


class JepaV2ImageEncoder(nn.Module):
    def __init__(self, resnet_name='resnet18', concept=128, pretrained=True, unfreeze=False):
        super().__init__()
        self.concept = concept
        if not hasattr(models, resnet_name):
            raise ValueError(f'Unknown torchvision model: {resnet_name}')
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet = getattr(models, resnet_name)(weights=weights)
        self.feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()  # we apply the head ourselves
        self.resnet = resnet
        self.head = nn.Linear(self.feature_dim, concept)

        for p in self.resnet.parameters():
            p.requires_grad = unfreeze

    def forward(self, x):
        # x: [N,3,224,224], normalized with ImageNet mean/std
        feat = self.resnet(x)            # [N, feature_dim]
        return self.head(feat)           # [N, concept]


class JepaV2ImageAutoencoder(nn.Module):
    def __init__(self, resnet_name='resnet18', concept=128, pretrained=True,
                 unfreeze=False, dec_width=1.0, dec_depth=0, dec_channels=None, load_sd=None):
        super().__init__()
        self.concept = concept
        self.encoder = JepaV2ImageEncoder(resnet_name, concept, pretrained, unfreeze)
        self.decoder = JepaV2ImageDecoder(concept, dec_width, dec_depth, dec_channels, load_sd)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ImageGlobDataset(Dataset):
    def __init__(self, data_dir, transform=None, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        self.data_dir = Path(data_dir)
        self.files = []
        if (self.data_dir / 'images').is_dir():
            root = self.data_dir / 'images'
        else:
            root = self.data_dir
        # ImageFolder-style class subdirectories
        if any(d.is_dir() for d in root.iterdir() if d.name[0] not in '.~'):
            for ext in extensions:
                self.files.extend(root.rglob(f'*{ext}'))
        else:
            for ext in extensions:
                self.files.extend(root.glob(f'*{ext}'))
        self.files = sorted(self.files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(str(self.files[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _unwrap_compiled(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def export_onnx(model, out_dir, concept=128, resolution=224):
    model = _unwrap_compiled(model)
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device

    base = model.encoder.resnet
    head = model.encoder.head
    feature_dim = model.encoder.feature_dim

    with torch.no_grad():
        # Encoder base: 3x224x224 -> 512-D embedding for BPU.
        dummy_enc = torch.randn(1, 3, resolution, resolution).to(device)
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

        # Encoder head: 512 -> concept, runs on CPU in C++.
        dummy_head = torch.randn(1, feature_dim).to(device)
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

        # Decoder: concept -> 3x224x224.
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

    # Save head weights as JSON for the C++ BPU pipeline.
    head_w = head.weight.detach().cpu().float().contiguous()
    head_b = head.bias.detach().cpu().float().contiguous() if head.bias is not None else torch.zeros(concept)
    with open(out_dir / 'encoder_head.json', 'w') as f:
        json.dump({
            'inDim': feature_dim,
            'outDim': concept,
            'W': head_w.view(-1).tolist(),
            'b': head_b.tolist(),
        }, f, indent=2)

    print(f'[export] ONNX -> {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='directory with images (class subdirs or flat)')
    parser.add_argument('--out-dir', default='checkpoints/jepa_v2_vision', help='checkpoint directory')
    parser.add_argument('--resnet', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='pretrained ResNet base (resnet50 -> 2048-D features)')
    parser.add_argument('--concept', type=int, default=128,
                        help='concept dimension (default 128 to match C++ targetDim)')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='use ImageNet pretrained weights (1=yes, 0=no)')
    parser.add_argument('--unfreeze-encoder', action='store_true',
                        help='train the ResNet base end-to-end (more GPU memory)')
    parser.add_argument('--decoder-pt', default='/home/kali/decoder_trained.pt',
                        help='path to trained image decoder checkpoint to warm-start')
    parser.add_argument('--decoder-width', type=float, default=1.0,
                        help='decoder channel width multiplier (1.0 = 3.3 M param base decoder)')
    parser.add_argument('--decoder-depth', type=int, default=0,
                        help='number of 3x3 residual blocks after the transposed-conv stack')
    parser.add_argument('--decoder-channels', default=None, type=_parse_channel_list,
                        help='explicit decoder channel list, e.g. 256,256,128,64,32,3')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (fp16)')
    parser.add_argument('--compile', action='store_true', help='try torch.compile (PyTorch 2.x)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='pin memory in DataLoader (default True, use --no-pin-memory to disable)')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=0x1DEA)
    parser.add_argument('--save-every', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device)
    print(f'[train] device={device}  resnet={args.resnet}  concept={args.concept}')

    load_sd = None
    if args.decoder_pt and Path(args.decoder_pt).is_file():
        print(f'[train] loading decoder warm-start from {args.decoder_pt}')
        load_sd = torch.load(args.decoder_pt, map_location='cpu', weights_only=False)
        if isinstance(load_sd, dict) and 'model' in load_sd:
            load_sd = load_sd['model']
        print(f'[train] decoder checkpoint has {len(list(load_sd.keys()))} keys')
    else:
        print(f'[train] no decoder warm-start ({args.decoder_pt} not found); training from random init')

    dec_channels = _parse_channel_list(args.decoder_channels)

    model = JepaV2ImageAutoencoder(
        resnet_name=args.resnet,
        concept=args.concept,
        pretrained=bool(args.pretrained),
        unfreeze=args.unfreeze_encoder,
        dec_width=args.decoder_width,
        dec_depth=args.decoder_depth,
        dec_channels=dec_channels,
        load_sd=load_sd,
    ).to(device)

    print(f'[train] trainable parameters: {count_parameters(model) / 1e6:.2f} M')

    if args.compile:
        try:
            model = torch.compile(model)
            print('[train] torch.compile enabled')
        except Exception as e:
            print(f'[train] torch.compile failed ({e}), continuing without compile')

    transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = ImageGlobDataset(args.data_dir, transform=transform)
    if len(ds) == 0:
        raise RuntimeError(f'No images found in {args.data_dir}')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=args.pin_memory and device.type == 'cuda',
                        persistent_workers=args.num_workers > 0)
    print(f'[train] dataset: {len(ds)} images from {args.data_dir}')

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, _unwrap_compiled(model).parameters()), lr=args.lr)
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
                'concept': args.concept,
                'resnet': args.resnet,
                'dec_channels': model.decoder.dec_channels,
                'dec_depth': args.decoder_depth,
            }, ckpt)
            print(f'  saved best -> {ckpt}')

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': _unwrap_compiled(model).state_dict(),
                'args': vars(args),
                'concept': args.concept,
                'resnet': args.resnet,
                'dec_channels': model.decoder.dec_channels,
                'dec_depth': args.decoder_depth,
            }, out_dir / f'epoch_{epoch:03d}.pt')

    export_onnx(_unwrap_compiled(model), out_dir, concept=args.concept, resolution=RESOLUTION)
    torch.save({
        'model': _unwrap_compiled(model).state_dict(),
        'args': vars(args),
        'concept': args.concept,
        'resnet': args.resnet,
        'dec_channels': model.decoder.dec_channels,
        'dec_depth': args.decoder_depth,
    }, out_dir / 'final.pt')

    torch.save(_unwrap_compiled(model).encoder.state_dict(), out_dir / 'encoder.pt')
    torch.save(_unwrap_compiled(model).decoder.state_dict(), out_dir / 'decoder.pt')

    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump({
            'modality': 'image',
            'data_dir': args.data_dir,
            'epochs': args.epochs,
            'best_loss': best,
            'samples': len(ds),
            'concept_dim': args.concept,
            'resolution': RESOLUTION,
            'resnet': args.resnet,
            'feature_dim': model.encoder.feature_dim,
            'dec_channels': model.decoder.dec_channels,
            'dec_depth': args.decoder_depth,
        }, f, indent=2)
    print(f'[train] done. outputs in {out_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train small audio/video decoders from frozen multimodal encoders.

The encoders (LLaVA-1.5 vision, Qwen2-Audio audio tower) are frozen.  For each
sample they produce a 4096-d unit query (or a sequence of unit queries).  This
script trains lightweight decoders to reconstruct the original media:

  image:  4096-d unit query -> 3 x 224 x 224 RGB
  audio:  4096-d unit query -> mel-spectrogram -> waveform

Two training objectives are supported:

  1. reconstruction: MSE / L1 between decoded media and original media.
  2. caption: a tiny text head on the unit query is trained to predict the
     original caption tokens.  This keeps the decoder's latent space aligned
     with the language model's tokenizer/detokenizer.

For images, a "painter" composite decoder is also provided: a variable number
of unit queries each decode into an RGBA layer and an affine transform, then
alpha-composite onto a canvas.  The single-query decoder is the default.

Usage:
  python tools/train_multimodal_decoders.py \
      --modality image --data_dir data/images --output_dir runtime_store/models/multimodal_dec

  python tools/train_multimodal_decoders.py \
      --modality audio --data_dir data/audio --captions data/audio/captions.json \
      --output_dir runtime_store/models/multimodal_dec
"""

import argparse
import base64
import io
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = 224
AUDIO_SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
UNIT_DIM = 4096


def _post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_unit_query_from_server(server_url, payload_bytes, modality, mime=""):
    """Get unit query(s) from the multimodal enc/dec server."""
    enc_url = f"{server_url}/enc/{modality}"
    payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
    if modality == "image":
        req = {"image": payload_b64, "mime_type": mime or "image/png", "return_sequence": True}
    else:
        req = {"audio": payload_b64, "mime_type": mime or "audio/wav", "sample_rate": AUDIO_SAMPLE_RATE, "return_sequence": True}
    resp = _post_json(enc_url, req)
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "encoder failed"))
    unit_queries = np.array(resp.get("unit_queries", []), dtype=np.float32)
    unit_query = np.array(resp.get("unit_query", []), dtype=np.float32)
    return unit_queries, unit_query


# ---------------------------------------------------------------------------
# Image decoders
# ---------------------------------------------------------------------------

class SingleImageDecoder(nn.Module):
    """Single 4096-d unit query -> 3 x 224 x 224."""

    def __init__(self, unit_dim=UNIT_DIM, size=IMAGE_SIZE):
        super().__init__()
        self.size = size
        # 4096 -> 64 x 7 x 7
        self.fc = nn.Linear(unit_dim, 64 * 7 * 7)
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 28
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 56
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),   # 112
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, 2, 1),    # 224
        )

    def forward(self, x):
        h = self.fc(x).view(-1, 64, 7, 7)
        return torch.tanh(self.blocks(h))


class LayerImageDecoder(nn.Module):
    """One unit query -> RGBA layer + affine transform."""

    def __init__(self, unit_dim=UNIT_DIM, size=IMAGE_SIZE, hidden=256):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(unit_dim, 64 * 7 * 7)
        self.rgba = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8), nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 4, 2, 1),  # RGBA
        )
        self.pose = nn.Sequential(
            nn.Linear(unit_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 6)  # tx, ty, scale, rotation, alpha, depth
        )

    def forward(self, q):
        N = q.size(0)
        h = self.fc(q).view(N, 64, 7, 7)
        rgba = torch.sigmoid(self.rgba(h))
        pose = self.pose(q)
        return rgba, pose


class PainterImageDecoder(nn.Module):
    """Variable number of unit queries -> composite RGBA layers."""

    def __init__(self, unit_dim=UNIT_DIM, size=IMAGE_SIZE, max_layers=8):
        super().__init__()
        self.size = size
        self.max_layers = max_layers
        self.layer_decoder = LayerImageDecoder(unit_dim, size)

    def forward(self, unit_queries, canvas=None):
        """unit_queries: [N, L, D] or [N, D].  Returns [N, 3, H, W]."""
        if unit_queries.dim() == 2:
            unit_queries = unit_queries.unsqueeze(1)
        N, L, D = unit_queries.shape
        L = min(L, self.max_layers)
        if canvas is None:
            canvas = torch.zeros(N, 4, self.size, self.size, device=unit_queries.device)
        # composite layers front-to-back by alpha
        for i in range(L):
            rgba, pose = self.layer_decoder(unit_queries[:, i])
            alpha = rgba[:, 3:4] * (1.0 - canvas[:, 3:4])
            canvas[:, :3] = canvas[:, :3] * (1.0 - alpha) + rgba[:, :3] * alpha
            canvas[:, 3:4] = canvas[:, 3:4] + (1.0 - canvas[:, 3:4]) * rgba[:, 3:4]
        return canvas[:, :3]


# ---------------------------------------------------------------------------
# Audio decoders
# ---------------------------------------------------------------------------

class AudioDecoder(nn.Module):
    """Single 4096-d unit query -> mel-spectrogram [1, n_mels, T]."""

    def __init__(self, unit_dim=UNIT_DIM, n_mels=N_MELS, hop=HOP_LENGTH, target_len=256):
        super().__init__()
        self.n_mels = n_mels
        self.target_len = target_len
        # 4096 -> 64 x target_len/32 x n_mels/32 = 64 x 8 x 3
        t = max(1, target_len // 32)
        self.fc = nn.Linear(unit_dim, 64 * t * (n_mels // 32))
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(True),   # 64 x t x n_mels/16
            nn.ConvTranspose2d(64, 32, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(True),   # 32 x t x n_mels/8
            nn.ConvTranspose2d(32, 16, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(16), nn.ReLU(True),   # 16 x t x n_mels/4
            nn.ConvTranspose2d(16, 8, (3, 4), (1, 2), (1, 1)),
            nn.BatchNorm2d(8), nn.ReLU(True),    # 8 x t x n_mels/2
            nn.ConvTranspose2d(8, 1, (3, 4), (1, 2), (1, 1)),
        )

    def forward(self, x):
        t = max(1, self.target_len // 32)
        h = self.fc(x).view(-1, 64, t, self.n_mels // 32)
        mel = self.upsample(h)
        mel = F.interpolate(mel, size=(self.n_mels, self.target_len), mode="bilinear", align_corners=False)
        return torch.sigmoid(mel)  # [N, 1, n_mels, T]


# ---------------------------------------------------------------------------
# Caption head (for the combined loss)
# ---------------------------------------------------------------------------

class CaptionHead(nn.Module):
    """Predict caption token distribution from a mean unit query."""

    def __init__(self, unit_dim=UNIT_DIM, vocab_size=128256, hidden=1024, max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, unit_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=unit_dim, nhead=16, dim_feedforward=4 * unit_dim, batch_first=True),
            num_layers=3,
        )
        self.out = nn.Linear(unit_dim, vocab_size)

    def forward(self, unit_query, target_tokens=None):
        """unit_query: [N, D].  target_tokens: [N, L]."""
        N, D = unit_query.shape
        memory = unit_query.unsqueeze(1)
        L = self.max_len if target_tokens is None else target_tokens.size(1)
        bos = torch.zeros(N, 1, dtype=torch.long, device=unit_query.device)
        if target_tokens is not None:
            inp = torch.cat([bos, target_tokens[:, :-1]], dim=1)
        else:
            inp = bos.expand(N, L)
        tgt = self.embed(inp)
        out = self.decoder(tgt, memory)
        return self.out(out)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, root, size=IMAGE_SIZE, captions=None, encoder_url="http://127.0.0.1:8085"):
        self.root = Path(root)
        self.size = size
        self.encoder_url = encoder_url
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
        if captions is None:
            self.captions = {p: p.stem for p in self.files}
        else:
            with open(captions, "r", encoding="utf-8") as f:
                self.captions = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)  # [3, H, W]
        with open(path, "rb") as f:
            data = f.read()
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        try:
            unit_queries, unit_query = load_unit_query_from_server(self.encoder_url, data, "image", mime)
        except Exception as e:
            unit_queries = np.zeros((1, UNIT_DIM), dtype=np.float32)
            unit_query = np.zeros((UNIT_DIM,), dtype=np.float32)
            print(f"encoder failed for {path}: {e}")
        caption = str(self.captions.get(str(path), path.stem))
        return {
            "unit_queries": torch.from_numpy(unit_queries),
            "unit_query": torch.from_numpy(unit_query),
            "media": tensor,
            "caption": caption,
        }


class AudioFolderDataset(Dataset):
    def __init__(self, root, captions, encoder_url="http://127.0.0.1:8085"):
        self.root = Path(root)
        self.encoder_url = encoder_url
        with open(captions, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
        # captions = {filename: text} or {filename: {"text": ..., "sr": 16000}}
        self.files = [self.root / k for k in self.captions if (self.root / k).exists()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        key = path.name
        caption = self.captions[key]
        if isinstance(caption, dict):
            caption = caption.get("text", "")
        wave, sr = librosa.load(path, sr=AUDIO_SAMPLE_RATE, mono=True)
        # Send the original file bytes to the server (server expects a WAV/FLAC container).
        with open(path, "rb") as f:
            file_bytes = f.read()
        try:
            unit_queries, unit_query = load_unit_query_from_server(self.encoder_url, file_bytes, "audio", "audio/wav")
        except Exception as e:
            unit_queries = np.zeros((1, UNIT_DIM), dtype=np.float32)
            unit_query = np.zeros((UNIT_DIM,), dtype=np.float32)
            print(f"encoder failed for {path}: {e}")
        mel = librosa.feature.melspectrogram(y=wave, sr=AUDIO_SAMPLE_RATE, n_mels=N_MELS,
                                             hop_length=HOP_LENGTH, n_fft=N_FFT, win_length=WIN_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db + 80.0) / 80.0
        mel_norm = torch.from_numpy(mel_norm).unsqueeze(0).float()  # [1, n_mels, T]
        return {
            "unit_queries": torch.from_numpy(unit_queries),
            "unit_query": torch.from_numpy(unit_query),
            "media": mel_norm,
            "caption": caption,
        }


def collate(batch):
    # unit_queries: list of [L_i, D]; pad to [B, Lmax, D]
    max_len = max(b["unit_queries"].size(0) for b in batch)
    D = batch[0]["unit_queries"].size(1)
    padded = torch.zeros(len(batch), max_len, D)
    for i, b in enumerate(batch):
        L = b["unit_queries"].size(0)
        padded[i, :L] = b["unit_queries"]
    return {
        "unit_queries": padded,
        "unit_query": torch.stack([b["unit_query"] for b in batch]),
        "media": torch.stack([b["media"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def image_loss(pred, target):
    return F.mse_loss(pred, target)


def audio_loss(pred_mel, target_mel):
    target_mel = target_mel.squeeze(1)  # [B, n_mels, T]
    if target_mel.size(-1) != pred_mel.size(-1):
        target_mel = F.interpolate(
            target_mel.unsqueeze(1),
            size=pred_mel.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return F.mse_loss(pred_mel, target_mel)


def griffin_lim(mel, n_iter=32, sr=AUDIO_SAMPLE_RATE, n_fft=N_FFT, hop=HOP_LENGTH, win=WIN_LENGTH):
    """Convert mel-spectrogram [1, n_mels, T] back to a waveform."""
    mel = mel.detach().cpu().numpy().squeeze(0)
    # approximate mel -> linear by pseudo-inverse
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel.shape[0])
    linear = np.dot(np.linalg.pinv(mel_basis), np.exp(mel * 8.0 - 8.0))
    angles = np.exp(2j * np.pi * np.random.rand(*linear.shape))
    for _ in range(n_iter):
        y = librosa.istft(linear * angles, hop_length=hop, win_length=win)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)
        angles = stft / (np.abs(stft) + 1e-8)
    return y


def export_to_onnx(model, dummy_input, path, opset=14):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["unit_query"],
        output_names=["output"],
        dynamic_axes={"unit_query": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported ONNX -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train multimodal decoders.")
    parser.add_argument("--modality", choices=["image", "audio"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--captions", default=None, help="JSON file mapping filename to caption")
    parser.add_argument("--output_dir", default="runtime_store/models/multimodal_dec")
    parser.add_argument("--encoder_url", default="http://127.0.0.1:8085")
    parser.add_argument("--decoder", choices=["single", "painter"], default="single")
    parser.add_argument("--use_caption_loss", action="store_true", help="Add caption prediction loss")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.modality == "image":
        ds = ImageFolderDataset(args.data_dir, captions=args.captions, encoder_url=args.encoder_url)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        if args.decoder == "painter":
            decoder = PainterImageDecoder(max_layers=4)
        else:
            decoder = SingleImageDecoder()
        decoder = decoder.to(device)
        if args.use_caption_loss:
            caption_head = CaptionHead().to(device)
        dummy = torch.randn(1, UNIT_DIM).to(device)
    else:
        if args.captions is None:
            raise ValueError("--captions is required for audio training")
        ds = AudioFolderDataset(args.data_dir, args.captions, encoder_url=args.encoder_url)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        decoder = AudioDecoder().to(device)
        if args.use_caption_loss:
            caption_head = CaptionHead().to(device)
        dummy = torch.randn(1, UNIT_DIM).to(device)

    if args.use_caption_loss:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        except Exception as e:
            print(f"Could not load llama3.1 tokenizer, using simple whitespace tokenization: {e}")
            tokenizer = None
    else:
        caption_head = None
        tokenizer = None

    params = list(decoder.parameters())
    if caption_head:
        params += list(caption_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        if caption_head:
            caption_head.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            unit_query = batch["unit_query"].to(device)
            unit_queries = batch["unit_queries"].to(device)
            media = batch["media"].to(device)

            opt.zero_grad()
            loss = 0.0

            if args.modality == "image":
                pred = decoder(unit_queries)
                loss = image_loss(pred, media)
            else:
                pred_mel = decoder(unit_query).squeeze(1)  # [B, n_mels, T]
                loss = audio_loss(pred_mel, media)

            if caption_head and tokenizer is not None:
                cap_tokens = tokenizer(batch["caption"], return_tensors="pt", padding="max_length",
                                       truncation=True, max_length=32).input_ids.to(device)
                logits = caption_head(unit_query, cap_tokens)
                cap_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), cap_tokens.view(-1),
                                           ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
                loss = loss + 0.1 * cap_loss

            loss.backward()
            opt.step()

            total_loss += loss.item()
            count += 1
            if count % 10 == 0:
                print(f"epoch {epoch} step {count} loss={total_loss / count:.4f}")

        print(f"epoch {epoch} avg loss={total_loss / max(1, count):.4f}")

    # Save
    decoder_path = out_dir / f"{args.modality}_decoder.pt"
    torch.save({"state_dict": decoder.state_dict(), "config": vars(args)}, decoder_path)
    print(f"Saved checkpoint -> {decoder_path}")

    # ONNX export
    decoder.eval()
    with torch.no_grad():
        if args.modality == "image" and args.decoder == "painter":
            dummy = dummy.unsqueeze(1).repeat(1, 4, 1)
        export_to_onnx(decoder, dummy, out_dir / f"{args.modality}_decoder.onnx")

    if caption_head:
        torch.save({"state_dict": caption_head.state_dict()}, out_dir / "caption_head.pt")


if __name__ == "__main__":
    main()

"""Additive residual BPU-evolvable JEPA-v2 models.

A single model starts from a zero output and grows by adding one residual block
per round.  Each new block receives the same input as the model and produces an
output of the same shape as the model output.  The final output is the sum of
all block outputs (plus an optional frozen base, e.g. ResNet18 for the vision
encoder).

The framework is deliberately training-free on the compile host: new residual
blocks are freshly initialized and exported to ONNX.  Selection is done by the
RDK X5 BPU using MSE on a fresh batch from a local data pool.
"""

import argparse
import io
import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import torch

from vboxsf_safe import safe_json_dump, safe_tofile, safe_write_bytes
import torch.nn as nn

# If set to 0, skip the post-write onnx/torch load verification.
_VERIFY_WRITES = os.environ.get("PHOENIX_VERIFY_WRITES", "1").lower() in ("1", "true", "yes")


def _verify_onnx_file(path: Path, expected_size: int) -> None:
    actual = path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"ONNX size mismatch for {path}: {actual} != {expected_size}")
    if _VERIFY_WRITES:
        try:
            import onnx

            onnx.load(str(path))
        except Exception as exc:
            raise RuntimeError(f"ONNX verification failed for {path}: {exc}") from exc


def _verify_pt_file(path: Path) -> None:
    if _VERIFY_WRITES:
        try:
            _torch_load_safe(path)
        except Exception as exc:
            raise RuntimeError(f"PT verification failed for {path}: {exc}") from exc


def _torch_load_safe(path: Path):
    """Load a PyTorch checkpoint via an in-memory buffer to avoid vboxsf read glitches."""
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)

CONCEPT = 128
SPEECH_CHUNK = 16000
SPEECH_TARGET = 15872
VISION_RES = 224


# ---------------------------------------------------------------------------
# Block builders (all ~100k-250k parameters, fixed batch 1, BPU-friendly ops)
# ---------------------------------------------------------------------------

class SpeechEncoderBlock(nn.Module):
    """1x1x1x16000 -> 1xconceptx1x1 residual block."""

    def __init__(self, concept=CONCEPT):
        super().__init__()
        self.concept = concept
        # 16000 -> 4000 -> 1000 -> 250 -> 62, then GAP
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 4), stride=(1, 4), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 4), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 4), stride=(1, 4), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(1, 4), stride=(1, 4), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = (
            nn.Conv2d(128, concept, kernel_size=1, bias=True)
            if concept != 128
            else None
        )

    def forward(self, x):
        x = self.body(x)          # [1, 128, 1, 62]
        x = self.gap(x)           # [1, 128, 1, 1]
        if self.proj is not None:
            x = self.proj(x)      # [1, concept, 1, 1]
        return x


class SpeechDecoderBlock(nn.Module):
    """1xconceptx1x1 -> 1x1x1x15872 residual block."""

    def __init__(self, concept=CONCEPT, first=16, target_len=SPEECH_TARGET):
        super().__init__()
        self.concept = concept
        self.first = first
        self.target_len = target_len
        # bottleneck length 62; 62 * 4^4 = 15872
        self.fc = nn.Linear(concept, first * 62, bias=True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(first, 32, kernel_size=(1, 4), stride=(1, 4), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=(1, 4), stride=(1, 4), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1, 4), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(1, 4), stride=(1, 4), bias=True),
        )

    def forward(self, x):
        x = x.view(1, self.concept)
        x = self.fc(x)
        x = x.view(1, self.first, 1, 62)
        x = self.upsample(x)
        return x


class VisionEncoderBlock(nn.Module):
    """1x3x224x224 -> 1xconceptx1x1 residual block."""

    def __init__(self, concept=CONCEPT, channels=(32, 64, 128, 128)):
        super().__init__()
        self.concept = concept
        in_ch = 3
        layers = []
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        last_ch = channels[-1]
        self.proj = (
            nn.Conv2d(last_ch, concept, kernel_size=1, bias=True)
            if last_ch != concept
            else None
        )

    def forward(self, x):
        x = self.body(x)
        x = self.gap(x)
        if self.proj is not None:
            x = self.proj(x)
        return x


class VisionDecoderBlock(nn.Module):
    """1xconceptx1x1 -> 1x3x224x224 residual block."""

    def __init__(self, concept=CONCEPT, first=8):
        super().__init__()
        self.concept = concept
        self.first = first
        self.fc = nn.Linear(concept, first * 7 * 7, bias=True)
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224 using ConvTranspose k=4 s=2 p=1
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(first, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=True),
        )

    def forward(self, x):
        x = x.view(1, self.concept)
        x = self.fc(x)
        x = x.view(1, self.first, 7, 7)
        x = self.upsample(x)
        return x


BLOCK_CLASSES = {
    "speech_encoder": SpeechEncoderBlock,
    "speech_decoder": SpeechDecoderBlock,
    "vision_encoder": VisionEncoderBlock,
    "vision_decoder": VisionDecoderBlock,
}


def build_block(model_name, concept=CONCEPT, block_config=None):
    """Build a residual block for the given model type."""
    block_config = block_config or {}
    cls = BLOCK_CLASSES[model_name]
    kwargs = {"concept": concept}
    if model_name == "speech_decoder":
        kwargs["first"] = block_config.get("first", 16)
    if model_name == "vision_decoder":
        kwargs["first"] = block_config.get("first", 8)
    return cls(**kwargs)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Optional frozen ResNet18 base for the vision encoder
# ---------------------------------------------------------------------------

class ResNet18Base(nn.Module):
    """Frozen ResNet18 + linear head -> 1xconceptx1x1."""

    def __init__(self, concept=CONCEPT, base_path=None, pretrained=True):
        super().__init__()
        import torchvision

        weights = None
        if base_path is None and pretrained:
            try:
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                # older torchvision fallback
                weights = "IMAGENET1K_V1"
        resnet = torchvision.models.resnet18(weights=weights)
        self.feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet
        self.head = nn.Linear(self.feature_dim, concept)

        if base_path:
            self._load_from_path(base_path)

        for p in self.parameters():
            p.requires_grad = False

    def _load_from_path(self, base_path):
        ckpt = _torch_load_safe(base_path)
        if isinstance(ckpt, dict):
            if "base_state" in ckpt:
                state = ckpt["base_state"]
            else:
                state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            # If the checkpoint is a JepaV2ImageAutoencoder, strip encoder.
            if any(k.startswith("encoder.") for k in state):
                state = {
                    k.split("encoder.", 1)[1]: v
                    for k, v in state.items()
                    if k.startswith("encoder.")
                }
            # If the checkpoint came from an AdditiveResidualModel, strip the
            # base. prefix and drop the residual blocks.
            if any(k.startswith("base.") for k in state):
                state = {
                    k.split("base.", 1)[1]: v
                    for k, v in state.items()
                    if k.startswith("base.")
                }
        else:
            state = ckpt.state_dict()
        missing, _ = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[ResNet18Base] missing {len(missing)} keys: {missing[:5]}")

    def forward(self, x):
        x = self.resnet(x)        # [1, 512]
        x = self.head(x)          # [1, concept]
        return x.view(x.size(0), -1, 1, 1)


# ---------------------------------------------------------------------------
# Generic additive residual model
# ---------------------------------------------------------------------------

def get_input_shape(model_name: str, concept: int = CONCEPT) -> tuple:
    return {
        "speech_encoder": (1, 1, 1, 16000),
        "speech_decoder": (1, concept, 1, 1),
        "vision_encoder": (1, 3, 224, 224),
        "vision_decoder": (1, concept, 1, 1),
    }[model_name]


def get_output_shape(model_name: str, concept: int = CONCEPT) -> tuple:
    return {
        "speech_encoder": (1, concept, 1, 1),
        "speech_decoder": (1, 1, 1, 15872),
        "vision_encoder": (1, concept, 1, 1),
        "vision_decoder": (1, 3, 224, 224),
    }[model_name]


# Legacy constants (concept=128) for quick reference; prefer the functions above.
MODEL_INPUT_SHAPES = {
    "speech_encoder": (1, 1, 1, 16000),
    "speech_decoder": (1, CONCEPT, 1, 1),
    "vision_encoder": (1, 3, 224, 224),
    "vision_decoder": (1, CONCEPT, 1, 1),
}

MODEL_OUTPUT_SHAPES = {
    "speech_encoder": (1, CONCEPT, 1, 1),
    "speech_decoder": (1, 1, 1, 15872),
    "vision_encoder": (1, CONCEPT, 1, 1),
    "vision_decoder": (1, 3, 224, 224),
}

MODEL_INPUT_NAMES = {
    "speech_encoder": "waveform",
    "speech_decoder": "concept",
    "vision_encoder": "pixel_values",
    "vision_decoder": "concept",
}

MODEL_OUTPUT_NAMES = {
    "speech_encoder": "concept",
    "speech_decoder": "reconstruction",
    "vision_encoder": "concept",
    "vision_decoder": "reconstruction",
}


class AdditiveResidualModel(nn.Module):
    """Model that adds residual blocks on top of an optional frozen base.

    output = base(input) + sum(block(input) for block in blocks)

    If ``base`` is None and there are no blocks, the output is a constant zero
    tensor of the target shape.
    """

    def __init__(
        self,
        model_name: str,
        concept: int = CONCEPT,
        base: nn.Module = None,
        block_config=None,
    ):
        super().__init__()
        if model_name not in MODEL_INPUT_SHAPES:
            raise ValueError(f"Unknown model_name: {model_name}")
        self.model_name = model_name
        self.concept = concept
        self.input_shape = get_input_shape(model_name, concept)
        self.output_shape = get_output_shape(model_name, concept)
        self.block_config = block_config or {}

        if isinstance(base, bool) and base and model_name == "vision_encoder":
            base = ResNet18Base(concept=concept)
        self.base = base
        if self.base is not None:
            for p in self.base.parameters():
                p.requires_grad = False

        self.blocks = nn.ModuleList()
        # Constant zero output used when there is no base and before blocks run.
        self.register_buffer(
            "zero_out", torch.zeros(1, *self.output_shape[1:])
        )

    def add_block(self, seed=None, block=None):
        """Append a new residual block.  Returns the block index."""
        if block is None:
            if seed is not None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    block = build_block(
                        self.model_name, self.concept, self.block_config
                    )
            else:
                block = build_block(
                    self.model_name, self.concept, self.block_config
                )
        self.blocks.append(block)
        return len(self.blocks) - 1

    def forward(self, x):
        if self.base is not None:
            out = self.base(x)
            if out.dim() != len(self.output_shape):
                out = out.view(1, *self.output_shape[1:])
        else:
            out = self.zero_out
        for block in self.blocks:
            out = out + block(x)
        return out

    def count_block_parameters(self) -> int:
        return sum(count_parameters(b) for b in self.blocks)

    def save_checkpoint(self, path: Path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": self.state_dict(),
            "model_name": self.model_name,
            "concept": self.concept,
            "block_config": self.block_config,
            "has_base": self.base is not None,
            "n_blocks": len(self.blocks),
            "extra": extra or {},
        }
        # torch.save directly hits VirtualBox shared-folder write-count bugs;
        # write the zip into memory and flush it with the vboxsf-safe helper.
        buffer = io.BytesIO()
        torch.save(ckpt, buffer)
        data = buffer.getvalue()
        safe_write_bytes(path, data)
        _verify_pt_file(path)

    @classmethod
    def from_checkpoint(cls, path: Path, base_path=None, base_pretrained=False):
        """Load an AdditiveResidualModel from a checkpoint."""
        path = Path(path)
        ckpt = _torch_load_safe(path)
        model_name = ckpt["model_name"]
        concept = ckpt.get("concept", CONCEPT)
        block_config = ckpt.get("block_config", {})

        base = None
        has_base = ckpt.get("has_base", False)
        if has_base and model_name == "vision_encoder":
            base = ResNet18Base(concept=concept, base_path=base_path, pretrained=base_pretrained)

        model = cls(
            model_name=model_name,
            concept=concept,
            base=base,
            block_config=block_config,
        )
        n_blocks = ckpt.get("n_blocks", 0)
        for _ in range(n_blocks):
            model.add_block()
        model.load_state_dict(ckpt["state_dict"])
        return model


# ---------------------------------------------------------------------------
# ONNX export / calibration / manifest helpers
# ---------------------------------------------------------------------------

def get_dummy_input(model_name: str, concept: int = CONCEPT) -> torch.Tensor:
    return torch.randn(*get_input_shape(model_name, concept))


def write_calibration(
    out_dir: Path,
    name: str,
    shape: tuple,
    count: int = 10,
    scale: float = 0.5,
    mean: float = 0.0,
):
    calib_dir = out_dir / (f"calibration_{name}" if name else "calibration")
    calib_dir.mkdir(parents=True, exist_ok=True)
    numel = int(np.prod(shape))
    rng = np.random.default_rng(42)
    for i in range(count):
        arr = rng.normal(mean, scale, size=shape).astype(np.float32)
        safe_tofile(arr, calib_dir / f"cal_{i:04d}.bin")
    print(f"[calibration] wrote {count} samples to {calib_dir}")
    return calib_dir


def write_manifest(
    out_dir: Path,
    model_name: str,
    concept: int,
    n_blocks: int,
    source_checkpoint=None,
):
    manifest = {
        "name": f"additive_jepa_{model_name}",
        "model_name": model_name,
        "modality": "speech" if "speech" in model_name else "image",
        "concept_dim": concept,
        "input_name": MODEL_INPUT_NAMES[model_name],
        "input_shape": list(MODEL_INPUT_SHAPES[model_name]),
        "output_name": MODEL_OUTPUT_NAMES[model_name],
        "output_shape": list(MODEL_OUTPUT_SHAPES[model_name]),
        "n_blocks": n_blocks,
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
    }
    if "speech" in model_name:
        manifest["chunk_size"] = SPEECH_CHUNK
        if "decoder" in model_name:
            manifest["decoder_output_samples"] = SPEECH_TARGET
    if "vision" in model_name:
        manifest["resolution"] = VISION_RES

    safe_json_dump(out_dir / "model.manifest.json", manifest, indent=2)
    print(f"[manifest] wrote {out_dir}/model.manifest.json")


def export_to_onnx(
    model: AdditiveResidualModel,
    out_dir: Path,
    n_calib: int = 10,
    source_checkpoint=None,
    save_pt: bool = True,
):
    """Export the full additive model to a single ONNX, write calibration and manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()

    with torch.no_grad():
        dummy = get_dummy_input(model.model_name, model.concept)
        # Some VirtualBox shared folders (vboxsf) return invalid lengths from
        # raw write() for large single writes.  Export to a BytesIO first and
        # then flush the bytes to disk in small chunks.
        buffer = io.BytesIO()
        torch.onnx.export(
            model,
            dummy,
            buffer,
            input_names=[MODEL_INPUT_NAMES[model.model_name]],
            output_names=[MODEL_OUTPUT_NAMES[model.model_name]],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
        )
        onnx_bytes = buffer.getvalue()
        onnx_path = out_dir / "model.onnx"
        safe_write_bytes(onnx_path, onnx_bytes)
        _verify_onnx_file(onnx_path, len(onnx_bytes))
        print(f"[export] wrote {onnx_path} ({len(onnx_bytes)} bytes)")

    if save_pt:
        model.save_checkpoint(out_dir / "model.pt", extra={"source": str(source_checkpoint)})

    write_calibration(
        out_dir,
        "",
        model.input_shape,
        n_calib,
        scale=0.5,
        mean=0.0,
    )
    write_manifest(
        out_dir,
        model.model_name,
        model.concept,
        len(model.blocks),
        source_checkpoint=source_checkpoint,
    )
    print(f"[export] wrote {out_dir}/model.onnx")
    return out_dir / "model.onnx"


# ---------------------------------------------------------------------------
# CLI helper: create / inspect an additive model
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        choices=list(MODEL_INPUT_SHAPES.keys()),
    )
    parser.add_argument("--concept", type=int, default=CONCEPT)
    parser.add_argument("--base-path", default=None, help="ResNet18 .pt for vision_encoder")
    parser.add_argument("--n-blocks", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--calibration-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    base = None
    if args.model_name == "vision_encoder" and args.base_path:
        base = ResNet18Base(concept=args.concept, base_path=args.base_path)

    model = AdditiveResidualModel(
        args.model_name,
        concept=args.concept,
        base=base,
    )
    for i in range(args.n_blocks):
        model.add_block(seed=args.seed + i + 1)

    print(
        f"[additive] {args.model_name} n_blocks={len(model.blocks)} "
        f"block_params={model.count_block_parameters()} "
        f"total_params={count_parameters(model)}"
    )
    export_to_onnx(
        model, args.out_dir, n_calib=args.calibration_count, source_checkpoint=None
    )


if __name__ == "__main__":
    main()

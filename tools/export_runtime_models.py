"""Deploy trained audio/video autoencoders to the runtime layout used by C++ and BPU.

The C++ Phoenix runtime expects:
    runtime_store/models/additive_jepa/<variant>/best.onnx
    runtime_store/models/additive_jepa/<variant>/model.manifest.json

BPU compilation additionally uses:
    model_encoder.onnx / model_decoder.onnx
    model_encoder_head.onnx (image) + encoder_head.json

This module exports the *full* encoder/decoder as best.onnx for C++ local
ONNX, while keeping the split BPU artifacts in the same variant directories.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEPLOY_ROOT = ROOT / "runtime_store" / "models" / "additive_jepa"


def _write_manifest(path: Path, fields: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)


def _export_full_module(
    module: nn.Module,
    out_path: Path,
    dummy: torch.Tensor,
    input_name: str,
    output_name: str,
    dynamic_axes: Dict[str, Any] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval().cpu()
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy.cpu(),
            out_path,
            input_names=[input_name],
            output_names=[output_name],
            opset_version=11,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes=dynamic_axes or {input_name: {0: "batch"}, output_name: {0: "batch"}},
        )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        import shutil
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deploy_video_onnx(
    model: nn.Module,
    out_dir: Path,
    deploy_root: Path = DEFAULT_DEPLOY_ROOT,
    concept: int = 128,
    resolution: int = 224,
    feature_dim: int | None = None,
) -> None:
    """Export a trained VideoAutoencoder to runtime-ready best.onnx files.

    BPU artifacts (model_encoder.onnx, model_encoder_head.onnx,
    model_decoder.onnx, encoder_head.json) are expected to already exist in
    `out_dir`; they are copied to the matching deploy directories.  The full
    fused encoder and decoder are exported as best.onnx for C++ local ONNX.
    """
    model = model.cpu().eval() if hasattr(model, "eval") else model
    out_dir = Path(out_dir)
    deploy_root = Path(deploy_root)

    enc_dir = deploy_root / "vision_encoder"
    dec_dir = deploy_root / "vision_decoder"

    # Full encoder ONNX (resnet + head) for local C++ ONNX.
    _export_full_module(
        model.encoder,
        enc_dir / "best.onnx",
        torch.randn(1, 3, resolution, resolution),
        input_name="pixel_values",
        output_name="concept",
    )
    _write_manifest(
        enc_dir / "model.manifest.json",
        {
            "name": "video-encoder",
            "model_name": "vision_encoder",
            "modality": "image",
            "concept_dim": concept,
            "feature_dim": feature_dim or model.encoder.feature_dim,
            "resolution": resolution,
            "resnet": getattr(model.encoder, "resnet_name", "resnet18"),
            "input_name": "pixel_values",
            "input_shape": [1, 3, resolution, resolution],
            "output_name": "concept",
            "output_shape": [1, concept, 1, 1],
        },
    )

    # Full decoder ONNX for local C++ ONNX.
    _export_full_module(
        model.decoder,
        dec_dir / "best.onnx",
        torch.randn(1, concept, 1, 1),
        input_name="concept",
        output_name="reconstruction",
    )
    _write_manifest(
        dec_dir / "model.manifest.json",
        {
            "name": "video-decoder",
            "model_name": "vision_decoder",
            "modality": "image",
            "concept_dim": concept,
            "resolution": resolution,
            "input_name": "concept",
            "input_shape": [1, concept, 1, 1],
            "output_name": "reconstruction",
            "output_shape": [1, 3, resolution, resolution],
        },
    )

    # Copy BPU split artifacts if the caller placed them in out_dir.
    for name in ["model_encoder.onnx", "model_encoder_head.onnx", "encoder_head.json"]:
        _copy_if_exists(out_dir / name, enc_dir / name)
    _copy_if_exists(out_dir / "model_decoder.onnx", dec_dir / "model_decoder.onnx")

    print(f"[deploy] video runtime ONNX -> {deploy_root}/vision_{{encoder,decoder}}")


def deploy_audio_onnx(
    model: nn.Module,
    out_dir: Path,
    deploy_root: Path = DEFAULT_DEPLOY_ROOT,
    concept: int = 128,
    chunk_size: int = 16000,
    decoder_output_samples: int = 15872,
) -> None:
    """Export a trained AudioAutoencoder to runtime-ready best.onnx files."""
    model = model.cpu().eval() if hasattr(model, "eval") else model
    out_dir = Path(out_dir)
    deploy_root = Path(deploy_root)

    enc_dir = deploy_root / "speech_encoder"
    dec_dir = deploy_root / "speech_decoder"

    # Full audio encoder.
    _export_full_module(
        model.encoder,
        enc_dir / "best.onnx",
        torch.randn(1, 1, 1, chunk_size),
        input_name="waveform",
        output_name="concept",
    )
    _write_manifest(
        enc_dir / "model.manifest.json",
        {
            "name": "audio-16k-encoder",
            "model_name": "speech_encoder",
            "modality": "speech",
            "concept_dim": concept,
            "chunk_size": chunk_size,
            "decoder_output_samples": decoder_output_samples,
            "input_name": "waveform",
            "input_shape": [1, 1, 1, chunk_size],
            "output_name": "concept",
            "output_shape": [1, concept, 1, 1],
        },
    )

    # Full audio decoder.
    _export_full_module(
        model.decoder,
        dec_dir / "best.onnx",
        torch.randn(1, concept, 1, 1),
        input_name="concept",
        output_name="reconstruction",
    )
    _write_manifest(
        dec_dir / "model.manifest.json",
        {
            "name": "audio-16k-decoder",
            "model_name": "speech_decoder",
            "modality": "speech",
            "concept_dim": concept,
            "chunk_size": chunk_size,
            "decoder_output_samples": decoder_output_samples,
            "input_name": "concept",
            "input_shape": [1, concept, 1, 1],
            "output_name": "reconstruction",
            "output_shape": [1, 1, 1, decoder_output_samples],
        },
    )

    # Copy BPU split artifacts if the caller placed them in out_dir.
    _copy_if_exists(out_dir / "model_encoder.onnx", enc_dir / "model_encoder.onnx")
    _copy_if_exists(out_dir / "model_decoder.onnx", dec_dir / "model_decoder.onnx")

    print(f"[deploy] audio runtime ONNX -> {deploy_root}/speech_{{encoder,decoder}}")

#!/usr/bin/env python3
"""Export/prepare multimodal encoders for the Phoenix enc/dec server.

This script loads the vision (LLaVA-1.5) and audio (Qwen2-Audio) encoders and
exports the vision/audio tower + projector submodules to ONNX. The resulting
ONNX files can be used by `multimodal_encdec_server.py` without loading the
full Hugging Face multimodal LLMs.

If ONNX export fails for a model, the script falls back to saving the full
Hugging Face checkpoint to `runtime_store/models/multimodal_enc/` so the server
can still load it via `--image-model-dir` or `--audio-model-dir`.

Outputs:
  runtime_store/models/multimodal_enc/llava_vision_projector.onnx
  runtime_store/models/multimodal_enc/llava_image_preprocessor/
  runtime_store/models/multimodal_enc/qwen2_audio_connector.onnx
  runtime_store/models/multimodal_enc/qwen2_audio_feature_extractor/

Usage:
    python tools/prepare_multimodal_encoders.py \
        --out-dir runtime_store/models/multimodal_enc \
        --device cpu \
        --dtype float32
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "runtime_store" / "models" / "multimodal_enc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    """Pick a torch dtype, forcing float32 on CPU for half dtypes."""
    dtype_str = dtype_str.lower()
    if dtype_str in ("float16", "fp16") and device == "cpu":
        print("[warn] float16 is not well supported on CPU; using float32")
        dtype_str = "float32"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float32)


def get_nested_attr(obj: Any, *names: str) -> Optional[Any]:
    """Try a sequence of attribute names and return the first one that exists."""
    for name in names:
        if obj is None:
            return None
        obj = getattr(obj, name, None)
    return obj


def pick_attr_from_model(model: Any, *candidates: Tuple[str, ...]) -> Any:
    """Pick a nested attribute, trying both top-level and `model.model.*`."""
    for candidate in candidates:
        obj = get_nested_attr(model, *candidate)
        if obj is not None:
            return obj
    raise RuntimeError(f"Could not find any of {candidates} in model")


# ---------------------------------------------------------------------------
# Image encoder export
# ---------------------------------------------------------------------------

class LlavaImageEncoder(nn.Module):
    """Wrapper that runs just the CLIP vision tower + multimodal projector."""

    def __init__(self, vision_tower: nn.Module, projector: nn.Module, config: Any):
        super().__init__()
        self.vision_tower = vision_tower
        self.multi_modal_projector = projector
        self.vision_feature_layer = getattr(config, "vision_feature_layer", -2)
        self.vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", "default")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = image_outputs.hidden_states

        if isinstance(self.vision_feature_layer, int):
            selected = hidden_states[self.vision_feature_layer]
            if self.vision_feature_select_strategy == "default":
                selected = selected[:, 1:]
        else:
            if self.vision_feature_select_strategy == "default":
                selected = torch.cat([hidden_states[i][:, 1:] for i in self.vision_feature_layer], dim=-1)
            else:
                selected = torch.cat([hidden_states[i] for i in self.vision_feature_layer], dim=-1)

        return self.multi_modal_projector(selected)


def export_image_encoder(args: argparse.Namespace) -> Optional[Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.image_model_dir or args.image_model
    local_files_only = bool(args.image_model_dir)
    device = args.device
    dtype = resolve_dtype(args.dtype, device)

    print(f"[image] loading {model_path} (local_only={local_files_only}) ...")
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            )
        except Exception as e:
            if "accelerate" in str(e).lower():
                print(f"[image] accelerate not installed; loading without device_map")
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=local_files_only,
                )
                model = model.to(device)
            else:
                raise
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    except Exception as e:
        print(f"[image] failed to load model: {e}")
        traceback.print_exc()
        return None

    # Save the image preprocessor so the ONNX-only server can preprocess.
    preprocessor = getattr(processor, "image_processor", None)
    if preprocessor is None:
        preprocessor = getattr(processor, "feature_extractor", None)
    if preprocessor is not None:
        preproc_dir = out_dir / "llava_image_preprocessor"
        preproc_dir.mkdir(parents=True, exist_ok=True)
        try:
            preprocessor.save_pretrained(preproc_dir)
            print(f"[image] saved preprocessor to {preproc_dir}")
        except Exception as e:
            print(f"[image] warning: could not save preprocessor: {e}")

    # Wrap and export just the vision part.
    vision_tower = pick_attr_from_model(
        model, ("vision_tower",), ("model", "vision_tower")
    )
    projector = pick_attr_from_model(
        model, ("multi_modal_projector",), ("model", "multi_modal_projector")
    )

    enc = LlavaImageEncoder(vision_tower, projector, model.config)
    enc.eval().to(device=device, dtype=dtype)

    onnx_path = out_dir / "llava_vision_projector.onnx"
    dummy = torch.randn(1, 3, 336, 336, dtype=dtype, device=device)

    print(f"[image] exporting vision projector to {onnx_path} ...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                enc,
                dummy,
                onnx_path,
                input_names=["pixel_values"],
                output_names=["image_features"],
                opset_version=14,
                do_constant_folding=True,
                dynamic_axes={
                    "pixel_values": {0: "batch", 2: "height", 3: "width"},
                    "image_features": {0: "batch", 1: "num_queries"},
                },
            )
        print(f"[image] exported ONNX -> {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"[image] ONNX export failed: {e}")
        traceback.print_exc()

    # Fallback: save the full HF checkpoint.
    fallback_dir = out_dir / "llava-1.5-7b-hf_full"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    print(f"[image] falling back to full HF save -> {fallback_dir}")
    try:
        model.save_pretrained(fallback_dir, max_shard_size="2GB", safe_serialization=False)
        processor.save_pretrained(fallback_dir)
        print(f"[image] saved full model and processor to {fallback_dir}")
        print(f"[image] run the server with --image-model-dir {fallback_dir}")
        return fallback_dir
    except Exception as e2:
        print(f"[image] fallback save failed: {e2}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Audio encoder export
# ---------------------------------------------------------------------------

class QwenAudioEncoder(nn.Module):
    """Wrapper that runs just the Qwen2-Audio audio tower + projector."""

    def __init__(self, audio_tower: nn.Module, projector: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower
        self.multi_modal_projector = projector

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        audio_outputs = self.audio_tower(input_features)
        last_hidden_state = (
            audio_outputs.last_hidden_state
            if hasattr(audio_outputs, "last_hidden_state")
            else audio_outputs[0]
        )
        return self.multi_modal_projector(last_hidden_state)


def qwen_expected_mel_length(model: Any) -> int:
    """Compute the exact mel-frame length Qwen2AudioEncoder expects.

    Qwen2AudioEncoder enforces:
        input_features.shape[-1] == max_source_positions * conv1.stride * conv2.stride
    """
    audio_cfg = getattr(model.config, "audio_config", None) or model.config
    max_source_positions = getattr(audio_cfg, "max_source_positions", 1500)
    # conv1 stride is (1,), conv2 stride is (2,).
    return max_source_positions * 1 * 2


def export_audio_encoder(args: argparse.Namespace) -> Optional[Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.audio_model_dir or args.audio_model
    local_files_only = bool(args.audio_model_dir)
    device = args.device
    dtype = resolve_dtype(args.dtype, device)

    print(f"[audio] loading {model_path} (local_only={local_files_only}) ...")
    try:
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

        try:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
        except Exception as e:
            if "accelerate" in str(e).lower():
                print(f"[audio] accelerate not installed; loading without device_map")
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    local_files_only=local_files_only,
                    trust_remote_code=True,
                )
                model = model.to(device)
            else:
                raise
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[audio] failed to load model: {e}")
        traceback.print_exc()
        return None

    # Save the Whisper feature extractor so the ONNX-only server can preprocess.
    feature_extractor = getattr(processor, "feature_extractor", None)
    if feature_extractor is not None:
        fe_dir = out_dir / "qwen2_audio_feature_extractor"
        fe_dir.mkdir(parents=True, exist_ok=True)
        try:
            feature_extractor.save_pretrained(fe_dir)
            print(f"[audio] saved feature extractor to {fe_dir}")
        except Exception as e:
            print(f"[audio] warning: could not save feature extractor: {e}")

    audio_tower = pick_attr_from_model(
        model, ("audio_tower",), ("model", "audio_tower")
    )
    projector = pick_attr_from_model(
        model, ("multi_modal_projector",), ("model", "multi_modal_projector")
    )

    enc = QwenAudioEncoder(audio_tower, projector)
    enc.eval().to(device=device, dtype=dtype)

    mel_length = qwen_expected_mel_length(model)
    num_mel_bins = getattr(
        getattr(model.config, "audio_config", model.config), "num_mel_bins", 128
    )
    onnx_path = out_dir / "qwen2_audio_connector.onnx"
    dummy = torch.randn(1, num_mel_bins, mel_length, dtype=dtype, device=device)

    print(f"[audio] exporting audio connector to {onnx_path} (dummy {dummy.shape}) ...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                enc,
                dummy,
                onnx_path,
                input_names=["input_features"],
                output_names=["audio_features"],
                opset_version=14,
                do_constant_folding=True,
                dynamic_axes={
                    "input_features": {0: "batch"},
                    "audio_features": {0: "batch"},
                },
            )
        print(f"[audio] exported ONNX -> {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"[audio] ONNX export failed: {e}")
        traceback.print_exc()

    # Fallback: save the full HF checkpoint.
    fallback_dir = out_dir / "Qwen2-Audio-7B-Instruct_full"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    print(f"[audio] falling back to full HF save -> {fallback_dir}")
    try:
        model.save_pretrained(fallback_dir, max_shard_size="2GB", safe_serialization=False)
        processor.save_pretrained(fallback_dir)
        print(f"[audio] saved full model and processor to {fallback_dir}")
        print(f"[audio] run the server with --audio-model-dir {fallback_dir}")
        return fallback_dir
    except Exception as e2:
        print(f"[audio] fallback save failed: {e2}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare multimodal encoders for Phoenix")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--image-model",
        default="llava-hf/llava-1.5-7b-hf",
        help="Hugging Face image encoder id",
    )
    parser.add_argument(
        "--audio-model",
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Hugging Face audio encoder id",
    )
    parser.add_argument("--image-model-dir", default="", help="Local Hugging Face image model dir")
    parser.add_argument("--audio-model-dir", default="", help="Local Hugging Face audio model dir")
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="PyTorch dtype (default float32)",
    )
    parser.add_argument("--skip-image", action="store_true", help="Skip image encoder export")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio encoder export")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    print(f"[prepare] output directory: {args.out_dir}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    results = []
    if not args.skip_image:
        results.append(("image", export_image_encoder(args)))
    else:
        print("[image] skipped")

    if not args.skip_audio:
        results.append(("audio", export_audio_encoder(args)))
    else:
        print("[audio] skipped")

    print("\n[done] prepared artifacts:")
    for name, path in results:
        if path:
            print(f"  {name}: {path}")
        else:
            print(f"  {name}: failed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
import gc
import json
import os
import sys
import traceback
import warnings

# Avoid Windows cp65001/gbk encoding issues when torch.onnx prints emoji.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress noisy non-fatal warnings from torch.onnx / transformers / copyreg.
warnings.filterwarnings("ignore", message=".*dynamic_axes.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*fast processor.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*isinstance\\(treespec, LeafSpec\\).*", category=FutureWarning)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from multimodal_model_loader import resolve_model_path

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


def _split_aliases(text: str) -> List[str]:
    """Split a comma-separated list of model ids into a clean list."""
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


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


def _find_index_file(model_path: Path) -> Optional[Path]:
    """Find a sharded checkpoint index, preferring safetensors."""
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = model_path / name
        if p.is_file():
            return p
    return None


def _load_checkpoint_file(checkpoint_file: Path) -> Dict[str, torch.Tensor]:
    """Load a safetensors or PyTorch checkpoint file and return its state dict."""
    if checkpoint_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            return load_file(str(checkpoint_file), device="cpu")
        except Exception as e:
            print(f"[warn] safetensors load failed for {checkpoint_file}: {e}; trying torch.load")
    return torch.load(checkpoint_file, map_location="cpu", weights_only=True)


def _load_partial_state_dict(model_path: Path, prefixes: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load only the weights for the given prefixes from a (sharded) checkpoint.

    This is the key memory-saver: we don't need to load the whole 7B/7B+ model
    just to extract the tiny vision/audio tower and projector.
    """
    # Strip trailing dots so callers can use plain names like "vision_tower".
    result: Dict[str, Dict[str, torch.Tensor]] = {p.rstrip("."): {} for p in prefixes}
    index_file = _find_index_file(model_path)
    files_to_load: List[Path] = []

    if index_file is not None:
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        needed_files = set()
        for key, filename in weight_map.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    needed_files.add(model_path / filename)
                    break
        files_to_load = sorted(needed_files)
        if not files_to_load:
            raise FileNotFoundError(f"no weights matching {prefixes} found in {index_file}")
    else:
        candidates = sorted(model_path.glob("*.safetensors")) + sorted(
            model_path.glob("pytorch_model.bin")
        )
        if not candidates:
            raise FileNotFoundError(f"no checkpoint file found in {model_path}")
        files_to_load = candidates[:1]

    for checkpoint_file in files_to_load:
        state = _load_checkpoint_file(checkpoint_file)
        for key, tensor in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    if new_key.startswith("."):
                        new_key = new_key[1:]
                    result[prefix.rstrip(".")][new_key] = tensor
                    break
        del state
        gc.collect()

    return result


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


def _load_llava_image_tower(resolved_path: Path, dtype: torch.dtype, device: str):
    """Load only the CLIP vision tower + multimodal projector from the full LLaVA checkpoint."""
    from transformers import AutoProcessor, CLIPVisionModel, LlavaConfig
    from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

    config = LlavaConfig.from_pretrained(resolved_path, local_files_only=True)
    state = _load_partial_state_dict(resolved_path, ["vision_tower.", "multi_modal_projector."])

    print(f"[image] loading vision tower and projector from {resolved_path} ...")
    vision_tower = CLIPVisionModel(config.vision_config)
    vision_tower.load_state_dict(state["vision_tower"], strict=True)
    projector = LlavaMultiModalProjector(config)
    projector.load_state_dict(state["multi_modal_projector"], strict=True)

    enc = LlavaImageEncoder(vision_tower, projector, config)
    enc.eval().to(device=device, dtype=dtype)

    # Save the image preprocessor so the ONNX-only server can preprocess.
    processor = AutoProcessor.from_pretrained(resolved_path, local_files_only=True)
    return enc, processor


def export_image_encoder(args: argparse.Namespace) -> Optional[Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.image_model_dir or args.image_model
    allow_download = not args.no_hf_download and args.allow_hf_download
    device = args.device
    dtype = resolve_dtype(args.dtype, device)

    print(f"[image] resolving {model_path} (allow_download={allow_download}) ...")
    try:
        resolved_path = resolve_model_path(
            model_path,
            allow_download=allow_download,
            hf_endpoint=args.hf_endpoint,
            use_modelscope=args.use_modelscope,
            modelscope_aliases=_split_aliases(args.image_modelscope_id),
            modelscope_endpoint=args.modelscope_endpoint,
        )
        print(f"[image] model resolved to {resolved_path}")
    except Exception as e:
        print(f"[image] failed to resolve model: {e}")
        traceback.print_exc()
        return None

    try:
        enc, processor = _load_llava_image_tower(Path(resolved_path), dtype, device)
    except Exception as e:
        print(f"[image] failed to load vision tower: {e}")
        traceback.print_exc()
        return None

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
                opset_version=18,
                do_constant_folding=True,
                dynamic_axes={
                    "pixel_values": {0: "batch"},
                    "image_features": {0: "batch", 1: "num_queries"},
                },
                verbose=False,
            )
        print(f"[image] exported ONNX -> {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"[image] ONNX export failed: {e}")
        traceback.print_exc()

    # Fallback: save the tiny extracted vision tower so the server can use it directly.
    fallback_dir = out_dir / "llava-1.5-7b-hf_full"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    print(f"[image] falling back to HF vision tower save -> {fallback_dir}")
    try:
        enc.vision_tower.save_pretrained(fallback_dir / "vision_tower")
        enc.multi_modal_projector.state_dict()  # no save_pretrained for projector
        torch.save(enc.multi_modal_projector.state_dict(), fallback_dir / "multi_modal_projector.pt")
        preprocessor.save_pretrained(fallback_dir)
        print(f"[image] saved vision tower and projector to {fallback_dir}")
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


def qwen_expected_mel_length(config: Any) -> int:
    """Compute the exact mel-frame length Qwen2AudioEncoder expects.

    Qwen2AudioEncoder enforces:
        input_features.shape[-1] == max_source_positions * conv1.stride * conv2.stride
    """
    audio_cfg = getattr(config, "audio_config", None) or config
    max_source_positions = getattr(audio_cfg, "max_source_positions", 1500)
    # conv1 stride is (1,), conv2 stride is (2,).
    return max_source_positions * 1 * 2


def _load_qwen_audio_tower(resolved_path: Path, dtype: torch.dtype, device: str):
    """Load only the Qwen2-Audio encoder + projector from the full checkpoint."""
    from transformers import AutoProcessor, Qwen2AudioConfig
    from transformers.models.qwen2_audio.modeling_qwen2_audio import (
        Qwen2AudioEncoder,
        Qwen2AudioMultiModalProjector,
    )

    config = Qwen2AudioConfig.from_pretrained(resolved_path, local_files_only=True, trust_remote_code=True)
    state = _load_partial_state_dict(resolved_path, ["audio_tower.", "multi_modal_projector."])

    print(f"[audio] loading audio tower and projector from {resolved_path} ...")
    audio_tower = Qwen2AudioEncoder(config.audio_config)
    audio_tower.load_state_dict(state["audio_tower"], strict=True)
    projector = Qwen2AudioMultiModalProjector(config)
    projector.load_state_dict(state["multi_modal_projector"], strict=True)

    enc = QwenAudioEncoder(audio_tower, projector)
    enc.eval().to(device=device, dtype=dtype)

    processor = AutoProcessor.from_pretrained(
        resolved_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    return enc, processor, config


def export_audio_encoder(args: argparse.Namespace) -> Optional[Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.audio_model_dir or args.audio_model
    allow_download = not args.no_hf_download and args.allow_hf_download
    device = args.device
    dtype = resolve_dtype(args.dtype, device)

    print(f"[audio] resolving {model_path} (allow_download={allow_download}) ...")
    try:
        resolved_path = resolve_model_path(
            model_path,
            allow_download=allow_download,
            hf_endpoint=args.hf_endpoint,
            use_modelscope=args.use_modelscope,
            modelscope_aliases=_split_aliases(args.audio_modelscope_id),
            modelscope_endpoint=args.modelscope_endpoint,
        )
        print(f"[audio] model resolved to {resolved_path}")
    except Exception as e:
        print(f"[audio] failed to resolve model: {e}")
        traceback.print_exc()
        return None

    try:
        enc, processor, config = _load_qwen_audio_tower(Path(resolved_path), dtype, device)
    except Exception as e:
        print(f"[audio] failed to load audio tower: {e}")
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

    mel_length = qwen_expected_mel_length(config)
    num_mel_bins = getattr(config.audio_config, "num_mel_bins", 128)
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
                opset_version=18,
                do_constant_folding=True,
                dynamic_axes={
                    "input_features": {0: "batch"},
                    "audio_features": {0: "batch"},
                },
                verbose=False,
            )
        print(f"[audio] exported ONNX -> {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"[audio] ONNX export failed: {e}")
        traceback.print_exc()

    # Fallback: save the tiny extracted audio tower so the server can use it directly.
    fallback_dir = out_dir / "Qwen2-Audio-7B-Instruct_full"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    print(f"[audio] falling back to HF audio tower save -> {fallback_dir}")
    try:
        enc.audio_tower.save_pretrained(fallback_dir / "audio_tower")
        torch.save(enc.multi_modal_projector.state_dict(), fallback_dir / "multi_modal_projector.pt")
        processor.save_pretrained(fallback_dir)
        print(f"[audio] saved audio tower and projector to {fallback_dir}")
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

    # Network / mirror options for environments with limited HF access (e.g. mainland China).
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        default=True,
        help="Allow downloading from Hugging Face or a configured mirror (default: True)",
    )
    parser.add_argument(
        "--no-hf-download",
        action="store_true",
        help="Never download; fail immediately if the model is not cached or provided via --*-model-dir",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", ""),
        help="Hugging Face mirror endpoint, e.g. https://hf-mirror.com (env: HF_ENDPOINT)",
    )
    parser.add_argument(
        "--modelscope-endpoint",
        default=os.environ.get("MODELSCOPE_ENDPOINT", ""),
        help="ModelScope mirror endpoint (env: MODELSCOPE_ENDPOINT)",
    )
    parser.add_argument(
        "--use-modelscope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try ModelScope as a fallback if Hugging Face is unavailable (default: True)",
    )
    parser.add_argument(
        "--image-modelscope-id",
        default="",
        help="Optional comma-separated ModelScope ids for the image model",
    )
    parser.add_argument(
        "--audio-modelscope-id",
        default="",
        help="Optional comma-separated ModelScope ids for the audio model",
    )
    parser.add_argument(
        "--cn-mirror",
        action="store_true",
        help="Shortcut: set --hf-endpoint=https://hf-mirror.com and keep ModelScope fallback",
    )

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

    if args.cn_mirror and not args.hf_endpoint:
        args.hf_endpoint = "https://hf-mirror.com"
        print(f"[prepare] using China mirror: {args.hf_endpoint}")

    if args.no_hf_download:
        args.allow_hf_download = False

    if not args.image_model_dir and not args.audio_model_dir and not args.allow_hf_download:
        print(
            "[warn] No --*-model-dir provided and --no-hf-download is set. "
            "The script will not be able to load any models."
        )

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

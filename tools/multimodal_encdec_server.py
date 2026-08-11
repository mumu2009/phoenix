#!/usr/bin/env python3
"""Multimodal encoder/decoder HTTP service for the Phoenix project.

This is a self-contained HTTP server (stdlib only for the web layer) that
exposes endpoints for encoding/decoding images and audio using large
multimodal encoders (LLaVA-1.5 and Qwen2-Audio) and optional ONNX or PyTorch
decoders.

Endpoints:
  GET/POST /status       -> health and model load status
  POST     /enc/image    -> image -> 4096-D unit queries
  POST     /enc/audio    -> audio -> 4096-D unit queries
  POST     /dec/image    -> unit queries -> base64 image
  POST     /dec/audio    -> unit queries -> base64 audio

Usage:
    python tools/multimodal_encdec_server.py \
        --host 127.0.0.1 --port 8085 \
        --image-decoder runtime_store/models/multimodal_dec/image_decoder.onnx \
        --audio-decoder runtime_store/models/multimodal_dec/audio_decoder.onnx
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMAGE_ENCODER_ONNX = (
    PROJECT_ROOT / "runtime_store" / "models" / "multimodal_enc" / "llava_vision_projector.onnx"
)
DEFAULT_AUDIO_ENCODER_ONNX = (
    PROJECT_ROOT / "runtime_store" / "models" / "multimodal_enc" / "qwen2_audio_connector.onnx"
)

DEFAULT_IMAGE_PREPROCESSOR_DIR = (
    PROJECT_ROOT / "runtime_store" / "models" / "multimodal_enc" / "llava_image_preprocessor"
)
DEFAULT_AUDIO_FEATURE_EXTRACTOR_DIR = (
    PROJECT_ROOT / "runtime_store" / "models" / "multimodal_enc" / "qwen2_audio_feature_extractor"
)

DEFAULT_IMAGE_DECODER = (
    PROJECT_ROOT / "runtime_store" / "models" / "multimodal_dec" / "image_decoder.onnx"
)
DEFAULT_AUDIO_DECODER = (
    PROJECT_ROOT / "runtime_store" / "models" / "multimodal_dec" / "audio_decoder.onnx"
)

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27477011], dtype=np.float32)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base64 / bytes helpers
# ---------------------------------------------------------------------------

def b64_decode(text: str) -> bytes:
    """Decode a base64 string, returning raw bytes."""
    return base64.b64decode(text)


def b64_encode(data: bytes) -> str:
    """Encode raw bytes as a base64 ASCII string."""
    return base64.b64encode(data).decode("ascii")


def safe_json(data: Any, ensure_ascii: bool = False) -> bytes:
    """Return UTF-8 JSON bytes, converting tensors/ndarrays on the fly."""

    def tensor_default(o: Any) -> Any:
        if hasattr(o, "detach"):
            o = o.detach().cpu().numpy()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(data, default=tensor_default, ensure_ascii=ensure_ascii).encode("utf-8")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def read_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Read a WAV file from bytes and return (float32 mono samples, sample_rate)."""
    with io.BytesIO(data) as buf:
        with wave.open(buf, "rb") as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            raw = w.readframes(nframes)

    if sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 3:
        # 24-bit packed; not common.
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        samples = (
            (a[:, 2].astype(np.int32) << 16)
            | (a[:, 1].astype(np.int32) << 8)
            | a[:, 0].astype(np.int32)
        )
        # sign-extend
        samples = np.where(samples >= 1 << 23, samples - (1 << 24), samples).astype(np.int32)
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if nchannels > 1:
        samples = samples.reshape(-1, nchannels).mean(axis=1).astype(samples.dtype)

    return samples.astype(np.float32) / (np.iinfo(samples.dtype).max + 1), framerate


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample a 1-D float audio array to target_sr using librosa or scipy."""
    if orig_sr == target_sr:
        return samples

    try:
        import librosa

        return librosa.resample(samples.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr).astype(
            np.float32
        )
    except Exception:
        pass

    try:
        import scipy.signal

        num_samples = int(round(len(samples) * target_sr / orig_sr))
        return scipy.signal.resample(samples.astype(np.float64), num_samples).astype(np.float32)
    except Exception:
        pass

    raise RuntimeError(
        f"Audio sample rate {orig_sr} != target {target_sr} and neither librosa nor scipy is available"
    )


def write_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Write a float32 1-D audio array to a WAV file in memory and return bytes."""
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    int16_samples = (samples * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16_samples.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def pil_to_b64(img, mime_type: str = "image/png") -> str:
    """Convert a PIL image to base64 encoded bytes."""
    fmt = mime_type.split("/")[-1].upper()
    if fmt not in ("PNG", "JPEG", "JPG", "GIF", "BMP", "WEBP"):
        fmt = "PNG"
    if fmt == "JPG":
        fmt = "JPEG"
    buf = io.BytesIO()
    if fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return b64_encode(buf.getvalue())


def decode_image(data: bytes, width: Optional[int], height: Optional[int]) -> Any:
    """Decode image bytes with PIL and optionally resize."""
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"PIL is required for image decoding: {e}")

    img = Image.open(io.BytesIO(data)).convert("RGB")
    if width is not None and height is not None:
        img = img.resize((width, height), Image.BILINEAR)
    return img


def preprocess_image_for_onnx(
    img: Any, target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Preprocess a PIL image for the exported CLIP/LLaVA vision encoder."""
    from PIL import Image

    if target_size is None:
        target_size = (336, 336)
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...].astype(np.float32)


# ---------------------------------------------------------------------------
# Lazy model wrapper
# ---------------------------------------------------------------------------

class LazyModel:
    """Thread-safe lazy loader for encoder/decoder modules."""

    def __init__(self, name: str, loader):
        self.name = name
        self._loader = loader
        self._model: Optional[Any] = None
        self._error: Optional[str] = None
        self._loaded = False
        self._lock = threading.Lock()

    def get(self) -> Tuple[Optional[Any], Optional[str]]:
        if not self._loaded and self._error is None and self._model is None:
            with self._lock:
                if not self._loaded and self._error is None and self._model is None:
                    try:
                        self._model = self._loader()
                        self._loaded = True
                        self._error = None
                        log.info("[%s] loaded successfully", self.name)
                    except Exception as e:
                        self._error = f"{type(e).__name__}: {e}"
                        log.error("[%s] failed to load: %s", self.name, self._error)
        return self._model, self._error

    def status(self) -> Dict[str, Any]:
        model = None
        if self._loaded and self._model is not None:
            if isinstance(self._model, dict):
                model = self._model.get("name", str(self._model.get("path", "")))
            elif hasattr(self._model, "name"):
                model = self._model.name
            elif hasattr(self._model, "path"):
                model = str(self._model.path)
            else:
                model = self._model.__class__.__name__
        return {
            "loaded": self._loaded,
            "model": model if model else self.name,
            "error": self._error,
        }


# ---------------------------------------------------------------------------
# Server state and model loaders
# ---------------------------------------------------------------------------

class ServerState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._torch_dtype: Any = None

        self.image_encoder = LazyModel("image_encoder", self._load_image_encoder)
        self.audio_encoder = LazyModel("audio_encoder", self._load_audio_encoder)
        self.image_decoder = LazyModel("image_decoder", self._load_image_decoder)
        self.audio_decoder = LazyModel("audio_decoder", self._load_audio_decoder)

    # --- dtype / device helpers ------------------------------------------

    def _resolve_dtype(self) -> Any:
        import torch

        if self._torch_dtype is not None:
            return self._torch_dtype

        device = self.args.device.lower()
        dtype_str = self.args.dtype.lower()

        if dtype_str == "float16" and device == "cpu":
            log.warning("float16 on CPU is not well supported; using float32")
            dtype_str = "float32"

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        self._torch_dtype = dtype_map.get(dtype_str, torch.float32)
        return self._torch_dtype

    # --- model path helpers ------------------------------------------------

    def _image_model_path(self) -> str:
        if self.args.image_model_dir:
            return self.args.image_model_dir
        return self.args.image_model

    def _audio_model_path(self) -> str:
        if self.args.audio_model_dir:
            return self.args.audio_model_dir
        return self.args.audio_model

    def _is_hf_dir_set(self, attr: str) -> bool:
        return bool(getattr(self.args, attr))

    # --- image encoder loader ----------------------------------------------

    def _load_image_encoder(self) -> Any:
        # If the user provided an explicit --image-model-dir, prefer the full
        # Hugging Face model. Otherwise, prefer the prepared ONNX if it exists.
        if not self._is_hf_dir_set("image_model_dir"):
            onnx_path = self.args.image_encoder_onnx or DEFAULT_IMAGE_ENCODER_ONNX
            if Path(onnx_path).is_file():
                return self._load_image_encoder_onnx(onnx_path)

        return self._load_image_encoder_full()

    def _load_image_encoder_onnx(self, onnx_path: Path) -> Dict[str, Any]:
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(f"onnxruntime is required for the image encoder ONNX: {e}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)

        preprocessor = None
        if not self._is_hf_dir_set("image_model_dir"):
            if DEFAULT_IMAGE_PREPROCESSOR_DIR.is_dir():
                try:
                    from transformers import AutoImageProcessor

                    preprocessor = AutoImageProcessor.from_pretrained(str(DEFAULT_IMAGE_PREPROCESSOR_DIR))
                except Exception:
                    pass

        return {
            "type": "onnx",
            "name": "llava-1.5-7b-onnx",
            "path": str(onnx_path),
            "session": sess,
            "preprocessor": preprocessor,
        }

    def _load_image_encoder_full(self) -> Dict[str, Any]:
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            import torch
        except Exception as e:
            raise RuntimeError(f"transformers/torch are required for the full image encoder: {e}")

        model_path = self._image_model_path()
        dtype = self._resolve_dtype()

        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=self.args.device,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        except Exception as e:
            msg = str(e)
            if "accelerate" in msg.lower():
                log.warning("accelerate not installed; loading image encoder without device_map")
                try:
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        local_files_only=True,
                    )
                    model = model.to(self.args.device)
                except Exception as e2:
                    raise RuntimeError(f"Could not load image encoder from {model_path}: {e2}")
            else:
                raise RuntimeError(f"Could not load image encoder from {model_path}: {e}")

        try:
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"Could not load image processor from {model_path}: {e}")

        return {
            "type": "hf",
            "name": "llava-1.5-7b",
            "model": model,
            "processor": processor,
        }

    # --- audio encoder loader ----------------------------------------------

    def _load_audio_encoder(self) -> Any:
        if not self._is_hf_dir_set("audio_model_dir"):
            onnx_path = self.args.audio_encoder_onnx or DEFAULT_AUDIO_ENCODER_ONNX
            if Path(onnx_path).is_file():
                feature_extractor_dir = DEFAULT_AUDIO_FEATURE_EXTRACTOR_DIR
                if feature_extractor_dir.is_dir():
                    return self._load_audio_encoder_onnx(onnx_path)

        return self._load_audio_encoder_full()

    def _load_audio_encoder_onnx(self, onnx_path: Path) -> Dict[str, Any]:
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(f"onnxruntime is required for the audio encoder ONNX: {e}")

        try:
            from transformers import AutoFeatureExtractor
        except Exception as e:
            raise RuntimeError(f"transformers is required to load the audio feature extractor: {e}")

        feature_extractor = None
        if DEFAULT_AUDIO_FEATURE_EXTRACTOR_DIR.is_dir():
            try:
                feature_extractor = AutoFeatureExtractor.from_pretrained(
                    str(DEFAULT_AUDIO_FEATURE_EXTRACTOR_DIR)
                )
            except Exception as e:
                log.warning("Could not load audio feature extractor: %s", e)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)

        return {
            "type": "onnx",
            "name": "qwen2-audio-7b-onnx",
            "path": str(onnx_path),
            "session": sess,
            "feature_extractor": feature_extractor,
        }

    def _load_audio_encoder_full(self) -> Dict[str, Any]:
        try:
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            import torch
        except Exception as e:
            raise RuntimeError(f"transformers/torch are required for the full audio encoder: {e}")

        model_path = self._audio_model_path()
        dtype = self._resolve_dtype()

        try:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=self.args.device,
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            msg = str(e)
            if "accelerate" in msg.lower():
                log.warning("accelerate not installed; loading audio encoder without device_map")
                try:
                    model = Qwen2AudioForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=dtype,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                    model = model.to(self.args.device)
                except Exception as e2:
                    raise RuntimeError(f"Could not load audio encoder from {model_path}: {e2}")
            else:
                raise RuntimeError(f"Could not load audio encoder from {model_path}: {e}")

        try:
            processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(f"Could not load audio processor from {model_path}: {e}")

        return {
            "type": "hf",
            "name": "qwen2-audio-7b",
            "model": model,
            "processor": processor,
        }

    # --- decoder loaders ---------------------------------------------------

    def _load_image_decoder(self) -> Any:
        path = Path(self.args.image_decoder)
        if not path.is_file():
            raise FileNotFoundError(f"image decoder not found at {path}")
        return self._load_decoder(path)

    def _load_audio_decoder(self) -> Any:
        path = Path(self.args.audio_decoder)
        if not path.is_file():
            raise FileNotFoundError(f"audio decoder not found at {path}")
        return self._load_decoder(path)

    def _load_decoder(self, path: Path) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            try:
                import onnxruntime as ort
            except Exception as e:
                raise RuntimeError(f"onnxruntime is required for decoder {path}: {e}")

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(str(path), providers=providers)
            return {"type": "onnx", "path": str(path), "session": sess}

        if suffix in (".pt", ".pth", ".ckpt"):
            try:
                import torch
            except Exception as e:
                raise RuntimeError(f"torch is required for decoder {path}: {e}")

            # Try TorchScript first, then a plain torch.load checkpoint.
            try:
                model = torch.jit.load(str(path), map_location="cpu")
            except Exception:
                ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    model = ckpt.get("model", ckpt.get("state_dict", ckpt))
                else:
                    model = ckpt
            if not callable(model):
                raise RuntimeError(f"Loaded decoder from {path} is not callable")
            return {"type": "pt", "path": str(path), "model": model}

        raise ValueError(f"Unsupported decoder format: {path}")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _get_image_features_from_hf(
    state: ServerState, model_info: Dict[str, Any], img: Any
) -> np.ndarray:
    """Run the full LLaVA image encoder and return (1, N, 4096) features."""
    import torch

    model = model_info["model"]
    processor = model_info["processor"]
    dtype = state._resolve_dtype()
    device = state.args.device

    # Pre-process. Prefer the processor's image processor.
    image_processor = getattr(processor, "image_processor", None) or getattr(
        processor, "feature_extractor", None
    )

    if image_processor is not None:
        # If the image has already been resized by the caller, skip further resize/crop.
        try:
            w, h = img.size
            do_resize = w != 336 or h != 336
        except Exception:
            do_resize = True

        if do_resize:
            inputs = image_processor(images=[img], do_rescale=True, do_normalize=True, return_tensors="pt")
        else:
            inputs = image_processor(
                images=[img],
                do_resize=False,
                do_center_crop=False,
                do_rescale=True,
                do_normalize=True,
                return_tensors="pt",
            )
        pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    else:
        arr = preprocess_image_for_onnx(img)
        pixel_values = torch.from_numpy(arr).to(device=device, dtype=dtype)

    # Try the high-level get_image_features; fall back to manual vision_tower + projector.
    image_features = None
    get_image_features = getattr(model, "get_image_features", None) or getattr(
        getattr(model, "model", None), "get_image_features", None
    )

    if get_image_features is not None:
        try:
            with torch.no_grad():
                out = get_image_features(pixel_values=pixel_values)
            image_features = _extract_image_features_output(out)
        except Exception as e:
            log.debug("get_image_features failed, falling back: %s", e)
            image_features = None

    if image_features is None:
        image_features = _image_forward_manual(model, pixel_values)

    return image_features.detach().cpu().float().numpy()


def _extract_image_features_output(out: Any) -> Any:
    """Normalize the return value of get_image_features across versions."""
    import torch

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        return out[0]
    if hasattr(out, "pooler_output"):
        po = out.pooler_output
        if isinstance(po, (list, tuple)):
            return po[0]
        return po
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out


def _image_forward_manual(model: Any, pixel_values: Any) -> Any:
    """Manual fallback: vision_tower -> projector with layer selection."""
    import torch

    vision_tower = getattr(model, "vision_tower", None) or getattr(
        getattr(model, "model", None), "vision_tower", None
    )
    projector = getattr(model, "multi_modal_projector", None) or getattr(
        getattr(model, "model", None), "multi_modal_projector", None
    )

    if vision_tower is None or projector is None:
        raise RuntimeError("Could not locate vision_tower or multi_modal_projector")

    config = model.config
    vision_feature_layer = getattr(config, "vision_feature_layer", -2)
    vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", "default")

    with torch.no_grad():
        vision_outputs = vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = vision_outputs.hidden_states

        if isinstance(vision_feature_layer, int):
            selected = hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected = selected[:, 1:]
        else:
            if vision_feature_select_strategy == "default":
                selected = torch.cat([hidden_states[i][:, 1:] for i in vision_feature_layer], dim=-1)
            else:
                selected = torch.cat([hidden_states[i] for i in vision_feature_layer], dim=-1)

        image_features = projector(selected)
    return image_features


def _preprocess_image_with_processor(img: Any, preprocessor: Any, size: Tuple[int, int]) -> np.ndarray:
    """Use a saved AutoImageProcessor for preprocessing."""
    from PIL import Image

    img = img.resize(size, Image.BILINEAR)
    inputs = preprocessor(
        images=[img],
        do_resize=False,
        do_center_crop=False,
        do_rescale=True,
        do_normalize=True,
        return_tensors="np",
    )
    return inputs["pixel_values"].astype(np.float32)


def _extract_audio_tower_and_projector(model: Any) -> Tuple[Any, Any]:
    """Locate audio_tower and multi_modal_projector in a Qwen2Audio model."""
    audio_tower = getattr(model, "audio_tower", None) or getattr(
        getattr(model, "model", None), "audio_tower", None
    )
    projector = getattr(model, "multi_modal_projector", None) or getattr(
        getattr(model, "model", None), "multi_modal_projector", None
    )
    return audio_tower, projector


def _extract_qwen_audio_features(model: Any, input_features: Any) -> Any:
    """Run audio_tower and multi_modal_projector on (B, 128, 3000) input."""
    import torch

    audio_tower, projector = _extract_audio_tower_and_projector(model)
    if audio_tower is None or projector is None:
        raise RuntimeError("Could not locate audio_tower or multi_modal_projector")

    with torch.no_grad():
        audio_outputs = audio_tower(input_features)
        last_hidden_state = (
            audio_outputs.last_hidden_state
            if hasattr(audio_outputs, "last_hidden_state")
            else audio_outputs[0]
        )
        audio_features = projector(last_hidden_state)
    return audio_features


def _whisper_feature_extractor_to_input(
    feature_extractor: Any, samples: np.ndarray, return_attention_mask: bool = False
) -> Dict[str, Any]:
    """Call a Whisper feature extractor and pad to the Qwen2-Audio max length."""
    return feature_extractor(
        samples,
        sampling_rate=16000,
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=return_attention_mask,
    )


def _real_audio_output_length(feature_extractor: Any, n_real_samples: int) -> int:
    """Estimate the number of non-padding audio tokens after the audio tower.

    This mirrors the internal helper used by Qwen2Audio.
    """
    hop_length = getattr(feature_extractor, "hop_length", 160)
    n_fft = getattr(feature_extractor, "n_fft", 400)
    n_mel_frames = max(0, (n_real_samples - n_fft) // hop_length + 1)

    # After conv2 (k=3, s=2, p=1): (L - 1) // 2 + 1
    after_conv2 = (n_mel_frames - 1) // 2 + 1
    # After avg_pool1d(2, stride=2): (L - 2) // 2 + 1
    after_pool = (after_conv2 - 2) // 2 + 1
    return after_pool


def _encode_image(state: ServerState, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        raw_img = b64_decode(payload.get("image", ""))
    except Exception as e:
        return {"ok": False, "error": f"invalid base64 image: {e}"}

    width = payload.get("width")
    height = payload.get("height")
    return_sequence = bool(payload.get("return_sequence", True))

    try:
        img = decode_image(raw_img, width, height)
    except Exception as e:
        return {"ok": False, "error": f"could not decode image: {e}"}

    enc, err = state.image_encoder.get()
    if enc is None:
        return {"ok": False, "error": err or "image encoder not loaded"}

    try:
        if enc["type"] == "hf":
            feats = _get_image_features_from_hf(state, enc, img)
        elif enc["type"] == "onnx":
            w = width or img.size[0]
            h = height or img.size[1]
            if enc.get("preprocessor") is not None:
                arr = _preprocess_image_with_processor(img, enc["preprocessor"], (w, h))
            else:
                arr = preprocess_image_for_onnx(img, (w, h))
            feats = _run_onnx(enc["session"], arr, "image")
        else:
            return {"ok": False, "error": f"unknown image encoder type: {enc['type']}"}
    except Exception as e:
        log.exception("image encoding failed")
        return {"ok": False, "error": f"image encoding failed: {e}"}

    # feats shape: (batch, N, 4096); batch is 1.
    feats = np.squeeze(feats, axis=0) if feats.shape[0] == 1 else feats[0]
    mean_query = feats.mean(axis=0)

    result: Dict[str, Any] = {
        "ok": True,
        "model": enc.get("name", "llava-1.5-7b"),
        "dim": int(feats.shape[-1]),
        "n_queries": int(feats.shape[0]),
        "unit_query": mean_query.tolist(),
    }
    if return_sequence:
        result["unit_queries"] = feats.tolist()
    return result


def _encode_audio(state: ServerState, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        raw = b64_decode(payload.get("audio", ""))
    except Exception as e:
        return {"ok": False, "error": f"invalid base64 audio: {e}"}

    try:
        samples, sr = read_wav_bytes(raw)
    except Exception as e:
        return {"ok": False, "error": f"could not decode WAV: {e}"}

    requested_sr = int(payload.get("sample_rate", 16000))
    if sr != requested_sr:
        try:
            samples = resample_audio(samples, sr, requested_sr)
            sr = requested_sr
        except Exception as e:
            return {"ok": False, "error": f"could not resample audio: {e}"}

    if sr != 16000:
        try:
            samples = resample_audio(samples, sr, 16000)
            sr = 16000
        except Exception as e:
            return {"ok": False, "error": f"audio must be 16000 Hz: {e}"}

    real_sample_count = len(samples)
    return_sequence = bool(payload.get("return_sequence", True))

    enc, err = state.audio_encoder.get()
    if enc is None:
        return {"ok": False, "error": err or "audio encoder not loaded"}

    try:
        if enc["type"] == "hf":
            processor = enc["processor"]
            feature_extractor = getattr(processor, "feature_extractor", None)
            if feature_extractor is None:
                raise RuntimeError("Audio processor has no feature_extractor")

            inputs = _whisper_feature_extractor_to_input(
                feature_extractor, samples, return_attention_mask=True
            )
            input_features = inputs["input_features"].to(
                device=state.args.device, dtype=state._resolve_dtype()
            )
            audio_features = _extract_qwen_audio_features(enc["model"], input_features)
            feats = audio_features.detach().cpu().float().numpy()

            # If we have a real audio length, mask the padding tokens so the mean
            # is computed only over the actual content.
            if "feature_attention_mask" in inputs:
                real_len = _real_audio_output_length(feature_extractor, real_sample_count)
                real_len = min(real_len, feats.shape[1])
                if real_len > 0:
                    feats = feats[:, :real_len, :]

        elif enc["type"] == "onnx":
            feature_extractor = enc.get("feature_extractor")
            if feature_extractor is None:
                raise RuntimeError(
                    "Audio encoder ONNX requires the saved Qwen2Audio feature extractor"
                )

            inputs = _whisper_feature_extractor_to_input(
                feature_extractor, samples, return_attention_mask=False
            )
            arr = inputs["input_features"].cpu().numpy().astype(np.float32)
            feats = _run_onnx(enc["session"], arr, "audio")
        else:
            return {"ok": False, "error": f"unknown audio encoder type: {enc['type']}"}
    except Exception as e:
        log.exception("audio encoding failed")
        return {"ok": False, "error": f"audio encoding failed: {e}"}

    # feats shape: (batch, T, 4096)
    if feats.shape[0] != 1:
        feats = feats[:1]
    feats = np.squeeze(feats, axis=0)
    mean_query = feats.mean(axis=0)

    result: Dict[str, Any] = {
        "ok": True,
        "model": enc.get("name", "qwen2-audio-7b"),
        "dim": int(feats.shape[-1]),
        "n_queries": int(feats.shape[0]),
        "unit_query": mean_query.tolist(),
    }
    if return_sequence:
        result["unit_queries"] = feats.tolist()
    return result


# ---------------------------------------------------------------------------
# ONNX / PT decoder helpers
# ---------------------------------------------------------------------------

def _run_onnx(session: Any, arr: np.ndarray, modality: str) -> np.ndarray:
    """Run an ONNX session with a single input, returning the first output."""
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_meta.name: arr})[0]


def _prepare_decoder_input(
    mean_unit_query: List[float],
    unit_queries: Optional[List[List[float]]],
    input_shape: List[Union[int, str, None]],
) -> np.ndarray:
    """Reshape incoming concept vectors to match the decoder's expected input.

    Supports:
      - 2-D (batch, concept)
      - 3-D (batch, sequence, concept)
      - 4-D (batch, concept, 1, 1)
    """
    mean_arr = np.array(mean_unit_query, dtype=np.float32)
    if mean_arr.ndim == 1:
        mean_arr = mean_arr.reshape(1, -1)
    elif mean_arr.ndim == 2 and mean_arr.shape[0] != 1:
        mean_arr = mean_arr.mean(axis=0, keepdims=True)

    if unit_queries is not None:
        seq_arr = np.array(unit_queries, dtype=np.float32)
        if seq_arr.ndim == 2:
            seq_arr = seq_arr.reshape(1, *seq_arr.shape)
    else:
        seq_arr = mean_arr.reshape(1, 1, mean_arr.shape[-1])

    # Determine rank from the ONNX shape description.
    rank = len(input_shape)

    if rank == 2:
        return mean_arr
    if rank == 3:
        return seq_arr
    if rank >= 4:
        return mean_arr.reshape(mean_arr.shape[0], mean_arr.shape[1], 1, 1)
    return mean_arr


def _run_decoder_pt(model: Any, mean_unit_query: List[float], unit_queries: Optional[List[List[float]]]) -> Any:
    """Run a PyTorch decoder, trying a few common input shapes."""
    import torch

    mean_arr = np.array(mean_unit_query, dtype=np.float32)
    if mean_arr.ndim == 1:
        mean_arr = mean_arr.reshape(1, -1)
    elif mean_arr.ndim == 2 and mean_arr.shape[0] != 1:
        mean_arr = mean_arr.mean(axis=0, keepdims=True)

    candidates = []

    # 2-D (1, C)
    candidates.append(torch.from_numpy(mean_arr.astype(np.float32)))

    # 4-D (1, C, 1, 1)
    c = mean_arr.shape[1]
    candidates.append(torch.from_numpy(mean_arr.reshape(1, c, 1, 1).astype(np.float32)))

    # 3-D (1, N, C)
    if unit_queries is not None:
        seq = np.array(unit_queries, dtype=np.float32)
        if seq.ndim == 2:
            seq = seq.reshape(1, *seq.shape)
        elif seq.ndim == 1:
            seq = seq.reshape(1, 1, -1)
        candidates.append(torch.from_numpy(seq.astype(np.float32)))
    else:
        candidates.append(torch.from_numpy(mean_arr.reshape(1, 1, c).astype(np.float32)))

    last_err = None
    for x in candidates:
        try:
            with torch.no_grad():
                out = model(x)
            if isinstance(out, torch.Tensor):
                return out.detach().cpu().numpy()
            if isinstance(out, (list, tuple)) and len(out) == 1:
                return out[0].detach().cpu().numpy()
            return out
        except Exception as e:
            last_err = e
    raise RuntimeError(f"PyTorch decoder inference failed: {last_err}")


def _run_decoder_onnx(
    session: Any, mean_unit_query: List[float], unit_queries: Optional[List[List[float]]]
) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    arr = _prepare_decoder_input(mean_unit_query, unit_queries, input_meta.shape)
    return session.run([output_name], {input_meta.name: arr})[0]


def _decode_image(state: ServerState, payload: Dict[str, Any]) -> Dict[str, Any]:
    mean_unit_query = payload.get("mean_unit_query")
    unit_queries = payload.get("unit_queries")
    mime_type = payload.get("mime_type", "image/png")
    width = int(payload.get("width", 224))
    height = int(payload.get("height", 224))

    if not isinstance(mean_unit_query, list):
        return {"ok": False, "error": "mean_unit_query is required"}

    dec, err = state.image_decoder.get()
    if dec is None:
        return {"ok": False, "error": err or "image decoder not loaded"}

    try:
        if dec["type"] == "onnx":
            out = _run_decoder_onnx(dec["session"], mean_unit_query, unit_queries)
        elif dec["type"] == "pt":
            out = _run_decoder_pt(dec["model"], mean_unit_query, unit_queries)
        else:
            return {"ok": False, "error": f"unknown decoder type: {dec['type']}"}
    except Exception as e:
        log.exception("image decoding failed")
        return {"ok": False, "error": f"image decoding failed: {e}"}

    return _image_output_to_b64(out, mime_type, width, height)


def _image_output_to_b64(out: np.ndarray, mime_type: str, width: int, height: int) -> Dict[str, Any]:
    """Convert a decoder output array to a base64 image."""
    from PIL import Image

    # Flatten leading dimensions and bring to HWC form.
    while out.ndim > 4:
        out = out[0]

    if out.ndim == 4:
        out = out[0]  # (C, H, W)

    if out.ndim == 3:
        if out.shape[0] in (1, 3):
            # (C, H, W)
            out = out.transpose(1, 2, 0)
        # else assume (H, W, C)
    elif out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    elif out.ndim == 1:
        side = int(np.sqrt(out.size))
        out = out[: side * side].reshape(side, side)
        out = np.stack([out] * 3, axis=-1)

    # Resize to the requested output size if the decoder did not.
    if out.shape[0] != height or out.shape[1] != width:
        img = Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8))
        img = img.resize((width, height), Image.BILINEAR)
        out = np.array(img, dtype=np.float32) / 255.0

    # Normalize to [0, 1] then to uint8.
    if out.max() > 1.0:
        out = out / 255.0
    out = np.clip(out, 0.0, 1.0)
    uint8_img = (out * 255.0).astype(np.uint8)
    img = Image.fromarray(uint8_img)

    return {"ok": True, "image": pil_to_b64(img, mime_type)}


def _decode_audio(state: ServerState, payload: Dict[str, Any]) -> Dict[str, Any]:
    mean_unit_query = payload.get("mean_unit_query")
    unit_queries = payload.get("unit_queries")
    mime_type = payload.get("mime_type", "audio/wav")
    sample_rate = int(payload.get("sample_rate", 16000))
    length_hint = int(payload.get("length_hint", 16000))

    if not isinstance(mean_unit_query, list):
        return {"ok": False, "error": "mean_unit_query is required"}

    dec, err = state.audio_decoder.get()
    if dec is None:
        return {"ok": False, "error": err or "audio decoder not loaded"}

    try:
        if dec["type"] == "onnx":
            out = _run_decoder_onnx(dec["session"], mean_unit_query, unit_queries)
        elif dec["type"] == "pt":
            out = _run_decoder_pt(dec["model"], mean_unit_query, unit_queries)
        else:
            return {"ok": False, "error": f"unknown decoder type: {dec['type']}"}
    except Exception as e:
        log.exception("audio decoding failed")
        return {"ok": False, "error": f"audio decoding failed: {e}"}

    # Flatten output to 1-D audio waveform.
    while out.ndim > 1:
        out = out.reshape(-1)

    if out.size == 0:
        return {"ok": False, "error": "decoder produced empty audio"}

    # Clamp to [-1, 1] if the model is in that range; otherwise clip.
    if out.max() > 1.0 or out.min() < -1.0:
        out = np.clip(out, -1.0, 1.0)

    # Pad or trim to the requested length_hint.
    if out.size < length_hint:
        out = np.concatenate([out, np.zeros(length_hint - out.size, dtype=np.float32)])
    elif out.size > length_hint and length_hint > 0:
        out = out[:length_hint]

    if mime_type == "audio/wav":
        wav_bytes = write_wav_bytes(out, sample_rate)
        return {"ok": True, "audio": b64_encode(wav_bytes)}

    return {"ok": False, "error": f"unsupported audio mime type: {mime_type}"}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class MultimodalEncDecHandler(BaseHTTPRequestHandler):
    state: Optional[ServerState] = None

    def _set_headers(self, code: int = 200, length: Optional[int] = None):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_json(self, data: Any, code: int = 200):
        body = safe_json(data)
        self._set_headers(code, len(body))
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._set_headers(204)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send_json({"ok": False, "error": f"invalid JSON: {e}"}, 400)
            return None

    def do_GET(self):
        if self.path in ("/status", "/"):
            return self._send_json(self._status())
        self._send_json({"ok": False, "error": f"unknown endpoint: {self.path}"}, 404)

    def do_POST(self):
        if self.path == "/status":
            return self._send_json(self._status())

        if self.path == "/enc/image":
            payload = self._read_json()
            if payload is None:
                return
            return self._send_json(_encode_image(self.state, payload))

        if self.path == "/enc/audio":
            payload = self._read_json()
            if payload is None:
                return
            return self._send_json(_encode_audio(self.state, payload))

        if self.path == "/dec/image":
            payload = self._read_json()
            if payload is None:
                return
            return self._send_json(_decode_image(self.state, payload))

        if self.path == "/dec/audio":
            payload = self._read_json()
            if payload is None:
                return
            return self._send_json(_decode_audio(self.state, payload))

        self._send_json({"ok": False, "error": f"unknown endpoint: {self.path}"}, 404)

    def _status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "image_encoder": self.state.image_encoder.status(),
            "audio_encoder": self.state.audio_encoder.status(),
            "image_decoder": self.state.image_decoder.status(),
            "audio_decoder": self.state.audio_decoder.status(),
        }

    def log_message(self, fmt: str, *args):
        log.info(fmt % args)


# ---------------------------------------------------------------------------
# CLI and entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phoenix multimodal enc/dec HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8085, help="Bind port (default 8085)")

    parser.add_argument(
        "--image-model", default="llava-hf/llava-1.5-7b-hf", help="Hugging Face image encoder id"
    )
    parser.add_argument(
        "--audio-model", default="Qwen/Qwen2-Audio-7B-Instruct", help="Hugging Face audio encoder id"
    )
    parser.add_argument("--image-model-dir", default="", help="Local Hugging Face image model dir")
    parser.add_argument("--audio-model-dir", default="", help="Local Hugging Face audio model dir")

    parser.add_argument(
        "--image-encoder-onnx",
        default=str(DEFAULT_IMAGE_ENCODER_ONNX),
        help="Optional exported image encoder ONNX path",
    )
    parser.add_argument(
        "--audio-encoder-onnx",
        default=str(DEFAULT_AUDIO_ENCODER_ONNX),
        help="Optional exported audio encoder ONNX path",
    )

    parser.add_argument(
        "--image-decoder",
        default=str(DEFAULT_IMAGE_DECODER),
        help="Path to image decoder ONNX or .pt checkpoint",
    )
    parser.add_argument(
        "--audio-decoder",
        default=str(DEFAULT_AUDIO_DECODER),
        help="Path to audio decoder ONNX or .pt checkpoint",
    )

    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="PyTorch dtype (default float32)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    state = ServerState(args)
    MultimodalEncDecHandler.state = state

    server = ThreadingHTTPServer((args.host, args.port), MultimodalEncDecHandler)
    log.info("Multimodal enc/dec server listening on http://%s:%s", args.host, args.port)
    log.info("  Image encoder: %s", state._image_model_path() or args.image_encoder_onnx)
    log.info("  Audio encoder: %s", state._audio_model_path() or args.audio_encoder_onnx)
    log.info("  Image decoder: %s", args.image_decoder)
    log.info("  Audio decoder: %s", args.audio_decoder)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

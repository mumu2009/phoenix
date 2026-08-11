#!/usr/bin/env python3
"""One-click pipeline for Phoenix multimodal encoders and decoders.

This script wires together the four parts of the new v7.5+1 multimodal pipeline:

1. image encoder: LLaVA-1.5-7B vision tower + projector
2. audio encoder: Qwen2-Audio-7B audio tower + projector
3. image decoder: 4096-d unit query -> 3 x 224 x 224 RGB
4. audio decoder: 4096-d unit query -> mel-spectrogram -> waveform

It can download/prepare the encoders, start the enc/dec HTTP server, and train
both decoders in a single command.  For environments with limited Hugging Face
connectivity (e.g. mainland China) use `--cn-mirror` or set `--hf-endpoint`.

Usage:
    # Full pipeline with automatic dummy data (for smoke test):
    python tools/run_multimodal_encdec_pipeline.py --generate-dummy-data

    # Use your own data:
    python tools/run_multimodal_encdec_pipeline.py \
        --image-data-dir data/images \
        --audio-data-dir data/audio \
        --audio-captions data/audio/captions.json

    # China network:
    python tools/run_multimodal_encdec_pipeline.py --cn-mirror --generate-dummy-data

    # Skip encoder preparation (use already exported ONNX):
    python tools/run_multimodal_encdec_pipeline.py --no-prepare-encoders

    # Train only one decoder:
    python tools/run_multimodal_encdec_pipeline.py --no-train-audio
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

# Force UTF-8 for subprocess prints on Windows (torch.onnx can emit emoji/checkmarks).
if not os.environ.get("PYTHONIOENCODING"):
    os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../v6.0Alixander
TOOLS_DIR = Path(__file__).resolve().parent           # .../phoenix/tools

DEFAULT_IMAGE_DATA = PROJECT_ROOT / "data" / "images"
DEFAULT_AUDIO_DATA = PROJECT_ROOT / "data" / "audio"
DEFAULT_AUDIO_CAPTIONS = PROJECT_ROOT / "data" / "audio" / "captions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "runtime_store" / "models" / "multimodal_dec"
DEFAULT_ENCODER_PORT = 8085


def _status_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/status"


def _post_json(url: str, payload: dict, timeout: float = 5.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_server(port: int, timeout: float = 120.0) -> Dict[str, Any]:
    """Wait until the encoder server responds to /status."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(_status_url(port), timeout=2.0) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(
        f"encoder server did not start on port {port} within {timeout}s: {last_err}"
    )


def start_encoder_server(args: argparse.Namespace) -> subprocess.Popen:
    """Start the multimodal enc/dec HTTP server and wait for it to be ready."""
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "multimodal_encdec_server.py"),
        "--port",
        str(args.encoder_port),
        "--device",
        args.device,
    ]
    if args.hf_endpoint:
        cmd += ["--hf-endpoint", args.hf_endpoint]
    if args.modelscope_endpoint:
        cmd += ["--modelscope-endpoint", args.modelscope_endpoint]
    if args.use_modelscope:
        cmd += ["--use-modelscope"]
    else:
        cmd += ["--no-use-modelscope"]
    if args.cn_mirror:
        cmd += ["--cn-mirror"]
    if args.allow_hf_download:
        cmd += ["--allow-hf-download"]

    print(f"[pipeline] starting encoder server on port {args.encoder_port} ...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        status = wait_for_server(args.encoder_port)
        print(f"[pipeline] encoder server ready")
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    return proc


def stop_encoder_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    print("[pipeline] stopping encoder server ...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def prepare_encoders(args: argparse.Namespace) -> None:
    """Run prepare_multimodal_encoders.py to download/export encoder ONNX files."""
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "prepare_multimodal_encoders.py"),
        "--device",
        args.device,
    ]
    if args.cn_mirror:
        cmd += ["--cn-mirror"]
    if args.hf_endpoint:
        cmd += ["--hf-endpoint", args.hf_endpoint]
    if args.modelscope_endpoint:
        cmd += ["--modelscope-endpoint", args.modelscope_endpoint]
    if args.use_modelscope:
        cmd += ["--use-modelscope"]
    else:
        cmd += ["--no-use-modelscope"]
    if args.image_model:
        cmd += ["--image-model", args.image_model]
    if args.audio_model:
        cmd += ["--audio-model", args.audio_model]
    if args.image_model_dir:
        cmd += ["--image-model-dir", args.image_model_dir]
    if args.audio_model_dir:
        cmd += ["--audio-model-dir", args.audio_model_dir]
    if args.no_hf_download:
        cmd += ["--no-hf-download"]

    print("[pipeline] preparing encoders ...")
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        raise RuntimeError("prepare_multimodal_encoders.py failed")


def generate_dummy_data(args: argparse.Namespace) -> None:
    """Generate a tiny image/audio dataset for smoke testing."""
    import numpy as np
    from PIL import Image
    import wave

    image_dir = Path(args.image_data_dir)
    audio_dir = Path(args.audio_data_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    captions: Dict[str, str] = {}
    sr = 16000
    duration = 1.0
    n_samples = int(sr * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    for i in range(5):
        # image: random 224x224 RGB
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img_path = image_dir / f"sample_{i:03d}.png"
        img.save(img_path)

        # audio: 1s sine wave at increasing frequency
        freq = 220 + i * 55
        wave_data = (np.sin(2.0 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
        wav_path = audio_dir / f"sample_{i:03d}.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(wave_data.tobytes())

        captions[wav_path.name] = f"synthetic tone at {freq} Hz"

    captions_path = Path(args.audio_captions)
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)

    print(f"[pipeline] generated dummy data: {image_dir} and {audio_dir}")


def train_image_decoder(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "train_multimodal_decoders.py"),
        "--modality",
        "image",
        "--data_dir",
        args.image_data_dir,
        "--output_dir",
        args.output_dir,
        "--encoder_url",
        f"http://127.0.0.1:{args.encoder_port}",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--decoder",
        args.decoder,
    ]
    if args.use_caption_loss:
        cmd += ["--use_caption_loss"]

    print("[pipeline] training image decoder ...")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def train_audio_decoder(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "train_multimodal_decoders.py"),
        "--modality",
        "audio",
        "--data_dir",
        args.audio_data_dir,
        "--captions",
        args.audio_captions,
        "--output_dir",
        args.output_dir,
        "--encoder_url",
        f"http://127.0.0.1:{args.encoder_port}",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
    ]
    if args.use_caption_loss:
        cmd += ["--use_caption_loss"]

    print("[pipeline] training audio decoder ...")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download/prepare multimodal encoders and train decoders in one shot."
    )

    # Data
    parser.add_argument(
        "--image-data-dir",
        default=str(DEFAULT_IMAGE_DATA),
        help=f"Directory with training images (default {DEFAULT_IMAGE_DATA})",
    )
    parser.add_argument(
        "--audio-data-dir",
        default=str(DEFAULT_AUDIO_DATA),
        help=f"Directory with training audio files (default {DEFAULT_AUDIO_DATA})",
    )
    parser.add_argument(
        "--audio-captions",
        default=str(DEFAULT_AUDIO_CAPTIONS),
        help=f"JSON file mapping audio filename to caption (default {DEFAULT_AUDIO_CAPTIONS})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Where to save trained decoder checkpoints (default {DEFAULT_OUTPUT})",
    )

    # Stages
    parser.add_argument(
        "--prepare-encoders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download/export encoder ONNX files (default: True)",
    )
    parser.add_argument(
        "--train-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train image decoder (default: True)",
    )
    parser.add_argument(
        "--train-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train audio decoder (default: True)",
    )
    parser.add_argument(
        "--generate-dummy-data",
        action="store_true",
        help="Create a tiny synthetic image/audio dataset for testing",
    )

    # Encoder model selection
    parser.add_argument(
        "--image-model",
        default="llava-hf/llava-1.5-7b-hf",
        help="Hugging Face / local image encoder id",
    )
    parser.add_argument(
        "--audio-model",
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Hugging Face / local audio encoder id",
    )
    parser.add_argument("--image-model-dir", default="", help="Local image model dir")
    parser.add_argument("--audio-model-dir", default="", help="Local audio model dir")

    # Network / mirror
    parser.add_argument(
        "--cn-mirror",
        action="store_true",
        help="Shortcut: use https://hf-mirror.com and ModelScope fallback",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", ""),
        help="Hugging Face mirror endpoint (env: HF_ENDPOINT)",
    )
    parser.add_argument(
        "--modelscope-endpoint",
        default=os.environ.get("MODELSCOPE_ENDPOINT", ""),
        help="ModelScope endpoint (env: MODELSCOPE_ENDPOINT)",
    )
    parser.add_argument(
        "--use-modelscope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try ModelScope if HF is unavailable (default: True)",
    )
    parser.add_argument(
        "--no-hf-download",
        action="store_true",
        help="Never download encoders; fail if not cached/local",
    )
    parser.add_argument(
        "--allow-hf-download",
        action="store_true",
        help="Allow the server to download full HF models at runtime",
    )

    # Training hyperparameters
    parser.add_argument("--device", default="cpu", help="PyTorch device (cpu/cuda)")
    parser.add_argument("--encoder-port", type=int, default=DEFAULT_ENCODER_PORT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--decoder",
        choices=["single", "painter"],
        default="single",
        help="Image decoder architecture",
    )
    parser.add_argument(
        "--use-caption-loss",
        action="store_true",
        help="Add caption prediction head to the decoder training",
    )

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.generate_dummy_data:
        generate_dummy_data(args)

    # Validate data before the long encoder prep.
    if args.train_image and not Path(args.image_data_dir).exists():
        raise RuntimeError(
            f"image data not found: {args.image_data_dir}. "
            "Use --generate-dummy-data or provide a real dataset."
        )
    if args.train_audio and not Path(args.audio_data_dir).exists():
        raise RuntimeError(
            f"audio data not found: {args.audio_data_dir}. "
            "Use --generate-dummy-data or provide a real dataset."
        )
    if args.train_audio and not Path(args.audio_captions).exists():
        raise RuntimeError(
            f"audio captions not found: {args.audio_captions}. "
            "Use --generate-dummy-data or provide a real dataset."
        )

    server_proc: Optional[subprocess.Popen] = None
    try:
        if args.prepare_encoders:
            prepare_encoders(args)

        server_proc = start_encoder_server(args)

        if args.train_image:
            rc = train_image_decoder(args)
            if rc != 0:
                print("[pipeline] image decoder training failed", file=sys.stderr)

        if args.train_audio:
            rc = train_audio_decoder(args)
            if rc != 0:
                print("[pipeline] audio decoder training failed", file=sys.stderr)

    finally:
        stop_encoder_server(server_proc)

    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

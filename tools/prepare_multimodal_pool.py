#!/usr/bin/env python3
"""Build paired real-data pools for the four additive residual JPEA models.

The shared "Unit" concept is produced by an LLM text encoder (BGE-small-en)
from a text description of each sample.  No teacher model is required:
the text description is the supervision.

Pairs generated:
  speech_encoder:  audio (1,1,1,16000) -> concept (1,1,1,128)
  speech_decoder:  concept (1,1,1,128) -> audio (1,1,1,15872)
  vision_encoder:  image (1,3,224,224) -> concept (1,1,1,128)
  vision_decoder:  concept (1,1,1,128) -> image (1,3,224,224)

Audio: MUSAN 16 kHz WAV files.  Text is derived from the dataset's
folder/source metadata (real, not synthetic).

Images: Tiny-ImageNet-200.  Text is the class name from words.txt.

Usage:
    python tools/prepare_multimodal_pool.py \
        --musan-dir /home/kali/phoenix/datasets/musan_16k \
        --imagenet-dir /home/kali/datasets/tiny-imagenet-200 \
        --bge-dir /home/kali/models/bge-small-en \
        --out-dir /home/kali/phoenix/additive_work/pools_real \
        --concept 128 \
        --max-samples 5000
"""

import argparse
import csv
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image

# Allow running from repo root or tools/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from llm_concept_encoder import LlmConceptEncoder


def musan_text_description(path: Path) -> str:
    """Derive a natural language description from a MUSAN filename.

    Examples:
      musan_music_fma-western-art_music-fma-wa-0075.wav
        -> "a music recording from the Free Music Archive western art collection"
      musan_speech_librivox_speech-librivox-0048.wav
        -> "a human speech recording from LibriVox"
      musan_noise_free-sound_noise-free-sound-0004.wav
        -> "an environmental noise recording from freesound"
    """
    parts = path.stem.split("_")
    if len(parts) < 3:
        return "an audio recording"

    # MUSAN filenames: musan_<category>_<source>_...
    category = parts[1]
    source = parts[2]

    category_map = {
        "music": "music",
        "speech": "human speech",
        "noise": "environmental noise",
    }
    cat_text = category_map.get(category, category)

    source_map = {
        "fma": "the Free Music Archive",
        "fma-western-art": "the Free Music Archive western art collection",
        "jamendo": "Jamendo",
        "rfm": "the RFM music library",
        "hd-classical": "a high-definition classical music collection",
        "librivox": "LibriVox",
        "us-gov": "a U.S. government speech recording",
        "free-sound": "freesound",
    }
    src_text = source_map.get(source, source)

    article = "an" if cat_text.startswith(("a", "e", "i", "o", "u")) else "a"
    if src_text:
        return f"{article} {cat_text} recording from {src_text}"
    return f"{article} {cat_text} recording"


def load_musan_samples(
    musan_dir: Path,
    max_samples: int,
    seed: int,
    encoder_input_samples: int = 16000,
    decoder_target_samples: int = 15872,
):
    """Return list of (text, encoder_input, decoder_target) for MUSAN."""
    wav_files = sorted(Path(musan_dir).rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"no .wav files under {musan_dir}")

    rng = np.random.default_rng(seed)
    wav_files = wav_files[:max_samples]
    print(f"[musan] using {len(wav_files)} wav files")

    samples = []
    for wf in wav_files:
        text = musan_text_description(wf)
        try:
            with wave.open(str(wf), "rb") as w:
                nchannels = w.getnchannels()
                sampwidth = w.getsampwidth()
                framerate = w.getframerate()
                nframes = w.getnframes()

                if framerate != 16000:
                    print(f"[warn] {wf} samplerate {framerate}, skipping")
                    continue

                if nframes < encoder_input_samples:
                    # Read whole short file and pad.
                    raw = w.readframes(nframes)
                else:
                    # Seek to a random start and read only the needed samples.
                    start_sample = rng.integers(0, nframes - encoder_input_samples + 1)
                    w.setpos(start_sample)
                    raw = w.readframes(encoder_input_samples)
        except Exception as e:
            print(f"[warn] cannot read {wf}: {e}")
            continue

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16)
        elif sampwidth == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        else:
            print(f"[warn] {wf} unsupported sample width {sampwidth}, skipping")
            continue

        if nchannels > 1:
            data = data.reshape(-1, nchannels).mean(axis=1)

        data = data.astype(np.float32) / 32768.0

        # Pad if the file was too short.
        if len(data) < encoder_input_samples:
            data = np.concatenate([data, np.zeros(encoder_input_samples - len(data), dtype=np.float32)])

        chunk = data[:encoder_input_samples]
        enc_in = chunk.reshape(1, 1, 1, encoder_input_samples)
        dec_out = chunk[:decoder_target_samples].reshape(1, 1, 1, decoder_target_samples)
        samples.append({"text": text, "encoder_input": enc_in, "decoder_target": dec_out})

    return samples


def load_imagenet_samples(imagenet_dir: Path, max_samples: int, seed: int):
    """Return list of (text, encoder_input, decoder_target) for Tiny-ImageNet."""
    imagenet_dir = Path(imagenet_dir)
    words_path = imagenet_dir / "words.txt"
    if not words_path.is_file():
        raise FileNotFoundError(f"words.txt not found: {words_path}")

    # Build class -> text mapping.
    class_to_text = {}
    with open(words_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wnid, desc = line.split("\t", 1)
            # Take the first synonym as the canonical English text.
            class_to_text[wnid] = desc.split(",")[0].strip()

    # Train directory: imagenet_dir/train/<wnid>/images/<wnid>_*.JPEG
    train_dir = imagenet_dir / "train"
    image_files = []
    for wnid_dir in sorted(train_dir.iterdir()):
        if not wnid_dir.is_dir():
            continue
        wnid = wnid_dir.name
        images_dir = wnid_dir / "images"
        if not images_dir.is_dir():
            continue
        for img in sorted(images_dir.glob("*.JPEG")):
            image_files.append((img, wnid))

    if not image_files:
        raise FileNotFoundError(f"no training images under {imagenet_dir}")

    rng = np.random.default_rng(seed)
    rng.shuffle(image_files)
    image_files = image_files[:max_samples]
    print(f"[imagenet] using {len(image_files)} training images")

    samples = []
    for img_path, wnid in image_files:
        text = f"a photo of {class_to_text.get(wnid, wnid)}"
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot read {img_path}: {e}")
            continue

        # Resize and center-crop to 224x224.
        w, h = pil.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        pil = pil.crop((left, top, left + s, top + s))
        pil = pil.resize((224, 224), getattr(Image, "Resampling", Image).BILINEAR)
        arr = np.array(pil, dtype=np.float32) / 255.0  # HWC, 0..1
        arr = arr.transpose(2, 0, 1)  # CHW
        img_tensor = arr.reshape(1, 3, 224, 224).astype(np.float32)

        samples.append(
            {"text": text, "encoder_input": img_tensor, "decoder_target": img_tensor}
        )

    return samples


def build_pools(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = LlmConceptEncoder(args.bge_dir, args.concept)

    # Audio samples.
    if args.musan_dir:
        audio_samples = load_musan_samples(
            Path(args.musan_dir), args.max_samples_per_modality, args.seed
        )
    else:
        audio_samples = []
        print("[musan] skipped (no --musan-dir)")

    # Image samples.
    if args.imagenet_dir:
        image_samples = load_imagenet_samples(
            Path(args.imagenet_dir), args.max_samples_per_modality, args.seed
        )
    else:
        image_samples = []
        print("[imagenet] skipped (no --imagenet-dir)")

    print(f"[pools] audio samples={len(audio_samples)} image samples={len(image_samples)}")

    # Encode all unique texts.
    all_texts = [s["text"] for s in audio_samples + image_samples]
    unique_texts = list(dict.fromkeys(all_texts))  # preserve order, dedup
    if not unique_texts:
        raise RuntimeError("no text descriptions found; cannot build pools")

    print(f"[encode] encoding {len(unique_texts)} unique descriptions ...")
    unique_concepts = encoder.encode(unique_texts)  # (N, concept)
    text_to_concept = {t: unique_concepts[i] for i, t in enumerate(unique_texts)}

    # Per-sample concept vectors, expanded to the model's expected (1, concept, 1, 1) shape.
    def concept_for(sample):
        c = text_to_concept[sample["text"]]  # (concept,)
        return c.reshape(1, args.concept, 1, 1)

    # Save metadata.
    meta = {
        "concept_dim": args.concept,
        "n_audio": len(audio_samples),
        "n_image": len(image_samples),
        "bge_dir": str(args.bge_dir),
        "unique_texts": unique_texts,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    def write_model_pool(model_name, modality_inputs, modality_targets, concept_inputs):
        """Write (inputs.npy, targets.npy) that bpu_evolve_additive.py expects.

        inputs.npy shape:   (N, *model_input_shape)
        targets.npy shape:  (N, *model_output_shape)
        """
        pool_dir = out_dir / model_name
        pool_dir.mkdir(parents=True, exist_ok=True)

        if "encoder" in model_name:
            # encoder:  modality -> concept
            inputs = np.array(modality_inputs)   # (N, *input_shape)
            targets = np.array(concept_inputs)   # (N, 1, concept, 1, 1)
        else:
            # decoder:  concept -> modality
            inputs = np.array(concept_inputs)    # (N, 1, concept, 1, 1)
            targets = np.array(modality_targets) # (N, *output_shape)

        np.save(pool_dir / "inputs.npy", inputs.astype(np.float32))
        np.save(pool_dir / "targets.npy", targets.astype(np.float32))

        src_samples = audio_samples if "speech" in model_name else image_samples
        with open(pool_dir / "texts.json", "w", encoding="utf-8") as f:
            json.dump([s["text"] for s in src_samples], f, ensure_ascii=False, indent=2)

        print(f"[pool] {model_name}: inputs={inputs.shape} targets={targets.shape} -> {pool_dir}")

    if audio_samples:
        audio_inputs = [s["encoder_input"] for s in audio_samples]
        audio_targets = [s["decoder_target"] for s in audio_samples]
        audio_concepts = [concept_for(s) for s in audio_samples]
        write_model_pool("speech_encoder", audio_inputs, audio_targets, audio_concepts)
        write_model_pool("speech_decoder", audio_inputs, audio_targets, audio_concepts)

    if image_samples:
        image_inputs = [s["encoder_input"] for s in image_samples]
        image_targets = [s["decoder_target"] for s in image_samples]
        image_concepts = [concept_for(s) for s in image_samples]
        write_model_pool("vision_encoder", image_inputs, image_targets, image_concepts)
        write_model_pool("vision_decoder", image_inputs, image_targets, image_concepts)

    print(f"[done] wrote pools to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real multimodal data pools")
    parser.add_argument("--musan-dir", default="/home/kali/phoenix/datasets/musan_16k", help="MUSAN 16k wav directory")
    parser.add_argument("--imagenet-dir", default="/home/kali/datasets/tiny-imagenet-200", help="Tiny-ImageNet-200 directory")
    parser.add_argument("--bge-dir", default="/home/kali/models/bge-small-en", help="BGE model directory")
    parser.add_argument("--out-dir", required=True, help="Output pool directory")
    parser.add_argument("--concept", type=int, default=128, help="Concept dimension")
    parser.add_argument("--max-samples-per-modality", type=int, default=5000, help="Max samples from each modality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128, help="BGE tokenizer max length")
    args = parser.parse_args()

    build_pools(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

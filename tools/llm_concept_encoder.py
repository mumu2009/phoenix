#!/usr/bin/env python3
"""Encode text descriptions into a fixed-size "Unit" concept vector.

This uses a local BGE-small-en ONNX model downloaded via ModelScope, then
projects the 384-D mean-pooled embedding down to `concept_dim` (default 128)
using a fixed orthonormal projection matrix.  The result is a real-valued
float32 vector that serves as the shared concept target for speech/vision
encoders and the shared concept input for the corresponding decoders.

Usage:
    python tools/llm_concept_encoder.py \
        --bge-dir /home/kali/models/bge-small-en \
        --text "a photo of a goldfish" \
        --concept 128

Or encode a JSONL file in batch:
    python tools/llm_concept_encoder.py \
        --bge-dir /home/kali/models/bge-small-en \
        --in-file texts.jsonl \
        --out-file concepts.npy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def make_orthonormal_projection(in_dim: int, out_dim: int, seed: int = 42) -> np.ndarray:
    """Return a (in_dim, out_dim) matrix whose columns are orthonormal."""
    rng = np.random.default_rng(seed)
    # Use QR on a random Gaussian to get an orthonormal basis for R^out_dim.
    a = rng.standard_normal(size=(in_dim, out_dim), dtype=np.float64)
    q, _ = np.linalg.qr(a)
    return q.astype(np.float32)


def load_or_create_projection(projection_path: Path, in_dim: int, out_dim: int) -> np.ndarray:
    projection_path = Path(projection_path)
    if projection_path.is_file():
        proj = np.load(projection_path)
        if proj.shape != (in_dim, out_dim):
            raise ValueError(
                f"loaded projection shape {proj.shape} != expected {(in_dim, out_dim)}"
            )
        return proj.astype(np.float32)
    proj = make_orthonormal_projection(in_dim, out_dim)
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(projection_path, proj)
    return proj


class LlmConceptEncoder:
    """Wrap the BGE-small-en ONNX + tokenizer and the projection head."""

    def __init__(
        self,
        bge_dir: Path,
        concept_dim: int = 128,
        max_length: int = 128,
    ):
        self.bge_dir = Path(bge_dir)
        self.concept_dim = concept_dim
        self.max_length = max_length

        # Load tokenizer and enable padding/truncation.
        tok_path = self.bge_dir / "tokenizer.json"
        if not tok_path.is_file():
            raise FileNotFoundError(f"BGE tokenizer not found: {tok_path}")
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding(length=max_length)

        # Load ONNX model.  Prefer CPU (Kali has no GPU).
        onnx_path = self.bge_dir / "onnx" / "model.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"BGE ONNX not found: {onnx_path}")
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

        # Sentence-BERT ONNX model returns last_hidden_state (batch, seq, 384).
        self.hidden_dim = 384

        # Fixed orthonormal projection 384 -> concept_dim.
        proj_path = self.bge_dir / f"concept_projection_{concept_dim}.npy"
        self.projection = load_or_create_projection(
            proj_path, self.hidden_dim, concept_dim
        )

    def _mean_pool(
        self, last_hidden_state: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Mean-pool over non-padding tokens, returns (batch, hidden_dim)."""
        mask = attention_mask.astype(np.float32)[..., None]
        masked = last_hidden_state * mask
        sum_emb = masked.sum(axis=1)
        lengths = mask.sum(axis=1)
        # Avoid division by zero for empty sequences.
        lengths = np.maximum(lengths, 1e-9)
        return sum_emb / lengths

    def encode(self, texts) -> np.ndarray:
        """Encode a string or a list of strings into (N, concept_dim)."""
        if isinstance(texts, str):
            texts = [texts]

        all_input_ids = []
        all_attention = []
        all_type_ids = []
        for t in texts:
            enc = self.tokenizer.encode(str(t))
            all_input_ids.append(enc.ids)
            all_attention.append(enc.attention_mask)
            all_type_ids.append(enc.type_ids)

        input_ids = np.array(all_input_ids, dtype=np.int64)
        attention_mask = np.array(all_attention, dtype=np.int64)
        token_type_ids = np.array(all_type_ids, dtype=np.int64)

        last_hidden = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0]

        mean_emb = self._mean_pool(last_hidden, attention_mask)
        # Project to the shared concept space.
        concepts = mean_emb @ self.projection
        return concepts.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode text into Unit concepts")
    parser.add_argument(
        "--bge-dir",
        default="/home/kali/models/bge-small-en",
        help="Directory containing the BGE model and tokenizer",
    )
    parser.add_argument("--concept", type=int, default=128, help="Concept dimension")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument("--text", default=None, help="Single text to encode")
    parser.add_argument("--in-file", default=None, help="JSONL file with {'text': ...} per line")
    parser.add_argument("--out-file", default=None, help="Output .npy file for batch encoding")
    args = parser.parse_args()

    encoder = LlmConceptEncoder(args.bge_dir, args.concept, args.max_length)

    if args.text:
        print(encoder.encode(args.text).tolist())
        return 0

    if args.in_file and args.out_file:
        texts = []
        with open(args.in_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    texts.append(obj.get("text", ""))
                else:
                    texts.append(str(obj))
        if not texts:
            print("[warn] no texts found")
            return 1
        concepts = encoder.encode(texts)
        np.save(args.out_file, concepts)
        print(f"[encode] wrote {concepts.shape} concepts to {args.out_file}")
        return 0

    print("[error] provide --text or both --in-file and --out-file")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

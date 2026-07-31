#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import logging
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer

"""
model_deployment_edge_example.py - Minimal edge inference server for Phoenix v7.0.

This example implements the HTTP/JSON protocol that Phoenix's remote vision and
speech world models use.  It supports two request types:

* encode: receives a base64 encoded payload, returns an embedding of the
  requested `conceptDim`.
* decode: receives a concept vector and an optional `lengthHint`, returns a
  base64 encoded reconstructed payload.

You can run this on an edge device (Raspberry Pi, RDK X5, Jetson, desktop GPU,
etc.) and point Phoenix at it with:

    phoenix_main.exe \
        --vision-placement remote \
        --vision-remote-url http://edge-device-ip:5000/infer

A real deployment would replace the stub embedding/synthesis with an actual
model (Ollama vision, ONNX Runtime, PyTorch, RDK X5 BPU bridge, etc.).
"""

# 1x1 PNG placeholder used by decode for image requests.
PLACEHOLDER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKOSIwAAAABJRU5ErkJggg=="
)


def _concept_to_embedding(concept_dim: int, seed: bytes) -> list[float]:
    """Deterministic placeholder embedding from a byte seed."""
    digest = hashlib.sha256(seed).digest()
    return [
        float(((digest[i % len(digest)] - 128) / 128.0) * 0.5)
        for i in range(concept_dim)
    ]


def _decode_placeholder(modality: str, mime_type: str, length_hint: int) -> bytes:
    """Return a deterministic placeholder payload for a decode request."""
    if modality == "image":
        return base64.b64decode(PLACEHOLDER_PNG_B64)

    if modality == "audio":
        samples = length_hint if length_hint > 0 else 256
        return b"\x80" * samples

    # Unknown modality: a small JSON concept payload.
    return json.dumps(
        {"sourceModality": modality, "mimeType": mime_type}
    ).encode("utf-8")

class EdgeHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _encode(self, payload: dict) -> None:
        concept_dim = int(payload.get("conceptDim", 128))
        b64 = payload.get("payloadBase64", "")
        if not b64:
            self._send_json(400, {"ok": False, "error": "missing payloadBase64"})
            return

        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            self._send_json(400, {"ok": False, "error": f"base64 decode failed: {e}"})
            return

        embedding = _concept_to_embedding(concept_dim, raw)
        logging.info(
            "Processed %s encode request: mime=%s dim=%d raw_bytes=%d",
            payload.get("modality", "unknown"),
            payload.get("mimeType", "unknown"),
            concept_dim,
            len(raw),
        )
        self._send_json(
            200,
            {
                "ok": True,
                "modality": payload.get("modality"),
                "embedding": embedding,
                "backend": "model_deployment_edge_example.py",
            },
        )

    def _decode(self, payload: dict) -> None:
        modality = payload.get("modality")
        concept_vector = payload.get("conceptVector")
        if not isinstance(concept_vector, list) or not concept_vector:
            self._send_json(400, {"ok": False, "error": "missing conceptVector"})
            return

        length_hint = int(payload.get("lengthHint", 0))
        mime_type = payload.get("mimeType", "")

        # Use the first concept value as a stable seed to produce a consistent
        # placeholder payload for the same concept vector.
        seed = (
            struct.pack(f"<{len(concept_vector)}f", *concept_vector)
            + mime_type.encode("utf-8")
        )
        placeholder = _decode_placeholder(modality, mime_type, length_hint)

        logging.info(
            "Processed %s decode request: mime=%s dim=%d length_hint=%d",
            modality,
            mime_type,
            len(concept_vector),
            length_hint,
        )
        self._send_json(
            200,
            {
                "ok": True,
                "modality": modality,
                "mimeType": mime_type,
                "payloadBase64": base64.b64encode(placeholder).decode("ascii"),
                "backend": "model_deployment_edge_example.py",
            },
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            self._send_json(400, {"ok": False, "error": "empty body"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "invalid json"})
            return

        if payload.get("decode") is True:
            self._decode(payload)
        else:
            self._encode(payload)

    def log_message(self, fmt: str, *args) -> None:
        logging.info(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix v7.0 edge inference example")
    parser.add_argument("--host", default="0.0.0.0", help="bind address")
    parser.add_argument("--port", type=int, default=5000, help="bind port")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    server = HTTPServer((args.host, args.port), EdgeHandler)
    logging.info("Edge inference server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down")


if __name__ == "__main__":
    main()

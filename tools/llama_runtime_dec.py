#!/usr/bin/env python3
"""llama_runtime_dec.py - Runtime enc/infer/dec wrapper for llama3.1 8b.

This script runs a small HTTP server that conceptually splits the local
llama-server into three stages:

    enc:   tokens / text  -> token ids + embeddings
    infer: token ids      -> logits / next-token distribution
    dec:   token ids      -> text (and, for audio/video, routes to the
           appropriate JEPA decoder)

It exists because:
  * GGUF model files and `outsides/llamacpp` are in `.gitignore` and should
    not be committed.
  * llama.cpp modifications are tracked in `phoenix/llama_server_mods/` and
    compiled into `llama-server.exe`, but the *runtime* separation of decoders
    (especially audio/video) must live outside the binary.

The wrapper talks to the existing `llama-server` over HTTP:
    /tokenize      -> token ids
    /detokenize    -> text
    /completion    -> generated text / logprobs
    /embedding     -> token embeddings

For audio/video output the wrapper returns a JSON envelope describing the
required decoder call, or (if a local ONNX/JEPA decoder is configured) runs it.
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LlamaRuntimeDec:
    def __init__(self, base_url: str = "http://127.0.0.1:8082"):
        self.base_url = base_url.rstrip("/")
        self.props = self._get_props()

    def _get_props(self) -> Dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/props", timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}

    def enc(self, text: str, add_bos: bool = True, normalize: bool = False) -> Dict[str, Any]:
        """Text -> token ids + optional mean embedding."""
        tok = _post_json(
            f"{self.base_url}/tokenize",
            {"content": text, "add_bos": add_bos, "with_pieces": True},
        )
        result: Dict[str, Any] = {
            "tokens": tok.get("tokens", []),
            "pieces": tok.get("pieces", []),
            "text": text,
            "model": self.props.get("default_generation_settings", {}).get("model"),
        }
        if normalize:
            emb = _post_json(
                f"{self.base_url}/embedding",
                {"content": text, "image_data": []},
            )
            result["embedding"] = emb.get("embedding", [])
        return result

    def infer(self, tokens: List[int], n_predict: int = 1,
              temperature: float = 0.7, top_p: float = 0.9,
              logit_bias: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Run the main inference body on token ids and return the next tokens."""
        text = self.detokenize(tokens)
        payload: Dict[str, Any] = {
            "prompt": text,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "n_probs": 5,
            "stream": False,
        }
        if logit_bias:
            payload["logit_bias"] = logit_bias
        completion = _post_json(f"{self.base_url}/completion", payload)
        return {
            "content": completion.get("content", ""),
            "tokens_generated": completion.get("tokens_predicted", 0),
            "probs": completion.get("completion_probabilities", []),
            "timings": completion.get("timings", {}),
        }

    def dec_text(self, tokens: List[int]) -> str:
        """Token ids -> text."""
        return self.detokenize(tokens)

    def detokenize(self, tokens: List[int]) -> str:
        try:
            data = _post_json(
                f"{self.base_url}/detokenize",
                {"tokens": tokens},
            )
            return data.get("content", "")
        except Exception as e:
            return f"[dec error: {e}]"

    def dec_audio(self, concept_vector: List[float], sample_rate: int = 16000,
                  decoder_path: Optional[str] = None) -> Dict[str, Any]:
        """Audio concept vector -> waveform bytes (uses external decoder if configured)."""
        if decoder_path and os.path.exists(decoder_path):
            # Placeholder: an actual implementation would run the ONNX/JEPA speech decoder.
            return {
                "status": "decoder_configured",
                "decoder_path": decoder_path,
                "concept_dim": len(concept_vector),
                "sample_rate": sample_rate,
            }
        return {
            "status": "no_decoder",
            "hint": "Route this SemanticUnit to phoenix::io::MixedModalConceptBridge::decode "
                    "with MixedModalModality::Audio, or configure a local speech_decoder.onnx",
            "concept_dim": len(concept_vector),
            "sample_rate": sample_rate,
        }

    def dec_video(self, concept_vector: List[float], width: int = 224, height: int = 224,
                  frames: int = 1, decoder_path: Optional[str] = None) -> Dict[str, Any]:
        """Video concept vector -> frame bytes (uses external decoder if configured)."""
        if decoder_path and os.path.exists(decoder_path):
            # Placeholder: run an ONNX/JEPA image/sequence decoder per frame.
            return {
                "status": "decoder_configured",
                "decoder_path": decoder_path,
                "concept_dim": len(concept_vector),
                "width": width,
                "height": height,
                "frames": frames,
            }
        return {
            "status": "no_decoder",
            "hint": "Route this SemanticUnit to phoenix::io::MixedModalConceptBridge::decode "
                    "with MixedModalModality::Video, or configure a local video_decoder.onnx",
            "concept_dim": len(concept_vector),
            "width": width,
            "height": height,
            "frames": frames,
        }


class Handler(BaseHTTPRequestHandler):
    runtime: Optional[LlamaRuntimeDec] = None

    def _ok(self, data: Any, code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bad(self, msg: str, code: int = 400):
        self._ok({"error": msg}, code)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_len)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._bad("Invalid JSON")

        if self.path == "/enc":
            text = payload.get("text", "")
            norm = payload.get("normalize", False)
            if not text:
                return self._bad("text required")
            self._ok(self.runtime.enc(text, normalize=norm))

        elif self.path == "/infer":
            tokens = payload.get("tokens")
            if not isinstance(tokens, list):
                text = payload.get("text", "")
                if not text:
                    return self._bad("tokens or text required")
                tok = self.runtime.enc(text)
                tokens = tok["tokens"]
            n = payload.get("n_predict", 1)
            bias = payload.get("logit_bias")
            self._ok(self.runtime.infer(tokens, n_predict=n, logit_bias=bias))

        elif self.path == "/dec":
            tokens = payload.get("tokens")
            if isinstance(tokens, list) and tokens:
                self._ok({"text": self.runtime.dec_text(tokens)})
            else:
                text = payload.get("text", "")
                if not text:
                    return self._bad("tokens or text required")
                self._ok({"tokens": self.runtime.enc(text)["tokens"]})

        elif self.path == "/dec/audio":
            vec = payload.get("concept_vector", [])
            sr = payload.get("sample_rate", 16000)
            path = payload.get("decoder_path")
            self._ok(self.runtime.dec_audio(vec, sr, path))

        elif self.path == "/dec/video":
            vec = payload.get("concept_vector", [])
            w = payload.get("width", 224)
            h = payload.get("height", 224)
            f = payload.get("frames", 1)
            path = payload.get("decoder_path")
            self._ok(self.runtime.dec_video(vec, w, h, f, path))

        else:
            self._bad("Unknown endpoint", 404)

    def do_GET(self):
        if self.path == "/health":
            self._ok({"status": "ok", "llama_server": self.runtime.props})
        else:
            self._bad("Unknown endpoint", 404)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {fmt % args}\n")


def main():
    parser = argparse.ArgumentParser(description="Runtime enc/infer/dec wrapper for llama-server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8082",
                        help="llama-server base URL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    Handler.runtime = LlamaRuntimeDec(args.base_url)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"llama_runtime_dec listening on http://{args.host}:{args.port}")
    print(f"upstream llama-server: {args.base_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

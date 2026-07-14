"""
Lightweight HTTP proxy: translates phoenix /api/chat (Ollama format)
to llama-server /completion endpoint.

Usage: python llama_proxy.py --proxy-port 8083 --backend-port 8084
"""
import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen, Request as UReq

PROXY_PORT = 8083
BACKEND_URL = "http://127.0.0.1:8084"

LOG_FILE = os.environ.get("LLAMA_PROXY_LOG", "proxy_prompt.log")
_log_fh = None

# Markdown / chain-of-thought artifacts that sometimes leak into model replies.
MD_BOLD_RE = re.compile(r"\*\*+(.+?)\*\*+")
MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
MD_HEADER_RE = re.compile(r"^#+\s*", re.MULTILINE)
MD_LIST_RE = re.compile(r"^[\-*•]\s+", re.MULTILINE)
COT_PREFIX_RE = re.compile(
    r"^\s*(?:step\s*\d+[:.)]?|analyze|reasoning|first|second|third|finally|in\s+conclusion|the\s+answer\s+is)\s*[:.)]?\s*",
    re.IGNORECASE | re.MULTILINE,
)


def strip_format_artifacts(text: str) -> str:
    """Remove markdown formatting and chain-of-thought prefixes from a reply."""
    text = MD_HEADER_RE.sub("", text)
    text = MD_LIST_RE.sub("", text)
    text = MD_BOLD_RE.sub(r"\1", text)
    text = MD_ITALIC_RE.sub(r"\1", text)
    text = COT_PREFIX_RE.sub("", text)
    return text


def parse_context_hint(context_hint: str):
    """Parse Phoenix's contextHint into conversation messages and a summary note.

    The contextHint is produced by prepareChatContext and looks like:

        [Conversation history:
          User: ...
          Assistant: ...
          [Long-term memory summary|mode]: [mode] word1 word2 ...
          [Cross-session history]:
          ...
        ]

    Returns (history_messages, summary_text).  history_messages is a list of
    dicts with role/user/assistant.  summary_text is a single string that can be
    appended to the system prompt.
    """
    if not context_hint.startswith("[Conversation history:"):
        return [], context_hint

    history: list[dict[str, str]] = []
    summary_parts: list[str] = []
    cross_parts: list[str] = []
    in_cross = False

    for raw_line in context_hint.splitlines():
        line = raw_line.strip()
        if not line or line == "]":
            continue
        if line == "[Conversation history:":
            continue
        if line.startswith("[Long-term memory summary|"):
            colon = line.find(":")
            if colon != -1:
                summary_parts.append(line[colon + 1 :].strip())
            continue
        if line.startswith("[Cross-session history]:"):
            in_cross = True
            continue
        if in_cross:
            cross_parts.append(line)
            continue

        if line.startswith("User:"):
            history.append({"role": "user", "content": line[5:].strip()})
            continue
        if line.startswith("Assistant:"):
            history.append({"role": "assistant", "content": line[10:].strip()})
            continue

        # Continuation of the previous message's content.
        if history and not line.startswith("["):
            history[-1]["content"] += "\n" + line

    note_parts: list[str] = []
    if summary_parts:
        note_parts.append("Long-term memory summary: " + " ".join(summary_parts))
    if cross_parts:
        note_parts.append("Cross-session history:\n" + "\n".join(cross_parts))
    summary_text = "\n".join(note_parts)
    return history, summary_text


def _log(line: str) -> None:
    global _log_fh
    if _log_fh is None:
        try:
            _log_fh = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
        except Exception:
            pass
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()



class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

    def do_GET(self):
        body = json.dumps({"ok": True, "status": "proxy running"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        t0 = time.perf_counter()
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            p = json.loads(raw)
        except Exception:
            p = {}

        # Short-circuit style-adapter train_step: the benchmark explicitly disables it,
        # but Phoenix still sends it to the backend. Sending it to llama-server with an
        # empty prompt wastes ~50s per request. Return an empty OK instead.
        if self.path == "/api/style/adapter/train_step":
            body = json.dumps({"ok": True, "trained": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _log(f"[{t0:.6f}] PATH={self.path} SHORT_CIRCUIT=200 ELAPSED_MS={elapsed_ms:.2f}")
            print(f"[proxy] POST {self.path} -> 200 short-circuit", flush=True)
            return

        msgs = p.get("messages", []) if isinstance(p, dict) else []
        opts = p.get("options", {}) if isinstance(p, dict) else {}
        if isinstance(opts, dict):
            n_pred = opts.get("num_predict", 512)
        else:
            n_pred = 512
        if n_pred is None or (isinstance(n_pred, int) and n_pred <= 0):
            n_pred = 512
        # Phoenix uses camelCase maxTokens; OpenAI uses snake_case max_tokens
        if isinstance(p, dict):
            n_pred = p.get("max_tokens", p.get("maxTokens", n_pred))
        if not isinstance(n_pred, int) or n_pred <= 0:
            n_pred = 512

        # Build a chat-template prompt for the /completion endpoint.
        # This is the Llama-3.1 style template; it gives the model clear
        # user/assistant roles without relying on llama-server /api/chat.
        context_hint = p.get("contextHint", "").strip() if isinstance(p, dict) else ""

        def build_chat_prompt(messages):
            has_system = any(isinstance(m, dict) and m.get("role") == "system" for m in messages)
            parts = ["<|begin_of_text|>"]

            history_msgs, summary_text = parse_context_hint(context_hint) if context_hint else ([], "")

            if not has_system:
                system_text = (
                    "You are a helpful assistant. Answer directly and concisely. "
                    "Do not use markdown formatting, bullet points, or step-by-step reasoning unless explicitly asked."
                )
                if summary_text:
                    system_text += "\n\n" + summary_text
                elif context_hint:
                    system_text += "\n\nPrevious conversation summary (use only relevant parts):\n" + context_hint
                parts.append(
                    "<|start_header_id|>system<|end_header_id|>\n\n" + system_text + "<|eot_id|>"
                )
            else:
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "system":
                        extras = []
                        if summary_text:
                            extras.append(summary_text)
                        if context_hint and not history_msgs and not summary_text:
                            extras.append("Previous conversation summary (use only relevant parts):\n" + context_hint)
                        if extras:
                            m["content"] = str(m.get("content", "")) + "\n\n" + "\n\n".join(extras)
                        break

            combined = list(history_msgs)
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    # Already handled above; include it as the first system message.
                    parts.append("<|start_header_id|>system<|end_header_id|>\n\n" + m["content"] + "<|eot_id|>")
                    continue
                combined.append(m)

            for m in combined:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append("<|start_header_id|>system<|end_header_id|>\n\n" + content + "<|eot_id|>")
                elif role == "assistant":
                    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n" + content + "<|eot_id|>")
                else:
                    parts.append("<|start_header_id|>user<|end_header_id|>\n\n" + content + "<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        if msgs:
            prompt = build_chat_prompt(msgs)
        else:
            prompt = p.get("prompt", "") if isinstance(p, dict) else ""

        req_id = f"{t0:.6f}"
        _log(f"[{req_id}] PATH={self.path}")
        _log(f"[{req_id}] BODY={raw.decode('utf-8', errors='replace')[:2000]}")
        _log(f"[{req_id}] PROMPT={prompt[:2000]}")

        backend_payload = json.dumps({
            "prompt": prompt,
            "n_predict": n_pred,
            "stream": False,
        }).encode()

        try:
            req = UReq(
                BACKEND_URL + "/completion",
                data=backend_payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=300) as r:
                data = json.loads(r.read())
            reply = strip_format_artifacts(data.get("content", ""))
            
            # Support both Ollama format (/api/chat) and OpenAI format (/v1/chat/completions)
            if self.path == "/v1/chat/completions":
                body = json.dumps({
                    "id": "chatcmpl-" + str(hash(reply)),
                    "object": "chat.completion",
                    "created": int(__import__("time").time()),
                    "model": p.get("model", "llama-3.1-8b"),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": len(reply.split()), "total_tokens": len(reply.split())}
                }).encode()
            else:
                body = json.dumps({
                    "ok": True,
                    "message": {"role": "assistant", "content": reply},
                    "done": True,
                }).encode()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.send_response(200)
            _log(f"[{req_id}] STATUS=200 REPLY_LEN={len(reply)} ELAPSED_MS={elapsed_ms:.2f}")
            print(f"[proxy] POST {self.path} -> 200, reply_len={len(reply)}", flush=True)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            body = json.dumps({"ok": False, "error": str(e)}).encode()
            self.send_response(500)
            _log(f"[{req_id}] STATUS=500 ERROR={e} ELAPSED_MS={elapsed_ms:.2f}")
            print(f"[proxy] POST {self.path} -> 500, error={e}", flush=True)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-port", type=int, default=8083)
    parser.add_argument("--backend-port", type=int, default=8084)
    args = parser.parse_args()

    global BACKEND_URL
    BACKEND_URL = f"http://127.0.0.1:{args.backend_port}"

    server = ThreadingHTTPServer(("127.0.0.1", args.proxy_port), ProxyHandler)
    print(f"[proxy] listening on 127.0.0.1:{args.proxy_port} -> backend {BACKEND_URL}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

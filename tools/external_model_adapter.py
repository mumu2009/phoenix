import argparse
from dataclasses import dataclass
import json
import os
import re
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

CHAT_PATHS = {"/api/chat", "/v1/chat/completions"}
WORLD_CONTEXT_MARKERS = (
    "phoenix guidance shell",
    "world_",
    "gnn_stage2|",
    "capture path:",
    "micro-mipi",
    "v-jepa2",
    "vjepa2",
)

DEFAULT_CTX_SIZE = 10_000_000
DEFAULT_BATCH_SIZE = 512
DEFAULT_UBATCH_SIZE = 128
DEFAULT_ROPE_SCALING = "yarn"
DEFAULT_ROPE_FREQ_BASE = 0.0
DEFAULT_ROPE_FREQ_SCALE = 0.0
DEFAULT_YARN_ORIG_CTX = 4096
DEFAULT_YARN_EXT_FACTOR = 1.0
DEFAULT_YARN_ATTN_FACTOR = 1.0
DEFAULT_YARN_BETA_FAST = 32.0
DEFAULT_YARN_BETA_SLOW = 1.0


@dataclass(frozen=True)
class RuntimeLaunchOptions:
    ctx_size: int = DEFAULT_CTX_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    ubatch_size: int = DEFAULT_UBATCH_SIZE
    rope_scaling: str = DEFAULT_ROPE_SCALING
    rope_freq_base: float = DEFAULT_ROPE_FREQ_BASE
    rope_freq_scale: float = DEFAULT_ROPE_FREQ_SCALE
    yarn_orig_ctx: int = DEFAULT_YARN_ORIG_CTX
    yarn_ext_factor: float = DEFAULT_YARN_EXT_FACTOR
    yarn_attn_factor: float = DEFAULT_YARN_ATTN_FACTOR
    yarn_beta_fast: float = DEFAULT_YARN_BETA_FAST
    yarn_beta_slow: float = DEFAULT_YARN_BETA_SLOW
    lora_files: str = ""
    lora_init_without_apply: bool = False


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def format_float(value: float) -> str:
    return f"{value:g}"


def normalize_request_path(path: str) -> str:
    normalized = (path or "/").split("?", 1)[0].rstrip("/")
    return normalized or "/"


def normalize_message_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text = item["text"].strip()
                elif isinstance(item.get("content"), str):
                    text = item["content"].strip()
                else:
                    continue
            else:
                continue
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def compact_block(text: str) -> str:
    return "\n".join(line.strip() for line in text.replace("\r", "").splitlines() if line.strip())


def looks_like_world_context(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in WORLD_CONTEXT_MARKERS)


def build_prompt_from_messages(messages: list[object]) -> str:
    instruction_blocks: list[str] = []
    world_blocks: list[str] = []
    dialogue_blocks: list[tuple[str, str]] = []
    role_labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }

    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role", "user") or "user").strip().lower()
        content = compact_block(normalize_message_content(raw_message.get("content", "")))
        if not content:
            continue
        if role == "system":
            if looks_like_world_context(content):
                world_blocks.append(content)
            else:
                instruction_blocks.append(content)
            continue
        dialogue_blocks.append((role_labels.get(role, role.title() or "User"), content))

    sections: list[str] = []
    if instruction_blocks:
        sections.append("System instructions:\n" + "\n\n".join(instruction_blocks))
    if world_blocks:
        sections.append("World model context:\n" + "\n\n".join(world_blocks))
    for label, content in dialogue_blocks:
        sections.append(f"{label}:\n{content}")

    if not dialogue_blocks:
        sections.append("User:\n")
    if not dialogue_blocks or dialogue_blocks[-1][0] != "Assistant":
        sections.append("Assistant:")
    return "\n\n".join(section for section in sections if section)


def extract_prompt(payload: dict[str, object]) -> str:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        prompt = build_prompt_from_messages(messages)
        if prompt.strip():
            return prompt
    return compact_block(normalize_message_content(payload.get("prompt", "")))


def extract_max_tokens(payload: dict[str, object]) -> int:
    candidates: list[object] = [payload.get("max_tokens"), payload.get("maxTokens"), payload.get("num_predict")]
    options = payload.get("options")
    if isinstance(options, dict):
        candidates.append(options.get("num_predict"))
        candidates.append(options.get("max_tokens"))
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 64


def build_success_payload(path: str, model: str, reply: str) -> dict[str, object]:
    normalized_path = normalize_request_path(path)
    if normalized_path == "/v1/chat/completions":
        return {
            "id": "chatcmpl-phoenix-adapter",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    return {
        "message": {"role": "assistant", "content": reply},
        "model": model,
    }


def build_cli_command(
    provider: str,
    executable: Path,
    model: str,
    prompt: str,
    max_tokens: int,
    options: RuntimeLaunchOptions,
) -> list[str]:
    command = [
        str(executable),
        "-m",
        model,
        "-n",
        str(max(16, max_tokens)),
        "--ctx-size",
        str(max(4096, options.ctx_size)),
        "--batch-size",
        str(max(32, options.batch_size)),
        "--ubatch-size",
        str(max(1, options.ubatch_size)),
        "-t",
        str(max(1, (os.cpu_count() or 4) // 2)),
        "--temp",
        "0.7",
        "--simple-io",
        "--no-display-prompt",
        "--rope-scaling",
        options.rope_scaling or "none",
        "--rope-freq-base",
        format_float(max(0.0, options.rope_freq_base)),
        "--rope-freq-scale",
        format_float(max(0.0, options.rope_freq_scale)),
        "--yarn-orig-ctx",
        str(max(1, options.yarn_orig_ctx)),
        "--yarn-ext-factor",
        format_float(max(0.0, options.yarn_ext_factor)),
        "--yarn-attn-factor",
        format_float(max(0.0, options.yarn_attn_factor)),
        "--yarn-beta-fast",
        format_float(max(0.0, options.yarn_beta_fast)),
        "--yarn-beta-slow",
        format_float(max(0.0, options.yarn_beta_slow)),
    ]
    if provider in {"llamacpp", "bitnet"}:
        command.append("-cnv")
    
    # Add LoRA adapter arguments if configured
    if options.lora_files:
        for lora_file in options.lora_files.split(","):
            lora_file = lora_file.strip()
            if lora_file:
                command.extend(["--lora", lora_file])
        if options.lora_init_without_apply:
            command.append("--lora-init-without-apply")
    
    command.extend(["-p", prompt])
    return command


def find_executable(provider: str, engine_root: Path) -> Path | None:
    candidates: list[Path] = []
    if provider == "llamacpp":
        candidates.extend(
            [
                engine_root / "build-gcc" / "bin" / "llama-cli.exe",
                engine_root / "build-gcc" / "bin" / "llama-server.exe",
                engine_root / "build" / "bin" / "Release" / "llama-cli.exe",
                engine_root / "build" / "bin" / "llama-cli.exe",
                engine_root / "build" / "bin" / "llama-server.exe",
                engine_root / "build" / "bin" / "Release" / "llama-server.exe",
            ]
        )
    elif provider == "bitnet":
        candidates.extend(
            [
                engine_root / "build-gcc" / "bin" / "llama-cli.exe",
                engine_root / "build" / "bin" / "Release" / "llama-cli.exe",
                engine_root / "build" / "bin" / "llama-cli.exe",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def clean_reply(output: str, prompt: str) -> str:
    text = strip_ansi(output).replace("\r", "")
    if prompt and prompt in text:
        text = text.replace(prompt, "", 1)
    lines = [line.rstrip() for line in text.splitlines()]
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(("build:", "main:", "llama_", "system_info:", "sampling:", "generate:", "srv ")):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def run_cli(
    provider: str,
    executable: Path,
    model: str,
    prompt: str,
    max_tokens: int,
    options: RuntimeLaunchOptions,
) -> tuple[str, str]:
    command = build_cli_command(provider, executable, model, prompt, max_tokens, options)

    completed = subprocess.run(
        command,
        cwd=str(executable.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    combined = completed.stdout or ""
    if completed.stderr:
        combined = combined + ("\n" if combined else "") + completed.stderr
    reply = clean_reply(combined, prompt)
    if completed.returncode != 0 and not reply:
        return "", combined.strip() or f"{provider} cli exited with code {completed.returncode}"
    return reply, ""


class AdapterHandler(BaseHTTPRequestHandler):
    provider = ""
    executable: Path | None = None
    model = ""
    launch_options = RuntimeLaunchOptions()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/", "/ready"}:
            if not self.executable or not self.executable.exists():
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": "engine executable missing"})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "provider": self.provider, "model": self.model})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        normalized_path = normalize_request_path(self.path)
        if normalized_path not in CHAT_PATHS:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        if not self.executable or not self.executable.exists():
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": "engine executable missing"})
            return
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json"})
            return

        prompt = extract_prompt(payload if isinstance(payload, dict) else {})
        if not prompt:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing prompt"})
            return

        max_tokens = extract_max_tokens(payload if isinstance(payload, dict) else {})
        reply, error = run_cli(self.provider, self.executable, self.model, prompt, max_tokens, self.launch_options)
        if error and not reply:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": error})
            return

        self._send_json(HTTPStatus.OK, build_success_payload(normalized_path, self.model, reply))

    def log_message(self, format: str, *args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="External model HTTP adapter")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--brain-map", default="")
    parser.add_argument("--ctx-size", type=int, default=DEFAULT_CTX_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ubatch-size", type=int, default=DEFAULT_UBATCH_SIZE)
    parser.add_argument("--rope-scaling", default=DEFAULT_ROPE_SCALING)
    parser.add_argument("--rope-freq-base", type=float, default=DEFAULT_ROPE_FREQ_BASE)
    parser.add_argument("--rope-freq-scale", type=float, default=DEFAULT_ROPE_FREQ_SCALE)
    parser.add_argument("--yarn-orig-ctx", type=int, default=DEFAULT_YARN_ORIG_CTX)
    parser.add_argument("--yarn-ext-factor", type=float, default=DEFAULT_YARN_EXT_FACTOR)
    parser.add_argument("--yarn-attn-factor", type=float, default=DEFAULT_YARN_ATTN_FACTOR)
    parser.add_argument("--yarn-beta-fast", type=float, default=DEFAULT_YARN_BETA_FAST)
    parser.add_argument("--yarn-beta-slow", type=float, default=DEFAULT_YARN_BETA_SLOW)
    parser.add_argument("--lora-files", default="")
    parser.add_argument("--lora-init-without-apply", action="store_true")
    args = parser.parse_args()

    engine_root = Path(args.engine_root).resolve()
    executable = find_executable(args.provider, engine_root)
    if executable is None:
        print(f"[{args.provider}] engine executable missing under {engine_root}", file=sys.stderr)
        return 2

    AdapterHandler.provider = args.provider
    AdapterHandler.executable = executable
    AdapterHandler.model = args.model
    AdapterHandler.launch_options = RuntimeLaunchOptions(
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        rope_scaling=args.rope_scaling,
        rope_freq_base=args.rope_freq_base,
        rope_freq_scale=args.rope_freq_scale,
        yarn_orig_ctx=args.yarn_orig_ctx,
        yarn_ext_factor=args.yarn_ext_factor,
        yarn_attn_factor=args.yarn_attn_factor,
        yarn_beta_fast=args.yarn_beta_fast,
        yarn_beta_slow=args.yarn_beta_slow,
        lora_files=args.lora_files,
        lora_init_without_apply=args.lora_init_without_apply,
    )

    server = ThreadingHTTPServer((args.host, args.port), AdapterHandler)
    print(f"[{args.provider}] adapter listening on http://{args.host}:{args.port} using {executable}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
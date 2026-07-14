from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import random
import re
import statistics
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_PHOENIX_URL = "http://127.0.0.1:5081/api/chat"
DEFAULT_PHOENIX_TOKEN = "local-dev"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8083/v1/chat/completions"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_JSON_OUTPUT = "build/memory_tier_benchmark_v1.json"
DEFAULT_MD_OUTPUT = "build/memory_tier_benchmark_v1.md"
DEFAULT_CACHE_OUTPUT = "build/memory_tier_benchmark_v1_cache.json"
SCORING_VERSION = 2

WORD_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")


def _load_phoenix_json() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "config" / "phoenix.json"
    if not path.exists():
        raise RuntimeError(f"Missing single-source config: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _scenario_ranges() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    cfg = _load_phoenix_json()["scenarios"]
    s_max = int(cfg["short_dialogue_max_turns"])
    l_min = int(cfg["long_dialogue_min_turns"])
    l_max = int(cfg["long_dialogue_max_turns"])
    u_min = int(cfg["ultra_long_dialogue_min_turns"])
    u_max = int(cfg["ultra_long_dialogue_max_turns"])
    return (2, s_max), (l_min, l_max), (u_min, u_max)


def short_range() -> tuple[int, int]:
    return _scenario_ranges()[0]


def long_range() -> tuple[int, int]:
    return _scenario_ranges()[1]


def ultra_range() -> tuple[int, int]:
    return _scenario_ranges()[2]


@dataclass
class ProviderReply:
    ok: bool
    status: int
    latency_ms: float
    reply: str
    error: str


@dataclass
class ScenarioSample:
    scenario: str
    expected: str
    similarity: float
    success: bool
    latency_ms: float
    raw_reply: str = ""
    raw_error: str = ""
    http_status: int = 0


@dataclass
class ScenarioRunResult:
    samples: list[ScenarioSample]
    conversations_planned: int
    requests_sent: int
    interrupted: bool


@dataclass
class ScenarioRequestBudget:
    min_requests: int
    max_requests: int


def append_failed_sample(out: list[ScenarioSample], scenario: str, expected: str, latency_ms: float, reason: str, raw_reply: str = "", http_status: int = 0) -> None:
    print(f"[warn] {scenario} sample failed: {reason}", flush=True)
    out.append(ScenarioSample(scenario, expected, 0.0, False, latency_ms, raw_reply=raw_reply, raw_error=reason, http_status=http_status))


def result_with_counts(samples: list[ScenarioSample], conversations_planned: int, requests_sent: int, interrupted: bool = False) -> ScenarioRunResult:
    return ScenarioRunResult(
        samples=samples,
        conversations_planned=conversations_planned,
        requests_sent=requests_sent,
        interrupted=interrupted,
    )


def print_activity(provider: str, scenario: str, sample_index: int, sample_total: int, step: str, requests_sent: int) -> None:
    print(
        f"[activity] provider={provider} scenario={scenario} sample={sample_index}/{sample_total} step={step} benchmarkRequests={requests_sent}",
        flush=True,
    )


def print_progress(provider: str, scenario: str, sample_index: int, sample_total: int, requests_sent: int) -> None:
    print(
        f"[progress] {provider} {scenario} {sample_index}/{sample_total} benchmarkRequests={requests_sent}",
        flush=True,
    )


def short_dialogue_budget(sample_count: int) -> ScenarioRequestBudget:
    return ScenarioRequestBudget(sample_count, sample_count)


def memory_dialogue_budget(sample_count: int, min_turns: int, max_turns: int) -> ScenarioRequestBudget:
    return ScenarioRequestBudget(sample_count * min_turns, sample_count * max_turns)


def cross_session_budget(sample_count: int) -> ScenarioRequestBudget:
    requests = sample_count * 2
    return ScenarioRequestBudget(requests, requests)


def budget_to_dict(budget: ScenarioRequestBudget) -> dict[str, int]:
    return {
        "minRequests": budget.min_requests,
        "maxRequests": budget.max_requests,
    }


def total_budget(budgets: list[ScenarioRequestBudget]) -> ScenarioRequestBudget:
    return ScenarioRequestBudget(
        min_requests=sum(item.min_requests for item in budgets),
        max_requests=sum(item.max_requests for item in budgets),
    )


def cache_enabled_for_provider(provider: str) -> bool:
    return provider in {"llama_server", "ollama"}


def make_cache_key(provider: str, scenario: str, payload_parts: list[str]) -> str:
    joined = "\n".join([provider, scenario, f"score-v{SCORING_VERSION}"] + payload_parts)
    return hashlib.sha256(joined.encode("utf-8", errors="replace")).hexdigest()


def load_result_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": {}}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {"version": 1, "entries": {}}
    if not isinstance(parsed, dict):
        return {"version": 1, "entries": {}}
    entries = parsed.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"version": 1, "entries": entries}


def save_result_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def with_retries(fn, retries: int = 2, sleep_s: float = 0.6) -> ProviderReply:
    last: ProviderReply | None = None
    for attempt in range(retries + 1):
        out = fn()
        if out.status == 200 and out.ok:
            return out
        last = out
        transient = out.status in {0, 429, 500, 502, 503, 504} or "timed out" in out.error.lower()
        if transient and attempt < retries:
            time.sleep(sleep_s)
            continue
        return out
    return last if last is not None else ProviderReply(False, 0, 0.0, "", "unknown error")


def get_cached_sample(cache: dict[str, Any], key: str, expected: str) -> ScenarioSample | None:
    entries = cache.get("entries")
    if not isinstance(entries, dict):
        return None
    row = entries.get(key)
    if not isinstance(row, dict):
        return None
    if row.get("expected") != expected:
        return None
    similarity = row.get("similarity")
    success = row.get("success")
    latency_ms = row.get("latencyMs")
    if not isinstance(similarity, (int, float)):
        return None
    if not isinstance(success, bool):
        return None
    if not isinstance(latency_ms, (int, float)):
        return None
    if latency_ms < 0:
        return None
    # Only reuse request-level valid rows; transport / HTTP failures are never cached.
    if row.get("status") != 200:
        return None
    if not isinstance(row.get("reply"), str) or not row.get("reply").strip():
        return None
    return ScenarioSample(
        scenario=str(row.get("scenario", "")) or "",
        expected=expected,
        similarity=float(similarity),
        success=success,
        latency_ms=float(latency_ms),
    )


def put_cached_sample(cache: dict[str, Any], key: str, provider: str, sample: ScenarioSample, status: int, reply: str) -> None:
    if status != 200 or not reply.strip():
        return
    entries = cache.setdefault("entries", {})
    if not isinstance(entries, dict):
        return
    entries[key] = {
        "provider": provider,
        "scenario": sample.scenario,
        "scoreVersion": SCORING_VERSION,
        "expected": sample.expected,
        "similarity": sample.similarity,
        "success": sample.success,
        "latencyMs": sample.latency_ms,
        "status": status,
        "reply": reply,
        "cachedAt": now_str(),
    }


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = SPACE_RE.sub(" ", lowered)
    return lowered


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def token_f1(pred: str, ref: str) -> float:
    p = tokenize(pred)
    r = tokenize(ref)
    if not p or not r:
        return 0.0
    p_counts: dict[str, int] = {}
    r_counts: dict[str, int] = {}
    for token in p:
        p_counts[token] = p_counts.get(token, 0) + 1
    for token in r:
        r_counts[token] = r_counts.get(token, 0) + 1
    overlap = 0
    for token, count in p_counts.items():
        overlap += min(count, r_counts.get(token, 0))
    precision = overlap / max(1, len(p))
    recall = overlap / max(1, len(r))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def char_ngram_jaccard(pred: str, ref: str, n: int = 3) -> float:
    p = normalize_text(pred)
    r = normalize_text(ref)
    if len(p) < n or len(r) < n:
        return 0.0
    p_set = {p[i : i + n] for i in range(len(p) - n + 1)}
    r_set = {r[i : i + n] for i in range(len(r) - n + 1)}
    union = p_set | r_set
    if not union:
        return 0.0
    return len(p_set & r_set) / len(union)


def semantic_similarity(pred: str, ref: str) -> float:
    if not pred.strip() or not ref.strip():
        return 0.0
    f1 = token_f1(pred, ref)
    tri = char_ngram_jaccard(pred, ref)
    return 0.6 * f1 + 0.4 * tri


def semantic_similarity_short(pred: str, ref: str) -> float:
    base = semantic_similarity(pred, ref)
    pred_norm = normalize_text(pred)
    ref_norm = normalize_text(ref)
    if not pred_norm or not ref_norm:
        return base

    ref_tokens = tokenize(ref_norm)
    pred_tokens = tokenize(pred_norm)

    # Keyword-style answers (e.g. sports/business) are semantically correct
    # even when the model wraps them in a sentence.
    if ref_norm in pred_norm and len(ref_tokens) <= 4:
        return max(base, 0.90)

    if ref_tokens and len(ref_tokens) <= 4 and all(tok in pred_tokens for tok in ref_tokens):
        return max(base, 0.90)

    return base


def post_json(url: str, payload: dict[str, Any], timeout_s: float, headers: dict[str, str] | None = None) -> tuple[int, Any, str]:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Connection", "close")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            return int(resp.getcode()), parsed, raw
    except error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception as read_exc:
            raw = f"HTTPError {exc.code}: {exc.reason}; body-read-failed: {read_exc}"
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = None
        return int(exc.code), parsed, raw
    except Exception as exc:
        return 0, None, str(exc)


def build_context_hint(history: list[dict[str, str]]) -> str:
    """将 [{role, content}] 历史转换成 Phoenix contextHint 格式的字符串。"""
    if not history:
        return ""
    lines = ["[Conversation history:"]
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"  User: {content}")
        elif role == "assistant":
            lines.append(f"  Assistant: {content}")
    lines.append("]")
    return "\n".join(lines)


def call_phoenix_ingest(text: str, session_id: str, base_url: str, token: str, timeout_s: float) -> bool:
    """调用 /context/ingest 将事实写入 worldmodel 持久存储，供跨 session 检索。"""
    ingest_url = base_url.split("/api/")[0] + "/context/ingest"
    payload = {
        "sessionId": session_id,
        "text": text,
        "modality": "text",
    }
    try:
        status, parsed, _ = post_json(ingest_url, payload, timeout_s, {"Authorization": f"Bearer {token}"})
        return status == 200
    except Exception:
        return False


def call_phoenix_reset(session_id: str, base_url: str, token: str, timeout_s: float) -> bool:
    """调用 /context/reset 触发归档和跨 session 学习探测。"""
    reset_url = base_url.split("/api/")[0] + "/context/reset"
    payload = {
        "sessionId": session_id,
    }
    try:
        status, parsed, _ = post_json(reset_url, payload, timeout_s, {"Authorization": f"Bearer {token}"})
        return status == 200
    except Exception:
        return False


def call_phoenix(prompt: str, session_id: str, url: str, token: str, timeout_s: float, max_tokens: int = 64,
                 context_hint: str = "", context_mode: str = "") -> ProviderReply:
    t0 = time.perf_counter()
    payload = {
        "text": prompt,
        "sessionId": session_id,
        "maxTokens": int(max_tokens),
        "benchmarkSinglePass": True,
        "disableReasoningCritic": True,
        "disableExternalStyleTrainStep": True,
        "enableAddonToolContract": False,
        "enableGraphSelector": False,
        "disableLongQuerySafeMode": True,
    }
    if context_hint:
        payload["contextHint"] = context_hint
        payload["contextWeight"] = 0.95
        payload["contextMode"] = "short"
    if context_mode:
        payload["contextMode"] = context_mode

    # Only auto-register watcher if not externally managed
    if not _phoenix_watcher._externally_managed:
        _ensure_phoenix_watcher_registered(url)

    for _attempt in range(3):
        status, parsed, raw = post_json(url, payload, timeout_s, {"Authorization": f"Bearer {token}"})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        _conn_err = status == 0
        if _conn_err and _attempt < 2:
            # Only attempt restart if not externally managed
            if not _phoenix_watcher._externally_managed:
                print(f"[phoenix-watcher] call_phoenix attempt {_attempt+1} failed (status=0 raw={raw[:80] if raw else ''}), checking phoenix...", flush=True)
                restarted = _phoenix_watcher.restart_if_dead()
                if not restarted:
                    time.sleep(3)
            else:
                # In externally managed mode, just wait and retry
                time.sleep(3)
            continue

        reply = ""
        if isinstance(parsed, dict):
            result = parsed.get("result")
            if isinstance(result, dict) and isinstance(result.get("reply"), str):
                reply = result["reply"].strip()
            elif isinstance(parsed.get("reply"), str):
                reply = parsed["reply"].strip()

        ok = status == 200 and bool(reply)
        err = "" if ok else (raw[:300] if raw else f"status={status}")
        return ProviderReply(ok=ok, status=status, latency_ms=latency_ms, reply=reply, error=err)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return ProviderReply(ok=False, status=status, latency_ms=latency_ms, reply="", error=raw[:300] if raw else f"status={status}")


def call_llama(messages: list[dict[str, str]], url: str, model: str, timeout_s: float, max_tokens: int = 64) -> ProviderReply:
    def _once() -> ProviderReply:
        t0 = time.perf_counter()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.8,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        status, parsed, raw = post_json(url, payload, timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        reply = ""
        if isinstance(parsed, dict):
            choices = parsed.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                msg = choices[0].get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    reply = msg["content"].strip()

        ok = status == 200 and bool(reply)
        err = "" if ok else (raw[:300] if raw else f"status={status}")
        return ProviderReply(ok=ok, status=status, latency_ms=latency_ms, reply=reply, error=err)

    return with_retries(_once)


def extract_ollama_reply(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()
    return ""


def call_ollama(messages: list[dict[str, str]], url: str, model: str, timeout_s: float, context_window: int, max_tokens: int = 64) -> ProviderReply:
    def _once() -> ProviderReply:
        t0 = time.perf_counter()
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": context_window,
                "temperature": 0.0,
                "top_p": 0.8,
                "num_thread": 8,
            },
        }
        status, parsed, raw = post_json(url, payload, timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        reply = extract_ollama_reply(parsed)
        ok = status == 200 and bool(reply)
        err = "" if ok else (raw[:300] if raw else f"status={status}")
        return ProviderReply(ok=ok, status=status, latency_ms=latency_ms, reply=reply, error=err)

    return with_retries(_once)


def call_chat_provider(messages: list[dict[str, str]], provider: str, args: argparse.Namespace, timeout_s: float, max_tokens: int) -> ProviderReply:
    if provider == "llama_server":
        return call_llama(messages, args.llama_server_url, args.model, timeout_s, max_tokens)
    if provider == "ollama":
        return call_ollama(messages, args.ollama_url, args.ollama_model, timeout_s, args.context_window, max_tokens)
    raise RuntimeError(f"unsupported provider for chat call: {provider}")


def ensure_ascii_key(text: str, seed: int) -> str:
    chars = [c for c in text if c in string.ascii_letters + string.digits + " "]
    cleaned = "".join(chars).strip()
    words = cleaned.split()
    if not words:
        words = ["memory", "anchor", str(seed)]
    out = "_".join(words[:6]).lower()
    if len(out) < 8:
        out = f"anchor_{seed}_{out}"
    return out[:40]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


class PhoenixWatcher:
    """Monitors phoenix process and restarts it if it crashes."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._args: list[str] = []
        self._root: Path | None = None
        self._port: int = 5080
        self._log_path: Path | None = None
        self._externally_managed: bool = False  # When True, watcher won't auto-restart

    def register(self, proc: subprocess.Popen, args: list[str], root: Path, port: int, log_path: Path | None = None, externally_managed: bool = False) -> None:
        self._proc = proc
        self._args = args
        self._root = root
        self._port = port
        self._log_path = log_path
        self._externally_managed = externally_managed

    def reset(self) -> None:
        """Clear all registration state."""
        self._proc = None
        self._args = []
        self._root = None
        self._port = 5080
        self._log_path = None
        self._externally_managed = False

    def is_alive(self) -> bool:
        if self._proc is None:
            return True
        return self._proc.poll() is None

    def restart_if_dead(self) -> bool:
        # If externally managed (e.g., by --launch-local-stack), don't auto-restart
        if self._externally_managed:
            return False
        alive = self.is_alive()
        if not alive:
            pass
        elif not is_tcp_port_open("127.0.0.1", self._port, timeout_s=1.0):
            alive = False
        if alive:
            return False
        if not self._args or self._root is None:
            return False
        print("[phoenix-watcher] phoenix crashed — restarting...", flush=True)
        try:
            try:
                import glob as _glob
                for _pid_f in _glob.glob(str(self._root / "runtime_store" / "phoenix_main.pid")):
                    Path(_pid_f).unlink(missing_ok=True)
            except Exception:
                pass
            stdout_handle: Any = subprocess.DEVNULL
            if self._log_path is not None:
                stdout_handle = self._log_path.open("a", encoding="utf-8", errors="replace")
            self._proc = subprocess.Popen(
                self._args, cwd=str(self._root),
                stdout=stdout_handle, stderr=subprocess.STDOUT,
            )
            if not wait_tcp_port("127.0.0.1", self._port, 60):
                print("[phoenix-watcher] restarted phoenix did not come up in 60s", flush=True)
                return False
            time.sleep(3)
            from urllib.request import urlopen, Request as UReq
            body = json.dumps({"text": "OK", "sessionId": "watcher-warmup", "maxTokens": 4,
                               "benchmarkSinglePass": True}).encode()
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    req = UReq(f"http://127.0.0.1:{self._port}/api/chat", data=body, method="POST",
                               headers={"Content-Type": "application/json",
                                        "Authorization": "Bearer local-dev", "Connection": "close"})
                    with urlopen(req, timeout=30) as resp:
                        if resp.status == 200:
                            print("[phoenix-watcher] phoenix restarted and ready", flush=True)
                            return True
                except Exception:
                    time.sleep(2)
            print("[phoenix-watcher] restarted phoenix did not respond to warmup", flush=True)
        except Exception as exc:
            print(f"[phoenix-watcher] restart failed: {exc}", flush=True)
        return False


_phoenix_watcher = PhoenixWatcher()


def _ensure_phoenix_watcher_registered(url: str) -> None:
    """If watcher has no args yet (running as bench subprocess), auto-register using workspace.
    Skip auto-registration if watcher is already externally managed (e.g., by --launch-local-stack)."""
    if _phoenix_watcher._args or _phoenix_watcher._externally_managed:
        return
    try:
        import re as _re
        m = _re.search(r"127\.0\.0\.1:(\d+)", url)
        if not m:
            return
        port = int(m.group(1))
        root = workspace_root()
        llama_server_port = 8083
        phoenix_args = [
            str(root / "phoenix_main.exe"),
            f"--port={port}",
            "--study-port=5081",
            "--frontend-enabled=false",
            "--http-log=false",
            "--frontend-http-log=false",
            "--using-ollama=false",
            "--transformer-mode=llamacpp",
            f"--llamacpp-base-url=http://127.0.0.1:{llama_server_port}",
            "--external-auto-launch=false",
            "--bug-shooter=false",
            "--gguf-models-dir=GGUF_models",
            "--ai-count=1",
            "--disable-context-module=false",
            "--external-style-sim=false",
            "--http-log=true",
            "--robots-autoload=false",
            "--infer-workers=1",
            "--single-proc=true",
            "--disable-rl=true",
            "--disable-adv=true",
            "--disable-learning=true",
            "--disable-gnn-module=true",
        ]
        log_path = root / "build" / "phoenix_restart.log"
        _phoenix_watcher.register(None, phoenix_args, root, port, log_path)  # type: ignore[arg-type]
    except Exception:
        pass


def is_tcp_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        import socket

        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def wait_tcp_port(host: str, port: int, timeout_s: int = 60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_tcp_port_open(host, port, timeout_s=1.0):
            return True
        time.sleep(0.5)
    return False


def _warmup_llama_server(host: str, port: int, timeout_s: int = 300) -> bool:
    """Send one real completion request to fully load the model into memory."""
    from urllib.request import urlopen, Request as UReq
    payload = json.dumps({"prompt": "Hi", "n_predict": 4, "stream": False}).encode()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = UReq(f"http://{host}:{port}/completion", data=payload, method="POST",
                       headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=min(300, deadline - time.time())) as resp:
                resp.read()
            return True
        except Exception:
            time.sleep(2)
    return False


def start_local_phoenix_stack(
    root: Path,
    llama_server_port: int,
    phoenix_port: int,
    log_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    *,
    llama_threads: int = 8,
    llama_parallel: int = 1,
    llama_ctx_size: int = 4096,
) -> tuple[list[subprocess.Popen[str]], list[Any]]:
    processes: list[subprocess.Popen[str]] = []
    log_handles: list[Any] = []
    import os as _os
    _proc_env = {**_os.environ}
    if extra_env:
        _proc_env.update(extra_env)

    llama_server_exe = root / "outsides" / "llamacpp" / "build-gcc" / "bin" / "llama-server.exe"
    llama_model = root / "GGUF_models" / "blobs" / "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"

    # Resolve python executable — prefer project venv, fall back to sys.executable
    py_exe = root / ".venv" / "Scripts" / "python.exe"
    if not py_exe.exists():
        py_exe = Path(sys.executable)

    # llama-server.exe runs on llama_server_port+1 (direct /completion endpoint)
    # A lightweight proxy on llama_server_port translates phoenix's /api/chat -> /completion
    llama_backend_port = llama_server_port + 1

    already_warm = is_tcp_port_open("127.0.0.1", llama_backend_port, timeout_s=1.0)
    if not already_warm:
        llama_stdout: Any = subprocess.DEVNULL
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            llama_log = (log_dir / "llama_server.log").open("a", encoding="utf-8", errors="replace")
            log_handles.append(llama_log)
            llama_stdout = llama_log

        llama_server_args = [
            str(llama_server_exe),
            "-m", str(llama_model),
            "-c", str(llama_ctx_size),
            "-t", str(llama_threads),
            "-ngl", "0",
            "--port", str(llama_backend_port),
            "--host", "127.0.0.1",
        ]
        if llama_parallel > 1:
            llama_server_args.extend(["--parallel", str(llama_parallel)])
        processes.append(subprocess.Popen(llama_server_args, cwd=str(root), stdout=llama_stdout, stderr=subprocess.STDOUT, env=_proc_env))

        print(f"[stack] waiting for llama-server on port {llama_backend_port} ...", flush=True)
        if not wait_tcp_port("127.0.0.1", llama_backend_port, 120):
            raise RuntimeError(f"llama-server failed to listen on 127.0.0.1:{llama_backend_port}")

        # Warmup: send one real request so model is fully loaded before phoenix connects
        print("[stack] warming up llama-server (first inference may take a while) ...", flush=True)
        if not _warmup_llama_server("127.0.0.1", llama_backend_port, timeout_s=300):
            print("[stack] WARNING: llama-server warmup timed out; continuing anyway", flush=True)
        else:
            print("[stack] llama-server warm and ready", flush=True)

    if not is_tcp_port_open("127.0.0.1", llama_server_port, timeout_s=1.0):
        proxy_stdout: Any = subprocess.DEVNULL
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            proxy_log = (log_dir / "llama_proxy.log").open("a", encoding="utf-8", errors="replace")
            log_handles.append(proxy_log)
            proxy_stdout = proxy_log

        proxy_script = root / "tools" / "llama_proxy.py"
        proxy_args = [
            str(py_exe), str(proxy_script),
            "--proxy-port", str(llama_server_port),
            "--backend-port", str(llama_backend_port),
        ]
        processes.append(subprocess.Popen(proxy_args, cwd=str(root), stdout=proxy_stdout, stderr=subprocess.STDOUT, env=_proc_env))
        if not wait_tcp_port("127.0.0.1", llama_server_port, 10):
            raise RuntimeError(f"llama proxy failed to listen on 127.0.0.1:{llama_server_port}")

    phoenix_args = [
        str(root / "phoenix_main.exe"),
        f"--port={phoenix_port}",
        "--study-port=5081",
        "--frontend-enabled=false",
        "--http-log=false",
        "--frontend-http-log=false",
        "--using-ollama=false",
        "--transformer-mode=llamacpp",
        f"--llamacpp-base-url=http://127.0.0.1:{llama_server_port}",
        "--external-auto-launch=false",
        "--bug-shooter=false",
        "--gguf-models-dir=GGUF_models",
        "--ai-count=1",
        "--disable-context-module=false",
        "--external-style-sim=false",
        "--http-log=true",
        "--robots-autoload=false",
        "--infer-workers=1",
        "--single-proc=true",
        "--disable-rl=true",
        "--disable-adv=true",
        "--disable-learning=true",
        "--disable-gnn-module=true",
        "--disable-watchdog",
    ]
    if not is_tcp_port_open("127.0.0.1", phoenix_port, timeout_s=1.0):
        phoenix_stdout: Any = subprocess.DEVNULL
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            phoenix_log = (log_dir / "phoenix_main.log").open("a", encoding="utf-8", errors="replace")
            log_handles.append(phoenix_log)
            phoenix_stdout = phoenix_log
        _phoenix_proc = subprocess.Popen(phoenix_args, cwd=str(root), stdout=phoenix_stdout, stderr=subprocess.STDOUT, env=_proc_env)
        processes.append(_phoenix_proc)
        _phoenix_log_path = (log_dir / "phoenix_main.log") if log_dir is not None else None
        # Reset any auto-registered state from _ensure_phoenix_watcher_registered before registering externally-managed phoenix
        _phoenix_watcher.reset()
        _phoenix_watcher.register(_phoenix_proc, phoenix_args, root, phoenix_port, _phoenix_log_path, externally_managed=True)

        if not wait_tcp_port("127.0.0.1", phoenix_port, 60):
            raise RuntimeError(f"phoenix_main failed to listen on 127.0.0.1:{phoenix_port}")

        # Port is open but phoenix may still be initializing (robots warmup, snapshot restore).
        # Send real HTTP warmup requests until we get a 200 response.
        from urllib.request import urlopen, Request as UReq
        from urllib.error import URLError
        print(f"[stack] waiting for phoenix on port {phoenix_port} to finish initializing ...", flush=True)
        warmup_body = json.dumps({
            "text": "Reply with OK.",
            "sessionId": "stack-warmup-init",
            "maxTokens": 24,
            "benchmarkSinglePass": True,
            "disableReasoningCritic": True,
            "disableExternalStyleTrainStep": True,
            "enableAddonToolContract": False,
            "enableGraphSelector": False,
        }).encode()
        deadline = time.time() + 120
        phoenix_ready = False
        while time.time() < deadline:
            try:
                req = UReq(
                    f"http://127.0.0.1:{phoenix_port}/api/chat",
                    data=warmup_body,
                    method="POST",
                    headers={"Content-Type": "application/json", "Authorization": "Bearer local-dev", "Connection": "close"},
                )
                with urlopen(req, timeout=60) as resp:
                    if resp.status == 200:
                        phoenix_ready = True
                        break
            except Exception:
                time.sleep(2)
        if not phoenix_ready:
            print("[stack] WARNING: phoenix did not respond to warmup in time; continuing anyway", flush=True)
        else:
            print("[stack] phoenix fully initialized and ready", flush=True)

    return processes, log_handles


def _build_sweep_heatmap(sweep_results: list[dict[str, Any]], metric: str) -> str:
    """Render a Markdown table heatmap for a given metric across (concat_thresh x rnn_thresh) pairs."""
    if not sweep_results:
        return ""
    max_msg_values = sorted(set(r["max_messages"] for r in sweep_results))
    concat_values = sorted(set(r["concat_thresh"] for r in sweep_results))
    rnn_values = sorted(set(r["rnn_thresh"] for r in sweep_results))
    lines: list[str] = []
    for max_msg in max_msg_values:
        lines.append(f"\n#### maxMessages={max_msg}")
        header = "| concat_thresh \\ rnn_thresh | " + " | ".join(str(r) for r in rnn_values) + " |"
        sep = "|---" * (len(rnn_values) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        for ct in concat_values:
            row = f"| concat={ct} |"
            for rt in rnn_values:
                match = [r for r in sweep_results if r["max_messages"] == max_msg and r["concat_thresh"] == ct and r["rnn_thresh"] == rt]
                if match:
                    val = match[0].get(metric, float("nan"))
                    row += f" {val:.3f} |"
                else:
                    row += " - |"
            lines.append(row)
    return "\n".join(lines)


def run_threshold_sweep(
    root: Path,
    facts: list[str],
    qa_pairs: list[tuple[str, str]],
    args: argparse.Namespace,
    samples_per_combo: int = 5,
    concat_thresholds: list[int] | None = None,
    rnn_thresholds: list[int] | None = None,
    max_messages_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run a grid sweep of context mode thresholds and shortWindow sizes for Phoenix.
    For each combination, restarts the Phoenix stack with the given env vars, runs
    long_dialogue_5_15 and ultra_long_dialogue_15_plus, records success_rate /
    avg_semantic_sim / avg_latency_ms, and returns all results as a list of dicts.
    """
    if concat_thresholds is None:
        concat_thresholds = [3, 5, 8, 10]
    if rnn_thresholds is None:
        rnn_thresholds = [10, 15, 20, 25]
    if max_messages_values is None:
        max_messages_values = [5, 10, 20, 40]

    sweep_results: list[dict[str, Any]] = []
    stack_log_dir = (root / args.json_output).parent / "sweep_logs"

    for max_msg in max_messages_values:
        for ct in concat_thresholds:
            for rt in rnn_thresholds:
                if rt <= ct:
                    continue  # rnn threshold must be > concat threshold
                combo_label = f"maxMsg={max_msg} concat={ct} rnn={rt}"
                print(f"[sweep] testing {combo_label}", flush=True)

                extra_env = {
                    "FRONTEND_CONCAT_THRESH": str(ct),
                    "FRONTEND_RNN_THRESH": str(rt),
                    "FRONTEND_SHORT_WINDOW_MAX_MESSAGES": str(max_msg),
                }

                local_procs: list[Any] = []
                local_handles: list[Any] = []
                try:
                    local_procs, local_handles = start_local_phoenix_stack(
                        root, 8083, 5080, stack_log_dir, extra_env=extra_env
                    )
                    # Wait for phoenix to be ready
                    if not warmup_phoenix(args, skip_watcher=True):
                        print(f"[sweep] SKIP {combo_label}: phoenix warmup failed", flush=True)
                        continue

                    long_res = run_memory_dialogue(
                        samples_per_combo, facts, [q[0] for q in qa_pairs],
                        max(args.timeout, args.phoenix_timeout_floor),
                        "phoenix", args, args.seed + 100, "long_dialogue_5_15", *long_range(), {}
                    )
                    ultra_res = run_memory_dialogue(
                        samples_per_combo, facts, [q[0] for q in qa_pairs],
                        max(args.timeout, args.phoenix_timeout_floor),
                        "phoenix", args, args.seed + 101, "ultra_long_dialogue_15_plus", *ultra_range(), {}
                    )

                    def _summarize_samples(samples: list[ScenarioSample]) -> dict[str, float]:
                        if not samples:
                            return {"success_rate": 0.0, "avg_sim": 0.0, "avg_latency_ms": 0.0}
                        ok = [s for s in samples if s.success]
                        return {
                            "success_rate": len(ok) / len(samples),
                            "avg_sim": sum(s.similarity for s in samples) / len(samples),
                            "avg_latency_ms": sum(s.latency_ms for s in samples) / len(samples),
                        }

                    long_stats = _summarize_samples(long_res.samples)
                    ultra_stats = _summarize_samples(ultra_res.samples)
                    combo_result = {
                        "concat_thresh": ct,
                        "rnn_thresh": rt,
                        "max_messages": max_msg,
                        "long_success_rate": long_stats["success_rate"],
                        "long_avg_sim": long_stats["avg_sim"],
                        "long_avg_latency_ms": long_stats["avg_latency_ms"],
                        "ultra_success_rate": ultra_stats["success_rate"],
                        "ultra_avg_sim": ultra_stats["avg_sim"],
                        "ultra_avg_latency_ms": ultra_stats["avg_latency_ms"],
                        "combined_score": 0.5 * (long_stats["success_rate"] + ultra_stats["success_rate"]),
                    }
                    sweep_results.append(combo_result)
                    print(
                        f"[sweep] {combo_label}: long_sr={long_stats['success_rate']:.2f} "
                        f"ultra_sr={ultra_stats['success_rate']:.2f} "
                        f"long_lat={long_stats['avg_latency_ms']:.0f}ms "
                        f"ultra_lat={ultra_stats['avg_latency_ms']:.0f}ms",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[sweep] ERROR {combo_label}: {exc}", flush=True)
                finally:
                    for proc in local_procs:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    for h in local_handles:
                        try:
                            h.close()
                        except Exception:
                            pass
                    time.sleep(3)  # let ports free up before next launch

    return sweep_results


def build_sweep_markdown(sweep_results: list[dict[str, Any]]) -> str:
    """Render a full Markdown report for the threshold sweep."""
    lines: list[str] = []
    lines.append("# Phoenix Context Mode Threshold Sweep Report\n")
    if not sweep_results:
        lines.append("No results collected.")
        return "\n".join(lines)

    # Best combo by combined score
    best = max(sweep_results, key=lambda r: r["combined_score"])
    lines.append(f"**Best combo**: concat_thresh={best['concat_thresh']}, rnn_thresh={best['rnn_thresh']}, maxMessages={best['max_messages']}")
    lines.append(f"  - combined_score={best['combined_score']:.3f}, long_success={best['long_success_rate']:.2f}, ultra_success={best['ultra_success_rate']:.2f}")
    lines.append(f"  - long_latency={best['long_avg_latency_ms']:.0f}ms, ultra_latency={best['ultra_avg_latency_ms']:.0f}ms\n")

    # Full results table
    lines.append("## All Results\n")
    cols = ["concat_thresh", "rnn_thresh", "max_messages", "long_success_rate", "ultra_success_rate", "combined_score", "long_avg_latency_ms", "ultra_avg_latency_ms"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in sorted(sweep_results, key=lambda x: x["combined_score"], reverse=True):
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")

    lines.append("\n## Heatmaps (combined_score)")
    lines.append(_build_sweep_heatmap(sweep_results, "combined_score"))
    lines.append("\n## Heatmaps (long_avg_latency_ms)")
    lines.append(_build_sweep_heatmap(sweep_results, "long_avg_latency_ms"))
    return "\n".join(lines)


def load_gpt4all_pairs(path: Path, limit: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            if instruction and output:
                pairs.append((instruction, output))
            if len(pairs) >= limit:
                break
    return pairs


def load_wikitext_facts(path: Path, limit: int) -> list[str]:
    facts: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if len(line) < 60:
                continue
            if line.startswith("="):
                continue
            if "@-@" in line:
                line = line.replace("@-@", "-")
            facts.append(line)
            if len(facts) >= limit:
                break
    return facts


def pick_turn_count(rng: random.Random, minimum: int, maximum: int) -> int:
    return rng.randint(minimum, maximum)


def run_short_dialogue(sample_count: int, qa_pairs: list[tuple[str, str]], timeout_s: float, provider: str, args: argparse.Namespace, seed: int, result_cache: dict[str, Any]) -> ScenarioRunResult:
    rng = random.Random(seed)
    out: list[ScenarioSample] = []
    requests_sent = 0
    try:
        for i in range(sample_count):
            instruction, expected = qa_pairs[i % len(qa_pairs)]
            sid = f"memv1-short-{provider}-{i}"

            cache_key = ""
            if cache_enabled_for_provider(provider):
                cache_key = make_cache_key(provider, "short_dialogue", [instruction, expected, str(i), str(seed)])
                cached = get_cached_sample(result_cache, cache_key, expected)
                if cached is not None:
                    out.append(ScenarioSample("short_dialogue", expected, cached.similarity, cached.success, cached.latency_ms))
                    print_progress(provider, "short_dialogue", i + 1, sample_count, requests_sent)
                    continue

            if provider == "phoenix":
                requests_sent += 1
                print_activity(provider, "short_dialogue", i + 1, sample_count, "request", requests_sent)
                r = call_phoenix(instruction, sid, args.phoenix_url, args.phoenix_token, timeout_s, args.max_tokens_short, context_mode=args.phoenix_context_mode)
                if r.status == 0 or r.status >= 400:
                    append_failed_sample(out, "short_dialogue", expected, r.latency_ms, f"status={r.status} error={r.error}", raw_reply=r.reply, http_status=r.status)
                    print_progress(provider, "short_dialogue", i + 1, sample_count, requests_sent)
                    continue
                pred = r.reply
                latency = r.latency_ms
            else:
                messages = [{"role": "user", "content": instruction}]
                requests_sent += 1
                print_activity(provider, "short_dialogue", i + 1, sample_count, "request", requests_sent)
                r = call_chat_provider(messages, provider, args, timeout_s, args.max_tokens_short)
                pred = r.reply
                latency = r.latency_ms

            sim = semantic_similarity_short(pred, expected)
            sample = ScenarioSample("short_dialogue", expected, sim, sim >= 0.70, latency, raw_reply=r.reply, http_status=r.status)
            out.append(sample)
            if cache_key:
                put_cached_sample(result_cache, cache_key, provider, sample, r.status, r.reply)

            print_progress(provider, "short_dialogue", i + 1, sample_count, requests_sent)
    except KeyboardInterrupt:
        return result_with_counts(out, sample_count, requests_sent, interrupted=True)
    return result_with_counts(out, sample_count, requests_sent)


def run_memory_dialogue(sample_count: int, facts: list[str], fillers: list[str], timeout_s: float, provider: str, args: argparse.Namespace, seed: int, scenario: str, min_turns: int, max_turns: int, result_cache: dict[str, Any]) -> ScenarioRunResult:
    rng = random.Random(seed)
    out: list[ScenarioSample] = []
    requests_sent = 0

    try:
        for i in range(sample_count):
            fact = facts[i % len(facts)]
            key = ensure_ascii_key(fact, i)
            expected = key
            turn_count = pick_turn_count(rng, min_turns, max_turns)
            filler_count = max(0, turn_count - 2)

            store_prompt = (
                "Remember the following token for this conversation. "
                "Reply exactly with ACK. "
                f"MEMTAG: {key}"
            )
            recall_prompt = "Ignore any previous replies and context markers. Extract the MEMTAG value from the message and reply with the exact value only."

            cache_key = ""
            if cache_enabled_for_provider(provider):
                cache_parts = [store_prompt, recall_prompt, str(filler_count), str(i), str(seed)]
                for j in range(filler_count):
                    filler = fillers[(i + j) % len(fillers)]
                    if args.minimal_filler:
                        cache_parts.append(f"Filler turn {j + 1}. Reply exactly with OK.")
                    else:
                        cache_parts.append(f"Filler turn {j + 1}. Briefly summarize this in one sentence: {filler}")
                cache_key = make_cache_key(provider, scenario, cache_parts)
                cached = get_cached_sample(result_cache, cache_key, expected)
                if cached is not None:
                    out.append(ScenarioSample(scenario, expected, cached.similarity, cached.success, cached.latency_ms))
                    print_progress(provider, scenario, i + 1, sample_count, requests_sent)
                    continue

            if provider == "phoenix":
                sid = f"memv1-{scenario}-phoenix-{i}"
                # Phoenix's gateway does not keep multi-turn history by sessionId alone.
                # Build a contextHint from the conversation history so the model can see
                # the MEMTAG value during recall.
                history: list[dict[str, str]] = []
                requests_sent += 1
                print_activity(provider, scenario, i + 1, sample_count, "store", requests_sent)
                store_reply = call_phoenix(store_prompt, sid, args.phoenix_url, args.phoenix_token, timeout_s, args.max_tokens_store, context_mode=args.phoenix_context_mode)
                if store_reply.status == 0 or store_reply.status >= 400:
                    append_failed_sample(out, scenario, expected, store_reply.latency_ms, f"store status={store_reply.status} error={store_reply.error}", raw_reply=store_reply.reply, http_status=store_reply.status)
                    print_progress(provider, scenario, i + 1, sample_count, requests_sent)
                    continue
                history.append({"role": "user", "content": store_prompt})
                history.append({"role": "assistant", "content": store_reply.reply})
                for j in range(filler_count):
                    filler = fillers[(i + j) % len(fillers)]
                    if args.minimal_filler:
                        filler_prompt = f"Filler turn {j + 1}. Reply exactly with OK."
                    else:
                        filler_prompt = f"Filler turn {j + 1}. Briefly summarize this in one sentence: {filler}"
                    context_hint = build_context_hint(history)
                    requests_sent += 1
                    print_activity(provider, scenario, i + 1, sample_count, f"filler_{j + 1}_of_{filler_count}", requests_sent)
                    filler_reply = call_phoenix(filler_prompt, sid, args.phoenix_url, args.phoenix_token, timeout_s, args.max_tokens_filler, context_hint=context_hint, context_mode=args.phoenix_context_mode)
                    if filler_reply.status == 0 or filler_reply.status >= 400:
                        append_failed_sample(out, scenario, expected, filler_reply.latency_ms, f"filler status={filler_reply.status} error={filler_reply.error}", raw_reply=filler_reply.reply, http_status=filler_reply.status)
                        print_progress(provider, scenario, i + 1, sample_count, requests_sent)
                        break
                    history.append({"role": "user", "content": filler_prompt})
                    history.append({"role": "assistant", "content": filler_reply.reply})
                else:
                    context_hint = build_context_hint(history)
                    requests_sent += 1
                    print_activity(provider, scenario, i + 1, sample_count, "recall", requests_sent)
                    recall_text = recall_prompt
                    final = call_phoenix(recall_text, sid, args.phoenix_url, args.phoenix_token, timeout_s, args.max_tokens_recall, context_hint=context_hint, context_mode=args.phoenix_context_mode)
                    if final.status == 0 or final.status >= 400:
                        append_failed_sample(out, scenario, expected, final.latency_ms, f"recall status={final.status} error={final.error}", raw_reply=final.reply, http_status=final.status)
                        print_progress(provider, scenario, i + 1, sample_count, requests_sent)
                        continue
                    pred = final.reply
                    latency = final.latency_ms
                    sim = semantic_similarity(pred, expected)
                    out.append(ScenarioSample(scenario, expected, sim, sim >= 0.70, latency, raw_reply=pred, http_status=final.status))
                    print_progress(provider, scenario, i + 1, sample_count, requests_sent)
            else:
                messages: list[dict[str, str]] = []
                messages.append({"role": "user", "content": store_prompt})
                requests_sent += 1
                print_activity(provider, scenario, i + 1, sample_count, "store", requests_sent)
                store_reply = call_chat_provider(messages, provider, args, timeout_s, args.max_tokens_store)
                if store_reply.status == 0 or store_reply.status >= 400:
                    append_failed_sample(out, scenario, expected, store_reply.latency_ms, f"store status={store_reply.status} error={store_reply.error}", raw_reply=store_reply.reply, http_status=store_reply.status)
                    print_progress(provider, scenario, i + 1, sample_count, requests_sent)
                    continue
                messages.append({"role": "assistant", "content": store_reply.reply})
                for j in range(filler_count):
                    filler = fillers[(i + j) % len(fillers)]
                    if args.minimal_filler:
                        filler_prompt = f"Filler turn {j + 1}. Reply exactly with OK."
                    else:
                        filler_prompt = f"Filler turn {j + 1}. Briefly summarize this in one sentence: {filler}"
                    messages.append({"role": "user", "content": filler_prompt})
                    requests_sent += 1
                    print_activity(provider, scenario, i + 1, sample_count, f"filler_{j + 1}_of_{filler_count}", requests_sent)
                    filler_reply = call_chat_provider(messages, provider, args, timeout_s, args.max_tokens_filler)
                    if filler_reply.status == 0 or filler_reply.status >= 400:
                        append_failed_sample(out, scenario, expected, filler_reply.latency_ms, f"filler status={filler_reply.status} error={filler_reply.error}", raw_reply=filler_reply.reply, http_status=filler_reply.status)
                        print_progress(provider, scenario, i + 1, sample_count, requests_sent)
                        break
                    messages.append({"role": "assistant", "content": filler_reply.reply})
                else:
                    messages.append({"role": "user", "content": recall_prompt})
                    requests_sent += 1
                    print_activity(provider, scenario, i + 1, sample_count, "recall", requests_sent)
                    final = call_chat_provider(messages, provider, args, timeout_s, args.max_tokens_recall)
                    if final.status == 0 or final.status >= 400:
                        append_failed_sample(out, scenario, expected, final.latency_ms, f"recall status={final.status} error={final.error}", raw_reply=final.reply, http_status=final.status)
                        print_progress(provider, scenario, i + 1, sample_count, requests_sent)
                        continue
                    pred = final.reply
                    latency = final.latency_ms
                    sim = semantic_similarity(pred, expected)
                    sample = ScenarioSample(scenario, expected, sim, sim >= 0.70, latency, raw_reply=pred, http_status=final.status)
                    out.append(sample)
                    if cache_key:
                        put_cached_sample(result_cache, cache_key, provider, sample, final.status, final.reply)
                    print_progress(provider, scenario, i + 1, sample_count, requests_sent)
    except KeyboardInterrupt:
        return result_with_counts(out, sample_count, requests_sent, interrupted=True)

    return result_with_counts(out, sample_count, requests_sent)


def run_cross_session(sample_count: int, facts: list[str], timeout_s: float, provider: str, args: argparse.Namespace, seed: int, result_cache: dict[str, Any]) -> ScenarioRunResult:
    rng = random.Random(seed)
    out: list[ScenarioSample] = []
    requests_sent = 0

    try:
        for i in range(sample_count):
            fact = facts[i % len(facts)]
            key = ensure_ascii_key(fact, i + 10000)
            expected = key
            profile = f"profile_{rng.randint(1000, 9999)}"

            store_prompt = (
                "Remember this label and its value for future reference. "
                "Reply ACK only. "
                f"LABEL: {profile}; TOKEN: {key}"
            )
            recall_prompt = (
                f"What TOKEN value was stored under LABEL {profile}? "
                "Reply with the exact value only."
            )

            cache_key = ""
            if cache_enabled_for_provider(provider):
                cache_key = make_cache_key(provider, "cross_session", [store_prompt, recall_prompt, str(i), str(seed)])
                cached = get_cached_sample(result_cache, cache_key, expected)
                if cached is not None:
                    out.append(ScenarioSample("cross_session", expected, cached.similarity, cached.success, cached.latency_ms))
                    print_progress(provider, "cross_session", i + 1, sample_count, requests_sent)
                    continue

            if provider == "phoenix":
                sid_a = f"memv1-cross-a-{i}"
                sid_b = f"memv1-cross-b-{i}"
                requests_sent += 1
                print_activity(provider, "cross_session", i + 1, sample_count, "store", requests_sent)
                r_store = call_phoenix(store_prompt, sid_a, args.phoenix_url, args.phoenix_token, timeout_s, args.max_tokens_store, context_mode=args.phoenix_context_mode)
                if r_store.status == 0 or r_store.status >= 400:
                    append_failed_sample(out, "cross_session", expected, r_store.latency_ms, f"store status={r_store.status} error={r_store.error}", raw_reply=r_store.reply, http_status=r_store.status)
                    print_progress(provider, "cross_session", i + 1, sample_count, requests_sent)
                    continue
                # 跨 session 持久化：把 TOKEN 事实写入 worldmodel，供 session B 检索
                ingest_text = f"LABEL: {profile}; TOKEN: {key}"
                call_phoenix_ingest(ingest_text, sid_a, args.phoenix_url, args.phoenix_token, timeout_s)
                # 调用 /context/reset 触发归档和跨 session 学习探测
                call_phoenix_reset(sid_a, args.phoenix_url, args.phoenix_token, timeout_s)
                # 等待后台学习完成（简化处理，实际应轮询 learnInFlight）
                time.sleep(2.0)
                requests_sent += 1
                print_activity(provider, "cross_session", i + 1, sample_count, "recall", requests_sent)
                # Use the dedicated recall prompt only. The Phoenix context module (via /context/hint)
                # should retrieve the cross-session fact from episodic memory and inject it as contextHint.
                recall_text = recall_prompt
                final = call_phoenix(recall_text, sid_b, args.phoenix_url, args.phoenix_token, timeout_s, args.max_tokens_recall, context_mode=args.phoenix_context_mode)
                if final.status == 0 or final.status >= 400:
                    append_failed_sample(out, "cross_session", expected, final.latency_ms, f"recall status={final.status} error={final.error}", raw_reply=final.reply, http_status=final.status)
                    print_progress(provider, "cross_session", i + 1, sample_count, requests_sent)
                    continue
                pred = final.reply
                latency = final.latency_ms
            else:
                messages_a = [{"role": "user", "content": store_prompt}]
                requests_sent += 1
                print_activity(provider, "cross_session", i + 1, sample_count, "store", requests_sent)
                store_reply = call_chat_provider(messages_a, provider, args, timeout_s, args.max_tokens_store)
                if store_reply.status == 0 or store_reply.status >= 400:
                    append_failed_sample(out, "cross_session", expected, store_reply.latency_ms, f"store status={store_reply.status} error={store_reply.error}", raw_reply=store_reply.reply, http_status=store_reply.status)
                    print_progress(provider, "cross_session", i + 1, sample_count, requests_sent)
                    continue
                messages_b = [{"role": "user", "content": recall_prompt}]
                requests_sent += 1
                print_activity(provider, "cross_session", i + 1, sample_count, "recall", requests_sent)
                final = call_chat_provider(messages_b, provider, args, timeout_s, args.max_tokens_recall)
                if final.status == 0 or final.status >= 400:
                    append_failed_sample(out, "cross_session", expected, final.latency_ms, f"recall status={final.status} error={final.error}", raw_reply=final.reply, http_status=final.status)
                    print_progress(provider, "cross_session", i + 1, sample_count, requests_sent)
                    continue
                pred = final.reply
                latency = final.latency_ms

            sim = semantic_similarity(pred, expected)
            sample = ScenarioSample("cross_session", expected, sim, sim >= 0.70, latency, raw_reply=pred, http_status=final.status)
            out.append(sample)
            if cache_key:
                put_cached_sample(result_cache, cache_key, provider, sample, final.status, final.reply)

            print_progress(provider, "cross_session", i + 1, sample_count, requests_sent)
    except KeyboardInterrupt:
        return result_with_counts(out, sample_count, requests_sent, interrupted=True)

    return result_with_counts(out, sample_count, requests_sent)


def summarize(samples: list[ScenarioSample], run_result: ScenarioRunResult, request_budget: ScenarioRequestBudget) -> dict[str, Any]:
    latencies = [s.latency_ms for s in samples]
    sims = [s.similarity for s in samples]
    success = sum(1 for s in samples if s.success)
    total = len(samples)
    raw_samples = [
        {
            "index": idx,
            "scenario": s.scenario,
            "expected": s.expected,
            "rawReply": s.raw_reply,
            "httpStatus": s.http_status,
            "similarity": round(s.similarity, 4),
            "success": s.success,
            "latencyMs": round(s.latency_ms, 2),
            "error": s.raw_error if s.raw_error else None,
        }
        for idx, s in enumerate(samples)
    ]
    return {
        "samples": total,
        "conversationsPlanned": run_result.conversations_planned,
        "conversationsCompleted": total,
        "requestsSent": run_result.requests_sent,
        "requestBudget": budget_to_dict(request_budget),
        "interrupted": run_result.interrupted,
        "latencyMs": {
            "avg": round(sum(latencies) / max(1, total), 2),
            "median": round(statistics.median(latencies) if latencies else 0.0, 2),
            "p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else round(max(latencies) if latencies else 0.0, 2),
        },
        "successRateSemanticGe70": round((success * 100.0) / max(1, total), 2),
        "avgSemanticSimilarity": round(sum(sims) / max(1, total), 4),
        "rawSamples": raw_samples,
    }


SCENARIO_LABELS: dict[str, str] = {
    "short_dialogue": "Short",
    "long_dialogue_5_15": "Long(5-15)",
    "ultra_long_dialogue_15_plus": "Ultra(15+)",
    "cross_session": "CrossSession",
}

PROVIDER_BAR_CHARS: dict[str, str] = {
    "phoenix": "#",
    "llama_server": "*",
    "ollama": "o",
}


def _bar(value: float, width: int = 20, char: str = "#") -> str:
    filled = max(0, min(width, round(value / 100.0 * width)))
    return char * filled + "-" * (width - filled)


def build_markdown(report: dict[str, Any]) -> str:
    providers = list(report["providers"].keys())
    meta = report.get("metadata", {})
    interrupted_flag = meta.get("interrupted", False)
    scenarios_in_report: list[str] = []
    for pdata in report["providers"].values():
        for s in pdata.get("scenarios", {}):
            if s not in scenarios_in_report:
                scenarios_in_report.append(s)

    lines: list[str] = [
        "# Memory Tier Benchmark v1 — Report",
        "",
        f"**Generated at:** {report['generatedAt']}  ",
        f"**Sources:** {', '.join(meta.get('sources', []))}  ",
        f"**Semantic threshold:** >= 0.70  ",
        f"**Providers tested:** {', '.join(providers)}  ",
    ]
    if interrupted_flag:
        lines.append("\n> **WARNING: benchmark was interrupted — results may be partial.**\n")
    lines.append("")

    # ── 1. Success-rate comparison (Mermaid bar chart) ──────────────────────
    lines.append("## Success Rate by Scenario (%)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "Success Rate (semantic >= 0.70) %"')
    x_labels = [SCENARIO_LABELS.get(s, s) for s in scenarios_in_report]
    lines.append(f'    x-axis {json.dumps(x_labels)}')
    lines.append('    y-axis "Success Rate (%)" 0 --> 100')
    for provider in providers:
        pdata = report["providers"][provider]
        values = [
            pdata["scenarios"].get(s, {}).get("successRateSemanticGe70", 0.0)
            for s in scenarios_in_report
        ]
        values_str = "[" + ", ".join(str(round(v, 1)) for v in values) + "]"
        lines.append(f'    bar {values_str}')
    lines.append("```")
    lines.append("")

    # ── 2. Latency comparison (Mermaid bar chart) ────────────────────────────
    lines.append("## Average Latency by Scenario (ms)")
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "Average Latency (ms)"')
    lines.append(f'    x-axis {json.dumps(x_labels)}')
    lines.append('    y-axis "Latency (ms)"')
    for provider in providers:
        pdata = report["providers"][provider]
        values = [
            pdata["scenarios"].get(s, {}).get("latencyMs", {}).get("avg", 0.0)
            for s in scenarios_in_report
        ]
        values_str = "[" + ", ".join(str(round(v, 1)) for v in values) + "]"
        lines.append(f'    bar {values_str}')
    lines.append("```")
    lines.append("")

    # ── 3. Avg semantic similarity (Mermaid bar chart) ───────────────────────
    lines.append("## Average Semantic Similarity by Scenario")
    lines.append("")
    lines.append("```mermaid")
    lines.append("xychart-beta")
    lines.append('    title "Avg Semantic Similarity (0-1)"')
    lines.append(f'    x-axis {json.dumps(x_labels)}')
    lines.append('    y-axis "Similarity" 0 --> 1')
    for provider in providers:
        pdata = report["providers"][provider]
        values = [
            pdata["scenarios"].get(s, {}).get("avgSemanticSimilarity", 0.0)
            for s in scenarios_in_report
        ]
        values_str = "[" + ", ".join(str(round(v, 4)) for v in values) + "]"
        lines.append(f'    bar {values_str}')
    lines.append("```")
    lines.append("")

    # ── 4. Provider overview table ───────────────────────────────────────────
    lines.append("## Provider Summary Table")
    lines.append("")
    header_scenarios = [SCENARIO_LABELS.get(s, s) for s in scenarios_in_report]
    col_w = max(14, max((len(h) for h in header_scenarios), default=14))
    header_parts = ["Provider".ljust(16)] + [h.ljust(col_w) for h in header_scenarios]
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(["-" * (len(p) + 2) for p in header_parts]) + "|")
    for provider in providers:
        pdata = report["providers"][provider]
        row_parts = [provider.ljust(16)]
        for s in scenarios_in_report:
            sdata = pdata["scenarios"].get(s)
            if sdata:
                sr = sdata.get("successRateSemanticGe70", 0.0)
                lat = sdata.get("latencyMs", {}).get("avg", 0.0)
                cell = f"{sr:.1f}% / {lat:.0f}ms"
            else:
                cell = "N/A"
            row_parts.append(cell.ljust(col_w))
        lines.append("| " + " | ".join(row_parts) + " |")
    lines.append("")
    lines.append("_Format: `success% / avg-latency-ms`_")
    lines.append("")

    # ── 5. Detailed per-provider sections ───────────────────────────────────
    lines.append("## Detailed Results")
    lines.append("")
    for provider in providers:
        pdata = report["providers"][provider]
        lines.append(f"### Provider: `{provider}`")
        lines.append("")
        bar_char = PROVIDER_BAR_CHARS.get(provider, "#")
        for scenario in scenarios_in_report:
            sdata = pdata["scenarios"].get(scenario)
            if not sdata:
                continue
            sr = sdata.get("successRateSemanticGe70", 0.0)
            avg_lat = sdata["latencyMs"]["avg"]
            med_lat = sdata["latencyMs"]["median"]
            p95_lat = sdata["latencyMs"]["p95"]
            avg_sim = sdata.get("avgSemanticSimilarity", 0.0)
            n_samples = sdata.get("samples", 0)
            n_req = sdata.get("requestsSent", 0)
            label = SCENARIO_LABELS.get(scenario, scenario)
            bar_vis = _bar(sr, 20, bar_char)
            lines.append(f"#### {label}")
            lines.append("")
            lines.append(f"```")
            lines.append(f"Success rate : [{bar_vis}] {sr:.1f}%")
            lines.append(f"Latency      : avg={avg_lat:.1f}ms  median={med_lat:.1f}ms  p95={p95_lat:.1f}ms")
            lines.append(f"Similarity   : avg={avg_sim:.4f}")
            lines.append(f"Samples      : {n_samples}  requests_sent={n_req}")
            if sdata.get("interrupted"):
                lines.append("WARNING      : run was interrupted, partial data only")
            lines.append("```")
            lines.append("")

            # failure breakdown (first 10 failed samples)
            raw = sdata.get("rawSamples", [])
            failed = [r for r in raw if not r.get("success")]
            if failed:
                lines.append(f"<details><summary>Failed samples ({len(failed)} / {n_samples})</summary>")
                lines.append("")
                lines.append("| # | Expected | Reply | Sim | Latency | Error |")
                lines.append("|---|----------|-------|-----|---------|-------|")
                for row in failed[:20]:
                    exp = str(row.get("expected", ""))[:30].replace("|", "\\|")
                    rep = str(row.get("rawReply", "") or "")[:40].replace("|", "\\|") or "-"
                    err = str(row.get("error") or "")[:60].replace("|", "\\|") or "-"
                    sim_v = row.get("similarity", 0.0)
                    lat_v = row.get("latencyMs", 0.0)
                    lines.append(f"| {row.get('index','')} | {exp} | {rep} | {sim_v:.3f} | {lat_v:.0f}ms | {err} |")
                if len(failed) > 20:
                    lines.append(f"\n_... and {len(failed) - 20} more (see raw JSON for full list)_")
                lines.append("")
                lines.append("</details>")
                lines.append("")

    # ── 6. Cross-provider latency overview (Mermaid quadrant) ────────────────
    if len(providers) >= 2 and len(scenarios_in_report) >= 2:
        lines.append("## Latency vs Success Rate Overview")
        lines.append("")
        lines.append("```mermaid")
        lines.append("quadrantChart")
        lines.append('    title "Latency (ms) vs Success Rate (%)"')
        lines.append('    x-axis "Low Success" --> "High Success"')
        lines.append('    y-axis "Fast" --> "Slow"')
        lines.append('    quadrant-1 "High-success, Slow"')
        lines.append('    quadrant-2 "High-success, Fast"')
        lines.append('    quadrant-3 "Low-success, Fast"')
        lines.append('    quadrant-4 "Low-success, Slow"')
        all_lats: list[float] = []
        all_srs: list[float] = []
        for pdata in report["providers"].values():
            for sdata in pdata["scenarios"].values():
                if sdata.get("skipped"):
                    continue
                all_lats.append(sdata.get("latencyMs", {}).get("avg", 0.0))
                all_srs.append(sdata.get("successRateSemanticGe70", 0.0))
        max_lat = max(all_lats) if all_lats else 1.0
        for provider in providers:
            pdata = report["providers"][provider]
            for scenario, sdata in pdata["scenarios"].items():
                if sdata.get("skipped"):
                    continue
                sr = sdata.get("successRateSemanticGe70", 0.0)
                lat = sdata.get("latencyMs", {}).get("avg", 0.0)
                x = round(sr / 100.0, 3)
                y = round(lat / max(max_lat, 1.0), 3)
                label = f"{provider[:6]}-{SCENARIO_LABELS.get(scenario, scenario)[:6]}"
                lines.append(f'    {label}: [{x}, {y}]')
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append(f"_Raw data: see accompanying `.json` file_")
    lines.append("")
    return "\n".join(lines)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _save_checkpoint(report: dict, json_out: Path, md_out: Path) -> None:
    """Write partial results to disk after each scenario completes."""
    try:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_out.write_text(build_markdown(report), encoding="utf-8")
        print(f"[checkpoint] saved partial results to {json_out.name}", flush=True)
    except Exception as exc:
        print(f"[checkpoint] failed to save: {exc}", flush=True)


def warmup_phoenix(args: argparse.Namespace, skip_watcher: bool = False) -> bool:
    """Return True if phoenix is ready, False otherwise (never raises).
    If skip_watcher is True, don't use the phoenix-watcher mechanism (for --launch-local-stack)."""
    for attempt in range(1, args.warmup_retries + 1):
        try:
            if skip_watcher:
                # Direct HTTP request without using call_phoenix (which triggers watcher)
                from urllib.request import urlopen, Request as UReq
                body = json.dumps({
                    "text": "Reply exactly with OK.",
                    "sessionId": f"memv1-warmup-{attempt}",
                    "maxTokens": args.max_tokens_filler,
                    "benchmarkSinglePass": True,
                    "disableReasoningCritic": True,
                    "disableExternalStyleTrainStep": True,
                    "enableAddonToolContract": False,
                    "enableGraphSelector": False,
                    "disableLongQuerySafeMode": True,
                }).encode()
                req = UReq(args.phoenix_url, data=body, method="POST",
                           headers={"Content-Type": "application/json",
                                    "Authorization": f"Bearer {args.phoenix_token}",
                                    "Connection": "close"})
                with urlopen(req, timeout=args.warmup_timeout) as resp:
                    if resp.status == 200:
                        print(f"[warmup] phoenix ready on attempt {attempt} (direct)", flush=True)
                        return True
                    print(f"[warmup] phoenix attempt {attempt}/{args.warmup_retries} failed: status={resp.status}", flush=True)
            else:
                reply = call_phoenix(
                    "Reply exactly with OK.",
                    f"memv1-warmup-{attempt}",
                    args.phoenix_url,
                    args.phoenix_token,
                    args.warmup_timeout,
                    args.max_tokens_filler,
                )
                if reply.status == 200:
                    print(f"[warmup] phoenix ready on attempt {attempt}", flush=True)
                    return True
                print(
                    f"[warmup] phoenix attempt {attempt}/{args.warmup_retries} failed: status={reply.status} error={reply.error}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[warmup] phoenix attempt {attempt}/{args.warmup_retries} exception: {exc}", flush=True)
            if attempt < args.warmup_retries:
                time.sleep(3)
            continue
        if attempt < args.warmup_retries:
            time.sleep(3)
    print("[WARN] phoenix warmup failed after all retries — phoenix scenarios will be skipped", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Memory tier benchmark for short/long/ultra/cross-session dialogues")
    parser.add_argument("--sample-per-scenario", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--warmup-timeout", type=float, default=120.0)
    parser.add_argument("--warmup-retries", type=int, default=3)
    parser.add_argument("--phoenix-timeout-floor", type=float, default=120.0)
    parser.add_argument("--minimal-filler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--launch-local-stack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens-short", type=int, default=48)
    parser.add_argument("--max-tokens-store", type=int, default=8)
    parser.add_argument("--max-tokens-filler", type=int, default=4)
    parser.add_argument("--max-tokens-recall", type=int, default=24)
    parser.add_argument("--phoenix-context-mode", default="", help="force phoenix contextMode: concat, rnn, lstm, or auto")
    parser.add_argument("--sweep-thresholds", action="store_true", default=False, help="run context mode threshold sweep instead of normal benchmark")
    parser.add_argument("--sweep-samples", type=int, default=5, help="samples per combo in threshold sweep")
    parser.add_argument("--sweep-concat-thresholds", default="3,5,8,10", help="comma list of concat->rnn switch turns to sweep")
    parser.add_argument("--sweep-rnn-thresholds", default="10,15,20,25", help="comma list of rnn->lstm switch turns to sweep")
    parser.add_argument("--sweep-max-messages", default="5,10,20,40", help="comma list of shortWindow.maxMessages to sweep")
    parser.add_argument("--sweep-json-output", default="build/threshold_sweep_report.json", help="output path for sweep JSON")
    parser.add_argument("--sweep-md-output", default="build/threshold_sweep_report.md", help="output path for sweep Markdown")
    parser.add_argument("--phoenix-url", default=DEFAULT_PHOENIX_URL)
    parser.add_argument("--phoenix-token", default=DEFAULT_PHOENIX_TOKEN)
    parser.add_argument("--llama-server-url", default=DEFAULT_LLAMA_SERVER_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--providers", default="phoenix,llama_server", help="comma list: phoenix,llama_server,ollama")
    parser.add_argument(
        "--scenarios",
        default="short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus,cross_session",
        help="comma list: short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus,cross_session",
    )
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_OUTPUT)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    if args.sweep_thresholds:
        root = workspace_root()
        gpt_path = root / "tests" / "GPT4all" / "gpt4all.jsonl"
        wiki_path = root / "robots" / "wikitext-103-all.txt"
        if not gpt_path.exists() or not wiki_path.exists():
            raise SystemExit("required corpus files are missing under tests/GPT4all or robots/")
        sample_need_sw = max(10, args.sweep_samples * 5)
        qa_pairs_sw = load_gpt4all_pairs(gpt_path, sample_need_sw)
        facts_sw = load_wikitext_facts(wiki_path, sample_need_sw)
        concat_thresholds = [int(x.strip()) for x in args.sweep_concat_thresholds.split(",") if x.strip()]
        rnn_thresholds = [int(x.strip()) for x in args.sweep_rnn_thresholds.split(",") if x.strip()]
        max_messages_values = [int(x.strip()) for x in args.sweep_max_messages.split(",") if x.strip()]
        print(f"[sweep] starting threshold sweep: {len(concat_thresholds)*len(rnn_thresholds)*len(max_messages_values)} combos x {args.sweep_samples} samples", flush=True)
        sweep_results = run_threshold_sweep(
            root, facts_sw, qa_pairs_sw, args,
            samples_per_combo=args.sweep_samples,
            concat_thresholds=concat_thresholds,
            rnn_thresholds=rnn_thresholds,
            max_messages_values=max_messages_values,
        )
        sweep_json_out = root / args.sweep_json_output
        sweep_md_out = root / args.sweep_md_output
        sweep_json_out.parent.mkdir(parents=True, exist_ok=True)
        sweep_json_out.write_text(json.dumps(sweep_results, ensure_ascii=False, indent=2), encoding="utf-8")
        sweep_md_out.write_text(build_sweep_markdown(sweep_results), encoding="utf-8")
        print(f"[OK] sweep wrote {sweep_json_out}", flush=True)
        print(f"[OK] sweep wrote {sweep_md_out}", flush=True)
        return 0

    # Removed minimum sample constraint to allow smaller test sizes for CPU-only testing
    # if args.sample_per_scenario < 100:
    #     raise SystemExit("sample-per-scenario must be >= 100")

    root = workspace_root()
    gpt_path = root / "tests" / "GPT4all" / "gpt4all.jsonl"
    wiki_path = root / "robots" / "wikitext-103-all.txt"
    if not gpt_path.exists() or not wiki_path.exists():
        raise SystemExit("required corpus files are missing under tests/GPT4all or robots/")

    sample_need = args.sample_per_scenario
    phoenix_timeout_s = max(args.timeout, args.phoenix_timeout_floor)
    qa_pairs = load_gpt4all_pairs(gpt_path, sample_need * 30)
    facts = load_wikitext_facts(wiki_path, sample_need * 20)
    if len(qa_pairs) < sample_need * 2:
        raise SystemExit(f"insufficient GPT4all samples: {len(qa_pairs)}")
    if len(facts) < sample_need * 2:
        raise SystemExit(f"insufficient wikitext samples: {len(facts)}")

    providers = parse_csv_arg(args.providers)
    valid = {"phoenix", "llama_server", "ollama"}
    if any(p not in valid for p in providers):
        raise SystemExit("providers must be subset of phoenix,llama_server,ollama")

    selected_scenarios = parse_csv_arg(args.scenarios)
    valid_scenarios = {"short_dialogue", "long_dialogue_5_15", "ultra_long_dialogue_15_plus", "cross_session"}
    if any(s not in valid_scenarios for s in selected_scenarios):
        raise SystemExit("scenarios must be subset of short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus,cross_session")
    if not selected_scenarios:
        raise SystemExit("at least one scenario must be selected")

    cache_path = root / args.cache_path
    result_cache = load_result_cache(cache_path) if args.use_cache else {"version": 1, "entries": {}}

    report: dict[str, Any] = {
        "generatedAt": now_str(),
        "metadata": {
            "sources": [
                "tests/GPT4all/gpt4all.jsonl",
                "robots/wikitext-103-all.txt",
            ],
            "samplePerScenario": args.sample_per_scenario,
            "semanticThreshold": 0.70,
            "timeout": args.timeout,
            "phoenixRequestTimeout": phoenix_timeout_s,
            "providers": providers,
            "scenarios": selected_scenarios,
            "interrupted": False,
            "cachePath": str(args.cache_path),
            "cacheEnabled": bool(args.use_cache),
            "scoreVersion": SCORING_VERSION,
        },
        "providers": {},
    }

    l_min, l_max = long_range()
    u_min, u_max = ultra_range()
    scenario_budgets = {
        "short_dialogue": short_dialogue_budget(sample_need),
        "long_dialogue_5_15": memory_dialogue_budget(sample_need, l_min, l_max),
        "ultra_long_dialogue_15_plus": memory_dialogue_budget(sample_need, u_min, u_max),
        "cross_session": cross_session_budget(sample_need),
    }
    selected_budgets = {name: scenario_budgets[name] for name in selected_scenarios}
    overall_budget = total_budget(list(selected_budgets.values()))
    print(
        f"[budget] conversationsPerProvider={sample_need * len(selected_scenarios)} requestBudgetMin={overall_budget.min_requests} requestBudgetMax={overall_budget.max_requests}",
        flush=True,
    )
    for scenario_name, budget in selected_budgets.items():
        print(
            f"[budget] scenario={scenario_name} minRequests={budget.min_requests} maxRequests={budget.max_requests}",
            flush=True,
        )

    local_processes: list[Any] = []
    local_log_handles: list[Any] = []

    try:
        for provider in providers:
            if provider == "phoenix":
                phoenix_ready = warmup_phoenix(args, skip_watcher=args.launch_local_stack)
                if not phoenix_ready and args.launch_local_stack:
                    print("[info] phoenix endpoint not ready; launching local llama-server + phoenix stack", flush=True)
                    stack_log_dir = (root / args.json_output).parent / "stack_logs"
                    try:
                        local_processes, local_log_handles = start_local_phoenix_stack(root, 8083, 5080, stack_log_dir)
                        phoenix_ready = warmup_phoenix(args, skip_watcher=True)
                    except Exception as exc:
                        print(f"[ERROR] failed to launch local phoenix stack: {exc}", flush=True)
                        phoenix_ready = False
                if not phoenix_ready:
                    print(f"[SKIP] provider=phoenix is unavailable; recording empty results and continuing", flush=True)
                    empty_scenarios = {}
                    for s in selected_scenarios:
                        empty_scenarios[s] = summarize([], result_with_counts([], args.sample_per_scenario, 0), scenario_budgets[s])
                        empty_scenarios[s]["skipped"] = True
                        empty_scenarios[s]["skipReason"] = "warmup_failed"
                    report["providers"]["phoenix"] = {
                        "requestBudget": {
                            "perScenario": {name: budget_to_dict(budget) for name, budget in selected_budgets.items()},
                            "total": budget_to_dict(overall_budget),
                        },
                        "scenarios": empty_scenarios,
                        "skipped": True,
                        "skipReason": "warmup_failed",
                    }
                    continue

            long_result = result_with_counts([], sample_need, 0)
            short_result = result_with_counts([], sample_need, 0)
            ultra_result = result_with_counts([], sample_need, 0)
            cross_result = result_with_counts([], sample_need, 0)
            provider_interrupted = False

            _json_out_path = root / args.json_output
            _md_out_path = root / args.md_output

            if "short_dialogue" in selected_scenarios:
                print(f"[run] provider={provider} scenario=short_dialogue", flush=True)
                short_result = run_short_dialogue(sample_need, qa_pairs, phoenix_timeout_s if provider == "phoenix" else args.timeout, provider, args, args.seed + 1, result_cache)
                provider_interrupted = short_result.interrupted
                if "short_dialogue" in selected_scenarios:
                    report["providers"].setdefault(provider, {"scenarios": {}})["scenarios"]["short_dialogue"] = summarize(short_result.samples, short_result, scenario_budgets["short_dialogue"])
                _save_checkpoint(report, _json_out_path, _md_out_path)

            if not provider_interrupted and "long_dialogue_5_15" in selected_scenarios:
                print(f"[run] provider={provider} scenario=long_dialogue_5_15", flush=True)
                long_result = run_memory_dialogue(sample_need, facts, [q[0] for q in qa_pairs], phoenix_timeout_s if provider == "phoenix" else args.timeout, provider, args, args.seed + 2, "long_dialogue_5_15", *long_range(), result_cache)
                provider_interrupted = long_result.interrupted
                if "long_dialogue_5_15" in selected_scenarios:
                    report["providers"].setdefault(provider, {"scenarios": {}})["scenarios"]["long_dialogue_5_15"] = summarize(long_result.samples, long_result, scenario_budgets["long_dialogue_5_15"])
                _save_checkpoint(report, _json_out_path, _md_out_path)

            if not provider_interrupted and "ultra_long_dialogue_15_plus" in selected_scenarios:
                print(f"[run] provider={provider} scenario=ultra_long_dialogue_15_plus", flush=True)
                ultra_result = run_memory_dialogue(sample_need, facts, [q[0] for q in qa_pairs], phoenix_timeout_s if provider == "phoenix" else args.timeout, provider, args, args.seed + 3, "ultra_long_dialogue_15_plus", *ultra_range(), result_cache)
                provider_interrupted = ultra_result.interrupted
                if "ultra_long_dialogue_15_plus" in selected_scenarios:
                    report["providers"].setdefault(provider, {"scenarios": {}})["scenarios"]["ultra_long_dialogue_15_plus"] = summarize(ultra_result.samples, ultra_result, scenario_budgets["ultra_long_dialogue_15_plus"])
                _save_checkpoint(report, _json_out_path, _md_out_path)

            if not provider_interrupted and "cross_session" in selected_scenarios:
                print(f"[run] provider={provider} scenario=cross_session", flush=True)
                cross_result = run_cross_session(sample_need, facts, phoenix_timeout_s if provider == "phoenix" else args.timeout, provider, args, args.seed + 4, result_cache)
                provider_interrupted = cross_result.interrupted
                if "cross_session" in selected_scenarios:
                    report["providers"].setdefault(provider, {"scenarios": {}})["scenarios"]["cross_session"] = summarize(cross_result.samples, cross_result, scenario_budgets["cross_session"])
                _save_checkpoint(report, _json_out_path, _md_out_path)

            provider_scenarios = {}
            if "short_dialogue" in selected_scenarios:
                provider_scenarios["short_dialogue"] = summarize(short_result.samples, short_result, scenario_budgets["short_dialogue"])
            if "long_dialogue_5_15" in selected_scenarios:
                provider_scenarios["long_dialogue_5_15"] = summarize(long_result.samples, long_result, scenario_budgets["long_dialogue_5_15"])
            if "ultra_long_dialogue_15_plus" in selected_scenarios:
                provider_scenarios["ultra_long_dialogue_15_plus"] = summarize(ultra_result.samples, ultra_result, scenario_budgets["ultra_long_dialogue_15_plus"])
            if "cross_session" in selected_scenarios:
                provider_scenarios["cross_session"] = summarize(cross_result.samples, cross_result, scenario_budgets["cross_session"])

            report["providers"][provider] = {
                "requestBudget": {
                    "perScenario": {name: budget_to_dict(budget) for name, budget in selected_budgets.items()},
                    "total": budget_to_dict(overall_budget),
                },
                "scenarios": provider_scenarios
            }
            if provider_interrupted:
                report["metadata"]["interrupted"] = True
                print(f"[warn] benchmark interrupted while provider={provider}; writing partial report", flush=True)
                break
    except KeyboardInterrupt:
        report["metadata"]["interrupted"] = True
        print("[warn] benchmark interrupted by signal; writing partial report", flush=True)
    finally:
        for proc in local_processes:
            try:
                proc.terminate()
            except Exception:
                pass
        for handle in local_log_handles:
            try:
                handle.close()
            except Exception:
                pass

    json_out = root / args.json_output
    md_out = root / args.md_output
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.write_text(build_markdown(report), encoding="utf-8")
    if args.use_cache:
        save_result_cache(cache_path, result_cache)

    print(f"[OK] wrote {json_out}", flush=True)
    print(f"[OK] wrote {md_out}", flush=True)
    if report["metadata"].get("interrupted"):
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

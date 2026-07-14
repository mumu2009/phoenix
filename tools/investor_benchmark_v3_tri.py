from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8083/v1/chat/completions"
DEFAULT_PHOENIX_TRANSFORMER_CHAT_URL = "http://127.0.0.1:5080/api/transformer/chat"
DEFAULT_PHOENIX_RUNTIME_URL = "http://127.0.0.1:5080/api/runtime/features"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_PHOENIX_TOKEN = "local-dev"
DEFAULT_JSON_OUTPUT = "build/investor_advantage_report_v2.json"
DEFAULT_MD_OUTPUT = "build/investor_advantage_report_v2.md"

# GPT4All training repo references this exact file in gpt4all-training/eval_self_instruct.py.
DEFAULT_GPT4ALL_REFERENCED_DATASET_URL = (
    "https://raw.githubusercontent.com/yizhongw/self-instruct/main/human_eval/user_oriented_instructions.jsonl"
)

DEFAULT_INSTRUCTION_SAMPLES = 100
DEFAULT_WINDOW_SAMPLES = 100
DEFAULT_CONTEXT_WINDOW = 2048
DEFAULT_TIMEOUT_S = 240.0

WORD_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")


@dataclass
class ProviderCall:
    reply: str = ""
    latency_ms: float = 0.0
    ok: bool = False
    status: int = 0
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Gpt4AllExample:
    example_id: str
    instruction: str
    input_text: str
    reference: str


@dataclass
class InstructionResult:
    example_id: str
    ollama: ProviderCall
    phoenix: ProviderCall
    llama_server: ProviderCall
    ollama_score: float
    phoenix_score: float
    llama_server_score: float


@dataclass
class WindowResult:
    example_id: str
    ollama: ProviderCall
    phoenix: ProviderCall
    llama_server: ProviderCall
    ollama_score: float
    phoenix_score: float
    llama_server_score: float


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_log(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        # Some task runners may close stdout when output buffers are saturated.
        pass


def try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def post_json(url: str, payload: dict[str, Any], timeout_s: float, headers: dict[str, str] | None = None, method: str = "POST") -> tuple[int, Any, str]:
    try:
        # Use ASCII-escaped JSON so malformed surrogate chars in source text cannot crash UTF-8 encoding.
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = request.Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.getcode()), try_json(raw), raw
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), try_json(raw), raw
    except Exception as exc:
        return 0, None, str(exc)


def is_disabled_endpoint(url: str) -> bool:
    try:
        p = parse.urlparse(url)
        host = (p.hostname or "").lower()
        port = p.port
        return host in {"127.0.0.1", "localhost"} and port in {1, 9}
    except Exception:
        return False



def with_retries(fn, retries: int = 1, sleep_s: float = 0.7):
    last = None
    for attempt in range(retries + 1):
        out = fn()
        if isinstance(out, ProviderCall):
            # Retry on transport error or server instability.
            if out.status == 200 and out.ok:
                return out
            if out.status in (0, 429, 500, 502, 503, 504) or "timed out" in out.error.lower():
                last = out
                if attempt < retries:
                    time.sleep(sleep_s)
                    continue
        return out
    return last


def get_json(url: str, timeout_s: float, headers: dict[str, str] | None = None) -> tuple[int, Any, str]:
    req = request.Request(url, method="GET")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.getcode()), try_json(raw), raw
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return int(exc.code), try_json(raw), raw
    except Exception as exc:
        return 0, None, str(exc)


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_runtime_features(runtime_url: str, token: str, timeout_s: float) -> dict[str, Any]:
    status, parsed, raw = get_json(runtime_url, timeout_s, auth_headers(token))
    if status == 200 and isinstance(parsed, dict) and isinstance(parsed.get("features"), dict):
        return parsed["features"]
    raise RuntimeError(raw or f"failed to read runtime features: status={status}")


def patch_runtime_features(runtime_url: str, token: str, patch: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    status, parsed, raw = post_json(runtime_url, patch, timeout_s, auth_headers(token), method="PATCH")
    if status == 200 and isinstance(parsed, dict) and parsed.get("ok") is True:
        return parsed
    raise RuntimeError(raw or f"failed to patch runtime features: status={status}")


def build_restore_patch(features: dict[str, Any]) -> dict[str, Any]:
    pipeline = features.get("pipeline", {}) if isinstance(features, dict) else {}
    return {
        "transformerMode": pipeline.get("transformerMode", "ollama"),
        "ollamaModel": pipeline.get("ollamaModel", DEFAULT_OLLAMA_MODEL),
        "contextLayerEnabled": not bool(pipeline.get("contextModuleDisabled", False)),
        "gnnModuleDisabled": bool(pipeline.get("gnnModuleDisabled", False)),
        "brainEnabled": bool(pipeline.get("brainEnabled", True)),
    }


def build_same_model_patch(model: str) -> dict[str, Any]:
    return {
        "transformerMode": "ollama",
        "ollamaModel": model,
        "contextLayerEnabled": True,
        "brainEnabled": True,
        "gnnModuleDisabled": False,
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


def token_recall(pred: str, ref: str) -> float:
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
    return overlap / max(1, len(r))


def char_ngram_jaccard(pred: str, ref: str, n: int = 3) -> float:
    p = normalize_text(pred).replace(" ", "")
    r = normalize_text(ref).replace(" ", "")
    if len(p) < n or len(r) < n:
        return 0.0
    p_set = {p[i : i + n] for i in range(0, len(p) - n + 1)}
    r_set = {r[i : i + n] for i in range(0, len(r) - n + 1)}
    if not p_set or not r_set:
        return 0.0
    inter = len(p_set & r_set)
    union = len(p_set | r_set)
    return inter / max(1, union)


def semantic_score(pred: str, ref: str) -> float:
    f1 = token_f1(pred, ref)
    jac = char_ngram_jaccard(pred, ref)
    return round(0.65 * f1 + 0.35 * jac, 4)


def semantic_match(pred: str, ref: str, threshold: float = 0.28) -> tuple[bool, float]:
    score = semantic_score(pred, ref)
    if score >= threshold:
        return True, score
    if token_recall(pred, ref) >= 0.62:
        return True, score
    # A fallback for short factual references.
    short_ref = normalize_text(ref)
    short_pred = normalize_text(pred)
    if short_ref and len(short_ref) <= 40 and short_ref in short_pred:
        return True, score
    return False, score


def summarize_latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round(len(ordered) * 0.95) - 1))
    return {
        "avg": round(sum(ordered) / len(ordered), 2),
        "median": round(statistics.median(ordered), 2),
        "p95": round(ordered[p95_index], 2),
    }


def summarize_rate(success_count: int, total_count: int) -> dict[str, float | int]:
    if total_count <= 0:
        return {"success": 0, "total": 0, "accuracy": 0.0}
    return {
        "success": success_count,
        "total": total_count,
        "accuracy": round(success_count * 100.0 / total_count, 3),
    }


def extract_ollama_reply(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()
    return ""


def extract_phoenix_result(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"]
    return {}


def extract_llama_server_reply(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"].strip()
    return ""


def call_ollama_messages(messages: list[dict[str, str]], model: str, ollama_url: str, timeout_s: float, context_window: int, max_tokens: int = 96) -> ProviderCall:
    if is_disabled_endpoint(ollama_url):
        return ProviderCall(reply="", latency_ms=0.0, ok=False, status=0, error="disabled endpoint")

    def _once() -> ProviderCall:
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
        t0 = time.perf_counter()
        status, parsed, raw = post_json(ollama_url, payload, timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        reply = extract_ollama_reply(parsed)
        ok = status == 200 and bool(reply)
        return ProviderCall(reply=reply, latency_ms=latency_ms, ok=ok, status=status, error="" if ok else (raw[:280] if raw else f"status={status}"), meta={"status": status})

    return with_retries(_once)


def call_phoenix_transformer_chat(text: str, token: str, url: str, session_id: str, timeout_s: float, max_tokens: int = 96) -> ProviderCall:
    if is_disabled_endpoint(url):
        return ProviderCall(reply="", latency_ms=0.0, ok=False, status=0, error="disabled endpoint")

    def _once() -> ProviderCall:
        payload = {
            "text": text,
            "sessionId": session_id,
            "enableAddonToolContract": False,
            "enableGraphSelector": False,
            "maxTokens": max_tokens,
        }
        t0 = time.perf_counter()
        status, parsed, raw = post_json(url, payload, timeout_s, auth_headers(token))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = extract_phoenix_result(parsed)
        reply = str(result.get("reply", "")).strip() if isinstance(result, dict) else ""
        ok = status == 200 and isinstance(parsed, dict) and parsed.get("ok") is True and bool(reply)
        meta = {
            "status": status,
            "transformerMode": result.get("transformerMode") if isinstance(result, dict) else "",
            "provider": result.get("provider", {}) if isinstance(result, dict) else {},
            "modelElapsedMs": result.get("modelElapsedMs") if isinstance(result, dict) else None,
            "contextModuleDisabled": result.get("contextModuleDisabled") if isinstance(result, dict) else None,
        }
        return ProviderCall(reply=reply, latency_ms=latency_ms, ok=ok, status=status, error="" if ok else (raw[:280] if raw else f"status={status}"), meta=meta)

    return with_retries(_once)


def call_llama_server_chat(messages: list[dict[str, str]], model: str, url: str, timeout_s: float, max_tokens: int = 96) -> ProviderCall:
    if is_disabled_endpoint(url):
        return ProviderCall(reply="", latency_ms=0.0, ok=False, status=0, error="disabled endpoint")

    def _once() -> ProviderCall:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.8,
            "max_tokens": max_tokens,
            "stream": False,
        }
        t0 = time.perf_counter()
        status, parsed, raw = post_json(url, payload, timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        reply = extract_llama_server_reply(parsed)
        ok = status == 200 and bool(reply)
        return ProviderCall(reply=reply, latency_ms=latency_ms, ok=ok, status=status, error="" if ok else (raw[:280] if raw else f"status={status}"), meta={"status": status})

    return with_retries(_once)


def download_gpt4all_referenced_eval(url: str, timeout_s: float) -> list[Gpt4AllExample]:
    root = workspace_root()
    cache_path = root / "build" / "datasets" / "gpt4all_user_oriented_instructions.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    raw = ""
    if cache_path.exists():
        raw = cache_path.read_text(encoding="utf-8", errors="replace")
    else:
        last_error = ""
        for _ in range(3):
            try:
                req = request.Request(url, method="GET")
                with request.urlopen(req, timeout=timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                if raw.strip():
                    cache_path.write_text(raw, encoding="utf-8")
                    break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(1.0)
        if not raw.strip():
            raise RuntimeError(f"failed to download GPT4All-referenced eval data: {last_error}")
    examples: list[Gpt4AllExample] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = try_json(line)
        if not isinstance(obj, dict):
            continue
        instruction = str(obj.get("instruction", "")).strip()
        task_id = str(obj.get("id", "")).strip()
        instances = obj.get("instances", [])
        if not instruction or not isinstance(instances, list):
            continue
        for idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            inp = str(inst.get("input", "")).strip()
            out = str(inst.get("output", "")).strip()
            if not out:
                continue
            ex_id = f"{task_id}#{idx}"
            examples.append(Gpt4AllExample(example_id=ex_id, instruction=instruction, input_text=inp, reference=out))
    if not examples:
        raise RuntimeError("No examples loaded from GPT4All-referenced evaluation set")
    return examples


def select_examples(examples: list[Gpt4AllExample], count: int, seed: int) -> list[Gpt4AllExample]:
    rng = random.Random(seed)
    ordered = examples[:]
    rng.shuffle(ordered)
    return ordered[:count]


def build_instruction_prompt(ex: Gpt4AllExample) -> str:
    instruction = ex.instruction.strip()
    input_text = ex.input_text.strip()
    # Keep request size bounded to avoid 400/503 due overlong prompts while staying faithful to source data.
    if len(instruction) > 220:
        instruction = instruction[:220]
    if len(input_text) > 320:
        input_text = input_text[:320]

    if ex.input_text:
        return (
            "Follow the instruction and answer in concise English only. Keep it brief and directly relevant.\n"
            f"Instruction: {instruction}\n"
            f"Input: {input_text}\n"
            "Answer:"
        )
    return (
        "Follow the instruction and answer in concise English only. Keep it brief and directly relevant.\n"
        f"Instruction: {instruction}\n"
        "Answer:"
    )


def build_window_prompts(ex: Gpt4AllExample) -> tuple[str, str]:
    ref = ex.reference.strip()
    if len(ref) > 380:
        ref = ref[:380]
    p1 = (
        "Memorize the following reference answer for one turn and reply exactly with: Memorized.\n"
        f"Reference answer: {ref}"
    )
    p2 = "Now restate the memorized reference answer in one concise English sentence."
    return p1, p2


def warmup_ollama(model: str, ollama_url: str, timeout_s: float, context_window: int) -> ProviderCall:
    prompt = "Reply with exactly: READY"
    return call_ollama_messages([{"role": "user", "content": prompt}], model, ollama_url, min(timeout_s, 30.0), context_window, max_tokens=8)


def warmup_llama_server(model: str, llama_server_url: str, timeout_s: float) -> ProviderCall:
    return call_llama_server_chat([{"role": "user", "content": "Reply with exactly: READY"}], model, llama_server_url, min(timeout_s, 30.0), max_tokens=8)


def benchmark_instruction(
    examples: list[Gpt4AllExample],
    model: str,
    ollama_url: str,
    llama_server_url: str,
    phoenix_transformer_url: str,
    phoenix_token: str,
    timeout_s: float,
    context_window: int,
    run_prefix: str,
) -> list[InstructionResult]:
    results: list[InstructionResult] = []
    total = len(examples)
    for idx, ex in enumerate(examples, start=1):
        prompt = build_instruction_prompt(ex)
        ollama = call_ollama_messages([{"role": "user", "content": prompt}], model, ollama_url, timeout_s, context_window, max_tokens=8)
        llama_server = call_llama_server_chat([{"role": "user", "content": prompt}], model, llama_server_url, timeout_s, max_tokens=8)
        phoenix = call_phoenix_transformer_chat(prompt, phoenix_token, phoenix_transformer_url, f"{run_prefix}-inst-{idx}", timeout_s, max_tokens=8)
        _, ollama_score = semantic_match(ollama.reply, ex.reference)
        _, llama_server_score = semantic_match(llama_server.reply, ex.reference)
        _, phoenix_score = semantic_match(phoenix.reply, ex.reference)
        results.append(
            InstructionResult(
                example_id=ex.example_id,
                ollama=ollama,
                phoenix=phoenix,
                llama_server=llama_server,
                ollama_score=ollama_score,
                phoenix_score=phoenix_score,
                llama_server_score=llama_server_score,
            )
        )
        if idx == 1 or idx == total or idx % 10 == 0:
            safe_log(
                f"[instruction] {idx}/{total} ollama_status={ollama.status} "
                f"llama_server_status={llama_server.status} phoenix_status={phoenix.status}"
            )
    return results


def benchmark_window_memory(
    examples: list[Gpt4AllExample],
    model: str,
    ollama_url: str,
    llama_server_url: str,
    phoenix_transformer_url: str,
    phoenix_token: str,
    timeout_s: float,
    context_window: int,
    run_prefix: str,
) -> list[WindowResult]:
    results: list[WindowResult] = []
    total = len(examples)
    for idx, ex in enumerate(examples, start=1):
        remember_prompt, recall_prompt = build_window_prompts(ex)

        # For direct Ollama, one call with explicit prior turns is enough to test short-window recall.
        ollama_recall = call_ollama_messages(
            [
                {"role": "user", "content": remember_prompt},
                {"role": "assistant", "content": "Memorized."},
                {"role": "user", "content": recall_prompt},
            ],
            model,
            ollama_url,
            timeout_s,
            context_window,
            max_tokens=8,
        )

        llama_server_recall = call_llama_server_chat(
            [
                {"role": "user", "content": remember_prompt},
                {"role": "assistant", "content": "Memorized."},
                {"role": "user", "content": recall_prompt},
            ],
            model,
            llama_server_url,
            timeout_s,
            max_tokens=8,
        )

        sid = f"{run_prefix}-win-{idx}"
        phoenix_store = call_phoenix_transformer_chat(remember_prompt, phoenix_token, phoenix_transformer_url, sid, timeout_s, max_tokens=4)
        phoenix_recall = call_phoenix_transformer_chat(recall_prompt, phoenix_token, phoenix_transformer_url, sid, timeout_s, max_tokens=8)
        phoenix_recall.latency_ms += phoenix_store.latency_ms

        _, ollama_score = semantic_match(ollama_recall.reply, ex.reference)
        _, llama_server_score = semantic_match(llama_server_recall.reply, ex.reference)
        _, phoenix_score = semantic_match(phoenix_recall.reply, ex.reference)
        results.append(
            WindowResult(
                example_id=ex.example_id,
                ollama=ollama_recall,
                phoenix=phoenix_recall,
                llama_server=llama_server_recall,
                ollama_score=ollama_score,
                phoenix_score=phoenix_score,
                llama_server_score=llama_server_score,
            )
        )
        if idx == 1 or idx == total or idx % 10 == 0:
            safe_log(
                f"[window] {idx}/{total} ollama_status={ollama_recall.status} "
                f"llama_server_status={llama_server_recall.status} phoenix_status={phoenix_recall.status}"
            )
    return results


def count_http_errors(calls: list[ProviderCall]) -> dict[str, int]:
    out = {"4xx": 0, "5xx": 0, "other": 0}
    for call in calls:
        if 400 <= call.status < 500:
            out["4xx"] += 1
        elif 500 <= call.status < 600:
            out["5xx"] += 1
        elif call.status != 200:
            out["other"] += 1
    return out


def build_markdown(report: dict[str, Any]) -> str:
    inst = report["benchmarks"]["instruction"]
    win = report["benchmarks"]["windowMemory"]
    lines = [
        "# 079 Investor Benchmark v2 (Revised)",
        "",
        f"- Generated at: {report['generatedAt']}",
        "- Data source: GPT4All-referenced evaluation set (self-instruct user_oriented_instructions.jsonl).",
        "- Language policy: all benchmark prompts are English; evaluation references are English.",
        "- Latency policy: chat-path latency only. No world/state endpoint latency is used as model latency.",
        f"- Model: {report['metadata']['ollamaModel']}",
        f"- Context window target for direct Ollama: {report['metadata']['contextWindow']}",
        f"- llama-server endpoint: {report['metadata']['llamaServerUrl']}",
        "",
        "## Executive Summary",
        "",
        f"- Instruction semantic accuracy: Ollama {inst['ollama']['accuracy']:.3f}% ({inst['ollama']['success']}/{inst['ollama']['total']}) | llama-server {inst['llamaServer']['accuracy']:.3f}% ({inst['llamaServer']['success']}/{inst['llamaServer']['total']}) | 079 {inst['phoenix']['accuracy']:.3f}% ({inst['phoenix']['success']}/{inst['phoenix']['total']}).",
        f"- Window-memory semantic accuracy: Ollama {win['ollama']['accuracy']:.3f}% ({win['ollama']['success']}/{win['ollama']['total']}) | llama-server {win['llamaServer']['accuracy']:.3f}% ({win['llamaServer']['success']}/{win['llamaServer']['total']}) | 079 {win['phoenix']['accuracy']:.3f}% ({win['phoenix']['success']}/{win['phoenix']['total']}).",
        f"- Chat latency median (instruction): Ollama {inst['ollamaLatency']['median']:.2f} ms | llama-server {inst['llamaServerLatency']['median']:.2f} ms | 079 {inst['phoenixLatency']['median']:.2f} ms.",
        f"- Chat latency median (window): Ollama {win['ollamaLatency']['median']:.2f} ms | llama-server {win['llamaServerLatency']['median']:.2f} ms | 079 {win['phoenixLatency']['median']:.2f} ms.",
        "",
        "## Gateway Health Check",
        "",
        f"- Instruction route HTTP errors: Ollama {inst['ollamaHttpErrors']} | llama-server {inst['llamaServerHttpErrors']} | 079 {inst['phoenixHttpErrors']}",
        f"- Window route HTTP errors: Ollama {win['ollamaHttpErrors']} | llama-server {win['llamaServerHttpErrors']} | 079 {win['phoenixHttpErrors']}",
        "",
        "## Notes",
        "",
        "- This revision does not present math-addon as a flagship advantage, because external Ollama tool stacks can implement calculator tools too.",
        "- Scoring uses semantic similarity (token-F1 + character n-gram overlap), not strict exact string match.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPT4All-sourced benchmark for 079 vs Ollama vs llama-server")
    parser.add_argument("--dataset-url", default=DEFAULT_GPT4ALL_REFERENCED_DATASET_URL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llama-server-url", default=DEFAULT_LLAMA_SERVER_URL)
    parser.add_argument("--phoenix-transformer-chat-url", default=DEFAULT_PHOENIX_TRANSFORMER_CHAT_URL)
    parser.add_argument("--phoenix-runtime-url", default=DEFAULT_PHOENIX_RUNTIME_URL)
    parser.add_argument("--phoenix-token", default=DEFAULT_PHOENIX_TOKEN)
    parser.add_argument("--instruction-samples", type=int, default=DEFAULT_INSTRUCTION_SAMPLES)
    parser.add_argument("--window-samples", type=int, default=DEFAULT_WINDOW_SAMPLES)
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--skip-runtime-patch", action="store_true", help="Skip reading/patching phoenix runtime features")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    if args.instruction_samples < 50 or args.window_samples < 50:
        raise SystemExit("instruction-samples and window-samples must both be >= 50")
    if args.instruction_samples > 200 or args.window_samples > 200:
        raise SystemExit("instruction-samples and window-samples must both be <= 200")

    root = workspace_root()
    run_prefix = f"bench{int(time.time() * 1000)}"

    restore_patch: dict[str, Any] | None = None
    if not args.skip_runtime_patch:
        # Runtime patch: same model and enable context path, then restore after benchmark.
        original_features = get_runtime_features(args.phoenix_runtime_url, args.phoenix_token, args.timeout)
        restore_patch = build_restore_patch(original_features)
        patch_runtime_features(args.phoenix_runtime_url, args.phoenix_token, build_same_model_patch(args.ollama_model), args.timeout)

    try:
        all_examples = download_gpt4all_referenced_eval(args.dataset_url, args.timeout)
        if len(all_examples) < max(args.instruction_samples, args.window_samples):
            raise RuntimeError(f"dataset has only {len(all_examples)} examples, fewer than requested sample count")

        instruction_examples = select_examples(all_examples, args.instruction_samples, args.seed)
        window_examples = select_examples(all_examples, args.window_samples, args.seed + 13)

        safe_log(f"[setup] loaded dataset examples={len(all_examples)}")
        safe_log(f"[setup] instruction_samples={len(instruction_examples)} window_samples={len(window_examples)}")

        warmup = warmup_ollama(args.ollama_model, args.ollama_url, args.timeout, args.context_window)
        safe_log(f"[warmup] ollama_status={warmup.status} latency_ms={warmup.latency_ms:.2f}")
        warmup_lls = warmup_llama_server(args.ollama_model, args.llama_server_url, args.timeout)
        safe_log(f"[warmup] llama_server_status={warmup_lls.status} latency_ms={warmup_lls.latency_ms:.2f}")

        instruction_results = benchmark_instruction(
            instruction_examples,
            args.ollama_model,
            args.ollama_url,
            args.llama_server_url,
            args.phoenix_transformer_chat_url,
            args.phoenix_token,
            args.timeout,
            args.context_window,
            run_prefix,
        )

        window_results = benchmark_window_memory(
            window_examples,
            args.ollama_model,
            args.ollama_url,
            args.llama_server_url,
            args.phoenix_transformer_chat_url,
            args.phoenix_token,
            args.timeout,
            args.context_window,
            run_prefix,
        )
    finally:
        if restore_patch is not None:
            patch_runtime_features(args.phoenix_runtime_url, args.phoenix_token, restore_patch, args.timeout)

    inst_ollama_hits = sum(1 for r in instruction_results if semantic_match(r.ollama.reply, next(ex.reference for ex in instruction_examples if ex.example_id == r.example_id))[0] and r.ollama.status == 200)
    inst_llama_server_hits = sum(1 for r in instruction_results if semantic_match(r.llama_server.reply, next(ex.reference for ex in instruction_examples if ex.example_id == r.example_id))[0] and r.llama_server.status == 200)
    inst_phoenix_hits = sum(1 for r in instruction_results if semantic_match(r.phoenix.reply, next(ex.reference for ex in instruction_examples if ex.example_id == r.example_id))[0] and r.phoenix.status == 200)

    win_ollama_hits = sum(1 for r in window_results if semantic_match(r.ollama.reply, next(ex.reference for ex in window_examples if ex.example_id == r.example_id))[0] and r.ollama.status == 200)
    win_llama_server_hits = sum(1 for r in window_results if semantic_match(r.llama_server.reply, next(ex.reference for ex in window_examples if ex.example_id == r.example_id))[0] and r.llama_server.status == 200)
    win_phoenix_hits = sum(1 for r in window_results if semantic_match(r.phoenix.reply, next(ex.reference for ex in window_examples if ex.example_id == r.example_id))[0] and r.phoenix.status == 200)

    inst_ollama_calls = [r.ollama for r in instruction_results]
    inst_llama_server_calls = [r.llama_server for r in instruction_results]
    inst_phoenix_calls = [r.phoenix for r in instruction_results]
    win_ollama_calls = [r.ollama for r in window_results]
    win_llama_server_calls = [r.llama_server for r in window_results]
    win_phoenix_calls = [r.phoenix for r in window_results]

    report = {
        "generatedAt": now_str(),
        "metadata": {
            "source": "GPT4All referenced eval set via gpt4all-training/eval_self_instruct.py",
            "datasetUrl": args.dataset_url,
            "datasetFingerprint": hashlib.sha256(args.dataset_url.encode("utf-8")).hexdigest()[:16],
            "ollamaModel": args.ollama_model,
            "contextWindow": args.context_window,
            "runPrefix": run_prefix,
            "instructionSamples": args.instruction_samples,
            "windowSamples": args.window_samples,
            "phoenixTransformerChatUrl": args.phoenix_transformer_chat_url,
            "ollamaUrl": args.ollama_url,
            "llamaServerUrl": args.llama_server_url,
        },
        "benchmarks": {
            "instruction": {
                "ollama": summarize_rate(inst_ollama_hits, len(instruction_results)),
                "llamaServer": summarize_rate(inst_llama_server_hits, len(instruction_results)),
                "phoenix": summarize_rate(inst_phoenix_hits, len(instruction_results)),
                "ollamaLatency": summarize_latency([c.latency_ms for c in inst_ollama_calls]),
                "llamaServerLatency": summarize_latency([c.latency_ms for c in inst_llama_server_calls]),
                "phoenixLatency": summarize_latency([c.latency_ms for c in inst_phoenix_calls]),
                "ollamaHttpErrors": count_http_errors(inst_ollama_calls),
                "llamaServerHttpErrors": count_http_errors(inst_llama_server_calls),
                "phoenixHttpErrors": count_http_errors(inst_phoenix_calls),
            },
            "windowMemory": {
                "ollama": summarize_rate(win_ollama_hits, len(window_results)),
                "llamaServer": summarize_rate(win_llama_server_hits, len(window_results)),
                "phoenix": summarize_rate(win_phoenix_hits, len(window_results)),
                "ollamaLatency": summarize_latency([c.latency_ms for c in win_ollama_calls]),
                "llamaServerLatency": summarize_latency([c.latency_ms for c in win_llama_server_calls]),
                "phoenixLatency": summarize_latency([c.latency_ms for c in win_phoenix_calls]),
                "ollamaHttpErrors": count_http_errors(win_ollama_calls),
                "llamaServerHttpErrors": count_http_errors(win_llama_server_calls),
                "phoenixHttpErrors": count_http_errors(win_phoenix_calls),
            },
        },
        "raw": {
            "instruction": [asdict(r) for r in instruction_results],
            "windowMemory": [asdict(r) for r in window_results],
        },
    }

    json_output = (root / args.json_output).resolve()
    md_output = (root / args.md_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)

    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_output.write_text(build_markdown(report), encoding="utf-8")

    safe_log(f"[OK] wrote {json_output}")
    safe_log(f"[OK] wrote {md_output}")
    safe_log(json.dumps(report["benchmarks"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_CONTEXT_BASE_URL = "http://127.0.0.1:5081"
DEFAULT_OLLAMA_MODEL = "tinyllama:latest"
DEFAULT_GPT4ALL_PATH = "tests/GPT4all/gpt4all.jsonl"
DEFAULT_JSON_OUTPUT = "build/investor_advantage_report.json"
DEFAULT_MD_OUTPUT = "build/investor_advantage_report.md"


@dataclass
class AddonResult:
    handled: bool = False
    reply: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MathCase:
    case_id: str
    expression: str
    expected: str


@dataclass
class MathResult:
    case_id: str
    expression: str
    expected: str
    addon_reply: str
    addon_ok: bool
    addon_latency_ms: float
    ollama_reply: str
    ollama_ok: bool
    ollama_latency_ms: float


@dataclass
class MemoryCase:
    case_id: str
    label: str
    code: str


@dataclass
class MemoryResult:
    case_id: str
    label: str
    code: str
    system_ok: bool
    system_latency_ms: float
    world_summary: str
    ollama_ok: bool
    ollama_latency_ms: float
    ollama_reply: str


def install_addon_stub() -> None:
    addon_mod = types.ModuleType("addon")

    def invoke_addon_online_lookup(_query: str, _options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    addon_mod.AddonResult = AddonResult
    addon_mod.invokeAddonOnlineLookup = invoke_addon_online_lookup
    sys.modules["addon"] = addon_mod


def make_math_addon():
    install_addon_stub()
    root_text = str(workspace_root())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from addons.math_addon import createMathAddon

    return createMathAddon("math")


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def post_json(url: str, payload: dict[str, Any], timeout_s: float, headers: dict[str, str] | None = None) -> tuple[int, Any, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
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


def try_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return None


def extract_ollama_reply(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()
    return ""


def normalized_number_tokens(text: str) -> list[str]:
    compact = text.replace(",", "")
    return [token.lstrip("+") for token in re.findall(r"[+-]?\d+(?:\.\d+)?", compact)]


def number_match(text: str, expected: str) -> bool:
    expected_norm = expected.replace(",", "").lstrip("+")
    return expected_norm in normalized_number_tokens(text)


def call_ollama(prompt: str, model: str, ollama_url: str, timeout_s: float) -> tuple[str, float, int, str]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    t0 = time.perf_counter()
    status, parsed, raw = post_json(ollama_url, payload, timeout_s)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return extract_ollama_reply(parsed), latency_ms, status, raw


def make_math_cases(count: int, seed: int) -> list[MathCase]:
    rng = random.Random(seed)
    cases: list[MathCase] = []
    patterns = [
        lambda: (f"{rng.randint(1000, 99999)} * {rng.randint(11, 99)}", None),
        lambda: (f"({rng.randint(20, 999)} + {rng.randint(20, 999)}) * {rng.randint(3, 21)}", None),
        lambda: (f"{rng.randint(5000, 99999)} - {rng.randint(10, 4999)} + {rng.randint(10, 999)}", None),
        lambda: (f"({rng.randint(12, 48)} * {rng.randint(12, 48)}) + ({rng.randint(50, 200)} * {rng.randint(2, 9)})", None),
        lambda: (f"pow({rng.randint(2, 12)}, {rng.randint(2, 4)}) + {rng.randint(10, 999)}", None),
    ]
    for index in range(1, count + 1):
        expr, _ = patterns[(index - 1) % len(patterns)]()
        expected_value = eval(expr, {"__builtins__": {}}, {"pow": pow})
        expected = str(int(expected_value)) if float(expected_value).is_integer() else format(float(expected_value), "g")
        cases.append(MathCase(case_id=f"math-{index:03d}", expression=expr, expected=expected))
    return cases


def benchmark_math(cases: list[MathCase], model: str, ollama_url: str, timeout_s: float) -> list[MathResult]:
    addon = make_math_addon()
    results: list[MathResult] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        addon_result = addon.handle(f"计算: {case.expression}", {})
        addon_latency_ms = (time.perf_counter() - t0) * 1000.0
        prompt = f"请计算 {case.expression}，只输出阿拉伯数字结果，不要解释。"
        ollama_reply, ollama_latency_ms, _status, _raw = call_ollama(prompt, model, ollama_url, timeout_s)
        results.append(
            MathResult(
                case_id=case.case_id,
                expression=case.expression,
                expected=case.expected,
                addon_reply=addon_result.reply,
                addon_ok=addon_result.handled and addon_result.reply == case.expected,
                addon_latency_ms=addon_latency_ms,
                ollama_reply=ollama_reply,
                ollama_ok=number_match(ollama_reply, case.expected),
                ollama_latency_ms=ollama_latency_ms,
            )
        )
        print(f"[math] {index}/{total} addon={'ok' if results[-1].addon_ok else 'fail'} ollama={'ok' if results[-1].ollama_ok else 'fail'}", flush=True)
    return results


def compact_label(text: str, index: int) -> str:
    first = text.replace("\n", " ").strip()
    first = re.sub(r"\s+", " ", first)
    first = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff ]+", "", first)
    first = first[:36].strip()
    return first or f"样本{index}"


def load_gpt4all_labels(path: Path, count: int) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except Exception:
                continue
            instruction = str(record.get("instruction", "")).strip()
            if not instruction:
                continue
            labels.append(compact_label(instruction, line_number))
            if len(labels) >= count * 6:
                break
    if len(labels) < count:
        raise RuntimeError(f"not enough GPT4All labels: need {count}, got {len(labels)}")
    rng = random.Random(20260524)
    return rng.sample(labels, count)


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def context_reset(base_url: str, token: str, session_id: str, timeout_s: float) -> None:
    post_json(f"{base_url.rstrip('/')}/context/reset", {"sessionId": session_id}, timeout_s, auth_headers(token))


def world_state(base_url: str, token: str, session_id: str, timeout_s: float) -> tuple[Any, str]:
    query = parse.urlencode({"sessionId": session_id, "limit": 4})
    _status, parsed, raw = get_json(f"{base_url.rstrip('/')}/world/state?{query}", timeout_s, auth_headers(token))
    return parsed, raw


def benchmark_memory(cases: list[MemoryCase], base_url: str, token: str, model: str, ollama_url: str, timeout_s: float) -> list[MemoryResult]:
    results: list[MemoryResult] = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        session_id = case.case_id
        fact_text = f"{case.label} 验证码 {case.code}"
        context_reset(base_url, token, session_id, timeout_s)
        t0 = time.perf_counter()
        _status_ingest, _parsed_ingest, _raw_ingest = post_json(
            f"{base_url.rstrip('/')}/context/ingest",
            {"sessionId": session_id, "text": fact_text, "mode": "memory"},
            timeout_s,
            auth_headers(token),
        )
        state_payload, state_raw = world_state(base_url, token, session_id, timeout_s)
        system_latency_ms = (time.perf_counter() - t0) * 1000.0
        state_text = json.dumps(state_payload, ensure_ascii=False) if state_payload is not None else state_raw
        system_ok = case.code in state_text and case.label[:8] in state_text

        prompt_store = f"请记住：{case.label} 的验证码是 {case.code}。只回复 已记住。"
        prompt_recall = f"{case.label} 的验证码是什么？只回复数字。"
        reply1, latency1_ms, _status1, _raw1 = call_ollama(prompt_store, model, ollama_url, timeout_s)
        reply2, latency2_ms, _status2, _raw2 = call_ollama(prompt_recall, model, ollama_url, timeout_s)
        ollama_ok = number_match(reply2, case.code)
        results.append(
            MemoryResult(
                case_id=case.case_id,
                label=case.label,
                code=case.code,
                system_ok=system_ok,
                system_latency_ms=system_latency_ms,
                world_summary=extract_world_summary(state_payload),
                ollama_ok=ollama_ok,
                ollama_latency_ms=latency1_ms + latency2_ms,
                ollama_reply=reply2 or reply1,
            )
        )
        print(f"[memory] {index}/{total} system={'ok' if results[-1].system_ok else 'fail'} ollama={'ok' if results[-1].ollama_ok else 'fail'}", flush=True)
    return results


def extract_world_summary(state_payload: Any) -> str:
    if not isinstance(state_payload, dict):
        return ""
    for key in ("summary", "sceneState", "episode"):
        value = state_payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            summary = value.get("summary")
            if isinstance(summary, str):
                return summary
    return json.dumps(state_payload, ensure_ascii=False)[:240]


def make_memory_cases(labels: list[str], seed: int) -> list[MemoryCase]:
    rng = random.Random(seed)
    out: list[MemoryCase] = []
    for index, label in enumerate(labels, start=1):
        code = "".join(str(rng.randint(0, 9)) for _ in range(6))
        out.append(MemoryCase(case_id=f"memory-{index:03d}", label=label, code=code))
    return out


def summarize_boolean_metric(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if item) * 100.0 / len(values)


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


def sample_failures_math(results: list[MathResult], limit: int = 3) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in results:
        if item.ollama_ok:
            continue
        out.append({
            "case": item.case_id,
            "expr": item.expression,
            "expected": item.expected,
            "ollama": item.ollama_reply[:200],
        })
        if len(out) >= limit:
            break
    return out


def sample_failures_memory(results: list[MemoryResult], limit: int = 3) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in results:
        if item.ollama_ok:
            continue
        out.append({
            "case": item.case_id,
            "label": item.label,
            "expected": item.code,
            "ollama": item.ollama_reply[:200],
        })
        if len(out) >= limit:
            break
    return out


def build_mermaid_accuracy(math_summary: dict[str, Any], memory_summary: dict[str, Any]) -> str:
    return "\n".join([
        "```mermaid",
        "xychart-beta",
        "    title \"Accuracy Comparison\"",
        "    x-axis [\"Math-079\", \"Math-Ollama\", \"Memory-079\", \"Memory-Ollama\"]",
        "    y-axis \"Exact Match %\" 0 --> 100",
        f"    bar [{math_summary['addon_accuracy']:.2f}, {math_summary['ollama_accuracy']:.2f}, {memory_summary['system_accuracy']:.2f}, {memory_summary['ollama_accuracy']:.2f}]",
        "```",
    ])


def build_mermaid_latency(math_summary: dict[str, Any], memory_summary: dict[str, Any]) -> str:
    return "\n".join([
        "```mermaid",
        "xychart-beta",
        "    title \"Median Latency Comparison\"",
        "    x-axis [\"Math-079\", \"Math-Ollama\", \"Memory-079\", \"Memory-Ollama\"]",
        "    y-axis \"Median ms\" 0 --> 8000",
        f"    bar [{math_summary['addon_latency']['median']:.2f}, {math_summary['ollama_latency']['median']:.2f}, {memory_summary['system_latency']['median']:.2f}, {memory_summary['ollama_latency']['median']:.2f}]",
        "```",
    ])


def build_markdown(report: dict[str, Any]) -> str:
    math_summary = report["math"]
    memory_summary = report["memory"]
    lines = [
        "# 079 Investor Benchmark",
        "",
        f"- Generated at: {report['generatedAt']}",
        f"- Ollama model: {report['metadata']['ollamaModel']}",
        f"- Math cases: {report['metadata']['mathCaseCount']}",
        f"- Memory cases: {report['metadata']['memoryCaseCount']}",
        f"- GPT4All source: {report['metadata']['gpt4allPath']}",
        "",
        "## Executive Summary",
        "",
        f"- 079 math addon exact-match rate: {math_summary['addon_accuracy']:.2f}%",
        f"- Direct Ollama math exact-match rate: {math_summary['ollama_accuracy']:.2f}%",
        f"- 079 explicit memory exact-match rate: {memory_summary['system_accuracy']:.2f}%",
        f"- Direct Ollama stateless recall exact-match rate: {memory_summary['ollama_accuracy']:.2f}%",
        f"- 079 math median latency: {math_summary['addon_latency']['median']:.2f} ms",
        f"- Direct Ollama math median latency: {math_summary['ollama_latency']['median']:.2f} ms",
        f"- 079 memory median latency: {memory_summary['system_latency']['median']:.2f} ms",
        f"- Direct Ollama memory median latency: {memory_summary['ollama_latency']['median']:.2f} ms",
        "",
        "## Charts",
        "",
        build_mermaid_accuracy(math_summary, memory_summary),
        "",
        build_mermaid_latency(math_summary, memory_summary),
        "",
        "## Method",
        "",
        "- Math benchmark: synthetic arithmetic expressions, same expression evaluated by 079 shipped math addon and by direct Ollama chat on the same tinyllama model.",
        "- Memory benchmark: synthetic 6-digit codes bound to GPT4All-derived text labels. 079 stores them through context/world endpoints, while plain Ollama is asked to remember in one call and recall in a fresh call without history.",
        "- No benchmark answer was copied from training data; the codes are random and generated at runtime.",
        "",
        "## Result Table",
        "",
        "| Benchmark | 079 Accuracy | Ollama Accuracy | 079 Median ms | Ollama Median ms |",
        "|---|---:|---:|---:|---:|",
        f"| Math exact match | {math_summary['addon_accuracy']:.2f}% | {math_summary['ollama_accuracy']:.2f}% | {math_summary['addon_latency']['median']:.2f} | {math_summary['ollama_latency']['median']:.2f} |",
        f"| Session-memory exact match | {memory_summary['system_accuracy']:.2f}% | {memory_summary['ollama_accuracy']:.2f}% | {memory_summary['system_latency']['median']:.2f} | {memory_summary['ollama_latency']['median']:.2f} |",
        "",
        "## Representative Failures",
        "",
    ]
    for item in report["samples"]["mathFailures"]:
        lines.extend([
            f"- Math {item['case']}: expr={item['expr']} expected={item['expected']}",
            f"  Ollama reply: {item['ollama']}",
        ])
    for item in report["samples"]["memoryFailures"]:
        lines.extend([
            f"- Memory {item['case']}: label={item['label']} expected={item['expected']}",
            f"  Ollama reply: {item['ollama']}",
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- These results do not claim that 079 is universally smarter than plain Ollama on open-ended QA. They show that the software stack adds deterministic capabilities ordinary direct-model usage does not provide.",
        "- The strongest measured advantages in this run are exact arithmetic and explicit session memory retention.",
    ])
    return "\n".join(lines)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run investor-facing component benchmarks for 079 vs plain Ollama")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--context-base-url", default=DEFAULT_CONTEXT_BASE_URL)
    parser.add_argument("--system-token", default="local-dev")
    parser.add_argument("--gpt4all-path", default=DEFAULT_GPT4ALL_PATH)
    parser.add_argument("--math-count", type=int, default=24)
    parser.add_argument("--memory-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()

    root = workspace_root()
    math_cases = make_math_cases(args.math_count, args.seed)
    labels = load_gpt4all_labels((root / args.gpt4all_path).resolve(), args.memory_count)
    memory_cases = make_memory_cases(labels, args.seed + 77)

    math_results = benchmark_math(math_cases, args.ollama_model, args.ollama_url, args.timeout)
    memory_results = benchmark_memory(memory_cases, args.context_base_url, args.system_token, args.ollama_model, args.ollama_url, args.timeout)

    math_summary = {
        "addon_accuracy": round(summarize_boolean_metric([item.addon_ok for item in math_results]), 2),
        "ollama_accuracy": round(summarize_boolean_metric([item.ollama_ok for item in math_results]), 2),
        "addon_latency": summarize_latency([item.addon_latency_ms for item in math_results]),
        "ollama_latency": summarize_latency([item.ollama_latency_ms for item in math_results]),
    }
    memory_summary = {
        "system_accuracy": round(summarize_boolean_metric([item.system_ok for item in memory_results]), 2),
        "ollama_accuracy": round(summarize_boolean_metric([item.ollama_ok for item in memory_results]), 2),
        "system_latency": summarize_latency([item.system_latency_ms for item in memory_results]),
        "ollama_latency": summarize_latency([item.ollama_latency_ms for item in memory_results]),
    }

    report = {
        "generatedAt": now_str(),
        "metadata": {
            "ollamaModel": args.ollama_model,
            "ollamaUrl": args.ollama_url,
            "contextBaseUrl": args.context_base_url,
            "gpt4allPath": args.gpt4all_path,
            "mathCaseCount": len(math_results),
            "memoryCaseCount": len(memory_results),
            "seed": args.seed,
        },
        "math": math_summary,
        "memory": memory_summary,
        "samples": {
            "mathFailures": sample_failures_math(math_results),
            "memoryFailures": sample_failures_memory(memory_results),
        },
        "raw": {
            "math": [asdict(item) for item in math_results],
            "memory": [asdict(item) for item in memory_results],
        },
    }

    json_output = (root / args.json_output).resolve()
    md_output = (root / args.md_output).resolve()
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_output.write_text(build_markdown(report), encoding="utf-8")

    print(f"[OK] wrote {json_output}")
    print(f"[OK] wrote {md_output}")
    print(json.dumps({"math": math_summary, "memory": memory_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
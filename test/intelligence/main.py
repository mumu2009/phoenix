#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HAI_DIMENSIONS = (
    "factual_grounding",
    "reasoning_planning",
    "instruction_following",
    "language_control",
    "code_generation",
    "tool_alignment",
)

TASK_TYPE_TO_HAI = {
    "qa": ["factual_grounding"],
    "reasoning": ["reasoning_planning"],
    "formatting": ["instruction_following"],
    "translation": ["language_control"],
    "coding": ["code_generation"],
    "external_qa": ["factual_grounding", "instruction_following"],
}


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text.strip() if not ch.isspace())


def char_bigrams(text: str) -> List[str]:
    if len(text) <= 1:
        return [text] if text else []
    return [text[index:index + 2] for index in range(len(text) - 1)]


def split_keywords(text: str) -> List[str]:
    words: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if len(token) >= 2 and token not in seen:
            seen.add(token)
            words.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 4:
            for size in (2, 3, 4):
                for index in range(len(token) - size + 1):
                    fragment = token[index:index + size]
                    if fragment not in seen:
                        seen.add(fragment)
                        words.append(fragment)
    return words


def split_clauses(text: str) -> List[str]:
    clauses: List[str] = []
    for item in re.split(r"[，。；、,.;:：!！?？\n\r]+", text):
        normalized = normalize_text(item)
        if normalized:
            clauses.append(normalized)
    return clauses


def compute_clause_coverage(normalized_reply: str, reference: str) -> float:
    clauses = split_clauses(reference)
    if not normalized_reply or not clauses:
        return 0.0
    reply_grams = set(char_bigrams(normalized_reply))
    scores: List[float] = []
    for clause in clauses:
        if clause in normalized_reply:
            scores.append(1.0)
            continue
        clause_grams = set(char_bigrams(clause))
        gram_score = (len(reply_grams & clause_grams) / len(clause_grams)) if clause_grams else 0.0
        seq_score = SequenceMatcher(None, normalized_reply, clause).ratio()
        scores.append(max(gram_score, seq_score))
    return sum(scores) / len(scores)


def score_text(reply: str, reference: str) -> Tuple[float, Dict[str, float]]:
    normalized_reply = normalize_text(reply)
    normalized_reference = normalize_text(reference)
    if not normalized_reply or not normalized_reference:
        return 0.0, {"jaccard": 0.0, "sequence": 0.0, "keywordRecall": 0.0, "final": 0.0}

    grams_reply = set(char_bigrams(normalized_reply))
    grams_reference = set(char_bigrams(normalized_reference))
    union = grams_reply | grams_reference
    jaccard = (len(grams_reply & grams_reference) / len(union)) if union else 0.0
    sequence = SequenceMatcher(None, normalized_reply, normalized_reference).ratio()

    keywords = split_keywords(reference)
    if keywords:
        hits = sum(1 for keyword in keywords if keyword in normalized_reply)
        keyword_recall = hits / len(keywords)
    else:
        keyword_recall = 0.0

    clause_coverage = compute_clause_coverage(normalized_reply, reference)
    contains_reference = 1.0 if normalized_reference in normalized_reply else 0.0

    final = max(0.0, min(100.0, jaccard * 10.0 + sequence * 15.0 + keyword_recall * 25.0 + clause_coverage * 35.0 + contains_reference * 15.0))
    return final, {
        "jaccard": round(jaccard * 100.0, 2),
        "sequence": round(sequence * 100.0, 2),
        "keywordRecall": round(keyword_recall * 100.0, 2),
        "clauseCoverage": round(clause_coverage * 100.0, 2),
        "containsReference": round(contains_reference * 100.0, 2),
        "final": round(final, 2),
    }


def score_structured(reply: str, reference: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    score = 0.0
    lowered = reply.lower()

    expected_tool = str(reference.get("tool", "")).strip().lower()
    if expected_tool:
        tool_hit = 100.0 if expected_tool in lowered else 0.0
        metrics["tool"] = tool_hit
        score += tool_hit * 0.4

    result_text = str(reference.get("result", "")).strip().lower()
    if result_text:
        result_hit = 100.0 if result_text in lowered else 0.0
        metrics["result"] = result_hit
        score += result_hit * 0.4

    arguments = reference.get("arguments", {})
    if isinstance(arguments, dict) and arguments:
        total_args = len(arguments)
        arg_hits = sum(1 for value in arguments.values() if str(value).strip().lower() in lowered)
        arg_score = (arg_hits / total_args) * 100.0
        metrics["arguments"] = round(arg_score, 2)
        score += arg_score * 0.2

    if not metrics:
        return 0.0, {"final": 0.0}
    metrics["final"] = round(score, 2)
    return score, metrics


def post_json(url: str, payload: Dict[str, Any], timeout_s: float, headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return int(response.getcode()), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = str(exc)
        return int(exc.code), raw
    except Exception as exc:
        return 0, str(exc)


def discover_ollama_model(ollama_url: str, timeout_s: float) -> str:
    tags_url = ollama_url.rstrip("/")
    if tags_url.endswith("/api/chat"):
        tags_url = tags_url[:-len("/api/chat")] + "/api/tags"
    elif not tags_url.endswith("/api/tags"):
        tags_url = tags_url.rstrip("/") + "/api/tags"

    try:
        request_obj = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(request_obj, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return ""

    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict) and str(item.get("name", "")).strip()]
    preferred = ["llama3.1:8b", "llama3.1:latest", "tinyllama:latest", "qwen2.5:7b", "qwen2.5:14b", "gpt-oss:20b"]
    for candidate in preferred:
        if candidate in names:
            return candidate
    return names[0] if names else ""


def parse_system_reply(raw: str) -> Tuple[bool, str, str]:
    try:
        payload = json.loads(raw)
    except Exception:
        raw_text = raw.strip()
        if raw_text:
            return True, raw_text, ""
        return False, "", "invalid-json"
    if not isinstance(payload, dict):
        return False, "", "invalid-payload"
    if payload.get("ok") is False:
        return False, "", str(payload.get("error") or payload.get("message") or "system-error")

    candidates: List[str] = []
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ["reply", "transformerReply", "graphReply"]:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    for key in ["reply", "text", "answer"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    for reply in candidates:
        if reply:
            return True, reply, ""
    return False, "", "empty-reply"


def parse_ollama_reply(raw: str) -> Tuple[bool, str, str]:
    try:
        payload = json.loads(raw)
    except Exception:
        return False, "", "invalid-json"
    if not isinstance(payload, dict):
        return False, "", "invalid-payload"

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True, content.strip(), ""

    response = payload.get("response")
    if isinstance(response, str) and response.strip():
        return True, response.strip(), ""

    return False, "", str(payload.get("error", "empty-reply"))


@dataclass
class EvalResult:
    case_id: str
    task_type: str
    ok: bool
    status: int
    latency_ms: float
    prompt: str
    reply: str
    reference_type: str
    score: float
    score_detail: Dict[str, float]
    threshold: float = 0.0
    quality_pass: bool = False
    reply_chars: int = 0
    error: str = ""
    ollama_score: float = -1.0
    ollama_reply: str = ""
    ollama_error: str = ""


def infer_hai_dimensions(case: Dict[str, Any], task_type: str) -> List[str]:
    explicit = case.get("haiDimensions")
    if isinstance(explicit, list):
        return [str(item).strip() for item in explicit if str(item).strip()]
    if isinstance(explicit, str) and explicit.strip():
        return [explicit.strip()]
    return TASK_TYPE_TO_HAI.get(task_type, ["instruction_following"])


def compute_hai_summary(results: List[EvalResult], cases_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    dimension_buckets: Dict[str, List[float]] = {dimension: [] for dimension in HAI_DIMENSIONS}
    for item in results:
        if not item.ok:
            continue
        case = cases_by_id.get(item.case_id, {})
        for dimension in infer_hai_dimensions(case, item.task_type):
            if dimension not in dimension_buckets:
                dimension_buckets[dimension] = []
            dimension_buckets[dimension].append(item.score)

    populated = {dimension: values for dimension, values in dimension_buckets.items() if values}
    dimensions = {
        dimension: {
            "count": len(values),
            "scoreAvg": round(statistics.fmean(values), 2),
        }
        for dimension, values in populated.items()
    }
    overall = round(statistics.fmean(bucket["scoreAvg"] for bucket in dimensions.values()), 2) if dimensions else 0.0
    coverage = round(len(dimensions) / len(HAI_DIMENSIONS) * 100.0, 2) if HAI_DIMENSIONS else 0.0
    return {
        "overall": overall,
        "coverage": coverage,
        "dimensions": dimensions,
    }


def load_cases(file_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("cases file must be a JSON array")
    return payload


def run_system_case(endpoint: str, case: Dict[str, Any], timeout_s: float, system_token: str, max_tokens: int) -> EvalResult:
    prompt = str(case.get("input", ""))
    case_id = str(case.get("id", "unknown"))
    task_type = str(case.get("taskType", "unknown"))
    scoring = case.get("scoring", {}) if isinstance(case.get("scoring"), dict) else {}
    threshold = float(scoring.get("threshold", 0.0) or 0.0)
    payload = {
        "text": prompt,
        "sessionId": f"intel-{case_id}-{int(time.time() * 1000)}",
        "maxTokens": max(64, max_tokens),
    }
    headers = {"Authorization": f"Bearer {system_token}"} if system_token else None
    start = time.perf_counter()
    status, raw = post_json(endpoint, payload, timeout_s, headers=headers)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if status == 0:
        return EvalResult(case_id, task_type, False, status, latency_ms, prompt, "", "unknown", 0.0, {}, threshold=threshold, quality_pass=False, reply_chars=0, error=raw)

    ok, reply, error = parse_system_reply(raw)
    if not ok:
        return EvalResult(case_id, task_type, False, status, latency_ms, prompt, "", "unknown", 0.0, {}, threshold=threshold, quality_pass=False, reply_chars=0, error=error)

    reference = case.get("reference", "")
    if isinstance(reference, str):
        score, detail = score_text(reply, reference)
        reference_type = "text"
    elif isinstance(reference, dict):
        score, detail = score_structured(reply, reference)
        reference_type = "structured"
    else:
        score, detail = 0.0, {"final": 0.0}
        reference_type = "unknown"

    score = round(score, 2)
    quality_pass = (score >= threshold) if threshold > 0.0 else True
    return EvalResult(
        case_id,
        task_type,
        True,
        status,
        latency_ms,
        prompt,
        reply,
        reference_type,
        score,
        detail,
        threshold=threshold,
        quality_pass=quality_pass,
        reply_chars=len(reply),
    )


def run_ollama_case(ollama_url: str, model: str, prompt: str, timeout_s: float, max_tokens: int) -> Tuple[bool, str, str, float]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": max(64, max_tokens), "temperature": 0.2, "top_p": 0.9},
    }
    status, raw = post_json(ollama_url, payload, timeout_s)
    if status == 0:
        return False, "", raw, 0.0
    ok, reply, error = parse_ollama_reply(raw)
    return ok, reply, error, float(status)


def warmup_system(endpoint: str, timeout_s: float, system_token: str) -> None:
    warmup_endpoint = endpoint
    payload = {
        "text": "warmup",
        "sessionId": f"intel-warmup-{int(time.time() * 1000)}",
        "maxTokens": 16,
    }
    headers = {"Authorization": f"Bearer {system_token}"} if system_token else None
    warmup_timeout = max(timeout_s, 180.0)
    status, raw = post_json(warmup_endpoint, payload, warmup_timeout, headers=headers)
    if status == 0:
        print(f"[WARMUP] system warmup failed: {raw}")
        return
    ok, _reply, error = parse_system_reply(raw)
    if ok:
        print(f"[WARMUP] system warmup complete via {warmup_endpoint}")
    else:
        print(f"[WARMUP] system warmup returned no reply: {error}")


def summarize(results: List[EvalResult]) -> Dict[str, Any]:
    total = len(results)
    ok_results = [item for item in results if item.ok]
    failed_results = [item for item in results if not item.ok]
    latencies = [item.latency_ms for item in ok_results]
    scores = [item.score for item in ok_results]
    quality_passed = [item for item in ok_results if item.quality_pass]
    reply_chars = [item.reply_chars for item in ok_results]
    ollama_scores = [item.ollama_score for item in ok_results if item.ollama_score >= 0.0]
    score_deltas = [item.score - item.ollama_score for item in ok_results if item.ollama_score >= 0.0]

    task_type_summary: Dict[str, Any] = {}
    for item in results:
        bucket = task_type_summary.setdefault(
            item.task_type,
            {
                "total": 0,
                "transportPassed": 0,
                "qualityPassed": 0,
                "scoreAvg": 0.0,
                "latencyAvgMs": 0.0,
            },
        )
        bucket["total"] += 1
        if item.ok:
            bucket["transportPassed"] += 1
            bucket["qualityPassed"] += int(item.quality_pass)

    for task_type, bucket in task_type_summary.items():
        task_results = [item for item in ok_results if item.task_type == task_type]
        task_scores = [item.score for item in task_results]
        task_latencies = [item.latency_ms for item in task_results]
        bucket["scoreAvg"] = round(statistics.fmean(task_scores), 2) if task_scores else 0.0
        bucket["latencyAvgMs"] = round(statistics.fmean(task_latencies), 2) if task_latencies else 0.0
        bucket["qualityPassRate"] = round((bucket["qualityPassed"] / bucket["transportPassed"] * 100.0), 2) if bucket["transportPassed"] else 0.0

    return {
        "total": total,
        "transportPassed": len(ok_results),
        "transportFailed": len(failed_results),
        "transportPassRate": round((len(ok_results) / total * 100.0), 2) if total else 0.0,
        "qualityPassed": len(quality_passed),
        "qualityPassRate": round((len(quality_passed) / len(ok_results) * 100.0), 2) if ok_results else 0.0,
        "latencyAvgMs": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latencyP50Ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latencyP90Ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.9) - 1)], 2) if latencies else 0.0,
        "latencyMaxMs": round(max(latencies), 2) if latencies else 0.0,
        "latencyStdMs": round(statistics.pstdev(latencies), 2) if len(latencies) >= 2 else 0.0,
        "scoreAvg": round(statistics.fmean(scores), 2) if scores else 0.0,
        "scoreP50": round(statistics.median(scores), 2) if scores else 0.0,
        "scoreP90": round(sorted(scores)[max(0, int(len(scores) * 0.9) - 1)], 2) if scores else 0.0,
        "scoreMin": round(min(scores), 2) if scores else 0.0,
        "scoreMax": round(max(scores), 2) if scores else 0.0,
        "scoreStd": round(statistics.pstdev(scores), 2) if len(scores) >= 2 else 0.0,
        "replyCharsAvg": round(statistics.fmean(reply_chars), 2) if reply_chars else 0.0,
        "replyCharsP50": round(statistics.median(reply_chars), 2) if reply_chars else 0.0,
        "ollamaScoreAvg": round(statistics.fmean(ollama_scores), 2) if ollama_scores else -1.0,
        "scoreDeltaVsOllamaAvg": round(statistics.fmean(score_deltas), 2) if score_deltas else 0.0,
        "taskTypes": task_type_summary,
        "failures": [asdict(item) for item in failed_results],
    }


def build_markdown(summary: Dict[str, Any], hai: Dict[str, Any], gates: Dict[str, Any], results: List[EvalResult], system_url: str, ollama_url: str, cases_file: str) -> str:
    lines = [
        "# Intelligence Eval Report",
        "",
        f"- system: {system_url}",
        f"- ollama: {ollama_url or 'disabled'}",
        f"- cases: {cases_file}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total | {summary['total']} |",
        f"| transportPassed | {summary['transportPassed']} |",
        f"| transportFailed | {summary['transportFailed']} |",
        f"| transportPassRate | {summary['transportPassRate']}% |",
        f"| qualityPassed | {summary['qualityPassed']} |",
        f"| qualityPassRate | {summary['qualityPassRate']}% |",
        f"| latencyAvgMs | {summary['latencyAvgMs']} |",
        f"| latencyP50Ms | {summary['latencyP50Ms']} |",
        f"| latencyP90Ms | {summary['latencyP90Ms']} |",
        f"| latencyMaxMs | {summary['latencyMaxMs']} |",
        f"| latencyStdMs | {summary['latencyStdMs']} |",
        f"| scoreAvg | {summary['scoreAvg']} |",
        f"| scoreP50 | {summary['scoreP50']} |",
        f"| scoreP90 | {summary['scoreP90']} |",
        f"| scoreMin | {summary['scoreMin']} |",
        f"| scoreMax | {summary['scoreMax']} |",
        f"| scoreStd | {summary['scoreStd']} |",
        f"| replyCharsAvg | {summary['replyCharsAvg']} |",
        f"| replyCharsP50 | {summary['replyCharsP50']} |",
    ]
    if summary["ollamaScoreAvg"] >= 0.0:
        lines.append(f"| ollamaScoreAvg | {summary['ollamaScoreAvg']} |")
        lines.append(f"| scoreDeltaVsOllamaAvg | {summary['scoreDeltaVsOllamaAvg']} |")
    lines.extend([
        "",
        "## HAI",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| overall | {hai['overall']} |",
        f"| coverage | {hai['coverage']}% |",
        f"| gatesPassed | {str(gates['passed']).lower()} |",
    ])
    if hai["dimensions"]:
        lines.extend([
            "",
            "| dimension | count | scoreAvg |",
            "|---|---:|---:|",
        ])
        for dimension, bucket in sorted(hai["dimensions"].items()):
            lines.append(f"| {dimension} | {bucket['count']} | {bucket['scoreAvg']} |")
    lines.extend([
        "",
        "## By Task Type",
        "",
        "| taskType | total | transportPassed | qualityPassed | qualityPassRate | scoreAvg | latencyAvgMs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for task_type, bucket in sorted(summary["taskTypes"].items()):
        lines.append(
            f"| {task_type} | {bucket['total']} | {bucket['transportPassed']} | {bucket['qualityPassed']} | {bucket['qualityPassRate']}% | {bucket['scoreAvg']} | {bucket['latencyAvgMs']} |"
        )
    lines.extend([
        "",
        "## Details",
        "",
        "| id | taskType | ok | qualityPass | latencyMs | score | threshold | ollamaScore | replyChars | prompt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for item in results:
        prompt = item.prompt.replace("\n", " ")[:60]
        ollama_score = "-" if item.ollama_score < 0.0 else f"{item.ollama_score:.2f}"
        lines.append(
            f"| {item.case_id} | {item.task_type} | {str(item.ok).lower()} | {str(item.quality_pass).lower()} | {item.latency_ms:.2f} | {item.score:.2f} | {item.threshold:.2f} | {ollama_score} | {item.reply_chars} | {prompt} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intelligence evaluation against phoenix_main and optional direct Ollama")
    parser.add_argument("--system-url", default="http://127.0.0.1:5080/api/chat")
    parser.add_argument("--cases-file", default="test/intelligence/cases.quick.json")
    parser.add_argument("--output-json", default="build/intelligence_eval_report.json")
    parser.add_argument("--output-md", default="build/intelligence_eval_report.md")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--ollama-url", default="")
    parser.add_argument("--ollama-model", default="")
    parser.add_argument("--system-token", default="")
    parser.add_argument("--min-quality-pass-rate", type=float, default=0.0)
    parser.add_argument("--min-score-avg", type=float, default=0.0)
    parser.add_argument("--min-hai-score", type=float, default=0.0)
    parser.add_argument("--min-hai-coverage", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(Path(args.cases_file))
    cases_by_id = {str(case.get("id", "")): case for case in cases}
    results: List[EvalResult] = []
    warmup_system(args.system_url, args.timeout, args.system_token)
    ollama_model = (args.ollama_model or "").strip()
    if args.ollama_url and not ollama_model:
        ollama_model = discover_ollama_model(args.ollama_url, args.timeout)
        if ollama_model:
            print(f"[OLLAMA] discovered model: {ollama_model}")

    for case in cases:
        result = run_system_case(args.system_url, case, args.timeout, args.system_token, args.max_tokens)
        reference = case.get("reference", "")
        if args.ollama_url and ollama_model:
            ok, reply, error, _status = run_ollama_case(args.ollama_url, ollama_model, result.prompt, args.timeout, args.max_tokens)
            if ok:
                result.ollama_reply = reply
                if isinstance(reference, str):
                    ollama_score, _detail = score_text(reply, reference)
                    result.ollama_score = round(ollama_score, 2)
                elif isinstance(reference, dict):
                    ollama_score, _detail = score_structured(reply, reference)
                    result.ollama_score = round(ollama_score, 2)
            else:
                result.ollama_error = error
        results.append(result)

    summary = summarize(results)
    hai = compute_hai_summary(results, cases_by_id)
    gates = {
        "transport": summary["transportFailed"] == 0,
        "qualityPassRate": summary["qualityPassRate"] >= args.min_quality_pass_rate,
        "scoreAvg": summary["scoreAvg"] >= args.min_score_avg,
        "haiScore": hai["overall"] >= args.min_hai_score,
        "haiCoverage": hai["coverage"] >= args.min_hai_coverage,
    }
    gates["passed"] = all(gates.values())
    report = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "systemUrl": args.system_url,
        "ollamaUrl": args.ollama_url,
        "ollamaModel": ollama_model,
        "casesFile": args.cases_file,
        "summary": summary,
        "hai": hai,
        "gates": gates,
        "results": [asdict(item) for item in results],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(summary, hai, gates, results, args.system_url, args.ollama_url, args.cases_file), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gates["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
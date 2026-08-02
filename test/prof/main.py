#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import math
import random
import re
import shutil
import socket
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib import error, request
from urllib.parse import urlencode, urlparse, urlunparse


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


httpx = _optional_import("httpx")
np = _optional_import("numpy")
scipy_mod = _optional_import("scipy")
scipy_stats = getattr(scipy_mod, "stats", None) if scipy_mod is not None else None
psutil = _optional_import("psutil")
rouge_score_mod = _optional_import("rouge_score")
rouge_scorer = getattr(rouge_score_mod, "rouge_scorer", None) if rouge_score_mod is not None else None
sacrebleu = _optional_import("sacrebleu")


SYSTEM_RECOVERY_LOCK = threading.Lock()


DEFAULT_SYSTEM_URL = "http://127.0.0.1:5080/api/chat"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8082/v1/chat/completions"
DEFAULT_CASES_FILE = "test/intelligence/cases.baseline.json"
DEFAULT_EXTERNAL_PROMPTS_FILE = "questionaire.txt"
DEFAULT_EXTERNAL_ANSWERS_FILE = "answer_20260217-175925-v2.0Multi.txt"
DEFAULT_EXTERNAL_INDEX_FILE = "doc/external_dataset_index.json"
DEFAULT_QUESTIONNAIRE_FILE = "questionaire.txt"
DEFAULT_QUESTIONNAIRE_GLOBS = ["tests/*.txt"]
DEFAULT_QUESTIONNAIRE_LIMIT = 1000
DEFAULT_EXTERNAL_LIMIT = 20
DEFAULT_TESTS_DATASET_LIMIT = 1000
DEFAULT_TIMEOUT = 60.0
DEFAULT_SYSTEM_START_TIMEOUT = 30.0
DEFAULT_OLLAMA_WARMUP_TIMEOUT = 180.0
DEFAULT_CHECKPOINT_EVERY = 100
DEFAULT_STABILITY_CHECK_INTERVAL = 100
DEFAULT_STABILITY_MIN_SAMPLES = 200
DEFAULT_STABILITY_WINDOW = 3
DEFAULT_STABILITY_QUALITY_DELTA = 0.75
DEFAULT_STABILITY_BALANCED_DELTA = 0.75
DEFAULT_STABILITY_SUCCESS_RATE_DELTA = 0.25
DEFAULT_STABILITY_LATENCY_RATIO = 0.05
DEFAULT_STABILITY_LATENCY_DELTA_MS = 200.0
DEFAULT_BENCHMARK_CACHE_DIR = "runtime_store/prof_cache"
DEFAULT_BENCHMARK_LIMIT_PER_PRESET = 128
DEFAULT_DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_PREFERRED_MODELS = [
    "llama3.1:8b",
    "llama3.1:latest",
    "qwen2.5:7b",
    "tinyllama:latest",
    "qwen2.5:14b",
    "gpt-oss:20b",
]
DEFAULT_STANDARD_BENCHMARKS = [
    "gsm8k-main-test",
    "ai2-arc-challenge-test",
    "hellaswag-validation",
    "winogrande-xl-validation",
]

ROUTE_SYSTEM = "system"
ROUTE_OLLAMA = "ollama"
ROUTE_LLAMACPP = "llamacpp"


def infer_llamacpp_style(path: str) -> str:
    normalized = (path or "").rstrip("/") or "/"
    if normalized in {"/v1/chat/completions", "/chat/completions"}:
        return "openai-chat"
    if normalized == "/completion":
        return "llama-completion"
    return "ollama-chat"


def parse_url_base_path(url: str) -> Tuple[str, str]:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc
    if not netloc and parsed.path and "://" not in parsed.path:
        netloc = parsed.path
        parsed = urlparse(f"{scheme}://{netloc}")
    base = urlunparse((scheme, netloc, "", "", "", ""))
    path = (parsed.path or "").rstrip("/")
    return base, path


def build_llamacpp_request_candidates(url: str) -> List[Dict[str, str]]:
    base_url, requested_path = parse_url_base_path(url)
    requested_path = requested_path or ""
    candidates: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def add(path: str, source: str) -> None:
        normalized_path = path if path.startswith("/") else f"/{path}"
        full_url = f"{base_url}{normalized_path}"
        if full_url in seen:
            return
        seen.add(full_url)
        candidates.append({
            "url": full_url,
            "path": normalized_path,
            "style": infer_llamacpp_style(normalized_path),
            "source": source,
        })

    if requested_path and requested_path != "/":
        add(requested_path, "requested")
        if requested_path == "/api/chat":
            add("/v1/chat/completions", "fallback-openai")
            add("/chat/completions", "fallback-openai")
            add("/completion", "fallback-legacy")
    else:
        add("/v1/chat/completions", "default-openai")
        add("/chat/completions", "default-openai")
        add("/completion", "default-legacy")
        add("/api/chat", "default-adapter")
    return candidates


def llamacpp_label(url: str) -> str:
    _base, path = parse_url_base_path(url)
    return path or "/"


def build_direct_chat_payload(route_name: str,
                              prompt: str,
                              model: str,
                              max_tokens: int,
                              endpoint_style: str = "ollama-chat") -> Dict[str, Any]:
    token_limit = max(16, int(max_tokens))
    if route_name == ROUTE_LLAMACPP and endpoint_style == "openai-chat":
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": token_limit,
            "stream": False,
        }
        if model.strip():
            payload["model"] = model.strip()
        return payload
    if route_name == ROUTE_LLAMACPP and endpoint_style == "llama-completion":
        payload = {
            "prompt": prompt,
            "n_predict": token_limit,
            "stream": False,
        }
        if model.strip():
            payload["model"] = model.strip()
        return payload
    return {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"num_predict": token_limit},
    }


def resolve_llamacpp_endpoint(llamacpp_url: str, timeout_s: float) -> Dict[str, Any]:
    parsed = urlparse(llamacpp_url.strip())
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    requested_base, requested_path = parse_url_base_path(llamacpp_url)
    base_candidates: List[Tuple[str, str]] = [(requested_base, "requested")]
    if host in {"127.0.0.1", "localhost"} and port == 8080:
        alt_base = urlunparse((scheme, f"{host}:8082", "", "", "", ""))
        if alt_base != requested_base:
            base_candidates.append((alt_base, "fallback-port-8082"))

    probe_timeout = max(1.0, min(float(timeout_s), 5.0))
    errors: List[str] = []

    for base_url, source in base_candidates:
        base_parsed = urlparse(base_url)
        probe_host = base_parsed.hostname or host
        probe_port = base_parsed.port or (443 if (base_parsed.scheme or scheme) == "https" else 80)
        port_open = is_tcp_port_open(probe_host, probe_port, probe_timeout)
        health_ok = False
        for health_path in ["/health", "/v1/models"]:
            ok, _payload, err = get_json(f"{base_url}{health_path}", probe_timeout)
            if ok:
                health_ok = True
                break
            if err:
                errors.append(f"{base_url}{health_path}: {err}")

        candidates = build_llamacpp_request_candidates(base_url + (requested_path or ""))
        if health_ok:
            for candidate in candidates:
                if candidate["style"] in {"openai-chat", "llama-completion"}:
                    note = ""
                    if source != "requested":
                        note = f"auto-switched-port:{port}->{probe_port}"
                    elif requested_path == "/api/chat":
                        note = "auto-switched-protocol:/api/chat->/v1/chat/completions"
                    return {
                        "ready": True,
                        "url": candidate["url"],
                        "style": candidate["style"],
                        "base_url": base_url,
                        "note": note,
                    }
        if port_open and requested_path == "/api/chat" and source == "requested":
            return {
                "ready": True,
                "url": f"{base_url}/api/chat",
                "style": "ollama-chat",
                "base_url": base_url,
                "note": "adapter-endpoint-assumed",
            }
        if not port_open:
            errors.append(f"tcp-port-closed:{probe_host}:{probe_port}")

    hint = ""
    if host in {"127.0.0.1", "localhost"} and port == 8080:
        hint = " ; repo-default-llamacpp-port=8082 ; standard-endpoint=/v1/chat/completions"
    return {
        "ready": False,
        "url": llamacpp_url,
        "style": infer_llamacpp_style(requested_path),
        "base_url": requested_base,
        "note": "",
        "error": " | ".join(dict.fromkeys(errors)) + hint,
    }


def enabled_route_names(args: argparse.Namespace) -> List[str]:
    names = [ROUTE_SYSTEM, ROUTE_OLLAMA]
    if getattr(args, "enable_llamacpp", False):
        names.append(ROUTE_LLAMACPP)
    return names


@dataclass
class BenchCase:
    case_id: str
    suite: str
    task_type: str
    prompt: str
    reference: Optional[Any]
    threshold: float
    tags: List[str]


@dataclass
class RouteCaseResult:
    route: str
    round_index: int
    case_id: str
    suite: str
    task_type: str
    prompt: str
    ok: bool
    latency_ms: float
    speed_score: float
    status: int
    error: str
    reply: str
    quality_score: float = -1.0
    quality_detail: Optional[Dict[str, float]] = None
    quality_pass: bool = False
    threshold: float = 0.0


@dataclass
class BenchDeps:
    http_backend: str
    httpx_available: bool
    numpy_available: bool
    scipy_available: bool
    psutil_available: bool
    rouge_available: bool
    sacrebleu_available: bool


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def resolve_report_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    default_dir = Path(__file__).resolve().parent / "reports"
    if args.resume and not args.output and not args.json_output:
        out_path = default_dir / "resume_latest.md"
        json_out_path = default_dir / "resume_latest.json"
        return out_path, json_out_path
    out_path = Path(args.output) if args.output else default_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    json_out_path = Path(args.json_output) if args.json_output else out_path.with_suffix(".json")
    return out_path, json_out_path


def normalize_preset_names(preset_names: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in preset_names:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def load_resume_payload(json_out_path: Path, resume: bool) -> Dict[str, Any]:
    if not resume or not json_out_path.exists():
        return {}
    try:
        payload = json.loads(json_out_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def restore_route_payloads(resume_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    route_payloads: Dict[str, Dict[str, Any]] = {
        ROUTE_SYSTEM: {"results": [], "wall_ms": 0.0},
        ROUTE_OLLAMA: {"results": [], "wall_ms": 0.0},
        ROUTE_LLAMACPP: {"results": [], "wall_ms": 0.0},
    }
    routes = resume_payload.get("routes", {}) if isinstance(resume_payload.get("routes"), dict) else {}
    for route_name in [ROUTE_SYSTEM, ROUTE_OLLAMA, ROUTE_LLAMACPP]:
        route_doc = routes.get(route_name, {}) if isinstance(routes.get(route_name), dict) else {}
        route_payloads[route_name]["results"] = [
            route_result_from_dict(item)
            for item in route_doc.get("results", [])
            if isinstance(item, dict)
        ]
        route_payloads[route_name]["wall_ms"] = float(route_doc.get("wall_ms", 0.0) or 0.0)
    return route_payloads


def build_existing_result_keys(results: List[RouteCaseResult]) -> Set[Tuple[int, str]]:
    return {(item.round_index, item.case_id) for item in results}


def materialize_route_payloads(route_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for route_name, state in route_payloads.items():
        results = list(state.get("results", []))
        wall_ms = float(state.get("wall_ms", 0.0) or 0.0)
        out[route_name] = {
            "results": [result_to_dict(item) for item in results],
            "wall_ms": round(wall_ms, 2),
            "summary": summarize_route(results),
        }
    return out


def expected_case_count_for_route(args: argparse.Namespace, all_cases: List[BenchCase]) -> int:
    default_per_round = len(all_cases)
    planned_case_counts = getattr(args, "planned_case_counts_by_round", {}) or {}
    return sum(int(planned_case_counts.get(round_index, default_per_round) or 0) for round_index in range(1, max(1, int(args.rounds)) + 1))


def route_has_pending_results(route_name: str,
                              args: argparse.Namespace,
                              all_cases: List[BenchCase],
                              route_payloads: Dict[str, Dict[str, Any]]) -> bool:
    expected = expected_case_count_for_route(args, all_cases)
    done = len(route_payloads.get(route_name, {}).get("results", []))
    return done < expected


def build_progress_state(args: argparse.Namespace,
                         all_cases: List[BenchCase],
                         route_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    planned_case_counts = getattr(args, "planned_case_counts_by_round", {}) or {}
    expected_per_route = expected_case_count_for_route(args, all_cases)
    routes: Dict[str, Any] = {}
    complete = True
    for route_name in enabled_route_names(args):
        done = len(route_payloads.get(route_name, {}).get("results", []))
        route_done = done >= expected_per_route if expected_per_route > 0 else True
        routes[route_name] = {
            "completed": done,
            "expected": expected_per_route,
            "complete": route_done,
        }
        complete = complete and route_done
    return {
        "resume": bool(args.resume),
        "checkpointEvery": int(args.checkpoint_every),
        "complete": complete,
        "plannedCaseCountsByRound": planned_case_counts,
        "routes": routes,
    }


def persist_report_artifacts(args: argparse.Namespace,
                             out_path: Path,
                             json_out_path: Path,
                             quality_cases: List[BenchCase],
                             standard_cases: List[BenchCase],
                             external_cases: List[BenchCase],
                             questionnaire_cases: List[BenchCase],
                             route_payloads: Dict[str, Dict[str, Any]],
                             deps: BenchDeps,
                             system_management: Dict[str, Any],
                             ollama_management: Dict[str, Any],
                             external_index: Dict[str, Any],
                             benchmark_metadata: Dict[str, Any],
                             local_dataset_metadata: Dict[str, Any]) -> None:
    materialized_routes = materialize_route_payloads(route_payloads)
    system_results = list(route_payloads[ROUTE_SYSTEM].get("results", []))
    ollama_results = list(route_payloads[ROUTE_OLLAMA].get("results", []))
    llamacpp_results = list(route_payloads[ROUTE_LLAMACPP].get("results", []))
    comparisons = {
        "latency_significance": compare_latency_significance(system_results, ollama_results),
        "quality_delta": compare_quality_delta(system_results, ollama_results),
    }
    if getattr(args, "enable_llamacpp", False) or llamacpp_results:
        comparisons["llamacpp_vs_ollama"] = {
            "latency_significance": compare_latency_significance(llamacpp_results, ollama_results),
            "quality_delta": compare_quality_delta(llamacpp_results, ollama_results),
        }
    progress_state = build_progress_state(args, quality_cases + standard_cases + external_cases + questionnaire_cases, route_payloads)
    report = build_report(
        args,
        quality_cases,
        standard_cases,
        external_cases,
        questionnaire_cases,
        materialized_routes,
        comparisons,
        deps,
        system_management,
        ollama_management,
        external_index,
        benchmark_metadata,
        local_dataset_metadata,
        progress_state,
    )
    payload = build_report_payload(
        args,
        quality_cases,
        standard_cases,
        external_cases,
        questionnaire_cases,
        materialized_routes,
        comparisons,
        deps,
        system_management,
        ollama_management,
        external_index,
        benchmark_metadata,
        local_dataset_metadata,
        progress_state,
    )
    atomic_write_text(out_path, report)
    atomic_write_text(json_out_path, json.dumps(payload, ensure_ascii=False, indent=2))


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    idx = (len(values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - idx) + values[hi] * (idx - lo))


def bootstrap_ci_mean(values: List[float], bootstrap_rounds: int = 600) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    if np is None:
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) if len(values) > 1 else 0.0
        margin = 1.96 * sd / math.sqrt(max(1, len(values)))
        return max(0.0, mean - margin), mean + margin
    arr = np.array(values, dtype=float)
    n = len(values)
    means = []
    for _ in range(max(200, bootstrap_rounds)):
        idx = np.random.randint(0, n, size=n)
        means.append(float(np.mean(arr[idx])))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text.strip() if not ch.isspace())


def char_bigrams(text: str) -> List[str]:
    if len(text) <= 1:
        return [text] if text else []
    return [text[i:i + 2] for i in range(len(text) - 1)]


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
        seq_score = statistics.fmean([
            difflib_similarity(normalized_reply, clause),
            difflib_similarity(clause, normalized_reply),
        ])
        scores.append(max(gram_score, seq_score))
    return sum(scores) / len(scores)


def difflib_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, left, right).ratio()
    except Exception:
        return 0.0


def score_text_basic(reply: str, reference: str) -> float:
    a = normalize_text(reply)
    b = normalize_text(reference)
    if not a or not b:
        return 0.0
    grams_a = set(char_bigrams(a))
    grams_b = set(char_bigrams(b))
    jaccard = len(grams_a & grams_b) / max(1, len(grams_a | grams_b)) if grams_a and grams_b else 0.0
    keywords = split_keywords(reference)
    recall = (sum(1 for kw in keywords if kw and kw in a) / len(keywords)) if keywords else 0.0
    sequence = difflib_similarity(a, b)
    clause_coverage = compute_clause_coverage(a, reference)
    contains_reference = 1.0 if b in a else 0.0
    return max(0.0, min(100.0, jaccard * 10.0 + sequence * 15.0 + recall * 25.0 + clause_coverage * 35.0 + contains_reference * 15.0))


def term_frequency_bigrams(text: str) -> Dict[str, float]:
    grams = char_bigrams(normalize_text(text))
    freq: Dict[str, float] = {}
    for gram in grams:
        if not gram:
            continue
        freq[gram] = freq.get(gram, 0.0) + 1.0
    return freq


def cosine_similarity_score(reply: str, reference: str) -> float:
    left = term_frequency_bigrams(reply)
    right = term_frequency_bigrams(reference)
    if not left or not right:
        return 0.0
    dot = 0.0
    for key, value in left.items():
        dot += value * right.get(key, 0.0)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(100.0, (dot / (left_norm * right_norm)) * 100.0))


def score_text(reply: str, reference: str) -> Tuple[float, Dict[str, float]]:
    base = score_text_basic(reply, reference)
    cosine = cosine_similarity_score(reply, reference)
    detail: Dict[str, float] = {"basic": round(base, 2), "cosine": round(cosine, 2)}
    fused = base * 0.40 + cosine * 0.20
    extra_weight = 0.60

    if rouge_scorer is not None and reply.strip() and reference.strip():
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            rouge_l = float(scorer.score(reference, reply)["rougeL"].fmeasure) * 100.0
            detail["rougeL_f1"] = round(rouge_l, 2)
            fused += rouge_l * 0.15
            extra_weight += 0.15
        except Exception:
            pass

    if sacrebleu is not None and reply.strip() and reference.strip():
        try:
            chrf = float(sacrebleu.sentence_chrf(reply, [reference]).score)
            bleu = float(sacrebleu.sentence_bleu(reply, [reference]).score)
            detail["chrf"] = round(chrf, 2)
            detail["bleu"] = round(bleu, 2)
            fused += chrf * 0.15 + bleu * 0.10
            extra_weight += 0.25
        except Exception:
            pass

    if extra_weight < 1.0:
        fallback = (base + cosine) * 0.5
        fused += fallback * (1.0 - extra_weight)

    final = max(0.0, min(100.0, fused))
    detail["fused"] = round(final, 2)
    return final, detail


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


def compute_speed_score(latency_ms: float, timeout_s: float, ok: bool) -> float:
    if not ok:
        return 0.0
    ceiling = max(timeout_s * 1000.0, 1.0)
    return round(max(0.0, min(100.0, (1.0 - (latency_ms / ceiling)) * 100.0)), 2)


def post_json(url: str, payload: Dict[str, Any], timeout_s: float, headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.getcode()), resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = str(exc)
        return int(exc.code), raw
    except Exception as exc:
        return 0, str(exc)


def post_json_httpx(url: str,
                    payload: Dict[str, Any],
                    timeout_s: float,
                    headers: Optional[Dict[str, str]] = None,
                    client: Optional[Any] = None) -> Tuple[int, str]:
    if httpx is None:
        return 0, "httpx-not-installed"
    local_client = client
    close_after = False
    if local_client is None:
        local_client = httpx.Client(timeout=timeout_s)
        close_after = True
    try:
        resp = local_client.post(url, content=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json", **(headers or {})})
        return int(resp.status_code), resp.text
    except Exception as exc:
        return 0, str(exc)
    finally:
        if close_after:
            try:
                local_client.close()
            except Exception:
                pass


def get_json_httpx(url: str,
                   timeout_s: float,
                   headers: Optional[Dict[str, str]] = None,
                   client: Optional[Any] = None) -> Tuple[bool, Dict[str, Any], str]:
    if httpx is None:
        return False, {}, "httpx-not-installed"
    local_client = client
    close_after = False
    if local_client is None:
        local_client = httpx.Client(timeout=timeout_s, follow_redirects=True)
        close_after = True
    try:
        resp = local_client.get(url, headers=headers or {})
        if resp.status_code < 200 or resp.status_code >= 300:
            return False, {}, f"http-{resp.status_code}"
        payload = resp.json()
        if isinstance(payload, dict):
            return True, payload, ""
        return False, {}, "invalid-json-payload"
    except Exception as exc:
        return False, {}, str(exc)
    finally:
        if close_after:
            try:
                local_client.close()
            except Exception:
                pass


def get_json(url: str, timeout_s: float, headers: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, Any], str]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "phoenix-prof-benchmark/6.0",
    }
    if headers:
        request_headers.update(headers)

    if httpx is not None:
        ok, payload, err = get_json_httpx(url, timeout_s, request_headers)
        if ok:
            return True, payload, ""

    req = request.Request(url, method="GET")
    for key, value in request_headers.items():
            req.add_header(key, value)
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            if isinstance(payload, dict):
                return True, payload, ""
            return False, {}, "invalid-json-payload"
    except Exception as exc:
        return False, {}, str(exc)


def extract_token_from_doc(doc: Dict[str, Any]) -> str:
    candidates = [doc.get("token"), doc.get("access_token"), doc.get("accessToken"), doc.get("jwt")]
    if isinstance(doc.get("result"), dict):
        result = doc["result"]
        candidates.extend([result.get("token"), result.get("access_token"), result.get("accessToken"), result.get("jwt")])
    if isinstance(doc.get("data"), dict):
        data = doc["data"]
        candidates.extend([data.get("token"), data.get("access_token"), data.get("accessToken"), data.get("jwt")])
    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def parse_login_token(raw: str) -> Tuple[bool, str, str]:
    try:
        doc = json.loads(raw)
    except Exception:
        return False, "", "invalid-json"
    if not isinstance(doc, dict):
        return False, "", "invalid-payload"
    token = extract_token_from_doc(doc)
    if token:
        return True, token, ""
    return False, "", str(doc.get("error") or doc.get("message") or "token-not-found")


def parse_auth_response(raw: str) -> Tuple[str, str]:
    try:
        doc = json.loads(raw)
    except Exception:
        return "", ""
    if not isinstance(doc, dict):
        return "", ""
    token = extract_token_from_doc(doc)
    verify_token = ""
    candidates = [doc.get("verifyToken"), doc.get("verify_token")]
    if isinstance(doc.get("result"), dict):
        candidates.extend([doc["result"].get("verifyToken"), doc["result"].get("verify_token")])
    for item in candidates:
        if isinstance(item, str) and item.strip():
            verify_token = item.strip()
            break
    return token, verify_token


def build_auth_candidates(system_url: str, relative_paths: List[str], explicit_url: str = "") -> List[str]:
    if explicit_url.strip():
        return [explicit_url.strip()]
    parsed = urlparse(system_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port

    netlocs: List[str] = []
    if port:
        netlocs.append(f"{host}:{port}")
        if port == 5080:
            netlocs.append(f"{host}:5081")
        elif port == 5081:
            netlocs.append(f"{host}:5080")
    else:
        netlocs.append(host)

    out: List[str] = []
    seen = set()
    for netloc in netlocs:
        for rel in relative_paths:
            url = urlunparse((scheme, netloc, rel, "", "", ""))
            if url not in seen:
                seen.add(url)
                out.append(url)
    return out


def build_login_candidates(system_url: str, login_url: str) -> List[str]:
    return build_auth_candidates(system_url, ["/auth/login", "/api/auth/login"], login_url)


def login_and_get_token(system_url: str,
                        login_url: str,
                        username: str,
                        password: str,
                        timeout_s: float) -> Tuple[bool, str, str]:
    urls = build_login_candidates(system_url, login_url)
    last_err = ""
    for url in urls:
        status, raw = post_json(url, {"username": username, "password": password}, timeout_s)
        if status == 0:
            last_err = f"{url}: {raw}"
            continue
        if status < 200 or status >= 300:
            last_err = f"{url}: http-{status}"
            continue
        ok, token, err = parse_login_token(raw)
        if ok:
            return True, token, ""
        last_err = f"{url}: {err}"
    return False, "", last_err or "login failed"


def auto_register_and_get_token(system_url: str, timeout_s: float) -> Tuple[bool, str, str]:
    config_urls = build_auth_candidates(system_url, ["/auth/config", "/api/auth/config"])
    register_urls = build_auth_candidates(system_url, ["/auth/register", "/api/auth/register"])
    verify_urls = build_auth_candidates(system_url, ["/auth/verify", "/api/auth/verify"])
    bootstrap_urls = build_auth_candidates(system_url, ["/auth/bootstrap", "/api/auth/bootstrap"])

    allow_register = True
    allow_bootstrap = False
    for url in config_urls:
        ok, payload, _err = get_json(url, timeout_s)
        if ok:
            allow_register = bool(payload.get("allowRegister", allow_register))
            allow_bootstrap = bool(payload.get("allowBootstrap", allow_bootstrap))
            break

    username = f"bench_{int(time.time())}"
    password = "Bench@2026"
    email = f"{username}@example.com"
    payload = {"username": username, "password": password, "email": email}
    last_err = ""

    if allow_register:
        for url in register_urls:
            status, raw = post_json(url, payload, timeout_s)
            if status == 0:
                last_err = f"{url}: {raw}"
                continue
            if status not in (200, 201, 409):
                last_err = f"{url}: http-{status}"
                continue
            token, verify_token = parse_auth_response(raw)
            if token:
                return True, token, "auto-register"
            if verify_token:
                for verify_url in verify_urls:
                    verify_status, verify_raw = post_json(verify_url, {"username": username, "token": verify_token}, timeout_s)
                    if 200 <= verify_status < 300:
                        ok, login_token, err = login_and_get_token(system_url, "", username, password, timeout_s)
                        if ok:
                            return True, login_token, "auto-register-verify-login"
                        last_err = err
                    else:
                        last_err = f"{verify_url}: {verify_raw or verify_status}"
            ok, login_token, err = login_and_get_token(system_url, "", username, password, timeout_s)
            if ok:
                return True, login_token, "auto-register-login"
            last_err = err or last_err

    if allow_bootstrap:
        for url in bootstrap_urls:
            status, raw = post_json(url, payload, timeout_s)
            if status == 0:
                last_err = f"{url}: {raw}"
                continue
            if status < 200 or status >= 300:
                last_err = f"{url}: http-{status}"
                continue
            token, verify_token = parse_auth_response(raw)
            if token:
                return True, token, "auto-bootstrap"
            if verify_token:
                for verify_url in verify_urls:
                    verify_status, _ = post_json(verify_url, {"username": username, "token": verify_token}, timeout_s)
                    if 200 <= verify_status < 300:
                        ok, login_token, err = login_and_get_token(system_url, "", username, password, timeout_s)
                        if ok:
                            return True, login_token, "auto-bootstrap-verify-login"
                        last_err = err

    return False, "", last_err or "auto-register unavailable"


def workspace_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def system_origin(system_url: str) -> Tuple[str, str, int]:
    parsed = urlparse(system_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, host, port


def is_tcp_port_open(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def find_phoenix_executable() -> Optional[Path]:
    root = workspace_root_dir()
    for candidate in [root / "build" / "phoenix_main.exe", root / "phoenix_main.exe"]:
        if candidate.exists():
            return candidate
    return None


def start_phoenix_service(system_launch_command: Optional[List[str]] = None) -> Tuple[bool, str]:
    command = [str(item).strip() for item in (system_launch_command or []) if str(item).strip()]
    if not command:
        exe_path = find_phoenix_executable()
        if exe_path is None:
            return False, "phoenix_main.exe-not-found"
        command = [
            str(exe_path),
            "--gateway-host=127.0.0.1",
            "--port=5080",
            "--study-port=5081",
            "--base-dir=runtime_store",
            "--db-path=runtime_store/ai_store.sqlite",
            "--lmdb-dir=lmdb",
            "--robots-dir=robots",
            "--redis-url=redis://127.0.0.1:6379",
            "--log-mode=release",
            "--tests-autoload=true",
            "--robots-limit=10000000",
            "--robots-autoload=true",
            "--tests-dir=tests",
        ]
    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        subprocess.Popen(command, cwd=str(workspace_root_dir()), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
        return True, "started"
    except Exception as exc:
        return False, str(exc)


def parse_system_launch_command(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception as exc:
        raise ValueError(f"invalid system-launch-command-json: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("system-launch-command-json must be a JSON array")
    command = [str(item).strip() for item in payload if str(item).strip()]
    if not command:
        raise ValueError("system-launch-command-json must contain at least one argument")
    return command


def wait_for_system_ready(system_url: str, timeout_s: float) -> Tuple[bool, str]:
    _scheme, host, port = system_origin(system_url)
    deadline = time.time() + timeout_s
    last_err = f"tcp-connect-failed:{host}:{port}"
    while time.time() < deadline:
        if is_tcp_port_open(host, port, 1.5):
            return True, "ready"
        time.sleep(1.0)
    return False, last_err


def ensure_system_service(system_url: str,
                          auto_manage: bool,
                          timeout_s: float,
                          system_launch_command: Optional[List[str]] = None) -> Dict[str, Any]:
    _scheme, host, port = system_origin(system_url)
    if is_tcp_port_open(host, port, 1.5):
        return {"ready": True, "action": "reuse", "error": ""}
    if not auto_manage:
        return {"ready": False, "action": "skip", "error": f"system-port-unreachable:{host}:{port}"}
    started, message = start_phoenix_service(system_launch_command)
    if not started:
        return {"ready": False, "action": "start-failed", "error": message}
    ready, wait_message = wait_for_system_ready(system_url, timeout_s)
    return {"ready": ready, "action": "start", "error": "" if ready else wait_message}


def probe_system_chat(system_url: str,
                      system_token: str,
                      timeout_s: float,
                      http_backend: str,
                      httpx_client: Optional[Any]) -> Tuple[bool, str]:
    payload = {"text": "benchmark warmup ping", "maxTokens": 16, "sessionId": f"bench-health-{int(time.time() * 1000)}"}
    headers = {"Authorization": f"Bearer {system_token}"} if system_token else None
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    attempt = 0
    last_error = "system-probe-failed"

    while True:
        attempt += 1
        if http_backend == "httpx":
            status, raw = post_json_httpx(system_url, payload, timeout_s, headers=headers, client=httpx_client)
        else:
            status, raw = post_json(system_url, payload, timeout_s, headers=headers)

        if status == 0:
            ok = False
            err = raw
        else:
            ok, _reply, err = extract_system_reply(raw)
        if ok:
            return True, ""

        last_error = str(err or (f"http-{status}" if status else "system-probe-failed"))
        if time.monotonic() >= deadline or not is_transient_system_probe_error(status, last_error):
            return False, last_error

        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            return False, last_error
        time.sleep(min(2.0, 0.5 * attempt, remaining))


def recover_system_service(system_url: str,
                           system_token: str,
                           timeout_s: float,
                           http_backend: str,
                           httpx_client: Optional[Any],
                           auto_manage_system: bool,
                           system_start_timeout: float,
                           system_launch_command: Optional[List[str]] = None) -> Tuple[bool, str]:
    with SYSTEM_RECOVERY_LOCK:
        system_management = ensure_system_service(system_url, auto_manage_system, max(5.0, system_start_timeout), system_launch_command)
        if not system_management.get("ready"):
            return False, str(system_management.get("error", "system-recover-failed"))
        ready, err = probe_system_chat(system_url, system_token, max(timeout_s, 30.0), http_backend, httpx_client)
        if ready:
            return True, "recovered"
        return False, err


def extract_system_reply(raw: str) -> Tuple[bool, str, str]:
    try:
        doc = json.loads(raw)
    except Exception:
        raw_text = raw.strip()
        if raw_text:
            return True, raw_text, ""
        return False, "", "invalid-json"
    if not isinstance(doc, dict):
        return False, "", "invalid-payload"
    if doc.get("ok") is False:
        return False, "", str(doc.get("error") or doc.get("message") or "system-error")
    result = doc.get("result", doc)
    candidates: List[str] = []
    if isinstance(result, dict):
        for key in ["reply", "transformerReply", "graphReply"]:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    for key in ["reply", "text", "answer"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for item in candidates:
        if item:
            return True, item, ""
    return False, "", "empty-reply"


def is_transient_system_probe_error(status: int, err: str) -> bool:
    lowered = str(err or "").strip().lower()
    if any(marker in lowered for marker in ("invalid-token", "unauthorized", "forbidden", "jwt")):
        return False
    if status in {0, 502, 503, 504}:
        return True
    transient_markers = (
        "bad status",
        "request failed",
        "request timeout",
        "empty-reply",
        "connection refused",
        "connection reset",
        "timed out",
        "temporarily unavailable",
    )
    return any(marker in lowered for marker in transient_markers)


def extract_chat_reply(raw: str) -> Tuple[bool, str, str]:
    try:
        doc = json.loads(raw)
    except Exception:
        return False, "", "invalid-json"
    if not isinstance(doc, dict):
        return False, "", "invalid-payload"
    choices = doc.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return True, content.strip(), ""
            content = choice.get("text", "")
            if isinstance(content, str) and content.strip():
                return True, content.strip(), ""
    if isinstance(doc.get("message"), dict):
        content = doc["message"].get("content", "")
        if isinstance(content, str) and content.strip():
            return True, content.strip(), ""
    response = doc.get("response", "")
    if isinstance(response, str) and response.strip():
        return True, response.strip(), ""
    for key in ["content", "text", "reply", "answer"]:
        value = doc.get(key, "")
        if isinstance(value, str) and value.strip():
            return True, value.strip(), ""
    return False, "", str(doc.get("error", "empty-reply"))


def score_case_reply(reply: str, bench_case: BenchCase) -> Tuple[float, Dict[str, float], bool]:
    if bench_case.reference is None:
        return -1.0, {}, False
    if isinstance(bench_case.reference, str):
        score, detail = score_text(reply, bench_case.reference)
    elif isinstance(bench_case.reference, dict):
        score, detail = score_structured(reply, bench_case.reference)
    else:
        score, detail = 0.0, {"final": 0.0}
    passed = (score >= bench_case.threshold) if bench_case.threshold > 0.0 else True
    return round(score, 2), detail, passed


def load_quality_cases(file_path: str) -> List[BenchCase]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"cases file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, list):
        raise ValueError("cases file must be a JSON array")
    out: List[BenchCase] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        out.append(BenchCase(
            case_id=str(item.get("id", f"quality-{index:03d}")),
            suite="quality",
            task_type=str(item.get("taskType", "quality")),
            prompt=str(item.get("input", "")).strip(),
            reference=item.get("reference"),
            threshold=float((item.get("scoring") or {}).get("threshold", 0.0) or 0.0),
            tags=[str(tag) for tag in item.get("tags", []) if str(tag).strip()],
        ))
    return [case for case in out if case.prompt]


def resolve_questionnaire_files(single_file: str, file_list: List[str]) -> List[str]:
    root = workspace_root_dir()
    candidates: List[Path] = []
    if file_list:
        candidates.extend(Path(item) for item in file_list if str(item).strip())
    elif single_file.strip():
        candidates.append(Path(single_file.strip()))
    else:
        candidates.append(root / DEFAULT_QUESTIONNAIRE_FILE)
        for pattern in DEFAULT_QUESTIONNAIRE_GLOBS:
            candidates.extend(sorted(root.glob(pattern)))

    resolved: List[str] = []
    seen = set()
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else (root / candidate)
        if not path.exists() or not path.is_file():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(key)
    return resolved


def load_questionnaire_cases(file_paths: List[str], limit: int) -> List[BenchCase]:
    if not file_paths:
        return []
    cases: List[BenchCase] = []
    seen_prompts = set()
    for file_path in file_paths:
        path = Path(file_path)
        prompts = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        for index, prompt in enumerate(prompts, start=1):
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            cases.append(BenchCase(
                case_id=f"questionnaire-{path.stem}-{index:03d}",
                suite="questionnaire",
                task_type="questionnaire",
                prompt=prompt,
                reference=None,
                threshold=0.0,
                tags=["questionnaire", path.stem],
            ))
    return random_sample_cases(cases, limit)


def random_sample_cases(cases: List[BenchCase], limit: int) -> List[BenchCase]:
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    return random.sample(cases, limit)


def random_sample_records(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(records) <= limit:
        sampled = list(records)
        random.shuffle(sampled)
        return sampled
    return random.sample(records, limit)


def _normalize_external_record(record: Dict[str, Any], index: int, default_threshold: float, source_name: str) -> Optional[BenchCase]:
    prompt = ""
    for key in ["input", "prompt", "question", "instruction", "query"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break
    if not prompt:
        return None

    reference: Optional[Any] = None
    for key in ["reference", "answer", "expected", "output", "target", "gold"]:
        value = record.get(key)
        if isinstance(value, (str, dict)):
            if isinstance(value, str) and not value.strip():
                continue
            reference = value
            break

    scoring = record.get("scoring") if isinstance(record.get("scoring"), dict) else {}
    threshold = float(scoring.get("threshold", default_threshold) or default_threshold)
    task_type = str(record.get("taskType") or record.get("task_type") or record.get("category") or "external")
    tags_raw = record.get("tags", [])
    tags = [str(tag) for tag in tags_raw if str(tag).strip()] if isinstance(tags_raw, list) else [source_name]
    case_id = str(record.get("id") or f"external-{source_name}-{index:04d}")
    return BenchCase(case_id=case_id, suite="external", task_type=task_type, prompt=prompt, reference=reference, threshold=threshold, tags=tags)


def load_external_dataset_file(file_path: str, default_threshold: float, limit: int = 0) -> List[BenchCase]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"external dataset file not found: {path}")
    suffix = path.suffix.lower()
    source_name = path.stem
    raw_records: List[Dict[str, Any]] = []
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(payload, list):
            raw_records = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            for key in ["records", "items", "data", "cases", "samples"]:
                value = payload.get(key)
                if isinstance(value, list):
                    raw_records = [item for item in value if isinstance(item, dict)]
                    break
    elif suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                raw_records.append(obj)
    elif suffix in [".csv", ".tsv"]:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            raw_records = [dict(item) for item in reader]
    else:
        raise ValueError(f"unsupported external dataset file: {path}")

    sampled_records = random_sample_records(raw_records, limit)

    cases: List[BenchCase] = []
    for index, record in enumerate(sampled_records, start=1):
        case = _normalize_external_record(record, index, default_threshold, source_name)
        if case is not None:
            cases.append(case)
    return cases


def load_shared_local_qa_cases(file_paths: List[str], shared_limit: int, default_threshold: float) -> Tuple[List[BenchCase], List[BenchCase], Dict[str, Any]]:
    quality_pool: List[BenchCase] = []
    for file_path in file_paths:
        quality_pool.extend(load_external_dataset_file(file_path, default_threshold, 0))

    sampled_quality = random_sample_cases(quality_pool, shared_limit)
    questionnaire_cases: List[BenchCase] = []
    quality_cases: List[BenchCase] = []
    for index, case in enumerate(sampled_quality, start=1):
        quality_cases.append(BenchCase(
            case_id=f"shared-quality-{index:04d}",
            suite="quality",
            task_type=case.task_type,
            prompt=case.prompt,
            reference=case.reference,
            threshold=case.threshold,
            tags=list(case.tags) + ["shared-local-qa"],
        ))
        questionnaire_cases.append(BenchCase(
            case_id=f"shared-questionnaire-{index:04d}",
            suite="questionnaire",
            task_type="questionnaire",
            prompt=case.prompt,
            reference=None,
            threshold=0.0,
            tags=list(case.tags) + ["shared-local-qa", "questionnaire"],
        ))

    metadata = {
        "mode": "shared-local-qa",
        "files": list(file_paths),
        "poolCount": len(quality_pool),
        "sharedCount": len(sampled_quality),
        "qualityCount": len(quality_cases),
        "questionnaireCount": len(questionnaire_cases),
    }
    return quality_cases, questionnaire_cases, metadata


def load_external_line_pairs(prompts_file: str, answers_file: str, limit: int, default_threshold: float) -> List[BenchCase]:
    prompts_path = Path(prompts_file)
    answers_path = Path(answers_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"external prompts file not found: {prompts_path}")
    if not answers_path.exists():
        raise FileNotFoundError(f"external answers file not found: {answers_path}")
    prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    answers = [line.strip() for line in answers_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    pair_count = min(len(prompts), len(answers))
    indices = list(range(pair_count))
    if limit > 0 and len(indices) > limit:
        indices = random.sample(indices, limit)
    else:
        random.shuffle(indices)

    cases: List[BenchCase] = []
    for case_number, index in enumerate(indices, start=1):
        cases.append(BenchCase(
            case_id=f"external-line-{index + 1:04d}",
            suite="external",
            task_type="external_qa",
            prompt=prompts[index],
            reference=answers[index],
            threshold=default_threshold,
            tags=["external", "line-pair"],
        ))
    return cases


def load_external_index(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def discover_local_qa_dataset_files() -> List[str]:
    root = workspace_root_dir()
    candidates = [
        root / "tests" / "GPT4all" / "gpt4all.jsonl",
        root / "tests" / "gpt4all" / "gpt4all.jsonl",
    ]
    resolved: List[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(key)
    return resolved


def load_benchmark_catalog() -> Dict[str, Dict[str, Any]]:
    catalog_path = Path(__file__).resolve().with_name("benchmark_catalog.json")
    if not catalog_path.exists():
        return {}
    payload = json.loads(catalog_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, list):
        raise ValueError("benchmark catalog must be a JSON array")
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        preset_id = str(item.get("id", "")).strip()
        if not preset_id:
            continue
        catalog[preset_id] = item
    return catalog


def cache_file_for_preset(cache_dir: Path, preset_id: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", preset_id).strip("_") or "preset"
    return cache_dir / f"{safe_name}.json"


def fetch_rows_page(dataset: str,
                    dataset_rows_url: str,
                    config: str,
                    split: str,
                    offset: int,
                    length: int,
                    timeout_s: float,
                    retries: int) -> List[Dict[str, Any]]:
    query = urlencode({
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    })
    last_err = ""
    for attempt in range(max(1, retries + 1)):
        ok, payload, err = get_json(f"{dataset_rows_url}?{query}", timeout_s)
        if ok:
            rows = payload.get("rows", []) if isinstance(payload, dict) else []
            out: List[Dict[str, Any]] = []
            for item in rows:
                if isinstance(item, dict) and isinstance(item.get("row"), dict):
                    out.append(item["row"])
            return out
        last_err = err or f"failed to fetch dataset rows: {dataset}/{config}/{split}"
        if attempt + 1 < max(1, retries + 1):
            time.sleep(min(3.0, 0.8 * (attempt + 1)))
    raise RuntimeError(last_err)


def fetch_or_load_preset_rows(preset: Dict[str, Any],
                              cache_dir: Path,
                              refresh_cache: bool,
                              limit_per_preset: int,
                              timeout_s: float,
                              dataset_rows_url: str,
                              retries: int,
                              cache_only: bool) -> List[Dict[str, Any]]:
    preset_id = str(preset.get("id", "preset"))
    cache_path = cache_file_for_preset(cache_dir, preset_id)
    if cache_path.exists() and not refresh_cache:
        payload = json.loads(cache_path.read_text(encoding="utf-8", errors="replace"))
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        return [item for item in rows if isinstance(item, dict)]
    if cache_only:
        raise RuntimeError("cache-only-no-local-snapshot")

    dataset = str(preset.get("dataset", "")).strip()
    config = str(preset.get("config", "")).strip()
    split = str(preset.get("split", "")).strip()
    if not dataset or not config or not split:
        raise ValueError(f"invalid benchmark preset: {preset_id}")

    default_limit = int(preset.get("defaultLimit", DEFAULT_BENCHMARK_LIMIT_PER_PRESET) or DEFAULT_BENCHMARK_LIMIT_PER_PRESET)
    target = max(1, limit_per_preset if limit_per_preset > 0 else default_limit)
    rows: List[Dict[str, Any]] = []
    offset = 0
    page_size = min(100, target)
    while len(rows) < target:
        page = fetch_rows_page(dataset, dataset_rows_url, config, split, offset, min(page_size, target - len(rows)), timeout_s, retries)
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        if len(page) < min(page_size, target - len(rows) + len(page)):
            break

    payload = {
        "preset": preset_id,
        "dataset": dataset,
        "config": config,
        "split": split,
        "fetchedAt": now_str(),
        "rows": rows,
    }
    atomic_write_text(cache_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return rows


def _gsm8k_reference(answer: str) -> str:
    stripped = answer.strip()
    match = re.search(r"####\s*(.+)$", stripped, flags=re.MULTILINE)
    return match.group(1).strip() if match else stripped


def _build_mcq_prompt(question: str, options: List[Tuple[str, str]], instruction: str) -> str:
    option_lines = [f"{label}. {text}" if label else text for label, text in options if text]
    return f"{instruction}\n\n题目:\n{question}\n\n选项:\n" + "\n".join(option_lines) + "\n\n请直接给出最优答案。"


def build_case_from_preset_row(preset: Dict[str, Any], row: Dict[str, Any], index: int) -> Optional[BenchCase]:
    preset_id = str(preset.get("id", "preset"))
    builder = str(preset.get("builder", "qa")).strip().lower()
    threshold = float(preset.get("threshold", 0.0) or 0.0)
    task_type = str(preset.get("taskType", builder or "benchmark"))
    tags = [str(tag) for tag in preset.get("tags", []) if str(tag).strip()]
    case_id = f"{preset_id}-{index:04d}"

    if builder == "gsm8k":
        prompt = str(row.get("question", "")).strip()
        answer = _gsm8k_reference(str(row.get("answer", "")))
        if prompt and answer:
            return BenchCase(case_id=case_id, suite="standard", task_type=task_type, prompt=prompt, reference=answer, threshold=threshold, tags=tags)
        return None

    if builder == "qa":
        prompt = str(row.get(preset.get("promptField", "question"), "")).strip()
        reference = str(row.get(preset.get("referenceField", "answer"), "")).strip()
        if prompt and reference:
            return BenchCase(case_id=case_id, suite="standard", task_type=task_type, prompt=prompt, reference=reference, threshold=threshold, tags=tags)
        return None

    if builder == "arc_mcq":
        question = str(row.get("question", "")).strip()
        choices = row.get("choices", {}) if isinstance(row.get("choices"), dict) else {}
        labels = choices.get("label", []) if isinstance(choices.get("label"), list) else []
        texts = choices.get("text", []) if isinstance(choices.get("text"), list) else []
        options = [(str(label).strip(), str(text).strip()) for label, text in zip(labels, texts) if str(text).strip()]
        answer_key = str(row.get("answerKey", "")).strip()
        answer_text = ""
        for label, text in options:
            if label == answer_key:
                answer_text = text
                break
        if question and options and answer_text:
            prompt = _build_mcq_prompt(question, options, "请回答下列多项选择推理题")
            return BenchCase(case_id=case_id, suite="standard", task_type=task_type, prompt=prompt, reference=f"{answer_key}. {answer_text}", threshold=threshold, tags=tags)
        return None

    if builder == "hellaswag":
        context = str(row.get("ctx", "")).strip()
        if not context:
            ctx_a = str(row.get("ctx_a", "")).strip()
            ctx_b = str(row.get("ctx_b", "")).strip()
            context = (ctx_a + " " + ctx_b).strip()
        endings = [str(item).strip() for item in row.get("endings", []) if str(item).strip()] if isinstance(row.get("endings"), list) else []
        label = str(row.get("label", "")).strip()
        if context and endings and label.isdigit():
            label_index = int(label)
            if 0 <= label_index < len(endings):
                options = [(str(i), ending) for i, ending in enumerate(endings)]
                prompt = _build_mcq_prompt(context, options, "请选择最合理的后续句子")
                return BenchCase(case_id=case_id, suite="standard", task_type=task_type, prompt=prompt, reference=endings[label_index], threshold=threshold, tags=tags)
        return None

    if builder == "winogrande":
        sentence = str(row.get("sentence", "")).strip()
        option1 = str(row.get("option1", "")).strip()
        option2 = str(row.get("option2", "")).strip()
        answer = str(row.get("answer", "")).strip()
        answer_text = option1 if answer == "1" else option2 if answer == "2" else ""
        if sentence and option1 and option2 and answer_text:
            prompt = _build_mcq_prompt(sentence, [("1", option1), ("2", option2)], "请为下列句子空缺处选择最合适的选项")
            return BenchCase(case_id=case_id, suite="standard", task_type=task_type, prompt=prompt, reference=answer_text, threshold=threshold, tags=tags)
        return None

    return None


def load_standard_benchmark_cases(preset_names: List[str],
                                  cache_dir: str,
                                  refresh_cache: bool,
                                  limit_per_preset: int,
                                  timeout_s: float,
                                  dataset_rows_url: str,
                                  retries: int,
                                  cache_only: bool) -> Tuple[List[BenchCase], Dict[str, Any]]:
    catalog = load_benchmark_catalog()
    selected = normalize_preset_names(preset_names)
    if not selected:
        return [], {"selected": [], "loaded": [], "failed": []}

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    cases: List[BenchCase] = []
    loaded: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    source_reachable = True
    source_error = ""
    for preset_name in selected:
        preset = catalog.get(preset_name)
        if preset is None:
            failed.append({"preset": preset_name, "error": "preset-not-found"})
            continue
        cache_exists = cache_file_for_preset(cache_path, preset_name).exists()
        if not cache_exists and not refresh_cache and not source_reachable:
            failed.append({"preset": preset_name, "error": f"dataset-source-unreachable:{source_error}"})
            continue
        try:
            rows = fetch_or_load_preset_rows(preset, cache_path, refresh_cache, limit_per_preset, timeout_s, dataset_rows_url, retries, cache_only)
            random.shuffle(rows)
            preset_cases = [
                case
                for index, row in enumerate(rows, start=1)
                for case in [build_case_from_preset_row(preset, row, index)]
                if case is not None
            ]
            cases.extend(preset_cases)
            loaded.append({
                "preset": preset_name,
                "count": len(preset_cases),
                "dataset": preset.get("dataset", ""),
                "config": preset.get("config", ""),
                "split": preset.get("split", ""),
            })
        except Exception as exc:
            error_text = str(exc)
            failed.append({"preset": preset_name, "error": error_text})
            lowered = error_text.lower()
            if not cache_exists and not cache_only and any(marker in lowered for marker in ["timed out", "connecttimeout", "10060", "name or service not known", "temporary failure", "nodename nor servname"]):
                source_reachable = False
                source_error = error_text
    return cases, {"selected": selected, "loaded": loaded, "failed": failed, "sourceReachable": source_reachable, "sourceError": source_error}


def load_prompt_answer_cases(prompts_file: str, answers_file: str) -> List[BenchCase]:
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_path}")
    prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    answers: List[str] = []
    if answers_file:
        answers_path = Path(answers_file)
        if not answers_path.exists():
            raise FileNotFoundError(f"answers file not found: {answers_path}")
        answers = [line.strip() for line in answers_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if len(answers) != len(prompts):
            raise ValueError(f"answers count mismatch: prompts={len(prompts)}, answers={len(answers)}")
    return [
        BenchCase(
            case_id=f"prompt-{index:03d}",
            suite="quality" if answers else "prompt",
            task_type="prompt",
            prompt=prompt,
            reference=answers[index - 1] if answers else None,
            threshold=0.0,
            tags=["prompt"],
        )
        for index, prompt in enumerate(prompts, start=1)
    ]


def parse_parameter_size_billions(model_doc: Dict[str, Any]) -> float:
    details = model_doc.get("details", {}) if isinstance(model_doc.get("details"), dict) else {}
    raw = str(details.get("parameter_size", "") or "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*B", raw, flags=re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def parse_ollama_list_output(raw: str) -> List[str]:
    names: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name"):
            continue
        parts = stripped.split()
        if parts:
            names.append(parts[0].strip())
    return names


def ollama_command() -> str:
    return shutil.which("ollama") or shutil.which("ollama.exe") or ""


def kill_ollama_processes() -> int:
    if psutil is not None:
        killed = 0
        for proc in psutil.process_iter(["name"]):
            try:
                name = str(proc.info.get("name") or "").lower()
                if "ollama" in name:
                    proc.kill()
                    killed += 1
            except Exception:
                continue
        return killed
    if Path("C:/Windows/System32/taskkill.exe").exists():
        subprocess.run(["taskkill", "/IM", "ollama.exe", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return 1
    return 0


def ollama_tags_url(ollama_url: str) -> str:
    base = ollama_url.rstrip("/")
    if base.endswith("/api/chat"):
        return base[:-len("/api/chat")] + "/api/tags"
    if base.endswith("/api/generate"):
        return base[:-len("/api/generate")] + "/api/tags"
    if base.endswith("/api/tags"):
        return base
    return base + "/api/tags"


def probe_ollama(ollama_url: str, timeout_s: float) -> Tuple[bool, List[Dict[str, Any]], str]:
    ok, payload, err = get_json(ollama_tags_url(ollama_url), timeout_s)
    if not ok:
        return False, [], err
    models = payload.get("models", []) if isinstance(payload, dict) else []
    docs = [item for item in models if isinstance(item, dict)]
    return True, docs, ""


def start_ollama_server() -> Tuple[bool, str]:
    cmd = ollama_command()
    if not cmd:
        return False, "ollama-command-not-found"
    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        subprocess.Popen([cmd, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)
        return True, "started"
    except Exception as exc:
        return False, str(exc)


def wait_for_ollama_ready(ollama_url: str, timeout_s: float) -> Tuple[bool, List[Dict[str, Any]], str]:
    deadline = time.time() + timeout_s
    last_err = ""
    while time.time() < deadline:
        ok, models, err = probe_ollama(ollama_url, 3.0)
        if ok:
            return True, models, "ready"
        last_err = err
        time.sleep(1.0)
    return False, [], last_err or "ollama-ready-timeout"


def ensure_ollama_service(ollama_url: str, auto_manage: bool, restart_unhealthy: bool) -> Dict[str, Any]:
    ok, models, err = probe_ollama(ollama_url, 5.0)
    if ok:
        return {"ready": True, "action": "reuse", "models": models, "error": ""}
    if not auto_manage:
        return {"ready": False, "action": "skip", "models": [], "error": err}
    killed = 0
    if restart_unhealthy:
        killed = kill_ollama_processes()
        time.sleep(1.0)
    started, start_msg = start_ollama_server()
    if not started:
        return {"ready": False, "action": "start-failed", "models": [], "error": start_msg, "killed": killed}
    ready, models, wait_msg = wait_for_ollama_ready(ollama_url, 30.0)
    return {
        "ready": ready,
        "action": "restart" if killed else "start",
        "models": models,
        "error": "" if ready else wait_msg,
        "killed": killed,
    }


def choose_best_model(model_docs: List[Dict[str, Any]], explicit_model: str, preferred_models: List[str]) -> str:
    names = [str(doc.get("name", "")).strip() for doc in model_docs if str(doc.get("name", "")).strip()]
    if explicit_model.strip() and explicit_model.strip() in names:
        return explicit_model.strip()
    for candidate in preferred_models:
        if candidate in names:
            return candidate

    scored: List[Tuple[Tuple[float, float, str], str]] = []
    for doc in model_docs:
        name = str(doc.get("name", "")).strip()
        if not name:
            continue
        lowered = name.lower()
        if any(skip in lowered for skip in ["embed", "embedding", "rerank"]):
            continue
        size_b = parse_parameter_size_billions(doc)
        unknown_penalty = 1.0 if size_b <= 0.0 else 0.0
        scored.append(((unknown_penalty, -(size_b or 0.0), name), name))
    if scored:
        scored.sort(key=lambda item: item[0])
        return scored[0][1]
    return names[0] if names else ""


def warmup_ollama_model(ollama_url: str,
                        model: str,
                        timeout_s: float,
                        http_backend: str,
                        httpx_client: Optional[Any]) -> Dict[str, Any]:
    payload = build_direct_chat_payload(ROUTE_OLLAMA, "只回复 yes", model, 16)
    if http_backend == "httpx":
        status, raw = post_json_httpx(ollama_url, payload, timeout_s, client=httpx_client)
    else:
        status, raw = post_json(ollama_url, payload, timeout_s)
    ok, reply, err = extract_chat_reply(raw) if status != 0 else (False, "", raw)
    return {"ok": ok, "status": status, "reply": reply, "error": err}


def load_cli_model_names() -> List[str]:
    cmd = ollama_command()
    if not cmd:
        return []
    try:
        cp = subprocess.run([cmd, "list"], capture_output=True, text=True, check=False, encoding="utf-8", errors="replace")
        if cp.returncode != 0:
            return []
        return parse_ollama_list_output(cp.stdout)
    except Exception:
        return []


def discover_or_choose_model(args: argparse.Namespace, model_docs: List[Dict[str, Any]]) -> str:
    preferred = [item.strip() for item in (args.preferred_models or "").split(",") if item.strip()]
    preferred = preferred or list(DEFAULT_PREFERRED_MODELS)
    chosen = choose_best_model(model_docs, args.ollama_model, preferred)
    if chosen:
        return chosen
    cli_names = load_cli_model_names()
    if args.ollama_model and args.ollama_model in cli_names:
        return args.ollama_model
    for candidate in preferred:
        if candidate in cli_names:
            return candidate
    return cli_names[0] if cli_names else ""


def run_route_case(route_name: str,
                   bench_case: BenchCase,
                   round_index: int,
                   timeout_s: float,
                   system_url: str,
                   system_token: str,
                   system_launch_command: Optional[List[str]],
                   ollama_url: str,
                   ollama_model: str,
                   llamacpp_url: str,
                   llamacpp_model: str,
                   max_tokens: int,
                   http_backend: str,
                   httpx_client: Optional[Any],
                   system_retry_count: int,
                   auto_manage_system: bool,
                   system_start_timeout: float,
                   llamacpp_style: str = "ollama-chat") -> RouteCaseResult:
    headers = {"Authorization": f"Bearer {system_token}"} if system_token and route_name == "system" else None
    if route_name == ROUTE_SYSTEM:
        payload = {"text": bench_case.prompt, "sessionId": f"bench-{round_index}-{bench_case.case_id}-{int(time.time() * 1000)}"}
        attempts = max(1, int(system_retry_count) + 1)
        status = 0
        raw = ""
        latency_ms = 0.0
        ok = False
        reply = ""
        err = ""
        for attempt in range(attempts):
            t0 = time.perf_counter()
            if http_backend == "httpx":
                status, raw = post_json_httpx(system_url, payload, timeout_s, headers=headers, client=httpx_client)
            else:
                status, raw = post_json(system_url, payload, timeout_s, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if status == 0:
                err = raw
            else:
                ok, reply, err = extract_system_reply(raw)
                if ok:
                    break
            if attempt + 1 < attempts and (status == 0 or "disconnected" in str(err).lower()):
                if status == 0 and auto_manage_system:
                    recovered, recover_msg = recover_system_service(
                        system_url,
                        system_token,
                        timeout_s,
                        http_backend,
                        httpx_client,
                        auto_manage_system,
                        system_start_timeout,
                        system_launch_command,
                    )
                    if not recovered:
                        raw = recover_msg
                        break
                time.sleep(0.8)
                continue
            if status == 0:
                return RouteCaseResult(route_name, round_index, bench_case.case_id, bench_case.suite, bench_case.task_type, bench_case.prompt, False, latency_ms, compute_speed_score(latency_ms, timeout_s, False), status, raw, "", threshold=bench_case.threshold)
    else:
        target_url = ollama_url if route_name == ROUTE_OLLAMA else llamacpp_url
        target_model = ollama_model if route_name == ROUTE_OLLAMA else llamacpp_model
        endpoint_style = "ollama-chat" if route_name == ROUTE_OLLAMA else llamacpp_style
        payload = build_direct_chat_payload(route_name, bench_case.prompt, target_model, max_tokens, endpoint_style)
        t0 = time.perf_counter()
        if http_backend == "httpx":
            status, raw = post_json_httpx(target_url, payload, timeout_s, client=httpx_client)
        else:
            status, raw = post_json(target_url, payload, timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if status == 0:
            return RouteCaseResult(route_name, round_index, bench_case.case_id, bench_case.suite, bench_case.task_type, bench_case.prompt, False, latency_ms, compute_speed_score(latency_ms, timeout_s, False), status, raw, "", threshold=bench_case.threshold)
        ok, reply, err = extract_chat_reply(raw)

    speed_score = compute_speed_score(latency_ms, timeout_s, ok)
    if not ok:
        return RouteCaseResult(route_name, round_index, bench_case.case_id, bench_case.suite, bench_case.task_type, bench_case.prompt, False, latency_ms, speed_score, status, err, "", threshold=bench_case.threshold)
    quality_score, quality_detail, quality_pass = score_case_reply(reply, bench_case)
    return RouteCaseResult(route_name, round_index, bench_case.case_id, bench_case.suite, bench_case.task_type, bench_case.prompt, True, latency_ms, speed_score, status, "", reply, quality_score=quality_score, quality_detail=quality_detail, quality_pass=quality_pass, threshold=bench_case.threshold)


def run_route_cases(route_name: str,
                    cases: List[BenchCase],
                    round_index: int,
                    concurrency: int,
                    timeout_s: float,
                    system_url: str,
                    system_token: str,
                    system_launch_command: Optional[List[str]],
                    ollama_url: str,
                    ollama_model: str,
                    llamacpp_url: str,
                    llamacpp_model: str,
                    max_tokens: int,
                    http_backend: str,
                    httpx_client: Optional[Any],
                    shuffle_cases: bool,
                    system_retry_count: int,
                    auto_manage_system: bool,
                    system_start_timeout: float,
                    llamacpp_style: str = "ollama-chat",
                    completed_keys: Optional[Set[Tuple[int, str]]] = None,
                    checkpoint_every: int = 0,
                    progress_callback: Optional[Callable[[List[RouteCaseResult], float], None]] = None) -> Tuple[List[RouteCaseResult], float]:
    skip_keys = completed_keys or set()
    tasks = [case for case in cases if (round_index, case.case_id) not in skip_keys]
    if shuffle_cases:
        random.shuffle(tasks)
    if not tasks:
        return [], 0.0
    started = time.perf_counter()
    results: List[RouteCaseResult] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_map = {
            executor.submit(
                run_route_case,
                route_name,
                bench_case,
                round_index,
                timeout_s,
                system_url,
                system_token,
                system_launch_command,
                ollama_url,
                ollama_model,
                llamacpp_url,
                llamacpp_model,
                max_tokens,
                http_backend,
                httpx_client,
                system_retry_count,
                auto_manage_system,
                system_start_timeout,
                llamacpp_style,
            ): bench_case
            for bench_case in tasks
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception as exc:
                bench_case = future_map[future]
                results.append(RouteCaseResult(route_name, round_index, bench_case.case_id, bench_case.suite, bench_case.task_type, bench_case.prompt, False, 0.0, 0.0, 0, f"runner-exception:{exc}", "", threshold=bench_case.threshold))
            if progress_callback is not None and checkpoint_every > 0 and len(results) % checkpoint_every == 0:
                progress_callback(list(results), (time.perf_counter() - started) * 1000.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if progress_callback is not None and results and (checkpoint_every <= 0 or len(results) % checkpoint_every != 0):
        progress_callback(list(results), elapsed_ms)
    results.sort(key=lambda item: (item.round_index, item.suite, item.case_id, item.prompt))
    return results, elapsed_ms


def current_round_results(results: List[RouteCaseResult], round_index: int) -> List[RouteCaseResult]:
    return [item for item in results if item.round_index == round_index]


def build_round_case_plan(cases: List[BenchCase], shuffle_cases: bool) -> List[BenchCase]:
    planned = list(cases)
    if shuffle_cases:
        random.shuffle(planned)
    return planned


def build_stability_snapshot(results: List[RouteCaseResult]) -> Dict[str, Any]:
    summary = summarize_route(results)
    snapshot = {
        "samples": summary["total"],
        "success_rate": summary["success_rate"],
        "latency_avg_ms": summary["latency_avg_ms"],
        "speed_score_avg": summary["speed_score_avg"],
        "quality_avg": summary["quality_avg"],
        "balanced_score": summary["balanced_score"],
    }
    return snapshot


def evaluate_route_stability(history: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    if len(history) < args.stability_window:
        return False, {"reason": "insufficient-window", "checks": len(history)}
    latest = history[-1]
    if int(latest.get("samples", 0) or 0) < args.stability_min_samples:
        return False, {"reason": "insufficient-samples", "samples": int(latest.get("samples", 0) or 0)}

    window = history[-args.stability_window:]
    metrics: Dict[str, Dict[str, float]] = {}
    stable = True

    def _collect(metric_name: str) -> List[float]:
        values: List[float] = []
        for item in window:
            value = item.get(metric_name)
            if value is None:
                continue
            number = float(value)
            if metric_name in {"quality_avg", "balanced_score"} and number < 0.0:
                continue
            values.append(number)
        return values

    metric_values = {
        "success_rate": _collect("success_rate"),
        "quality_avg": _collect("quality_avg"),
        "balanced_score": _collect("balanced_score"),
        "latency_avg_ms": _collect("latency_avg_ms"),
    }
    thresholds = {
        "success_rate": float(args.stability_success_rate_delta),
        "quality_avg": float(args.stability_quality_delta),
        "balanced_score": float(args.stability_balanced_delta),
    }

    for metric_name, values in metric_values.items():
        if len(values) < 2:
            continue
        diff = max(values) - min(values)
        if metric_name == "latency_avg_ms":
            baseline = max(1.0, statistics.fmean(values))
            allowed = max(float(args.stability_latency_delta_ms), baseline * float(args.stability_latency_ratio))
        else:
            allowed = thresholds[metric_name]
        metrics[metric_name] = {
            "range": round(diff, 4),
            "allowed": round(allowed, 4),
        }
        if diff > allowed:
            stable = False

    if not metrics:
        return False, {"reason": "no-comparable-metrics", "checks": len(history)}
    return stable, {
        "reason": "stable" if stable else "range-exceeded",
        "checks": len(history),
        "samples": int(latest.get("samples", 0) or 0),
        "metrics": metrics,
    }


def evaluate_round_stability(system_results: List[RouteCaseResult],
                             ollama_results: List[RouteCaseResult],
                             history: List[Dict[str, Any]],
                             args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    snapshot = {
        "system": build_stability_snapshot(system_results),
        "ollama": build_stability_snapshot(ollama_results),
    }
    history.append(snapshot)
    system_stable, system_detail = evaluate_route_stability([item["system"] for item in history], args)
    ollama_stable, ollama_detail = evaluate_route_stability([item["ollama"] for item in history], args)
    return system_stable and ollama_stable, {
        "stable": system_stable and ollama_stable,
        "samples": min(int(snapshot["system"].get("samples", 0) or 0), int(snapshot["ollama"].get("samples", 0) or 0)),
        "system": system_detail,
        "ollama": ollama_detail,
    }


def result_to_dict(item: RouteCaseResult) -> Dict[str, Any]:
    return {
        "route": item.route,
        "round": item.round_index,
        "case_id": item.case_id,
        "suite": item.suite,
        "task_type": item.task_type,
        "prompt": item.prompt,
        "ok": item.ok,
        "latency_ms": round(item.latency_ms, 2),
        "speed_score": item.speed_score,
        "status": item.status,
        "error": item.error,
        "reply": item.reply,
        "quality_score": item.quality_score,
        "quality_detail": item.quality_detail or {},
        "quality_pass": item.quality_pass,
        "threshold": item.threshold,
    }


def summarize_route(results: List[RouteCaseResult]) -> Dict[str, Any]:
    total = len(results)
    oks = [item for item in results if item.ok]
    failed = [item for item in results if not item.ok]
    latencies = sorted(item.latency_ms for item in oks)
    speed_scores = sorted(item.speed_score for item in results)
    quality_suites = {"quality", "external", "standard"}
    quality_all = [item for item in results if item.suite in quality_suites]
    quality_ok = [item for item in quality_all if item.ok]
    quality_scores = sorted(item.quality_score for item in quality_ok if item.quality_score >= 0.0)
    reply_chars = [len(item.reply) for item in oks]
    quality_passes = [item for item in quality_ok if item.quality_pass]
    latency_ci_low, latency_ci_high = bootstrap_ci_mean(latencies)
    balanced_score = -1.0
    if quality_scores:
        balanced_score = round((statistics.fmean(speed_scores) * 0.4) + (statistics.fmean(quality_scores) * 0.6), 2)

    task_types: Dict[str, Any] = {}
    for item in quality_all:
        bucket = task_types.setdefault(item.task_type, {"total": 0, "transportPassed": 0, "qualityPassed": 0, "scoreAvg": 0.0, "latencyAvgMs": 0.0})
        bucket["total"] += 1
        if item.ok:
            bucket["transportPassed"] += 1
            bucket["qualityPassed"] += int(item.quality_pass)
    for task_type, bucket in task_types.items():
        task_items = [item for item in quality_ok if item.task_type == task_type]
        task_scores = [item.quality_score for item in task_items if item.quality_score >= 0.0]
        task_latencies = [item.latency_ms for item in task_items]
        bucket["scoreAvg"] = round(statistics.fmean(task_scores), 2) if task_scores else 0.0
        bucket["latencyAvgMs"] = round(statistics.fmean(task_latencies), 2) if task_latencies else 0.0
        bucket["qualityPassRate"] = round((bucket["qualityPassed"] / bucket["transportPassed"] * 100.0), 2) if bucket["transportPassed"] else 0.0

    round_stats: List[Dict[str, Any]] = []
    for round_index in sorted({item.round_index for item in results}):
        round_items = [item for item in results if item.round_index == round_index]
        round_quality = [item for item in round_items if item.suite in quality_suites and item.ok and item.quality_score >= 0.0]
        round_lat = [item.latency_ms for item in round_items if item.ok]
        round_speed = [item.speed_score for item in round_items]
        round_stats.append({
            "round": round_index,
            "successRate": round(sum(1 for item in round_items if item.ok) / len(round_items) * 100.0, 2) if round_items else 0.0,
            "latencyAvgMs": round(statistics.fmean(round_lat), 2) if round_lat else 0.0,
            "speedScoreAvg": round(statistics.fmean(round_speed), 2) if round_speed else 0.0,
            "qualityAvg": round(statistics.fmean([item.quality_score for item in round_quality]), 2) if round_quality else -1.0,
        })

    return {
        "total": total,
        "ok": len(oks),
        "failed": len(failed),
        "success_rate": round((len(oks) / total * 100.0), 2) if total else 0.0,
        "latency_avg_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latency_p50_ms": round(pct(latencies, 0.50), 2),
        "latency_p90_ms": round(pct(latencies, 0.90), 2),
        "latency_p95_ms": round(pct(latencies, 0.95), 2),
        "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
        "latency_std_ms": round(statistics.pstdev(latencies), 2) if len(latencies) >= 2 else 0.0,
        "latency_ci95_low_ms": round(latency_ci_low, 2),
        "latency_ci95_high_ms": round(latency_ci_high, 2),
        "reply_chars_avg": round(statistics.fmean(reply_chars), 2) if reply_chars else 0.0,
        "reply_chars_p50": round(pct(sorted(float(value) for value in reply_chars), 0.50), 2) if reply_chars else 0.0,
        "speed_score_avg": round(statistics.fmean(speed_scores), 2) if speed_scores else 0.0,
        "speed_score_p50": round(pct(speed_scores, 0.50), 2) if speed_scores else 0.0,
        "quality_total": len(quality_all),
        "quality_ok": len(quality_ok),
        "quality_passed": len(quality_passes),
        "quality_pass_rate": round((len(quality_passes) / len(quality_ok) * 100.0), 2) if quality_ok else 0.0,
        "quality_avg": round(statistics.fmean(quality_scores), 2) if quality_scores else -1.0,
        "quality_p50": round(pct(quality_scores, 0.50), 2) if quality_scores else -1.0,
        "quality_p90": round(pct(quality_scores, 0.90), 2) if quality_scores else -1.0,
        "balanced_score": balanced_score,
        "task_types": task_types,
        "rounds": round_stats,
        "errors": [result_to_dict(item) for item in failed],
    }


def route_result_from_dict(doc: Dict[str, Any]) -> RouteCaseResult:
    return RouteCaseResult(
        route=str(doc.get("route", "")),
        round_index=int(doc.get("round", 0) or 0),
        case_id=str(doc.get("case_id", "")),
        suite=str(doc.get("suite", "")),
        task_type=str(doc.get("task_type", "")),
        prompt=str(doc.get("prompt", "")),
        ok=bool(doc.get("ok", False)),
        latency_ms=float(doc.get("latency_ms", 0.0) or 0.0),
        speed_score=float(doc.get("speed_score", 0.0) or 0.0),
        status=int(doc.get("status", 0) or 0),
        error=str(doc.get("error", "")),
        reply=str(doc.get("reply", "")),
        quality_score=float(doc.get("quality_score", -1.0) if doc.get("quality_score") is not None else -1.0),
        quality_detail=doc.get("quality_detail", {}) if isinstance(doc.get("quality_detail"), dict) else {},
        quality_pass=bool(doc.get("quality_pass", False)),
        threshold=float(doc.get("threshold", 0.0) or 0.0),
    )


def compare_latency_significance(system_results: List[RouteCaseResult], ollama_results: List[RouteCaseResult]) -> Dict[str, Any]:
    a = [item.latency_ms for item in system_results if item.ok]
    b = [item.latency_ms for item in ollama_results if item.ok]
    if not a or not b:
        return {"test": "n/a", "p_value": -1.0, "note": "insufficient-samples"}
    if scipy_stats is not None:
        try:
            stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            return {"test": "mannwhitneyu", "stat": float(stat), "p_value": float(p)}
        except Exception:
            pass
    return {"test": "fallback-delta", "p_value": -1.0, "delta_mean_ms": round(statistics.fmean(a) - statistics.fmean(b), 2)}


def compare_quality_delta(system_results: List[RouteCaseResult], ollama_results: List[RouteCaseResult]) -> Dict[str, Any]:
    quality_suites = {"quality", "external", "standard"}
    ollama_map = {(item.round_index, item.case_id): item for item in ollama_results if item.suite in quality_suites}
    deltas: List[float] = []
    for item in system_results:
        if item.suite not in quality_suites or item.quality_score < 0.0:
            continue
        peer = ollama_map.get((item.round_index, item.case_id))
        if peer and peer.quality_score >= 0.0:
            deltas.append(item.quality_score - peer.quality_score)
    return {"avg_delta": round(statistics.fmean(deltas), 2) if deltas else None, "count": len(deltas)}


def format_quality_delta(delta_doc: Dict[str, Any]) -> str:
    count = int(delta_doc.get("count", 0) or 0)
    avg_delta = delta_doc.get("avg_delta")
    if count <= 0 or avg_delta is None:
        return "N/A"
    return f"{float(avg_delta):.2f}"


def md_route_summary(name: str, summary: Dict[str, Any], wall_ms: float) -> str:
    qps = (summary["ok"] / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0
    lines = [
        f"### {name}",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 总请求数 | {summary['total']} |",
        f"| 成功数 | {summary['ok']} |",
        f"| 失败数 | {summary['failed']} |",
        f"| 成功率 | {summary['success_rate']:.2f}% |",
        f"| 平均延迟(ms) | {summary['latency_avg_ms']:.2f} |",
        f"| P95 延迟(ms) | {summary['latency_p95_ms']:.2f} |",
        f"| 延迟标准差(ms) | {summary['latency_std_ms']:.2f} |",
        f"| 延迟均值95%CI(ms) | [{summary['latency_ci95_low_ms']:.2f}, {summary['latency_ci95_high_ms']:.2f}] |",
        f"| 实测吞吐(成功QPS) | {qps:.2f} |",
        f"| 速度分平均(0-100) | {summary['speed_score_avg']:.2f} |",
        f"| 质量样本数 | {summary['quality_total']} |",
        f"| 智能度平均(0-100) | {summary['quality_avg']:.2f} |",
        f"| 质量通过率 | {summary['quality_pass_rate']:.2f}% |",
        f"| 综合分(速度40/智能60) | {summary['balanced_score']:.2f} |",
    ]
    lines.append("")
    return "\n".join(lines)


def md_rounds(route_name: str, rounds: List[Dict[str, Any]]) -> str:
    if not rounds:
        return f"### {route_name} 轮次统计\n\n无数据。\n"
    lines = [f"### {route_name} 轮次统计", "", "| 轮次 | 成功率 | 平均延迟(ms) | 速度分平均 | 智能度平均 |", "|---:|---:|---:|---:|---:|"]
    for item in rounds:
        lines.append(f"| {item['round']} | {item['successRate']:.2f}% | {item['latencyAvgMs']:.2f} | {item['speedScoreAvg']:.2f} | {item['qualityAvg']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def md_task_types(route_name: str, task_types: Dict[str, Any]) -> str:
    if not task_types:
        return f"### {route_name} 任务分布\n\n无质量评测数据。\n"
    lines = [f"### {route_name} 任务分布", "", "| taskType | total | transportPassed | qualityPassed | qualityPassRate | scoreAvg | latencyAvgMs |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task_type, bucket in sorted(task_types.items()):
        lines.append(f"| {task_type} | {bucket['total']} | {bucket['transportPassed']} | {bucket['qualityPassed']} | {bucket['qualityPassRate']:.2f}% | {bucket['scoreAvg']:.2f} | {bucket['latencyAvgMs']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def md_errors(title: str, errors: List[Dict[str, Any]], max_items: int = 12) -> str:
    if not errors:
        return f"### {title}\n\n无失败样本。\n"
    lines = [f"### {title}", "", "| 轮次 | 用例 | 状态码 | 延迟(ms) | 错误 | Prompt(截断) |", "|---:|---|---:|---:|---|---|"]
    for item in errors[:max_items]:
        prompt = str(item.get("prompt", "")).replace("\n", " ")
        if len(prompt) > 72:
            prompt = prompt[:69] + "..."
        err = str(item.get("error", "")).replace("\n", " ")
        if len(err) > 120:
            err = err[:117] + "..."
        lines.append(f"| {item.get('round', 0)} | {item.get('case_id', 'n/a')} | {item.get('status', 0)} | {float(item.get('latency_ms', 0.0)):.2f} | {err} | {prompt} |")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace,
                 quality_cases: List[BenchCase],
                 standard_cases: List[BenchCase],
                 external_cases: List[BenchCase],
                 questionnaire_cases: List[BenchCase],
                 route_payloads: Dict[str, Dict[str, Any]],
                 comparisons: Dict[str, Any],
                 deps: BenchDeps,
                 system_management: Dict[str, Any],
                 ollama_management: Dict[str, Any],
                 external_index: Dict[str, Any],
                 benchmark_metadata: Dict[str, Any],
                 local_dataset_metadata: Dict[str, Any],
                 progress_state: Dict[str, Any]) -> str:
    system_summary = route_payloads[ROUTE_SYSTEM]["summary"]
    ollama_summary = route_payloads[ROUTE_OLLAMA]["summary"]
    llamacpp_enabled = bool(getattr(args, "enable_llamacpp", False))
    llamacpp_payload = route_payloads.get(ROUTE_LLAMACPP, {"summary": summarize_route([]), "wall_ms": 0.0})
    llamacpp_summary = llamacpp_payload["summary"]
    better_latency = "系统API" if 0 < system_summary["latency_avg_ms"] < (ollama_summary["latency_avg_ms"] or float("inf")) else "直连Ollama"
    better_quality = "系统API" if system_summary["quality_avg"] >= ollama_summary["quality_avg"] else "直连Ollama"
    better_balance = "系统API" if system_summary["balanced_score"] >= ollama_summary["balanced_score"] else "直连Ollama"
    questionnaire_files = getattr(args, "resolved_questionnaire_files", []) or []
    stability_state = getattr(args, "stability_state", {}) or {}
    stability_rounds = stability_state.get("rounds", {}) if isinstance(stability_state.get("rounds"), dict) else {}
    local_dataset_count = int(local_dataset_metadata.get("count", 0) or 0)
    manual_external_count = max(0, len(external_cases) - local_dataset_count)
    report_complete = bool(progress_state.get("complete", False))
    title = "# AI 综合基准报告" if report_complete else "# AI 综合基准报告（未完成快照）"

    lines = [title, ""]
    if not report_complete:
        lines.extend([
            "## 运行状态",
            "",
            "- 当前报告是未完成快照。若某一路由计数未满或为 0，表示该路由尚未跑完或进程被中断，不能据此判定模型能力为 0。",
            "",
        ])
    lines.extend([
        f"- 生成时间: {now_str()}",
        f"- 系统API: `{args.system_url}`",
        f"- Ollama直连: `{args.ollama_url}`",
        f"- llama.cpp直连: `{args.llamacpp_url if llamacpp_enabled else 'disabled'}`",
        f"- llama.cpp接口样式: `{getattr(args, 'llamacpp_api_style', 'disabled') if llamacpp_enabled else 'disabled'}`",
        f"- 选中模型: `{args.ollama_model}`",
        f"- llama.cpp模型: `{args.llamacpp_model if llamacpp_enabled else 'disabled'}`",
        f"- 质量基线用例: `{args.cases_file}`，样本数={len(quality_cases)}",
        f"- 标准基准集: `{', '.join(benchmark_metadata.get('selected', [])) or 'disabled'}`，样本数={len(standard_cases)}",
        f"- 标准基准加载: 成功={len(benchmark_metadata.get('loaded', []))}，失败={len(benchmark_metadata.get('failed', []))}，缓存目录=`{args.benchmark_cache_dir}`，源=`{args.dataset_server_url}`，重试={args.benchmark_fetch_retries}，cacheOnly={args.benchmark_cache_only}，源可达={benchmark_metadata.get('sourceReachable', True)}",
        f"- 质量问卷配对样本: 文件数={len(args.external_dataset_files)}，样本数={manual_external_count}，上限={args.external_limit}，配对文本源=`{args.external_prompts_file}` / `{args.external_answers_file}`",
        f"- 本地 tests QA 数据集: 自动发现={args.auto_discover_tests_datasets}，文件数={len(local_dataset_metadata.get('files', []))}，样本数={local_dataset_metadata.get('count', 0)}，上限={args.tests_dataset_limit}",
        f"- GPT4all 共享抽样: {args.shared_local_qa}，共享样本数={local_dataset_metadata.get('sharedCount', 0)}",
        f"- 速度问卷样本: 文件数={len(questionnaire_files)}，样本数={len(questionnaire_cases)}，上限={args.questionnaire_limit}",
        f"- 轮次: {args.rounds}",
        f"- 并发: {args.concurrency}",
        f"- 超时(s): {args.timeout}",
        f"- 随机采样: {args.shuffle_cases}，随机种子={args.random_seed}",
        f"- 稳定性早停: {args.stability_stop}，检查间隔={args.stability_check_interval}，最小样本={args.stability_min_samples}，窗口={args.stability_window}",
        f"- Ollama线程覆盖(API): disabled，兼容参数请求值={args.ollama_num_thread}",
        f"- HTTP后端: {deps.http_backend}",
        f"- System自动管理: {args.auto_manage_system}，动作={system_management.get('action', 'n/a')}，就绪={system_management.get('ready', False)}",
        f"- Ollama自动管理: {args.auto_manage_ollama}，动作={ollama_management.get('action', 'n/a')}，就绪={ollama_management.get('ready', False)}",
        f"- 断点续测: {args.resume}，快照间隔={args.checkpoint_every}，完成状态={progress_state.get('complete', False)}",
        f"- 路由进度: system={progress_state.get('routes', {}).get(ROUTE_SYSTEM, {}).get('completed', 0)}/{progress_state.get('routes', {}).get(ROUTE_SYSTEM, {}).get('expected', 0)}，ollama={progress_state.get('routes', {}).get(ROUTE_OLLAMA, {}).get('completed', 0)}/{progress_state.get('routes', {}).get(ROUTE_OLLAMA, {}).get('expected', 0)}，llamacpp={progress_state.get('routes', {}).get(ROUTE_LLAMACPP, {}).get('completed', 0)}/{progress_state.get('routes', {}).get(ROUTE_LLAMACPP, {}).get('expected', 0)}",
        f"- 外部索引激活数据集: {external_index.get('activeDataset', 'n/a')}",
        f"- 可用库: httpx={deps.httpx_available}, numpy={deps.numpy_available}, scipy={deps.scipy_available}, psutil={deps.psutil_available}, rouge={deps.rouge_available}, sacrebleu={deps.sacrebleu_available}",
        "",
        "## 结论摘要",
        "",
        f"- 速度更优: **{better_latency}**",
        f"- 智能度更优: **{better_quality}**",
        f"- 综合分更优: **{better_balance}**",
        f"- 质量分差(System - Ollama): **{format_quality_delta(comparisons['quality_delta'])}**",
    ])
    if llamacpp_enabled or llamacpp_summary["total"] > 0:
        llama_cmp = comparisons.get("llamacpp_vs_ollama", {}) if isinstance(comparisons.get("llamacpp_vs_ollama"), dict) else {}
        llama_quality_delta = llama_cmp.get("quality_delta", {}) if isinstance(llama_cmp.get("quality_delta"), dict) else {}
        lines.append(f"- 质量分差(llama.cpp - Ollama): **{format_quality_delta(llama_quality_delta)}**")
    if stability_rounds:
        early_stop_rounds = sum(1 for item in stability_rounds.values() if isinstance(item, dict) and item.get("stoppedEarly"))
        lines.append(f"- 稳定性早停触发轮次: **{early_stop_rounds}**")
    latency_sig = comparisons["latency_significance"]
    if latency_sig.get("test") == "mannwhitneyu":
        lines.append(f"- 延迟显著性: `{latency_sig['test']}`, p={latency_sig['p_value']:.6f}")
    else:
        lines.append(f"- 延迟显著性: `{latency_sig.get('test', 'n/a')}`")

    if stability_rounds:
        lines.extend([
            "",
            "## 稳定性早停详情",
            "",
            "| 轮次 | 计划样本 | 实际样本 | 早停 | 最近检查样本 | 原因 |",
            "|---:|---:|---:|---|---:|---|",
        ])
        for round_key in sorted(stability_rounds, key=lambda item: int(item)):
            item = stability_rounds.get(round_key, {}) if isinstance(stability_rounds.get(round_key), dict) else {}
            last_check = item.get("lastCheck", {}) if isinstance(item.get("lastCheck"), dict) else {}
            lines.append(
                f"| {round_key} | {int(item.get('planned', 0) or 0)} | {int(item.get('executed', 0) or 0)} | {'yes' if item.get('stoppedEarly') else 'no'} | {int(last_check.get('samples', 0) or 0)} | {item.get('reason', 'n/a')} |"
            )
        lines.append("")

    lines.extend([
        "",
        md_route_summary(f"当前系统 API（{llamacpp_label(args.system_url)}）", system_summary, route_payloads[ROUTE_SYSTEM]["wall_ms"]),
        md_route_summary(f"直连 Ollama（{llamacpp_label(args.ollama_url)}）", ollama_summary, route_payloads[ROUTE_OLLAMA]["wall_ms"]),
        md_rounds("系统 API", system_summary["rounds"]),
        md_rounds("直连 Ollama", ollama_summary["rounds"]),
        md_task_types("系统 API", system_summary["task_types"]),
        md_task_types("直连 Ollama", ollama_summary["task_types"]),
        md_errors("系统 API 错误详情", system_summary["errors"]),
        md_errors("Ollama 错误详情", ollama_summary["errors"]),
    ])
    if llamacpp_enabled or llamacpp_summary["total"] > 0:
        lines.extend([
            md_route_summary(f"直连 llama.cpp（{llamacpp_label(args.llamacpp_url)}）", llamacpp_summary, llamacpp_payload["wall_ms"]),
            md_rounds("直连 llama.cpp", llamacpp_summary["rounds"]),
            md_task_types("直连 llama.cpp", llamacpp_summary["task_types"]),
            md_errors("llama.cpp 错误详情", llamacpp_summary["errors"]),
        ])
    return "\n".join(lines)


def build_report_payload(args: argparse.Namespace,
                         quality_cases: List[BenchCase],
                         standard_cases: List[BenchCase],
                         external_cases: List[BenchCase],
                         questionnaire_cases: List[BenchCase],
                         route_payloads: Dict[str, Dict[str, Any]],
                         comparisons: Dict[str, Any],
                         deps: BenchDeps,
                         system_management: Dict[str, Any],
                         ollama_management: Dict[str, Any],
                         external_index: Dict[str, Any],
                         benchmark_metadata: Dict[str, Any],
                         local_dataset_metadata: Dict[str, Any],
                         progress_state: Dict[str, Any]) -> Dict[str, Any]:
    local_dataset_count = int(local_dataset_metadata.get("count", 0) or 0)
    manual_external_count = max(0, len(external_cases) - local_dataset_count)
    return {
        "generatedAt": now_str(),
        "metadata": {
            "system_url": args.system_url,
            "ollama_url": args.ollama_url,
            "llamacpp_url": args.llamacpp_url,
            "llamacpp_api_style": getattr(args, "llamacpp_api_style", "ollama-chat"),
            "llamacpp_requested_url": getattr(args, "llamacpp_requested_url", args.llamacpp_url),
            "ollama_model": args.ollama_model,
            "llamacpp_model": args.llamacpp_model,
            "enable_llamacpp": args.enable_llamacpp,
            "random_seed": args.random_seed,
            "shuffle_cases": args.shuffle_cases,
            "ollama_num_thread_requested": args.ollama_num_thread,
            "ollama_num_thread_applied": False,
            "ollama_warmup_timeout": args.ollama_warmup_timeout,
            "cases_file": args.cases_file,
            "benchmark_presets": args.benchmark_presets,
            "benchmark_cache_dir": args.benchmark_cache_dir,
            "benchmark_limit_per_preset": args.benchmark_limit_per_preset,
            "dataset_server_url": args.dataset_server_url,
            "benchmark_fetch_retries": args.benchmark_fetch_retries,
            "benchmark_cache_only": args.benchmark_cache_only,
            "external_dataset_files": args.external_dataset_files,
            "auto_discover_tests_datasets": args.auto_discover_tests_datasets,
            "shared_local_qa": args.shared_local_qa,
            "tests_dataset_limit": args.tests_dataset_limit,
            "external_prompts_file": args.external_prompts_file,
            "external_answers_file": args.external_answers_file,
            "external_index_file": args.external_index_file,
            "external_limit": args.external_limit,
            "questionnaire_file": args.questionnaire_file,
            "questionnaire_files": getattr(args, "resolved_questionnaire_files", []),
            "quality_case_count": len(quality_cases),
            "standard_case_count": len(standard_cases),
            "external_case_count": manual_external_count,
            "local_tests_dataset_case_count": local_dataset_count,
            "questionnaire_count": len(questionnaire_cases),
            "rounds": args.rounds,
            "concurrency": args.concurrency,
            "timeout_s": args.timeout,
            "max_tokens": args.max_tokens,
            "http_backend": deps.http_backend,
            "questionnaire_limit": args.questionnaire_limit,
            "resume": args.resume,
            "checkpoint_every": args.checkpoint_every,
            "stability": getattr(args, "stability_state", {}),
        },
        "deps": {
            "httpx": deps.httpx_available,
            "numpy": deps.numpy_available,
            "scipy": deps.scipy_available,
            "psutil": deps.psutil_available,
            "rouge": deps.rouge_available,
            "sacrebleu": deps.sacrebleu_available,
        },
        "system_management": system_management,
        "ollama_management": ollama_management,
        "benchmark_metadata": benchmark_metadata,
        "local_dataset_metadata": local_dataset_metadata,
        "progress": progress_state,
        "external_index": external_index,
        "routes": route_payloads,
        "comparisons": comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run combined speed + intelligence benchmark against system API and direct Ollama")
    parser.add_argument("--system-url", default=DEFAULT_SYSTEM_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llamacpp-url", default=DEFAULT_LLAMACPP_URL)
    parser.add_argument("--ollama-model", default="")
    parser.add_argument("--llamacpp-model", default="llamacpp")
    parser.add_argument("--enable-llamacpp", action="store_true", help="Run an additional direct llama.cpp route via --llamacpp-url and keep existing routes/results for comparison")
    parser.add_argument("--preferred-models", default=",".join(DEFAULT_PREFERRED_MODELS), help="Comma separated preferred Ollama model names")
    parser.add_argument("--auto-manage-ollama", action="store_true", default=True)
    parser.add_argument("--no-auto-manage-ollama", action="store_false", dest="auto_manage_ollama")
    parser.add_argument("--restart-unhealthy-ollama", action="store_true", default=True)
    parser.add_argument("--no-restart-unhealthy-ollama", action="store_false", dest="restart_unhealthy_ollama")
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--external-dataset-files", nargs="*", default=[])
    parser.add_argument("--auto-discover-tests-datasets", action="store_true", default=True)
    parser.add_argument("--no-auto-discover-tests-datasets", action="store_false", dest="auto_discover_tests_datasets")
    parser.add_argument("--shared-local-qa", action="store_true", help="Sample both quality and questionnaire cases from the same auto-discovered local QA dataset pool, such as tests/GPT4all")
    parser.add_argument("--tests-dataset-limit", type=int, default=DEFAULT_TESTS_DATASET_LIMIT, help="Maximum QA pairs auto-loaded from local tests datasets such as tests/GPT4all")
    parser.add_argument("--benchmark-presets", nargs="*", default=list(DEFAULT_STANDARD_BENCHMARKS), help="Public benchmark presets downloaded via Hugging Face datasets-server")
    parser.add_argument("--benchmark-cache-dir", default=DEFAULT_BENCHMARK_CACHE_DIR)
    parser.add_argument("--benchmark-limit-per-preset", type=int, default=DEFAULT_BENCHMARK_LIMIT_PER_PRESET)
    parser.add_argument("--dataset-server-url", default=DEFAULT_DATASET_SERVER_ROWS_URL)
    parser.add_argument("--benchmark-fetch-retries", type=int, default=2)
    parser.add_argument("--benchmark-cache-only", action="store_true")
    parser.add_argument("--refresh-benchmark-cache", action="store_true")
    parser.add_argument("--no-standard-benchmarks", action="store_true")
    parser.add_argument("--external-prompts-file", default=DEFAULT_EXTERNAL_PROMPTS_FILE)
    parser.add_argument("--external-answers-file", default=DEFAULT_EXTERNAL_ANSWERS_FILE)
    parser.add_argument("--external-index-file", default=DEFAULT_EXTERNAL_INDEX_FILE)
    parser.add_argument("--external-limit", type=int, default=DEFAULT_EXTERNAL_LIMIT, help="0 means use all external paired cases")
    parser.add_argument("--external-threshold", type=float, default=5.0)
    parser.add_argument("--questionnaire-file", default=DEFAULT_QUESTIONNAIRE_FILE)
    parser.add_argument("--questionnaire-files", nargs="*", default=[])
    parser.add_argument("--questionnaire-limit", type=int, default=DEFAULT_QUESTIONNAIRE_LIMIT, help="Total questionnaire prompt cap; default 1000, 0 means use all questionnaire prompts")
    parser.add_argument("--prompts-file", default="", help="Legacy prompts file; when provided and cases-file is empty, script falls back to prompt/answer mode")
    parser.add_argument("--answers-file", default="")
    parser.add_argument("--system-token", default="")
    parser.add_argument("--system-launch-command-json", default="")
    parser.add_argument("--login-url", default="")
    parser.add_argument("--auth-username", default="")
    parser.add_argument("--auth-password", default="")
    parser.add_argument("--require-login", action="store_true")
    parser.add_argument("--auto-manage-system", action="store_true", default=True)
    parser.add_argument("--no-auto-manage-system", action="store_false", dest="auto_manage_system")
    parser.add_argument("--system-start-timeout", type=float, default=DEFAULT_SYSTEM_START_TIMEOUT)
    parser.add_argument("--system-retry-count", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2, help="How many full-suite rounds to run")
    parser.add_argument("--repeats", type=int, default=0, help="Legacy alias; used when rounds is not provided explicitly")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ollama-num-thread", type=int, default=0, help="Deprecated compatibility flag; ignored to avoid Ollama model reload")
    parser.add_argument("--ollama-warmup-timeout", type=float, default=DEFAULT_OLLAMA_WARMUP_TIMEOUT, help="Warmup timeout for initial Ollama model load")
    parser.add_argument("--http-backend", choices=["auto", "urllib", "httpx"], default="auto")
    parser.add_argument("--warmup", type=int, default=1, help="Round 1 preheat count for direct Ollama")
    parser.add_argument("--shuffle-cases", action="store_true", default=True)
    parser.add_argument("--no-shuffle-cases", action="store_false", dest="shuffle_cases")
    parser.add_argument("--random-seed", type=int, default=0, help="0 means auto-generate a seed for randomized sampling")
    parser.add_argument("--stability-stop", action="store_true", default=True)
    parser.add_argument("--no-stability-stop", action="store_false", dest="stability_stop")
    parser.add_argument("--stability-check-interval", type=int, default=DEFAULT_STABILITY_CHECK_INTERVAL, help="Check early-stop stability after each N paired samples")
    parser.add_argument("--stability-min-samples", type=int, default=DEFAULT_STABILITY_MIN_SAMPLES, help="Minimum paired samples required before early-stop can trigger")
    parser.add_argument("--stability-window", type=int, default=DEFAULT_STABILITY_WINDOW, help="How many recent stability checkpoints must stay within thresholds")
    parser.add_argument("--stability-quality-delta", type=float, default=DEFAULT_STABILITY_QUALITY_DELTA)
    parser.add_argument("--stability-balanced-delta", type=float, default=DEFAULT_STABILITY_BALANCED_DELTA)
    parser.add_argument("--stability-success-rate-delta", type=float, default=DEFAULT_STABILITY_SUCCESS_RATE_DELTA)
    parser.add_argument("--stability-latency-ratio", type=float, default=DEFAULT_STABILITY_LATENCY_RATIO)
    parser.add_argument("--stability-latency-delta-ms", type=float, default=DEFAULT_STABILITY_LATENCY_DELTA_MS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--output", default="")
    parser.add_argument("--json-output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats > 0 and args.rounds <= 0:
        args.rounds = args.repeats
    args.rounds = max(1, args.rounds)

    http_backend = args.http_backend
    if http_backend == "auto":
        http_backend = "httpx" if httpx is not None else "urllib"
    if http_backend == "httpx" and httpx is None:
        raise RuntimeError("http-backend=httpx but httpx not installed")

    deps = BenchDeps(
        http_backend=http_backend,
        httpx_available=httpx is not None,
        numpy_available=np is not None,
        scipy_available=scipy_stats is not None,
        psutil_available=psutil is not None,
        rouge_available=rouge_scorer is not None,
        sacrebleu_available=sacrebleu is not None,
    )

    if args.no_standard_benchmarks:
        args.benchmark_presets = []
    args.benchmark_presets = normalize_preset_names(args.benchmark_presets)
    args.checkpoint_every = max(0, int(args.checkpoint_every))
    args.ollama_num_thread = max(0, int(args.ollama_num_thread))
    args.ollama_warmup_timeout = max(args.timeout, float(args.ollama_warmup_timeout))
    args.benchmark_fetch_retries = max(0, int(args.benchmark_fetch_retries))
    args.tests_dataset_limit = max(0, int(args.tests_dataset_limit))
    args.random_seed = int(args.random_seed or 0)
    args.stability_check_interval = max(1, int(args.stability_check_interval))
    args.stability_min_samples = max(1, int(args.stability_min_samples))
    args.stability_window = max(2, int(args.stability_window))
    args.stability_quality_delta = max(0.0, float(args.stability_quality_delta))
    args.stability_balanced_delta = max(0.0, float(args.stability_balanced_delta))
    args.stability_success_rate_delta = max(0.0, float(args.stability_success_rate_delta))
    args.stability_latency_ratio = max(0.0, float(args.stability_latency_ratio))
    args.stability_latency_delta_ms = max(0.0, float(args.stability_latency_delta_ms))

    if args.ollama_num_thread > 0:
        print(f"[OLLAMA] ignore --ollama-num-thread={args.ollama_num_thread} to avoid forcing model reload")

    out_path, json_out_path = resolve_report_paths(args)
    resume_payload = load_resume_payload(json_out_path, args.resume)
    resume_metadata = resume_payload.get("metadata", {}) if isinstance(resume_payload.get("metadata"), dict) else {}
    if not args.ollama_model:
        args.ollama_model = str(resume_metadata.get("ollama_model", "") or "")
    if not args.llamacpp_model:
        args.llamacpp_model = str(resume_metadata.get("llamacpp_model", "") or "")
    if args.random_seed <= 0:
        args.random_seed = int(resume_metadata.get("random_seed", 0) or 0)
    if args.random_seed <= 0:
        args.random_seed = int(time.time_ns() & 0x7FFFFFFF)
    random.seed(args.random_seed)
    args.llamacpp_requested_url = args.llamacpp_url
    args.llamacpp_api_style = infer_llamacpp_style(parse_url_base_path(args.llamacpp_url)[1])
    args.system_launch_command = parse_system_launch_command(args.system_launch_command_json)

    route_payloads = restore_route_payloads(resume_payload) if resume_payload else {
        ROUTE_SYSTEM: {"results": [], "wall_ms": 0.0},
        ROUTE_OLLAMA: {"results": [], "wall_ms": 0.0},
        ROUTE_LLAMACPP: {"results": [], "wall_ms": 0.0},
    }
    if args.resume and resume_payload:
        print(f"[RESUME] loaded snapshot: {json_out_path}")
        print(f"[RESUME] system={len(route_payloads[ROUTE_SYSTEM]['results'])}, ollama={len(route_payloads[ROUTE_OLLAMA]['results'])}, llamacpp={len(route_payloads[ROUTE_LLAMACPP]['results'])}")

    local_dataset_files = discover_local_qa_dataset_files() if args.auto_discover_tests_datasets else []
    local_dataset_metadata = {
        "files": local_dataset_files,
        "count": 0,
        "sharedCount": 0,
    }
    quality_cases: List[BenchCase] = []
    external_cases: List[BenchCase] = []
    questionnaire_cases: List[BenchCase] = []

    if args.shared_local_qa:
        if not local_dataset_files:
            raise RuntimeError("shared-local-qa enabled but no auto-discovered local QA dataset was found")
        quality_cases, questionnaire_cases, shared_metadata = load_shared_local_qa_cases(
            local_dataset_files,
            args.tests_dataset_limit,
            args.external_threshold,
        )
        if not quality_cases or not questionnaire_cases:
            raise RuntimeError("shared-local-qa enabled but no usable GPT4all/local QA samples were loaded")
        local_dataset_metadata.update(shared_metadata)
        local_dataset_metadata["count"] = len(quality_cases)
        args.cases_file = "shared-local-qa://tests/GPT4all"
        args.external_prompts_file = ""
        args.external_answers_file = ""
        args.questionnaire_file = ""
        args.questionnaire_files = []
        args.resolved_questionnaire_files = []
    else:
        if args.cases_file:
            quality_cases = load_quality_cases(args.cases_file)
        elif args.prompts_file:
            quality_cases = load_prompt_answer_cases(args.prompts_file, args.answers_file)
        else:
            quality_cases = load_quality_cases(DEFAULT_CASES_FILE)
            args.cases_file = DEFAULT_CASES_FILE
    standard_cases, benchmark_metadata = load_standard_benchmark_cases(
        args.benchmark_presets,
        args.benchmark_cache_dir,
        args.refresh_benchmark_cache,
        args.benchmark_limit_per_preset,
        args.timeout,
        args.dataset_server_url,
        args.benchmark_fetch_retries,
        args.benchmark_cache_only,
    )
    for failed in benchmark_metadata.get("failed", []):
        print(f"[BENCH] preset failed: {failed.get('preset', 'unknown')} => {failed.get('error', 'unknown-error')}")
    for file_path in args.external_dataset_files:
        external_cases.extend(load_external_dataset_file(file_path, args.external_threshold, args.external_limit))
    if not args.shared_local_qa:
        for file_path in local_dataset_files:
            local_cases = load_external_dataset_file(file_path, args.external_threshold, args.tests_dataset_limit)
            external_cases.extend(local_cases)
            local_dataset_metadata["count"] += len(local_cases)
        if args.external_prompts_file and args.external_answers_file:
            external_cases.extend(load_external_line_pairs(args.external_prompts_file, args.external_answers_file, args.external_limit, args.external_threshold))

    external_index = load_external_index(args.external_index_file)

    if not args.shared_local_qa:
        args.resolved_questionnaire_files = resolve_questionnaire_files(args.questionnaire_file, args.questionnaire_files)
        questionnaire_cases = load_questionnaire_cases(args.resolved_questionnaire_files, max(0, args.questionnaire_limit))
    all_cases = quality_cases + standard_cases + external_cases + questionnaire_cases
    if not all_cases:
        raise RuntimeError("no benchmark cases loaded")
    args.planned_case_counts_by_round = {round_index: len(all_cases) for round_index in range(1, args.rounds + 1)}
    restored_stability = resume_metadata.get("stability", {}) if isinstance(resume_metadata.get("stability"), dict) else {}
    args.stability_state = {
        "enabled": bool(args.stability_stop),
        "randomSeed": int(args.random_seed),
        "checkInterval": int(args.stability_check_interval),
        "minSamples": int(args.stability_min_samples),
        "window": int(args.stability_window),
        "qualityDelta": float(args.stability_quality_delta),
        "balancedDelta": float(args.stability_balanced_delta),
        "successRateDelta": float(args.stability_success_rate_delta),
        "latencyRatio": float(args.stability_latency_ratio),
        "latencyDeltaMs": float(args.stability_latency_delta_ms),
        "rounds": restored_stability.get("rounds", {}) if isinstance(restored_stability.get("rounds"), dict) else {},
    }
    for round_key, round_doc in args.stability_state["rounds"].items():
        if isinstance(round_doc, dict):
            try:
                args.planned_case_counts_by_round[int(round_key)] = int(round_doc.get("executed", round_doc.get("planned", len(all_cases))) or len(all_cases))
            except Exception:
                continue

    shared_httpx_client = httpx.Client(timeout=args.timeout) if http_backend == "httpx" else None

    system_management: Dict[str, Any] = {"ready": True, "action": "skip", "error": ""}
    ollama_management: Dict[str, Any] = {"ready": True, "action": "skip", "error": "", "models": []}
    system_token = (args.system_token or "").strip()

    system_pending = route_has_pending_results(ROUTE_SYSTEM, args, all_cases, route_payloads)
    ollama_pending = route_has_pending_results(ROUTE_OLLAMA, args, all_cases, route_payloads)
    llamacpp_pending = args.enable_llamacpp and route_has_pending_results(ROUTE_LLAMACPP, args, all_cases, route_payloads)

    if system_pending:
        system_management = ensure_system_service(args.system_url, args.auto_manage_system, max(5.0, args.system_start_timeout), args.system_launch_command)
        if not system_management.get("ready"):
            raise RuntimeError(f"system service not ready: {system_management.get('error', 'unknown-error')}")

        if not system_token and args.auth_username and args.auth_password:
            ok, token, err = login_and_get_token(args.system_url, args.login_url, args.auth_username, args.auth_password, args.timeout)
            if ok:
                system_token = token
                print("[AUTH] login success, JWT acquired")
            else:
                print(f"[AUTH] login failed: {err}")
        if not system_token:
            ok, token, source = auto_register_and_get_token(args.system_url, args.timeout)
            if ok:
                system_token = token
                print(f"[AUTH] acquired JWT via {source}")
            else:
                print(f"[AUTH] auto-register unavailable: {source}")
        if args.require_login and not system_token:
            raise RuntimeError("require-login enabled but no JWT available")

        system_ready, system_err = probe_system_chat(args.system_url, system_token, max(args.timeout, 30.0), http_backend, shared_httpx_client)
        if not system_ready and args.auto_manage_system:
            recovered, recover_msg = recover_system_service(
                args.system_url,
                system_token,
                max(args.timeout, 30.0),
                http_backend,
                shared_httpx_client,
                args.auto_manage_system,
                args.system_start_timeout,
                args.system_launch_command,
            )
            if recovered:
                system_ready, system_err = probe_system_chat(args.system_url, system_token, max(args.timeout, 30.0), http_backend, shared_httpx_client)
            else:
                system_err = recover_msg
        if not system_ready:
            raise RuntimeError(f"system chat not ready: {system_err}")
        print(f"[SYSTEM] warmup probe ok auth={'yes' if system_token else 'no'}")
    elif args.require_login and not system_token:
        raise RuntimeError("require-login enabled but no JWT available")

    if ollama_pending:
        ollama_management = ensure_ollama_service(args.ollama_url, args.auto_manage_ollama, args.restart_unhealthy_ollama)
        if not ollama_management.get("ready"):
            raise RuntimeError(f"ollama not ready: {ollama_management.get('error', 'unknown-error')}")

        args.ollama_model = discover_or_choose_model(args, ollama_management.get("models", []))
        if not args.ollama_model:
            raise RuntimeError("failed to select an Ollama model")
        print(f"[OLLAMA] selected model: {args.ollama_model}")

        if args.warmup > 0:
            warmup = warmup_ollama_model(args.ollama_url, args.ollama_model, args.ollama_warmup_timeout, http_backend, shared_httpx_client)
            if not warmup.get("ok") and args.auto_manage_ollama and args.restart_unhealthy_ollama:
                print(f"[OLLAMA] warmup failed once: {warmup.get('error', '')}; retrying service recovery")
                ollama_management = ensure_ollama_service(args.ollama_url, True, True)
                if not ollama_management.get("ready"):
                    raise RuntimeError(f"ollama recover failed: {ollama_management.get('error', 'unknown-error')}")
                args.ollama_model = discover_or_choose_model(args, ollama_management.get("models", []))
                if not args.ollama_model:
                    raise RuntimeError("failed to re-select an Ollama model after recovery")
                warmup = warmup_ollama_model(args.ollama_url, args.ollama_model, args.ollama_warmup_timeout, http_backend, shared_httpx_client)
            if not warmup.get("ok"):
                raise RuntimeError(f"ollama chat not ready: {warmup.get('error', 'warmup failed')}")
            print(f"[OLLAMA] warmup complete reply={warmup.get('reply', '')[:40]}")

    if llamacpp_pending:
        llama_probe = resolve_llamacpp_endpoint(args.llamacpp_url, max(args.timeout, 10.0))
        if not llama_probe.get("ready"):
            raise RuntimeError(f"llama.cpp not ready: {llama_probe.get('error', 'unknown-error')}")
        args.llamacpp_url = str(llama_probe.get("url", args.llamacpp_url))
        args.llamacpp_api_style = str(llama_probe.get("style", args.llamacpp_api_style))
        if llama_probe.get("note"):
            print(f"[LLAMACPP] {llama_probe.get('note')}")
        print(f"[LLAMACPP] selected endpoint: {args.llamacpp_url} style={args.llamacpp_api_style}")

    for round_index in range(1, args.rounds + 1):
        print(f"[ROUND] {round_index}/{args.rounds} starting")
        planned_cases = build_round_case_plan(all_cases, args.shuffle_cases)
        round_state = args.stability_state["rounds"].get(str(round_index), {}) if isinstance(args.stability_state["rounds"].get(str(round_index)), dict) else {}
        round_state.setdefault("planned", len(planned_cases))
        round_state.setdefault("executed", 0)
        round_state.setdefault("checks", 0)
        round_state.setdefault("stoppedEarly", False)
        round_state.setdefault("reason", "planned-complete")
        round_state.setdefault("lastCheck", {})
        args.stability_state["rounds"][str(round_index)] = round_state
        if round_state.get("stoppedEarly"):
            restored_count = max(0, min(len(planned_cases), int(round_state.get("executed", round_state.get("planned", len(planned_cases))) or len(planned_cases))))
            planned_cases = planned_cases[:restored_count]
            args.planned_case_counts_by_round[round_index] = restored_count

        existing_system_keys = build_existing_result_keys(route_payloads[ROUTE_SYSTEM]["results"])
        existing_ollama_keys = build_existing_result_keys(route_payloads[ROUTE_OLLAMA]["results"])
        existing_llamacpp_keys = build_existing_result_keys(route_payloads[ROUTE_LLAMACPP]["results"])
        system_complete = sum(1 for case in planned_cases if (round_index, case.case_id) in existing_system_keys) == len(planned_cases)
        ollama_complete = sum(1 for case in planned_cases if (round_index, case.case_id) in existing_ollama_keys) == len(planned_cases)
        llamacpp_complete = (not args.enable_llamacpp) or (sum(1 for case in planned_cases if (round_index, case.case_id) in existing_llamacpp_keys) == len(planned_cases))
        if system_complete and ollama_complete and llamacpp_complete:
            round_state["executed"] = len(planned_cases)
            args.planned_case_counts_by_round[round_index] = len(planned_cases)
            print(f"[RESUME] skip round={round_index}, all enabled routes already present")
        else:
            stability_history: List[Dict[str, Any]] = []
            executed_count = 0
            for start in range(0, len(planned_cases), args.stability_check_interval):
                batch_cases = planned_cases[start:start + args.stability_check_interval]

                system_base_results = list(route_payloads["system"]["results"])
                system_base_wall = float(route_payloads["system"]["wall_ms"])

                def system_progress(current_results: List[RouteCaseResult], current_wall_ms: float) -> None:
                    snapshot_routes = {
                        ROUTE_SYSTEM: {"results": system_base_results + current_results, "wall_ms": system_base_wall + current_wall_ms},
                        ROUTE_OLLAMA: {"results": list(route_payloads[ROUTE_OLLAMA]["results"]), "wall_ms": float(route_payloads[ROUTE_OLLAMA]["wall_ms"])},
                        ROUTE_LLAMACPP: {"results": list(route_payloads[ROUTE_LLAMACPP]["results"]), "wall_ms": float(route_payloads[ROUTE_LLAMACPP]["wall_ms"])},
                    }
                    persist_report_artifacts(args, out_path, json_out_path, quality_cases, standard_cases, external_cases, questionnaire_cases, snapshot_routes, deps, system_management, ollama_management, external_index, benchmark_metadata, local_dataset_metadata)

                system_results, system_wall_ms = run_route_cases(
                    route_name=ROUTE_SYSTEM,
                    cases=batch_cases,
                    round_index=round_index,
                    concurrency=args.concurrency,
                    timeout_s=args.timeout,
                    system_url=args.system_url,
                    system_token=system_token,
                    system_launch_command=args.system_launch_command,
                    ollama_url=args.ollama_url,
                    ollama_model=args.ollama_model,
                    llamacpp_url=args.llamacpp_url,
                    llamacpp_model=args.llamacpp_model,
                    max_tokens=args.max_tokens,
                    http_backend=http_backend,
                    httpx_client=shared_httpx_client,
                    shuffle_cases=False,
                    system_retry_count=args.system_retry_count,
                    auto_manage_system=args.auto_manage_system,
                    system_start_timeout=args.system_start_timeout,
                    llamacpp_style=args.llamacpp_api_style,
                    completed_keys=existing_system_keys,
                    checkpoint_every=args.checkpoint_every,
                    progress_callback=system_progress,
                )
                route_payloads[ROUTE_SYSTEM]["results"].extend(system_results)
                route_payloads[ROUTE_SYSTEM]["wall_ms"] += system_wall_ms
                existing_system_keys.update((round_index, case.case_id) for case in batch_cases)

                ollama_base_results = list(route_payloads[ROUTE_OLLAMA]["results"])
                ollama_base_wall = float(route_payloads[ROUTE_OLLAMA]["wall_ms"])

                def ollama_progress(current_results: List[RouteCaseResult], current_wall_ms: float) -> None:
                    snapshot_routes = {
                        ROUTE_SYSTEM: {"results": list(route_payloads[ROUTE_SYSTEM]["results"]), "wall_ms": float(route_payloads[ROUTE_SYSTEM]["wall_ms"])},
                        ROUTE_OLLAMA: {"results": ollama_base_results + current_results, "wall_ms": ollama_base_wall + current_wall_ms},
                        ROUTE_LLAMACPP: {"results": list(route_payloads[ROUTE_LLAMACPP]["results"]), "wall_ms": float(route_payloads[ROUTE_LLAMACPP]["wall_ms"])},
                    }
                    persist_report_artifacts(args, out_path, json_out_path, quality_cases, standard_cases, external_cases, questionnaire_cases, snapshot_routes, deps, system_management, ollama_management, external_index, benchmark_metadata, local_dataset_metadata)

                ollama_results, ollama_wall_ms = run_route_cases(
                    route_name=ROUTE_OLLAMA,
                    cases=batch_cases,
                    round_index=round_index,
                    concurrency=args.concurrency,
                    timeout_s=args.timeout,
                    system_url=args.system_url,
                    system_token=system_token,
                    system_launch_command=args.system_launch_command,
                    ollama_url=args.ollama_url,
                    ollama_model=args.ollama_model,
                    llamacpp_url=args.llamacpp_url,
                    llamacpp_model=args.llamacpp_model,
                    max_tokens=args.max_tokens,
                    http_backend=http_backend,
                    httpx_client=shared_httpx_client,
                    shuffle_cases=False,
                    system_retry_count=args.system_retry_count,
                    auto_manage_system=args.auto_manage_system,
                    system_start_timeout=args.system_start_timeout,
                    llamacpp_style=args.llamacpp_api_style,
                    completed_keys=existing_ollama_keys,
                    checkpoint_every=args.checkpoint_every,
                    progress_callback=ollama_progress,
                )
                route_payloads[ROUTE_OLLAMA]["results"].extend(ollama_results)
                route_payloads[ROUTE_OLLAMA]["wall_ms"] += ollama_wall_ms
                existing_ollama_keys.update((round_index, case.case_id) for case in batch_cases)

                if args.enable_llamacpp:
                    llamacpp_base_results = list(route_payloads[ROUTE_LLAMACPP]["results"])
                    llamacpp_base_wall = float(route_payloads[ROUTE_LLAMACPP]["wall_ms"])

                    def llamacpp_progress(current_results: List[RouteCaseResult], current_wall_ms: float) -> None:
                        snapshot_routes = {
                            ROUTE_SYSTEM: {"results": list(route_payloads[ROUTE_SYSTEM]["results"]), "wall_ms": float(route_payloads[ROUTE_SYSTEM]["wall_ms"])},
                            ROUTE_OLLAMA: {"results": list(route_payloads[ROUTE_OLLAMA]["results"]), "wall_ms": float(route_payloads[ROUTE_OLLAMA]["wall_ms"])},
                            ROUTE_LLAMACPP: {"results": llamacpp_base_results + current_results, "wall_ms": llamacpp_base_wall + current_wall_ms},
                        }
                        persist_report_artifacts(args, out_path, json_out_path, quality_cases, standard_cases, external_cases, questionnaire_cases, snapshot_routes, deps, system_management, ollama_management, external_index, benchmark_metadata, local_dataset_metadata)

                    llamacpp_results, llamacpp_wall_ms = run_route_cases(
                        route_name=ROUTE_LLAMACPP,
                        cases=batch_cases,
                        round_index=round_index,
                        concurrency=args.concurrency,
                        timeout_s=args.timeout,
                        system_url=args.system_url,
                        system_token=system_token,
                        system_launch_command=args.system_launch_command,
                        ollama_url=args.ollama_url,
                        ollama_model=args.ollama_model,
                        llamacpp_url=args.llamacpp_url,
                        llamacpp_model=args.llamacpp_model,
                        max_tokens=args.max_tokens,
                        http_backend=http_backend,
                        httpx_client=shared_httpx_client,
                        shuffle_cases=False,
                        system_retry_count=args.system_retry_count,
                        auto_manage_system=args.auto_manage_system,
                        system_start_timeout=args.system_start_timeout,
                        llamacpp_style=args.llamacpp_api_style,
                        completed_keys=existing_llamacpp_keys,
                        checkpoint_every=args.checkpoint_every,
                        progress_callback=llamacpp_progress,
                    )
                    route_payloads[ROUTE_LLAMACPP]["results"].extend(llamacpp_results)
                    route_payloads[ROUTE_LLAMACPP]["wall_ms"] += llamacpp_wall_ms
                    existing_llamacpp_keys.update((round_index, case.case_id) for case in batch_cases)

                executed_count = min(len(planned_cases), start + len(batch_cases))
                round_state["executed"] = executed_count
                round_state["checks"] = int(round_state.get("checks", 0) or 0) + 1
                persist_report_artifacts(args, out_path, json_out_path, quality_cases, standard_cases, external_cases, questionnaire_cases, route_payloads, deps, system_management, ollama_management, external_index, benchmark_metadata, local_dataset_metadata)

                if args.stability_stop and executed_count < len(planned_cases):
                    round_system_results = current_round_results(route_payloads[ROUTE_SYSTEM]["results"], round_index)
                    round_ollama_results = current_round_results(route_payloads[ROUTE_OLLAMA]["results"], round_index)
                    stable, stable_detail = evaluate_round_stability(round_system_results, round_ollama_results, stability_history, args)
                    round_state["lastCheck"] = stable_detail
                    if stable:
                        round_state["stoppedEarly"] = True
                        round_state["reason"] = "metrics-stable"
                        args.planned_case_counts_by_round[round_index] = executed_count
                        print(f"[ROUND] {round_index} early stop at {executed_count}/{len(planned_cases)} paired samples")
                        break

            if not round_state.get("stoppedEarly"):
                round_state["reason"] = "planned-complete"
                round_state["executed"] = len(planned_cases)
                args.planned_case_counts_by_round[round_index] = len(planned_cases)
        persist_report_artifacts(args, out_path, json_out_path, quality_cases, standard_cases, external_cases, questionnaire_cases, route_payloads, deps, system_management, ollama_management, external_index, benchmark_metadata, local_dataset_metadata)

    system_result_objects = list(route_payloads[ROUTE_SYSTEM]["results"])
    ollama_result_objects = list(route_payloads[ROUTE_OLLAMA]["results"])
    llamacpp_result_objects = list(route_payloads[ROUTE_LLAMACPP]["results"])
    materialized_routes = materialize_route_payloads(route_payloads)
    comparisons = {
        "latency_significance": compare_latency_significance(system_result_objects, ollama_result_objects),
        "quality_delta": compare_quality_delta(system_result_objects, ollama_result_objects),
    }
    if args.enable_llamacpp or llamacpp_result_objects:
        comparisons["llamacpp_vs_ollama"] = {
            "latency_significance": compare_latency_significance(llamacpp_result_objects, ollama_result_objects),
            "quality_delta": compare_quality_delta(llamacpp_result_objects, ollama_result_objects),
        }
    progress_state = build_progress_state(args, all_cases, route_payloads)
    report = build_report(args, quality_cases, standard_cases, external_cases, questionnaire_cases, materialized_routes, comparisons, deps, system_management, ollama_management, external_index, benchmark_metadata, local_dataset_metadata, progress_state)
    payload = build_report_payload(args, quality_cases, standard_cases, external_cases, questionnaire_cases, materialized_routes, comparisons, deps, system_management, ollama_management, external_index, benchmark_metadata, local_dataset_metadata, progress_state)

    atomic_write_text(out_path, report)
    atomic_write_text(json_out_path, json.dumps(payload, ensure_ascii=False, indent=2))

    if shared_httpx_client is not None:
        try:
            shared_httpx_client.close()
        except Exception:
            pass

    print(f"[OK] report written: {out_path}")
    print(f"[OK] json written: {json_out_path}")
    finished_ok = progress_state.get("complete", False)
    for route_name in enabled_route_names(args):
        finished_ok = finished_ok and materialized_routes[route_name]["summary"]["failed"] == 0
    return 0 if finished_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

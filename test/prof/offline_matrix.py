#!/usr/bin/env python3
import argparse
import json
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request
from urllib.parse import urlparse


DEFAULT_PLAN_FILE = "test/prof/offline_matrix_plan.json"
DEFAULT_SYSTEM_URL = "http://127.0.0.1:5080/api/chat"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_LLAMACPP_URL = "http://127.0.0.1:8082/v1/chat/completions"
DEFAULT_HAI_CASES_FILE = "test/intelligence/cases.baseline.json"


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_plan(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("presets"), list):
        return [dict(item) for item in payload["presets"] if isinstance(item, dict)]
    raise ValueError("plan file must be a JSON array or an object containing presets[]")


def as_arg_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        return shlex.split(value, posix=False)
    return []


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if not text:
        return default
    return text not in {"0", "false", "off", "no", "disabled"}


def to_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def format_metric(value: Any, missing: str = "n/a") -> str:
    try:
        if value is None:
            return missing
        number = float(value)
    except Exception:
        return missing
    if number < 0.0:
        return missing
    return f"{number:.2f}"


def csv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(ch in text for ch in {",", '"', "\n", "\r"}):
        return '"' + text.replace('"', '""') + '"'
    return text


def summary_base_name(output_dir: Path) -> str:
    return f"{output_dir.name or 'offline_matrix'}_summary"


def task_type_buckets(summary: Any) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    for key in ("taskTypes", "task_types"):
        value = summary.get(key)
        if isinstance(value, dict):
            return value
    return {}


def classify_result_status(benchmark_exit_code: int,
                           benchmark_payload: Dict[str, Any],
                           hai_enabled: bool,
                           hai_exit_code: Optional[int],
                           hai_payload: Dict[str, Any]) -> str:
    benchmark_ready = bool(benchmark_payload)
    hai_ready = (not hai_enabled) or bool(hai_payload)
    benchmark_ok = benchmark_exit_code == 0 and benchmark_ready
    hai_ok = (not hai_enabled) or (hai_exit_code == 0 and hai_ready)
    if benchmark_ok and hai_ok:
        return "ok"
    if benchmark_ready and hai_ready:
        return "regression"
    return "error"


def write_text_copies(paths: Sequence[Path], content: str) -> None:
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def build_component_args(preset: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    config_path = str(preset.get("componentConfig", "") or "").strip()
    if config_path:
        args.append(f"--component-config={config_path}")
    components = str(preset.get("components", "") or "").strip()
    if components:
        args.append(f"--components={components}")
    return args


def component_label(preset: Dict[str, Any]) -> str:
    labels: List[str] = []
    config_path = str(preset.get("componentConfig", "") or "").strip()
    components = str(preset.get("components", "") or "").strip()
    if config_path:
        summary = summarize_component_config(config_path)
        labels.append(summary or config_path)
    if components:
        labels.append(f"CLI:{components}")
    return " + ".join(labels) if labels else "default"


def benchmark_mode(preset: Dict[str, Any]) -> str:
    mode = str(preset.get("benchmarkMode", "shared-local-qa") or "shared-local-qa").strip().lower()
    if mode in {"shared", "shared-local-qa", "shared_local_qa", "gpt4all", "gpt4all-shared"}:
        return "shared-local-qa"
    return mode or "shared-local-qa"


def hai_cases_file(preset: Dict[str, Any]) -> str:
    return str(preset.get("haiCasesFile", DEFAULT_HAI_CASES_FILE) or DEFAULT_HAI_CASES_FILE).strip()


def flatten_component_json(value: Any, prefix: str = "") -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            flattened.update(flatten_component_json(child, next_prefix))
        return flattened
    if prefix:
        if isinstance(value, bool):
            flattened[prefix] = "true" if value else "false"
        else:
            flattened[prefix] = str(value)
    return flattened


def load_component_selection(config_path: str) -> Dict[str, str]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_absolute():
        path = workspace_root() / path
    if not path.exists() or not path.is_file():
        return {}
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            return {}
        return flatten_component_json(payload)
    if suffix == ".xml":
        flattened: Dict[str, str] = {}
        pattern = re.compile(r'<component\s+[^>]*path="([^"]+)"[^>]*(?:enabled="([^"]+)"|value="([^"]+)")[^>]*/?>', re.IGNORECASE)
        for match in pattern.finditer(text):
            key = str(match.group(1) or "").strip()
            enabled = match.group(2)
            value = match.group(3)
            if not key:
                continue
            flattened[key] = (enabled if enabled is not None else value or "").strip()
        return flattened
    return {}


def summarize_component_config(config_path: str) -> str:
    flattened = load_component_selection(config_path)
    if not flattened:
        return ""
    ordered = [f"{key}={flattened[key]}" for key in sorted(flattened)]
    return "; ".join(ordered)


def system_origin(system_url: str) -> Tuple[str, str, int]:
    parsed = urlparse(system_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, host, port


def system_base_url(system_url: str) -> str:
    scheme, host, port = system_origin(system_url)
    return f"{scheme}://{host}:{port}"


def system_health_url(system_url: str) -> str:
    return system_base_url(system_url) + "/api/model/lifecycle"


def is_tcp_port_open(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def wait_http_ready(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = "service-not-ready"
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=min(5.0, timeout_s)) as response:
                if response.status < 500:
                    return
        except error.HTTPError as exc:
            if exc.code < 500:
                return
            last_error = f"http-{exc.code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"wait_http_ready timeout: {url} ({last_error})")


def post_json(url: str,
              payload: Dict[str, Any],
              timeout_s: float,
              headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
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

    result = payload.get("result", payload)
    candidates: List[str] = []
    if isinstance(result, dict):
        for key in ["reply", "transformerReply", "graphReply"]:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    for key in ["reply", "text", "answer"]:
        value = payload.get(key)
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
    return any(
        marker in lowered
        for marker in (
            "connection refused",
            "actively refused",
            "timed out",
            "timeout",
            "bad gateway",
            "service unavailable",
            "empty-reply",
        )
    )


def wait_system_chat_ready(system_url: str, system_token: str, timeout_s: float) -> None:
    payload = {
        "text": "hai warmup ping",
        "maxTokens": 16,
        "sessionId": f"hai-health-{int(time.time() * 1000)}",
    }
    headers = {"Authorization": f"Bearer {system_token}"} if system_token else None
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    attempt = 0
    last_error = "system-chat-probe-failed"
    per_request_timeout = max(15.0, min(30.0, float(timeout_s)))

    while True:
        attempt += 1
        status, raw = post_json(system_url, payload, per_request_timeout, headers=headers)
        if status == 0:
            ok = False
            err = raw
        else:
            ok, _reply, err = parse_system_reply(raw)
        if ok:
            return

        last_error = str(err or (f"http-{status}" if status else "system-chat-probe-failed"))
        if time.monotonic() >= deadline or not is_transient_system_probe_error(status, last_error):
            break

        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            break
        time.sleep(min(2.0, 0.5 * attempt, remaining))

    raise RuntimeError(f"wait_system_chat_ready timeout: {system_url} ({last_error})")


def find_phoenix_executable() -> Optional[Path]:
    root = workspace_root()
    for candidate in [root / "phoenix_main.exe", root / "build" / "phoenix_main.exe"]:
        if candidate.exists():
            return candidate
    return None


def find_python_executable(explicit: str) -> Path:
    if explicit:
        return Path(explicit)
    local_python = workspace_root() / "Python314" / "python.exe"
    if local_python.exists():
        return local_python
    return Path(sys.executable)


def has_gguf_magic(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except Exception:
        return False


def iter_manifest_backed_gguf_blobs(gguf_root: Path) -> List[Path]:
    manifests_root = gguf_root / "manifests"
    blobs_root = gguf_root / "blobs"
    if not manifests_root.exists() or not blobs_root.exists():
        return []

    candidates: List[Path] = []
    seen: set[str] = set()
    for manifest_path in manifests_root.rglob("*"):
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        layers = payload.get("layers") if isinstance(payload, dict) else None
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            media_type = str(layer.get("mediaType", "") or "").strip().lower()
            digest = str(layer.get("digest", "") or "").strip()
            if media_type != "application/vnd.ollama.image.model" or not digest.startswith("sha256:"):
                continue
            blob_path = blobs_root / f"sha256-{digest.split(':', 1)[1]}"
            normalized = str(blob_path.resolve()) if blob_path.exists() else str(blob_path)
            if normalized in seen:
                continue
            seen.add(normalized)
            if blob_path.is_file() and has_gguf_magic(blob_path):
                candidates.append(blob_path)
    return candidates


def has_any_gguf(gguf_root: Optional[Path] = None) -> bool:
    gguf_root = gguf_root or (workspace_root() / "GGUF_models")
    if not gguf_root.exists():
        return False
    if any(path.is_file() for path in gguf_root.rglob("*.gguf")):
        return True
    return bool(iter_manifest_backed_gguf_blobs(gguf_root))


def kill_named_processes(names: Sequence[str]) -> None:
    if sys.platform != "win32":
        return
    for name in names:
        subprocess.run(["taskkill", "/IM", name, "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def build_system_args(preset: Dict[str, Any], system_url: str) -> List[str]:
    _scheme, host, port = system_origin(system_url)
    study_port = int(preset.get("studyPort", 5081 if port != 5081 else 5082))
    args = [
        f"--gateway-host={host}",
        f"--port={port}",
        f"--study-port={study_port}",
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
    backend = str(preset.get("backend", "ollama") or "ollama").strip().lower()
    if backend:
        args.append(f"--transformer-mode={backend}")
    ollama_model = str(preset.get("ollamaModel", "") or "").strip()
    if ollama_model:
        args.append(f"--ollama-model={ollama_model}")
    llamacpp_model = str(preset.get("llamacppModel", "") or "").strip()
    if llamacpp_model:
        args.append(f"--llamacpp-model={llamacpp_model}")
    bitnet_model = str(preset.get("bitnetModel", "") or "").strip()
    if bitnet_model:
        args.append(f"--bitnet-model={bitnet_model}")
    args.extend(build_component_args(preset))
    args.extend(as_arg_list(preset.get("systemArgs", [])))
    return args


def should_skip_preset(preset: Dict[str, Any]) -> str:
    backend = str(preset.get("backend", "") or "").strip().lower()
    if backend == "llamacpp" and bool(preset.get("requireGguf", False)) and not has_any_gguf():
        return "llamacpp preset requires local GGUF models"
    return ""


def direct_llamacpp_ready(llamacpp_url: str, timeout_s: float) -> bool:
    _scheme, host, port = system_origin(llamacpp_url)
    return is_tcp_port_open(host, port, timeout_s)


def should_enable_direct_llamacpp(preset: Dict[str, Any]) -> bool:
    backend = str(preset.get("backend", "") or "").strip().lower()
    return as_bool(preset.get("enableDirectLlamacpp"), backend == "llamacpp")


def run_command(command: Sequence[str], cwd: Path) -> int:
    completed = subprocess.run(list(command), cwd=str(cwd), check=False)
    return int(completed.returncode)


def start_system_process(exe_path: Path, preset: Dict[str, Any], system_url: str) -> subprocess.Popen:
    system_args = [str(exe_path), *build_system_args(preset, system_url)]
    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return subprocess.Popen(
        system_args,
        cwd=str(workspace_root()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def stop_system_process(process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    try:
        process.terminate()
    except Exception:
        pass


def ensure_system_ready_for_hai(process: Optional[subprocess.Popen],
                                exe_path: Path,
                                preset: Dict[str, Any],
                                args: argparse.Namespace) -> Tuple[Optional[subprocess.Popen], bool, str]:
    health_url = system_health_url(args.system_url)
    process_alive = process is not None and process.poll() is None
    probe_timeout = max(3.0, min(args.startup_timeout / 3.0, 12.0))
    chat_timeout = max(args.timeout, args.startup_timeout, 30.0)
    failure_reason = ""
    try:
        wait_http_ready(health_url, probe_timeout)
        wait_system_chat_ready(args.system_url, args.system_token, chat_timeout)
        health_ready = True
    except Exception as exc:
        health_ready = False
        failure_reason = str(exc)

    if process_alive and health_ready:
        return process, False, ""

    reason = "system process exited after benchmark"
    if process_alive and failure_reason:
        reason = failure_reason

    stop_system_process(process)
    kill_named_processes(["phoenix_main.exe", "bug_shooter.exe"])
    restarted = start_system_process(exe_path, preset, args.system_url)
    wait_http_ready(health_url, args.startup_timeout)
    wait_system_chat_ready(args.system_url, args.system_token, chat_timeout)
    return restarted, True, reason


def safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in name.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "preset"


def read_report_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def route_summary(payload: Dict[str, Any], route_name: str) -> Dict[str, Any]:
    routes = payload.get("routes", {}) if isinstance(payload, dict) else {}
    route_doc = routes.get(route_name, {}) if isinstance(routes, dict) else {}
    summary = route_doc.get("summary", {}) if isinstance(route_doc, dict) else {}
    return summary if isinstance(summary, dict) else {}


def benchmark_quality_source(payload: Dict[str, Any]) -> str:
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("cases_file", "") or "").strip()


def benchmark_quality_delta(payload: Dict[str, Any]) -> Optional[float]:
    comparisons = payload.get("comparisons", {}) if isinstance(payload, dict) else {}
    if not isinstance(comparisons, dict):
        return None
    quality_delta = comparisons.get("quality_delta", {})
    if not isinstance(quality_delta, dict):
        return None
    avg_delta = quality_delta.get("avg_delta")
    try:
        return float(avg_delta) if avg_delta is not None else None
    except Exception:
        return None


def extract_hai_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    hai = payload.get("hai", {}) if isinstance(payload, dict) else {}
    gates = payload.get("gates", {}) if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(hai, dict):
        hai = {}
    if not isinstance(gates, dict):
        gates = {}
    return {
        "total": int(summary.get("total", 0) or 0),
        "transportPassed": int(summary.get("transportPassed", 0) or 0),
        "transportFailed": int(summary.get("transportFailed", 0) or 0),
        "transportPassRate": to_float(summary.get("transportPassRate", 0.0), 0.0),
        "qualityPassed": int(summary.get("qualityPassed", 0) or 0),
        "qualityPassRate": to_float(summary.get("qualityPassRate", 0.0), 0.0),
        "scoreAvg": to_float(summary.get("scoreAvg", -1.0), -1.0),
        "ollamaScoreAvg": to_float(summary.get("ollamaScoreAvg", -1.0), -1.0),
        "scoreDeltaVsOllamaAvg": to_float(summary.get("scoreDeltaVsOllamaAvg", 0.0), 0.0),
        "overall": to_float(hai.get("overall", -1.0), -1.0),
        "coverage": to_float(hai.get("coverage", 0.0), 0.0),
        "gatesPassed": bool(gates.get("passed", False)),
        "taskTypes": task_type_buckets(summary),
    }


def build_benchmark_command(args: argparse.Namespace,
                            python_exe: Path,
                            exe_path: Path,
                            preset: Dict[str, Any],
                            report_md: Path,
                            report_json: Path) -> List[str]:
    system_launch_command = json.dumps([str(exe_path), *build_system_args(preset, args.system_url)], ensure_ascii=False)
    command = [
        str(python_exe),
        str(workspace_root() / "test" / "prof" / "main.py"),
        "--system-url", args.system_url,
        "--ollama-url", args.ollama_url,
        "--llamacpp-url", args.llamacpp_url,
        "--ollama-model", str(preset.get("benchmarkOllamaModel", args.ollama_model)),
        "--system-token", args.system_token,
        "--system-launch-command-json", system_launch_command,
        "--no-standard-benchmarks",
        "--benchmark-cache-only",
        "--rounds", str(args.rounds),
        "--concurrency", str(args.concurrency),
        "--timeout", str(args.timeout),
        "--max-tokens", str(args.max_tokens),
        "--ollama-warmup-timeout", str(args.ollama_warmup_timeout),
        "--stability-check-interval", str(max(4, min(args.tests_dataset_limit, 8))),
        "--stability-min-samples", str(max(4, min(args.tests_dataset_limit, 8))),
        "--stability-window", "2",
        "--output", str(report_md),
        "--json-output", str(report_json),
    ]
    if benchmark_mode(preset) == "shared-local-qa":
        command.extend([
            "--shared-local-qa",
            "--tests-dataset-limit", str(args.tests_dataset_limit),
        ])
    else:
        command.extend(as_arg_list(preset.get("benchmarkArgs", [])))
    if should_enable_direct_llamacpp(preset) and not args.skip_direct_llamacpp and direct_llamacpp_ready(args.llamacpp_url, 2.0):
        command.append("--enable-llamacpp")
    command.extend(as_arg_list(preset.get("benchArgs", [])))
    return command


def build_hai_command(args: argparse.Namespace,
                      python_exe: Path,
                      preset: Dict[str, Any],
                      report_md: Path,
                      report_json: Path) -> List[str]:
    ollama_model = str(preset.get("haiOllamaModel", preset.get("benchmarkOllamaModel", args.ollama_model)) or "").strip()
    command = [
        str(python_exe),
        str(workspace_root() / "test" / "intelligence" / "main.py"),
        "--system-url", args.system_url,
        "--system-token", args.system_token,
        "--ollama-url", args.ollama_url,
        "--cases-file", hai_cases_file(preset),
        "--output-md", str(report_md),
        "--output-json", str(report_json),
        "--timeout", str(args.timeout),
        "--max-tokens", str(args.max_tokens),
    ]
    if ollama_model:
        command.extend(["--ollama-model", ollama_model])
    command.extend(as_arg_list(preset.get("haiArgs", [])))
    return command


def run_single_preset(args: argparse.Namespace,
                      python_exe: Path,
                      preset: Dict[str, Any],
                      output_dir: Path) -> Dict[str, Any]:
    skip_reason = should_skip_preset(preset)
    base_result = {
        "name": str(preset.get("name", "unnamed")),
        "backend": str(preset.get("backend", "ollama")),
        "benchmarkMode": benchmark_mode(preset),
        "componentConfig": str(preset.get("componentConfig", "") or ""),
        "components": str(preset.get("components", "") or ""),
        "componentLabel": component_label(preset),
        "systemArgs": as_arg_list(preset.get("systemArgs", [])),
        "haiEnabled": as_bool(preset.get("runHai", False), False),
        "haiCasesFile": hai_cases_file(preset),
        "haiSystemRestarted": False,
        "haiSystemRestartReason": "",
    }
    if skip_reason:
        return {
            **base_result,
            "status": "skipped",
            "reason": skip_reason,
            "exitCode": None,
            "haiExitCode": None,
            "qualitySource": "",
            "qualityDeltaVsOllama": None,
            "reportMarkdown": "",
            "reportJson": "",
            "haiReportMarkdown": "",
            "haiReportJson": "",
            "hai": {},
            "system": {},
            "ollama": {},
            "llamacpp": {},
        }

    exe_path = find_phoenix_executable()
    if exe_path is None:
        raise RuntimeError("phoenix_main.exe not found")

    preset_name = str(preset.get("name", "unnamed"))
    preset_dir = output_dir / safe_name(preset_name)
    preset_dir.mkdir(parents=True, exist_ok=True)
    report_md = preset_dir / "benchmark_report.md"
    report_json = preset_dir / "benchmark_report.json"
    hai_md = preset_dir / "hai_eval_report.md"
    hai_json = preset_dir / "hai_eval_report.json"

    kill_named_processes(["phoenix_main.exe", "bug_shooter.exe"])
    process = start_system_process(exe_path, preset, args.system_url)
    try:
        wait_http_ready(system_health_url(args.system_url), args.startup_timeout)
        command = build_benchmark_command(args, python_exe, exe_path, preset, report_md, report_json)
        exit_code = run_command(command, workspace_root())
        payload = read_report_payload(report_json)
        hai_enabled = as_bool(preset.get("runHai", False), False)
        hai_exit_code: Optional[int] = None
        hai_payload: Dict[str, Any] = {}
        hai_system_restarted = False
        hai_restart_reason = ""
        if hai_enabled:
            process, hai_system_restarted, hai_restart_reason = ensure_system_ready_for_hai(process, exe_path, preset, args)
            hai_command = build_hai_command(args, python_exe, preset, hai_md, hai_json)
            hai_exit_code = run_command(hai_command, workspace_root())
            hai_payload = read_report_payload(hai_json)
        status = classify_result_status(exit_code, payload, hai_enabled, hai_exit_code, hai_payload)
        return {
            **base_result,
            "name": preset_name,
            "backend": str(preset.get("backend", "ollama")),
            "haiEnabled": hai_enabled,
            "haiSystemRestarted": hai_system_restarted,
            "haiSystemRestartReason": hai_restart_reason,
            "status": status,
            "reason": "",
            "exitCode": exit_code,
            "haiExitCode": hai_exit_code,
            "qualitySource": benchmark_quality_source(payload),
            "qualityDeltaVsOllama": benchmark_quality_delta(payload),
            "reportMarkdown": str(report_md),
            "reportJson": str(report_json),
            "haiReportMarkdown": str(hai_md) if hai_enabled else "",
            "haiReportJson": str(hai_json) if hai_enabled else "",
            "hai": extract_hai_summary(hai_payload) if hai_enabled else {},
            "system": route_summary(payload, "system"),
            "ollama": route_summary(payload, "ollama"),
            "llamacpp": route_summary(payload, "llamacpp"),
        }
    except Exception as exc:
        return {
            **base_result,
            "status": "error",
            "reason": str(exc),
            "exitCode": None,
            "haiExitCode": None,
            "qualitySource": "",
            "qualityDeltaVsOllama": None,
            "reportMarkdown": str(report_md),
            "reportJson": str(report_json),
            "haiReportMarkdown": str(hai_md) if as_bool(preset.get("runHai", False), False) else "",
            "haiReportJson": str(hai_json) if as_bool(preset.get("runHai", False), False) else "",
            "hai": {},
            "system": {},
            "ollama": {},
            "llamacpp": {},
        }
    finally:
        stop_system_process(process)
        kill_named_processes(["phoenix_main.exe", "bug_shooter.exe"])


def write_aggregate_report(results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = summary_base_name(output_dir)
    counts = {
        "ok": sum(1 for result in results if result.get("status") == "ok"),
        "regression": sum(1 for result in results if result.get("status") == "regression"),
        "error": sum(1 for result in results if result.get("status") == "error"),
        "skipped": sum(1 for result in results if result.get("status") == "skipped"),
    }
    payload = {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
        "results": results,
    }

    primary_json = output_dir / f"{base_name}.json"
    primary_md = output_dir / f"{base_name}.md"
    primary_csv = output_dir / f"{base_name}.csv"
    alias_json = output_dir.parent / f"{base_name}.json"
    alias_md = output_dir.parent / f"{base_name}.md"
    alias_csv = output_dir.parent / f"{base_name}.csv"

    write_text_copies(
        [primary_json, alias_json],
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )

    overview_rows: List[Dict[str, str]] = []
    benchmark_task_rows: List[Dict[str, str]] = []
    hai_task_rows: List[Dict[str, str]] = []

    for result in results:
        system_summary = result.get("system", {}) if isinstance(result.get("system"), dict) else {}
        ollama_summary = result.get("ollama", {}) if isinstance(result.get("ollama"), dict) else {}
        hai_summary = result.get("hai", {}) if isinstance(result.get("hai"), dict) else {}
        overview_row = {
            "Preset": str(result.get("name", "unnamed")),
            "Backend": str(result.get("backend", "n/a")),
            "Status": str(result.get("status", "n/a")),
            "Reason": str(result.get("reason", "") or "-"),
            "BenchmarkMode": str(result.get("benchmarkMode", "shared-local-qa")),
            "ComponentSelection": str(result.get("componentLabel", "") or "default"),
            "SystemArgs": " ".join(result.get("systemArgs", [])) if isinstance(result.get("systemArgs"), list) and result.get("systemArgs") else "n/a",
            "QualitySource": str(result.get("qualitySource", "") or "n/a"),
            "SystemSuccessRate": format_metric(system_summary.get("success_rate")),
            "SystemQuality": format_metric(system_summary.get("quality_avg")),
            "SystemBalanced": format_metric(system_summary.get("balanced_score")),
            "OllamaQuality": format_metric(ollama_summary.get("quality_avg")),
            "DeltaVsOllama": format_metric(result.get("qualityDeltaVsOllama")),
            "HAITransportPassRate": format_metric(hai_summary.get("transportPassRate")),
            "HAIQualityPassRate": format_metric(hai_summary.get("qualityPassRate")),
            "HAIOverall": format_metric(hai_summary.get("overall")),
            "HAIScore": format_metric(hai_summary.get("scoreAvg")),
            "HAIDeltaVsOllama": format_metric(hai_summary.get("scoreDeltaVsOllamaAvg")),
            "HAIGatesPassed": str(bool(hai_summary.get("gatesPassed", False))).lower() if result.get("haiEnabled") else "disabled",
            "BenchmarkReport": str(result.get("reportMarkdown", "") or "n/a"),
            "HAIReport": str(result.get("haiReportMarkdown", "") or ("disabled" if not result.get("haiEnabled") else "n/a")),
        }
        overview_rows.append(overview_row)

        system_task_types = task_type_buckets(system_summary)
        if system_task_types:
            for task_type, bucket in sorted(system_task_types.items()):
                benchmark_task_rows.append({
                    "Preset": overview_row["Preset"],
                    "Status": overview_row["Status"],
                    "TaskType": str(task_type),
                    "Total": str(int(bucket.get("total", 0) or 0)),
                    "TransportPassed": str(int(bucket.get("transportPassed", 0) or 0)),
                    "QualityPassed": str(int(bucket.get("qualityPassed", 0) or 0)),
                    "QualityPassRate": format_metric(bucket.get("qualityPassRate")),
                    "ScoreAvg": format_metric(bucket.get("scoreAvg")),
                    "LatencyAvgMs": format_metric(bucket.get("latencyAvgMs")),
                })
        else:
            benchmark_task_rows.append({
                "Preset": overview_row["Preset"],
                "Status": overview_row["Status"],
                "TaskType": "-",
                "Total": "0",
                "TransportPassed": "0",
                "QualityPassed": "0",
                "QualityPassRate": "n/a",
                "ScoreAvg": "n/a",
                "LatencyAvgMs": "n/a",
            })

        hai_task_types = task_type_buckets(hai_summary)
        if hai_task_types:
            for task_type, bucket in sorted(hai_task_types.items()):
                hai_task_rows.append({
                    "Preset": overview_row["Preset"],
                    "Status": overview_row["Status"],
                    "TaskType": str(task_type),
                    "Total": str(int(bucket.get("total", 0) or 0)),
                    "TransportPassed": str(int(bucket.get("transportPassed", 0) or 0)),
                    "QualityPassed": str(int(bucket.get("qualityPassed", 0) or 0)),
                    "QualityPassRate": format_metric(bucket.get("qualityPassRate")),
                    "ScoreAvg": format_metric(bucket.get("scoreAvg")),
                    "LatencyAvgMs": format_metric(bucket.get("latencyAvgMs")),
                })
        else:
            hai_task_rows.append({
                "Preset": overview_row["Preset"],
                "Status": overview_row["Status"],
                "TaskType": "-",
                "Total": "0",
                "TransportPassed": "0",
                "QualityPassed": "0",
                "QualityPassRate": "n/a",
                "ScoreAvg": "n/a",
                "LatencyAvgMs": "n/a",
            })

    lines = [
        "# Offline Matrix Summary",
        "",
        f"- Generated At: {payload['generatedAt']}",
        f"- Status Counts: ok={counts['ok']}, regression={counts['regression']}, error={counts['error']}, skipped={counts['skipped']}",
        f"- Summary JSON: {primary_json}",
        f"- Summary CSV: {primary_csv}",
        f"- Root Alias Markdown: {alias_md}",
        "- 注意：综合 benchmark 默认质量源是 shared-local-qa/GPT4all 抽样；HAI 报告使用固定 baseline 用例。两者不能直接当作同一分数口径比较。",
        "",
        "## Full Matrix",
        "",
        "| Preset | Backend | Status | Reason | Benchmark Mode | Component Selection | System Args | Quality Source | System Success | System Quality | System Balanced | Ollama Quality | Delta | HAI Transport | HAI Quality | HAI Overall | HAI Score | HAI Delta | HAI Gates | Benchmark Report | HAI Report |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in overview_rows:
        lines.append(
            "| {Preset} | {Backend} | {Status} | {Reason} | {BenchmarkMode} | {ComponentSelection} | {SystemArgs} | {QualitySource} | {SystemSuccessRate} | {SystemQuality} | {SystemBalanced} | {OllamaQuality} | {DeltaVsOllama} | {HAITransportPassRate} | {HAIQualityPassRate} | {HAIOverall} | {HAIScore} | {HAIDeltaVsOllama} | {HAIGatesPassed} | {BenchmarkReport} | {HAIReport} |".format(
                **row,
            )
        )
    lines.extend([
        "",
        "## Benchmark Task Matrix",
        "",
        "| Preset | Status | Task Type | Total | Transport Passed | Quality Passed | Quality Pass Rate | Score Avg | Latency Avg(ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in benchmark_task_rows:
        lines.extend([
            "| {Preset} | {Status} | {TaskType} | {Total} | {TransportPassed} | {QualityPassed} | {QualityPassRate} | {ScoreAvg} | {LatencyAvgMs} |".format(**row),
        ])
    lines.extend([
        "",
        "## HAI Task Matrix",
        "",
        "| Preset | Status | Task Type | Total | Transport Passed | Quality Passed | Quality Pass Rate | Score Avg | Latency Avg(ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in hai_task_rows:
        lines.append(
            "| {Preset} | {Status} | {TaskType} | {Total} | {TransportPassed} | {QualityPassed} | {QualityPassRate} | {ScoreAvg} | {LatencyAvgMs} |".format(**row)
        )

    write_text_copies([primary_md, alias_md], "\n".join(lines) + "\n")

    csv_headers = [
        "Preset",
        "Backend",
        "Status",
        "Reason",
        "BenchmarkMode",
        "ComponentSelection",
        "SystemArgs",
        "QualitySource",
        "SystemSuccessRate",
        "SystemQuality",
        "SystemBalanced",
        "OllamaQuality",
        "DeltaVsOllama",
        "HAITransportPassRate",
        "HAIQualityPassRate",
        "HAIOverall",
        "HAIScore",
        "HAIDeltaVsOllama",
        "HAIGatesPassed",
        "BenchmarkReport",
        "HAIReport",
    ]
    csv_lines = [",".join(csv_headers)]
    for row in overview_rows:
        csv_lines.append(",".join(csv_cell(row.get(header, "")) for header in csv_headers))
    write_text_copies([primary_csv, alias_csv], "\n".join(csv_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline benchmark matrix for Phoenix presets on Windows")
    parser.add_argument("--plan-file", default=DEFAULT_PLAN_FILE)
    parser.add_argument("--output-dir", default="build/offline_matrix")
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--system-url", default=DEFAULT_SYSTEM_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--llamacpp-url", default=DEFAULT_LLAMACPP_URL)
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument("--system-token", default="local-dev")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--ollama-warmup-timeout", type=float, default=240.0)
    parser.add_argument("--tests-dataset-limit", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--skip-direct-llamacpp", action="store_true")
    parser.add_argument("--strict-exit", action="store_true", help="Return non-zero if any preset regresses or errors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_exe = find_python_executable(args.python_exe)
    plan = load_plan(Path(args.plan_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for preset in plan:
        result = run_single_preset(args, python_exe, preset, output_dir)
        results.append(result)
        print(f"[MATRIX] {result.get('name', 'unnamed')} => {result.get('status', 'unknown')}")

    write_aggregate_report(results, output_dir)
    ok_count = sum(1 for result in results if result.get("status") == "ok")
    regression_count = sum(1 for result in results if result.get("status") == "regression")
    error_count = sum(1 for result in results if result.get("status") == "error")
    skipped_count = sum(1 for result in results if result.get("status") == "skipped")
    print(f"[SUMMARY] ok={ok_count} regression={regression_count} error={error_count} skipped={skipped_count}")
    if error_count:
        return 1
    if args.strict_exit and regression_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
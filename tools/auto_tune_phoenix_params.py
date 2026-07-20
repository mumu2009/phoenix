#!/usr/bin/env python3
"""
auto_tune_phoenix_params.py

Auto-tune hard-coded Phoenix magic numbers by running small memory-tier
benchmarks and writing the best configuration to config/phoenix.json.

Tunable parameters (first pass):
  - context window / similarity / chunking thresholds
  - benchmark scenario turn thresholds (the 5/15 boundaries)
  - llama-server thread/parallel/ctx-size strategy knobs

The script is intentionally conservative with sample counts so you can run a
"large-scale" sweep without spending days. Increase --samples for production
quality.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path
from typing import Any


def get_by_dotpath(obj: Any, dot_path: str) -> Any:
    cur = obj
    for part in dot_path.split("."):
        cur = cur[part]
    return cur


def set_by_dotpath(obj: Any, dot_path: str, value: Any) -> None:
    cur = obj
    parts = dot_path.split(".")
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def parse_vary_values(text: str) -> list[Any]:
    text = text.strip()
    if not text.startswith("["):
        text = "[" + text + "]"
    return json.loads(text)


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    d = root_dir() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_param_space() -> dict[str, Any]:
    """Parameter grid.  The lists are deliberately broad so the random
    sub-grid does not cluster near the old estimate."""
    return {
        "context": {
            "maxTokens": [2048, 4096, 8192, 16384],
            "reservedSystemTokens": [128, 256, 512, 768],
            "importanceThreshold": [0.1, 0.2, 0.3, 0.5, 0.7],
            "similarityThreshold": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            "semanticChunkSize": [128, 256, 512, 1024],
            "attentionSink": {
                "sinkTokens": [64, 128, 256, 512],
                "sinkImportance": [0.03, 0.05, 0.1, 0.2, 0.3],
            },
        },
        "scenarios": {
            # Turn-count boundaries that separate short / long / ultra.
            # Names stay the same, the actual ranges become data-driven.
            "short_dialogue_max_turns": [2, 3, 4],
            "long_dialogue_min_turns": [4, 5, 6, 7],
            "long_dialogue_max_turns": [10, 12, 15, 18],
            "ultra_long_dialogue_min_turns": [16, 18, 20, 24],
            "ultra_long_dialogue_max_turns": [22, 25, 28, 32],
        },
        "llama_server": {
            "threads": [4, 6, 8, 12],
            "parallel": [1, 2, 4],
            "ctx_size": [2048, 4096, 8192, 16384],
        },
    }


def sim_param_space() -> dict[str, Any]:
    """Parameter space for the similarityThreshold re-measurement.
    All other parameters are pinned to the current best-known values."""
    return {
        "context": {
            "maxTokens": [8192],
            "reservedSystemTokens": [256],
            "importanceThreshold": [0.3],
            "similarityThreshold": [0.6],
            "semanticChunkSize": [256],
            "attentionSink": {
                "sinkTokens": [256],
                "sinkImportance": [0.05],
            },
        },
        "scenarios": {
            "short_dialogue_max_turns": [3],
            "long_dialogue_min_turns": [5],
            "long_dialogue_max_turns": [10],
            "ultra_long_dialogue_min_turns": [20],
            "ultra_long_dialogue_max_turns": [22],
        },
        "llama_server": {
            "threads": [12],
            "parallel": [1],
            "ctx_size": [8192],
        },
    }


def other_param_space(similarity: float | None) -> dict[str, Any]:
    """Parameter space for the remaining magic numbers.
    similarityThreshold is included as a broad range so the search can
    verify or override the value discovered in the sim sweep."""
    return {
        "context": {
            "maxTokens": [2048, 4096, 8192, 16384],
            "reservedSystemTokens": [128, 256, 512, 768],
            "importanceThreshold": [0.1, 0.2, 0.3, 0.5, 0.7],
            "similarityThreshold": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            "semanticChunkSize": [128, 256, 512, 1024],
            "attentionSink": {
                "sinkTokens": [64, 128, 256, 512],
                "sinkImportance": [0.03, 0.05, 0.1, 0.2, 0.3],
            },
        },
        "scenarios": {
            "short_dialogue_max_turns": [2, 3, 4],
            "long_dialogue_min_turns": [4, 5, 6, 7],
            "long_dialogue_max_turns": [10, 12, 15, 18],
            "ultra_long_dialogue_min_turns": [16, 18, 20, 24],
            "ultra_long_dialogue_max_turns": [22, 25, 28, 32],
        },
        "llama_server": {
            "threads": [4, 6, 8, 12],
            "parallel": [1, 2, 4],
            "ctx_size": [2048, 4096, 8192, 16384],
        },
    }


def default_fixed_params() -> dict[str, Any]:
    """Non-tuned defaults, written to the final JSON because production code
    reads every value from this file and has no fallbacks."""
    return {
        "context": {
            "enableSemanticSearch": True,
            "enableHierarchical": True,
            "hierarchicalLevels": {
                "global": 1024,
                "workspace": 2048,
                "file": 4096,
                "function": 512,
            },
        },
        "memebarrier": {
            "scanIntervalMs": 10000,
            "maliciousThreshold": 0.7,
            "minThreshold": 0.35,
            "maxThreshold": 0.95,
            "thresholdMomentum": 0.15,
            "minIsolationMargin": 0.05,
            "minConsecutiveHits": 2,
            "maxIsolatePerScan": 5,
            "adaptiveEnabled": True,
            "anomalyEnabled": True,
            "scoreWindowSize": 512,
            "useTextCNN": True,
            "useRecommender": True,
            "useTorchModels": True,
            "disableRateTarget": 0.01,
            "recommenderTopics": 24,
            "recommenderDim": 16,
            "weights": {
                "growth": 0.5,
                "outSkew": 0.25,
                "selfSkew": 0.15,
                "anomaly": 0.10,
            },
            "torch": {
                "maxLen": 48,
                "embDim": 64,
                "hidDim": 64,
                "adamLR": 0.001,
                "epochs": 2,
                "batch": 32,
                "lrDecay": 0.5,
                "patience": 1,
            },
        },
        "summary_model": {
            "vocabSize": 4096,
            "dModel": 64,
            "nHeads": 2,
            "nLayers": 1,
            "dFF": 128,
            "maxLen": 128,
            "maxTokens": 32,
            "lr": 0.001,
            "tokenizerMode": "bpe",
        },
        "model_defaults": {
            "decayFactor": 0.5,
            "maxMemeWords": 100,
            "minOverlapThreshold": 2,
            "memeNgramMin": 3,
            "memeNgramMax": 14,
            "maliciousThreshold": 0.7,
            "learningIterations": 3,
            "iteration": 5,
            "threshold": 3.0,
            "decay": 1.0,
            "decayK": 1.0,
            "maxLen": 16,
            "edgeWeight": 1.0,
            "activationType": "relu",
            "transferType": "linear",
            "activationCustom": "",
            "transferCustom": "",
            "mappingDepth": 1,
            "reflectionTopMemes": 18,
            "reflectionTopWords": 24,
            "reflectionMinScore": 1e-6,
        },
        "dialog": {
            "rlEvery": 20,
            "advEvery": 30,
            "gnnEvery": 40,
            "dialogAsyncLimit": 2,
        },
        "learning": {
            "rlMaxDocs": 64,
            "rlTopKWords": 30,
            "rlImprovementThreshold": 0.01,
            "rlCoverageWeight": 0.7,
            "rlUniquenessWeight": 0.3,
            "advMaxAdversaries": 64,
            "advNoiseLevel": 0.2,
            "advAttackRounds": 3,
            "advDefenseRounds": 3,
            "advBenchLimit": 50,
            "gnnGaMaxDocs": 32,
            "gnnGaPopulation": 6,
            "gnnGaGenerations": 1,
        },
    }


def make_config_candidate(
    context: dict[str, Any],
    scenarios: dict[str, int],
    llama: dict[str, Any],
    fixed: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one grid point into the JSON config structure."""
    cfg = {
        "version": 1,
        "source": "auto_tune_phoenix_params.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "context": {
            **fixed["context"],
            "maxTokens": context["maxTokens"],
            "reservedSystemTokens": context["reservedSystemTokens"],
            "importanceThreshold": context["importanceThreshold"],
            "similarityThreshold": context["similarityThreshold"],
            "semanticChunkSize": context["semanticChunkSize"],
            "attentionSink": {
                "sinkTokens": context["attentionSink"]["sinkTokens"],
                "sinkImportance": context["attentionSink"]["sinkImportance"],
            },
        },
        "scenarios": scenarios,
        "llama_server": llama,
    }
    for key, value in fixed.items():
        if key not in cfg:
            cfg[key] = value
    return cfg


def _atomic_write_json(path: Path, data: Any) -> Path:
    """Write JSON atomically with fsync and a .bak backup.

    A crash or power outage can leave the temp file behind, but it will never
    leave the target file in a half-written state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    backup = path.with_suffix(path.suffix + ".bak")
    if path.exists():
        path.replace(backup)
    tmp.replace(path)
    return path


def write_tuned_json(cfg: dict[str, Any], output_path: Path | None = None) -> Path:
    path = output_path if output_path else config_dir() / "phoenix.json"
    return _atomic_write_json(path, cfg)


def checkpoint_path() -> Path:
    return root_dir() / "build" / "tune_checkpoint.json"


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


_RELEVANT_ARGS = frozenset(
    {
        "samples",
        "rounds",
        "providers",
        "scenarios",
        "latency_budget",
        "run_timeout",
        "timeout",
        "warmup_timeout",
        "warmup_retries",
        "stall_seconds",
        "max_evals",
        "tune_mode",
        "fix_similarity",
        "vary_key",
        "vary_values",
        "best_of",
    }
)


def args_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {k: getattr(args, k) for k in _RELEVANT_ARGS}


def point_signature(point: dict[str, Any]) -> str:
    return json.dumps(point, sort_keys=True, ensure_ascii=False)


def cfg_to_point(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten a candidate config back into the point dictionary used by the grid."""
    ctx = cfg["context"]
    scn = cfg["scenarios"]
    llm = cfg["llama_server"]
    return {
        "context.maxTokens": ctx["maxTokens"],
        "context.reservedSystemTokens": ctx["reservedSystemTokens"],
        "context.importanceThreshold": ctx["importanceThreshold"],
        "context.similarityThreshold": ctx["similarityThreshold"],
        "context.semanticChunkSize": ctx["semanticChunkSize"],
        "context.attentionSink.sinkTokens": ctx["attentionSink"]["sinkTokens"],
        "context.attentionSink.sinkImportance": ctx["attentionSink"]["sinkImportance"],
        "scenarios.short_dialogue_max_turns": scn["short_dialogue_max_turns"],
        "scenarios.long_dialogue_min_turns": scn["long_dialogue_min_turns"],
        "scenarios.long_dialogue_max_turns": scn["long_dialogue_max_turns"],
        "scenarios.ultra_long_dialogue_min_turns": scn["ultra_long_dialogue_min_turns"],
        "scenarios.ultra_long_dialogue_max_turns": scn["ultra_long_dialogue_max_turns"],
        "llama_server.threads": llm["threads"],
        "llama_server.parallel": llm["parallel"],
        "llama_server.ctx_size": llm["ctx_size"],
    }


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_checkpoint(args: argparse.Namespace) -> dict[str, Any] | None:
    cp = checkpoint_path()
    tmp = cp.with_suffix(cp.suffix + ".tmp")
    bak = cp.with_suffix(cp.suffix + ".bak")

    if args.reset_state:
        for p in (cp, tmp, bak):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    for candidate in (cp, tmp, bak):
        if not candidate.exists():
            continue
        state = _load_json_file(candidate)
        if state is None:
            continue
        if state.get("args") != args_signature(args):
            print("[tune] existing checkpoint args mismatch; starting fresh", flush=True)
            return None
        print(f"[tune] restored checkpoint from {candidate.name}", flush=True)
        return state
    return None


def save_checkpoint(state: dict[str, Any]) -> None:
    _atomic_write_json(checkpoint_path(), state)


def build_candidates(args: argparse.Namespace, space: dict[str, Any], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the ordered list of grid points for this run."""
    if args.vary_key or args.vary_values:
        if not args.vary_key or not args.vary_values:
            raise ValueError("--vary-key and --vary-values must be used together")
        baseline_points = list_grid_points(space, args.max_evals)
        baseline_point = baseline_points[0]
        vary_values = parse_vary_values(args.vary_values)
        baseline_value = baseline_point[args.vary_key]
        if baseline_value not in vary_values:
            vary_values = [baseline_value] + vary_values
        points: list[dict[str, Any]] = []
        for value in vary_values:
            point = copy.deepcopy(baseline_point)
            point[args.vary_key] = value
            points.append(point)
        return points
    return list_grid_points(space, args.max_evals)


def _is_valid_threshold_point(point: dict[str, Any]) -> bool:
    """Ensure scenario turn ranges do not overlap or invert."""
    try:
        s = int(point.get("scenarios.short_dialogue_max_turns", 2))
        l_min = int(point.get("scenarios.long_dialogue_min_turns", 5))
        l_max = int(point.get("scenarios.long_dialogue_max_turns", 15))
        u_min = int(point.get("scenarios.ultra_long_dialogue_min_turns", 16))
        u_max = int(point.get("scenarios.ultra_long_dialogue_max_turns", 22))
    except (TypeError, ValueError):
        return False
    return (
        1 <= s < l_min <= l_max < u_min <= u_max <= 100
    )


def list_grid_points(space: dict[str, Any], max_evals: int) -> list[dict[str, Any]]:
    """Build a random sample of the flattened parameter grid.

    Nested dicts are flattened by joining keys with '.'.  The full Cartesian
    product is never materialized, so the grid can be huge without running out
    of memory. Invalid scenario thresholds are filtered out."""
    flat: dict[str, list[Any]] = {}

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            flat[prefix] = obj
        else:
            raise ValueError(f"unexpected value in parameter space at {prefix}: {obj!r}")

    walk(space)
    keys = list(flat.keys())
    value_lists = [flat[k] for k in keys]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max_evals * 1000
    while len(out) < max_evals and attempts < max_attempts:
        combo = [random.choice(v) for v in value_lists]
        attempts += 1
        point = dict(zip(keys, combo))
        if not _is_valid_threshold_point(point):
            continue
        sig = point_signature(point)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(point)
    return out


def unflatten_point(point: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    context: dict[str, Any] = {"attentionSink": {}}
    scenarios: dict[str, int] = {}
    llama: dict[str, Any] = {}

    for key, value in point.items():
        parts = key.split(".")
        if parts[0] == "context":
            if len(parts) == 2:
                context[parts[1]] = value
            elif len(parts) == 3 and parts[1] == "attentionSink":
                context["attentionSink"][parts[2]] = value
        elif parts[0] == "scenarios":
            scenarios[parts[1]] = int(value)
        elif parts[0] == "llama_server":
            llama[parts[1]] = value
    return context, scenarios, llama


def kill_residual_processes() -> None:
    """Best-effort cleanup of leftover phoenix/llama-server processes.
    Also terminate leftover Python helper scripts (llama_proxy, TUI, etc.)
    so they do not hold ports across runs."""
    for name in ("phoenix_main", "llama-server"):
        try:
            subprocess.run(
                ["taskkill", "/F", "/IM", f"{name}.exe"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except Exception:
            pass
    # Terminate leftover Python helper scripts by command line, but never
    # kill the auto-tune script itself.
    for script in ("llama_proxy.py", "memory_tier_benchmark_v1.py", "run_memory_tier_benchmark_tui.py"):
        try:
            subprocess.run(
                [
                    "wmic", "process",
                    "where", f"CommandLine like '%{script}%'",
                    "call", "Terminate",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except Exception:
            pass


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
TUI_SCENARIO_RE = re.compile(r"Scenario:\s+(\d+)/(\d+)\s+\(([^)]*)\)")
TUI_SAMPLE_RE = re.compile(r"Test In Scenario:\s+(\d+)/(\d+)")
TUI_TOTAL_RE = re.compile(r"Total Completed:\s+(\d+)/(\d+)")


def _reader_enqueue(pipe: Any, out_q: "queue.Queue[str]") -> None:
    try:
        for line in iter(pipe.readline, ""):
            out_q.put(line.rstrip("\n"))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def run_one_benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Run the TUI benchmark once with the candidate config and stream progress."""
    write_tuned_json(cfg)
    kill_residual_processes()
    time.sleep(1.0)

    out_prefix = f"auto_tune_{int(time.time() * 1000)}"
    py = root_dir() / ".venv" / "Scripts" / "python.exe"
    tui = root_dir() / "tools" / "run_memory_tier_benchmark_tui.py"

    cmd = [
        str(py),
        str(tui),
        "--sample-per-scenario", str(args.samples),
        "--rounds", str(args.rounds),
        "--providers", args.providers,
        "--scenarios", args.scenarios,
        "--timeout", str(args.timeout),
        "--warmup-timeout", str(args.warmup_timeout),
        "--warmup-retries", str(args.warmup_retries),
        "--context-window", str(cfg["context"]["maxTokens"]),
        "--llama-threads", str(cfg["llama_server"]["threads"]),
        "--llama-parallel", str(cfg["llama_server"]["parallel"]),
        "--llama-ctx-size", str(cfg["llama_server"]["ctx_size"]),
        "--fps", "0.2",
        "--stall-seconds", str(args.stall_seconds),
        "--out-prefix", out_prefix,
    ]
    print("[tune] running:", " ".join(cmd), flush=True)

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    start_ts = time.time()
    timed_out = False
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root_dir()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
    except Exception as e:
        print(f"[tune] failed to start benchmark: {e}", flush=True)
        return None

    out_q: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=_reader_enqueue, args=(proc.stdout, out_q), daemon=True)
    reader.start()

    last_scenario = ""
    last_sample = (-1, -1)
    last_total = (-1, -1)
    try:
        while True:
            if proc.poll() is not None:
                break
            if time.time() - start_ts > args.run_timeout:
                timed_out = True
                print("[tune] benchmark timed out; terminating", flush=True)
                proc.kill()
                proc.wait()
                break
            try:
                line = out_q.get(timeout=0.5)
            except queue.Empty:
                continue
            clean = ANSI_RE.sub("", line).strip()
            if not clean:
                continue
            if "Stall: YES" in clean or "Warning:" in clean or "Traceback" in clean or "[ERROR]" in clean:
                print(f"[tune] {clean}", flush=True)

            m = TUI_SCENARIO_RE.search(clean)
            if m:
                last_scenario = f"{m.group(3)} {m.group(1)}/{m.group(2)}"

            m = TUI_SAMPLE_RE.search(clean)
            if m:
                s_idx, s_total = int(m.group(1)), int(m.group(2))
                if (s_idx, s_total) != last_sample:
                    last_sample = (s_idx, s_total)
                    print(f"[progress] scenario={last_scenario} sample={s_idx}/{s_total}", flush=True)

            m = TUI_TOTAL_RE.search(clean)
            if m:
                t_idx, t_total = int(m.group(1)), int(m.group(2))
                if (t_idx, t_total) != last_total:
                    last_total = (t_idx, t_total)
                    print(f"[progress] total_completed={t_idx}/{t_total}", flush=True)
    except KeyboardInterrupt:
        print("[tune] benchmark interrupted; terminating", flush=True)
        if proc is not None:
            proc.kill()
            proc.wait()
        raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        try:
            reader.join(timeout=2)
        except Exception:
            pass

    if timed_out:
        return None

    # Locate the JSON reports written by the TUI (one per provider/round).
    build_dir = root_dir() / "build"
    best_dir: Path | None = None
    best_mtime = 0.0
    for child in build_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith(out_prefix):
            continue
        if child.stat().st_mtime > best_mtime:
            best_dir = child
            best_mtime = child.stat().st_mtime

    if best_dir is None:
        print("[tune] no output directory found; discarding candidate", flush=True)
        return None

    reports = list(best_dir.glob("memory_tier_benchmark_v1_*_round*.json"))
    if not reports:
        print("[tune] no round JSON found; discarding candidate", flush=True)
        return None

    return summarize_reports(reports)


FORMAT_WORD_RE = re.compile(r"\w+", re.UNICODE)


def is_format_issue(raw_reply: str, expected: str) -> bool:
    """判断失败是否由模型输出的格式/包装词导致，但答案内容正确。
    例如 pred 里包含了 expected 的每个词，只是被额外句子包裹。
    """
    exp_tokens = [t for t in FORMAT_WORD_RE.findall(expected.lower()) if t]
    if not exp_tokens:
        return False
    raw_tokens = set(FORMAT_WORD_RE.findall(raw_reply.lower()))
    return all(tok in raw_tokens for tok in exp_tokens)


def summarize_reports(report_paths: list[Path]) -> dict[str, Any]:
    """Aggregate latency, success rate, average similarity and format issues across all provider/round reports."""
    latencies: list[float] = []
    similarities: list[float] = []
    successes = 0
    total = 0
    format_issues = 0
    raw: list[dict[str, Any]] = []
    for path in report_paths:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        raw.append(report)
        for provider, pdata in report.get("providers", {}).items():
            for scenario, sdata in pdata.get("scenarios", {}).items():
                lat = sdata.get("latencyMs", {}).get("avg", 0.0)
                if isinstance(lat, (int, float)):
                    latencies.append(float(lat))
                sample_count = sdata.get("samples", 0)
                if not isinstance(sample_count, int):
                    sample_count = 0
                # successRateSemanticGe70 is a percentage string or float
                sr = sdata.get("successRateSemanticGe70", 0.0)
                if isinstance(sr, str):
                    try:
                        sr = float(sr)
                    except ValueError:
                        sr = 0.0
                successes += int(round(sample_count * sr / 100.0))
                total += sample_count

                # Per-sample similarity for continuous scoring and format issue detection
                for sample in sdata.get("rawSamples", []):
                    sim = sample.get("similarity")
                    if isinstance(sim, (int, float)):
                        similarities.append(float(sim))
                    if not sample.get("success", False):
                        if is_format_issue(sample.get("rawReply", ""), sample.get("expected", "")):
                            format_issues += 1
    success_rate = successes / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    return {
        "success_rate": success_rate,
        "avg_similarity": avg_similarity,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": max_latency,
        "format_issues": format_issues,
        "raw": raw,
    }


def score_candidate(metrics: dict[str, Any], latency_budget_ms: float) -> float:
    """Higher is better.  Score is driven by the continuous average similarity
    (not just the pass/fail count) and mild latency penalties."""
    avg_similarity = metrics.get("avg_similarity", 0.0)
    avg_latency = metrics.get("avg_latency_ms", 0.0)
    max_latency = metrics.get("max_latency_ms", 0.0)
    if max_latency > latency_budget_ms:
        return -1e9
    score = avg_similarity * 100.0
    score -= (avg_latency / 1000.0) * 0.5
    score -= (max_latency / 1000.0) * 0.3
    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune Phoenix magic numbers")
    parser.add_argument("--samples", type=int, default=5, help="samples per scenario per run")
    parser.add_argument("--rounds", type=int, default=1, help="rounds per run")
    parser.add_argument("--providers", default="phoenix", help="providers to benchmark")
    parser.add_argument("--scenarios", default="short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus", help="scenarios")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--warmup-timeout", type=float, default=120.0)
    parser.add_argument("--warmup-retries", type=int, default=3)
    parser.add_argument("--stall-seconds", type=int, default=180)
    parser.add_argument("--run-timeout", type=float, default=1800.0, help="per-candidate benchmark timeout")
    parser.add_argument("--latency-budget", type=float, default=120000.0, help="max acceptable latency in ms")
    parser.add_argument("--max-evals", type=int, default=12, help="maximum configurations to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="generate config without running benchmarks")
    parser.add_argument("--output-json", default="config/phoenix.json")
    parser.add_argument("--best-of", type=int, default=1, help="keep N best configs")
    parser.add_argument("--vary-key", default=None, help="dot-path of the single parameter to vary (controlled experiment)")
    parser.add_argument("--vary-values", default=None, help="comma-separated list of values for --vary-key")
    parser.add_argument("--tune-mode", default=None, choices=["sim", "other"], help="sim: sweep similarityThreshold; other: sweep remaining magic numbers")
    parser.add_argument("--fix-similarity", type=float, default=None, help="fixed similarityThreshold for tune-mode=other")
    parser.add_argument("--reset-state", action="store_true", help="ignore/delete existing checkpoint and start fresh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tune_mode == "sim":
        space = sim_param_space()
    elif args.tune_mode == "other":
        space = other_param_space(args.fix_similarity)
    else:
        space = default_param_space()
    fixed = default_fixed_params()

    output_json_path = Path(args.output_json)
    if not output_json_path.is_absolute():
        output_json_path = root_dir() / output_json_path

    if args.dry_run:
        points = build_candidates(args, space, fixed)
        cfg = make_config_candidate(*unflatten_point(points[0]), fixed)
        path = write_tuned_json(cfg, output_json_path)
        print(f"[tune] dry-run wrote {path}", flush=True)
        return 0

    state = load_checkpoint(args)
    if state is None:
        points = build_candidates(args, space, fixed)
        state = {
            "version": 1,
            "args": args_signature(args),
            "tune_mode": args.tune_mode,
            "started_at": time.time(),
            "last_updated": time.time(),
            "total": len(points),
            "completed": [],
            "completed_count": 0,
            "points": points,
            "results": [],
            "best": [],
        }
        save_checkpoint(state)
        print(f"[tune] starting fresh: tune_mode={args.tune_mode} candidates={len(points)}", flush=True)
    else:
        points = state["points"]
        print(f"[tune] resuming from checkpoint: {state.get('completed_count', 0)}/{len(points)} candidate(s) completed", flush=True)

    total = len(points)
    completed_sigs = set(state.get("completed", []))
    results = state.get("results", [])
    best = state.get("best", [])
    start_ts = state.get("started_at", time.time())

    interrupted = False
    try:
        for i, point in enumerate(points, start=1):
            sig = point_signature(point)
            if sig in completed_sigs:
                print(f"[tune] candidate {i}/{total} already completed, skipping", flush=True)
                continue
            cfg = make_config_candidate(*unflatten_point(point), fixed)
            elapsed = time.time() - start_ts
            done = i - 1
            if done > 0:
                avg = elapsed / done
                eta = _format_duration(avg * (total - done))
            else:
                eta = "n/a"
            print(
                f"\n[tune] === candidate {i}/{total} ({done * 100.0 / total:.1f}% done, elapsed {_format_duration(elapsed)}, ETA {eta}) ===",
                flush=True,
            )
            print(f"[tune] point={point}", flush=True)
            metrics = run_one_benchmark(cfg, args)
            if metrics is None:
                score = None
                print(f"[tune] candidate {i}/{total} failed or timed out", flush=True)
            else:
                score = score_candidate(metrics, args.latency_budget)
                print(
                    f"[tune] -> success_rate={metrics['success_rate']:.2%} "
                    f"avg_similarity={metrics['avg_similarity']:.3f} "
                    f"avg_latency={metrics['avg_latency_ms']:.1f}ms "
                    f"max_latency={metrics['max_latency_ms']:.1f}ms "
                    f"format_issues={metrics['format_issues']} "
                    f"score={score:.3f}",
                    flush=True,
                )
            results.append(
                {
                    "signature": sig,
                    "point": point,
                    "score": score,
                    "metrics": metrics,
                    "cfg": cfg,
                    "completed_at": time.time(),
                }
            )
            if score is not None:
                best.append({"score": score, "cfg": cfg, "metrics": metrics})
                best.sort(key=lambda x: x["score"], reverse=True)
                best = best[: args.best_of]
            completed_sigs.add(sig)
            state.update(
                {
                    "completed": list(completed_sigs),
                    "completed_count": len(completed_sigs),
                    "results": results,
                    "best": best,
                    "last_updated": time.time(),
                }
            )
            save_checkpoint(state)
            if best:
                top = best[0]
                print(
                    f"[tune] best so far: score={top['score']:.3f} "
                    f"avg_similarity={top['metrics']['avg_similarity']:.3f} "
                    f"avg_latency={top['metrics']['avg_latency_ms']:.1f}ms",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("[tune] interrupted by user; checkpoint saved", flush=True)
        save_checkpoint(state)
        interrupted = True
    finally:
        kill_residual_processes()

    if interrupted:
        return 130

    if not best:
        print("[tune] no successful configuration; keeping defaults", flush=True)
        return 1

    top = best[0]
    out_path = write_tuned_json(top["cfg"], output_json_path)
    print(f"\n[tune] best config written to {out_path}", flush=True)
    print(f"[tune] score={top['score']:.3f} metrics={top['metrics']}", flush=True)

    if not args.reset_state:
        try:
            checkpoint_path().unlink()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
auto_tune_phoenix_params.py

Single-variable / grid / OFAT auto-tuner for Phoenix runtime parameters.

Scope:
  * remote Windows box -> --scope non-vision-speech
    Tunes context, dialog, learning, memebarrier, summary, model_defaults,
    llama_server, frontend, edge-platform (non-NPU), emotion, spark, etc.
    Benchmark is the memory-tier TUI (text/dialog latency & cosine similarity).

  * RDK X5 -> --scope vision-speech
    Tunes vision.*, jpea.*, edge_platform.npu.*, speech.*.
    Benchmark is a vision/speech latency & accuracy test against the local
    phoenix_main HTTP endpoint or a supplied command.

Single-variable experiment:
  python tools/auto_tune_phoenix_params.py --scope non-vision-speech \
      --vary-key context.maxTokens --vary-values "[2048,4096,8192]"

Config-driven OFAT (one factor at a time) sweep:
  python tools/auto_tune_phoenix_params.py --param-space config/universal_optimizer_schema.json \
      --strategy ofat --max-evals 100

Ordered full grid:
  python tools/auto_tune_phoenix_params.py --strategy grid --max-evals 24
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import queue
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_by_dotpath(obj: Any, dot_path: str) -> Any:
    cur = obj
    for part in dot_path.split("."):
        cur = cur[part]
    return cur


def set_by_dotpath(obj: Any, dot_path: str, value: Any) -> None:
    cur = obj
    parts = dot_path.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
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


def base_config() -> dict[str, Any]:
    """Load the current config as the template, so all non-tuned keys stay intact."""
    path = config_dir() / "phoenix.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Parameter spaces
# ---------------------------------------------------------------------------

VISION_SPEECH_PREFIXES = {"vision", "jpea", "speech", "edge_platform"}


def non_vision_speech_param_space() -> dict[str, Any]:
    """All tunable dot-paths that do NOT relate to vision/speech/X5 NPU runtime."""
    return {
        "context": {
            "maxTokens": [2048, 4096, 8192, 16384],
            "reservedSystemTokens": [128, 256, 512, 768],
            "importanceThreshold": [0.1, 0.2, 0.3, 0.5, 0.7],
            "similarityThreshold": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
            "semanticChunkSize": [128, 256, 512, 1024],
            "rnnTopK": [10, 25, 50, 100],
            "attentionSink": {
                "sinkTokens": [64, 128, 256, 512],
                "sinkImportance": [0.03, 0.05, 0.1, 0.2, 0.3],
            },
            "embeddings": {
                "dim": [64, 128, 256],
                "window": [2, 4, 8],
                "minCount": [1, 2, 3, 5],
                "maxVocab": [1000, 3000, 6000],
                "maxFiles": [100, 400, 800],
            },
            "torch": {
                "maxLen": [32, 64, 128],
                "embDim": [64, 128, 256],
                "hidDim": [64, 128, 256],
                "epochs": [2, 4, 8],
                "batch": [8, 16, 32],
                "maxFiles": [120, 240, 480],
            },
            "adaptive": {
                "concatThresh": [3, 5, 8, 12],
                "rnnThresh": [10, 15, 25, 40],
                "shortWindowMaxMessages": [20, 40, 80],
                "shortWindowMaxTokens": [1024, 2048, 4096],
                "targetLatencyMs": [5000.0, 15000.0, 30000.0],
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
        "memebarrier": {
            "scanIntervalMs": [5000, 10000, 30000],
            "maliciousThreshold": [0.5, 0.7, 0.85],
            "minThreshold": [0.25, 0.35, 0.5],
            "maxThreshold": [0.8, 0.95, 0.99],
            "thresholdMomentum": [0.05, 0.15, 0.3],
            "minIsolationMargin": [0.03, 0.05, 0.1],
            "minConsecutiveHits": [1, 2, 3],
            "maxIsolatePerScan": [3, 5, 10],
            "scoreWindowSize": [256, 512, 1024],
            "recommenderTopics": [12, 24, 48],
            "recommenderDim": [8, 16, 32],
            "torch": {
                "maxLen": [24, 48, 96],
                "embDim": [32, 64, 128],
                "hidDim": [32, 64, 128],
                "adamLR": [0.0005, 0.001, 0.002],
                "epochs": [1, 2, 4],
                "batch": [16, 32, 64],
            },
        },
        "summary_model": {
            "vocabSize": [2048, 4096, 8192],
            "dModel": [32, 64, 128],
            "nHeads": [1, 2, 4],
            "nLayers": [1, 2, 3],
            "dFF": [64, 128, 256],
            "maxLen": [64, 128, 256],
            "maxTokens": [16, 32, 64],
            "lr": [0.0005, 0.001, 0.002],
        },
        "model_defaults": {
            "decayFactor": [0.25, 0.5, 0.75],
            "maxMemeWords": [50, 100, 200],
            "minOverlapThreshold": [1, 2, 3],
            "memeNgramMin": [2, 3, 4],
            "memeNgramMax": [10, 14, 20],
            "learningIterations": [1, 3, 5],
            "threshold": [2.0, 3.0, 5.0],
            "decay": [0.5, 1.0, 2.0],
            "edgeWeight": [0.5, 1.0, 2.0],
        },
        "dialog": {
            "rlEvery": [10, 20, 40],
            "advEvery": [20, 30, 60],
            "gnnEvery": [20, 40, 80],
            "dialogAsyncLimit": [1, 2, 4],
        },
        "learning": {
            "rlMaxDocs": [32, 64, 128],
            "rlTopKWords": [15, 30, 60],
            "rlImprovementThreshold": [0.005, 0.01, 0.02],
            "rlCoverageWeight": [0.5, 0.7, 0.9],
            "rlUniquenessWeight": [0.1, 0.3, 0.5],
            "advMaxAdversaries": [32, 64, 128],
            "advNoiseLevel": [0.1, 0.2, 0.3],
            "advAttackRounds": [2, 3, 5],
            "advDefenseRounds": [2, 3, 5],
            "advBenchLimit": [30, 50, 100],
            "gnnGaMaxDocs": [16, 32, 64],
            "gnnGaPopulation": [4, 6, 10],
            "gnnGaGenerations": [1, 2, 3],
        },
        "emotion": {
            "learningRate": [0.01, 0.05, 0.1],
            "maxBias": [1.0, 2.0, 3.0],
            "minBias": [-3.0, -2.0, -1.0],
            "decay": [0.9, 0.95, 0.99],
            "momentum": [0.5, 0.9, 0.99],
            "tokenBoostExponent": [0.8, 1.2, 1.6],
            "minTokenScore": [0.01, 0.05, 0.1],
        },
        "spark": {
            "gnnScheduler": {
                "enabled": [True, False],
                "minAffinity": [0.2, 0.3, 0.5],
                "maxLayers": [2, 3, 4],
                "perturbations": [1, 2, 4],
                "useHistory": [True, False],
            }
        },
        "mechanical_mind": {
            "enabled": [True, False],
            "textThreshold": [0.5, 0.58, 0.7],
            "tokenThreshold": [0.5, 0.6, 0.7],
            "emotionInfluence": [0.1, 0.3, 0.5],
        },
        "partial_cache": {
            "enabled": [True, False],
            "tolerance": [0.0, 0.05, 0.1],
            "maxEntries": [1024, 2048, 4096],
            "ttlMs": [60000, 120000, 300000],
        },
        "knowledge_probe": {
            "probeTimeoutMs": [30000, 60000, 120000],
            "knownSimThreshold": [0.4, 0.5, 0.6],
            "crossSessionLearnEnabled": [True, False],
        },
        "world_model": {
            "agentCount": [10, 50, 100],
            "mapWidth": [50, 100, 200],
            "mapHeight": [50, 100, 200],
            "mapDepth": [2, 3, 5],
            "physicsSubsteps": [2, 4, 8],
        },
        "frontend_server": {
            "httpThreads": [2, 4, 8],
            "embeddingWorkers": [1, 2, 4],
            "rnnWorkers": [1, 2, 4],
            "contextWorkers": [1, 2, 4],
            "episodicWorkers": [1, 2, 3],
        },
        "chat": {
            "queueWaitMs": [90000, 180000, 300000],
            "upstreamTimeoutMs": [120000, 360000, 600000],
            "maxInFlight": [1, 2, 4],
        },
        "api": {
            "upstreamTimeoutMs": [30000, 45000, 60000],
        },
        "auth": {
            "allowRegister": [True, False],
            "allowLocalTokenFallback": [True, False],
            "preferLocalToken": [True, False],
            "requireEmailVerify": [True, False],
        },
        "edge_platform": {
            "enabled": [True, False],
            "preferredComputeBackend": ["auto", "cpu", "npu"],
            "maxComputeInflight": [1, 2, 4],
        },
    }


def vision_speech_param_space() -> dict[str, Any]:
    """Tunable dot-paths for RDK X5 vision/speech runtime."""
    return {
        "vision": {
            "confidenceThreshold": [0.25, 0.35, 0.5],
            "nmsThreshold": [0.3, 0.45, 0.6],
            "minBoxAreaRatio": [0.0005, 0.001, 0.005],
            "maxBoxes": [40, 80, 160],
            "embeddingDim": [8, 18, 32],
            "colorSpace": ["bgr", "rgb"],
            "pixelMean": [[0.485, 0.456, 0.406], [0.0, 0.0, 0.0]],
            "pixelStd": [[0.229, 0.224, 0.225], [1.0, 1.0, 1.0]],
            "torch": {
                "cnnInput": [128, 224, 320],
                "yoloInput": [320, 416, 608],
                "yoloS": [5, 7, 11],
                "yoloB": [2, 3, 5],
                "yoloScore": [0.15, 0.25, 0.4],
                "yoloNms": [0.35, 0.45, 0.6],
                "batch": [1, 4, 8],
                "lr": [0.0005, 0.001, 0.002],
            },
            "yolo": {
                "input": [416, 640, 960],
            },
            "cnn": {
                "input": [128, 224, 320],
                "topK": [3, 5, 10],
            },
        },
        "jpea": {
            "image": {
                "conceptDim": [64, 128, 256],
                "backend": ["auto", "bpu", "cpu"],
            },
            "camera": {
                "width": [640, 1280, 1920],
                "height": [480, 720, 1080],
                "fps": [15, 30, 60],
            },
        },
        "speech": {
            "noiseFloorDb": [-50.0, -45.0, -40.0],
            "minSpeechDurationMs": [100, 250, 500],
            "vadFrameMs": [20, 30, 50],
            "maxTracks": [1, 3, 5],
            "asrConfidenceThreshold": [0.5, 0.6, 0.75],
            "featureWindowMs": [20, 25, 40],
            "featureHopMs": [5, 10, 20],
        },
        "edge_platform": {
            "npu": {
                "enabled": [True, False],
                "vmMb": [2048, 4096, 8192],
                "hotWeightsLimit": [4, 12, 32],
                "hotPromoteHits": [1, 2, 4],
                "unitCount": [1, 10, 19],
                "probeThreshold": [0.2, 0.35, 0.5],
                "spiSpeedHz": [50000000, 120000000, 150000000],
                "spiMode": [0, 1, 2, 3],
            },
        },
    }


def sim_param_space() -> dict[str, Any]:
    """Narrowed context-only sweep, useful as a quick sanity pass."""
    out = copy.deepcopy(non_vision_speech_param_space())
    # Pin everything except context.similarityThreshold
    for key in list(out.keys()):
        if key != "context":
            del out[key]
    # Disable nested context attention-sink/chunk/embeddings/torch/adaptive
    for key in list(out["context"].keys()):
        if key not in ("similarityThreshold", "maxTokens"):
            del out["context"][key]
    out["context"]["maxTokens"] = [8192]
    out["context"]["similarityThreshold"] = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    return out


def _space_to_flat(space: dict[str, Any]) -> dict[str, list[Any]]:
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
    return flat


def _values_for_spec(spec: dict[str, Any]) -> list[Any]:
    """Convert a schema parameter spec into a concrete list of values to try."""
    if "values" in spec:
        return list(spec["values"])

    typ = spec.get("type", "string")

    if typ == "bool":
        return [False, True]

    if typ == "enum" or "choices" in spec:
        return list(spec.get("choices", []))

    if typ in ("int", "float"):
        mn = spec.get("min")
        mx = spec.get("max")
        if mn is None or mx is None:
            raise ValueError(f"{spec.get('path')} needs min/max")

        step = spec.get("step")
        count = spec.get("count")

        if typ == "int":
            mn = int(mn)
            mx = int(mx)
            if step is not None:
                step = int(step)
                return list(range(mn, mx + 1, step))
            if count is not None:
                count = max(2, int(count))
                if count == 2:
                    return [mn, mx]
                step = max(1, (mx - mn) // (count - 1))
                return list(sorted(set(range(mn, mx + 1, step))))[:count]
            # old schema without step/count: default to 5 evenly spaced values
            step = max(1, (mx - mn) // 4)
            return list(sorted(set(range(mn, mx + 1, step))))[:5]

        if typ == "float":
            mn = float(mn)
            mx = float(mx)
            if step is not None:
                step = float(step)
                out = []
                v = mn
                while v <= mx + 1e-9:
                    out.append(round(v, 6))
                    v += step
                return out
            if count is not None:
                count = max(2, int(count))
                if count == 2:
                    return [mn, mx]
                step = (mx - mn) / (count - 1)
                return [round(mn + i * step, 6) for i in range(count)]
            step = (mx - mn) / 4
            return [round(mn + i * step, 6) for i in range(5)]

    raise ValueError(f"unsupported parameter spec type: {typ!r}")


def load_param_space_from_schema(path: Path, filter_prefixes: list[str] | None = None) -> dict[str, list[Any]]:
    """Load a parameter space from a JSON schema file.

    Supported formats:
      * { "parameters": [ { "path": "...", "type": "int", "min": a, "max": b, "step": c }, ... ] }
      * { "dot.path": [v1, v2, ...], ... }
      * [ { "path": "...", "values": [...] }, ... ]
    """
    data = _load_json_file(path)
    if data is None:
        raise ValueError(f"could not load parameter space from {path}")

    params: list[dict[str, Any]] = []
    if isinstance(data, list):
        params = data
    elif isinstance(data, dict):
        if "parameters" in data and isinstance(data["parameters"], list):
            params = data["parameters"]
        else:
            # Treat as flat {dot-path: [values]} space
            return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    else:
        raise ValueError(f"unrecognized parameter space format in {path}")

    space: dict[str, list[Any]] = {}
    for spec in params:
        dot_path = spec.get("path") or spec.get("dot_path")
        if not dot_path:
            continue
        if filter_prefixes:
            if not any(str(dot_path).startswith(p) for p in filter_prefixes):
                continue
        try:
            values = _values_for_spec(spec)
        except ValueError as e:
            print(f"[tune] skipping {dot_path}: {e}", flush=True)
            continue
        if not values:
            continue
        space[dot_path] = values

    if not space:
        raise ValueError(f"no tunable parameters found in {path}")
    return space


def random_points(space: dict[str, Any], max_evals: int) -> list[dict[str, Any]]:
    """Random sampling with de-duplication."""
    flat = _space_to_flat(space)
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
        sig = point_signature(point)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(point)
    return out


def grid_points(space: dict[str, Any], max_evals: int) -> list[dict[str, Any]]:
    """Deterministic full grid (Cartesian product) truncated to max_evals."""
    flat = _space_to_flat(space)
    keys = list(flat.keys())
    value_lists = [flat[k] for k in keys]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        if len(out) >= max_evals:
            break
        out.append(dict(zip(keys, combo)))
    return out


def ofat_points(space: dict[str, Any], max_evals: int) -> list[dict[str, Any]]:
    """One-factor-at-a-time: baseline + each value of each parameter."""
    flat = _space_to_flat(space)
    keys = list(flat.keys())
    if not keys:
        return []
    baseline = {k: flat[k][0] for k in keys}
    out: list[dict[str, Any]] = [baseline]
    for k in keys:
        for v in flat[k]:
            if v == baseline[k]:
                continue
            if len(out) >= max_evals:
                return out
            point = copy.deepcopy(baseline)
            point[k] = v
            out.append(point)
    return out


# ---------------------------------------------------------------------------
# Config write / checkpoint
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> Path:
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


_RELEVANT_ARGS = frozenset({
    "samples", "rounds", "providers", "scenarios", "latency_budget", "run_timeout",
    "timeout", "warmup_timeout", "warmup_retries", "stall_seconds", "max_evals",
    "tune_mode", "fix_similarity", "vary_key", "vary_values", "best_of", "scope",
    "benchmark_command", "x5_host", "x5_port", "x5_warmup",
    "param_space", "param_filter", "strategy",
})


def args_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {k: getattr(args, k) for k in _RELEVANT_ARGS}


def point_signature(point: dict[str, Any]) -> str:
    return json.dumps(point, sort_keys=True, ensure_ascii=False)


def cfg_to_point(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten a candidate config back into the point dictionary used by the grid."""
    point: dict[str, Any] = {}
    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, (list, str, int, float, bool, type(None))):
            point[prefix] = obj
    walk(cfg)
    return point


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


def make_config_candidate(point: dict[str, Any]) -> dict[str, Any]:
    """Overlay the point onto the current config template."""
    cfg = copy.deepcopy(base_config())
    cfg["version"] = 1
    cfg["source"] = "auto_tune_phoenix_params.py"
    cfg["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    for dot_path, value in point.items():
        set_by_dotpath(cfg, dot_path, value)
    return cfg


def build_candidates(args: argparse.Namespace, space: dict[str, Any]) -> list[dict[str, Any]]:
    if args.vary_key or args.vary_values:
        if not args.vary_key or not args.vary_values:
            raise ValueError("--vary-key and --vary-values must be used together")
        baseline_point = make_point(args, ofat_points(space, 1)[0])
        vary_values = parse_vary_values(args.vary_values)
        if args.vary_key not in baseline_point:
            baseline_point[args.vary_key] = vary_values[0]
        baseline_value = baseline_point[args.vary_key]
        if baseline_value not in vary_values:
            vary_values = [baseline_value] + vary_values
        points: list[dict[str, Any]] = []
        for value in vary_values:
            point = copy.deepcopy(baseline_point)
            point[args.vary_key] = value
            points.append(point)
        return points

    if args.strategy == "ofat":
        points = ofat_points(space, args.max_evals)
    elif args.strategy == "grid":
        points = grid_points(space, args.max_evals)
    else:
        points = random_points(space, args.max_evals)
    return [make_point(args, p) for p in points]


def make_point(args: argparse.Namespace, point: dict[str, Any]) -> dict[str, Any]:
    """Apply sanity filters for scenario turn ranges."""
    try:
        s = int(point.get("scenarios.short_dialogue_max_turns", 2))
        l_min = int(point.get("scenarios.long_dialogue_min_turns", 5))
        l_max = int(point.get("scenarios.long_dialogue_max_turns", 15))
        u_min = int(point.get("scenarios.ultra_long_dialogue_min_turns", 16))
        u_max = int(point.get("scenarios.ultra_long_dialogue_max_turns", 22))
        if not (1 <= s < l_min <= l_max < u_min <= u_max <= 100):
            point["scenarios.long_dialogue_min_turns"] = s + 1
            point["scenarios.long_dialogue_max_turns"] = max(l_max, s + 2)
            point["scenarios.ultra_long_dialogue_min_turns"] = max(u_min, l_max + 1)
            point["scenarios.ultra_long_dialogue_max_turns"] = max(u_max, u_min + 1)
    except (TypeError, ValueError):
        pass
    return point


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def kill_residual_processes() -> None:
    for name in ("phoenix_main", "llama-server"):
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/IM", f"{name}.exe"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=10,
                )
            else:
                subprocess.run(
                    ["pkill", "-9", "-f", name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=10,
                )
        except Exception:
            pass
    for script in ("llama_proxy.py", "memory_tier_benchmark_v1.py", "run_memory_tier_benchmark_tui.py"):
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["wmic", "process", "where", f"CommandLine like '%{script}%'", "call", "Terminate"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=10,
                )
            else:
                subprocess.run(
                    ["pkill", "-9", "-f", script],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=10,
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


def run_text_benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    write_tuned_json(cfg)
    kill_residual_processes()
    time.sleep(1.0)

    out_prefix = f"auto_tune_{int(time.time() * 1000)}"
    venv = root_dir() / ".venv"
    if venv.exists():
        if platform.system() == "Windows":
            py = venv / "Scripts" / "python.exe"
        else:
            py = venv / "bin" / "python"
    else:
        py = Path(sys.executable)
    tui = root_dir() / "tools" / "run_memory_tier_benchmark_tui.py"
    if not tui.exists():
        print(f"[tune] benchmark script not found: {tui}", flush=True)
        return None

    cmd = [
        str(py), str(tui),
        "--sample-per-scenario", str(args.samples),
        "--rounds", str(args.rounds),
        "--providers", args.providers,
        "--scenarios", args.scenarios,
        "--timeout", str(args.timeout),
        "--warmup-timeout", str(args.warmup_timeout),
        "--warmup-retries", str(args.warmup_retries),
        "--context-window", str(cfg.get("context", {}).get("maxTokens", 8192)),
        "--llama-threads", str(cfg.get("llama_server", {}).get("threads", 6)),
        "--llama-parallel", str(cfg.get("llama_server", {}).get("parallel", 1)),
        "--llama-ctx-size", str(cfg.get("llama_server", {}).get("ctx_size", 4096)),
        "--fps", "0.2",
        "--stall-seconds", str(args.stall_seconds),
        "--out-prefix", out_prefix,
    ]
    print("[tune] running:", " ".join(cmd), flush=True)

    creationflags = 0
    if platform.system() == "Windows":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    start_ts = time.time()
    timed_out = False
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(root_dir()),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
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
    exp_tokens = [t for t in FORMAT_WORD_RE.findall(expected.lower()) if t]
    if not exp_tokens:
        return False
    raw_tokens = set(FORMAT_WORD_RE.findall(raw_reply.lower()))
    return all(tok in raw_tokens for tok in exp_tokens)


def summarize_reports(report_paths: list[Path]) -> dict[str, Any]:
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
                sr = sdata.get("successRateSemanticGe70", 0.0)
                if isinstance(sr, str):
                    try:
                        sr = float(sr)
                    except ValueError:
                        sr = 0.0
                successes += int(round(sample_count * sr / 100.0))
                total += sample_count

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


def run_x5_vision_speech_benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """X5 vision/speech benchmark placeholder.

    Strategy:
      1. (Optionally) restart the local phoenix_main in the background on X5.
      2. POST a sample image to the local /camera/analyze endpoint and/or
         a sample audio file to /speech/analyze.
      3. Measure end-to-end latency and any reported confidence/box count.

    This can be replaced with an arbitrary shell command via --benchmark-command.
    """
    if args.benchmark_command:
        return run_custom_benchmark(cfg, args)

    host = args.x5_host or "127.0.0.1"
    port = args.x5_port
    url = f"http://{host}:{port}/camera/analyze"
    try:
        import urllib.request
        sample_image = root_dir() / "runtime_store" / "sample.jpg"
        if not sample_image.exists():
            # If no sample image, skip measurement but do not fail.
            print("[tune] no runtime_store/sample.jpg; using synthetic metrics", flush=True)
            return {"avg_latency_ms": 0, "max_latency_ms": 0, "success_rate": 0, "avg_similarity": 0, "format_issues": 0, "raw": {}}
        start = time.time()
        with open(sample_image, "rb") as f:
            req = urllib.request.Request(url, data=f.read(), method="POST")
            req.add_header("Content-Type", "image/jpeg")
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                body = resp.read()
        elapsed_ms = (time.time() - start) * 1000
        result = json.loads(body)
        conf = float(result.get("confidence", 0.5))
        boxes = int(result.get("boxes", 1))
        return {
            "avg_latency_ms": elapsed_ms,
            "max_latency_ms": elapsed_ms,
            "success_rate": min(1.0, boxes / max(1, cfg.get("vision", {}).get("maxBoxes", 80))),
            "avg_similarity": conf,
            "format_issues": 0,
            "raw": result,
        }
    except Exception as e:
        print(f"[tune] X5 benchmark failed: {e}", flush=True)
        return None


def run_custom_benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    cmd = [arg.replace("{{config}}", str(config_dir() / "phoenix.json")) for arg in args.benchmark_command.split()]
    try:
        start = time.time()
        proc = subprocess.run(cmd, cwd=str(root_dir()), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=args.run_timeout)
        elapsed_ms = (time.time() - start) * 1000
        if proc.returncode != 0:
            print(f"[tune] custom benchmark failed: {proc.stderr}", flush=True)
            return None
        # Try to parse last JSON line as metrics
        for line in reversed(proc.stdout.strip().splitlines()):
            if not line.strip():
                continue
            try:
                metrics = json.loads(line)
                metrics.setdefault("avg_latency_ms", elapsed_ms)
                metrics.setdefault("max_latency_ms", elapsed_ms)
                return metrics
            except json.JSONDecodeError:
                continue
        return {"avg_latency_ms": elapsed_ms, "max_latency_ms": elapsed_ms, "success_rate": 0, "avg_similarity": 0, "format_issues": 0, "raw": proc.stdout}
    except Exception as e:
        print(f"[tune] custom benchmark error: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidate(metrics: dict[str, Any], latency_budget_ms: float) -> float:
    avg_similarity = metrics.get("avg_similarity", 0.0)
    avg_latency = metrics.get("avg_latency_ms", 0.0)
    max_latency = metrics.get("max_latency_ms", 0.0)
    if max_latency > latency_budget_ms:
        return -1e9
    score = avg_similarity * 100.0
    score -= (avg_latency / 1000.0) * 0.5
    score -= (max_latency / 1000.0) * 0.3
    return score


# ---------------------------------------------------------------------------
# Args / main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune Phoenix runtime parameters")
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
    parser.add_argument("--vary-key", default=None, help="dot-path of the single parameter to vary")
    parser.add_argument("--vary-values", default=None, help="comma or JSON list of values for --vary-key")
    parser.add_argument("--tune-mode", default=None, choices=["sim", "other"], help="sim: sweep similarityThreshold; other: sweep remaining magic numbers")
    parser.add_argument("--fix-similarity", type=float, default=None, help="fixed similarityThreshold for tune-mode=other")
    parser.add_argument("--reset-state", action="store_true", help="ignore/delete existing checkpoint and start fresh")
    parser.add_argument("--scope", default="non-vision-speech", choices=["non-vision-speech", "vision-speech", "sim"], help="which parameter domain to tune")
    parser.add_argument("--param-space", type=Path, default=None, help="JSON schema file with parameter ranges/choices (overrides built-in scope)")
    parser.add_argument("--param-filter", default=None, help="comma-separated dot-path prefixes to keep when --param-space is used")
    parser.add_argument("--strategy", default="ofat", choices=["ofat", "grid", "random"], help="ofat: one factor at a time; grid: ordered cartesian product; random: random samples")
    parser.add_argument("--benchmark-command", default=None, help="custom shell command to run instead of the default benchmark")
    parser.add_argument("--x5-host", default=None, help="X5 HTTP host for vision/speech benchmark")
    parser.add_argument("--x5-port", type=int, default=5081, help="X5 HTTP port")
    parser.add_argument("--x5-warmup", type=float, default=3.0, help="seconds to wait after restarting X5 phoenix_main")
    return parser.parse_args()


def choose_param_space(args: argparse.Namespace) -> dict[str, Any]:
    if args.param_space:
        path = Path(args.param_space)
        if not path.is_absolute():
            path = root_dir() / path
        prefixes = None
        if args.param_filter:
            prefixes = [p.strip() for p in args.param_filter.split(",") if p.strip()]
        return load_param_space_from_schema(path, prefixes)
    if args.tune_mode == "sim":
        return sim_param_space()
    if args.scope == "vision-speech":
        return vision_speech_param_space()
    return non_vision_speech_param_space()


def main() -> int:
    args = parse_args()
    space = choose_param_space(args)

    output_json_path = Path(args.output_json)
    if not output_json_path.is_absolute():
        output_json_path = root_dir() / output_json_path

    if args.dry_run:
        points = build_candidates(args, space)
        cfg = make_config_candidate(points[0])
        dry_path = output_json_path.with_stem(output_json_path.stem + ".dryrun")
        path = write_tuned_json(cfg, dry_path)
        print(f"[tune] dry-run wrote {path}", flush=True)
        return 0

    state = load_checkpoint(args)
    if state is None:
        points = build_candidates(args, space)
        state = {
            "version": 1,
            "args": args_signature(args),
            "points": points,
            "evaluated": [],
            "best": [],
        }
    else:
        points = state.get("points", [])

    if not points:
        print("[tune] no candidate points to evaluate", flush=True)
        return 1

    evaluated_sigs = {point_signature(e["point"]) for e in state.get("evaluated", [])}
    remaining = [p for p in points if point_signature(p) not in evaluated_sigs]
    print(f"[tune] {len(remaining)}/{len(points)} candidates remaining", flush=True)

    for point in remaining:
        sig = point_signature(point)
        print(f"\n[tune] evaluating {sig}", flush=True)
        cfg = make_config_candidate(point)

        if args.scope == "vision-speech":
            metrics = run_x5_vision_speech_benchmark(cfg, args)
        else:
            metrics = run_text_benchmark(cfg, args)

        if metrics is None:
            print(f"[tune] candidate {sig} failed; recording as timeout", flush=True)
            metrics = {"success_rate": 0, "avg_similarity": 0, "avg_latency_ms": args.latency_budget, "max_latency_ms": args.latency_budget, "format_issues": 0, "raw": []}

        score = score_candidate(metrics, args.latency_budget)
        entry = {"point": point, "metrics": metrics, "score": score, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        state["evaluated"].append(entry)

        best = state.get("best", [])
        best.append(entry)
        best.sort(key=lambda x: x["score"], reverse=True)
        state["best"] = best[: args.best_of]
        save_checkpoint(state)

        best_score = state["best"][0]["score"] if state["best"] else -1e9
        print(f"[tune] score={score:.4f} best={best_score:.4f}", flush=True)

    if state["best"]:
        best_point = state["best"][0]["point"]
        best_cfg = make_config_candidate(best_point)
        write_tuned_json(best_cfg, output_json_path)
        print(f"[tune] best config written to {output_json_path}", flush=True)
        print(f"[tune] best point: {json.dumps(best_point, sort_keys=True, ensure_ascii=False)}", flush=True)
    else:
        print("[tune] no successful evaluations", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

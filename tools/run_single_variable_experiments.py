#!/usr/bin/env python3
"""
run_single_variable_experiments.py

Controlled single-variable experiment driver for the Phoenix universal optimizer.

For each remaining untuned hyperparameter in universal_optimizer_schema.json,
this script runs `universal_optimizer.py --strategy single --path <dot-path>`
against a fixed baseline config.  Results are saved per parameter and a CSV
summary is generated.

Usage:
    # generate experiment plan and dry-run all candidates
    python tools/run_single_variable_experiments.py --dry-run

    # run the actual single-variable experiments for all remaining params
    python tools/run_single_variable_experiments.py --execute --max-evals 6

    # run only a specific module
    python tools/run_single_variable_experiments.py --execute --module learning
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def schema_path() -> Path:
    return root_dir() / "config" / "universal_optimizer_schema.json"


def config_path() -> Path:
    return root_dir() / "config" / "phoenix.json"


def default_state_dir() -> Path:
    return root_dir() / "experiments" / "single_variable"


def resolve_state_dir(value: str | None) -> Path:
    path = Path(value) if value else default_state_dir()
    if not path.is_absolute():
        path = root_dir() / path
    return path.resolve()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_state(path: Path, state_dir: Path) -> str:
    return str(path.resolve().relative_to(state_dir.resolve()))


def load_state(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text(encoding="utf-8"))


def load_config() -> dict[str, Any]:
    return json.loads(config_path().read_text(encoding="utf-8"))


def get_parameters(schema: dict[str, Any], module: str | None) -> list[dict[str, Any]]:
    """Return untuned parameters, optionally filtered by module."""
    out: list[dict[str, Any]] = []
    for p in schema.get("parameters", []):
        if p.get("tuned", False):
            continue
        if module is not None and p.get("module") != module:
            continue
        out.append(p)
    out.sort(key=lambda x: x["path"])
    return out


def experiment_signature(params: list[dict[str, Any]], args: argparse.Namespace) -> str:
    payload = {
        "schema": file_hash(schema_path()),
        "parameters": params,
        "settings": {
            "max_evals": args.max_evals,
            "latency_budget": args.latency_budget,
            "run_timeout": args.run_timeout,
            "timeout": args.timeout,
            "warmup_timeout": args.warmup_timeout,
            "warmup_retries": args.warmup_retries,
            "stall_seconds": args.stall_seconds,
            "samples": args.samples,
            "rounds": args.rounds,
            "providers": args.providers,
            "scenarios": args.scenarios,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_optimizer_args(args: argparse.Namespace, path: str) -> list[str]:
    cmd = [
        sys.executable,
        str(root_dir() / "tools" / "universal_optimizer.py"),
        "--strategy", "single",
        "--path", path,
        "--max-evals", str(args.max_evals),
        "--latency-budget", str(args.latency_budget),
        "--run-timeout", str(args.run_timeout),
        "--timeout", str(args.timeout),
        "--warmup-timeout", str(args.warmup_timeout),
        "--warmup-retries", str(args.warmup_retries),
        "--stall-seconds", str(args.stall_seconds),
        "--samples", str(args.samples),
        "--rounds", str(args.rounds),
        "--providers", args.providers,
        "--scenarios", args.scenarios,
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def create_state(state_dir: Path, params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    state_dir.mkdir(parents=True, exist_ok=True)
    baseline = state_dir / "baseline_phoenix.json"
    shutil.copy2(config_path(), baseline)
    plan = {
        "version": 1,
        "created_at": time.time(),
        "schema_sha256": file_hash(schema_path()),
        "baseline_sha256": file_hash(baseline),
        "parameters": params,
        "settings": {
            "max_evals": args.max_evals,
            "latency_budget": args.latency_budget,
            "run_timeout": args.run_timeout,
            "timeout": args.timeout,
            "warmup_timeout": args.warmup_timeout,
            "warmup_retries": args.warmup_retries,
            "stall_seconds": args.stall_seconds,
            "samples": args.samples,
            "rounds": args.rounds,
            "providers": args.providers,
            "scenarios": args.scenarios,
        },
    }
    atomic_write_json(state_dir / "experiment_plan.json", plan)
    return {
        "version": 1,
        "signature": experiment_signature(params, args),
        "created_at": time.time(),
        "last_updated": time.time(),
        "baseline": "baseline_phoenix.json",
        "plan": "experiment_plan.json",
        "entries": {p["path"]: {"status": "pending"} for p in params},
        "runs": [{"host": platform.node(), "platform": platform.platform(), "started_at": time.time()}],
    }


def load_or_create_state(state_dir: Path, params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    state_path = state_dir / "experiment_state.json"
    state = load_state(state_path)
    expected = experiment_signature(params, args)
    if state is None:
        state = create_state(state_dir, params, args)
        atomic_write_json(state_path, state)
        print(f"[exp] created portable state: {state_path}")
        return state
    if args.reset_state:
        state = create_state(state_dir, params, args)
        atomic_write_json(state_path, state)
        print(f"[exp] reset portable state: {state_path}")
        return state
    if state.get("signature") != expected:
        raise ValueError("existing state differs from the current experiment plan; use a new --state-dir or --reset-state")
    state.setdefault("runs", []).append({"host": platform.node(), "platform": platform.platform(), "resumed_at": time.time()})
    state["last_updated"] = time.time()
    atomic_write_json(state_path, state)
    return state


def run_experiment(path: str, args: argparse.Namespace, state_dir: Path) -> dict[str, Any]:
    cmd = build_optimizer_args(args, path)
    out = state_dir / "logs" / f"{path.replace('.', '_')}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[exp] running single-variable experiment: {path}", flush=True)
    proc = subprocess.run(cmd, cwd=str(root_dir()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out.write_text(proc.stdout, encoding="utf-8")
    print(f"[exp] finished: {path} (exit={proc.returncode})", flush=True)

    best_score = -float("inf")
    best_value = None
    for line in proc.stdout.splitlines():
        if "score=" not in line:
            continue
        try:
            score = float(line.split("score=")[-1].split()[0])
            if score > best_score:
                best_score = score
                best_value = line.split("=")[1].split()[0] if "=" in line else None
        except (IndexError, ValueError):
            continue

    return {
        "status": "dry_run" if args.dry_run else ("completed" if proc.returncode == 0 else "failed"),
        "exit_code": proc.returncode,
        "best_score": best_score if best_score > -1e8 else None,
        "best_value": best_value,
        "log": relative_to_state(out, state_dir),
        "completed_at": time.time(),
        "host": platform.node(),
    }


def write_plan_csv(params: list[dict[str, Any]], state_dir: Path) -> Path:
    path = state_dir / "experiment_plan.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "module", "type", "min", "max", "default", "choices"])
        for p in params:
            writer.writerow([p["path"], p.get("module", ""), p.get("type", ""), p.get("min", ""), p.get("max", ""), p.get("default", ""), ";".join(str(c) for c in p.get("choices", []))])
    return path


def write_results_csv(state: dict[str, Any], state_dir: Path) -> Path:
    path = state_dir / "experiment_results.csv"
    fieldnames = ["path", "status", "exit_code", "best_score", "best_value", "log", "host", "started_at", "completed_at"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for param, entry in sorted(state["entries"].items()):
            writer.writerow({"path": param, **entry})
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled single-variable experiments for Phoenix")
    parser.add_argument("--execute", action="store_true", help="run actual experiments (default: plan only)")
    parser.add_argument("--module", default=None, help="limit to one module")
    parser.add_argument("--reverse", action="store_true", help="run selected parameters in reverse dot-path order")
    parser.add_argument("--skip", action="append", default=[], help="comma-separated parameter dot-paths to exclude; may be repeated")
    parser.add_argument("--state-dir", default=None, help="portable experiment bundle directory")
    parser.add_argument("--reset-state", action="store_true", help="replace the state in --state-dir")
    parser.add_argument("--max-evals", type=int, default=12, help="max evaluations per parameter")
    parser.add_argument("--latency-budget", type=float, default=120000.0, help="latency budget ms")
    parser.add_argument("--run-timeout", type=float, default=1800.0, help="per-experiment timeout")
    parser.add_argument("--timeout", type=float, default=90.0, help="single benchmark timeout")
    parser.add_argument("--warmup-timeout", type=float, default=120.0, help="warmup timeout")
    parser.add_argument("--warmup-retries", type=int, default=3, help="warmup retries")
    parser.add_argument("--stall-seconds", type=int, default=180, help="stall detection seconds")
    parser.add_argument("--samples", type=int, default=5, help="samples per scenario")
    parser.add_argument("--rounds", type=int, default=1, help="rounds per run")
    parser.add_argument("--providers", default="phoenix", help="providers")
    parser.add_argument("--scenarios", default="short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus", help="scenarios")
    parser.add_argument("--dry-run", action="store_true", help="pass --dry-run to optimizer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_dir = resolve_state_dir(args.state_dir)
    all_params = get_parameters(load_schema(), args.module)
    skip_paths = {path.strip() for group in args.skip for path in group.split(",") if path.strip()}
    available_paths = {p["path"] for p in all_params}
    unknown_paths = skip_paths - available_paths
    if unknown_paths:
        raise ValueError(f"--skip contains paths outside the selected experiment: {', '.join(sorted(unknown_paths))}")
    params = [p for p in all_params if p["path"] not in skip_paths]
    if args.reverse:
        params.reverse()
    state = load_or_create_state(state_dir, params, args)
    write_plan_csv(params, state_dir)
    print(f"[exp] portable state: {state_dir}")
    print(f"[exp] {len(params)} parameters selected for single-variable experiments")

    if not args.execute and not args.dry_run:
        print("[exp] plan only; use --execute or --dry-run to run experiments")
        return 0

    baseline = state_dir / state["baseline"]
    if not baseline.exists():
        raise FileNotFoundError(f"missing baseline in state bundle: {baseline}")
    state_path = state_dir / "experiment_state.json"
    interrupted = False
    try:
        for p in params:
            path = p["path"]
            entry = state["entries"].get(path, {})
            if entry.get("status") == "completed":
                print(f"[exp] already completed, skipping: {path}")
                continue
            shutil.copy2(baseline, config_path())
            state["entries"][path] = {"status": "running", "started_at": time.time(), "host": platform.node()}
            state["last_updated"] = time.time()
            atomic_write_json(state_path, state)
            result = run_experiment(path, args, state_dir)
            shutil.copy2(baseline, config_path())
            state["entries"][path] = result
            state["last_updated"] = time.time()
            atomic_write_json(state_path, state)
            write_results_csv(state, state_dir)
    except KeyboardInterrupt:
        interrupted = True
        print("[exp] interrupted; portable state saved", flush=True)
    finally:
        shutil.copy2(baseline, config_path())
        state["last_updated"] = time.time()
        atomic_write_json(state_path, state)
        results_csv = write_results_csv(state, state_dir)
        print(f"[exp] results written: {results_csv}")

    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())

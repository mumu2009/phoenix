#!/usr/bin/env python3
"""
universal_optimizer.py

A runtime-capable "universal optimizer" for Phoenix AI modules.

This tool treats the whole config/phoenix.json as a tunable control surface.
Each tunable parameter belongs to a module (context, memebarrier, model_defaults,
dialog, learning, ...). The user can enable or disable per-module tuning.

Supported operating modes:
  - single-variable: sweep one parameter while all others stay pinned.
  - grid: sweep all tunable parameters from the schema grid.
  - random: random search within module bounds.
  - bayesian: lightweight Gaussian-process Bayesian optimization.
  - pbt: Population-Based Training (exploit + explore) across modules.

The optimizer reuses the existing memory-tier benchmark runner for feedback and
writes the best discovered config to config/phoenix.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        SKLEARN_AVAILABLE = True
    except Exception:
        SKLEARN_AVAILABLE = False


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    d = root_dir() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_json(path: Path, data: Any) -> Path:
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


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def schema_path() -> Path:
    return config_dir() / "universal_optimizer_schema.json"


def default_optimizer_config() -> dict[str, Any]:
    """Default settings for the universal optimizer section."""
    return {
        "enabled": True,
        "log_dir": str(root_dir() / "build" / "universal_optimizer"),
        "modules": {
            "context": True,
            "scenarios": False,
            "llama_server": False,
            "memebarrier": True,
            "summary_model": True,
            "model_defaults": True,
            "dialog": True,
            "learning": True,
        },
        "strategy": "single",
        "max_evals": 12,
        "latency_budget_ms": 120000.0,
        "run_timeout": 1800.0,
        "samples": 5,
        "rounds": 1,
        "providers": "phoenix",
        "scenarios": "short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus",
    }


def load_schema() -> dict[str, Any]:
    path = schema_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing schema: {path}")
    schema = load_json(path)
    if not schema:
        raise ValueError("Cannot parse schema")
    return schema


def load_config() -> dict[str, Any]:
    path = config_dir() / "phoenix.json"
    cfg = load_json(path)
    if cfg is None:
        raise FileNotFoundError(f"Missing config: {path}")
    return cfg


def save_config(cfg: dict[str, Any]) -> Path:
    path = config_dir() / "phoenix.json"
    return _atomic_write_json(path, cfg)


def ensure_optimizer_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if "universal_optimizer" not in cfg:
        cfg["universal_optimizer"] = default_optimizer_config()
    else:
        base = default_optimizer_config()
        for k, v in base.items():
            if k not in cfg["universal_optimizer"]:
                cfg["universal_optimizer"][k] = v
        if "modules" in cfg["universal_optimizer"]:
            for mod, default in base["modules"].items():
                if mod not in cfg["universal_optimizer"]["modules"]:
                    cfg["universal_optimizer"]["modules"][mod] = default
    return cfg


def filtered_parameters(schema: dict[str, Any], enabled_modules: dict[str, bool]) -> list[dict[str, Any]]:
    """Return parameters whose module is enabled and that are not yet tuned."""
    out = []
    for p in schema.get("parameters", []):
        mod = p.get("module", "")
        if enabled_modules.get(mod, False) and not p.get("tuned", False):
            out.append(p)
    return out


def normalize_value(p: dict[str, Any], value: Any) -> Any:
    """Clip values to bounds and convert types."""
    typ = p.get("type", "string")
    if typ == "int":
        v = int(value)
        mn = p.get("min")
        mx = p.get("max")
        if mn is not None:
            v = max(mn, v)
        if mx is not None:
            v = min(mx, v)
        return v
    if typ == "float":
        v = float(value)
        mn = p.get("min")
        mx = p.get("max")
        if mn is not None:
            v = max(mn, v)
        if mx is not None:
            v = min(mx, v)
        return v
    if typ == "bool":
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    if typ == "enum":
        choices = p.get("choices", [])
        if value in choices:
            return value
        if str(value) in choices:
            return str(value)
        return p.get("default")
    return value


def random_value(p: dict[str, Any]) -> Any:
    typ = p.get("type", "string")
    if typ == "int":
        mn = p.get("min", 0)
        mx = p.get("max", mn + 1)
        return random.randint(mn, mx)
    if typ == "float":
        mn = p.get("min", 0.0)
        mx = p.get("max", 1.0)
        return mn + random.random() * (mx - mn)
    if typ == "bool":
        return random.choice([True, False])
    if typ == "enum":
        choices = p.get("choices", [])
        return random.choice(choices) if choices else p.get("default")
    return p.get("default")


def candidate_from_params(base_cfg: dict[str, Any], overrides: list[tuple[dict[str, Any], Any]]) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    for p, value in overrides:
        value = normalize_value(p, value)
        set_by_dotpath(cfg, p["path"], value)
    return cfg


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_benchmark(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    """Run the memory-tier benchmark using the existing auto_tune helper."""
    from auto_tune_phoenix_params import write_tuned_json, run_one_benchmark
    out_path = config_dir() / "phoenix.json"
    write_tuned_json(cfg, out_path)
    return run_one_benchmark(cfg, args)


def score_candidate(metrics: dict[str, Any] | None, latency_budget_ms: float) -> float:
    """Higher is better."""
    if metrics is None:
        return -1e9
    avg_similarity = metrics.get("avg_similarity", 0.0)
    avg_latency = metrics.get("avg_latency_ms", 0.0)
    max_latency = metrics.get("max_latency_ms", 0.0)
    if max_latency > latency_budget_ms:
        return -1e9
    score = avg_similarity * 100.0
    score -= (avg_latency / 1000.0) * 0.5
    score -= (max_latency / 1000.0) * 0.3
    return score


def single_variable_sweep(base_cfg: dict[str, Any], params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Run single-variable sweeps for every remaining parameter, one at a time."""
    best = copy.deepcopy(base_cfg)
    best_score = -float("inf")
    for p in params:
        path = p["path"]
        print(f"\n[uo] === single-variable sweep: {path} ===", flush=True)
        values = build_sweep_values(p)
        local_best = copy.deepcopy(base_cfg)
        local_best_score = -float("inf")
        for v in values:
            cfg = copy.deepcopy(base_cfg)
            set_by_dotpath(cfg, path, normalize_value(p, v))
            metrics = run_benchmark(cfg, args)
            score = score_candidate(metrics, args.latency_budget)
            print(f"[uo] {path}={v} score={score:.3f}", flush=True)
            if score > local_best_score:
                local_best_score = score
                local_best = cfg
        if local_best_score > best_score:
            best_score = local_best_score
            best = local_best
    return best


def build_sweep_values(p: dict[str, Any]) -> list[Any]:
    """Generate a small grid for a parameter."""
    typ = p.get("type", "string")
    if typ == "enum":
        return p.get("choices", [p.get("default")])
    if typ == "bool":
        return [True, False]
    if typ == "int":
        mn = p.get("min", 0)
        mx = p.get("max", mn + 1)
        width = mx - mn
        if width <= 6:
            return list(range(mn, mx + 1))
        if width <= 20:
            step = max(1, width // 5)
            vals = list(range(mn, mx + 1, step))
            if vals[-1] != mx:
                vals.append(mx)
            return vals
        step = max(1, width // 10)
        vals = list(range(mn, mx + 1, step))
        if vals[-1] != mx:
            vals.append(mx)
        return vals
    if typ == "float":
        mn = p.get("min", 0.0)
        mx = p.get("max", 1.0)
        width = mx - mn
        if width <= 0.1:
            return [mn, (mn + mx) / 2, mx]
        return [mn + i * (mx - mn) / 9 for i in range(10)]
    return [p.get("default")]


def random_sweep(base_cfg: dict[str, Any], params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    best = copy.deepcopy(base_cfg)
    best_score = -float("inf")
    for i in range(1, args.max_evals + 1):
        overrides = [(p, random_value(p)) for p in params]
        cfg = candidate_from_params(base_cfg, overrides)
        print(f"\n[uo] === random candidate {i}/{args.max_evals} ===", flush=True)
        metrics = run_benchmark(cfg, args)
        score = score_candidate(metrics, args.latency_budget)
        print(f"[uo] score={score:.3f}", flush=True)
        if score > best_score:
            best_score = score
            best = cfg
    return best


def grid_search(base_cfg: dict[str, Any], params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Exhaustive grid for small spaces, otherwise random subset."""
    values_per_param = [build_sweep_values(p) for p in params]
    total = 1
    for vs in values_per_param:
        total *= len(vs)
    if total > args.max_evals * 2:
        print(f"[uo] grid too large ({total} points); using random subset", flush=True)
        return random_sweep(base_cfg, params, args)
    import itertools
    best = copy.deepcopy(base_cfg)
    best_score = -float("inf")
    for i, combo in enumerate(itertools.product(*values_per_param)):
        overrides = list(zip(params, combo))
        cfg = candidate_from_params(base_cfg, overrides)
        print(f"\n[uo] === grid point {i + 1}/{total} ===", flush=True)
        metrics = run_benchmark(cfg, args)
        score = score_candidate(metrics, args.latency_budget)
        print(f"[uo] score={score:.3f}", flush=True)
        if score > best_score:
            best_score = score
            best = cfg
    return best


def bayesian_optimization(base_cfg: dict[str, Any], params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Lightweight GP-based Bayesian optimization. Requires scikit-learn."""
    if not SKLEARN_AVAILABLE:
        print("[uo] sklearn not available, falling back to random search", flush=True)
        return random_sweep(base_cfg, params, args)

    float_params = [p for p in params if p.get("type") in ("int", "float")]
    if not float_params:
        print("[uo] no numeric params for BO, falling back to random", flush=True)
        return random_sweep(base_cfg, params, args)

    bounds = [(p.get("min", 0.0), p.get("max", 1.0)) for p in float_params]
    for i, p in enumerate(float_params):
        if p.get("type") == "int":
            bounds[i] = (float(bounds[i][0]), float(bounds[i][1]))

    def _scale_to_unit(x: list[float]) -> list[float]:
        return [(x[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0] + 1e-9) for i in range(len(x))]

    def _scale_from_unit(x: list[float]) -> list[float]:
        return [x[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0] for i in range(len(x))]

    X: list[list[float]] = []
    y: list[float] = []
    best = copy.deepcopy(base_cfg)
    best_score = -float("inf")

    for i in range(1, args.max_evals + 1):
        if i <= 3 or len(X) < 3:
            raw = [random.uniform(b[0], b[1]) for b in bounds]
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
            gp.fit([_scale_to_unit(x) for x in X], y)
            candidates = [[random.uniform(b[0], b[1]) for b in bounds] for _ in range(100)]
            means, sigmas = gp.predict([_scale_to_unit(c) for c in candidates], return_std=True)
            best_idx = int((means + 1.96 * sigmas).argmax())
            raw = candidates[best_idx]

        overrides = [(p, normalize_value(p, raw[j])) for j, p in enumerate(float_params)]
        cfg = candidate_from_params(base_cfg, overrides)
        print(f"\n[uo] === BO candidate {i}/{args.max_evals} ===", flush=True)
        metrics = run_benchmark(cfg, args)
        score = score_candidate(metrics, args.latency_budget)
        print(f"[uo] score={score:.3f}", flush=True)
        X.append(raw)
        y.append(score)
        if score > best_score:
            best_score = score
            best = cfg
    return best


def population_based_training(base_cfg: dict[str, Any], params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Population-Based Training with exploit/explore."""
    pop_size = min(4, args.max_evals)
    generations = max(1, args.max_evals // pop_size)
    float_params = [p for p in params if p.get("type") in ("int", "float")]
    if not float_params:
        return random_sweep(base_cfg, params, args)

    population = []
    for m in range(pop_size):
        overrides = [(p, random_value(p)) for p in float_params]
        cfg = candidate_from_params(base_cfg, overrides)
        print(f"\n[uo] === PBT init member {m + 1}/{pop_size} ===", flush=True)
        metrics = run_benchmark(cfg, args)
        score = score_candidate(metrics, args.latency_budget)
        population.append({"overrides": overrides, "score": score, "cfg": cfg})

    for g in range(1, generations):
        population.sort(key=lambda x: x["score"], reverse=True)
        print(f"\n[uo] === PBT generation {g + 1}/{generations} best={population[0]['score']:.3f} ===", flush=True)
        new_pop = []
        for rank, member in enumerate(population):
            if rank < pop_size // 2:
                new_pop.append(member)
                continue
            donor = population[0]
            overrides = []
            for (p, v) in donor["overrides"]:
                if random.random() < 0.2:
                    v = random_value(p)
                else:
                    if p.get("type") == "int":
                        v = normalize_value(p, int(v) + random.choice([-1, 1]))
                    else:
                        v = normalize_value(p, v + random.gauss(0.0, (p.get("max", 1.0) - p.get("min", 0.0)) * 0.05))
                overrides.append((p, v))
            cfg = candidate_from_params(base_cfg, overrides)
            metrics = run_benchmark(cfg, args)
            score = score_candidate(metrics, args.latency_budget)
            new_pop.append({"overrides": overrides, "score": score, "cfg": cfg})
        population = new_pop

    population.sort(key=lambda x: x["score"], reverse=True)
    return population[0]["cfg"]


def run_strategy(cfg: dict[str, Any], params: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    strategy = args.strategy
    if strategy == "single":
        return single_variable_sweep(cfg, params, args)
    if strategy == "grid":
        return grid_search(cfg, params, args)
    if strategy == "random":
        return random_sweep(cfg, params, args)
    if strategy == "bayesian":
        return bayesian_optimization(cfg, params, args)
    if strategy == "pbt":
        return population_based_training(cfg, params, args)
    raise ValueError(f"Unknown strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal runtime optimizer for Phoenix")
    parser.add_argument("--strategy", default="single", choices=["single", "grid", "random", "bayesian", "pbt"], help="search strategy")
    parser.add_argument("--module", default=None, help="optimize only one module (e.g., model_defaults)")
    parser.add_argument("--path", default=None, help="optimize only one parameter dot-path")
    parser.add_argument("--max-evals", type=int, default=12, help="maximum evaluations")
    parser.add_argument("--latency-budget", type=float, default=120000.0, help="max acceptable latency in ms")
    parser.add_argument("--timeout", type=float, default=90.0, help="single benchmark timeout")
    parser.add_argument("--run-timeout", type=float, default=1800.0, help="per-candidate benchmark timeout")
    parser.add_argument("--warmup-timeout", type=float, default=120.0, help="warmup timeout")
    parser.add_argument("--warmup-retries", type=int, default=3, help="warmup retries")
    parser.add_argument("--stall-seconds", type=int, default=180, help="stall detection seconds")
    parser.add_argument("--samples", type=int, default=5, help="samples per scenario")
    parser.add_argument("--rounds", type=int, default=1, help="rounds per run")
    parser.add_argument("--providers", default="phoenix", help="providers")
    parser.add_argument("--scenarios", default="short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus", help="scenarios")
    parser.add_argument("--dry-run", action="store_true", help="print first candidate without running benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    cfg = ensure_optimizer_config(cfg)
    schema = load_schema()
    modules = cfg["universal_optimizer"].get("modules", {})
    enabled = {m: modules.get(m, False) for m in schema["modules"]}

    if args.module:
        for m in enabled:
            enabled[m] = (m == args.module)

    params = filtered_parameters(schema, enabled)
    if args.path:
        params = [p for p in params if p["path"] == args.path]
    if not params:
        print("[uo] no tunable parameters for selected modules", flush=True)
        return 0

    print(f"[uo] strategy={args.strategy} enabled_modules={enabled}", flush=True)
    print(f"[uo] optimizing {len(params)} parameter(s)", flush=True)

    if args.dry_run:
        if args.strategy == "single":
            print(f"[uo] dry-run first single-variable sweep: {params[0]['path']}")
        else:
            overrides = [(p, random_value(p)) for p in params]
            cfg = candidate_from_params(cfg, overrides)
            print(f"[uo] dry-run candidate: {json.dumps(cfg, indent=2, ensure_ascii=False)}")
        return 0

    best = run_strategy(cfg, params, args)
    out_path = save_config(best)
    print(f"\n[uo] best config written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

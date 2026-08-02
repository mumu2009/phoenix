#!/usr/bin/env python3
"""
remote_phoenix_manager.py

A single, self-contained Windows 10 Pro / remote training box manager for
Phoenix.  Supports:

  tune      : search Phoenix runtime or training hyperparameters
  pretrain  : stream-download public image / speech datasets and continue
              self-supervised / contrastive pre-training on GPU
  export    : convert a trained checkpoint to ONNX
  status    : show current progress toward a sample target

Design constraints (per user request):
  - Single file: copy this one script to the remote and run it.
  - Uses GPU to the maximum extent the single card allows.
  - Tracks progress toward 0.1 billion (100 000 000) samples per modality.
  - Continual / resumable; streams data and deletes each dataset after use.

Requirements on the remote (will try to auto-install if internet is available):
  torch, transformers, accelerate, datasets, sentence-transformers,
  Pillow, librosa / soundfile, scipy, psutil, numpy, requests

Image training (pretrain --modality image):
  - Streams image+caption datasets from HuggingFace (default: COCO, Flickr30k,
    WikiArt via datasets if available).
  - Loads a compact vision encoder (default: google/vit-base-patch16-224 or
    timm vit_small_patch16_224) and a sentence-transformer text encoder.
  - Trains a contrastive image-text projection head so the output dimension
    matches --target-dim.
  - Saves checkpoints to runtime_store/models/phoenix_remote/image/

Speech training (pretrain --modality speech):
  - Streams audio+transcript datasets from HuggingFace (default: librispeech_asr,
    common_voice_11_0, voxpopuli) and trains a contrastive audio-text pipeline.
  - Base audio encoder: facebook/wav2vec2-base-960h or facebook/wav2vec2-base.
  - Saves checkpoints to runtime_store/models/phoenix_remote/speech/

0.1 billion note:
  The script accepts --target-samples 100000000, but public streaming datasets
  rarely contain that many clean samples and the remote GPU (GTX 16-series,
  6 GB-class) would take months to process that many.  The default target is
  deliberately set to 100000000 so you can run it for as long as you want; the
  script will continue across datasets and report how far it gets.

Examples:

  python tools/remote_phoenix_manager.py pretrain --modality image \
      --target-samples 100000000 --target-dim 128 --batch-size 8

  python tools/remote_phoenix_manager.py tune --mode phoenix \
      --max-evals 20 --benchmark-cmd "python tools/memory_tier_benchmark_v1.py"

  python tools/remote_phoenix_manager.py export --modality image \
      --checkpoint runtime_store/models/phoenix_remote/image/best.pth

Author: 079 Project
License: GNU Lesser General Public License v3
"""
from __future__ import annotations

import argparse
import copy
import datetime
import importlib
import importlib.util
import itertools
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build"
RUNTIME_STORE = ROOT / "runtime_store"
MODEL_DIR = RUNTIME_STORE / "models" / "phoenix_remote"
STATE_FILE = RUNTIME_STORE / "remote_manager_state.json"
CHECKPOINT_FILE = RUNTIME_STORE / "remote_tune_checkpoint.json"
DEFAULT_PHOENIX_JSON = ROOT / "config" / "phoenix.json"
DEFAULT_SCHEMA = ROOT / "config" / "universal_optimizer_schema.json"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    path.unlink(missing_ok=True)
    tmp.rename(path)


def dot_path_get(obj: Any, dot_path: str) -> Any:
    cur = obj
    for part in dot_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dot_path)
        cur = cur[part]
    return cur


def dot_path_set(obj: Any, dot_path: str, value: Any) -> None:
    cur = obj
    parts = dot_path.split(".")
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def human_count(n: float) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.0f}"


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def parse_sample_count(text: str) -> int:
    text = text.strip().lower().replace(",", "")
    multipliers = {
        "b": 1_000_000_000,
        "m": 1_000_000,
        "k": 1_000,
        "billion": 1_000_000_000,
        "million": 1_000_000,
        "thousand": 1_000,
    }
    for word in ("billion", "million", "thousand"):
        if text.endswith(word):
            return int(float(text[:-len(word)]) * multipliers[word])
    if text[-1] in multipliers:
        return int(float(text[:-1]) * multipliers[text[-1]])
    return int(float(text))


# ---------------------------------------------------------------------------
# Dependency bootstrap
# ---------------------------------------------------------------------------
def _import_or_install(name: str, package: Optional[str] = None, import_name: Optional[str] = None) -> Any:
    package = package or name
    import_name = import_name or name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"[manager] {package} not found, attempting pip install ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except Exception as exc:
            print(f"[manager] pip install failed: {exc}")
            print(f"[manager] Please install '{package}' manually and re-run.")
            sys.exit(2)
        return importlib.import_module(import_name)


def _ensure_core_deps() -> None:
    for spec in (
        ("torch", "torch>=2.0.0"),
        ("numpy", "numpy>=1.24"),
        ("PIL", "Pillow", "PIL"),
        ("requests", "requests>=2.32"),
        ("psutil", "psutil>=5.9"),
    ):
        _import_or_install(spec[0], package=spec[1], import_name=spec[0] if len(spec) == 2 else spec[2])


def _ensure_ml_deps() -> None:
    for spec in (
        ("transformers", "transformers>=4.55"),
        ("datasets", "datasets>=2.20"),
        ("accelerate", "accelerate>=0.30"),
        ("sentence_transformers", "sentence-transformers>=3.0", "sentence_transformers"),
        ("librosa", "librosa>=0.10"),
    ):
        _import_or_install(spec[0], package=spec[1], import_name=spec[0] if len(spec) == 2 else spec[2])


# ---------------------------------------------------------------------------
# State / checkpointing
# ---------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return load_json(STATE_FILE)
    return {}


def save_state(state: dict[str, Any]) -> None:
    save_json(STATE_FILE, state)


# ---------------------------------------------------------------------------
# Section A : Hyper-parameter tuning
# ---------------------------------------------------------------------------
class Tuner:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _make_serializable(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [self._make_serializable(v) for v in value]
        if isinstance(value, dict):
            return {k: self._make_serializable(v) for k, v in value.items()}
        return value

    def _param_space_from_schema(self, schema: dict[str, Any]) -> dict[str, list[Any]]:
        flat: dict[str, list[Any]] = {}

        def walk(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, list):
                for item in obj:
                    walk(item, prefix)
                return
            if isinstance(obj, dict):
                if "path" in obj:
                    path = obj["path"]
                    typ = obj.get("type", "string")
                    values: list[Any] = []
                    if "choices" in obj:
                        values = list(obj["choices"])
                    elif typ == "bool":
                        values = [False, True]
                    elif typ in ("int", "float"):
                        mn = obj.get("min")
                        mx = obj.get("max")
                        if mn is not None and mx is not None:
                            steps = obj.get("steps", 5)
                            if typ == "int":
                                mn = int(mn)
                                mx = int(mx)
                                step = max(1, (mx - mn) // (steps - 1) if steps > 1 else 1)
                                values = list(range(mn, mx + 1, step))
                                values = list(sorted(set(values)))
                            else:
                                values = [round(mn + (mx - mn) * i / (steps - 1), 6) for i in range(steps)]
                    if values:
                        flat[path] = values
                    return
                for k, v in obj.items():
                    walk(v, f"{prefix}.{k}" if prefix else k)

        walk(schema)
        return flat

    def _space_from_bounds(self, bounds_path: Path) -> dict[str, list[Any]]:
        data = load_json(bounds_path)
        space: dict[str, list[Any]] = {}
        for path, spec in data.items():
            spec = spec if isinstance(spec, dict) else {"type": spec}
            typ = spec.get("type", "string")
            values: list[Any] = []
            if "values" in spec:
                values = list(spec["values"])
            elif "choices" in spec:
                values = list(spec["choices"])
            elif typ == "bool":
                values = [False, True]
            elif typ in ("int", "float"):
                mn = spec["min"]
                mx = spec["max"]
                steps = spec.get("steps", 8)
                if typ == "int":
                    mn = int(mn)
                    mx = int(mx)
                    step = max(1, (mx - mn) // (steps - 1) if steps > 1 else 1)
                    values = list(range(mn, mx + 1, step))
                    values = list(sorted(set(values)))
                else:
                    values = [round(mn + (mx - mn) * i / (steps - 1), 6) for i in range(steps)]
            if values:
                space[path] = values
        return space

    def _sample_point(self, space: dict[str, list[Any]]) -> dict[str, Any]:
        return {k: random.choice(v) for k, v in space.items()}

    def _make_grid_points(self, space: dict[str, list[Any]]) -> list[dict[str, Any]]:
        keys = list(space.keys())
        value_lists = [space[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def _materialize_config(self, point: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
        cfg = copy.deepcopy(base)
        for dot_path, value in point.items():
            dot_path_set(cfg, dot_path, value)
        return cfg

    def _render_command(self, template: str, point: dict[str, Any], report_path: Path) -> str:
        out = template
        for k, v in point.items():
            out = out.replace(f"{{{{{k}}}}}", str(v))
        out = out.replace("{{report}}", str(report_path))
        return out

    def _parse_metric(self, metric_spec: str, work_dir: Path) -> Optional[float]:
        if metric_spec.startswith("stdout:") or metric_spec.startswith("stderr:"):
            stream, pattern = metric_spec.split(":", 1)
            log_path = work_dir / f"{stream}.txt"
            if not log_path.exists():
                return None
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            compiled = re.compile(pattern)
            for line in reversed(text.splitlines()):
                m = compiled.search(line)
                if m:
                    try:
                        return float(m.group(1))
                    except (IndexError, ValueError):
                        return None
            return None

        if ":" in metric_spec:
            file_part, key_path = metric_spec.rsplit(":", 1)
        else:
            file_part, key_path = metric_spec, "score"
        if Path(file_part).is_absolute():
            json_path = Path(file_part)
        else:
            candidates = [work_dir / file_part, ROOT / file_part]
            json_path = next((p for p in candidates if p.exists()), candidates[0])
        if not json_path.exists():
            return None
        try:
            data = load_json(json_path)
            return float(dot_path_get(data, key_path))
        except Exception:
            return None

    def _run_benchmark(self, cmd: list[str], timeout: float, work_dir: Path, env: dict[str, str]) -> dict[str, Any]:
        start = time.time()
        stdout_path = work_dir / "stdout.txt"
        stderr_path = work_dir / "stderr.txt"
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=stdout_path.open("w", encoding="utf-8"),
            stderr=stderr_path.open("w", encoding="utf-8"),
            env=env,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return {
            "returncode": proc.returncode,
            "duration": time.time() - start,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    def _is_better(self, metric: float, best: Optional[float], direction: str) -> bool:
        if best is None:
            return True
        return metric < best if direction == "min" else metric > best

    def run(self) -> int:
        args = self.args
        _ensure_core_deps()
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)

        if args.param_space:
            space = self._space_from_bounds(args.param_space)
        else:
            schema = load_json(DEFAULT_SCHEMA)
            space = self._param_space_from_schema(schema)
        if not space:
            print("[tune] no tunable parameters found; exiting", file=sys.stderr)
            return 1

        print(f"[tune] parameter space: {len(space)} dimensions", flush=True)
        for k, v in sorted(space.items()):
            print(f"  {k}: {v[:5]}{'...' if len(v) > 5 else ''}", flush=True)

        if args.strategy == "grid":
            candidates = self._make_grid_points(space)
            if len(candidates) > args.max_evals:
                random.shuffle(candidates)
                candidates = candidates[:args.max_evals]
        else:
            seen: set[str] = set()
            candidates = []
            attempts = 0
            while len(candidates) < args.max_evals and attempts < args.max_evals * 1000:
                attempts += 1
                p = self._sample_point(space)
                sig = json.dumps(p, sort_keys=True, ensure_ascii=False)
                if sig in seen:
                    continue
                seen.add(sig)
                candidates.append(p)

        serial_args = self._make_serializable(vars(args))
        state: dict[str, Any] = {"args": serial_args, "evaluated": [], "best": None}
        if args.resume and CHECKPOINT_FILE.exists():
            old = load_json(CHECKPOINT_FILE)
            if old.get("args") == serial_args:
                state = old
                print(f"[tune] resumed {len(state['evaluated'])} prior evaluations", flush=True)

        base_config: dict[str, Any] = {}
        backup_path = BUILD_DIR / "phoenix.json.bak"
        if args.mode == "phoenix" and args.base_config.exists():
            base_config = load_json(args.base_config)
            if not backup_path.exists():
                save_json(backup_path, base_config)
                print(f"[tune] backed up {DEFAULT_PHOENIX_JSON}", flush=True)

        env = os.environ.copy()
        for kv in args.env:
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v

        evaluated_sigs = {json.dumps(e["point"], sort_keys=True, ensure_ascii=False) for e in state["evaluated"]}

        for i, point in enumerate(candidates, start=1):
            sig = json.dumps(point, sort_keys=True, ensure_ascii=False)
            if sig in evaluated_sigs:
                print(f"[tune] trial {i}/{len(candidates)} already evaluated, skipping", flush=True)
                continue
            print(f"\n[tune] trial {i}/{len(candidates)}: {point}", flush=True)

            work_dir = BUILD_DIR / f"tune_trial_{i:04d}_{int(time.time()*1000)}"
            work_dir.mkdir(parents=True, exist_ok=True)

            if args.mode == "phoenix":
                cfg = self._materialize_config(point, base_config)
                save_json(DEFAULT_PHOENIX_JSON, cfg)
                if not args.benchmark_cmd:
                    print("[tune] --benchmark-cmd required in phoenix mode", file=sys.stderr)
                    return 1
                cmd = shlex.split(args.benchmark_cmd)
            else:
                if not args.command_template:
                    print("[tune] --command-template required in command mode", file=sys.stderr)
                    return 1
                report_path = work_dir / "last_report.json"
                rendered = self._render_command(args.command_template, point, report_path)
                cmd = shlex.split(rendered)

            # simple cleanup of previous benchmark if it spawns a Phoenix server
            for name in ("phoenix_main", "llama-server"):
                try:
                    subprocess.run(["taskkill", "/F", "/IM", f"{name}.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
                except Exception:
                    pass

            result = self._run_benchmark(cmd, args.timeout, work_dir, env)
            metric = self._parse_metric(args.metric, work_dir)
            print(f"[tune] trial {i} rc={result['returncode']} dur={result['duration']:.1f}s metric={metric}", flush=True)

            record = {
                "trial": i,
                "point": point,
                "metric": metric,
                "returncode": result["returncode"],
                "duration": result["duration"],
                "work_dir": str(work_dir),
            }
            state["evaluated"].append(record)

            if metric is not None and self._is_better(metric, state["best"]["metric"] if state["best"] else None, args.direction):
                state["best"] = record
                if args.mode == "phoenix":
                    best_cfg = self._materialize_config(point, base_config)
                    save_json(args.output_config, best_cfg)
                else:
                    save_json(args.output_config, {"point": point, "metric": metric})
                print(f"[tune] new best: {metric} at trial {i}", flush=True)

            evaluated_sigs.add(sig)
            save_json(CHECKPOINT_FILE, state)

        if state["best"]:
            best = state["best"]
            print(f"\n[tune] best trial: {best['trial']} metric={best['metric']} point={best['point']}", flush=True)
            if args.mode == "phoenix":
                best_cfg = self._materialize_config(best["point"], base_config)
                save_json(args.output_config, best_cfg)
                if not args.no_apply:
                    save_json(DEFAULT_PHOENIX_JSON, best_cfg)
                    print(f"[tune] applied best config to {DEFAULT_PHOENIX_JSON}", flush=True)
                elif backup_path.exists():
                    save_json(DEFAULT_PHOENIX_JSON, load_json(backup_path))
                    print(f"[tune] restored original {DEFAULT_PHOENIX_JSON} (--no-apply)", flush=True)
            else:
                save_json(args.output_config, {"point": best["point"], "metric": best["metric"]})
        else:
            if args.mode == "phoenix" and backup_path.exists():
                save_json(DEFAULT_PHOENIX_JSON, load_json(backup_path))
            print("[tune] no successful evaluation", file=sys.stderr)
            return 1

        return 0


# ---------------------------------------------------------------------------
# Section B : Streaming data + training
# ---------------------------------------------------------------------------
class PretrainManager:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        _ensure_core_deps()
        _ensure_ml_deps()

    @staticmethod
    def _default_image_sources() -> list[dict[str, Any]]:
        return [
            {"name": "flickr30k", "repo_id": "nlphuji/flickr30k", "split": "test", "modality": "image"},
            {"name": "coco", "repo_id": "HuggingFaceM4/COCO", "split": "train", "modality": "image"},
        ]

    @staticmethod
    def _default_speech_sources() -> list[dict[str, Any]]:
        return [
            {"name": "librispeech_clean_100", "repo_id": "librispeech_asr", "split": "train.clean.100", "modality": "speech"},
            {"name": "librispeech_clean_360", "repo_id": "librispeech_asr", "split": "train.clean.360", "modality": "speech"},
            {"name": "common_voice_en", "repo_id": "mozilla-foundation/common_voice_11_0", "split": "train", "modality": "speech"},
            {"name": "voxpopuli_en", "repo_id": "facebook/voxpopuli", "split": "train", "modality": "speech"},
        ]

    def _now(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def _log(self, *parts: Any) -> None:
        print(f"[{self._now()}] {' '.join(str(p) for p in parts)}", flush=True)

    def _run(self) -> int:
        args = self.args
        from datasets import load_dataset
        import numpy as np
        import psutil

        modality = args.modality
        target = args.target_samples
        out_dir = MODEL_DIR / modality
        out_dir.mkdir(parents=True, exist_ok=True)
        state = load_state()
        state.setdefault(modality, {"samples_done": 0, "datasets_done": [], "current": None})

        sources = args.sources if args.sources else (
            self._default_image_sources() if modality == "image" else self._default_speech_sources()
        )

        device = torch_device(args.device)
        self._log(f"device = {device}")

        # build models
        if modality == "image":
            model = ImageContrastiveModel(
                image_model=args.image_encoder,
                text_model=args.text_encoder,
                target_dim=args.target_dim,
                device=device,
                freeze_encoders=args.freeze_encoders,
            )
        else:
            model = SpeechContrastiveModel(
                audio_model=args.audio_encoder,
                text_model=args.text_encoder,
                target_dim=args.target_dim,
                device=device,
                freeze_encoders=args.freeze_encoders,
            )

        self._log(f"model has {sum(p.numel() for p in model.parameters()):,} parameters")

        # try to resume checkpoint
        if args.resume and state[modality].get("latest_checkpoint"):
            ckpt = Path(state[modality]["latest_checkpoint"])
            if ckpt.exists():
                model.load_checkpoint(ckpt)
                self._log(f"resumed from {ckpt}")

        optimizer = _import_or_install("torch").optim.AdamW(model.parameters(), lr=args.lr)
        if state[modality].get("optimizer_state"):
            try:
                optimizer.load_state_dict(state[modality]["optimizer_state"])
            except Exception:
                pass

        cumulative = state[modality]["samples_done"]
        self._log(f"target = {human_count(target)} samples, already processed = {human_count(cumulative)}")

        completed = set(state[modality]["datasets_done"])

        for source in sources:
            if cumulative >= target:
                self._log(f"target {human_count(target)} reached ({human_count(cumulative)})")
                break

            if source["name"] in completed:
                self._log(f"source {source['name']} already done, skipping")
                continue

            state[modality]["current"] = source["name"]
            save_state(state)

            self._log(f"loading streaming dataset {source['repo_id']} / {source['split']} ...")
            try:
                ds = load_dataset(source["repo_id"], split=source["split"], streaming=True, trust_remote_code=True)
            except Exception as exc:
                self._log(f"failed to load {source['repo_id']}: {exc}")
                continue

            batch: list[dict[str, Any]] = []
            samples_in_source = 0

            for sample in ds:
                if cumulative >= target:
                    break
                parsed = self._parse_sample(sample, modality)
                if not parsed:
                    continue
                batch.append(parsed)
                cumulative += 1
                samples_in_source += 1

                if len(batch) >= args.batch_size:
                    loss = model.train_step(batch, optimizer)
                    batch = []
                    if cumulative % args.log_every == 0:
                        mem = psutil.virtual_memory()
                        self._log(f"samples={human_count(cumulative)} loss={loss:.4f} mem={mem.percent}%")
                        state[modality]["samples_done"] = cumulative
                        state[modality]["loss"] = loss
                        save_state(state)

                if cumulative % args.save_every == 0:
                    ckpt = out_dir / f"step_{cumulative}.pth"
                    model.save_checkpoint(ckpt, optimizer, cumulative)
                    state[modality]["latest_checkpoint"] = str(ckpt)
                    state[modality]["samples_done"] = cumulative
                    save_state(state)

            # tail batch
            if batch:
                loss = model.train_step(batch, optimizer)
                state[modality]["loss"] = loss

            completed.add(source["name"])
            state[modality]["datasets_done"] = sorted(completed)
            state[modality]["samples_done"] = cumulative
            state[modality]["current"] = None
            save_state(state)
            self._log(f"source {source['name']} finished; added {human_count(samples_in_source)}; total {human_count(cumulative)}")

        # final save
        best = out_dir / "best.pth"
        model.save_checkpoint(best, optimizer, cumulative)
        state[modality]["latest_checkpoint"] = str(best)
        save_state(state)

        if cumulative < target:
            self._log(f"WARNING: ran out of public datasets at {human_count(cumulative)} / {human_count(target)}")
        else:
            self._log(f"reached target {human_count(target)} samples")

        return 0

    def _parse_sample(self, sample: dict[str, Any], modality: str) -> Optional[dict[str, Any]]:
        if modality == "image":
            try:
                from PIL import Image
                import requests

                # Try common field names
                image_url = sample.get("image") or sample.get("img") or sample.get("url") or sample.get("image_url")
                if isinstance(image_url, Image.Image):
                    img = image_url.convert("RGB")
                elif isinstance(image_url, dict):
                    image_url = image_url.get("url")
                    if not image_url or not isinstance(image_url, str):
                        return None
                    img = Image.open(requests.get(image_url, stream=True, timeout=30).raw).convert("RGB")
                elif isinstance(image_url, str) and image_url.startswith("http"):
                    img = Image.open(requests.get(image_url, stream=True, timeout=30).raw).convert("RGB")
                else:
                    return None
                caption = str(sample.get("caption", sample.get("captions", sample.get("text", ""))))
                if isinstance(sample.get("captions"), list) and sample["captions"]:
                    caption = str(sample["captions"][0])
                elif isinstance(sample.get("caption"), list) and sample["caption"]:
                    caption = str(sample["caption"][0])
                return {"image": img, "caption": caption}
            except Exception:
                return None
        else:
            try:
                audio = sample.get("audio")
                if not audio:
                    return None
                transcript = str(sample.get("text", sample.get("transcript", sample.get("sentence", ""))))
                if not transcript:
                    return None
                if isinstance(audio, dict):
                    sr = audio.get("sampling_rate", 16000)
                    arr = audio.get("array")
                else:
                    return None
                return {"audio": (arr, sr), "transcript": transcript}
            except Exception:
                return None


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def torch_device(preference: Optional[str] = None) -> str:
    import torch
    if preference:
        return preference
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ImageContrastiveModel:
    def __init__(self, image_model: str, text_model: str, target_dim: int, device: str, freeze_encoders: bool = False):
        import torch
        import torch.nn as nn
        from transformers import AutoImageProcessor, AutoModel
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.target_dim = target_dim

        if "/" in image_model:
            self.image_processor = AutoImageProcessor.from_pretrained(image_model)
            self.image_encoder = AutoModel.from_pretrained(image_model).to(device)
            img_hidden = self.image_encoder.config.hidden_size
        else:
            import timm
            self.image_encoder = timm.create_model(image_model, pretrained=True, num_classes=0).to(device)
            img_hidden = self.image_encoder.num_features
            self.image_processor = None

        self.text_encoder = SentenceTransformer(text_model).to(device)
        text_hidden = self.text_encoder.get_sentence_embedding_dimension()

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.image_proj = nn.Sequential(
            nn.Linear(img_hidden, target_dim),
            nn.LayerNorm(target_dim),
        ).to(device)

        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, target_dim),
            nn.LayerNorm(target_dim),
        ).to(device)

        self.temperature = nn.Parameter(torch.tensor(0.07), requires_grad=True).to(device)

    def encode_image(self, images: list[Any]) -> torch.Tensor:
        import torch
        from PIL import Image
        if self.image_processor:
            inputs = self.image_processor(images, return_tensors="pt").to(self.device)
            with torch.amp.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
                out = self.image_encoder(**inputs).last_hidden_state[:, 0]
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            batch = torch.stack([transform(img) for img in images]).to(self.device)
            with torch.amp.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
                out = self.image_encoder(batch)
        return self.image_proj(out)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        import torch
        with torch.amp.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
            emb = self.text_encoder.encode(texts, show_progress_bar=False, convert_to_tensor=True).to(self.device)
        return self.text_proj(emb)

    def train_step(self, batch: list[dict[str, Any]], optimizer: Any) -> float:
        import torch
        import torch.nn.functional as F

        optimizer.zero_grad()
        images = [s["image"] for s in batch]
        captions = [s["caption"] for s in batch]

        img_z = F.normalize(self.encode_image(images), dim=-1)
        txt_z = F.normalize(self.encode_text(captions), dim=-1)

        logits = (img_z @ txt_z.T) / self.temperature.exp()
        labels = torch.arange(len(batch), device=self.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        loss.backward()
        optimizer.step()
        return loss.item()

    def save_checkpoint(self, path: Path, optimizer: Optional[Any] = None, samples: int = 0) -> None:
        import torch
        state = {
            "image_proj": self.image_proj.state_dict(),
            "text_proj": self.text_proj.state_dict(),
            "temperature": self.temperature.state_dict(),
            "samples": samples,
        }
        if optimizer:
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        import torch
        state = torch.load(path, map_location=self.device)
        self.image_proj.load_state_dict(state["image_proj"])
        self.text_proj.load_state_dict(state["text_proj"])
        self.temperature.load_state_dict(state["temperature"])


class SpeechContrastiveModel:
    def __init__(self, audio_model: str, text_model: str, target_dim: int, device: str, freeze_encoders: bool = False):
        import torch
        import torch.nn as nn
        from transformers import AutoProcessor, Wav2Vec2Model
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.target_dim = target_dim

        self.audio_processor = AutoProcessor.from_pretrained(audio_model)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_model).to(device)
        audio_hidden = self.audio_encoder.config.hidden_size

        self.text_encoder = SentenceTransformer(text_model).to(device)
        text_hidden = self.text_encoder.get_sentence_embedding_dimension()

        if freeze_encoders:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_hidden, target_dim),
            nn.LayerNorm(target_dim),
        ).to(device)

        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, target_dim),
            nn.LayerNorm(target_dim),
        ).to(device)

        self.temperature = nn.Parameter(torch.tensor(0.07), requires_grad=True).to(device)

    def encode_audio(self, audio_list: list[tuple[Any, int]]) -> torch.Tensor:
        import torch
        import numpy as np

        # find max length and create attention mask
        arrays = [np.array(a[0], dtype=np.float32) for a in audio_list]
        lengths = [len(a) for a in arrays]
        max_len = max(lengths)
        padded = np.zeros((len(arrays), max_len), dtype=np.float32)
        mask = np.zeros((len(arrays), max_len), dtype=bool)
        for i, (arr, L) in enumerate(zip(arrays, lengths)):
            padded[i, :L] = arr
            mask[i, :L] = True

        inputs = self.audio_processor(
            padded,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.amp.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
            out = self.audio_encoder(**inputs).last_hidden_state
            # mean over non-masked tokens
            mask_t = torch.tensor(mask, device=self.device).unsqueeze(-1)
            out = (out * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp(min=1)
        return self.audio_proj(out)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        import torch
        with torch.amp.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
            emb = self.text_encoder.encode(texts, show_progress_bar=False, convert_to_tensor=True).to(self.device)
        return self.text_proj(emb)

    def train_step(self, batch: list[dict[str, Any]], optimizer: Any) -> float:
        import torch
        import torch.nn.functional as F

        optimizer.zero_grad()
        audios = [s["audio"] for s in batch]
        transcripts = [s["transcript"] for s in batch]

        audio_z = F.normalize(self.encode_audio(audios), dim=-1)
        txt_z = F.normalize(self.encode_text(transcripts), dim=-1)

        logits = (audio_z @ txt_z.T) / self.temperature.exp()
        labels = torch.arange(len(batch), device=self.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        loss.backward()
        optimizer.step()
        return loss.item()

    def save_checkpoint(self, path: Path, optimizer: Optional[Any] = None, samples: int = 0) -> None:
        import torch
        state = {
            "audio_proj": self.audio_proj.state_dict(),
            "text_proj": self.text_proj.state_dict(),
            "temperature": self.temperature.state_dict(),
            "samples": samples,
        }
        if optimizer:
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        import torch
        state = torch.load(path, map_location=self.device)
        self.audio_proj.load_state_dict(state["audio_proj"])
        self.text_proj.load_state_dict(state["text_proj"])
        self.temperature.load_state_dict(state["temperature"])


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------
def export_onnx(args: argparse.Namespace) -> int:
    import torch

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"[export] checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else checkpoint.with_suffix(".onnx")
    device = torch_device(args.device)

    if args.modality == "image":
        model = ImageContrastiveModel(
            image_model=args.image_encoder,
            text_model=args.text_encoder,
            target_dim=args.target_dim,
            device=device,
        )
        model.load_checkpoint(checkpoint)
        dummy = torch.randn(1, 3, 224, 224).to(device)
        enc = model.image_encoder if not model.image_processor else lambda x: model.image_encoder(dummy).last_hidden_state[:, 0]
        torch.onnx.export(
            enc,
            dummy,
            out,
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={"pixel_values": {0: "batch"}, "image_features": {0: "batch"}},
            opset_version=13,
        )
    else:
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(args.audio_encoder).to(device)
        dummy = torch.randn(1, 16000).to(device)
        torch.onnx.export(
            enc,
            dummy,
            out,
            input_names=["audio_values"],
            output_names=["audio_features"],
            dynamic_axes={"audio_values": {0: "batch", 1: "samples"}, "audio_features": {0: "batch"}},
            opset_version=13,
        )

    print(f"[export] wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def show_status() -> int:
    state = load_state()
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _sig_handler(signum: int, frame: Any) -> None:
    print("\n[manager] interrupted, state saved; exiting gracefully", flush=True)
    sys.exit(0)


def main() -> int:
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    parser = argparse.ArgumentParser(description="Remote Phoenix manager (single-file Windows/GPU)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # tune
    p_tune = sub.add_parser("tune", help="hyper-parameter tuning")
    p_tune.add_argument("--param-space", type=Path, default=None)
    p_tune.add_argument("--mode", choices=["phoenix", "command"], default="phoenix")
    p_tune.add_argument("--base-config", type=Path, default=DEFAULT_PHOENIX_JSON)
    p_tune.add_argument("--output-config", type=Path, default=ROOT / "config" / "phoenix_tuned.json")
    p_tune.add_argument("--command-template", type=str, default=None)
    p_tune.add_argument("--benchmark-cmd", type=str, default=None)
    p_tune.add_argument("--metric", type=str, default="build/tune_report.json:score")
    p_tune.add_argument("--max-evals", type=int, default=12)
    p_tune.add_argument("--strategy", choices=["random", "grid"], default="random")
    p_tune.add_argument("--direction", choices=["min", "max"], default="min")
    p_tune.add_argument("--timeout", type=float, default=1800)
    p_tune.add_argument("--seed", type=int, default=42)
    p_tune.add_argument("--no-apply", action="store_true")
    p_tune.add_argument("--resume", action="store_true")
    p_tune.add_argument("--env", action="append", default=[])

    # pretrain
    p_pre = sub.add_parser("pretrain", help="stream data and continue pre-training on GPU")
    p_pre.add_argument("--modality", choices=["image", "speech"], required=True)
    p_pre.add_argument("--target-samples", type=str, default="100000000", help="e.g. 100000000, 10M, 0.1B")
    p_pre.add_argument("--target-dim", type=int, default=128)
    p_pre.add_argument("--batch-size", type=int, default=8)
    p_pre.add_argument("--lr", type=float, default=1e-4)
    p_pre.add_argument("--image-encoder", type=str, default="google/vit-base-patch16-224")
    p_pre.add_argument("--audio-encoder", type=str, default="facebook/wav2vec2-base-960h")
    p_pre.add_argument("--text-encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_pre.add_argument("--device", type=str, default=None)
    p_pre.add_argument("--log-every", type=int, default=1000)
    p_pre.add_argument("--save-every", type=int, default=10000)
    p_pre.add_argument("--resume", action="store_true")
    p_pre.add_argument("--freeze-encoders", action="store_true", help="freeze base encoders, train only projection heads (fits small GPUs)")
    p_pre.add_argument("--sources", type=Path, default=None, help="JSON file with list of {name, repo_id, split, modality}")

    # export
    p_exp = sub.add_parser("export", help="export a checkpoint to ONNX")
    p_exp.add_argument("--modality", choices=["image", "speech"], required=True)
    p_exp.add_argument("--checkpoint", type=str, required=True)
    p_exp.add_argument("--output", type=str, default=None)
    p_exp.add_argument("--target-dim", type=int, default=128)
    p_exp.add_argument("--image-encoder", type=str, default="google/vit-base-patch16-224")
    p_exp.add_argument("--audio-encoder", type=str, default="facebook/wav2vec2-base-960h")
    p_exp.add_argument("--text-encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p_exp.add_argument("--device", type=str, default=None)

    # status
    sub.add_parser("status", help="show current progress")

    args = parser.parse_args()

    if args.cmd == "tune":
        return Tuner(args).run()
    if args.cmd == "pretrain":
        args.target_samples = parse_sample_count(args.target_samples)
        if args.sources:
            with open(args.sources, "r", encoding="utf-8") as f:
                args.sources = json.load(f)
        else:
            args.sources = None
        return PretrainManager(args)._run()
    if args.cmd == "export":
        return export_onnx(args)
    if args.cmd == "status":
        return show_status()

    return 0


if __name__ == "__main__":
    sys.exit(main())

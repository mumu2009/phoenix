"""
Headless launcher for phoenix_main.exe.
Reads runtime_store/start_079_launcher.json, kills stale processes,
starts phoenix (which auto-launches llamacpp via external_auto_launch).
Prints the PID and waits for the process to be ready on the gateway port.
Usage: python tools/launch_headless.py [--wait-port 5080] [--wait-timeout 60]
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent


def load_config(root: Path) -> dict:
    cfg_path = root / "runtime_store" / "start_079_launcher.json"
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)["values"]


def kill_stale(root: Path) -> None:
    print("[headless] killing stale phoenix_main.exe / bug_shooter.exe ...", flush=True)
    for name in ["phoenix_main.exe", "bug_shooter.exe"]:
        subprocess.run(
            ["taskkill", "/IM", name, "/F"],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    # Kill anything listening on 8082 (llamacpp adapter)
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL, cwd=root
        )
        for line in out.splitlines():
            if ":8082 " in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                print(f"[headless] killing PID {pid} on port 8082", flush=True)
                subprocess.run(
                    ["taskkill", "/PID", pid, "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                )
    except Exception:
        pass
    time.sleep(2)


def wait_port(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def ensure_gguf_extension(model_raw: str, root: Path) -> str:
    """
    If the model path exists but has no .gguf suffix (Ollama blob format),
    create a hard-link with .gguf suffix next to it and return the new path.
    Returns the original string if it already ends with .gguf or does not exist.
    """
    if not model_raw:
        return model_raw
    p = Path(model_raw)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        return model_raw
    if p.suffix.lower() == ".gguf":
        return model_raw
    # Create hard-link with .gguf extension
    link = p.with_suffix(".gguf")
    if not link.exists():
        try:
            os.link(p, link)
            print(f"[headless] created hard-link: {link.name}", flush=True)
        except Exception as exc:
            print(f"[headless] WARNING: could not create hard-link ({exc}); trying symlink", flush=True)
            try:
                link.symlink_to(p)
                print(f"[headless] created symlink: {link.name}", flush=True)
            except Exception as exc2:
                print(f"[headless] WARNING: symlink also failed ({exc2}); using original path", flush=True)
                return model_raw
    return str(link)


def build_phoenix_args(root: Path, cfg: dict) -> list[str]:
    exe = root / str(cfg.get("phoenix_executable", "phoenix_main.exe"))
    args = [str(exe)]

    def add(flag: str, key: str, default: str = "") -> None:
        val = str(cfg.get(key, default)).strip()
        if val:
            args.append(f"--{flag}={val}")

    def add_bool(flag: str, key: str, default: bool = False) -> None:
        val = cfg.get(key, default)
        if isinstance(val, str):
            val = val.lower() in ("true", "1", "yes")
        args.append(f"--{flag}={'true' if val else 'false'}")

    add("gateway-host", "gateway_host", "127.0.0.1")
    add("port", "gateway_port", "5080")
    add("study-port", "study_port", "5081")
    add("base-dir", "base_dir", "runtime_store")
    add("db-path", "db_path", "runtime_store/ai_store.sqlite")
    add("log-mode", "log_mode", "release")
    add("transformer-mode", "transformer_mode", "llamacpp")

    # Resolve model path — fix Ollama blob paths that lack .gguf extension
    raw_model = str(cfg.get("llamacpp_model", "")).strip()
    resolved_model = ensure_gguf_extension(raw_model, root)
    if resolved_model:
        args.append(f"--llamacpp-model={resolved_model}")

    add("llamacpp-base-url", "llamacpp_base_url", "http://127.0.0.1:8082")
    add("llamacpp-ctx-size", "llamacpp_ctx_size", "8192")
    add("llamacpp-batch-size", "llamacpp_batch_size", "16")
    add("llamacpp-ubatch-size", "llamacpp_ubatch_size", "8")
    add("gguf-models-dir", "gguf_models_dir", "GGUF_models")
    add("robots-dir", "robots_dir", "robots")
    add("lmdb-dir", "lmdb_dir", "lmdb")
    add("redis-url", "redis_url", "redis://127.0.0.1:6379")
    add_bool("external-auto-launch", "external_auto_launch", True)
    add_bool("tests-autoload", "tests_autoload", True)
    add_bool("robots-autoload", "robots_autoload", True)
    add_bool("frontend-enabled", "frontend_enabled", True)
    add_bool("inference-enabled", "inference_enabled", True)
    return args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-port", type=int, default=5080)
    parser.add_argument("--wait-timeout", type=int, default=60)
    parser.add_argument("--no-kill", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    cfg = load_config(root)

    if not args.no_kill:
        kill_stale(root)

    phoenix_args = build_phoenix_args(root, cfg)
    print(f"[headless] launching: {phoenix_args[0]}", flush=True)

    proc = subprocess.Popen(phoenix_args, cwd=root)
    print(f"[headless] PID={proc.pid}", flush=True)

    host = str(cfg.get("gateway_host", "127.0.0.1"))
    port = int(str(cfg.get("gateway_port", "5080")))
    print(f"[headless] waiting for {host}:{port} (up to {args.wait_timeout}s)...", flush=True)

    if wait_port(host, port, args.wait_timeout):
        print(f"[headless] phoenix ready on port {port}", flush=True)
        return 0
    else:
        print(f"[headless] TIMEOUT: phoenix did not bind port {port} within {args.wait_timeout}s", flush=True)
        proc.terminate()
        return 1


if __name__ == "__main__":
    sys.exit(main())

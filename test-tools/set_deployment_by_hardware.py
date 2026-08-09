#!/usr/bin/env python3
"""Detect target hardware and set model_deployment.localBackend in config/phoenix.json.

Never guesses from model file type.  It looks at hardware/runtime evidence:
- PHOENIX_EDGE_DEVICE env var (rdk_x5, rk3588, jetson_nano, ...)
- platform.machine()
- presence of Horizon BPU runtime on Linux (hobot-dnn / hb_dnn.h)
- availability of GPU runtime

The explicit localBackend values are then written to config/phoenix.json so the
C++ runtime has exactly one path and no fallback opportunity.
"""
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "phoenix.json"


def has_horizon_bpu() -> bool:
    """True if this machine has the Horizon BPU runtime headers available."""
    if platform.system() != "Linux":
        return False
    return (
        Path("/usr/include/dnn/hb_dnn.h").exists()
        or Path("/usr/local/include/dnn/hb_dnn.h").exists()
        or any(p.exists() for p in Path("/").glob("**/hb_dnn.h"))
    )


def has_rk3588_runtime() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return shutil.which("rknnlite") is not None or any(
            p.exists() for p in Path("/").glob("**/librknn_api*")
        )
    except Exception:
        return False


def has_tensorrt() -> bool:
    if platform.system() == "Windows":
        return False
    return (
        shutil.which("trtexec") is not None
        or Path("/usr/local/lib/libnvinfer.so").exists()
    )


def detect_backend() -> str:
    env = os.environ.get("PHOENIX_EDGE_DEVICE", "").lower()
    if env in ("rdk_x5", "rdk_s100"):
        return "bpu"
    if env == "rk3588":
        return "rknn"
    if env == "jetson_nano":
        return "trt"

    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        if has_horizon_bpu():
            return "bpu"
        if has_rk3588_runtime():
            return "rknn"
        if has_tensorrt():
            return "trt"
        return "cpu"  # plain aarch64 ORT

    if machine in ("x86_64", "amd64", "i386", "i686"):
        return "cpu"

    return "cpu"


def main():
    backend = detect_backend()
    if not CONFIG.exists():
        print(f"ERROR: {CONFIG} not found", file=sys.stderr)
        sys.exit(1)

    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("model_deployment", {})
    for k in ("vision", "speech"):
        cfg["model_deployment"].setdefault(k, {})
        cfg["model_deployment"][k]["localBackend"] = backend

    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"[set_deployment_by_hardware] detected={backend} backend={backend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

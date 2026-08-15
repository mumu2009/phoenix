#!/usr/bin/env python3
"""
simulate_edge_devices.py

Use VirtualBox / QEMU or a local ONNX Runtime fallback to evaluate the four
image/audio models on edge platforms that are not physically available.

Supported target profiles:
  - rdk_x5 / rdk_s100 : Horizon BPU  (.bin)
  - rk3588            : Rockchip NPU (.rknn, fallback .onnx)
  - jetson_nano       : NVIDIA TensorRT (.trt, fallback .onnx)

If a VM disk image is configured in config/edge_devices.json the script will
attempt to start the VM via VBoxManage or qemu-system-aarch64, wait for the
SSH port, deploy the models, and run the platform-specific evaluator.

If no VM is configured, the script falls back to local ONNX Runtime using the
best ONNX artifacts under runtime_store/models/ijepa/<variant>/.

Usage:
    # Run the full local-ONNX fallback for all configured devices
    python tools/simulate_edge_devices.py --fallback

    # Start a QEMU VM for rdk_x5 (needs an aarch64 Linux image)
    python tools/simulate_edge_devices.py --device rdk_x5 \
        --vm-image /path/to/rdk_x5_qemu.qcow2 --vm-backend qemu

    # VirtualBox
    python tools/simulate_edge_devices.py --device jetson_nano \
        --vm-image /path/to/jetson_nano.ova --vm-backend vbox
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


def log(*parts: Any) -> None:
    print("[simulate]", *parts, flush=True)


def _run(cmd: List[str], **kwargs: Any) -> tuple[int, str, str]:
    log("$", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# VM management
# ---------------------------------------------------------------------------
class VMController:
    """Start / stop a VirtualBox or QEMU VM and wait for SSH."""

    def __init__(self, device: str, image: Path, backend: str, ssh_port: int = 2222):
        self.device = device
        self.image = Path(image)
        self.backend = backend
        self.ssh_port = ssh_port
        self.name = f"phoenix_sim_{device}"
        self._proc: Optional[subprocess.Popen] = None

    def start(self, timeout: int = 180) -> bool:
        if not self.image.exists():
            log(f"VM image not found: {self.image}")
            return False

        if self.backend == "vbox":
            return self._start_vbox(timeout)
        if self.backend == "qemu":
            return self._start_qemu(timeout)
        log(f"Unknown VM backend: {self.backend}")
        return False

    def _start_vbox(self, timeout: int) -> bool:
        # Import the OVA/VDI if it is not already registered
        rc, out, _ = _run(["VBoxManage", "list", "vms"])
        if self.name not in out:
            if self.image.suffix == ".ova":
                _run(["VBoxManage", "import", str(self.image), "--vsys", "0", "--vmname", self.name])
            else:
                _run(["VBoxManage", "createvm", "--name", self.name, "--register"])
                _run(["VBoxManage", "storagectl", self.name, "--name", "SATA", "--add", "sata"])
                _run(["VBoxManage", "storageattach", self.name, "--storagectl", "SATA",
                      "--port", "0", "--device", "0", "--type", "hdd", "--medium", str(self.image)])
            _run(["VBoxManage", "modifyvm", self.name, "--natpf1", f"ssh,tcp,,{self.ssh_port},,22"])
        _run(["VBoxManage", "startvm", self.name, "--type", "headless"])
        return self._wait_ssh(timeout)

    def _start_qemu(self, timeout: int) -> bool:
        # Common aarch64/QEMU smoke: replace with your real command
        qcmd = [
            "qemu-system-aarch64",
            "-machine", "virt",
            "-cpu", "cortex-a72",
            "-m", "4096",
            "-smp", "4",
            "-hda", str(self.image),
            "-netdev", "user,id=net0,hostfwd=tcp::{}-:22".format(self.ssh_port),
            "-device", "virtio-net-device,netdev=net0",
            "-nographic",
        ]
        log("Starting QEMU:", " ".join(qcmd))
        self._proc = subprocess.Popen(qcmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self._wait_ssh(timeout)

    def _wait_ssh(self, timeout: int) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with socket.create_connection(("127.0.0.1", self.ssh_port), timeout=2):
                    log(f"VM SSH reachable on port {self.ssh_port}")
                    return True
            except OSError:
                time.sleep(2.0)
        log("VM SSH did not become reachable")
        return False

    def stop(self) -> None:
        if self.backend == "vbox":
            _run(["VBoxManage", "controlvm", self.name, "poweroff"])
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Local ONNX fallback
# ---------------------------------------------------------------------------
def local_ort_eval(variant_dir: Path, kind: str) -> Dict[str, Any]:
    """Run a local ONNX Runtime inference as a hardware surrogate."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        return {"error": f"onnxruntime not installed: {e}"}

    onnx = variant_dir / f"model_{kind}.onnx"
    if not onnx.exists():
        onnx = variant_dir / "best.onnx"
    if not onnx.exists():
        return {"error": f"no {kind} ONNX found in {variant_dir}"}

    sess = ort.InferenceSession(str(onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dummy: Optional[np.ndarray] = None
    name = variant_dir.name
    if "vision" in name and kind == "encoder":
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    elif "vision" in name and kind == "decoder":
        dummy = np.random.randn(1, 128, 1, 1).astype(np.float32)
    elif "speech" in name and kind == "encoder":
        dummy = np.random.randn(1, 1, 1, 16000).astype(np.float32)
    elif "speech" in name and kind == "decoder":
        dummy = np.random.randn(1, 128, 1, 1).astype(np.float32)
    else:
        return {"error": f"unknown variant/kind: {name}/{kind}"}

    out = sess.run(None, {inp.name: dummy})
    arr = out[0]
    finite = bool(np.all(np.isfinite(arr)))
    nonzero = bool(np.any(np.abs(arr) > 1e-7))
    std = float(arr.std())
    return {
        "onnx": str(onnx),
        "input_shape": list(dummy.shape),
        "output_shape": list(arr.shape),
        "finite": finite,
        "nonzero": nonzero,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": std,
        "ok": finite and nonzero and std > 1e-7,
    }


def run_fallback() -> Dict[str, Any]:
    """Evaluate all four variants locally with ONNX Runtime."""
    report: Dict[str, Any] = {}
    candidates = [
        ROOT / "runtime_store" / "models" / "ijepa" / "remote_trained",
        ROOT / "runtime_store" / "models" / "additive_jepa",
    ]
    variants = {
        "vision_encoder": "vision_encoder",
        "vision_decoder": "vision_decoder",
        "speech_encoder": "speech_encoder",
        "speech_decoder": "speech_decoder",
    }
    for name, subdir in variants.items():
        kind = "encoder" if "encoder" in name else "decoder"
        d = None
        for c in candidates:
            if (c / subdir).is_dir():
                d = c / subdir
                break
        if d is None:
            report[name] = {"error": f"no directory for {name} in any fallback root"}
            continue
        report[name] = local_ort_eval(d, kind)
    return report


# ---------------------------------------------------------------------------
# Remote-VM evaluation (placeholder / template)
# ---------------------------------------------------------------------------
def run_vm_eval(device: str, vm_image: Path, backend: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Start a VM, deploy artifacts, and run the platform evaluator."""
    vm = VMController(device, vm_image, backend, ssh_port=cfg.get("vm_ssh_port", 2222))
    try:
        if not vm.start(timeout=cfg.get("vm_boot_timeout", 180)):
            return {"error": "VM did not start"}

        # Connect via edge_device_manager.  The device config uses 127.0.0.1:ssh_port.
        sys.path.insert(0, str(ROOT / "tools"))
        from edge_device_manager import EdgeDevice

        dev_cfg = {
            "type": device,
            "host": "127.0.0.1",
            "port": vm.ssh_port,
            "user": cfg.get("vm_user", "root"),
            "pass": cfg.get("vm_pass", "root"),
            "auth": "plain",
        }
        with EdgeDevice(dev_cfg) as edge:
            remote_root = cfg.get("remote_root", "/root/phoenix")
            edge.mkdir_p(remote_root)
            # Placeholder: deploy ONNX/BIN files and run the right evaluator.
            # The actual evaluator commands are platform-specific and need the
            # corresponding runtime libraries installed in the VM.
            if device.startswith("rdk"):
                cmd = f"cd {remote_root} && ./tools/x5_bpu_smoke --run"
            elif device == "rk3588":
                cmd = f"cd {remote_root} && python3 tools/rk3588_npu_evaluate.py"
            else:
                cmd = f"cd {remote_root} && python3 tools/jetson_trt_evaluate.py"
            rc, out, err = edge.exec(cmd)
            return {"rc": rc, "stdout": out, "stderr": err}
    finally:
        vm.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Simulate edge device execution via VM or local ORT")
    p.add_argument("--device", default="rdk_x5",
                   choices=["rdk_x5", "rdk_s100", "rk3588", "jetson_nano"],
                   help="Target edge profile")
    p.add_argument("--vm-image", type=Path, default=None,
                   help="Path to .ova, .vdi, or .qcow2 VM image")
    p.add_argument("--vm-backend", default="qemu", choices=["qemu", "vbox"],
                   help="Virtual machine backend")
    p.add_argument("--vm-user", default="root")
    p.add_argument("--vm-pass", default="root")
    p.add_argument("--vm-ssh-port", type=int, default=2222)
    p.add_argument("--vm-boot-timeout", type=int, default=180)
    p.add_argument("--fallback", action="store_true",
                   help="Ignore VM and run local ONNX Runtime fallback")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "edge_devices.json",
                   help="Edge device registry (if no --fallback)")
    args = p.parse_args()

    # Make sure edge_device_manager is importable even on Windows for fallback.
    sys.path.insert(0, str(ROOT / "tools"))

    cfg: Dict[str, Any] = {}
    if args.config.exists():
        with args.config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    if args.fallback or not args.vm_image:
        log("Running local ONNX Runtime fallback (no VM)")
        report = run_fallback()
    else:
        cfg["vm_user"] = args.vm_user
        cfg["vm_pass"] = args.vm_pass
        cfg["vm_ssh_port"] = args.vm_ssh_port
        cfg["vm_boot_timeout"] = args.vm_boot_timeout
        report = run_vm_eval(args.device, args.vm_image, args.vm_backend, cfg)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)

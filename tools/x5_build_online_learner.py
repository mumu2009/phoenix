#!/usr/bin/env python3
"""
x5_build_online_learner.py

Cross-compile the Eigen-based C++ online learner on the RDK X5 and run a quick
synthetic test.  This gives you a NumPy-free training path for small task heads.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

X5_HOST = "192.168.0.107"
X5_USER = "root"
X5_PREFIX = "/opt/x5_online_learner"


def ssh(cmd: str, timeout: int = 120) -> int:
    print(f"[x5] {cmd}", flush=True)
    return subprocess.call(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{X5_USER}@{X5_HOST}", cmd],
        timeout=timeout,
    )


def scp_to(local: Path, remote: str) -> None:
    subprocess.check_call(
        ["scp", "-o", "StrictHostKeyChecking=no", str(local), f"{X5_USER}@{X5_HOST}:{remote}"],
        timeout=120,
    )


def main() -> int:
    src = Path(__file__).with_name("x5_online_learner.cpp")
    if not src.exists():
        print(f"[x5] source not found: {src}", file=sys.stderr)
        return 1

    ssh(f"mkdir -p {X5_PREFIX}")
    scp_to(src, f"{X5_PREFIX}/x5_online_learner.cpp")

    rc = ssh(
        f"g++ -std=c++17 -O3 -I/usr/include/eigen3 -o {X5_PREFIX}/x5_online_learner "
        f"{X5_PREFIX}/x5_online_learner.cpp",
        timeout=180,
    )
    if rc != 0:
        return rc

    # quick synthetic test
    rc = ssh(
        f"cd {X5_PREFIX} && ./x5_online_learner synth --out /tmp/x5_test_data.bin --n 1200 --in 64 --out-dim 10",
        timeout=30,
    )
    if rc != 0:
        return rc
    rc = ssh(
        f"cd {X5_PREFIX} && ./x5_online_learner train "
        f"--data /tmp/x5_test_data.bin --n 1000 --in 64 --out 10 --hidden 64 32 "
        f"--epochs 10 --batch-size 32 --lr 0.01 --save /tmp/x5_head.bin",
        timeout=120,
    )
    if rc != 0:
        return rc
    rc = ssh(
        f"cd {X5_PREFIX} && ./x5_online_learner predict "
        f"--data /tmp/x5_test_data.bin --n 200 --in 64 --out 10 --hidden 64 32 "
        f"--load /tmp/x5_head.bin",
        timeout=30,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())

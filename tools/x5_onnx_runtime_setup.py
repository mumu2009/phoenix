#!/usr/bin/env python3
"""
x5_onnx_runtime_setup.py

Sets up a minimal C++ ONNX Runtime build chain on the RDK X5 by reusing the
Python wheel's shared library and fetching the matching C/C++ headers from the
official GitHub release.  Then it builds a tiny C++ runner to verify the chain.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


X5_HOST = "192.168.0.107"
X5_USER = "root"
ORT_VERSION = "1.23.2"
ORT_TARBALL = f"onnxruntime-linux-aarch64-{ORT_VERSION}.tgz"
ORT_URL = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ORT_TARBALL}"


def x5_path(p: Path) -> str:
    """Return a POSIX-style absolute path for X5."""
    return "/" + p.as_posix().lstrip("/")


def ssh(cmd: str, timeout: int = 120, check: bool = True) -> int:
    print(f"[x5] {cmd}", flush=True)
    rc = subprocess.call(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{X5_USER}@{X5_HOST}", cmd],
        timeout=timeout,
    )
    if check and rc != 0:
        raise RuntimeError(f"SSH command failed with {rc}: {cmd}")
    return rc


def ssh_out(cmd: str, timeout: int = 120) -> str:
    return subprocess.check_output(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"{X5_USER}@{X5_HOST}", cmd],
        timeout=timeout,
        text=True,
    ).strip()


def scp_to(local: Path, remote: str) -> None:
    subprocess.check_call(
        ["scp", "-o", "StrictHostKeyChecking=no", str(local), f"{X5_USER}@{X5_HOST}:{remote}"],
        timeout=120,
    )


def find_x5_ort_lib() -> str:
    out = ssh_out(
        "python3 -c 'import onnxruntime; print(onnxruntime.__file__)' 2>/dev/null"
    )
    site = Path(out).parent
    candidates = [
        site / "capi" / f"libonnxruntime.so.{ORT_VERSION}",
        site / "capi" / "libonnxruntime.so",
        site / f"libonnxruntime.so.{ORT_VERSION}",
        site / "libonnxruntime.so",
    ]
    for c in candidates:
        if ssh_out(f"test -f {c} && echo ok || echo missing") == "ok":
            return str(c)
    out = ssh_out(
        "find /usr/local/lib/python3.10/dist-packages -name 'libonnxruntime.so*' | head -1"
    )
    return out


def setup(prefix: Path) -> int:
    inc_dir = prefix / "include"
    lib_dir = prefix / "lib"
    src_dir = prefix / "src"
    for d in (prefix, inc_dir, lib_dir, src_dir):
        ssh(f"mkdir -p {x5_path(d)}")

    tarball = prefix / ORT_TARBALL
    exists = ssh_out(f"test -f {x5_path(tarball)} && echo ok || echo missing")
    if exists != "ok":
        print("[x5] downloading release archive ...", flush=True)
        # Prefer curl, fall back to wget
        rc = ssh(
            f"cd {x5_path(prefix)} && (curl -L -o {ORT_TARBALL} {ORT_URL} || wget -q {ORT_URL} -O {ORT_TARBALL})",
            timeout=300,
            check=False,
        )
        if rc != 0 or ssh_out(f"test -f {x5_path(tarball)} && echo ok || echo missing") != "ok":
            raise RuntimeError(f"could not download {ORT_URL}")

    print("[x5] extracting headers and runtime ...", flush=True)
    top_dir = f"onnxruntime-linux-aarch64-{ORT_VERSION}"
    ssh(
        f"cd {x5_path(prefix)} && rm -rf {x5_path(inc_dir)} {x5_path(lib_dir)} {top_dir} && "
        f"tar -xzf {ORT_TARBALL} && "
        f"mkdir -p {x5_path(inc_dir)} {x5_path(lib_dir)} && "
        f"cp -r {top_dir}/include/* {x5_path(inc_dir)}/ && "
        f"cp {top_dir}/lib/libonnxruntime.so* {x5_path(lib_dir)}/",
        timeout=120,
    )

    # Make sure libonnxruntime.so is available.  Prefer the release .so,
    # but fall back to the Python wheel's .so.
    so_path = find_x5_ort_lib()
    if not so_path:
        raise RuntimeError("could not find libonnxruntime.so on X5")
    so_name = Path(so_path).name
    ssh(
        f"cd {x5_path(lib_dir)} && rm -f libonnxruntime.so* && "
        f"cp {so_path} . && "
        f"ln -s {so_name} libonnxruntime.so.1 && "
        f"ln -s libonnxruntime.so.1 libonnxruntime.so"
    )

    # Copy runner C++
    runner_cpp = src_dir / "x5_onnx_runner.cpp"
    runner_local = Path(__file__).with_name("x5_onnx_runner.cpp")
    if not runner_local.exists():
        runner_local = Path("C:/Users/木木/AppData/Local/Temp/x5_onnx_runner.cpp")
    scp_to(runner_local, x5_path(runner_cpp))

    # Write Makefile
    makefile = src_dir / "Makefile"
    make_local = Path(tempfile.gettempdir()) / "Makefile.x5"
    make_text = f"""CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -I{x5_path(inc_dir)}
LDFLAGS = -L{x5_path(lib_dir)} -Wl,-rpath,{x5_path(lib_dir)} -lonnxruntime

x5_onnx_runner: x5_onnx_runner.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: x5_onnx_runner
	LD_LIBRARY_PATH={x5_path(lib_dir)} ./x5_onnx_runner $(MODEL)
"""
    make_local.write_text(make_text, encoding="utf-8")
    scp_to(make_local, x5_path(makefile))

    print("[x5] compiling runner ...", flush=True)
    ssh(f"cd {x5_path(src_dir)} && make x5_onnx_runner", timeout=180)

    print(f"[x5] built: {x5_path(src_dir)}/x5_onnx_runner")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/opt/x5_onnx_runtime", help="install prefix on X5")
    parser.add_argument("--test-model", default="", help="optional ONNX model to run after build")
    args = parser.parse_args()

    prefix = Path(args.prefix.lstrip("/"))
    try:
        setup(prefix)
        if args.test_model:
            lib_dir = prefix / "lib"
            src_dir = prefix / "src"
            ssh(
                f"cd {x5_path(src_dir)} && LD_LIBRARY_PATH={x5_path(lib_dir)} ./x5_onnx_runner {args.test_model}",
                timeout=60,
            )
        return 0
    except Exception as e:
        print(f"[x5] setup failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

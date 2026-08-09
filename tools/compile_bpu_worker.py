#!/usr/bin/env python3
"""Persistent BPU compilation worker that runs inside the Docker container.

This script imports hb_mapper ONCE at startup, then loops reading compile jobs
from a socket (UNIX domain socket at /tmp/bpu_compile.sock).  Each job is a
JSON object; multiple candidates can be compiled in parallel using the worker
pool.

Protocol (newline-delimited JSON over UNIX socket):
  -> {"action": "compile", "jobs": [<job>, ...], "parallel": 2}
  <- {"status": "ok", "results": [{"job_id": "...", "ok": true, "bin": "..."}, ...]}

  -> {"action": "ping"}
  <- {"status": "ok", "msg": "pong"}

  -> {"action": "shutdown"}
  <- {"status": "ok", "msg": "bye"}

Single-job format:
  {
    "job_id": "candidate_0000",
    "model_name": "speech_encoder",
    "onnx_path": "/workspace/...",
    "calib_dir": "/workspace/...",
    "input_name": "waveform",
    "input_shape": "1x1x1x16000",
    "out_dir": "/workspace/...",
    "per_channel": true,
    "calib_type": "kl",
    "march": "bayes-e"
  }

Usage:
  # Inside the Docker container:
  python3 /workspace/tools/compile_bpu_worker.py

  # Or with a custom socket path:
  python3 /workspace/tools/compile_bpu_worker.py --sock /tmp/my.sock

  # Or in batch mode (single JSON on stdin, no socket):
  echo '{"action":"compile","jobs":[...]}' | python3 /workspace/tools/compile_bpu_worker.py --batch
"""

import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path

# ---------------------------------------------------------------------------
# hb_mapper initialization (done once)
# ---------------------------------------------------------------------------

def _init_hb_mapper():
    """Import hb_mapper (and the patch) once. Returns True on success."""
    try:
        sys.path.insert(0, "/workspace/tools")
        import hb_mapper_patch  # noqa: F401
        from horizon_tc_ui.hb_mapper import main as _hb_main
        return True
    except Exception as e:
        print(f"[worker] WARNING: hb_mapper import failed: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Single compilation job (runs in a subprocess)
# ---------------------------------------------------------------------------

def _compile_one(job: dict) -> dict:
    """Compile a single ONNX model to BPU .bin.

    Uses the compile_bpu_jepa_v2.sh script (which calls run_hb_mapper.py)
    so each compilation runs in its own subprocess with proper hb_mapper
    patching.  The parent process (this worker) stays alive between jobs.
    """
    job_id = job.get("job_id", "unknown")
    result = {"job_id": job_id, "ok": False, "bin": None, "error": None}

    try:
        model_name = job["model_name"]
        onnx_path = job["onnx_path"]
        calib_dir = job["calib_dir"]
        input_name = job["input_name"]
        input_shape = job["input_shape"]
        out_dir = job["out_dir"]
        per_channel = job.get("per_channel", True)
        calib_type = job.get("calib_type", "kl")
        march = job.get("march", "bayes-e")

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "mapper_work"), exist_ok=True)

        # Use the same compile script that the old Docker flow uses.
        # compile_bpu_jepa_v2.sh handles:
        #   - ONNX IR version fix
        #   - YAML config generation
        #   - hb_mapper invocation via run_hb_mapper.py
        #   - .bin output copying
        cmd = [
            "bash", "/workspace/tools/compile_bpu_jepa_v2.sh",
            "--model-name", model_name,
            "--onnx", onnx_path,
            "--calib-dir", calib_dir,
            "--input-name", input_name,
            "--input-shape", input_shape,
            "--out-dir", out_dir,
            "--per-channel", str(per_channel),
            "--calib-type", calib_type,
            "--march", march,
            "--run-hb-mapper", "/workspace/tools/run_hb_mapper.py",
        ]

        log_path = os.path.join(out_dir, "compile.log")
        with open(log_path, "w") as logf:
            proc = subprocess.run(
                cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=900,
            )

        if proc.returncode != 0:
            result["error"] = f"compile script returned {proc.returncode}"
            return result

        # Check for output
        import glob
        final_bin = os.path.join(out_dir, f"{model_name}.bin")
        if os.path.isfile(final_bin):
            result["ok"] = True
            result["bin"] = final_bin
        else:
            result["error"] = "no .bin produced"

    except subprocess.TimeoutExpired:
        result["error"] = "compile timed out (900s)"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _compile_batch(jobs: list, parallel: int = 2) -> list:
    """Compile a batch of jobs using thread-based parallelism.

    Each job spawns a subprocess (bash compile_bpu_jepa_v2.sh) which handles
    its own hb_mapper import.  By running inside a persistent container we
    avoid the ~5s Docker cold-start for each candidate.  The thread pool
    enables multiple hb_mapper processes to run concurrently.
    """
    results = []
    if not jobs:
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed
    parallel = min(parallel, len(jobs))

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        future_to_job = {pool.submit(_compile_one, job): job for job in jobs}
        for future in as_completed(future_to_job):
            try:
                result = future.result(timeout=960)
            except Exception as e:
                job = future_to_job[future]
                result = {
                    "job_id": job.get("job_id", "unknown"),
                    "ok": False,
                    "bin": None,
                    "error": f"thread error: {e}",
                }
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Socket server
# ---------------------------------------------------------------------------

SOCK_PATH = "/tmp/bpu_compile.sock"

def _handle_client(conn):
    """Handle one client connection (newline-delimited JSON)."""
    buf = b""
    try:
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    resp = {"status": "error", "error": f"invalid JSON: {e}"}
                    conn.sendall(json.dumps(resp).encode() + b"\n")
                    continue

                action = msg.get("action", "")
                if action == "ping":
                    resp = {"status": "ok", "msg": "pong"}
                elif action == "shutdown":
                    resp = {"status": "ok", "msg": "bye"}
                    conn.sendall(json.dumps(resp).encode() + b"\n")
                    conn.close()
                    os._exit(0)
                elif action == "compile":
                    jobs = msg.get("jobs", [])
                    parallel = msg.get("parallel", 2)
                    t0 = time.time()
                    results = _compile_batch(jobs, parallel)
                    elapsed = time.time() - t0
                    resp = {
                        "status": "ok",
                        "results": results,
                        "elapsed": round(elapsed, 2),
                    }
                else:
                    resp = {"status": "error", "error": f"unknown action: {action}"}

                conn.sendall(json.dumps(resp).encode() + b"\n")
    except Exception as e:
        print(f"[worker] client error: {e}", file=sys.stderr)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_server(sock_path: str):
    """Run the UNIX domain socket server."""
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(4)
    os.chmod(sock_path, 0o666)
    print(f"[worker] listening on {sock_path}", flush=True)

    def _shutdown(sig, frame):
        print(f"[worker] shutting down (signal {sig})", flush=True)
        server.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        try:
            conn, _ = server.accept()
            t = threading.Thread(target=_handle_client, args=(conn,), daemon=True)
            t.start()
        except OSError:
            break


# ---------------------------------------------------------------------------
# Batch mode (stdin/stdout, no socket)
# ---------------------------------------------------------------------------

def run_batch():
    """Read a single JSON request from stdin, process, write response to stdout."""
    data = sys.stdin.read()
    try:
        msg = json.loads(data)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1

    action = msg.get("action", "compile")
    if action == "compile":
        jobs = msg.get("jobs", [])
        parallel = msg.get("parallel", 2)
        t0 = time.time()
        results = _compile_batch(jobs, parallel)
        elapsed = time.time() - t0
        resp = {"status": "ok", "results": results, "elapsed": round(elapsed, 2)}
    elif action == "ping":
        resp = {"status": "ok", "msg": "pong"}
    else:
        resp = {"status": "error", "error": f"unknown action: {action}"}

    print(json.dumps(resp), flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Persistent BPU compilation worker")
    parser.add_argument("--sock", default=SOCK_PATH, help="UNIX socket path")
    parser.add_argument("--batch", action="store_true", help="Batch mode: read JSON from stdin")
    args = parser.parse_args()

    if args.batch:
        sys.exit(run_batch())

    # Pre-import hb_mapper in the main process (child processes inherit the import)
    print("[worker] pre-importing hb_mapper ...", flush=True)
    t0 = time.time()
    ok = _init_hb_mapper()
    print(f"[worker] hb_mapper ready in {time.time()-t0:.1f}s (ok={ok})", flush=True)

    run_server(args.sock)


if __name__ == "__main__":
    main()

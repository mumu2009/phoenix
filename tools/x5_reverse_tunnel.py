#!/usr/bin/env python3
"""Reverse SSH tunnel: Windows -> Kali, forwarding Kali:<port> -> X5:22.

Run this on the Windows host when the Kali VM cannot route to the RDK X5
(192.168.0.107) directly.  After it starts, on Kali use:

    ssh -p 2222 root@127.0.0.1

or set `bpu_evolve_additive.py` / `run_all_additive_training.py` to:

    --x5-host 127.0.0.1 --x5-port 2222 --x5-user root --x5-pass root

The default port is 2222.  If the port is already in use on Kali, the script
will try the next few ports (2223, 2224, ...).  The corresponding
`edge_devices.json` entry must match the chosen port.
"""
import argparse
import atexit
import os
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import paramiko

LOCK_PATH = Path(__file__).resolve().parent / ".x5_tunnel.lock"

DEFAULT_KALI_HOST = "192.168.0.100"
DEFAULT_KALI_USER = "kali"
DEFAULT_KALI_PASS = "kali"
DEFAULT_X5_HOST = "192.168.0.107"
DEFAULT_X5_PORT = 22
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 2222


def handler(chan, host, port):
    sock = socket.socket()
    try:
        sock.connect((host, port))
    except Exception as e:
        print(f"! tunnel connect to {host}:{port} failed: {e}")
        chan.close()
        return

    while True:
        r, w, x = select.select([sock, chan], [], [])
        if sock in r:
            data = sock.recv(1024)
            if len(data) == 0:
                break
            chan.send(data)
        if chan in r:
            data = chan.recv(1024)
            if len(data) == 0:
                break
            sock.send(data)

    chan.close()
    sock.close()


def reverse_port_forward_tunnel(transport, bind_host, bind_port, x5_host, x5_port):
    transport.request_port_forward(bind_host, bind_port)
    print(f"[tunnel] Kali {bind_host}:{bind_port} -> {x5_host}:{x5_port}", flush=True)
    while True:
        chan = transport.accept(1000)
        if chan is None:
            continue
        thr = threading.Thread(target=handler, args=(chan, x5_host, x5_port), daemon=True)
        thr.start()


def _try_bind(client, bind_host, bind_port, x5_host, x5_port, max_attempts=8):
    for attempt in range(max_attempts):
        port = bind_port + attempt
        try:
            reverse_port_forward_tunnel(client.get_transport(), bind_host, port, x5_host, x5_port)
            return port
        except paramiko.ssh_exception.SSHException as e:
            if "TCP forwarding request denied" in str(e) or "forwarding request denied" in str(e):
                print(f"[tunnel] port {port} denied on Kali (already in use or TCP forwarding disabled).", flush=True)
                if attempt < max_attempts - 1:
                    print(f"[tunnel] retrying with port {port + 1} ...", flush=True)
                    time.sleep(0.2)
                    continue
            raise
    raise RuntimeError(f"could not establish remote forward after {max_attempts} ports")


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for a Windows PID (no extra dependencies)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return False
    return str(pid) in out


def _kill_pid(pid: int) -> None:
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True, timeout=10)
    except Exception as exc:
        print(f"[tunnel] failed to kill stale tunnel PID {pid}: {exc}", file=sys.stderr, flush=True)


def _acquire_single_instance_lock() -> None:
    """Kill any previous ``x5_reverse_tunnel.py`` instance and take over.

    This is launched both by a background orchestration script and,
    independently, by hand in a foreground terminal whenever the user wants
    to make sure the tunnel is healthy.  There is no way for the user to
    know about or reach a background instance to stop it first, and running
    two tunnels at once previously left stale/duplicate reverse-forward
    registrations on Kali (picking a different fallback port, silently
    routing new connections through a half-dead tunnel).  Since the tunnel
    is stateless, it is always safe to kill whichever instance was here
    first and take over.
    """
    if LOCK_PATH.is_file():
        try:
            old_pid = int(LOCK_PATH.read_text().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None and old_pid != os.getpid() and _pid_alive(old_pid):
            print(f"[tunnel] killing previous tunnel instance (PID {old_pid}) and taking over", flush=True)
            _kill_pid(old_pid)
            for _ in range(20):  # up to ~10s for it to actually exit
                if not _pid_alive(old_pid):
                    break
                time.sleep(0.5)
    LOCK_PATH.write_text(str(os.getpid()))
    atexit.register(lambda: LOCK_PATH.unlink(missing_ok=True))


def main():
    _acquire_single_instance_lock()
    parser = argparse.ArgumentParser()
    parser.add_argument("--kali-host", default=DEFAULT_KALI_HOST)
    parser.add_argument("--kali-user", default=DEFAULT_KALI_USER)
    parser.add_argument("--kali-pass", default=DEFAULT_KALI_PASS)
    parser.add_argument("--x5-host", default=DEFAULT_X5_HOST)
    parser.add_argument("--x5-port", type=int, default=DEFAULT_X5_PORT)
    parser.add_argument("--bind-host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--bind-port", type=int, default=DEFAULT_BIND_PORT)
    args = parser.parse_args()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(args.kali_host, username=args.kali_user, password=args.kali_pass, timeout=30)
        client.get_transport().set_keepalive(30)
    except Exception as e:
        print(f"[tunnel] failed to connect to Kali: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print(f"[tunnel] connected to Kali {args.kali_host}", flush=True)
    try:
        used_port = _try_bind(client, args.bind_host, args.bind_port, args.x5_host, args.x5_port)
        print(f"[tunnel] active. On Kali use: ssh -p {used_port} root@{args.bind_host}", flush=True)
        print(f"[tunnel] press Ctrl+C to stop", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[tunnel] closing", flush=True)
    except Exception as e:
        print(f"[tunnel] error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

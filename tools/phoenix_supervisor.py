#!/usr/bin/env python3
"""Phoenix 运行时监督器：分进程探活、断点落盘、只拉起死掉的服务。

不 curl llama /health（--parallel 1 会 cancel 在途生成）。
探活只用本机 pid / 127.0.0.1:8082 是否在听，不 LAN 打 8082。
重启不 POST /api/mission/assign，也不恢复脏 mission。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import phoenix_process_guard as pguard
    import phoenix_runtime_amp as amp
    import phoenix_runtime_opt as ropt
    import phoenix_shm_channel as shmchan
    import phoenix_speculative_bridge as specbridge
except ImportError:
    pguard = None  # type: ignore[misc, assignment]
    amp = None  # type: ignore[misc, assignment]
    ropt = None  # type: ignore[misc, assignment]
    shmchan = None  # type: ignore[misc, assignment]
    specbridge = None  # type: ignore[misc, assignment]

BUILTIN_DENY = ("2678077", "6477259", "9374278")
PRODUCT = "phoenix"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_mission_id(raw: str) -> str:
    return "".join(ch for ch in str(raw) if ch.isdigit())


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def supervisor_cfg(phoenix_json: Path) -> dict[str, Any]:
    root = load_json(phoenix_json)
    main = root.get("main") if isinstance(root.get("main"), dict) else {}
    sup = main.get("supervisor") if isinstance(main.get("supervisor"), dict) else {}
    services = main.get("services") if isinstance(main.get("services"), dict) else {}
    checkpoint = main.get("checkpoint") if isinstance(main.get("checkpoint"), dict) else {}
    return {"main": main, "supervisor": sup, "services": services, "checkpoint": checkpoint}


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, 0, pid)
            if not handle:
                return False
            wait = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)
            ctypes.windll.kernel32.CloseHandle(handle)
            return wait == 0x00000102  # WAIT_TIMEOUT => still running
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def port_listening(host: str, port: int, timeout: float = 0.4) -> bool:
    if port <= 0:
        return False
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def read_pid_file(path: Path) -> int:
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not text:
        return 0
    digits = "".join(ch for ch in text[0] if ch.isdigit())
    return int(digits) if digits else 0


def refuse_llama_health(role: str, url: str) -> bool:
    return role in ("llama", "inference") and "/health" in (url or "").lower()


def is_loopback_host(host: str) -> bool:
    return (host or "").strip().lower() in ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def llama_probe_host(host: str) -> str:
    """Never probe llama :8082 over LAN (X5 binds 127.0.0.1 only)."""
    h = (host or "").strip().lower()
    if h == "::1":
        return "::1"
    return "127.0.0.1"


def rss_mb(pid: int) -> int:
    if pid <= 0:
        return 0
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(0x1000, 0, pid)
            if not handle:
                return 0
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            try:
                get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            except AttributeError:
                get_info = k32.K32GetProcessMemoryInfo
            ok = get_info(handle, ctypes.byref(counters), counters.cb)
            k32.CloseHandle(handle)
            if not ok:
                return 0
            return int(counters.WorkingSetSize) // (1024 * 1024)
        except OSError:
            return 0
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return 0
    for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) // 1024
    return 0


def merge_deny(cfg_list: Any, file_list: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for src in (BUILTIN_DENY, cfg_list or [], file_list or []):
        for item in src:
            nid = normalize_mission_id(str(item))
            if nid and nid not in seen:
                seen.add(nid)
                out.append(nid)
    return out


def can_restore_mission(mission_id: str, deny: list[str], resume_flag: bool) -> bool:
    nid = normalize_mission_id(mission_id)
    if not nid or not resume_flag:
        return False
    return nid not in {normalize_mission_id(x) for x in deny}


def default_services(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    services = cfg["services"]
    gw = services.get("gateway") if isinstance(services.get("gateway"), dict) else {}
    fe = services.get("frontend") if isinstance(services.get("frontend"), dict) else {}
    llama = services.get("llama") if isinstance(services.get("llama"), dict) else {}
    return [
        {
            "role": "gateway",
            "port": int(gw.get("port") or 5080),
            "host": str(gw.get("host") or "127.0.0.1"),
            "maxRssMb": int(gw.get("maxRssMb") or 4096),
            "cmd_env": "PHOENIX_GATEWAY_CMD",
        },
        {
            "role": "frontend",
            "port": int(fe.get("port") or 5081),
            "host": str(fe.get("host") or "127.0.0.1"),
            "maxRssMb": int(fe.get("maxRssMb") or 0),
            "cmd_env": "PHOENIX_FRONTEND_CMD",
        },
        {
            "role": "llama",
            "port": int(llama.get("port") or 8082),
            "host": llama_probe_host(str(llama.get("host") or "127.0.0.1")),
            "maxRssMb": 0,
            "cmd_env": "PHOENIX_LLAMA_CMD",
            "restartIfDead": bool(llama.get("restartIfDead", False)),
        },
    ]


def observe(spec: dict[str, Any], pid_dir: Path) -> dict[str, Any]:
    role = spec["role"]
    pid = read_pid_file(pid_dir / f"{role}.pid")
    alive_pid = pid_running(pid)
    probe_host = spec["host"]
    if role in ("llama", "inference"):
        probe_host = llama_probe_host(str(spec.get("host") or "127.0.0.1"))
    alive_port = port_listening(probe_host, int(spec["port"]))
    alive = alive_pid or alive_port
    rss = rss_mb(pid) if alive_pid else 0
    max_rss = int(spec.get("maxRssMb") or 0)
    # RSS fuse is gateway-only; llama is never a fuse target.
    over_mem = bool(role == "gateway" and max_rss > 0 and rss >= max_rss)
    if refuse_llama_health(role, str(spec.get("probeUrl") or "")):
        raise RuntimeError("never curl llama /health (--parallel 1 cancels generation)")
    return {
        "role": role,
        "pid": pid,
        "port": int(spec["port"]),
        "alive": alive,
        "rssMb": rss,
        "overMem": over_mem,
        "lastCleanCheckpoint": utc_now() if (alive and not over_mem) else "",
        "probe": "pid_or_listen",
    }


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def stop_service(role: str, pid: int) -> None:
    if role == "llama" or role == "inference" or pid <= 0:
        return
    if pguard is not None:
        pguard.cleanup_tree(role, pid)
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T"],
            capture_output=True,
            check=False,
        )
        return
    try:
        os.killpg(pid, 15)
    except OSError:
        try:
            os.kill(pid, 15)
        except OSError:
            return


def start_service(
    role: str,
    cmd: str,
    cwd: Path,
    extra_env: dict[str, str],
    overlay: dict[str, Any] | None = None,
) -> int:
    if role in ("llama-draft", "draft"):
        print("[supervisor] refuse second llama/draft server (slot steal)", flush=True)
        return 0
    if not cmd.strip():
        print(f"[supervisor] no start command for {role}", flush=True)
        return 0
    env = os.environ.copy()
    env.update(extra_env)
    env["AI_SUPERVISOR_MANAGED"] = "1"
    if os.name == "nt":
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, shell=True)
    else:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, shell=True, start_new_session=True)
    print(f"[supervisor] started {role} pid={proc.pid}", flush=True)
    if overlay and pguard is not None:
        attached = pguard.attach_child(role, int(proc.pid), overlay)
        if attached.get("degraded"):
            print(f"[supervisor] process_guard degrade role={role}", flush=True)
    return int(proc.pid)


def plan_and_act(
    observed: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    cwd: Path,
    dry_run: bool,
    overlay: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    spec_by_role = {s["role"]: s for s in specs}
    decisions: list[dict[str, str]] = []
    for svc in observed:
        role = svc["role"]
        spec = spec_by_role[role]
        if role in ("llama-draft", "draft") or spec.get("secondLlamaServer"):
            decisions.append(
                {"role": role, "action": "leave", "reason": "parallel_1_forbids_second_slot_server"}
            )
            continue
        if svc.get("overMem") and role == "gateway":
            decisions.append({"role": role, "action": "restart", "reason": "rss_fuse"})
            if not dry_run:
                stop_service(role, int(svc.get("pid") or 0))
                cmd = os.environ.get(str(spec["cmd_env"]), "")
                start_service(role, cmd, cwd, {}, overlay)
            continue
        if svc["alive"]:
            decisions.append({"role": role, "action": "leave", "reason": "alive"})
            continue
        if role == "llama" and not spec.get("restartIfDead"):
            decisions.append({"role": role, "action": "leave", "reason": "llama_dead_no_autostart"})
            continue
        decisions.append({"role": role, "action": "start", "reason": "dead"})
        if dry_run:
            continue
        cmd = os.environ.get(str(spec["cmd_env"]), "")
        pid = start_service(role, cmd, cwd, {}, overlay)
        if pid:
            (cwd / "runtime_store" / "pids").mkdir(parents=True, exist_ok=True)
    return decisions


def refuse_dirty_resume(last_mission: str, deny: list[str]) -> None:
    nid = normalize_mission_id(last_mission)
    if nid and nid in set(deny):
        print(
            f"[supervisor] refuse restore dirty mission {nid}; "
            "will not assign or resume",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix process supervisor")
    parser.add_argument("--phoenix-json", default="", help="config/phoenix.json 路径")
    parser.add_argument("--once", action="store_true", help="只巡检一轮")
    parser.add_argument("--dry-run", action="store_true", help="只规划不拉起")
    parser.add_argument("--interval-ms", type=int, default=0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    phoenix_json = Path(args.phoenix_json) if args.phoenix_json else repo / "config" / "phoenix.json"
    cfg = supervisor_cfg(phoenix_json)
    sup = cfg["supervisor"]
    ck_cfg = cfg["checkpoint"]
    pid_dir = Path(sup.get("pidDir") or "runtime_store/pids")
    if not pid_dir.is_absolute():
        pid_dir = repo / pid_dir
    ck_path = Path(ck_cfg.get("path") or sup.get("checkpointPath") or "runtime_store/supervisor_checkpoint.json")
    if not ck_path.is_absolute():
        ck_path = repo / ck_path
    interval_ms = args.interval_ms or int(sup.get("pollIntervalMs") or 5000)
    specs = default_services(cfg)
    overlay = ropt.load_overlay(load_json(phoenix_json)) if ropt is not None else None
    if os.environ.get("PHOENIX_LLAMA_DRAFT_CMD"):
        print(
            "[supervisor] refuse PHOENIX_LLAMA_DRAFT_CMD; draft must share the live "
            "llama-server (--parallel 1)",
            flush=True,
        )
    if overlay and specbridge is not None:
        draft_plan = specbridge.plan_speculative(
            overlay,
            llama_already_running=True,
            want_standalone_draft_server=bool(os.environ.get("PHOENIX_LLAMA_DRAFT_CMD")),
        )
        if draft_plan.get("refuse"):
            print(f"[supervisor] speculative {draft_plan['reason']}", flush=True)
    shm = None
    if overlay and overlay.get("flags", {}).get("shm_ipc") and shmchan is not None:
        shm = shmchan.ShmChannel()
        shm_cfg = overlay["shm_ipc"]
        if not shm.open(
            shm_cfg["name"],
            int(shm_cfg["bytes"]),
            repo / "runtime_store" / "shm",
            fallback=str(shm_cfg.get("fallback") or "file"),
        ):
            print("[supervisor] shm_ipc degrade to none", flush=True)
            shm = None
        else:
            print(f"[supervisor] shm_ipc backend={shm.backend}", flush=True)
    prev = load_json(ck_path)
    deny = merge_deny(sup.get("denyResumeMissions"), prev.get("denyResumeMissions"))
    last_mission = str(prev.get("lastMissionId") or "")
    refuse_dirty_resume(last_mission, deny)
    if last_mission and not can_restore_mission(last_mission, deny, False):
        last_mission = last_mission  # keep id for audit, never resume

    print(
        "[supervisor] product=phoenix llama_probe=pid_or_listen "
        "never_curl_llama_health=1 never_assign_mission=1 "
        f"runtime_opt={overlay['flags'] if overlay else {}}",
        flush=True,
    )

    while True:
        observed = [observe(spec, pid_dir) for spec in specs]
        decisions = plan_and_act(observed, specs, repo, args.dry_run, overlay)
        press = amp.host_pressure() if amp is not None else {"diskFreePct": 100, "swapUsedPct": 0}
        survive = (
            amp.plan_survive(press, overlay or {"flags": {}}, 1)
            if amp is not None
            else {"reason": "amp_missing", "leaveLlama": True, "maxInFlight": 1}
        )
        alerts_path = repo / "runtime_store" / "security_alerts.json"
        alert_agg = {"count": 0, "reason": "no_sidecar"}
        if overlay and overlay.get("flags", {}).get("security_agg") and alerts_path.is_file():
            try:
                raw_alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
                items = raw_alerts if isinstance(raw_alerts, list) else raw_alerts.get("alerts") or []
                if amp is not None:
                    alert_agg = amp.aggregate_alerts(list(items))
            except (OSError, json.JSONDecodeError):
                alert_agg = {"count": 0, "reason": "alerts_unreadable"}
        if shm is not None:
            if amp is not None:
                shm.write(amp.encode_frame("heartbeat", "supervisor", shmchan.heartbeat_payload(observed)))
            else:
                shm.write(shmchan.heartbeat_payload(observed))
        payload = {
            "version": 1,
            "product": PRODUCT,
            "writtenAt": utc_now(),
            "services": observed,
            "decisions": decisions,
            "denyResumeMissions": deny,
            "lastMissionId": last_mission,
            "resumeMissions": False,
            "llamaHealth": "do_not_curl_/health",
            "hostPressure": press,
            "survivability": survive,
            "securityAgg": alert_agg,
        }
        write_checkpoint(ck_path, payload)
        for d in decisions:
            print(f"[supervisor] {d['role']} -> {d['action']} ({d['reason']})", flush=True)
        if args.once:
            if shm is not None:
                shm.close(unlink=False)
            return 0
        time.sleep(max(0.2, interval_ms / 1000.0))


if __name__ == "__main__":
    sys.exit(main())

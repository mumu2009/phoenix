#!/usr/bin/env python3
"""Host unit checks for phoenix_supervisor policy (no RDK, no llama /health)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phoenix_supervisor as s


def test_checkpoint_roundtrip() -> None:
    payload = {
        "version": 1,
        "product": "phoenix",
        "writtenAt": "2026-09-11T21:00:00+00:00",
        "services": [
            {"role": "gateway", "pid": 10, "port": 5080, "alive": False},
            {"role": "llama", "pid": 22, "port": 8082, "alive": True},
        ],
        "denyResumeMissions": list(s.BUILTIN_DENY),
        "lastMissionId": "1111111",
        "resumeMissions": False,
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ck.json"
        s.write_checkpoint(p, payload)
        back = s.load_json(p)
    assert back["product"] == "phoenix"
    assert back["services"][1]["role"] == "llama"
    assert back["services"][1]["pid"] == 22


def test_dirty_mission_refuse() -> None:
    deny = s.merge_deny(["2678077"], [])
    assert "6477259" in deny and "9374278" in deny
    assert not s.can_restore_mission("2678077", deny, True)
    assert not s.can_restore_mission("mission-6477259", deny, True)
    assert s.can_restore_mission("5550001", deny, True)
    assert not s.can_restore_mission("5550001", deny, False)


def test_restart_gateway_leaves_llama() -> None:
    specs = [
        {"role": "gateway", "cmd_env": "PHOENIX_GATEWAY_CMD", "restartIfDead": True},
        {"role": "llama", "cmd_env": "PHOENIX_LLAMA_CMD", "restartIfDead": False},
    ]
    observed = [
        {"role": "gateway", "alive": False},
        {"role": "llama", "alive": True},
    ]
    decisions = s.plan_and_act(observed, specs, Path("."), dry_run=True)
    by = {d["role"]: d for d in decisions}
    assert by["gateway"]["action"] == "start"
    assert by["llama"]["action"] == "leave"
    assert by["llama"]["reason"] == "alive"


def test_refuse_llama_health_and_second_server() -> None:
    assert s.refuse_llama_health("llama", "http://127.0.0.1:8082/health")
    assert s.llama_probe_host("192.168.1.107") == "127.0.0.1"
    assert s.llama_probe_host("10.0.0.8") == "127.0.0.1"
    specs = [
        {"role": "llama-draft", "cmd_env": "PHOENIX_LLAMA_DRAFT_CMD", "secondLlamaServer": True},
        {"role": "llama", "cmd_env": "PHOENIX_LLAMA_CMD", "restartIfDead": False},
    ]
    observed = [
        {"role": "llama-draft", "alive": False},
        {"role": "llama", "alive": True},
    ]
    decisions = s.plan_and_act(observed, specs, Path("."), dry_run=True)
    by = {d["role"]: d for d in decisions}
    assert by["llama-draft"]["action"] == "leave"
    assert by["llama"]["action"] == "leave"


def test_rss_fuse_restarts_gateway_not_llama() -> None:
    specs = [
        {"role": "gateway", "cmd_env": "PHOENIX_GATEWAY_CMD", "restartIfDead": True},
        {"role": "llama", "cmd_env": "PHOENIX_LLAMA_CMD", "restartIfDead": False},
    ]
    observed = [
        {"role": "gateway", "alive": True, "overMem": True, "pid": 11},
        {"role": "llama", "alive": True, "overMem": True, "pid": 22},
    ]
    decisions = s.plan_and_act(observed, specs, Path("."), dry_run=True)
    by = {d["role"]: d for d in decisions}
    assert by["gateway"]["action"] == "restart"
    assert by["gateway"]["reason"] == "rss_fuse"
    assert by["llama"]["action"] == "leave"
    assert by["llama"]["reason"] == "alive"


def main() -> int:
    test_checkpoint_roundtrip()
    test_dirty_mission_refuse()
    test_restart_gateway_leaves_llama()
    test_rss_fuse_restarts_gateway_not_llama()
    test_refuse_llama_health_and_second_server()
    print("supervisor policy tests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

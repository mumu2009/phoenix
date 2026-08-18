from __future__ import annotations

import json as _json
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from addon import AddonResult

# Parity with the C++ cli_json_addon: whitelist gate + direct exec (no shell).
# Whitelist from env PHOENIX_CLI_TOOLS_JSON: [{"name","command","args","timeoutMs","json"}].


def _registry() -> Dict[str, Dict[str, Any]]:
    raw = os.environ.get("PHOENIX_CLI_TOOLS_JSON", "[]")
    try:
        items = _json.loads(raw)
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("name") and item.get("command"):
                out[item["name"]] = item
    return out


def run_cli_json_command(tool: str, args: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
    reg = _registry()
    tpl = reg.get(tool)
    if not tpl:
        return {"ok": False, "error": f"cli tool not whitelisted: {tool}"}
    cmd = [tpl["command"]] + list(tpl.get("args", [])) + list(args)
    timeout = int(options.get("timeoutMs", tpl.get("timeoutMs", 5000)))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout / 1000.0, shell=False
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "tool": tool, "timedOut": True,
                "error": f"command timed out after {timeout}ms"}
    except FileNotFoundError as e:
        return {"ok": False, "tool": tool, "error": str(e)}
    out: Dict[str, Any] = {
        "tool": tool, "exitCode": proc.returncode, "timedOut": False,
        "ok": proc.returncode == 0,
    }
    reply = proc.stdout or ""
    if tpl.get("json", tpl.get("jsonOutput", True)):
        try:
            out["json"] = _json.loads(proc.stdout)
            reply = proc.stdout.strip()
        except Exception:
            out["text"] = proc.stdout
    else:
        out["text"] = proc.stdout
    if len(reply) > int(tpl.get("maxReply", 2000)):
        reply = reply[: int(tpl.get("maxReply", 2000))]
    out["reply"] = reply
    if proc.stderr:
        out["stderr"] = proc.stderr[:400]
    return out


class CliJsonAddon:
    def __init__(self, name: str) -> None:
        self._name = name or "cli-json"

    def name(self) -> str:
        return self._name

    def type(self) -> str:
        return "cli-json"

    def handle(self, text: str, payload: Dict[str, Any]) -> AddonResult:
        addon_type = str(payload.get("__addonType", "")).strip().lower()
        if addon_type not in {"", "cli-json", "cli", "clijson"}:
            return AddonResult()
        tool = str(payload.get("__cliTool", ""))
        rest = (text or "").strip()
        if not tool:
            parts = shlex.split(rest)
            if not parts:
                return AddonResult()
            tool = parts[0]
            rest = " ".join(parts[1:])
        if not tool:
            return AddonResult()
        out = run_cli_json_command(tool, shlex.split(rest), payload)
        res = AddonResult(handled=True, reply=out.get("reply", ""),
                          meta={"addon": "cli-json", "name": self._name, "result": out})
        if not out.get("ok"):
            res.reply = "[cli-json error] " + out.get("error", "command failed")
        return res


def createCliJsonAddon(name: str) -> CliJsonAddon:
    return CliJsonAddon(name or "cli-json")

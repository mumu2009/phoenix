from __future__ import annotations

from typing import List, Optional

from addons.cli_json_addon import createCliJsonAddon
from addons.math_addon import createMathAddon
from addons.search_addon import createSearchAddon


def createBuiltinAddon(typ: str, name: str):
    t = (typ or "").strip().lower()
    if t == "math":
        return createMathAddon(name or "math")
    if t == "search":
        return createSearchAddon(name or "search")
    if t in {"cli-json", "cli", "clijson"}:
        return createCliJsonAddon(name or "cli-json")
    return None


def createDefaultBuiltinAddons() -> List:
    return [createMathAddon("math"), createSearchAddon("search"), createCliJsonAddon("cli-json")]

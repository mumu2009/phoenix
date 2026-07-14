from __future__ import annotations

from typing import List, Optional

from addons.math_addon import createMathAddon
from addons.search_addon import createSearchAddon


def createBuiltinAddon(typ: str, name: str):
    t = (typ or "").strip().lower()
    if t == "math":
        return createMathAddon(name or "math")
    if t == "search":
        return createSearchAddon(name or "search")
    return None


def createDefaultBuiltinAddons() -> List:
    return [createMathAddon("math"), createSearchAddon("search")]

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from addon import AddonResult


class MathAddon:
    def __init__(self, name: str) -> None:
        self._name = name or "math"

    def name(self) -> str:
        return self._name

    def type(self) -> str:
        return "math"

    def handle(self, text: str, payload: Dict[str, Any]) -> AddonResult:
        addon_type = str(payload.get("__addonType", "")).strip().lower()
        if addon_type not in {"", "math", "calculator", "calc"}:
            return AddonResult()
        expr = text.strip()
        if not expr:
            return AddonResult()
        expr = re.sub(r"^(math|calc|计算)\s*[:：]?\s*", "", expr, flags=re.IGNORECASE)
        if not expr:
            return AddonResult()
        try:
            value = _safe_eval(expr)
        except Exception:
            return AddonResult()
        return AddonResult(
            handled=True,
            reply=f"{value:.10g}",
            meta={"addon": "math", "name": self._name, "expr": expr},
        )


def createMathAddon(name: str) -> MathAddon:
    return MathAddon(name or "math")


def _safe_eval(expr: str) -> float:
    allowed = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "tanh": math.tanh,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }
    code = compile(expr, "<math-addon>", "eval")
    for name in code.co_names:
        if name not in allowed:
            raise ValueError("unsafe expression")
    value = eval(code, {"__builtins__": {}}, allowed)
    if not isinstance(value, (int, float)):
        raise ValueError("non numeric")
    if not math.isfinite(float(value)):
        raise ValueError("invalid numeric")
    return float(value)

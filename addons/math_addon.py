from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional

from addon import AddonResult

# Parity with the C++ math_addon (exact bignum/rational mode + float mode).
# CPython ints are already arbitrary precision; Fraction gives exact rationals.


def _strip_prefix(expr: str) -> str:
    s = expr.strip()
    low = s.lower()
    for p in ("math:", "calc:", "math=", "calc=", "计算:", "计算"):
        if low.startswith(p):
            return s[len(p):].strip()
    return s


def evaluate_math_expression(expr: str) -> Dict[str, Any]:
    """Exact+float evaluator mirroring the C++ one.

    Exact when the expression only uses integers and exact operators/functions;
    floats otherwise.  Safe: whitelisted AST, no builtins, no attribute access.
    """
    import ast

    src = _strip_prefix(expr)
    if not src:
        raise ValueError("empty expression")

    ALLOWED_NAMES = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
        "exp": math.exp, "ln": math.log, "log": math.log, "log2": math.log2,
        "log10": math.log10, "sqrt": math.sqrt, "cbrt": math.cbrt if hasattr(math, "cbrt") else (lambda x: x ** (1/3)),
        "abs": abs, "floor": math.floor, "ceil": math.ceil, "round": round,
        "trunc": math.trunc, "sign": lambda x: (x > 0) - (x < 0),
        "expm1": math.expm1, "log1p": math.log1p, "erf": math.erf,
        "gamma": math.gamma, "pow": pow, "atan2": math.atan2,
        "hypot": math.hypot, "mod": lambda a, b: a % b, "fmod": math.fmod,
        "gcd": math.gcd, "lcm": math.lcm, "min": min, "max": max,
        "sum": sum,
        "pi": math.pi, "e": math.e, "tau": math.tau,
        "phi": (1 + 5 ** 0.5) / 2,
        "frac": Fraction, "Fraction": Fraction,
    }

    tree = ast.parse(src, mode="eval")
    names: List[str] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            visit(child)
        for field in ("id",):
            pass

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            pass

    for n in names:
        if n not in ALLOWED_NAMES:
            raise ValueError(f"unknown name '{n}'")

    # Fraction-aware evaluation: convert integer constants so exact arithmetic
    # is preserved where possible.
    class ExactTransformer(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.Constant:
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return ast.copy_location(ast.Constant(Fraction(node.value)), node)
            return node

    tree = ExactTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<math-addon>", "eval")
    value = eval(code, {"__builtins__": {}}, dict(ALLOWED_NAMES))

    if isinstance(value, Fraction):
        if value.denominator == 1:
            return {"ok": True, "value": str(value.numerator), "exact": True, "mode": "exact"}
        # try terminating decimal, else fraction form
        d = value.denominator
        while d % 2 == 0:
            d //= 2
        while d % 5 == 0:
            d //= 5
        if d == 1:
            return {"ok": True, "value": format(float(value), ".15g"), "exact": True, "mode": "exact"}
        return {"ok": True, "value": f"{value.numerator}/{value.denominator}", "exact": True, "mode": "exact"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, int):
            return {"ok": True, "value": str(value), "exact": True, "mode": "exact"}
        if not math.isfinite(value):
            raise ValueError("non-finite result")
        return {"ok": True, "value": format(value, ".15g"), "exact": False, "mode": "float"}
    raise ValueError("non-numeric result")


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
        expr = _strip_prefix(text)
        if not expr:
            return AddonResult()
        try:
            out = evaluate_math_expression(expr)
        except Exception as e:
            return AddonResult(handled=True, reply=f"[math error] {e}",
                               meta={"addon": "math", "name": self._name, "expr": expr})
        return AddonResult(handled=True, reply=out["value"],
                           meta={"addon": "math", "name": self._name, "result": out})


def createMathAddon(name: str) -> MathAddon:
    return MathAddon(name or "math")

#!/usr/bin/env python3
"""Audit phoenix.json defaults against the C++ source.

Scans the C++ source for resolveConfig / cfgOr / resolveConfigAsString dot paths,
plus the dot-path map in model_deployment.cpp, and reports:
- code dot paths with no default in phoenix.json
- phoenix.json leaf keys not referenced from code
- total counts
"""

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def flatten_config(data, prefix=""):
    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                if v:
                    out.update(flatten_config(v, new_p))
                else:
                    out[new_p] = v
            else:
                out[new_p] = v
    return out


def all_prefixes(keys):
    """Return every dot-path prefix present in the given leaf keys so that
    code can query an intermediate object (e.g. cfgOr<json>("v7.ahead"))."""
    out = set(keys)
    for key in keys:
        parts = key.split(".")
        for i in range(1, len(parts)):
            out.add(".".join(parts[:i]))
    return out


def dot_paths_from_code():
    pattern = re.compile(
        r'(?:resolveConfig|cfgOr|resolveConfigAsString)(?:<[^>]+>)?\s*\(\s*"([^"]+)"'
    )
    paths = set()
    source_roots = [ROOT, ROOT / "main_hub_parts", ROOT / "module_overrides", ROOT / "addons", ROOT / "tools"]
    for r in source_roots:
        if not r.is_dir():
            continue
        for dirpath, _, files in os.walk(r):
            for name in files:
                if Path(name).suffix not in {".cpp", ".hpp", ".inc", ".h"}:
                    continue
                p = Path(dirpath) / name
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except (PermissionError, OSError):
                    continue
                if p.name == "model_deployment.cpp":
                    for m in re.finditer(r'return "([^"]+)";', text):
                        s = m.group(1)
                        if s.startswith("model_deployment."):
                            paths.add(s)
                for m in pattern.finditer(text):
                    paths.add(m.group(1))
    return paths


def main():
    with (ROOT / "config/phoenix.json").open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg_leaves = flatten_config(cfg)
    cfg_prefixes = all_prefixes(cfg_leaves.keys())
    code_keys = dot_paths_from_code()

    missing = sorted(code_keys - cfg_prefixes)

    def covered_by_code(leaf):
        for code_key in code_keys:
            if leaf == code_key or leaf.startswith(code_key + "."):
                return True
        return False

    unused = sorted([k for k in cfg_leaves if not covered_by_code(k)])

    print(f"code dot paths:        {len(code_keys)}")
    print(f"config leaf keys:      {len(cfg_leaves)}")
    print(f"missing in config:     {len(missing)}")
    print(f"unused in config:      {len(unused)}")

    if missing:
        print("\nMissing defaults (add to phoenix.json):")
        for k in missing:
            print(f"  {k}")
    if unused:
        print("\nUnused config keys (no C++ reference):")
        for k in unused:
            print(f"  {k}")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

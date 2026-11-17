"""Q1 demand probe for the symbolic-reasoning decision gate.

doc/v10.1/reasoning_research_plan.md requires evidence BEFORE the symbolic
layer is enabled.  Q1 is: do missions actually ask questions that need
deterministic reasoning?

This tool is deliberately simple and auditable:

  * it reads every mission deliverable under runtime_store/mission_workspace,
  * classifies the mission by transparent keyword sets (a PROXY, not an
    oracle - the report says so),
  * counts deliberator ticks from the long-run logs for the sample-size leg
    of the gate,
  * prints the raw counts and the gate verdict.

Gate (from the plan): >= 200 ticks AND >= 30 missions AND >= 20% of missions
with at least one class hit.  Anything less => "no demand", and the symbolic
layer stays opt-in.

Usage: python tools/reasoning_demand_probe.py [--root runtime_store] [--json out.json]
"""
import argparse
import glob
import io
import json
import os
import re
import sys

CLASSES = {
    "constraint": [
        "constraint", "consistent", "consistency", "invariant", "must not",
        "must never", "violates", "validates",
        "\u7ea6\u675f", "\u4e00\u81f4\u6027", "\u4e0d\u5f97", "\u8fdd\u53cd",
    ],
    "prerequisite": [
        "prerequisite", "requires", "depends on", "implies", "only if",
        "before you can", "precondition",
        "\u524d\u7f6e", "\u4f9d\u8d56", "\u8574\u542b",
    ],
    "box_completion": [
        "child box", "helper box", "sub-task", "subtask", "still open",
        "remaining work", "unfinished",
        "\u5b50\u76d2", "\u672a\u5b8c\u6210", "\u5269\u4f59\u5de5\u4f5c",
    ],
}

QUESTION_CUES = ["?", "\u662f\u5426", "\u54ea\u4e9b", "\u5982\u4f55\u4fdd\u8bc1",
                 "can we", "is it", "does it", "which"]


def classify(text):
    low = text.lower()
    hits = {}
    for name, keys in CLASSES.items():
        n = sum(low.count(k.lower()) for k in keys)
        if n:
            hits[name] = n
    asked = any(c.lower() in low for c in QUESTION_CUES)
    return hits, asked


def mission_goal(ws_dir):
    """Best-effort goal text: the first heading of the deliverable, else the
    directory name (the workspace does not persist the goal separately)."""
    path = os.path.join(ws_dir, "deliverable.md")
    if os.path.isfile(path):
        try:
            with io.open(path, encoding="utf-8", errors="replace") as f:
                head = f.read(400)
            for line in head.splitlines():
                line = line.strip()
                if line:
                    return line[:160]
        except Exception:
            pass
    return os.path.basename(ws_dir)


def count_ticks(root):
    pats = [
        os.path.join(root, "long_run", "gateway.log"),
        os.path.join(root, "long_run", "gateway.log.*"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p))
    ticks = 0
    for f in files:
        try:
            with io.open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "[mission-deliberate]" in line and "tokBudget" in line:
                        ticks += 1
        except Exception:
            continue
    return ticks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runtime_store")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ws_root = os.path.join(args.root, "mission_workspace")
    if not os.path.isdir(ws_root):
        sys.exit("no mission_workspace under %s" % args.root)

    missions = []
    for name in sorted(os.listdir(ws_root)):
        d = os.path.join(ws_root, name)
        if not os.path.isdir(d):
            continue
        deliv = os.path.join(d, "deliverable.md")
        if not os.path.isfile(deliv):
            continue
        try:
            with io.open(deliv, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:
            continue
        if not text.strip():
            continue
        hits, asked = classify(mission_goal(d) + "\n" + text)
        missions.append({"id": name, "chars": len(text), "hits": hits,
                         "asked": asked})

    total = len(missions)
    per_class = {k: 0 for k in CLASSES}
    any_hit = 0
    for m in missions:
        if m["hits"]:
            any_hit += 1
        for k in m["hits"]:
            per_class[k] += 1
    asked = sum(1 for m in missions if m["asked"])
    ticks = count_ticks(args.root)

    ratio = (any_hit / total) if total else 0.0
    gate = (ticks >= 200) and (total >= 30) and (ratio >= 0.20)

    print("== Q1 symbolic-demand probe (keyword PROXY, not an oracle) ==")
    print("missions_with_deliverable : %d" % total)
    print("deliberator ticks in logs : %d" % ticks)
    print("missions with >=1 hit     : %d (%.1f%%)" % (any_hit, 100 * ratio))
    for k, n in per_class.items():
        print("  %-16s %d (%.1f%%)" % (k, n, 100.0 * n / total if total else 0))
    print("missions asking a question: %d (%.1f%%)"
          % (asked, 100.0 * asked / total if total else 0))
    print("gate (ticks>=200, missions>=30, ratio>=20%%): %s"
          % ("PASS" if gate else "FAIL -> no demand, keep logic.enabled=false"))

    if args.json:
        with io.open(args.json, "w", encoding="utf-8") as f:
            json.dump({"total": total, "ticks": ticks, "anyHit": any_hit,
                       "ratio": ratio, "perClass": per_class, "asked": asked,
                       "gate": gate,
                       "sample": missions[:50]}, f, ensure_ascii=False, indent=2)
        print("detail -> %s" % args.json)


if __name__ == "__main__":
    main()

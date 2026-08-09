#!/usr/bin/env python3
"""C-Eval benchmark through Phoenix /api/chat (the same backend the web UI calls).

Tests the augmented llama-3.1 (GNN, context, etc.), not the raw llama-server.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aa_bench_datasets as bench_ds
import ui_e2e_test as ui

ROOT = Path(__file__).resolve().parent.parent
RE_ANSWER = re.compile(r"(?<![A-Za-z0-9])[A-D](?![A-Za-z0-9])")
BASE = "http://127.0.0.1:5080"
AUTH_BASE = "http://127.0.0.1:5081"
DEFAULT_TOKEN = "local-dev"


def log(msg: str) -> None:
    print(msg, flush=True)
    ui.log(msg)




def build_prompt(q: dict) -> str:
    return (
        f"问题：{q['question']}\n"
        f"A. {q['A']}\n"
        f"B. {q['B']}\n"
        f"C. {q['C']}\n"
        f"D. {q['D']}\n"
        "请只输出正确选项的一个字母（A/B/C/D），不要解释。"
    )


def api_call(method: str, path: str, body: dict | None = None, token: str = "", base: str = BASE) -> dict:
    url = base + path
    data = json.dumps(body or {}, ensure_ascii=False).encode("utf-8") if body is not None else b""
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json; charset=utf-8")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_choice(text: str) -> str | None:
    m = RE_ANSWER.search(text)
    if not m:
        return None
    return m.group(0).upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", choices=["on", "off"], default="on")
    parser.add_argument("--per-subject", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--dataset", choices=["ceval", "cmmlu"], default="ceval")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.gnn == "off":
        os.environ["AI_DISABLE_GNN_MODULE"] = "1"
    elif args.gnn == "on":
        os.environ.pop("AI_DISABLE_GNN_MODULE", None)

    procs = ui.start_phoenix()
    if procs is None:
        raise RuntimeError("phoenix_main is already running or failed to start")

    try:
        # Wait a moment for the gateway to be ready.
        for _ in range(30):
            try:
                with urllib.request.urlopen(BASE + "/api/world/status", timeout=3):
                    break
            except Exception:
                time.sleep(1)

        token = DEFAULT_TOKEN

        if args.output is None:
            args.output = ROOT / "build" / "tmp" / f"{args.dataset}_bench.json"
        questions = bench_ds.load_dataset(args.dataset, per_subject=args.per_subject)
        if args.max_questions > 0:
            questions = questions[:args.max_questions]
        log(f"dataset={args.dataset} GNN={args.gnn}, total questions={len(questions)}")

        correct = 0
        results = []
        for idx, q in enumerate(questions, 1):
            log(f"[{idx}/{len(questions)}] {q['subject']} id={q['id']}")
            prompt = build_prompt(q)
            reply = ""
            try:
                out = api_call("POST", "/api/chat", {"text": prompt, "token": token, "maxTokens": 24}, token=token)
                reply = out.get("result", {}).get("reply", "")
                if not reply and out.get("ok"):
                    # Some versions put the answer directly in "reply" at top level.
                    reply = out.get("reply", "")
            except urllib.error.HTTPError as e:
                log(f"  HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}")
                reply = ""
            choice = extract_choice(reply or "")
            ok = choice == q["answer"]
            correct += int(ok)
            results.append({
                "subject": q["subject"],
                "id": q["id"],
                "expected": q["answer"],
                "predicted": choice or "",
                "reply": reply,
                "correct": ok,
            })
            log(f"  expected={q['answer']} predicted={choice or 'NONE'} correct={ok}")
    finally:
        for p in reversed(procs):
            if p is None:
                continue
            try:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:
                    p.kill()
            except Exception as e:
                log(f"WARN: process stop failed: {e}")

    total = len(questions)
    accuracy = correct / total if total else 0.0
    stats = {"correct": correct, "total": total, "accuracy": accuracy, "results": results}
    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"{args.dataset} accuracy (GNN {args.gnn}): {correct}/{total} = {accuracy:.4%}")
    log(f"results written to {args.output}")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

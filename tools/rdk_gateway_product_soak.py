#!/usr/bin/env python3
"""Gateway product soak: health, login, frontend chat, 429, ops, module CRUD read.

Never curls llama /health. Never assigns missions. 429 chat-busy is a boundary.
Chat timeout/slow is not down. lastOkAge only follows health/status/security.
Old monitor verdict=PASS is not used here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http(
    url: str,
    method: str = "GET",
    body: dict | None = None,
    token: str = "",
    timeout: float = 60.0,
):
    data = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        status = e.code
        elapsed = time.perf_counter() - t0
        try:
            doc = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            doc = {"ok": False, "error": "invalid-json", "raw": raw[:240]}
        return elapsed, status, doc
    except Exception as e:
        return time.perf_counter() - t0, 0, {"ok": False, "error": str(e)}
    elapsed = time.perf_counter() - t0
    try:
        doc = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        doc = {"ok": False, "error": "invalid-json", "raw": raw[:240]}
    return elapsed, status, doc


def url_path(url: str) -> str:
    if "://" in url:
        url = "/" + url.split("/", 3)[-1]
    return url


CHAT_TIMEOUT_SEC = 300.0
CHAT_SLOW_SEC = 90.0
LIVENESS_MARKERS = ("/api/health", "/api/system/status", "/security/")


def is_chat_path(path: str) -> bool:
    return "/api/chat" in url_path(path)


def is_liveness_path(path: str) -> bool:
    p = url_path(path)
    return any(m in p for m in LIVENESS_MARKERS)


def is_timeout_error(err: str) -> bool:
    e = err.lower()
    return any(
        n in e
        for n in (
            "timed out",
            "timeout",
            "time out",
            "read timed out",
            "the read operation timed out",
        )
    )


def is_connect_down_error(err: str) -> bool:
    e = err.lower()
    return any(
        n in e
        for n in (
            "connection refused",
            "actively refused",
            "connection reset",
            "failed to connect",
            "no connection could be made",
            "network is unreachable",
            "no route to host",
            "name or service not known",
            "getaddrinfo",
            "10061",
            "10054",
            "errno 111",
            "[errno 111]",
        )
    )


def classify(status: int, doc: dict, path: str, elapsed: float = 0.0) -> str:
    err = str(doc.get("error") or "")
    p = url_path(path)
    chat = is_chat_path(p)
    if status == 429 or err == "chat-busy":
        return "busy"
    if status == 0:
        if is_timeout_error(err):
            return "timeout"
        if is_connect_down_error(err) or not err:
            return "down"
        if chat:
            return "timeout"
        return "down"
    if "/security/resources/construct" in p or "/security/resources/deploy" in p:
        return "ok" if status == 404 else "error"
    if 200 <= status < 300:
        if chat and elapsed >= CHAT_SLOW_SEC:
            return "slow"
        return "ok"
    if "/api/modules/" in p and status in (401, 403, 404):
        return "boundary"
    return "error"


def probe_doc_ok(doc: dict) -> tuple[bool, str]:
    """Default-off: GET probe must not show enable/plant/spread."""
    items = doc.get("items")
    row = items[0] if isinstance(items, list) and items else doc
    if not isinstance(row, dict):
        return False, "probe-not-object"
    if row.get("allowInertProbe") is True:
        return False, "allowInertProbe-on"
    if row.get("probeEnabled") is True:
        return False, "probeEnabled-on"
    if row.get("planted") is True:
        return False, "planted"
    traces = row.get("traces") or []
    act = row.get("activation") or {}
    if traces or (isinstance(act, dict) and act):
        return False, "probe-spread"
    return True, ""


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self.by_path: Counter[str] = Counter()
        self.started = time.time()
        self.last_ok = 0.0
        self.fatal = ""

    def add(self, path: str, kind: str, elapsed: float) -> None:
        with self.lock:
            self.counts[kind] += 1
            self.by_path[f"{path}:{kind}"] += 1
            if kind == "ok" and is_liveness_path(path):
                self.last_ok = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "elapsedSec": int(time.time() - self.started),
                "counts": dict(self.counts),
                "byPath": dict(self.by_path),
                "lastOkAgeSec": None if not self.last_ok else int(time.time() - self.last_ok),
                "fatal": self.fatal,
            }


def login(front: str, user: str, password: str) -> str:
    elapsed, status, doc = http(
        front + "/auth/login",
        method="POST",
        body={"username": user, "password": password},
        timeout=20,
    )
    print(f"{now_iso()} login status={status} dt={elapsed:.2f} keys={list(doc)[:8]}", flush=True)
    token = str(doc.get("token") or doc.get("accessToken") or "")
    if status == 200 and token:
        return token
    elapsed, status, doc = http(front + "/auth/config", timeout=15)
    print(f"{now_iso()} auth/config status={status} bootstrap={doc.get('allowBootstrap')}", flush=True)
    if doc.get("allowBootstrap"):
        elapsed, status, doc = http(
            front + "/auth/bootstrap",
            method="POST",
            body={
                "username": user,
                "password": password,
                "email": "ops@localhost",
            },
            timeout=20,
        )
        print(f"{now_iso()} bootstrap status={status} dt={elapsed:.2f}", flush=True)
        token = str(doc.get("token") or doc.get("accessToken") or "")
        if token:
            return token
    if not token:
        elapsed, status, doc = http(
            front + "/auth/login",
            method="POST",
            body={"username": user, "password": password},
            timeout=20,
        )
        token = str(doc.get("token") or doc.get("accessToken") or "")
    return token


def cycle(front: str, gateway: str, token: str, stats: Stats, chat_n: int, crud_token: str) -> None:
    gw = gateway.rstrip("/")
    checks = [
        (front + "/api/health", "GET", None, token, 15),
        (front + "/api/ops/monitor", "GET", None, token, 20),
        (front + "/api/ops/modules", "GET", None, token, 20),
        (gw + "/api/system/status", "GET", None, token, 20),
        (gw + "/api/modules/session/resources/meta", "GET", None, crud_token, 20),
        (gw + "/api/modules/security/resources/stats", "GET", None, crud_token, 20),
        (gw + "/api/modules/security/resources/identify", "GET", None, crud_token, 20),
        (gw + "/api/modules/security/resources/alerts", "GET", None, crud_token, 20),
        (gw + "/api/modules/security/resources/defense", "GET", None, crud_token, 20),
        (gw + "/api/modules/security/resources/probe", "GET", None, crud_token, 20),
        (gw + "/api/modules/security/resources/construct", "GET", None, crud_token, 15),
        (gw + "/api/modules/security/resources/deploy", "GET", None, crud_token, 15),
    ]
    for url, method, body, tok, to in checks:
        elapsed, status, doc = http(url, method, body, tok, to)
        kind = classify(status, doc, url, elapsed)
        if "/security/resources/probe" in url and kind == "ok":
            okp, why = probe_doc_ok(doc)
            if not okp:
                kind = "error"
                with stats.lock:
                    stats.fatal = why
                print(f"{now_iso()} FAIL probe-default-off {why} {doc}", flush=True)
        stats.add(url, kind, elapsed)
        if kind == "down":
            print(f"{now_iso()} down {url} {doc}", flush=True)
        if kind == "error" and "/security/" in url:
            print(f"{now_iso()} security {url} status={status} {doc}", flush=True)

    body = {
        "text": f"Reply with one word: OK. n={chat_n}",
        "sessionId": f"product-soak-{chat_n % 3}",
        "maxTokens": 8,
        "enableGraphSelector": False,
    }
    elapsed, status, doc = http(front + "/api/chat", "POST", body, token, CHAT_TIMEOUT_SEC)
    kind = classify(status, doc, "/api/chat", elapsed)
    stats.add("/api/chat", kind, elapsed)
    if kind in ("timeout", "slow"):
        print(f"{now_iso()} {kind} /api/chat dt={elapsed:.2f} status={status} {doc}", flush=True)


def burst_429(front: str, token: str, stats: Stats) -> None:
    def one(i: int) -> None:
        body = {
            "text": f"Reply with one word: OK. burst={i}",
            "sessionId": f"burst-{i}",
            "maxTokens": 8,
            "enableGraphSelector": False,
        }
        elapsed, status, doc = http(front + "/api/chat", "POST", body, token, CHAT_TIMEOUT_SEC)
        stats.add("/api/chat", classify(status, doc, "/api/chat", elapsed), elapsed)

    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--front", default="http://192.168.1.107:5081")
    p.add_argument("--gateway", default="http://192.168.1.107:5080")
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--user", default="soakadmin")
    p.add_argument("--password", default="admin123")
    p.add_argument("--crud-token", default="local-dev")
    p.add_argument("--report", default="phoenix/tools/_tmp_gateway_product_soak_rdk.json")
    p.add_argument("--log-every-sec", type=int, default=1200)
    args = p.parse_args()

    elapsed, st, doc = http(args.front + "/api/health", timeout=15)
    print(f"{now_iso()} probe front/api/health status={st} dt={elapsed:.2f}", flush=True)
    if st == 0:
        print("FAIL: frontend not reachable", flush=True)
        return 2
    elapsed, st, doc = http(args.gateway + "/api/system/status", timeout=15)
    print(f"{now_iso()} probe gateway/status status={st} dt={elapsed:.2f}", flush=True)
    if st == 0:
        print("FAIL: gateway not reachable (do not soak a dead process)", flush=True)
        return 2

    token = login(args.front, args.user, args.password)
    if not token:
        print("WARN: no login token; continuing with empty bearer (local fallback may apply)", flush=True)

    stats = Stats()
    stop = threading.Event()
    n = 0

    def loop() -> None:
        nonlocal n
        while not stop.is_set():
            n += 1
            cycle(args.front, args.gateway, token, stats, n, args.crud_token)
            if n % 16 == 8:
                burst_429(args.front, token, stats)
            if stop.wait(12):
                break

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    deadline = time.time() + args.hours * 3600.0
    next_log = time.time() + args.log_every_sec
    rc = 0
    print(f"{now_iso()} start hours={args.hours} front={args.front}", flush=True)
    try:
        while time.time() < deadline:
            time.sleep(5)
            snap = stats.snapshot()
            last_ok_age = snap.get("lastOkAgeSec")
            if last_ok_age is None:
                last_ok_age = 999
            if snap.get("fatal"):
                print(f"{now_iso()} FAIL fatal {snap}", flush=True)
                rc = 2
                break
            if snap["counts"].get("error", 0) >= 8 and snap["counts"].get("ok", 0) == 0:
                print(f"{now_iso()} FAIL errors {snap}", flush=True)
                rc = 2
                break
            if snap["counts"].get("down", 0) >= 6 and last_ok_age > 90:
                print(f"{now_iso()} FAIL down {snap}", flush=True)
                rc = 2
                break
            if time.time() >= next_log:
                print(f"{now_iso()} progress {json.dumps(snap, ensure_ascii=False)}", flush=True)
                write_json(args.report, {"at": now_iso(), "progress": snap, "verdict": "RUNNING"})
                next_log = time.time() + args.log_every_sec
    except KeyboardInterrupt:
        rc = 130
    stop.set()
    t.join(timeout=4)
    snap = stats.snapshot()
    verdict = "FAIL" if rc not in (0, 130) else "RUNNING_DONE"
    if rc == 0 and snap["counts"].get("ok", 0) == 0:
        verdict = "FAIL"
        rc = 2
    elif rc == 0:
        verdict = "COMPLETE"
    out = {"at": now_iso(), "verdict": verdict, "final": snap, "front": args.front}
    write_json(args.report, out)
    print(f"{now_iso()} {verdict} {json.dumps(out, ensure_ascii=False)}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

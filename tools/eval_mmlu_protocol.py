#!/usr/bin/env python3
"""eval_mmlu_protocol.py — M8 evaluation protocol harness for Phoenix.

WHAT THIS IS
------------
A runnable implementation of the "small-sample statistics protocol" (M8 in
doc/v10.1/methods_backlog.md).  It has three modes:

  legacy      n=40 uniform sample, seed 20261108 — the historical sentinel.
              Reproduces tools/mmlu_subset_eval.py's SAMPLING EXACTLY
              (pandas 3 `df.sample(n, random_state=seed)` ==
              `numpy.random.RandomState(seed).choice(len(df), n, replace=False)`,
              verified on this box 2026-11-17), so scores stay comparable with
              runtime_store/long_run/mmlu_history.jsonl.
  stratified  n=400 stratified by Subject (Hamilton largest-remainder),
              fixed seed — the new primary (±0.025 at p~0.48).
  paired      McNemar exact two-sided + paired bootstrap 95% CI, comparing
              two arms (two endpoints/builds, or two previously saved runs).

WHY STDLIB-ONLY
---------------
The box has python 3.11 + numpy + pandas but **no openai package**, and the
project rule is "do not install things to make a test run".  The eval path
therefore uses only the standard library (urllib/csv/json/random).  numpy is
optional and used *only* to reproduce the legacy pandas sampling exactly.

SAFETY (does not disturb the live run)
--------------------------------------
* Never compiles anything, never calls compile.bat, never touches gateway
  routes.  It is a pure HTTP client to an OpenAI-compatible /v1 endpoint.
* Default endpoint is the shared llama-server (http://127.0.0.1:8082/v1) to
  match the legacy tool; pass --base-url to use a sidecar port instead.
* --self-test runs entirely against an in-process mock HTTP server (no load
  on the production model at all).
* It does not edit config/, does not write product files; outputs go to
  build/tmp/mmlu_protocol/ (or --out).

USAGE
-----
  # legacy sentinel (40 items, continuity) against the shared llama-server
  python tools/eval_mmlu_protocol.py --mode legacy --n 40 --seed 20261108 \
      --base-url http://127.0.0.1:8082/v1 --tag legacy

  # new primary: 400 items stratified by subject
  python tools/eval_mmlu_protocol.py --mode stratified --n 400 --seed 20261117

  # paired comparison of two builds/endpoints over the same stratified items
  python tools/eval_mmlu_protocol.py --mode paired --n 400 --seed 20261117 \
      --base-url-a http://127.0.0.1:5101/v1 --model-a phoenix-next \
      --base-url-b http://127.0.0.1:8082/v1

  # paired analysis of two previously saved runs (no network at all)
  python tools/eval_mmlu_protocol.py --mode paired \
      --a-json build/tmp/mmlu_protocol/stratified-...a.json \
      --b-json build/tmp/mmlu_protocol/stratified-...b.json

  # power analysis table (no network)
  python tools/eval_mmlu_protocol.py --power-table

  # prove all plumbing without touching any real service
  python tools/eval_mmlu_protocol.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOL_VERSION = "phoenix.mmlu_protocol.v1"
DEFAULT_CSV_CANDIDATES = [
    "build/tmp/mmlu.csv",
    "tools/data/mmlu.csv",
    "runtime_store/mmlu.csv",
    "data/mmlu.csv",
    "mmlu.csv",
]
REQUIRED_COLUMNS = ("Question", "A", "B", "C", "D", "Answer", "Subject")

# Historical sentinel parameters (tools/mmlu_subset_eval.py defaults).
LEGACY_N = 40
LEGACY_SEED = 20261108


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def project_root_from_tool() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def find_csv(explicit: str | None) -> str:
    if explicit:
        cand = os.path.abspath(explicit)
        if not os.path.isfile(cand):
            die("--csv not found: %s" % cand)
        return cand
    env = os.environ.get("PHOENIX_MMLU_CSV")
    if env:
        cand = os.path.abspath(env)
        if os.path.isfile(cand):
            return cand
    root = project_root_from_tool()
    for rel in DEFAULT_CSV_CANDIDATES:
        cand = os.path.abspath(os.path.join(root, rel))
        if os.path.isfile(cand):
            return cand
    die("no MMLU csv found; looked for %s under %s (set --csv or PHOENIX_MMLU_CSV)"
        % (", ".join(DEFAULT_CSV_CANDIDATES), root))
    raise RuntimeError("unreachable")


def die(msg: str, code: int = 2) -> None:
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class Record(dict):
    """One MMLU item. dict so it serialises directly."""
    pass


def normalise_gold(answer: str, options: dict[str, str]) -> str | None:
    """MMLU Answer column is normally the letter (A-D).  Be tolerant if a row
    instead repeats the option text: map the matching option to its letter."""
    ans = (answer or "").strip()
    if len(ans) == 1 and ans.upper() in "ABCD":
        return ans.upper()
    if not ans:
        return None
    for letter in "ABCD":
        if options.get(letter, "").strip() == ans:
            return letter
    return None


def load_records(csv_path: str) -> tuple[list[Record], dict]:
    """Load and normalise the MMLU CSV.  Returns (records, dataset_meta).

    IMPORTANT — keep the frame identical to the historical pandas path.
    `tools/mmlu_subset_eval.py` reads the CSV with pandas and samples row
    positions over the *full* frame.  A handful of rows have one empty option
    cell; pandas turns that cell into NaN and the legacy prompt therefore
    contains the literal string "nan" (`A) nan`).  We keep those rows (flagged
    `nan_option=True`) and render "" as "nan", so both the sampling frame size
    and the prompt text match the legacy tool byte-for-byte.  Dropping them
    would silently change every legacy sample (14,042 -> 14,032 positions).
    """
    records: list[Record] = []
    excluded = {"missing_question": 0, "missing_subject": 0, "bad_answer": 0}
    nan_option_rows = 0
    subject_sizes: dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            die("csv %s missing required columns %s (header: %s)"
                % (csv_path, missing, header))
        for row_index, row in enumerate(reader):
            # Keep raw text for prompt building: pandas does not strip CSV fields,
            # and the legacy prompt is built from raw values (leading/trailing
            # spaces included).  Stripped copies are used only for validation.
            q_raw = row.get("Question") or ""
            q = q_raw.strip()
            subject = (row.get("Subject") or "").strip()
            options_raw = {L: (row.get(L) or "") for L in "ABCD"}
            options_stripped = {L: v.strip() for L, v in options_raw.items()}
            if not q:
                excluded["missing_question"] += 1
                continue
            if not subject:
                excluded["missing_subject"] += 1
                continue
            gold = normalise_gold(row.get("Answer") or "", options_stripped)
            if gold is None:
                excluded["bad_answer"] += 1
                continue
            has_nan_option = any(not v for v in options_stripped.values())
            if has_nan_option:
                nan_option_rows += 1
            subject_sizes[subject] = subject_sizes.get(subject, 0) + 1
            records.append(Record(
                key="mmlu-%05d" % row_index,
                row_index=row_index,
                qsha=sha1_text(q),
                subject=subject,
                question=q_raw,
                options=options_raw,
                gold=gold,
                nan_option=has_nan_option,
            ))
    meta = {
        "path": os.path.abspath(csv_path),
        "sha256": sha256_file(csv_path),
        "bytes": os.path.getsize(csv_path),
        "records_total": len(records) + sum(excluded.values()),
        "records_eligible": len(records),
        "rows_with_nan_option": nan_option_rows,
        "excluded": excluded,
        "subjects": len(subject_sizes),
        "subject_sizes": dict(sorted(subject_sizes.items())),
        "note": ("MMLU test split; CSV physical line count is larger than record "
                 "count because some Question fields contain embedded newlines "
                 "inside quotes. Use this record count, not `wc -l`. Empty option "
                 "cells are rendered as 'nan' to match pandas' legacy prompt."),
    }
    if not records:
        die("no eligible records in %s" % csv_path)
    return records, meta


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def _hamilton(allocation_quota: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder (Hamilton) apportionment of n seats over subjects."""
    base = {s: int(math.floor(q)) for s, q in allocation_quota.items()}
    used = sum(base.values())
    remainders = sorted(
        ((allocation_quota[s] - base[s], s) for s in allocation_quota),
        key=lambda t: (-t[0], t[1]),
    )
    i = 0
    while used < n:
        _, s = remainders[i % len(remainders)]
        base[s] += 1
        used += 1
        i += 1
    # floor never exceeds quota, so base is a valid allocation; used == n now
    return base


def sample_legacy(records: list[Record], n: int, seed: int) -> tuple[list[Record], dict]:
    """Exact reproduction of pandas DataFrame.sample(n, random_state=seed).

    Verified 2026-11-17 on this box: pandas 3.0.6 `df.sample(n, random_state=s)`
    returns the same row positions as `RandomState(s).choice(len, n, replace=False)`.
    numpy is optional; without it we fall back to a documented RandomState-like
    approximation and say so in the metadata.
    """
    N = len(records)
    if n > N:
        die("--n %d exceeds eligible records %d" % (n, N))
    try:
        import numpy as np  # noqa: F401
        rs = np.random.RandomState(seed)
        idx = [int(i) for i in rs.choice(N, size=n, replace=False)]
        method = "legacy-uniform (numpy RandomState; == pandas 3 sample for int seed)"
        numpy_used = True
    except Exception:
        rng = random.Random(seed)
        idx = rng.sample(range(N), n)
        method = ("legacy-uniform (stdlib fallback; item identity may differ "
                  "from historical pandas runs)")
        numpy_used = False
    items = [records[i] for i in idx]
    meta = {"method": method, "seed": seed, "n": n, "numpy_used": numpy_used,
            "row_positions": idx[:8] + ["..."] if len(idx) > 8 else idx}
    return items, meta


def sample_stratified(records: list[Record], n: int, seed: int) -> tuple[list[Record], dict]:
    """Proportional stratified sample by Subject.

    Allocation: quota_s = n * size_s / N, floors via Hamilton largest remainder.
    Within-subject order: deterministic per (seed, subject) via
    RandomState(sha256(seed|subject) first 8 bytes).  Item order is
    subject-sorted so a run is reproducible from the metadata alone.
    """
    N = len(records)
    if n > N:
        die("--n %d exceeds eligible records %d" % (n, N))
    groups: dict[str, list[Record]] = {}
    for r in records:
        groups.setdefault(r["subject"], []).append(r)
    quota = {s: n * len(g) / N for s, g in groups.items()}
    alloc = _hamilton(quota, n)
    chosen: list[Record] = []
    for subject in sorted(groups):
        group = groups[subject]
        k = alloc[subject]
        if k <= 0:
            continue
        digest = hashlib.sha256(("%d|%s" % (seed, subject)).encode()).digest()
        per_subject_seed = int.from_bytes(digest[:8], "big") % (2 ** 32)
        rng = random.Random(per_subject_seed)
        pick = rng.sample(range(len(group)), k)
        chosen.extend(group[i] for i in sorted(pick))
    meta = {
        "method": "stratified-by-subject (Hamilton largest remainder, "
                  "within-subject seeded sample)",
        "seed": seed,
        "n": n,
        "allocation": {s: alloc[s] for s in sorted(alloc)},
        "allocation_min": min(alloc.values()),
        "allocation_max": max(alloc.values()),
    }
    return chosen, meta


# ---------------------------------------------------------------------------
# prompting / parsing / http
# ---------------------------------------------------------------------------

def make_prompt(rec: Record) -> str:
    o = rec["options"]

    def opt(v: str) -> str:
        # pandas turns empty cells into NaN -> legacy prompt contains "nan";
        # non-empty cells are used verbatim (including any surrounding spaces).
        return v if v != "" else "nan"

    return ("%s\nA) %s\nB) %s\nC) %s\nD) %s\nAnswer:"
            % (rec["question"], opt(o["A"]), opt(o["B"]), opt(o["C"]), opt(o["D"])))


def parse_pred_legacy(raw: str) -> str | None:
    """Same rule as tools/mmlu_subset_eval.py: first A-D letter anywhere."""
    for ch in raw or "":
        if ch.upper() in "ABCD":
            return ch.upper()
    return None


_WORD_LETTER = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_pred_strict(raw: str) -> str | None:
    """Prefer a leading letter, else a standalone token; None if ambiguous."""
    s = (raw or "").strip()
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    m = _WORD_LETTER.search(s)
    return m.group(1).upper() if m else None


def http_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def resolve_model(base_url: str, explicit: str | None, timeout: float) -> str:
    if explicit:
        return explicit
    try:
        models = http_json(base_url.rstrip("/") + "/models", None, timeout)
        data = models.get("data") or []
        if data and data[0].get("id"):
            return str(data[0]["id"])
    except Exception:
        pass
    return "(unspecified; server default)"


def call_chat(base_url: str, model: str, prompt: str, *,
              temperature: float, max_tokens: int, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    out = http_json(base_url.rstrip("/") + "/chat/completions", payload, timeout)
    choices = out.get("choices") or []
    if not choices:
        raise RuntimeError("no choices in response: %r" % (out,))
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        content = choices[0].get("text") or ""
    return str(content)


# ---------------------------------------------------------------------------
# evaluation loop + summary
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def eval_arm(items: list[Record], *, base_url: str, model: str,
             temperature: float, max_tokens: int, timeout: float,
             retries: int, max_errors: int, progress: bool,
             arm_label: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    errors = 0
    latencies: list[float] = []
    n = len(items)
    correct = 0
    t0 = time.time()
    for i, rec in enumerate(items):
        prompt = make_prompt(rec)
        raw = ""
        err = ""
        lat = 0.0
        for attempt in range(retries + 1):
            t1 = time.time()
            try:
                raw = call_chat(base_url, model, prompt,
                                temperature=temperature,
                                max_tokens=max_tokens, timeout=timeout)
                lat = (time.time() - t1) * 1000.0
                err = ""
                break
            except Exception as exc:  # noqa: BLE001 - we record and continue
                lat = (time.time() - t1) * 1000.0
                err = "%s: %s" % (type(exc).__name__, exc)
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        pred_legacy = parse_pred_legacy(raw) if not err else None
        pred_strict = parse_pred_strict(raw) if not err else None
        ok = (pred_legacy == rec["gold"])
        if err:
            errors += 1
        if ok:
            correct += 1
        latencies.append(lat)
        rows.append({
            "key": rec["key"],
            "qsha": rec["qsha"],
            "subject": rec["subject"],
            "gold": rec["gold"],
            "pred_legacy": pred_legacy,
            "pred_strict": pred_strict,
            "correct": bool(ok),
            "raw": raw[:200],
            "latency_ms": round(lat, 1),
            "error": err,
            "arm": arm_label,
        })
        if progress and (i + 1) % 10 == 0:
            print("  [%s] %d/%d correct=%d errors=%d"
                  % (arm_label, i + 1, n, correct, errors), flush=True)
        if max_errors and errors > max_errors:
            raise RuntimeError(
                "aborting arm %s: %d errors (> --max-errors %d); last: %s"
                % (arm_label, errors, max_errors, rows[-1]["error"]))
    elapsed = time.time() - t0
    lat_sorted = sorted(latencies)
    per_subject: dict[str, dict] = {}
    for r in rows:
        s = per_subject.setdefault(r["subject"], {"correct": 0, "total": 0})
        s["total"] += 1
        s["correct"] += 1 if r["correct"] else 0
    summary = {
        "n": n,
        "correct": correct,
        "score": (correct / n) if n else None,
        "wilson_ci95": wilson_ci(correct, n),
        "errors": errors,
        "strict_score": (sum(1 for r in rows if r["pred_strict"] == r["gold"]) / n) if n else None,
        "elapsed_s": round(elapsed, 1),
        "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "latency_ms_p50": lat_sorted[len(lat_sorted) // 2] if lat_sorted else None,
        "latency_ms_p95": lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95))] if lat_sorted else None,
        "per_subject": {k: per_subject[k] for k in sorted(per_subject)},
    }
    return rows, summary


# ---------------------------------------------------------------------------
# paired analysis (McNemar exact + paired bootstrap CI)
# ---------------------------------------------------------------------------

def _binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    d = b + c
    if d == 0:
        return 1.0
    tail = sum(_binom_pmf(i, d, 0.5) for i in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail)


def paired_bootstrap_delta(rows_a: list[dict], rows_b: list[dict], resamples: int,
                           seed: int) -> dict:
    """Paired bootstrap 95% CI for delta = score_a - score_b, plus McNemar."""
    if len(rows_a) != len(rows_b):
        die("paired runs differ in length: %d vs %d" % (len(rows_a), len(rows_b)))
    n = len(rows_a)
    for ra, rb in zip(rows_a, rows_b):
        if ra["key"] != rb["key"]:
            die("paired runs are not aligned: %s vs %s" % (ra["key"], rb["key"]))
    b = c = 0
    same_both_correct = same_both_wrong = 0
    for ra, rb in zip(rows_a, rows_b):
        ok_a, ok_b = ra["correct"], rb["correct"]
        if ok_a and not ok_b:
            b += 1
        elif ok_b and not ok_a:
            c += 1
        elif ok_a and ok_b:
            same_both_correct += 1
        else:
            same_both_wrong += 1
    delta = (sum(1 for r in rows_a if r["correct"]) -
             sum(1 for r in rows_b if r["correct"])) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(1 for i in idx if rows_a[i]["correct"])
        db = sum(1 for i in idx if rows_b[i]["correct"])
        deltas.append((da - db) / n)
    deltas.sort()
    lo = deltas[max(0, int(0.025 * resamples) - 1)]
    hi = deltas[min(resamples - 1, int(0.975 * resamples))]
    return {
        "n_pairs": n,
        "a_correct": sum(1 for r in rows_a if r["correct"]),
        "b_correct": sum(1 for r in rows_b if r["correct"]),
        "b_a_only": b,
        "c_b_only": c,
        "discordant": b + c,
        "same_both_correct": same_both_correct,
        "same_both_wrong": same_both_wrong,
        "delta_pp": round(delta * 100.0, 3),
        "mcnemar_exact_p_two_sided": mcnemar_exact_p(b, c),
        "bootstrap_resamples": resamples,
        "bootstrap_ci95_pp": [round(lo * 100.0, 3), round(hi * 100.0, 3)],
        "bootstrap_ci95_delta": [round(lo, 6), round(hi, 6)],
        "note": ("b = A-only-correct, c = B-only-correct. Positive delta_pp means "
                 "A > B. Bootstrap is paired resampling of items (same items in "
                 "both arms); McNemar is the exact two-sided binomial test."),
    }


# ---------------------------------------------------------------------------
# power analysis (exact, no normal approximation for the discrete tests)
# ---------------------------------------------------------------------------

def _reject_region_single(n: int, p0: float, alpha: float, sided: str) -> tuple[int, int]:
    """Return (k_lo, k_hi) such that values <=k_lo or >=k_hi are rejected.
    For one-sided upper test, k_lo is -1 (no lower rejection)."""
    if sided == "two":
        k_hi = n + 1
        for k in range(n, -1, -1):
            # upper tail P(X >= k)
            if sum(_binom_pmf(i, n, p0) for i in range(k, n + 1)) <= alpha / 2.0:
                k_hi = k
            else:
                break
        k_lo = -1
        for k in range(0, n + 1):
            if sum(_binom_pmf(i, n, p0) for i in range(0, k + 1)) <= alpha / 2.0:
                k_lo = k
            else:
                break
        return k_lo, k_hi
    # one-sided: H1 p > p0
    k_hi = n + 1
    for k in range(n, -1, -1):
        if sum(_binom_pmf(i, n, p0) for i in range(k, n + 1)) <= alpha:
            k_hi = k
        else:
            break
    return -1, k_hi


def mde_single_arm(n: int, p0: float, alpha: float, sided: str,
                   power: float = 0.8, step: float = 0.001) -> float | None:
    """Smallest absolute score difference d such that an exact binomial test
    at level alpha has >= power probability of rejecting H0: p=p0."""
    k_lo, k_hi = _reject_region_single(n, p0, alpha, sided)
    if k_hi > n:
        return None
    d = 0.0
    while d <= 0.5:
        p1 = min(1.0, p0 + d)
        pw = sum(_binom_pmf(i, n, p1) for i in range(0, k_lo + 1)) if k_lo >= 0 else 0.0
        pw += sum(_binom_pmf(i, n, p1) for i in range(k_hi, n + 1))
        if pw >= power:
            return round(d, 4)
        d += step
    return None


def mde_mcnemar(n: int, discordance: float, alpha: float = 0.05,
                power: float = 0.8) -> dict:
    """Smallest detectable NET accuracy difference (pp) for an exact two-sided
    McNemar test, given n items and a discordance rate (b+c)/n.

    Under H1 the net difference delta = (b-c)/n; with D = b+c fixed,
    b ~ Binomial(D, p1), p1 = (D + delta*n) / (2D).  We scan delta in units of
    1/n and find the smallest value whose exact rejection probability >= power.
    """
    D = int(round(discordance * n))
    if D <= 0:
        return {"discordance": discordance, "D": 0, "mde_pp": None,
                "note": "no discordant pairs possible"}
    # critical net (absolute |b-c|) for two-sided exact McNemar at alpha
    crit = None
    for net in range(0, D + 1, 2):
        b = (D + net) // 2
        c = D - b
        if mcnemar_exact_p(b, c) <= alpha:
            crit = net
            break
    if crit is None:
        return {"discordance": discordance, "D": D, "mde_pp": None,
                "note": "even a 100%% flip cannot reach alpha=%.3f" % alpha}
    best = None
    for net in range(crit, D + 1, 2):
        p1 = (D + net) / (2.0 * D)
        # power: P(b in rejection region | b ~ Bin(D, p1)) = P(net(b) >= crit)
        pw = 0.0
        for b in range(0, D + 1):
            net_b = abs(2 * b - D)
            if net_b >= crit:
                pw += _binom_pmf(b, D, p1)
        if pw >= power:
            best = net
            break
    if best is None:
        return {"discordance": discordance, "D": D, "mde_pp": None,
                "note": "power target unreachable at this n/discordance"}
    return {"discordance": discordance, "D": D,
            "mde_pp": round(100.0 * best / n, 2),
            "critical_net_pairs": crit,
            "note": "exact two-sided McNemar, alpha=0.05, power=0.8"}


def power_table(p0: float = 0.4777, alpha: float = 0.05) -> list[dict]:
    rows = []
    for n in (40, 200, 400):
        one = mde_single_arm(n, p0, alpha, "one")
        two = mde_single_arm(n, p0, alpha, "two")
        row = {
            "n": n,
            "single_arm_one_sided_pp": None if one is None else round(one * 100, 1),
            "single_arm_two_sided_pp": None if two is None else round(two * 100, 1),
        }
        for disc in (0.10, 0.20, 0.30):
            m = mde_mcnemar(n, disc, alpha=alpha)
            row["paired_mde_pp_discordance_%d" % int(disc * 100)] = m["mde_pp"]
        rows.append(row)
    return rows


def print_power_table(p0: float = 0.4777, alpha: float = 0.05) -> None:
    rows = power_table(p0, alpha)
    print("Power analysis: alpha=%.2f, power=0.80, baseline p0=%.4f "
          "(measured 8B mean on the 40-item sentinel)" % (alpha, p0))
    print()
    hdr = ("| n | single-arm 1-sided MDE | single-arm 2-sided MDE | "
           "paired MDE @10% disc | paired MDE @20% disc | paired MDE @30% disc |")
    print(hdr)
    print("|---|---|---|---|---|---|")
    for r in rows:
        def fmt(v):
            return "n/a" if v is None else ("%.1f pp" % v)
        print("| %d | %s | %s | %s | %s | %s |" % (
            r["n"],
            fmt(r["single_arm_one_sided_pp"]),
            fmt(r["single_arm_two_sided_pp"]),
            fmt(r["paired_mde_pp_discordance_10"]),
            fmt(r["paired_mde_pp_discordance_20"]),
            fmt(r["paired_mde_pp_discordance_30"]),
        ))
    print()
    print("Legend: single-arm = score vs fixed reference (regression sentinel); "
          "paired = before/after on the same items (McNemar, exact). "
          "'n/a' means alpha/power is unreachable at that n/discordance "
          "(no amount of luck rejects).")


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

ITEM_COLUMNS = ["key", "subject", "gold", "pred_legacy", "pred_strict",
                "correct", "raw", "latency_ms", "error", "arm"]


def write_run(outdir: str, tag: str, mode: str, payload: dict,
              items: list[dict]) -> dict:
    os.makedirs(outdir, exist_ok=True)
    stamp = utc_stamp()
    base = "%s-%s-n%d-seed%s-%s" % (tag, mode, payload.get("sampling", {}).get("n", 0),
                                    payload.get("sampling", {}).get("seed", "na"), stamp)
    json_path = os.path.join(outdir, base + ".json")
    csv_path = os.path.join(outdir, base + ".csv")
    payload["outputs"] = {"json": os.path.abspath(json_path),
                          "csv": os.path.abspath(csv_path)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_COLUMNS)
        w.writeheader()
        for row in items:
            w.writerow({k: row.get(k) for k in ITEM_COLUMNS})
    return {"json": os.path.abspath(json_path), "csv": os.path.abspath(csv_path)}


def write_paired(outdir: str, tag: str, payload: dict,
                 discordant: list[dict]) -> dict:
    os.makedirs(outdir, exist_ok=True)
    stamp = utc_stamp()
    base = "%s-paired-n%d-seed%s-%s" % (
        tag, payload.get("sampling", {}).get("n", 0),
        payload.get("sampling", {}).get("seed", "na"), stamp)
    json_path = os.path.join(outdir, base + ".json")
    csv_path = os.path.join(outdir, base + ".csv")
    payload["outputs"] = {"json": os.path.abspath(json_path),
                          "csv": os.path.abspath(csv_path)}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cols = ["key", "subject", "gold", "pred_a", "correct_a",
            "pred_b", "correct_b", "type"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in discordant:
            w.writerow({k: row.get(k) for k in cols})
    return {"json": os.path.abspath(json_path), "csv": os.path.abspath(csv_path)}


def load_run(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# self-test (in-process mock, zero external load)
# ---------------------------------------------------------------------------

class _MockHandler(BaseHTTPRequestHandler):
    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # aiohttp-style models endpoint
        self._send({"object": "list", "data": [{"id": "mock-model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        model = str(req.get("model", ""))
        if model == "mock-correct":
            # deterministic pseudo-random letter, independent of gold (tests plumbing)
            content = "B"
        elif model == "mock-flip":
            content = "C"
        else:
            content = "A"
        self._send({"id": "mock", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": content}}]})

    def log_message(self, *args):  # silence
        pass


def self_test(csv_path: str, outdir: str) -> int:
    print("== self-test: dataset ==")
    records, meta = load_records(csv_path)
    print("  path=%s records=%d subjects=%d sha256=%s"
          % (meta["path"], meta["records_eligible"], meta["subjects"],
             meta["sha256"][:16] + "..."))

    print("== self-test: stratified sampling determinism ==")
    a, ma = sample_stratified(records, 40, 20261117)
    b, mb = sample_stratified(records, 40, 20261117)
    assert [r["key"] for r in a] == [r["key"] for r in b], "stratified not deterministic"
    assert len(a) == 40
    assert sum(ma["allocation"].values()) == 40
    print("  ok: 40 items, allocation sums to 40, deterministic")

    print("== self-test: legacy sampling determinism ==")
    la, mla = sample_legacy(records, 40, LEGACY_SEED)
    lb, _ = sample_legacy(records, 40, LEGACY_SEED)
    assert [r["key"] for r in la] == [r["key"] for r in lb]
    print("  ok: 40 items, deterministic (numpy_used=%s)" % mla["numpy_used"])

    print("== self-test: HTTP + parse + CSV/JSON against in-process mock ==")
    host, port = "127.0.0.1", 0
    srv = ThreadingHTTPServer((host, port), _MockHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base_url = "http://%s:%d/v1" % (host, port)
    items = a[:20]
    rows, summary = eval_arm(items, base_url=base_url, model="mock-correct",
                             temperature=0.0, max_tokens=8, timeout=10,
                             retries=0, max_errors=100, progress=False,
                             arm_label="mock")
    payload = {
        "schema": TOOL_VERSION, "tool": "tools/eval_mmlu_protocol.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "self-test", "label": "self-test",
        "dataset": {k: v for k, v in meta.items() if k != "subject_sizes"},
        "sampling": ma, "endpoint": {"base_url": base_url, "model": "mock-correct"},
        "summary": summary, "items": rows,
    }
    paths = write_run(outdir, "selftest", "stratified", payload, rows)
    srv.shutdown()
    assert os.path.isfile(paths["json"]) and os.path.isfile(paths["csv"])
    assert summary["n"] == 20 and summary["errors"] == 0
    assert 0.0 <= summary["score"] <= 1.0
    print("  ok: json=%s csv=%s score=%.3f"
          % (paths["json"], paths["csv"], summary["score"]))

    print("== self-test: paired McNemar + bootstrap on synthetic counts ==")
    rows_a, rows_b = [], []
    b_count, c_count, both_ok, both_bad = 30, 10, 300, 60
    pairs = ([(True, False)] * b_count +      # A-only correct -> b
             [(False, True)] * c_count +      # B-only correct -> c
             [(True, True)] * both_ok +
             [(False, False)] * both_bad)
    for i, (ok_a, ok_b) in enumerate(pairs):
        base = {"key": "syn-%04d" % i, "subject": "syn", "gold": "A"}
        rows_a.append(dict(base, correct=ok_a))
        rows_b.append(dict(base, correct=ok_b))
    analysis = paired_bootstrap_delta(rows_a, rows_b, resamples=1000, seed=7)
    assert analysis["b_a_only"] == b_count, analysis["b_a_only"]
    assert analysis["c_b_only"] == c_count, analysis["c_b_only"]
    assert abs(analysis["delta_pp"] - 5.0) < 1e-9, analysis["delta_pp"]
    assert analysis["mcnemar_exact_p_two_sided"] < 0.01
    assert analysis["bootstrap_ci95_pp"][0] > 0
    print("  ok: delta=%.2fpp p=%.2e CI=%s"
          % (analysis["delta_pp"], analysis["mcnemar_exact_p_two_sided"],
             analysis["bootstrap_ci95_pp"]))

    print("== self-test: paired --interleave end-to-end against mock ==")
    srv2 = ThreadingHTTPServer((host, 0), _MockHandler)
    port2 = srv2.server_address[1]
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    base2 = "http://%s:%d/v1" % (host, port2)
    rc = main([
        "--mode", "paired", "--n", "20", "--seed", "1",
        "--base-url-a", base2, "--model-a", "mock-correct",
        "--base-url-b", base2, "--model-b", "mock-flip",
        "--interleave", "--out", outdir, "--tag", "selftest-paired",
        "--max-errors", "100",
    ])
    srv2.shutdown()
    assert rc == 0, "paired interleave self-test returned %s" % rc
    print("  ok: paired interleave path wrote outputs")

    print("== self-test: power table sanity ==")
    rows = power_table()
    n40 = rows[0]
    assert n40["single_arm_one_sided_pp"] is not None
    assert 10.0 <= n40["single_arm_one_sided_pp"] <= 25.0, n40
    assert rows[1]["paired_mde_pp_discordance_20"] is not None
    print("  ok: n=40 one-sided MDE=%.1fpp, n=400 two-sided MDE=%.1fpp"
          % (n40["single_arm_one_sided_pp"],
             rows[2]["single_arm_two_sided_pp"]))
    print("ALL SELF-TESTS PASSED")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MMLU evaluation protocol harness (M8): legacy sentinel, "
                    "stratified primary, paired McNemar/bootstrap.")
    ap.add_argument("--mode", choices=["legacy", "stratified", "paired"],
                    default="stratified")
    ap.add_argument("--n", type=int, default=None,
                    help="sample size (default: 40 legacy / 400 stratified|paired)")
    ap.add_argument("--seed", type=int, default=None,
                    help="sampling seed (default: 20261108 legacy / 20261117 others)")
    ap.add_argument("--csv", default=None, help="path to full MMLU csv")
    ap.add_argument("--base-url", default="http://127.0.0.1:8082/v1",
                    help="OpenAI-compatible endpoint for single-arm modes")
    ap.add_argument("--base-url-a", default=None, help="paired mode: arm A endpoint")
    ap.add_argument("--base-url-b", default=None, help="paired mode: arm B endpoint")
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-a", default=None)
    ap.add_argument("--model-b", default=None)
    ap.add_argument("--a-json", default=None, help="paired mode: saved run A json")
    ap.add_argument("--b-json", default=None, help="paired mode: saved run B json")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-errors", type=int, default=10)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--interleave", action="store_true",
                    help="paired: alternate A/B order per item (reduces drift bias)")
    ap.add_argument("--out", default=os.path.join("build", "tmp", "mmlu_protocol"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--power-table", action="store_true",
                    help="print the power analysis table and exit (no network)")
    ap.add_argument("--self-test", action="store_true",
                    help="run in-process mock tests (no real service touched)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = os.path.abspath(args.out)
    root = project_root_from_tool()
    os.chdir(root) if os.path.abspath(os.getcwd()) != root else None

    if args.power_table:
        print_power_table()
        return 0

    if args.self_test:
        csv_path = find_csv(args.csv)
        return self_test(csv_path, outdir)

    saved_paired = (args.mode == "paired" and bool(args.a_json and args.b_json))
    if saved_paired:
        # Truly offline: dataset/sampling come from the saved run JSONs, so the
        # CSV does not even need to exist for the analysis to run.
        records, dataset_meta = [], {}
        print("paired(saved): offline analysis (no dataset/sampling needed up front)")
    else:
        csv_path = find_csv(args.csv)
        records, dataset_meta = load_records(csv_path)
        # keep subject_sizes out of the JSON payload (large, not needed per run)
        dataset_meta = {k: v for k, v in dataset_meta.items() if k != "subject_sizes"}
        print("dataset: %s" % dataset_meta["path"])
        print("  records_eligible=%d subjects=%d sha256=%s"
              % (dataset_meta["records_eligible"], dataset_meta["subjects"],
                 dataset_meta["sha256"][:16] + "..."))

    mode = args.mode
    n = args.n if args.n is not None else (LEGACY_N if mode == "legacy" else 400)
    seed = args.seed if args.seed is not None else (
        LEGACY_SEED if mode == "legacy" else 20261117)
    dataset = dataset_meta
    if saved_paired:
        # Offline analysis: sampling metadata comes from the saved runs; never
        # resample (a resample would not be the items that were actually run).
        sampling = None
        items: list[Record] = []
    elif mode == "legacy":
        items, sampling = sample_legacy(records, n, seed)
    else:
        items, sampling = sample_stratified(records, n, seed)
    if not saved_paired:
        print("sampling: %s" % sampling["method"])
        print("  n=%d seed=%d" % (n, seed))

    common = {
        "schema": TOOL_VERSION,
        "tool": "tools/eval_mmlu_protocol.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "dataset": dataset_meta,
        "sampling": sampling,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
    }

    if mode in ("legacy", "stratified"):
        base_url = args.base_url
        if "127.0.0.1:8082" in base_url or "localhost:8082" in base_url:
            print("WARNING: --base-url targets the shared llama-server on 8082 "
                  "(same endpoint the live supervisor uses). This run adds slot "
                  "contention but does NOT touch the gateway or compile anything. "
                  "Use a sidecar port for scored comparisons.", file=sys.stderr)
        model = resolve_model(base_url, args.model, args.timeout)
        print("endpoint: %s model=%s temperature=%s max_tokens=%d"
              % (base_url, model, args.temperature, args.max_tokens))
        rows, summary = eval_arm(items, base_url=base_url, model=model,
                                 temperature=args.temperature,
                                 max_tokens=args.max_tokens,
                                 timeout=args.timeout, retries=args.retries,
                                 max_errors=args.max_errors, progress=True,
                                 arm_label=mode)
        payload = dict(common)
        payload.update({
            "mode": mode,
            "label": args.tag or mode,
            "endpoint": {"base_url": base_url, "model": model,
                         "temperature": args.temperature,
                         "max_tokens": args.max_tokens},
            "summary": summary,
            "items": rows,
        })
        paths = write_run(outdir, args.tag or mode, mode, payload, rows)
        print()
        print("RESULT %s: %d/%d = %.4f  wilson95=[%.4f, %.4f]  errors=%d  %.1fs"
              % (mode, summary["correct"], summary["n"], summary["score"],
                 summary["wilson_ci95"][0], summary["wilson_ci95"][1],
                 summary["errors"], summary["elapsed_s"]))
        print("JSON: %s" % paths["json"])
        print("CSV : %s" % paths["csv"])
        return 0

    # ---------------------------- paired mode ----------------------------
    have_saved = bool(args.a_json and args.b_json)
    have_endpoints = bool(args.base_url_a and args.base_url_b)
    if not (have_saved or have_endpoints):
        die("paired mode needs --a-json/--b-json OR --base-url-a/--base-url-b")
    if have_saved and have_endpoints:
        die("paired mode: give either saved JSONs or endpoints, not both")

    if have_saved:
        run_a = load_run(args.a_json)
        run_b = load_run(args.b_json)
        rows_a, rows_b = run_a["items"], run_b["items"]
        arms = {"a": {"json": os.path.abspath(args.a_json),
                      "summary": run_a.get("summary"),
                      "endpoint": run_a.get("endpoint")},
                "b": {"json": os.path.abspath(args.b_json),
                      "summary": run_b.get("summary"),
                      "endpoint": run_b.get("endpoint")}}
        sampling = run_a.get("sampling") or sampling
        dataset = run_a.get("dataset") or dataset_meta
        print("paired (saved): A=%s (%d items) B=%s (%d items)"
              % (args.a_json, len(rows_a), args.b_json, len(rows_b)))
    else:
        model_a = resolve_model(args.base_url_a, args.model_a, args.timeout)
        model_b = resolve_model(args.base_url_b, args.model_b, args.timeout)
        print("endpoint A: %s model=%s" % (args.base_url_a, model_a))
        print("endpoint B: %s model=%s" % (args.base_url_b, model_b))
        if args.interleave:
            rng = random.Random(seed)
            order = ["a", "b"] * (n // 2) + (["a"] if n % 2 else [])
            rng.shuffle(order)
            rows_a_map: dict[str, dict] = {}
            rows_b_map: dict[str, dict] = {}
            for rec, first in zip(items, order):
                calls = [(first, rec)]
                second = "b" if first == "a" else "a"
                calls.append((second, rec))
                for arm, r2 in calls:
                    base_url = args.base_url_a if arm == "a" else args.base_url_b
                    model = model_a if arm == "a" else model_b
                    raw = ""
                    err = ""
                    t1 = time.time()
                    try:
                        raw = call_chat(base_url, model, make_prompt(r2),
                                        temperature=args.temperature,
                                        max_tokens=args.max_tokens,
                                        timeout=args.timeout)
                    except Exception as exc:  # noqa: BLE001
                        err = "%s: %s" % (type(exc).__name__, exc)
                    lat = (time.time() - t1) * 1000.0
                    row = {
                        "key": r2["key"], "qsha": r2["qsha"],
                        "subject": r2["subject"], "gold": r2["gold"],
                        "pred_legacy": parse_pred_legacy(raw) if not err else None,
                        "pred_strict": parse_pred_strict(raw) if not err else None,
                        "correct": (parse_pred_legacy(raw) == r2["gold"]) if not err else False,
                        "raw": raw[:200], "latency_ms": round(lat, 1),
                        "error": err, "arm": arm,
                    }
                    (rows_a_map if arm == "a" else rows_b_map)[r2["key"]] = row
            rows_a = [rows_a_map[r["key"]] for r in items]
            rows_b = [rows_b_map[r["key"]] for r in items]
            print("  interleaved %d items x 2 arms" % n)
        else:
            rows_a, _ = eval_arm(items, base_url=args.base_url_a, model=model_a,
                                 temperature=args.temperature,
                                 max_tokens=args.max_tokens, timeout=args.timeout,
                                 retries=args.retries, max_errors=args.max_errors,
                                 progress=True, arm_label="A")
            rows_b, _ = eval_arm(items, base_url=args.base_url_b, model=model_b,
                                 temperature=args.temperature,
                                 max_tokens=args.max_tokens, timeout=args.timeout,
                                 retries=args.retries, max_errors=args.max_errors,
                                 progress=True, arm_label="B")
        arms = {"a": {"base_url": args.base_url_a, "model": model_a},
                "b": {"base_url": args.base_url_b, "model": model_b}}

    analysis = paired_bootstrap_delta(rows_a, rows_b,
                                      resamples=args.resamples, seed=seed)
    discordant = []
    for ra, rb in zip(rows_a, rows_b):
        if ra["correct"] == rb["correct"]:
            continue
        discordant.append({
            "key": ra["key"], "subject": ra["subject"], "gold": ra["gold"],
            "pred_a": ra["pred_legacy"], "correct_a": ra["correct"],
            "pred_b": rb["pred_legacy"], "correct_b": rb["correct"],
            "type": "b(a-only-correct)" if ra["correct"] else "c(b-only-correct)",
        })
    payload = dict(common)
    payload.update({
        "mode": "paired",
        "label": args.tag or "paired",
        "sampling": sampling,
        "dataset": dataset,
        "arms": arms,
        "analysis": analysis,
        "discordant_items": discordant,
        "items_a": rows_a,
        "items_b": rows_b,
    })
    paths = write_paired(outdir, args.tag or "paired", payload, discordant)
    print()
    print("PAIRED RESULT: n=%d  A=%d/%d  B=%d/%d  delta=%.2fpp  "
          "McNemar exact p=%.3g  bootstrap95=[%.2f, %.2f]pp"
          % (analysis["n_pairs"], analysis["a_correct"], analysis["n_pairs"],
             analysis["b_correct"], analysis["n_pairs"], analysis["delta_pp"],
             analysis["mcnemar_exact_p_two_sided"],
             analysis["bootstrap_ci95_pp"][0], analysis["bootstrap_ci95_pp"][1]))
    print("discordant pairs: b(A only)=%d c(B only)=%d"
          % (analysis["b_a_only"], analysis["c_b_only"]))
    print("JSON: %s" % paths["json"])
    print("CSV : %s" % paths["csv"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

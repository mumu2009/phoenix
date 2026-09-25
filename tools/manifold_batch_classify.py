# manifold_batch_classify.py -- classify a dialogue-serial batch into
# memes (stable zero-gradient points) vs anti-memes (replication failures),
# with anti-meme subtypes. Same judgment as the seedbudget10 analysis:
#   meme      = chain never stopped early AND alphaSame on every judge round
#               (r >= JudgeAfter) -- the 17e6cec4 pattern (6/6).
#   anti-meme = anything that stopped early or failed a judge round:
#     no-carrier      screen/compose stage: no corpus sentence carrier at all
#                     (field prior = 0, unignitable; skipped by the serial)
#     ignition-fail   r1 inPresent=false: seed carrier lost the id before the
#                     first generation
#     saddle          replicated on >=1 judge round, then failed a later judge
#                     round (the 6d698abe / 2e177b09 pattern: perfect-copy
#                     point next to the collapse basin)
#     collapse        stopped by the collapse interceptor (token loop /
#                     single-token dominance, no healthy recompose carrier)
#     decay           failed the first judge round without prior replication
#     dirty           empty/exam-dump output stopped the chain
# Also recomputes the manifold field factors per group and the spearman
# rho(S, survival rounds) over every group with rounds, so the field formula
# can be re-validated on the enlarged sample.
#
# Usage: python tools/manifold_batch_classify.py [dialogue.json ...]
# Default: runtime_store/screenbatch32_dialogue.json
# Output: runtime_store/manifold_batch_classify.json + stdout table.
import json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runtime_store" / "manifold_batch_classify.json"

TOK = re.compile(r"[a-z]+")
JUDGE_AFTER = 3
ROUNDS = 6


def text_stats(text: str):
    toks = TOK.findall(text.lower())
    n = len(toks)
    if n == 0:
        return {"tokens": 0, "entropy": 0.0, "top1Frac": 0.0, "maxRun": 0}
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    h = -sum((c / n) * math.log2(c / n) for c in freq.values())
    top1 = max(freq.values()) / n
    run = 1
    best = 1
    for i in range(1, n):
        if toks[i] == toks[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return {"tokens": n, "entropy": round(h, 4), "top1Frac": round(top1, 4),
            "maxRun": best}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0.0


def classify_group(g):
    """Return (verdict, subtype, detail) for one serial group."""
    if g.get("skipped"):
        return ("antimeme", "no-carrier", str(g.get("skipped")))
    rounds = g.get("rounds") or []
    if not rounds:
        return ("antimeme", "no-rounds", "no rounds recorded")
    r1 = rounds[0]
    det_in = r1.get("detectIn") or {}
    if not det_in.get("present", False):
        return ("antimeme", "ignition-fail", "r1 carrier lost id pre-generate")
    last = rounds[-1]
    if last.get("dirty") or last.get("exam"):
        return ("antimeme", "dirty", "empty/exam-dump output")
    if last.get("collapsed") and last.get("intercepted"):
        return ("antimeme", "collapse",
                "maxRun=%s top1=%s" % ((last.get("wipe") or {}).get("outMaxRun"),
                                       (last.get("wipe") or {}).get("outTop1Frac")))
    judge_rounds = [r for r in rounds if r.get("judge")]
    judge_same = [bool((r.get("wipe") or {}).get("alphaSame"))
                  for r in judge_rounds]
    replicated_before_death = any(judge_same)
    completed = len(rounds) >= ROUNDS and all(judge_same)
    if completed:
        return ("meme", "stable",
                "rounds=%d alphaSame=%d" % (len(rounds), sum(judge_same)))
    if replicated_before_death:
        return ("antimeme", "saddle",
                "judgeSame=%s rounds=%d" % (judge_same, len(rounds)))
    return ("antimeme", "decay", "judgeSame=%s rounds=%d"
            % (judge_same, len(rounds)))


def group_factors(g):
    rounds = g.get("rounds") or []
    w1 = (rounds[0].get("wipe") or {}) if rounds else {}
    typ_need = w1.get("typicalNeed", 0)
    r1cos = w1.get("alphaCos", 0.0)
    stats = [text_stats(r.get("output", "")) for r in rounds]
    redundancy = min(typ_need, 3) / 3.0
    ignition = 1.0 if r1cos > 0 else 0.0
    buffer_ = max(0.0, 1.0 - abs(r1cos - math.sqrt(0.5)))
    entropy = min(1.0, (stats[0]["entropy"] / 8.0) if stats else 0.0)
    s = redundancy * ignition * buffer_ * entropy
    return {
        "typNeed": typ_need, "r1cos": round(r1cos, 4),
        "r1Entropy": stats[0]["entropy"] if stats else 0.0,
        "redundancy": round(redundancy, 4), "ignition": ignition,
        "buffer": round(buffer_, 4), "entropyFactor": round(entropy, 4),
        "fieldScore": round(s, 4),
    }


def main():
    srcs = [Path(a) for a in sys.argv[1:]] or [
        ROOT / "runtime_store" / "screenbatch32_dialogue.json"]
    rows = []
    for src in srcs:
        if not src.exists():
            print(f"skip missing {src}")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        for g in data.get("groups", []):
            verdict, subtype, detail = classify_group(g)
            f = group_factors(g)
            comp = g.get("compose") or {}
            rows.append({
                "id": g["id"], "pole": g.get("pole"),
                "significance": g.get("significance"),
                "source": src.name,
                "verdict": verdict, "subtype": subtype, "detail": detail,
                "rounds": len(g.get("rounds") or []),
                "alphaSameRounds": g.get("alphaSameRounds", 0),
                "composeExclusive": comp.get("composeExclusive", True),
                "screenFieldScore": g.get("fieldScore"),
                **f,
            })
    memes = [r for r in rows if r["verdict"] == "meme"]
    antis = [r for r in rows if r["verdict"] == "antimeme"]
    by_sub = {}
    for r in antis:
        by_sub.setdefault(r["subtype"], []).append(r["id"])
    scored = [r for r in rows if r["rounds"] > 0]
    rho = spearman([r["fieldScore"] for r in scored],
                   [r["rounds"] for r in scored]) if len(scored) > 2 else 0.0
    report = {
        "sources": [s.name for s in srcs],
        "judgeAfter": JUDGE_AFTER, "rounds": ROUNDS,
        "memeCount": len(memes), "antimemeCount": len(antis),
        "antimemeSubtypes": {k: len(v) for k, v in sorted(by_sub.items())},
        "spearmanFieldVsSurvivalRounds": round(rho, 4),
        "memes": [r["id"] for r in memes],
        "antimemes": [{k: r[k] for k in ("id", "subtype", "detail")}
                      for r in antis],
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"memes={len(memes)} antimemes={len(antis)} "
          f"subtypes={report['antimemeSubtypes']} spearman={rho:.4f}")
    print(f"{'id':44} {'pole':6} {'verdict':9} {'subtype':13} {'need':>4} "
          f"{'r1cos':>6} {'r1H':>5} {'S':>6} {'rounds':>6}")
    for r in sorted(rows, key=lambda r: (r["verdict"] != "meme",
                                         -r["fieldScore"])):
        print(f"{r['id']:44} {str(r['pole']):6} {r['verdict']:9} "
              f"{r['subtype']:13} {r['typNeed']:>4} {r['r1cos']:>6.3f} "
              f"{r['r1Entropy']:>5.2f} {r['fieldScore']:>6.3f} "
              f"{r['rounds']:>6}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# manifold_field_analysis.py -- verify the manifold/field hypothesis on the
# dialogue serial experiment: a meme candidate survives iff it sits at a
# STABLE zero-gradient point of the carrier-continuation field.
#
# Factors per meme:
#   redundancy = min(typNeed,3)/3      (typical-set size, degenerate=1/3)
#   ignition   = 1 if r1 alphaCos > 0  (first reingest hit the typical set)
#   buffer     = 1 - |r1cos - 1/sqrt(2)|  (distance to the half-hit attractor;
#               perfect 1.0 copies are saddle points, not attractors)
#   entropy    = r1 output unigram entropy / 8 bits, clamped to [0,1]
#                (verbatim-repeat carriers are entropically adjacent to the
#                token-collapse basin: "aalborg aalborg ...")
#   S = redundancy * ignition * buffer * entropy
#
# Output: runtime_store/manifold_field_analysis.json + stdout table.
import json, math, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runtime_store" / "seedbudget10_dialogue.json"
OUT = ROOT / "runtime_store" / "manifold_field_analysis.json"

TOK = re.compile(r"[a-z]+")

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

def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = []
    for g in data["groups"]:
        mid = g["id"].replace("meme_p_", "")[:8]
        rounds = g["rounds"]
        if not rounds:
            continue
        w1 = rounds[0].get("wipe") or {}
        typ_need = w1.get("typicalNeed", 0)
        r1cos = w1.get("alphaCos", 0.0)
        stats = [text_stats(r.get("output", "")) for r in rounds]
        cos_seq = [(r.get("wipe") or {}).get("alphaCos", 0.0) for r in rounds]
        live_seq = [(r.get("wipe") or {}).get("liveUnits", 0) for r in rounds]
        redundancy = min(typ_need, 3) / 3.0
        ignition = 1.0 if r1cos > 0 else 0.0
        buffer_ = max(0.0, 1.0 - abs(r1cos - math.sqrt(0.5)))
        entropy = min(1.0, (stats[0]["entropy"] / 8.0) if stats else 0.0)
        s = redundancy * ignition * buffer_ * entropy
        # collapse detector: token loop or single-token dominance
        collapse_rounds = [i + 1 for i, st in enumerate(stats)
                           if st["maxRun"] >= 8 or st["top1Frac"] >= 0.5]
        rows.append({
            "id": mid, "pole": g["pole"], "rounds": len(rounds),
            "alphaSameRounds": g.get("alphaSameRounds", 0),
            "typNeed": typ_need, "r1cos": round(r1cos, 4),
            "cosSeq": [round(c, 4) for c in cos_seq],
            "liveUnitsSeq": live_seq,
            "r1Entropy": stats[0]["entropy"] if stats else 0.0,
            "r1Top1Frac": stats[0]["top1Frac"] if stats else 0.0,
            "r1MaxRun": stats[0]["maxRun"] if stats else 0,
            "collapseRounds": collapse_rounds,
            "redundancy": round(redundancy, 4),
            "ignition": ignition,
            "buffer": round(buffer_, 4),
            "entropyFactor": round(entropy, 4),
            "fieldScore": round(s, 4),
        })
    rows.sort(key=lambda r: -r["fieldScore"])
    svals = [r["fieldScore"] for r in rows]
    surv = [r["rounds"] for r in rows]
    rho = spearman(svals, surv)
    report = {
        "source": SRC.name,
        "hypothesis": ("meme candidates live on a manifold; true memes sit at "
                       "STABLE zero-gradient field points (half-hit attractor "
                       "cos=1/sqrt(2)), perfect-copy points (cos=1, typNeed=1) "
                       "are saddles adjacent to the token-collapse basin"),
        "spearmanFieldVsSurvivalRounds": round(rho, 4),
        "memes": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"spearman(fieldScore, survivalRounds) = {rho:.4f}")
    print(f"{'id':10} {'pole':6} {'need':>4} {'r1cos':>6} {'r1H':>6} "
          f"{'run':>4} {'coll':>10} {'S':>7} {'rounds':>6}")
    for r in rows:
        print(f"{r['id']:10} {r['pole']:6} {r['typNeed']:>4} "
              f"{r['r1cos']:>6.3f} {r['r1Entropy']:>6.2f} "
              f"{r['r1MaxRun']:>4} {str(r['collapseRounds']):>10} "
              f"{r['fieldScore']:>7.3f} {r['rounds']:>6}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    sys.exit(main())

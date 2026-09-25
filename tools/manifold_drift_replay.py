# manifold_drift_replay.py -- replay every recorded dialogue-serial output
# through the enhanced wipe-ingest instrument (no llama needed) and extract
# word-level typical-set drift: which identity words each round kept (hit)
# vs dropped (miss), plus collapse health. Focus: the round-5 decay point.
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "meme_barrier_instrument.exe"
SNAP = ROOT / "runtime_store" / "gnn_instrument_merge.txt"
SRC = ROOT / "runtime_store" / "seedbudget10_dialogue.json"
OUT = ROOT / "runtime_store" / "manifold_drift_replay.json"
FOCUS = ("6d698abe", "2e177b09", "17e6cec4", "ad748b03")

def wipe(meme_id: str, text: str):
    p = subprocess.run(
        [str(EXE), "wipe-ingest", "--snap", str(SNAP), "--units", "256",
         "--meme", meme_id],
        input=text, capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=ROOT)
    if p.returncode != 0:
        return {"error": p.stderr.strip()[:200]}
    return json.loads(p.stdout)

def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    report = {}
    for g in data["groups"]:
        short = g["id"].replace("meme_p_", "")[:8]
        if short not in FOCUS:
            continue
        rounds = []
        for r in g["rounds"]:
            out = r.get("output", "")
            if not out:
                continue
            w = wipe(g["id"], out)
            rounds.append({
                "round": r["round"],
                "alphaCos": w.get("alphaCos"),
                "alphaSame": w.get("alphaSame"),
                "typicalFrozen": w.get("typicalFrozen"),
                "hit": w.get("typicalHitWords"),
                "miss": w.get("typicalMissWords"),
                "entropy": w.get("outEntropy"),
                "maxRun": w.get("outMaxRun"),
                "collapsed": w.get("collapsed"),
                "decayWarn": w.get("decayWarn"),
                "liveUnits": w.get("liveUnits"),
            })
        report[short] = {"pole": g["pole"], "rounds": rounds}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    for short, d in report.items():
        print(f"=== {short} ({d['pole']}) ===")
        for r in d["rounds"]:
            print(f"  r{r['round']}: cos={r['alphaCos']:.3f} "
                  f"hit={r['hit']} miss={r['miss']} "
                  f"H={r['entropy']:.2f} run={r['maxRun']} "
                  f"coll={r['collapsed']} warn={r['decayWarn']} "
                  f"live={r['liveUnits']}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    sys.exit(main())

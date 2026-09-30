# manifold_export_candidates.py -- 把 screen 筛出的模因候选导出成
# 远程串行包 candidates.json：在本地用冻结仪器预合成载体（compose，
# requireExclusive 优先、失败回退非独占，与 ps1 的 Get-Compose 一致），
# 并用冻结图 detect 验证载体呈现。远程机器因此不需要语料库，只需要
# snap + candidates.json + meme_remote_serial.exe。
#
# Usage:
#   python tools/manifold_export_candidates.py \
#       --screen runtime_store/farm_screen_1792.json \
#       --snap   runtime_store/farm_instrument_1792.txt \
#       --out    runtime_store/remote_candidates_1792.json \
#       [--corpus robots/wikitext-103-all.txt] [--units 256] \
#       [--take 40] [--only id1,id2] [--skip-file serial.json ...]
#
# --skip-file 可重复：串行结果 json（读 groups[].id）或每行一个 id 的
# 文本；已在本地测过的 id 不再导出，避免与农场重复。
import argparse, json, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / "meme_barrier_instrument.exe"

DETECT_RE = re.compile(
    r"present=(\d) activation=([0-9eE.+-]+) peak=([0-9eE.+-]+) "
    r"rank=(-?\d+) lift=([0-9eE.+-]+) null=([0-9eE.+-]+)")


def load_skip_ids(files):
    ids = set()
    for f in files or []:
        p = Path(f)
        if not p.exists():
            print(f"skip-file missing: {p}", file=sys.stderr)
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix.lower() == ".json":
            try:
                data = json.loads(txt)
                for g in data.get("groups", []):
                    if g.get("id"):
                        ids.add(g["id"])
                for m in data.get("memes", []):
                    if isinstance(m, dict) and m.get("id"):
                        ids.add(m["id"])
                    elif isinstance(m, str):
                        ids.add(m)
                continue
            except Exception:
                pass
        for line in txt.splitlines():
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def run_compose(snap, corpus, units, meme_id, require_exclusive):
    cmd = [str(EXE), "compose", "--snap", str(snap), "--corpus",
           str(corpus), "--units", str(units), "--meme", meme_id]
    if require_exclusive:
        cmd.append("--require-exclusive")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode == 4:
        return None, 4  # 无（独占）载体
    if p.returncode != 0:
        raise RuntimeError(f"compose exit {p.returncode}: {p.stdout} {p.stderr}")
    return json.loads(p.stdout), 0


def run_detect(snap, meme_id, text):
    p = subprocess.run([str(EXE), "detect", "--snap", str(snap), "--meme",
                        meme_id], input=text, capture_output=True, text=True,
                       encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"detect exit {p.returncode}: {p.stdout}")
    m = DETECT_RE.search(p.stdout)
    if not m:
        raise RuntimeError(f"detect parse failed: {p.stdout}")
    return {"present": m.group(1) == "1", "activation": float(m.group(2)),
            "peak": float(m.group(3)), "rank": int(m.group(4)),
            "lift": float(m.group(5)), "nullScore": float(m.group(6))}


def main():
    global EXE  # --exe 默认值引用它，global 声明必须在函数最前
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True)
    ap.add_argument("--snap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "robots" /
                                            "wikitext-103-all.txt"))
    ap.add_argument("--units", type=int, default=256)
    ap.add_argument("--take", type=int, default=0, help="最多导出 N 个")
    ap.add_argument("--only", default="", help="只导出这些 id（逗号分隔）")
    ap.add_argument("--skip-file", action="append", default=[])
    ap.add_argument("--exe", default=str(EXE),
                    help="instrument 可执行文件路径（RDK X5 等精益构建时"
                         "指到 meme_barrier_instrument 精益版）")
    ap.add_argument("--farm-filter", action="store_true",
                    help="应用农场 build_batch 同款过滤：typicalNeed==2 且 "
                         "carrierEntropy>0 且 typicalCover>0，按组合分 "
                         "z(wordCount)-z(carrierEntropy) 降序（2026-09 远端 "
                         "审计：熵降序方向用反，词多熵低者存活约 2.5x）")
    args = ap.parse_args()

    EXE = Path(args.exe)

    screen_p = Path(args.screen)
    snap_p = Path(args.snap)
    corpus_p = Path(args.corpus)
    if not EXE.exists():
        sys.exit(f"missing {EXE}")
    if not corpus_p.exists():
        alt = ROOT / "robots" / "wikitext-101-all.txt"
        if alt.exists():
            corpus_p = alt
        else:
            sys.exit("missing corpus")
    screen = json.loads(screen_p.read_text(encoding="utf-8"))
    memes = screen.get("memes", [])
    if args.farm_filter:
        memes = [m for m in memes
                 if m.get("typicalNeed") == 2
                 and (m.get("carrierEntropy") or 0) > 0
                 and (m.get("typicalCover") or 0) > 0]
        # 组合分排序（替代旧的 carrierEntropy 降序——审计证明该方向反了）。
        # combo = z(wordCount) - z(carrierEntropy)：两字段无共线，
        # 十分位存活 0.64->1.62 准单调，AUC(>=2臂强模因)=0.65。
        def _z(vals):
            n = len(vals)
            if n == 0:
                return []
            mu = sum(vals) / n
            var = sum((v - mu) ** 2 for v in vals) / n
            sd = var ** 0.5
            if sd < 1e-12:
                return [0.0] * n
            return [(v - mu) / sd for v in vals]
        zw = _z([float(m.get("wordCount") or 0) for m in memes])
        ze = _z([float(m.get("carrierEntropy") or 0) for m in memes])
        for m, w, e in zip(memes, zw, ze):
            m["comboScore"] = w - e
        memes.sort(key=lambda m: -m["comboScore"])
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = load_skip_ids(args.skip_file)

    out_memes, skipped = [], []
    t0 = time.time()
    n_seen = 0
    for m in memes:
        mid = m["id"]
        if only and mid not in only:
            continue
        n_seen += 1
        if mid in skip:
            skipped.append({"id": mid, "reason": "already-tested"})
            continue
        if args.take and len(out_memes) >= args.take:
            break
        comp, rc = run_compose(snap_p, corpus_p, args.units, mid, True)
        excl = True
        if rc == 4:
            comp, rc = run_compose(snap_p, corpus_p, args.units, mid, False)
            excl = False
        if rc == 4 or comp is None:
            skipped.append({"id": mid, "reason": "no-carrier"})
            print(f"  skip {mid}: no corpus carrier")
            continue
        text = comp.get("text", "")
        det = run_detect(snap_p, mid, text)
        el = time.time() - t0
        print(f"  [{len(out_memes)+1}] {mid} present={det['present']} "
              f"lift={det['lift']:.2f} excl={excl} chars={len(text)} "
              f"({el:.0f}s)")
        if not det["present"]:
            skipped.append({"id": mid, "reason": "compose-not-present"})
            continue
        comp["composeExclusive"] = excl
        out_memes.append({
            "id": mid, "pole": m.get("pole"),
            "rankMost": m.get("rankMost"), "rankLeast": m.get("rankLeast"),
            "significance": m.get("significance"),
            "wordCount": m.get("wordCount"),
            # screen 全字段保留（审计用，避免回 join screen 文件）
            "contentN": m.get("contentN"),
            "contentSpec": m.get("contentSpec"),
            "carrierEntropy": m.get("carrierEntropy"),
            "fieldScore": m.get("fieldScore"),
            "selfPeak": m.get("selfPeak"),
            "selfRank": m.get("selfRank"),
            "typicalNeed": m.get("typicalNeed"),
            "typicalCover": m.get("typicalCover"),
            "comboScore": m.get("comboScore"),
            "seedBag": m.get("carrier"),
            "seedRag": text,
            "composeExclusive": excl,
            "compose": comp,
        })

    doc = {
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "kind": "meme_remote_serial candidates v1",
        "snap": str(snap_p), "screen": str(screen_p),
        "corpus": str(corpus_p), "units": args.units,
        "memes": out_memes, "skipped": skipped,
    }
    out_p = Path(args.out)
    out_p.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"exported {len(out_memes)} candidates, skipped {len(skipped)} "
          f"-> {out_p} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

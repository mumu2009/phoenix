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


def run_compose_batch(snap, corpus, units, ids_file, skip=0):
    """一次进程批量 compose（instrument compose-batch，倒排缓存同进程
    生效：N 个模因 = 1 次建索引 + N 次命中查询）。
    返回 {id: compose 结果}；无载体的 id 不在结果里。"""
    cmd = [str(EXE), "compose-batch", "--snap", str(snap), "--corpus",
           str(corpus), "--units", str(units), "--memes-file",
           str(ids_file)]
    if skip > 0:
        # 片内载体搜索：instrument 2026-09-30 起在 --skip>0 时把
        # rawCorpus 切到 [skip, skip+units)
        cmd += ["--skip", str(skip)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"compose-batch exit {p.returncode}: "
                           f"{p.stdout[:500]} {p.stderr[:500]}")
    # strict=False：instrument jsonEscape 未转义 \x00-\x1f 控制字符
    # （语料里的 \x0c 等进入 compose text），宽松解析入口容错；
    # 本脚本写出的 candidates json 由 json.dump 正规转义，下游干净。
    return json.loads(p.stdout, strict=False).get("results", {})


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
    ap.add_argument("--skip", type=int, default=0,
                    help="该片在 corpus 中的起始 unit（透传 compose，"
                         "启用片内载体搜索；0=全语料）")
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
    screen = json.loads(screen_p.read_text(encoding="utf-8"), strict=False)
    memes = screen.get("memes", [])
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = load_skip_ids(args.skip_file)

    # 1) 预过滤：typicalNeed==2（screen 字段，图内计算）。
    #    carrierEntropy/typicalCover 由 compose-batch 实测后补判——
    #    screen 阶段不再逐个 compose（2026-09-30：全池 compose 是
    #    O(pool x 语料)，是 screen 慢的根源）。
    if args.farm_filter:
        memes = [m for m in memes if m.get("typicalNeed") == 2]
    cand = []
    for m in memes:
        mid = m["id"]
        if only and mid not in only:
            continue
        if mid in skip:
            skipped_pre = {"id": mid, "reason": "already-tested"}
            cand.append((m, None, skipped_pre))
            continue
        cand.append((m, None, None))

    # 2) compose-batch 一次进程（倒排缓存：1 次建索引 + N 次命中）
    t0 = time.time()
    todo = [m for m, _, sk in cand if sk is None]
    out_p = Path(args.out)
    ids_file = out_p.with_suffix(".ids.txt")
    ids_file.write_text("\n".join(m["id"] for m in todo), encoding="utf-8")
    comp_map = run_compose_batch(snap_p, corpus_p, args.units, ids_file,
                                 args.skip) if todo else {}
    print(f"compose-batch: {len(comp_map)}/{len(todo)} carriers "
          f"({time.time()-t0:.0f}s)")

    # 3) 过滤：no-carrier / not present / farm: entropy>0 & cover>0
    skipped = [sk for _, _, sk in cand if sk]
    pool = []
    for m, _, sk in cand:
        if sk is not None:
            continue
        mid = m["id"]
        comp = comp_map.get(mid)
        if comp is None:
            skipped.append({"id": mid, "reason": "no-carrier"})
            continue
        if not comp.get("present"):
            skipped.append({"id": mid, "reason": "compose-not-present"})
            continue
        if args.farm_filter and not ((comp.get("carrierEntropy") or 0) > 0
                                     and (comp.get("typicalCover") or 0) > 0):
            skipped.append({"id": mid, "reason": "farm-filter-post"})
            continue
        pool.append((m, comp))

    # 4) 组合分排序 combo_v2 = 0.835*z(log1p(dfMax)) - 0.255*z(carrierEntropy)
    #    （2026-09-30 logistic 学习权重，1922 模因 5-fold CV AUC 0.704，
    #    校准极佳 D1 预测 7.31% 实际 7.29%。机理：枢纽搭接是主效应(OR=2.53)，
    #    载体低熵次之；wordCount 控制 dfMax 后系数转负——词数效应完全是
    #    dfMax 的中介，无独立贡献；交互项/GBDT 均无增益，线性即上限。
    #    旧 screen 无 dfMax 时退化为熵单负向。）
    if args.farm_filter and pool:
        def _z(vals):
            n = len(vals)
            mu = sum(vals) / n
            sd = (sum((v - mu) ** 2 for v in vals) / n) ** 0.5
            return [(v - mu) / sd if sd > 1e-12 else 0.0 for v in vals]
        import math as _math
        ze = _z([float(c.get("carrierEntropy") or 0) for _, c in pool])
        has_df = any(m.get("dfMax") for m, _ in pool)
        zd = _z([_math.log1p(float(m.get("dfMax") or 0)) for m, _ in pool]) if has_df else [0.0] * len(pool)
        for (m, c), e, d in zip(pool, ze, zd):
            m["comboScore"] = 0.835 * d - 0.255 * e
        pool.sort(key=lambda mc: -mc[0]["comboScore"])

    # 5) take + 组装
    out_memes = []
    for m, comp in pool[:args.take if args.take else None]:
        mid = m["id"]
        text = comp.get("text", "")
        excl = bool(comp.get("composeExclusive"))
        print(f"  [{len(out_memes)+1}] {mid} present=True "
              f"lift={comp.get('lift', 0):.2f} excl={excl} chars={len(text)}")
        out_memes.append({
            "id": mid, "pole": m.get("pole"),
            "rankMost": m.get("rankMost"), "rankLeast": m.get("rankLeast"),
            "significance": m.get("significance"),
            "wordCount": m.get("wordCount"),
            # screen 全字段保留（审计用，避免回 join screen 文件）
            "contentN": m.get("contentN"),
            "contentSpec": m.get("contentSpec"),
            "dfMax": m.get("dfMax"),
            "deg": m.get("deg"),
            "top1Frac": m.get("top1Frac"),
            # entropy/cover/fieldScore 用 compose 实测值（screen 已不填）
            "carrierEntropy": comp.get("carrierEntropy"),
            "fieldScore": comp.get("fieldScore"),
            "selfPeak": m.get("selfPeak"),
            "selfRank": m.get("selfRank"),
            "typicalNeed": m.get("typicalNeed"),
            "typicalCover": comp.get("typicalCover"),
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

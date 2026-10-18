# remote_slice_pipeline_v4.py -- v4 未筛选随机候选包管线。
# 与 v2/v3 的根本区别：候选是 screen 全池（--take 100000，most+least
# 两极合取=全池覆盖）里 typicalNeed==2 的**均匀随机抽样**（每片 220，
# 固定 seed 可复现），不按 comboScore 排序截断——comboScore 仍记录，
# 供回收结果后公正计算 AUC（v3 的 range restriction 教训）。
#
# 用法（phoenix/ 下，幂等，可中断重跑）：
#   python tools\remote_slice_pipeline_v4.py --corpus arxiv
#   python tools\remote_slice_pipeline_v4.py --corpus stackexchange
#   python tools\remote_slice_pipeline_v4.py --corpus wikipedia2023
# 每语料 6 片（skip=0..2560 step 512）。多语料并行 = 开多个进程。
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RS = ROOT / "runtime_store"
EXE = ROOT / "build" / "meme_barrier_instrument.exe"
EXPORTER = ROOT / "tools" / "manifold_export_candidates.py"
STATE = RS / "farm_state.json"
SLICES = RS / "remote_package_v4" / "slices"
LOG = RS / "remote_slice_pipeline_v4.log"

CORPORA = {
    "arxiv": ROOT / "robots" / "corpus_arxiv.txt",
    "cnndailymail": ROOT / "robots" / "corpus_cnndailymail.txt",
    "wikipedia2023": ROOT / "robots" / "corpus_wikipedia2023.txt",
}
UNITS_PER_SLICE = 512
SLICES_PER_CORPUS = 6
RANDOM_TAKE = 220       # 每片随机抽样数（compose 淘汰 no-carrier 后 ~120）
SAMPLE_SEED = 20261002  # 固定 seed：抽样可复现
MIN_POOL = 50           # 全池 need==2 少于此数的片不值得做


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, stdout_to=None, timeout_h=5):
    log("run: " + " ".join(str(c) for c in cmd)[:160])
    fh = (open(stdout_to, "w", encoding="utf-8") if stdout_to
          else subprocess.DEVNULL)
    # stderr 独立 .err 文件（2026-10-01 教训：instrument 的
    # [screen-profile] 计时行混入 stdout 会污染 screen json）
    err_fh = (open(str(stdout_to) + ".err", "w", encoding="utf-8")
              if stdout_to else subprocess.DEVNULL)
    try:
        with subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                              stderr=err_fh) as p:
            try:
                rc = p.wait(timeout=int(timeout_h * 3600))
            except subprocess.TimeoutExpired:
                p.kill()
                raise RuntimeError("timeout: %s" % cmd[0])
    finally:
        if fh is not subprocess.DEVNULL:
            fh.close()
        if err_fh is not subprocess.DEVNULL:
            err_fh.close()
    if rc != 0:
        raise RuntimeError("exit %d: %s" % (rc, cmd[0]))


def load_screen(path):
    """严格解析失败时剔除 [screen-profile] 等非 json 尾行重试。
    空文件（另一实例刚创建、正在跑）统一抛 JSONDecodeError。"""
    txt = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(txt, strict=False)
    except json.JSONDecodeError:
        lines = txt.splitlines()
        ends = [i for i, ln in enumerate(lines) if ln.rstrip() == "}"]
        if not ends:
            raise json.JSONDecodeError("empty/no-json screen", txt, 0)
        return json.loads("\n".join(lines[:ends[-1] + 1]), strict=False)


def build_screen(corpus_key, skip):
    corpus = CORPORA[corpus_key]
    snap = RS / ("farm_instrument_v4_%s_%d.txt" % (corpus_key, skip))
    screen = RS / ("farm_screen_v4_%s_%d.json" % (corpus_key, skip))
    if screen.exists() and snap.exists():
        try:
            return snap, load_screen(screen)
        except json.JSONDecodeError:
            # 0 字节且刚被修改 = 另一实例正在跑这片——不要重筛（会撞
            # 写同一文件），直接报错退出本进程
            age = time.time() - screen.stat().st_mtime
            if screen.stat().st_size == 0 and age < 1800:
                raise RuntimeError(
                    "screen %s is being written by another instance "
                    "(0 bytes, %ds old); refusing to double-run"
                    % (screen.name, age))
            log("stale/broken screen %s - re-screening" % screen.name)
    # snap 缓存陷阱（2026-10-01）：screen 对已存在 snap 直接 loadText
    # 跳过 ingest——若 snap 是旧参数建的必须删掉重建
    if snap.exists() and not screen.exists():
        log("warn: snap exists without screen, deleting stale snap %s"
            % snap.name)
        snap.unlink()
    run([str(EXE), "screen", "--corpus", str(corpus), "--snap", str(snap),
         "--units", str(UNITS_PER_SLICE), "--skip", str(skip),
         "--take", "100000", "--pool", "eligible", "--relax"],
        stdout_to=screen)
    return snap, load_screen(screen)


def collect_tested_ids():
    ids = set()
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8"))
        ids.update(st.get("testedIds", []))
    for pat in ("remote_package_v2/slices/candidates_*.json",
                "remote_package_v3/slices/candidates_*.json",
                "remote_package_v4/slices/candidates_*.json"):
        for p in RS.glob(pat):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                ids.update(m["id"] for m in doc.get("memes", []))
            except (json.JSONDecodeError, KeyError):
                log("warn: broken candidates %s, ignored" % p.name)
    return ids


def export_slice(corpus_key, skip, snap, screen, tested_txt):
    out = SLICES / ("candidates_v4_%s_%d.json" % (corpus_key, skip))
    if out.exists():
        log("%s/%d: candidates already exported, reuse" % (corpus_key, skip))
        return out
    pool = [m for m in screen.get("memes", []) if m.get("typicalNeed") == 2]
    if len(pool) < MIN_POOL:
        log("%s/%d: pool too small (%d), skip slice"
            % (corpus_key, skip, len(pool)))
        return None
    rng = random.Random(SAMPLE_SEED + skip)
    take = min(RANDOM_TAKE, len(pool))
    picked = rng.sample(pool, take)
    ids_file = RS / ("_tmp_v4_only_%s_%d.txt" % (corpus_key, skip))
    ids_file.write_text("\n".join(m["id"] for m in picked),
                        encoding="utf-8")
    log("%s/%d: pool=%d need==2, random sampled %d (seed=%d)"
        % (corpus_key, skip, len(pool), take, SAMPLE_SEED + skip))
    run([sys.executable, str(EXPORTER),
         "--screen", str(RS / ("farm_screen_v4_%s_%d.json"
                               % (corpus_key, skip))),
         "--snap", str(snap),
         "--out", str(out),
         "--corpus", str(CORPORA[corpus_key]),
         "--skip", str(skip),
         "--farm-filter", "--take", "0",
         "--only-file", str(ids_file),
         "--skip-file", str(tested_txt)])
    ids_file.unlink(missing_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--slices", type=int, default=SLICES_PER_CORPUS)
    ap.add_argument("--start", type=int, default=0,
                    help="起始片序号（skip = start*512；分链并行时用）")
    args = ap.parse_args()

    if not CORPORA[args.corpus].exists():
        log("ERROR: corpus missing: %s" % CORPORA[args.corpus])
        return 1
    SLICES.mkdir(parents=True, exist_ok=True)
    log("v4 pipeline start: corpus=%s slices=%d start=%d"
        % (args.corpus, args.slices, args.start))

    made = 0
    for i in range(args.start, args.start + args.slices):
        skip = i * UNITS_PER_SLICE
        snap, screen = build_screen(args.corpus, skip)
        units = screen.get("ingestedUnits", 0)
        if units < 200:
            log("corpus exhausted at skip=%d (units=%s); stop"
                % (skip, units))
            break
        tested = collect_tested_ids()
        tested_txt = RS / "remote_v4_tested_ids.txt"
        tested_txt.write_text("\n".join(sorted(tested)), encoding="utf-8")
        out = export_slice(args.corpus, skip, snap, screen, tested_txt)
        if out is None:
            continue
        dst_snap = SLICES / ("snap_v4_%s_%d.txt" % (args.corpus, skip))
        if not dst_snap.exists():
            dst_snap.write_bytes(snap.read_bytes())
        doc = json.loads(out.read_text(encoding="utf-8"))
        log("%s/%d: exported %d candidates"
            % (args.corpus, skip, len(doc.get("memes", []))))
        made += 1

    log("v4 pipeline done: %s, %d slices" % (args.corpus, made))
    return 0


if __name__ == "__main__":
    sys.exit(main())

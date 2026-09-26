# remote_slice_pipeline.py -- 批量生产远端实验切片（screen -> 过滤去重 ->
# 导出候选 -> 拷入 remote_package_v2/slices/）。全程幂等：已存在的产物
# 直接跳过，可随时中断重跑。
#
# 用法（仓库根目录 phoenix/ 下）：
#   python tools\remote_slice_pipeline.py --start-skip 23040 --count 8
#
# 每个切片 skip=K 的产物：
#   runtime_store/farm_instrument_K.txt      冻结图仪器（screen snap）
#   runtime_store/farm_screen_K.json         screen 原始结果
#   runtime_store/remote_package_v2/slices/snap_K.txt + candidates_K.json
#
# 去重来源（每切片导出前重新收集）：
#   farm_state.json 的 testedIds + 已导出的一切 candidates_*.json 的 id
#   （模因 id 是内容寻址的，跨仪器会撞 id，必须跨切片去重）
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RS = ROOT / "runtime_store"
EXE = ROOT / "build" / "meme_barrier_instrument.exe"
EXPORTER = ROOT / "tools" / "manifold_export_candidates.py"
CORPUS = ROOT / "robots" / "wikitext-103-all.txt"
STATE = RS / "farm_state.json"
SLICES = RS / "remote_package_v2" / "slices"
LOG = RS / "remote_slice_pipeline.log"

UNITS_PER_INSTRUMENT = 512
MIN_SLICE_UNITS = 200   # 尾部小切片 = 语料耗尽
MIN_CANDIDATES = 20     # 候选太少的切片不值得占一个远端进程槽
TAKE_PER_SLICE = 65


def log(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd, stdout_to=None, timeout_h=4):
    log("run: " + " ".join(str(c) for c in cmd)[:160])
    fh = (open(stdout_to, "w", encoding="utf-8") if stdout_to
          else subprocess.DEVNULL)
    try:
        with subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh,
                              stderr=subprocess.STDOUT) as p:
            try:
                rc = p.wait(timeout=int(timeout_h * 3600))
            except subprocess.TimeoutExpired:
                p.kill()
                raise RuntimeError("timeout: %s" % cmd[0])
    finally:
        if fh is not subprocess.DEVNULL:
            fh.close()
    if rc != 0:
        raise RuntimeError("exit %d: %s" % (rc, cmd[0]))


def collect_tested_ids():
    """farm_state testedIds + 已导出 candidates 的 id（跨切片去重）。"""
    ids = set()
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8"))
        ids.update(st.get("testedIds", []))
    if SLICES.exists():
        for p in SLICES.glob("candidates_*.json"):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
                ids.update(m["id"] for m in doc.get("memes", []))
            except (json.JSONDecodeError, KeyError):
                log("warn: broken candidates file %s, ignored" % p.name)
    return ids


def build_instrument(skip):
    snap = RS / ("farm_instrument_%d.txt" % skip)
    screen = RS / ("farm_screen_%d.json" % skip)
    if screen.exists() and snap.exists():
        try:
            return snap, json.loads(screen.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("stale/broken screen %s - re-screening" % screen.name)
    run([str(EXE), "screen", "--corpus", str(CORPUS), "--snap", str(snap),
         "--units", str(UNITS_PER_INSTRUMENT), "--skip", str(skip),
         "--take", "400", "--pool", "eligible", "--relax"],
        stdout_to=screen)
    return snap, json.loads(screen.read_text(encoding="utf-8"))


def export_slice(skip, snap, screen, tested_txt):
    out = SLICES / ("candidates_%d.json" % skip)
    if out.exists():
        log("skip=%d: candidates already exported, reuse" % skip)
        return out
    run([sys.executable, str(EXPORTER),
         "--screen", str(RS / ("farm_screen_%d.json" % skip)),
         "--snap", str(snap),
         "--out", str(out),
         "--farm-filter", "--take", str(TAKE_PER_SLICE),
         "--skip-file", str(tested_txt)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-skip", type=int, required=True)
    ap.add_argument("--count", type=int, default=4)
    args = ap.parse_args()

    SLICES.mkdir(parents=True, exist_ok=True)
    log("pipeline start: skip=%d count=%d" % (args.start_skip, args.count))

    made = 0
    for i in range(args.count):
        skip = args.start_skip + i * UNITS_PER_INSTRUMENT
        snap, screen = build_instrument(skip)
        units = screen.get("ingestedUnits", 0)
        if units < MIN_SLICE_UNITS:
            log("corpus exhausted at skip=%d (units=%s); stop"
                % (skip, units))
            break
        # 农场同款过滤，先看这个切片还剩多少干净候选
        cands = [m for m in screen.get("memes", [])
                 if m.get("typicalNeed") == 2
                 and (m.get("carrierEntropy") or 0) > 0
                 and (m.get("typicalCover") or 0) > 0]
        log("skip=%d: %d clean need=2 candidates in screen" % (skip, len(cands)))
        if len(cands) < MIN_CANDIDATES:
            log("skip=%d: too few candidates (%d), skip slice"
                % (skip, len(cands)))
            continue
        # 导出前重新收集已测 id（farm_state 可能刚合并过新结果）
        tested = collect_tested_ids()
        tested_txt = RS / "remote_slice_tested_ids.txt"
        tested_txt.write_text("\n".join(sorted(tested)), encoding="utf-8")
        export_slice(skip, snap, screen, tested_txt)
        # snap 拷入切片目录（远端包自包含）
        dst_snap = SLICES / ("snap_%d.txt" % skip)
        if not dst_snap.exists():
            dst_snap.write_bytes(snap.read_bytes())
        doc = json.loads((SLICES / ("candidates_%d.json" % skip))
                         .read_text(encoding="utf-8"))
        log("skip=%d: exported %d candidates" % (skip, len(doc.get("memes", []))))
        made += 1

    log("pipeline done: %d slices ready in %s" % (made, SLICES))
    return 0


if __name__ == "__main__":
    sys.exit(main())

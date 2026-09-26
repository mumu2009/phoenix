# remote_merge_results.py -- 批量分类远端传回的结果文件，并把**基线臂**
# (llama3.1:8b，digest 校验过的同一模型) 合并进 farm_state.json。
# 其他模型臂只分类出报告，不合并（跨模型统计不可混）。
#
# 用法（仓库根目录 phoenix/ 下）：
#   python tools\remote_merge_results.py --rounds 10 runtime_store\remote_package_v2\results_*.json
#   python tools\remote_merge_results.py --rounds 10 --all-arms ...   # 连非基线臂也合并（一般不要）
#
# 幂等：已合并过的组 id 不会重复入账（以 farm_state.testedIds 为准）。
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RS = ROOT / "runtime_store"
CLASSIFY = ROOT / "tools" / "manifold_batch_classify.py"
CLASSIFY_OUT = RS / "manifold_batch_classify.json"
STATE = RS / "farm_state.json"

BASELINE_TAG = "llama3.1_8b"   # results_<tag>_<skip>.json 命名约定


def classify_one(path, rounds):
    run = subprocess.run(
        [sys.executable, str(CLASSIFY), "--rounds", str(rounds), str(path)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if run.returncode != 0:
        print(run.stdout)
        print(run.stderr, file=sys.stderr)
        raise RuntimeError("classify failed for %s" % path)
    return json.loads(CLASSIFY_OUT.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="结果文件（支持通配符）")
    ap.add_argument("--rounds", type=int, default=10,
                    help="实验运行的轮数（判定阈值；默认 10，v1 包是 6）")
    ap.add_argument("--all-arms", action="store_true",
                    help="把非基线臂也合并进 farm_state（一般不要）")
    args = ap.parse_args()

    files = []
    for a in args.files:
        if "*" in a or "?" in a:
            pat = a if Path(a).is_absolute() else str(ROOT / a)
            files.extend(sorted(glob.glob(pat)))
        else:
            files.append(a)
    if not files:
        print("no files matched")
        return 1

    st = json.loads(STATE.read_text(encoding="utf-8"))
    tested = set(st["testedIds"])
    meme_set = set(st["memes"])

    total_new_memes = 0
    for f in files:
        p = Path(f)
        is_baseline = p.name.startswith("results_%s_" % BASELINE_TAG) \
            or p.name == "results.json"
        rep = classify_one(p, args.rounds)
        arm = "baseline" if is_baseline else "SIDE-ARM"
        print("[%s] %s: memes=%d antimemes=%d rho=%.3f"
              % (arm, p.name, rep["memeCount"], rep["antimemeCount"],
                 rep["spearmanFieldVsSurvivalRounds"]))
        if not is_baseline and not args.all_arms:
            print("  side arm: report only, NOT merged into farm_state")
            continue
        new_ids, new_memes, new_antis = 0, 0, 0
        for row in rep["rows"]:
            rid = row["id"]
            if rid in tested:
                continue
            tested.add(rid)
            st["testedIds"].append(rid)
            new_ids += 1
            if row["verdict"] == "meme":
                if rid not in meme_set:
                    meme_set.add(rid)
                    st["memes"].append(rid)
                    new_memes += 1
            elif row["verdict"] == "antimeme":
                st["antimemes"] += 1
                new_antis += 1
        st["batches"].append({
            "file": p.name, "groups": len(rep["rows"]),
            "memes": rep["memeCount"], "antimemes": rep["antimemeCount"],
            "spearman": rep["spearmanFieldVsSurvivalRounds"],
            "rounds": args.rounds,
            "finishedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print("  merged: +%d tested, +%d memes, +%d antimemes"
              % (new_ids, new_memes, new_antis))
        total_new_memes += new_memes

    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False),
                     encoding="utf-8")
    print("farm_state: memes=%d/88 tested=%d antimemes=%d (+%d memes this run)"
          % (len(st["memes"]), len(st["testedIds"]), st["antimemes"],
             total_new_memes))
    return 0


if __name__ == "__main__":
    sys.exit(main())

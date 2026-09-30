# manifold_merge_results.py -- 把远端串行结果（results_*.json，多模型
# 多切片）按">=2 臂投票"口径合并进 farm_state.json：
#
#   臂通过：该模型臂内 presentRounds >= --min-alive（默认 5）
#   强模因：>= --min-arms（默认 2）个独立模型臂通过
#
# 设计依据（2026-09 远端审计）：逐模因判定噪声主导（同模型
# test-retest phi=0.238），单臂判定不可靠；>=2 臂通过是高置信集
# （33/1922）。llama3.1 x mistral 重叠 4.2x 随机期望（p=0.014），
# 说明跨臂一致性度量的是模因内在可复制性。
#
# Usage:
#   python tools/manifold_merge_results.py \
#       --results runtime_store/remote_package_v2_results_20260930/results \
#       [--farm-state runtime_store/farm_state.json] \
#       [--min-alive 5] [--min-arms 2] [--apply]
#
# 默认 dry-run：只打印报告不改文件。--apply 才写回 farm_state。
import argparse, json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="results_*.json 所在目录，或单个 json 文件")
    ap.add_argument("--farm-state",
                    default=str(ROOT / "runtime_store" / "farm_state.json"))
    ap.add_argument("--min-alive", type=int, default=5,
                    help="臂通过阈值：presentRounds >= N（默认 5）")
    ap.add_argument("--min-arms", type=int, default=2,
                    help="强模因阈值：>= N 个模型臂通过（默认 2）")
    ap.add_argument("--apply", action="store_true",
                    help="实际写回 farm_state（默认 dry-run）")
    args = ap.parse_args()

    rp = Path(args.results)
    files = sorted(rp.glob("results_*.json")) if rp.is_dir() else [rp]
    if not files:
        sys.exit(f"no results_*.json under {rp}")

    # id -> {model: presentRounds}
    arms = defaultdict(dict)
    file_stats = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        model = d.get("modelName") or "?"
        groups = d.get("groups", [])
        n_pass = 0
        for g in groups:
            mid = g.get("id")
            if not mid:
                continue
            pr = g.get("presentRounds", 0)
            arms[mid][model] = max(arms[mid].get(model, 0), pr)
            if pr >= args.min_alive:
                n_pass += 1
        file_stats.append((f.name, model, len(groups), n_pass,
                           d.get("rounds", "?")))

    print(f"files: {len(files)}   unique memes tested: {len(arms)}")
    print(f"{'file':<44} {'model':<14} {'grp':>4} {'R':>3} "
          f"{'armPass':>7}")
    for name, model, ng, npass, rounds in file_stats:
        print(f"{name:<44} {model:<14} {ng:>4} {rounds!s:>3} {npass:>7}")

    strong = {mid: passes for mid, a in arms.items()
              if len([1 for pr in a.values() if pr >= args.min_alive])
              >= args.min_arms
              for passes in [sorted(m for m, pr in a.items()
                                    if pr >= args.min_alive)]}
    print(f"\nstrong memes (>={args.min_arms} arms x alive>={args.min_alive})"
          f": {len(strong)}")
    for mid, passes in sorted(strong.items()):
        print(f"  {mid}  arms={','.join(passes)}")

    fs_p = Path(args.farm_state)
    fs = json.loads(fs_p.read_text(encoding="utf-8")) \
        if fs_p.exists() else {"testedIds": [], "memes": [],
                               "antimemes": 0, "batches": []}
    have_memes = set(fs.get("memes", []))
    have_tested = set(fs.get("testedIds", []))

    new_memes = [m for m in sorted(strong) if m not in have_memes]
    new_tested = [m for m in sorted(arms) if m not in have_tested]
    # antimemes：新测且未入选（含既往已入选者不重复计）
    new_anti = [m for m in new_tested if m not in strong]

    print(f"\nfarm_state: memes {len(have_memes)} -> "
          f"{len(have_memes) + len(new_memes)} (+{len(new_memes)})")
    print(f"            testedIds {len(have_tested)} -> "
          f"{len(have_tested) + len(new_tested)} (+{len(new_tested)})")
    print(f"            antimemes {fs.get('antimemes', 0)} -> "
          f"{fs.get('antimemes', 0) + len(new_anti)} (+{len(new_anti)})")

    if not args.apply:
        print("\ndry-run. pass --apply to write farm_state.")
        return 0

    fs.setdefault("memes", []).extend(new_memes)
    fs.setdefault("testedIds", []).extend(new_tested)
    fs["antimemes"] = fs.get("antimemes", 0) + len(new_anti)
    fs.setdefault("batches", []).append({
        "file": str(rp),
        "groups": len(arms),
        "memes": len(new_memes),
        "antimemes": len(new_anti),
        "rounds": "multi",
        "arms": f">={args.min_arms}x alive>={args.min_alive}",
        "finishedAt": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"),
    })
    fs_p.write_text(json.dumps(fs, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\nwrote {fs_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

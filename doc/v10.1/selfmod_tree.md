# Branch-Tree Self-Modification Harness（分支树自我修改脚手架）— v10.3

Status: tooling delivered, dry-run verified.
Tool: `tools/selfmod_tree.ps1` · State: `runtime_store/selfmod/tree.json` · Spec lineage:
`doc/v10.1/self_modification_protocol.md` (rails), `doc/v10.1/branch_batched_selfmod.md` (W=3 flat batching).

This document defines the **tree** replacement for flat W=3 batching. One candidate
change = one git branch; branches are children of the branch they were derived
from, so the candidate space is a tree:

```
main
 └─ selfmod/<ts>-n0001-<slug>              (candidate 1)
     ├─ selfmod/<ts>-n0002-<slug>          (refinement of candidate 1)
     └─ selfmod/<ts>-n0003-<slug>          (alternative refinement)
 └─ selfmod/<ts>-n0004-<slug>              (candidate 2 from main)
```

The inner loop explores that tree **with a bias, like a chess engine**:

1. **Best-first expansion** — expand the highest-scoring open node first.
2. **Prune** — keep the top-k frontier by score, remove the rest (worktree + branch).
3. **Select** — at the end keep the single best root→leaf path; a *human* adopts it.

The Lead agent never creates branches. Only the harness creates `selfmod/*`
branches, always inside a throw-away worktree under `build\tmp\selfmod\<node>`,
never in the main working tree. The harness never merges.

---

## 1. Node schema

`runtime_store/selfmod/tree.json` is the single source of truth. Every node is a
candidate branch:

```json
{
  "id": "n0002",
  "parent": "n0001",
  "goal": "raise agi.efe.horizon 3 -> 4 and keep tick latency flat",
  "status": "scored",
  "branch": "selfmod/20261117-140102-n0002-dryrun-probe-b",
  "baseBranch": "selfmod/20261117-140055-n0001-dryrun-probe-a",
  "baseCommit": "f2ca5c35b7...",
  "commit": "945da128d3...",
  "worktree": "E:\\...\\phoenix\\build\\tmp\\selfmod\\n0002",
  "patch": { "path": "runtime_store/selfmod/staging/<id>/patch.diff",
             "bytes": 512, "paths": ["autonomy_stack.cpp"], "productDiff": true },
  "diff": { "files": ["autonomy_stack.cpp"], "added": 3, "removed": 1,
            "total": 4, "productDiff": true },
  "built": { "ok": true, "exe": "build\\tmp\\selfmod\\phoenix_main.n0002.exe",
             "exitCode": 0, "durationMin": 21.4, "log": "runtime_store/selfmod/nodes/n0002/build_n0002.log" },
  "gtestBuild": null,
  "score": {
    "total": 0.87,
    "pass": 1.0, "soak": 1.0, "targetDelta": 0.35,
    "weights": { "pass": 0.5, "soak": 0.3, "target": 0.2 },
    "runner": { "path": "build\\tmp\\selfmod\\n0002\\gtest_runner.n0002.exe",
                "provenance": "node" },
    "tests": { "passed": 331, "total": 331,
               "targeted": { "passed": 63, "total": 63, "exitCode": 0 },
               "wide": { "passed": 268, "total": 268, "exitCode": 0 } },
    "soakDetail": { "port": 5211, "healthy": true, "samples": 60, "downSamples": 0,
                    "crashed": false, "stability": 1.0,
                    "csv": "runtime_store/selfmod/nodes/n0002/soak_n0002.csv" }
  },
  "evidence": [
    { "kind": "patch",  "path": ".../patch.diff",      "ts": "...", "summary": "512 bytes, 1 file(s), productDiff=True" },
    { "kind": "build",  "path": ".../build_n0002.log", "ts": "...", "summary": "exit=0 ok=True 21.4 min" },
    { "kind": "gtest",  "path": ".../gtest_targeted_n0002.log", "ts": "...", "summary": "targeted 63/63 exit=0" },
    { "kind": "gtest",  "path": ".../gtest_wide_n0002.log",     "ts": "...", "summary": "wide 268/268 exit=0" },
    { "kind": "soak",   "path": ".../soak_n0002.csv",           "ts": "...", "summary": "stability=1.0 samples=60 down=0 port=5211" },
    { "kind": "score",  "path": ".../score_n0002.json",         "ts": "...", "summary": "score=0.87 (pass=1.0 soak=1.0 target=0.35)" }
  ],
  "createdBy": "selfmod_tree.ps1",
  "createdAt": "2026-11-17T14:01:02Z"
}
```

Only `{id, parent, goal, score, evidence}` are required by the design; the other
fields make the node self-describing and evidence-complete (protocol §2.6).
`status` is one of:

| status | meaning |
|---|---|
| `baseline` | the `main` root node |
| `applied` | worktree + branch + commit exist, not built yet |
| `built` / `build_failed` | sidecar exe exists / build failed (fail-closed) |
| `scored` | gtest + soak finished, `score.total` valid |
| `unverified` | diff touches compiled sources but no node runner could be built → **cannot be selected** |
| `pruned` / `cleaned` | worktree + branch removed (kept in JSON as history) |
| `patch_failed` / `expand_failed` | worktree creation or `git apply` failed; rolled back to nothing |

The flat `baseline` block stores the main commit and the optional target-metric
baseline. The `selection` block records the last `select` recommendation
(`adopted` stays `false` — only a human changes it).

---

## 2. Expansion bias (chess-engine style)

`expand <parent>` always creates a **child of an existing node**, so expansion is
a tree walk, not a flat batch:

- **Open node** = a leaf (`status` ∈ `{applied, built, scored}`) with no open children.
- **Best-first rule**: the next node a human/agent should expand is the open node
  with the highest `score.total`; ties are broken by **shorter diff**
  (`diff.total` = added+removed lines), then by node id. This is the same
  shorter-diff preference already mandated by `branch_batched_selfmod.md` §5,
  and it keeps the tree biased toward small, reviewable changes.
- **Branch naming**: `selfmod/<UTC yyyyMMdd-HHmmss>-<id>-<slug>`.
  Slug from the changed target or `-Slug`; branch is unique by construction.
- **One commit per node branch** (`selfmod(<id>): <goal>`). The commit is created
  inside the worktree only, so cherry-pick/rollback stay single-point.
- **Base** = the parent node's branch, so a child inherits its parent's commit
  and diff. A "best path" is therefore a real derivation chain, and adopting the
  leaf with `cherry-pick` replays that chain only if the intermediate commits are
  replayed too — the harness's `select` prints the **merge** form for the whole
  branch (which contains the chain) and notes the cherry-pick form for a
  single-commit node.

Why a tree instead of flat W=3: flat batching can only compare siblings that all
start from `main`; a tree lets a promising change be *refined* (candidate 2 under
candidate 1) while alternatives from `main` are still explored. The bias keeps
the cost bounded: only the frontier is expanded, never the whole tree.

---

## 3. Scoring (`score <node>`)

```
score = 0.50 * pass_ratio
      + 0.30 * soak_stability
      + 0.20 * clamp(target_delta, -1, 1)          # total clamped to [-1, 1]
```

- **pass_ratio** = `passed / total` over two gtest runs:
  - targeted: `LogicEngine.*:PluginSemantics.*:SummonSelect.*:ExploreWidth.*:PopulationFeatures.*:RetrievalDiversity.*:StudyConfigStateSplit.*:AgiActionDispatchTest.*:CognitionHardeningTest.*` (63 tests)
  - wide regression: `*Autonomy*:*Mission*:*Study*:*Population*:*Summon*:*Memetic*:*TinyExpert*:*Expert*` (268 tests)
- **soak_stability** = `max(0, 1 - min(1, (down + crash + fatal)/samples))`,
  measured by launching the node's **own sidecar exe** on a rotated free port
  (`-PortBase`, default 5196+) with its own pid-file and `AI_BASE_DIR`, polling
  `GET /api/health` every `-IntervalSec` for `-SoakMinutes` (default 25; the
  protocol asks 20-30 min). The live gateway and llama-servers are never touched;
  the sidecar is stopped by PID only, and only its own process tree.
- **target_delta** = relative improvement of an optional target metric.
  Set `-TargetMetricCommand '<pwsh snippet printing a number>'`; the first
  measurement on `main` becomes the baseline, later nodes are compared as
  `clamp((node - base)/max(|base|,1e-9), -1, 1)`. Without the option the term is
  0.0 (the pipeline itself is still fully exercised).
- **Unverifiable nodes score nothing.** A diff that touches a compiled
  translation unit (`.cpp/.cc/.cxx/.c/.h/.hpp/.hh/.hxx/.inc/.ipp`) needs a gtest
  runner built *from that node*; the main-tree `gtest_runner.exe` is a sound
  oracle **only** for non-product diffs (docs/tools-only). For product diffs use
  `score <node> -BuildGtest` (builds `gtest_runner.<id>.exe` in the worktree,
  copying the gitignored `tests/` tree from main first — ~30-60 min). If neither
  runner is available the node becomes `unverified` and is never selected.
- **minScore** (default `0.75`): nodes below it are dropped by `prune` even when
  they are inside the top-k, matching `branch_batched_selfmod.md` §5.

Evidence (gtest logs, soak CSV, score JSON, build log) is copied to
`runtime_store/selfmod/nodes/<id>/` before any worktree is removed.

---

## 4. Pruning

`prune [-K 3]` — chess-engine frontier pruning:

1. Frontier = non-main leaves that are not pruned/cleaned.
2. Rank scored frontier nodes by `score.total` desc, then `diff.total` asc, then id.
3. Keep the top-k (default **k = 3**, configurable) and additionally keep any
   kept node with `score.total >= minScore`; drop everything else.
4. Dropped nodes: `git worktree remove --force` + `git branch -D` (only branches
   whose name starts `selfmod/` and whose id is recorded with
   `createdBy: selfmod_tree.ps1`). The node stays in tree.json as `pruned` with
   `pruneReason` — history is never rewritten.
5. Unscored frontier nodes are reported and **kept** (pruning must not kill a
   candidate before it is scored). `-DryRun` prints the decision table only.

`k` is a hard frontier width, not a target: W=3 means "at most 3 open scored
nodes alive". Smaller k = more exploitation, larger k = more exploration.

---

## 5. Selection and adoption

`select` prints:

- the best root→leaf path with per-node score, diff size, goal, status, evidence;
- the exact local git commands a human would run:

```
git -C <repo> log --oneline main..<best-branch>
git -C <repo> diff --stat main...<best-branch>
git -C <repo> checkout main
git -C <repo> merge --no-ff <best-branch> -m "selfmod: adopt <id> (score=...)"
#   or replay the single commit:
git -C <repo> cherry-pick <sha>
cmd /c compile_gtest.bat
gtest_runner.exe --gtest_filter="<targeted filter>"
gtest_runner.exe --gtest_filter="<wide filter>"
git -C <repo> tag selfmod/<ts>-merged-<id>
# rollback (human decision only): git -C <repo> reset --hard selfmod/<previous-good-tag>
```

The harness itself executes **none** of these: no merge, no checkout of main, no
push, no reset. `select` records the recommendation in `tree.json.selection`
with `adopted=false`.

---

## 6. Hard rails (inherited, and how the harness enforces them)

| rail (protocol) | enforcement in `selfmod_tree.ps1` |
|---|---|
| never run `compile.bat` (kills the live gateway) | node builds are a generated sidecar batch (`compile_node_<id>.bat`) compiling to `phoenix_main.<id>.exe` with per-node `TMP`; `selfcheck` refuses to start while a main-tree `compile.bat`/`compile_gtest.bat` is running |
| never disturb the live gateway / llama-servers | every command reads the live pid/port before and after and warns if it changed; soak runs a *second* exe with `CONTROLLER_PORT` on a rotated free port, `AI_PID_FILE` + `AI_BASE_DIR` in its own artifact dir, `AI_FRONTEND_ENABLED=false`; teardown only stops PIDs it spawned (never by image name, and refuses the live pid) |
| path whitelist (`selfModify.allowPaths`) | patch paths are parsed from the diff and validated against `main_hub_parts/`, `addons/`, `doc/`, `tools/`, `config/phoenix.json`, `*.hpp`, `autonomy_stack.*`; anything else is rejected before a worktree is created |
| deny paths (`selfModify.denyPaths`) | `.git/`, `build/`, `outsides/`, `GGUF_models/`, `runtime_store/`, `Python314/`, `poppler-*`, `models/`, `datasets/`, `node_modules/`, `hacktests/`, `catastrophe/`, plus binary extensions and absolute/`..` paths |
| patch size cap | `selfModify.maxPatchBytes = 204800` (200 KB) enforced |
| auto-revert on failure | the harness never writes to main, so a failed build/test/soak **is** a revert: the node is marked failed/unverified, all evidence is kept, and the node becomes prunable. Worktree-creation/`git apply` failures remove the fresh worktree + branch immediately |
| no push / rebase / reset --hard / history rewrite | those commands do not exist anywhere in the harness; `select` prints them only as human guidance, labelled "not run by the harness" |
| never delete branches it did not create | prune/cleanup/teardown delete only `selfmod/*` branches of nodes recorded with `createdBy: selfmod_tree.ps1`; `teardown` refuses any other node |
| evidence per change (protocol §2.6) | every node carries `evidence[]` (patch, build log, gtest logs, soak CSV, score JSON, commit) under `runtime_store/selfmod/`; failures also write `runtime_store/selfmod/last_failure.json` |

L3 (architecture-level changes) still requires a human signature per the
protocol; the tree harness cannot bypass that — it only automates L2 staging,
scoring and pruning.

---

## 7. Usage

```powershell
# one-time
pwsh -NoProfile -File tools/selfmod_tree.ps1 init
pwsh -NoProfile -File tools/selfmod_tree.ps1 selfcheck

# stage a candidate as a CHILD of an existing node (or of main)
pwsh -NoProfile -File tools/selfmod_tree.ps1 expand main `
  -Patch runtime_store/selfmod/staging/n0001/patch.diff `
  -Goal "agi.efe.horizon 3 -> 4, keep tick flat" -Slug efe-horizon-4
# expands + sidecar-builds; add -NoBuild to stage only
# use "build <id>" to (re)build later

# score it (targeted + wide gtest + soak)
pwsh -NoProfile -File tools/selfmod_tree.ps1 score n0001 -SoakMinutes 25

# if the diff touched product sources, build the node test runner first:
pwsh -NoProfile -File tools/selfmod_tree.ps1 score n0001 -BuildGtest -SoakMinutes 25

# frontier management
pwsh -NoProfile -File tools/selfmod_tree.ps1 prune -K 3          # execute
pwsh -NoProfile -File tools/selfmod_tree.ps1 prune -K 3 -DryRun  # plan only
pwsh -NoProfile -File tools/selfmod_tree.ps1 status

# final recommendation (prints commands, executes none)
pwsh -NoProfile -File tools/selfmod_tree.ps1 select

# housekeeping
pwsh -NoProfile -File tools/selfmod_tree.ps1 cleanup -Force
pwsh -NoProfile -File tools/selfmod_tree.ps1 teardown n0003 -Force
```

Cost model (measured on this host, 56 logical CPUs):

| step | cost |
|---|---|
| `expand -NoBuild` (worktree + apply + commit) | ~3 s |
| sidecar node build (`phoenix_main.<id>.exe`, `-O3`, 68 TUs) | ~20-25 min |
| node gtest build (`gtest_runner.<id>.exe`, `-O0 -g1`) | ~30-60 min |
| targeted + wide gtest with an existing runner | ~1-3 min |
| soak (default) | 25 min |

A real **W=3 tree round** therefore costs roughly `3 * 25 min` build + `3 * 1 min`
tests + `3 * 25 min` soak ≈ **2.5-3 h serial**, or ~1 h wall if node builds and
soaks run in parallel — compare with flat batching (same cost) but the tree can
spend the second round *refining the winner* instead of restarting from main.

### Next command for a real W=3 tree run

```powershell
# 0) confirm the live long run is up and main is green
pwsh -NoProfile -File tools/selfmod_tree.ps1 selfcheck

# 1) stage W=3 candidates (three children of main; patches prepared by the
#    patch generator from the previous pipeline, or by hand)
$patches = @(
  @{ id='n0001'; p='runtime_store/selfmod/staging/t1/patch.diff'; g='agi.efe.horizon 3->4' },
  @{ id='n0002'; p='runtime_store/selfmod/staging/t2/patch.diff'; g='agi.efe.horizon 3->5' },
  @{ id='n0003'; p='runtime_store/selfmod/staging/t3/patch.diff'; g='mission.recallDiversity 0.7' }
)
foreach ($c in $patches) {
  pwsh -NoProfile -File tools/selfmod_tree.ps1 expand main -Id $c.id -Patch $c.p -Goal $c.g -NoBuild
}
# 2) build the three sidecar exes in parallel (separate shells / jobs)
foreach ($c in $patches) { Start-Job { pwsh -NoProfile -File tools/selfmod_tree.ps1 build $using:c.id } }
# 3) score each (build node gtest runner too for product diffs)
foreach ($c in $patches) { pwsh -NoProfile -File tools/selfmod_tree.ps1 score $c.id -BuildGtest -SoakMinutes 25 }
# 4) prune to the top-3 frontier, then select the best path
pwsh -NoProfile -File tools/selfmod_tree.ps1 prune -K 3
pwsh -NoProfile -File tools/selfmod_tree.ps1 select
# 5) refine the winner instead of restarting: expand the highest-scoring node
pwsh -NoProfile -File tools/selfmod_tree.ps1 expand <best-id> -Patch <refinement.diff> -Goal "..."
# 6) a human adopts the final path with the commands printed by select
```

---

## 8. Dry-run (mechanics proof)

Performed by the harness on a harmless change (comment-only markdown scratch files
under `doc/v10.1/`, no product code, no build inputs). Full transcript and
artifacts: `runtime_store/selfmod/nodes/` and `runtime_store/selfmod/dryrun/`.
The ablation below is the recorded result of the dry run.

| step | node | result |
|---|---|---|
| `selfcheck` | — | live gateway port 5097 pid 23920 healthy; main clean at start; toolchain/conan/source-list OK |
| hard-rail probe | — | patch touching `build/evil.cpp` rejected **before** any worktree creation (`deny path: build/`); no node, no branch, main untouched |
| `expand main -NoBuild` | n0001 (probe A, +3 lines) | worktree `build/tmp/selfmod/n0001` + branch `selfmod/20261117-140055-n0001-dryrun-probe-a`, one commit `5e8c55a` |
| `expand main -NoBuild` | n0002 (probe B, +7 lines) | worktree + branch `selfmod/20261117-140102-n0002-dryrun-probe-b`, one commit `945da12`; deliberately larger diff to exercise the tie-break |
| `build` | n0001 | sidecar `phoenix_main.n0001.exe` (44.9 MB) built in **22.0 min** from the worktree; `compile.bat` never run; gateway stayed up |
| `build` | n0002 | sidecar `phoenix_main.n0002.exe` (44.9 MB) built in **22.3 min**; both builds ran in parallel, main tree untouched |
| `score` | n0001 | targeted **63/63** + wide **268/268** (shared main runner, sound because the diff is non-product); soak on rotated port 5196, healthy after 34 s, 12/12 samples up, no crash → **0.8** |
| `score` | n0002 | targeted **63/63** + wide **268/268**; soak port 5196, 12/12 samples up → **0.8** (equal score → shorter-diff tie-break decides) |
| `prune -K 1` | n0001 | kept (diff 3 lines); **n0002 pruned**: worktree `build/tmp/selfmod/n0002` + branch `selfmod/...n0002...` removed |
| `select` | n0001 | best path `main -> n0001`, per-node score/evidence + adoption commands printed; no merge executed |
| `cleanup -Force` | all | every harness worktree/branch removed; `git worktree list` left only main; `git branch --list 'selfmod/*'` empty; `build/tmp/selfmod/` empty; harness paths clean in `git status` |

Dry-run soak used an abbreviated window (`-SoakSeconds 120`) because it is a
mechanics proof; the real `score` default is 25 min (`-SoakMinutes 25`). The
recorded sidecar run reached `/api/health` in 34 s and stayed up for the whole
window, so the real 25-minute soak uses the same proven launch path.

Two build/soak issues surfaced and were fixed during the dry run (kept here as
operational knowledge):

1. node sources resolve `python314/include/Python.h` through the repo-root
   include dir → the generated sidecar build adds `-I<main-root>` **after**
   `-I<worktree>`, so node headers still shadow main;
2. the worktree exe needs `python314.dll`/`poppler.dll` and the shared model
   corpus → the soak child gets a scoped `PATH` (main root, `Python314`,
   poppler `bin`, mingw `bin`) plus `AI_GGUF_MODELS_DIR` / `AI_ROBOTS_DIR` /
   `AI_LEMMA_CSV` overrides pointing at the main tree. The node's own
   `AI_BASE_DIR` and pid file stay inside `runtime_store/selfmod/nodes/<id>/`,
   so no runtime state is shared with the live long run.

---

## 9. 中文版（Chinese section）

### 9.1 设计

系统自我改进时，**每个候选改动创建一个 git 分支**，分支的父节点是它派生自的分支，
于是候选空间是一棵树（`main -> 候选 -> 子候选 ...`）。内循环像国际象棋引擎一样
带偏置地探索这棵树：

1. **最佳优先扩展**：先扩展当前分数最高的开放节点（叶子）；
2. **剪枝**：只保留分数 top-k 的边界节点（默认 k=3），其余删除（工作树+分支）；
3. **选择**：最终只保留一条最佳根→叶路径，由**人类**按 `select` 打印的 git 命令合入。

Lead 代理不创建分支；只有脚手架在 `build\tmp\selfmod\<节点>` 的临时工作树里创建
`selfmod/*` 分支，主工作树永远不被改动，脚手架也从不执行 merge。

### 9.2 节点模式

`runtime_store/selfmod/tree.json` 是唯一事实来源，每个节点至少包含
`{id, parent, goal, score, evidence}`，另带 `status/branch/baseBranch/baseCommit/commit/
worktree/patch/diff/built/gtestBuild` 等自描述字段。状态机：
`baseline → applied → built/build_failed → scored`；
`pruned/cleaned` 保留在 JSON 中作为历史；`unverified` 表示 diff 触及编译源文件却没有
从该节点构建 gtest runner，此类节点**永不被选中**。

### 9.3 扩展偏置

- 开放节点 = 没有开放子节点的叶子；
- 最佳优先：按 `score.total` 降序；同分按 **diff 更短**（增删行数少）、再按 id；
- 分支命名 `selfmod/<UTC时间戳>-<id>-<slug>`，每分支**只允许一个 commit**；
- 子节点基于父节点分支，因此"最佳路径"是真实派生链。

### 9.4 打分

```
score = 0.5*通过率 + 0.3*soak稳定性 + 0.2*clamp(目标指标增量,-1,1)
```

- 通过率 = 定向 63 项 + 宽回归 268 项 gtest 的 `passed/total`；
- soak：用节点自己的侧车 exe 在**轮转的空闲端口**上启动第二个网关（独立 pid 文件、
  独立 `AI_BASE_DIR`、前端关闭），每 `-IntervalSec` 探测 `/api/health`，
  默认 25 分钟；**不碰**正在运行的长跑网关与 llama-servers，只按 PID 停止自己启动的进程；
- 目标增量：可选 `-TargetMetricCommand`，在 main 上取基线，节点相对基线归一化；
- 未验证节点不得分、不可选；产品源码改动必须 `-BuildGtest` 从节点构建测试运行器；
- 分数低于 `minScore`（默认 0.75）的节点即使进入 top-k 也会被剪掉。

### 9.5 剪枝与选择

`prune -K 3`：边界 = 非 main 的未剪枝叶子；按分数降序、diff 升序排序，保留前 k，
并额外保留分数仍 ≥ minScore 的节点；其余执行 `git worktree remove --force` +
`git branch -D`（只删 `selfmod/` 且记录在 tree.json 里的分支）。未打分的叶子会保留
并提示，避免"还没验证就被杀"。`select` 打印最佳路径、每节点分数与证据，以及人类
合入所需的**精确 git 命令**（merge 或 cherry-pick、重跑测试、打 tag、回滚），
脚手架本身不执行其中任何一条。

### 9.6 硬性护栏（继承协议，脚手架落实）

- 绝不运行 `compile.bat`（会杀死正在跑的长跑网关）：节点构建是生成的侧车批处理，
  输出 `phoenix_main.<id>.exe`，使用节点自己的 `TMP`；
- 绝不动网关/llama-servers：只在轮转端口起第二个 exe，独立 pid 与数据目录，
  只按 PID 停止自己启动的进程，并拒绝停止活动网关 PID；
- 补丁白名单：`main_hub_parts/`、`addons/`、`doc/`、`tools/`、`config/phoenix.json`、
  `*.hpp`、`autonomy_stack.*`；命中 `.git/`、`build/`、`outsides/`、`runtime_store/`、
  `GGUF_models/`、`Python314/`、二进制扩展名、绝对路径或 `..` 一律拒绝；
- 补丁上限 200 KB；
- **失败即回退**：脚手架从不写 main，所以构建/测试/soak 失败天然等于回退——节点标记
  失败、证据保留、可直接剪枝；worktree 创建或 `git apply` 失败会立即删除新工作树与分支；
- 无 push / rebase / reset --hard / 历史改写；只删除自己创建的 `selfmod/*` 分支；
- 每个改动都留证据（补丁、构建日志、gtest 日志、soak CSV、score JSON），失败写
  `runtime_store/selfmod/last_failure.json`。

### 9.7 成本与真实的 W=3 树轮次

单节点：扩展 ~3 s；侧车构建 20-25 min；节点 gtest 构建 30-60 min；gem 测试 1-3 min；
soak 25 min。W=3 串行约 2.5-3 h，并行（构建与 soak 错峰）约 1 h 墙钟。与扁平批处理
成本相同，但第二轮可以**在最优点上继续细化**，而不是从 main 重来。

真实 W=3 的首条命令见第 7 节末尾的 PowerShell 片段，核心是：

```powershell
pwsh -NoProfile -File tools/selfmod_tree.ps1 selfcheck
# 三个 candidate 各 expand 一次（作为 main 的子节点），并行 build，再 score -BuildGtest
pwsh -NoProfile -File tools/selfmod_tree.ps1 prune -K 3
pwsh -NoProfile -File tools/selfmod_tree.ps1 select
```

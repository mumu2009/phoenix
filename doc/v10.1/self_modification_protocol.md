# 自我修改协议（Self-Modification Protocol）— v10.2

> 目标：以 **AGI/自主进化层** 为基点，让系统开始修改自身架构。
> 本文不是"让模型自由重写仓库"的许可，而是**护栏**。没有护栏的自我修改
> 等于不可回滚的破坏。

## 0. 现状（事实）

- 已经存在的能力：SELF-MOD 任务链（`SELF-MOD:` goal → coder expert →
  `FILE/OLD/NEW` 补丁 → `code_edit_parser` → 工作区写入）、AGI EFE 规划器、
  动作注册表（`math/search/computer/script/replicate/logic`）、配置热更新。
- 尚缺的是**执行护栏**：谁允许把补丁写回仓库、写之前是否已有可回滚点、
  写之后是否强制编译+测试、失败如何自动回退。
- 因此本轮交付：git 检查点脚本 + 本协议 + `selfModify.enabled=false` 硬开关。
  **运行时自动改写仓库尚未开启**——先有护栏，再开门。

## 1. 三级权限

| 级别 | 对象 | 是否需要编译 | 护栏 |
|---|---|---|---|
| **L1 参数级** | `config/phoenix.json` 中可热更新的键（采样温度、`addons.semantic*`、`mission.recallDiversity`、`summon.select*`、slot 数、`autonomyLoop.intervalSec`、`study.*` 开关） | 否 | 热更新 watcher 自动生效；改前仍需 checkpoint（便于对比/回滚） |
| **L2 代码补丁** | 白名单路径内的 `.cpp/.hpp/.inc` | 是 | checkpoint → 打补丁 → `compile.bat` → gtest 定向+宽回归 → 30 min soak → 保留或回滚 |
| **L3 架构级** | 新增模块、改动数据流、改协议/契约 | 是 | L2 全部 + **人工审批**（人必须在场签字） |

## 2. 硬性 rails（不可协商）

1. **每次自改前必须 checkpoint**：`tools/self_mod_checkpoint.ps1` 会
   `git add -A` → commit → 打 tag `selfmod/<UTC时间戳>`；失败则中止自改。
2. **白名单路径**（`selfModify.allowPaths`）：默认只允许
   `main_hub_parts/`、`autonomy_stack.*`、`addons/`、`*.hpp`、`config/phoenix.json`、
   `doc/`、`tools/`。
3. **禁止路径**（`selfModify.denyPaths`）：`.git/`、`build/`、`outsides/`（vendored）、
   `GGUF_models/`、`runtime_store/`（数据）、`Python314/`、`*.dll/*.exe`。
4. **禁止** `git push` / rebase / reset --hard / 改写历史；只允许本地 commit 与 revert。
5. **失败即回滚**：编译失败、gtest 失败、或 soak 崩溃 → 自动 `git revert` 到
   checkpoint tag，并把失败原因写入 `runtime_store/self_mod/last_failure.json`。
6. **每次改动留证据**：diff、编译日志路径、gtest 汇总、soak 结果，写进
   `runtime_store/self_mod/<timestamp>/report.json`。
7. **AGI 规划器不直接执行补丁**：它只能"提出"（`self_mod propose`），
   执行由 rail 进程完成，且必须通过上面 5 条。

## 3. 执行流程（L2）

```
1. tools/self_mod_checkpoint.ps1                 # 可回滚点 + tag
2. AGI 生成补丁 -> runtime_store/self_mod/staging/<id>/patch.diff
3. rail 校验：路径白名单/禁改路径/diff 大小上限（默认 200 KB）
4. git apply --check 通过后 apply
5. compile.bat（新增 -Wl,-Map 便于后续取证）
6. gtest 定向套件（logic/plugin/summon/explore/population/retrieval/autonomy）
7. 宽回归 268 项（37 套件）
8. soak 30 min：0 崩溃 + up 率 >= 90% + GPU 均值不下降
9. 全通过 -> 保留改动并 commit；任一失败 -> git revert <tag>
```

## 4. 与 AGI 的关系（为什么以它为基点）

AGI 层已经具备三件事，正好是自我修改需要的：**(a) 动作选择**（EFE 在注册动作里挑，
包含 `replicate` 等改变自身结构的动作）、**(b) 价值学习**（TD(0) 以
`benefit/harm netUtility` 为奖励——"改完更好用"可以被学到）、**(c) 惊奇信号**
（前向模型预测误差回灌为 Novelty，驱动探索新结构）。

所以自我修改的第一形态是：**AGI 用 EFE 规划 `self_mod` 动作 → 生成候选补丁 →
护栏执行 → 结果（编译/测试/soak 通过与否）作为奖励回到 TD(0)**。
这使得"改自己"成为可学习、可退化（学不到就回到不改）的过程，而不是一次性赌博。

## 5. 开关与默认值

| 键 | 默认 | 含义 |
|---|---|---|
| `selfModify.enabled` | `false` | 总开关；关时 `self_mod` 动作不注册 |
| `selfModify.allowPaths` | 见 §2.2 | 允许改写的路径前缀 |
| `selfModify.denyPaths` | 见 §2.3 | 永不允许 |
| `selfModify.maxPatchBytes` | `204800` | 单次补丁上限 |
| `selfModify.requireTests` | `true` | 必须跑 gtest 才能保留 |
| `selfModify.soakMinutes` | `30` | 保留前的最短长跑 |

开启顺序建议：先在 L1（热改参数）观察 24 h，再开 L2 让 AGI 只做"小补丁"
（单文件、<2 KB），最后才讨论 L3。

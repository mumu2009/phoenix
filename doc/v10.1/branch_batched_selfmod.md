# 分支批量化架构更新（Branch-Batched Architecture Update）— v10.2

> 用户设想：用 git 分支把"架构候选"并行展开（**保持宽度**），而不是一次只赌
> 一个改法；每个分支独立验证，按分数保留。这是我们上 L3 的正确姿势。
> 本文补齐 §self_modification_protocol.md 里 L3 的批量化部分。

## 1. 为什么不"单线改到底"

单线自我修改 = 贪心一次上路，一旦走错就要回滚再试，成本是 O(次数 × 编译时长)。
分支批量化把探索宽度显式化：N 个候选**同时**存在，互不覆盖，验证结果可比较。
这与项目里已有的宽度原则一致（`pickSoftmax` 退火、`pickExplorePath`、`mmrSelect`）。

## 2. 分支模型

```
main                      ← 只有通过全部护栏的候选才合并回来
 └─ selfmod/<id>-<slug>   ← 每个候选一个短命分支（≤1 次改动）
```

- 分支命名：`selfmod/<utcTS>-<slug>`，slug 取自改动目标（如 `agi-efe-horizon`）。
- **每个分支只允许一次 commit**，便于 cherry-pick / 单点回滚。
- 基线永远是可跑的 main（= 最近一个 `selfmod/<ts>` tag）。

## 3. 宽度（W）怎么定

| W | 含义 | 建议 |
|---|---|---|
| 1 | 单线 | 只适合修正类改动 |
| **3–5** | 推荐 | 同一目标的多候选（不同实现/不同超参），编译与测试可错峰 |
| >6 | 过度 | 机器只有 8 核 + 3 个 llama slot，再多会让"测试"本身成为噪声 |

配置：`selfModify.batchWidth`（默认 3）。**宽度是硬上限**，不是目标值。

## 4. 批量化流水线（每个候选独立走）

```
00 基线确认      git tag selfmod/<ts>；确保 main 绿（gtest 63/63）
01 建分支        git switch -c selfmod/<ts>-<slug>
02 打补丁        AGI 产出 patch.diff -> 白名单/禁改/大小校验
03 编译          侧车编译：-o phoenix_main.next.exe（不打断正在跑的长跑！）
04 定向 gtest    logic/plugin/summon/explore/population/retrieval/security
05 宽回归        268 项（37 套件）
06 短长跑        30 min soak（侧车二进制，端口轮转自动避开）
07 打分          score = 0.5*测试通过率 + 0.3*soak稳定性 + 0.2*目标指标增量
08 保留/丢弃     通过者按分数排序；只合并 top-1（可选 top-2 若互不冲突）
```

**关键：03–06 全程不 kill 正在运行的网关**——侧车二进制 + soak 监控继续跑。
切换只在 07 之后，用"停 3 秒 → 换 exe → 起"的原子窗口完成。

## 5. 打分函数（可复现，不用主观判断）

```
score =
  0.50 * (定向+宽回归通过率)                 # 0..1
+ 0.30 * soak 稳定性 (1 - min(1, crash+down/采样数))
+ 0.20 * 目标指标相对基线增量 (clamp(-1..1))
```

- 目标指标随改动类型不同：性能类用 `tick 耗时/GPU 均值`；能力类用
  `goalsRun/完成率`；安全类用 `误报率`。
- `score < 0.75` 一律丢弃；同分取改动行数更少者（更小 diff 更可维护）。

## 6. 合并规则

1. 只合 `score` 最高且 **diff 与其它候选无重叠文件** 的分支；
2. 合并后重跑 06 的 soak 30 min，作为"合并后仍然是绿的"证据；
3. 合并提交信息必须带：分支名、score、测试摘要、soak 摘要；
4. 合并后立即 `git tag selfmod/<ts>-merged`，下一个候选以此为基线。

## 7. L3（架构级）的额外要求

- L3 候选**必须**同时提供：`doc/v10.1/` 下的一页设计与"可回滚点"；
- 必须由人审批（协议 §2）——批量化不能绕过这一点；
- L3 的分支隔离额外加一条：候选分支里**不允许**改 `main` 的既有公共契约
  （`main_hub_parts/` 对外 struct/route 签名），只能新增；
- 若候选需要新依赖（新库/新进程），先单独提交依赖提案分支。

## 8. 工具

| 脚本 | 作用 |
|---|---|
| `tools/self_mod_checkpoint.ps1` | 自改前打 tag（已交付） |
| `build/tmp/compile_next.bat` | 侧车编译到 `phoenix_main.next.exe`（不打断长跑） |
| `build/tmp/soak_monitor.ps1` | 侧车 soak 采样（已交付，窗口/间隔可配） |
| `build/tmp/nightly_check.ps1` | 重编 gtest + 定向套件 + soak 汇总（已交付） |

## 9. 现状与下一步

- 本轮：护栏（checkpoint + 协议 + 豁免区）+ 侧车编译通道就位；
- 下一步：在 `selfModify.enabled=true` 下跑**第一批 W=3** 的候选（建议目标：
  `agi.efe.horizon` 3→4/5/6 或 `mission.recallDiversity` 0.6/0.7/0.8 —— 都是
  L1/L2 边界上的小改动，可先验证流水线本身）；
- 流水线稳了再放开 L3。

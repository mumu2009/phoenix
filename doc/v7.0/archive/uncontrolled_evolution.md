# 封存：不受控的自进化设想（(1+λ)-ES 进程内选择）

**状态：封存（2026-08）。** 本文保留原始设想与封存原因，供未来在独立 develop 分支重启。

---

## 1. 原始设想（曾写于 mission_layer.md §4）

- 基因组：`MissionGenome`（SubconsciousProfile + learningRate）；
- 变异：`spawnChild(parent, rate)` 高斯扰动 + clamp；
- 选择：适应度 fitness = −completionTime，进程内保留最优父本，(1+λ)-ES 跨代循环自动运行；
- 目标：让实例群体围绕"更快完成任务"自我优化，无需人为干预。

## 2. 封存原因

1. **目的定位**：模型的最终目的是**服务于人的需求**。不受控的自我进化意味着实例群体按自己的适应度函数自主繁殖选择——即使目标无害，这也把"系统行为由谁决定"从人手里交了出去。这与项目"任何模块皆可被用户自由使用/废止"的原则冲突。
2. **没有消费者**：生产代码中不存在会读取进程内 fitness/elite 的跨代循环（`spawnMissionChild` 零生产调用点）。在无人读的字段里写数据是猜测性死代码。
3. **编排归属**：选择在生物学上发生于"实例之间"（盒子里多个 Meeseeks 赛跑），而 `MissionLifecycle` 是单实例内状态机。真正的 (1+λ) 编排属于监督者/网关层；该层尚未存在，不应被文档提前"实现"。

## 3. 保留在代码中的部分（复制的权力交给实例）

- `MissionGenome::mutate`（遗传+变异）——复制时的多样性来源；
- `MissionLifecycle::replicate(rate)`——实例自由复制的原语：变异自身基因组、记录 `MissionChild`、绑定同一目标；
- `replicate` 规划器动作（category="replicate"）——实例经 EFE 规划自主选择何时复制，**无固定触发条件、无接班**；
- `mission.maxReplicas`（默认 4）——唯一护栏，防繁殖风暴；
- `MissionLifecycle::stats().completionTimeMs` / `children`——人类监督信号；
- 协议层验证：`testing_methodology.md` §9.5 模拟多代重试验证"完成时间下降"这一统计性质。

**明确不在代码中**：fitness 字段、最优父本保留、按完成时间的自动选择、跨代自动循环、接班（hand-off）——这些属于封存的"不受控进化"部分。
## 4. 重启条件（若未来需要）

- 存在一个真实的监督者/调度层（网关或多实例运行时）需要跨代选择；
- 用户明确同意"任务域内自动繁殖选择"的边界（如：仅限目标域化实例、人类可随时叫停）；
- 在独立 develop 分支实现：进程内 elite 保留 + λ 子代循环 + fitness 记录，并补齐运行时测试。

## 5. 当时的实现记录（可追溯）

- 2026-08 修订前，文档声称"选择 = 完成时间、构成 (1+λ)-ES"，但代码仅有 `spawnChild`（变异）、`markComplete`（计数器）与 `completeCount_`；选择与 fitness 从未实现。审计结论：该表述属设计意图，非运行时事实。本文与 `mission_layer.md` §4 的修订即由此审计触发。
---

## 重开记录（v8.x，2026-08，用户批准）

- **触发**：用户明确要求『实现自主进化』（不再满足于 pressure 驱动的固定循环与无进化的复制）。
- **重开范围（受控子集）**：本文原始设想中的进程内 (1+λ)-ES 适应度竞赛**仍保持封存**；
  重开的只是**历史选择环**：变异（MissionGenome 白名单）+ 完成记录（lineage 审计）
  + softmax 加权变异步长（不淘汰个体）。设计见 doc/v8.3/autonomous_evolution.md。
- **新护栏**：白名单硬编码、全审计 + 人工 reset、默认关闭（mission.evolution.enabled=false）、
  E-stop 不受影响、人类 interject 永远优先。
- **再次封存条件**：若观察到不受控行为（lineage 被用于在线淘汰/权限扩张/绕过 E-stop），
  立即将 mission.evolution.enabled 设回 false 并重新封存本文。

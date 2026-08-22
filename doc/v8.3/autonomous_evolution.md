# 自主进化 / 自主循环 / 自主推理（v8.x C 阶段设计）

**目标（用户）**：让 AI 通过自身探索实现自主进化、自主循环、自主推理——而不是
pressure 这类『TDD 范式』驱动的固定循环，也不是没有进化的复制。

## 1. 三层能力与实现

### C1 自主推理（计划先行）——已实现
- deliberator 第 5 种回复形态 `{"plan":"..."}` 写入工作区 `plan.md`；
- 每 tick 先注入 plan（≤500 字符）再动笔/调工具；
- 门槛：deliverable ≥ minChars/3 才允许写计划（先有极简草稿再规划）；
- 人类否决权：`/api/mission/file` 可随时读写 plan.md。

### C2 自主循环（AI 决定节奏与内容）——已实现
- 第 6 种形态 `{"loop":{"action":"pause|replicate|continue","reason","ticks"}}`；
- pause：manager `loopPauseTicks_` 跳过 N tick（≤ mission.maxSelfPauseTicks=6，deliberator 侧钳制）；
- replicate：直接 spawnMissionChild（子盒 parentId=自身，与 v8.2 血统一致）；
- continue：无操作确认。
- 人类 interject/amendGoal 永远优先；E-stop 在最外层不受影响。

### C3 自主进化（历史选择环）——已实现，默认关
- **变异**：`MissionGenome::mutate`（subconscious 参数 + learningRate 高斯变异，白名单
  硬编码在 mutate 内，不可经配置扩展）；
- **评估**：mission 完成时 `markComplete` 记录 `(genome 指纹, 完成时间, 覆盖度)` 进 lineage
  （内存内，上限 1000 条，`/api/mission/lineage` 全审计）；
- **选择**：下次 replicate 的变异步长 = `effectiveMutationRate`——历史完成越快/覆盖越高，
  步长收缩（0.5×），反之放大（2.0×）。**只影响新变异分布，不淘汰任何个体**（非在线竞赛）；
- **开关**：`mission.evolution.enabled` 默认 false；关闭时 lineage 仍记录（纯审计）。

## 2. 元闭环（C4）

```
自主循环探索（C2 工具/子盒）
      │ 完成（v8.3 done 自验收）
      ▼
经验沉淀（B2 mission_experience.json，≤200 条）
      │ 相似目标检索注入静态前缀
      ▼
lineage 塑造下次变异步长（C3 softmax 加权）
      │
      ▼
GNN 大纲随经验进化（B3：完成经验经 learnFromDialog 回灌学习管线，图真实加边/强化）
      │
      └──► 下一个任务的起点更好 ──► 循环
```

任一开关关闭：系统退化到 v8.3 行为（有 done 自验收、无 plan/loop/进化）。

## 3. 安全边界

1. **白名单**：可变异字段固定在 `MissionGenome::mutate`，配置无法扩展；
2. **历史选择**：无在线淘汰、无精英池、无适应度竞赛——选择只影响变异分布；
3. **全审计**：lineage 每条记录指纹/时间/覆盖度，`GET /api/mission/lineage`；
4. **人工重置**：`POST /api/mission/lineage/reset`；
5. **默认关**：`mission.evolution.enabled=false`；
6. **E-stop**：EmergencyStop 在循环最外层，任何进化状态不能禁用它；
7. **人类优先**：interject/amendGoal 永远优先于 AI 的 plan/loop 意图。

## 4. 与原封印文档的关系
- `doc/v7.0/archive/uncontrolled_evolution.md` 的封印条件是『人类未批准不受控进化』；
- 用户于 v8.x 明确要求自主进化 → 按文档重开，叠加本节安全边界；
- 重开记录见该文档尾注。

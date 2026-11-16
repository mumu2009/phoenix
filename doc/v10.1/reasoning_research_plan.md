# 符号推理层调研与评估计划（先调研、后决策）

> 本文是**决策门**，不是设计文档。它存在的理由：上一轮在没有调研的情况下就把
> Prolog/Datalog 引擎接进了 AGI 动作注册表，这是流程违规。现在该引擎默认关闭
> （`logic.enabled=false`），必须先回答下面三个问题才允许启用。

## 0. 当前状态（事实，不是判断）

- `logic_engine.hpp`：header-only，事实 + Horn 子句 + 前向链（不动点）+ 后向链（SLD）。
  已修两处缺陷（未编译验证）：
  1. **规则改名必须共享映射**：原来 head 与 body 分别改名（head `X#1`、body `X#2`），
     头部绑定传不到 body，所有规则静默返回 0 解；
  2. **需要总步数预算**：仅靠深度上界时，循环规则库（p:-q, q:-p）在 depth=24 下
     是 2^24 次归结——gtest 实测跑满 248 s CPU 仍未结束。现 `maxSteps` 默认 20000，
     耗尽即 fail-closed。
- 已注册为 AGI 动作 **但默认不注册**（`logic.enabled=false`）；`ExecuteLogicTool`
  断言"未注册时必须拒绝"。
- 已知**不能**支撑启用的事实：`gnnMs ≈ 1 ms` 而一次生成是秒级 → 符号层不是吞吐
  杠杆；没有任何测量显示 mission 需要"证明"。

## 1. 要回答的三个问题（缺一不可）

### Q1 需求频率：任务里真的出现需要确定性推理的提问吗？
- 数据：`runtime_store/long_run/gateway.log` 中的 mission goal + deliverable 全文
  （已有数十个 scope）。
- 判据：按三类标注每个 mission ——
  (a) 约束一致性（"与已声明约束是否一致"），
  (b) 前置条件/依赖（"X 是否蕴含 Y"），
  (c) 子盒完成度（"哪些 box 仍未完成"）。
- 通过门：**≥200 个 tick、≥30 个 mission** 里 (a)+(b)+(c) ≥ 20% 的 mission 至少
  提出一次此类问题。否则：不做（不是"以后再说"，是判定为无需求）。

### Q2 选择率：规划器会不会真的选它？
- 做法：离线重放 mission 的 (goal, draft tail, 工具清单) 到 `ActiveInferenceController`，
  统计 `logic` 被选中的比例（EFE 下）。
- 通过门：在 Q1 命中的 mission 上选择率 ≥ 30%，且**不**在 prose-only mission 上
  > 5%（否则是噪声选择）。

### Q3 效果：启用后是否可测地改善？
- 指标（每个 mission）：完成率（`reportMissionOutcome(goalAchieved=true)` 比例）、
  到完成的 tick 数、人工/规则评分（`deliverableInformationChars`、
  `looksLikeCollapsedProse` 命中率）、以及**工具结果被采纳率**（模型是否用了
  返回的绑定）。
- 设计：A/B 同任务集（≥30 任务/组），`logic.enabled` 为唯一变量，其它配置冻结。
- 通过门：完成率或 tick 数有统计显著改善（配对检验 p<0.05），且评分不下降。
  若只是"答案更严谨"但完成率不变 → 记为**可选增强**，默认仍关。

## 2. 若通过：最小集成方式（不是现在做）

1. 只在 mission 的 **验证阶段**调用（对已写好的 deliverable 做一致性检查），
   不进入生成路径——避免污染因果文本；
2. 事实/规则由**工具侧**从工作区文件抽取（`plan.md`、`constraints.md`），
   不让模型输出 Prolog 语法（框架适配模型，不反向）；
3. 结果以 `[verify:consistent]` / `[verify:conflict <atoms>]` 注入，作为
   `writeToolFeedback` 的一种，而不是拼进 prompt 正文。

## 3. 明确不做

- 不做否定即失败（NAF）、cut、算术求值——语义会悄悄改变；
- 不把 Prolog 引擎作为常驻服务 / MCP server（多一个进程、多一个失败点）；
- 不用它替代 GNN 或模型（两者解决不同问题：联想 vs 证明）。

## 4. 文献（调研起点，不是结论）

- Euclid-MCP：把 Prolog 确定性推理作为 MCP 工具暴露给 LLM
  <https://browse-export.arxiv.org/pdf/2607.21412>
- Prolog + 本体用于 NPC 决策 <https://hal.science/hal-04333537v1/document>
- MoE 路由中 base/expert 分工（对应"base 角色"待办）
  <https://arxiv-org.ezproxy.obspm.fr/html/2408.17280v1>
- LLM 是否会做离散算法（判断"让模型自己推理"是否够）
  <https://arxiv.org/html/2506.23408>

## 7. Q1 实测结果（2026-11-15，`tools/reasoning_demand_probe.py`）

数据源：`runtime_store/mission_workspace/*/deliverable.md`（有交付物的 mission）
+ `runtime_store/long_run/gateway.log` 的 deliberator tick 计数。

```
missions_with_deliverable : 842      （门槛 ≥30   ✅）
deliberator ticks         : 203      （门槛 ≥200  ✅）
含至少一类命中的 mission  : 91 (10.8%)（门槛 ≥20% ❌）
  constraint  61 (7.2%)
  prerequisite 40 (4.8%)
  box_completion 0 (0.0%)
提出疑问的 mission        : 378 (44.9%)
```

**结论：Q1 FAIL → 不存在可证明的需求，`logic.enabled` 保持 false。**

诚实说明：
- 分类器是**关键词代理**，不是语义判定；把同义改写计入会更宽松，但当前 10.8% 与门槛 20%
  差约 2 倍，合理调参不足以翻转结论；
- `box_completion` 为 0 是因为 deliverable 文本里很少出现"子盒/未完成"字样（该信息在
  `plan.md`/mission 状态里，不在交付物正文），这一类的代理效度较低；
- 44.9% 的 mission 提出疑问，但绝大多数是**生成型**提问（"写一个教程"），不是**判定型**
  提问（"这个计划与约束是否一致"）——这正是 Q3 要区分的。

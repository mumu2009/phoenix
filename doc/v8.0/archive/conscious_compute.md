# 封存：意识计算 / 集体意识计算 世界路由（conscious/collective compute）

> 状态：**已封存**（v8.0）。本文档是唯一权威记录；代码中不再提供这两个路由。
> 模板参照：`doc/v7.0/archive/uncontrolled_evolution.md`（封存原因 / 保留部分 / 重启条件 / 实现记录）。

## 1. 原始设想

- `world_model.hpp::buildConsciousComputePlan`（2546）与
  `buildCollectiveConsciousComputePlan`（2603/2846）：以"意识计算"与"集体意识计算"
  命名的世界状态推演计划，试图为世界模型产出"意识级"计算编排。
- 前端路由：`/world/conscious-compute`（frontend_server.cpp:6512）与
  `/world/collective-compute`（6567），返回上述计划。

## 2. 封存原因

1. **违反项目红线**："理论/科学驱动，不做伪科学"（AGENTS.md）。"意识计算"
  （conscious compute）与"集体意识计算"（collective conscious compute）没有
  可证伪的科学度量支撑，属命名层面的伪科学倾向——无论内部实现为何，暴露这样的
  API 就是在宣称系统具有它并不拥有的性质。
2. **没有消费者**：两个路由除自身外无任何调用方；不进 mission、不进 AGI/EFE 循环、
  不进聊天链路、前端 WorldPanel 也不展示。属"休眠 + 伪科学"双重不合格。
3. **可被更克制的命名替代**：`/world/cognitive`（6421，认知状态）与 `/world/brain`
  （6465，脑剖面）是同一功能的克制版本，继续保留。

## 3. 保留在代码中的部分

- 无。两个计划函数与两条路由随封存一并移除（v8.0）。
- `world_model.hpp` 其余部分（预测→对齐→校准闭环）不受影响。

## 4. 重启条件

同时满足以下条件才可恢复（缺一不可）：

1. 存在**明确的、可证伪的意识科学度量**（如 Integrated Information Theory 的
  Φ 值的可计算实现、Global Workspace 理论的广播-竞争形式化），并且该度量通过
  `doc/v7.0/testing_methodology.md` 的模块验收门槛（可复现、有基线对照）。
2. 存在真实消费者：mission 或 AGI/EFE 循环把该度量作为数值信号接入
  （对齐分模式，见 `doc/v8.0/config_world_audit.md` §2.1）。
3. 命名改为度量本身（如 `/world/iit-phi`），不再使用"意识"修辞。

## 5. 实现记录

- v8.0：审计判定封存（`doc/v8.0/config_world_audit.md` W5）；移除两个计划函数与
  两条路由；本文档建档。
- 审计确认 `/world/cognitive`、`/world/brain` 保留（克制命名、无伪科学主张）。

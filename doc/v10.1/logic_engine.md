# 符号推理层（Prolog/Datalog 风格）— v10.1

## 1. 为什么需要

Phoenix 的推理链是**概率的**：8B 模型写散文、GNN 给联想、tiny-expert 给打分。
它们都不能回答"**是否必然成立**"这类问题：

- 这个计划与已声明的约束是否一致？
- 哪些子盒（child box）仍未完成？
- 目标 X 是否蕴含前置条件 Y？

缺少一层可以对命题**证明或反驳**的确定性推理。

## 2. 设计（`logic_engine.hpp`，header-only）

| 组件 | 说明 |
|---|---|
| `Term` | 变量 / 原子 / 数字 |
| `Literal` | `pred(arg, ...)` |
| `Rule` | Horn 子句：`head :- body` |
| `Engine::saturate()` | 前向链（Datalog），规则反复求值到不动点 |
| `Engine::solve()` | 后向链（SLD 归结，带深度上界） |
| `Engine::runJson()` | JSON 门面，供工具分发直接调用 |

内建谓词：`=`、`!=`、`>`、`<`、`>=`、`=<`、`true`（仅对 ground 项求值，
数字原子也可以被解析成数）。

### 2.1 语义边界（明确不做，而不是静默降级）

- **不做** 否定即失败（negation as failure）、cut、算术表达式求值。
  否定在 Horn 子句里会悄悄改变语义，宁可明确不支持。
- 比较要求两侧 ground；非 ground 比较返回 false（fail closed）。
- 每次推导都有上界：`maxIterations` / `maxFacts` / `maxSolutions` / `maxDepth`。
  循环规则库会**在界内返回**，不会挂死 mission tick。

## 3. 复杂度

- 前向链（朴素求值）：`O(I · R · J)`，`I` 迭代数（≤ 规则库高度）、`R` 规则数、
  `J` 单条规则 body 的连接代价（≤ `F^|body|`，`F` 事实数）。实现里对 `J` 加了
  `cap=512` 的绑定上限，因此单轮是 `O(R · min(F^|body|, 512))`。
- 后向链：`O(b^d)`，`b` 每层分支（事实匹配 + 规则匹配），`d` 为上界深度；
  同样有 `maxSolutions` 截断。
- 空间：事实集 `O(F · arity)`，无额外索引（规模在千级事实以内，够用）。
  若未来事实数上万，应加谓词/首参索引——记为已知扩展点。

## 4. 与主链的接线

```
mission deliberator ──{"tool":"logic", arguments:{facts,rules,query}}──▶
  CognitionAutonomyManager::executeAgiAction(spec.addonType=="logic")
    └─▶ phoenix::logic::runJson(payload)
    └─▶ {"ok":true,"solutions":[{...}],...} 作为工具结果回注交付物
```

- AGI 动作注册：`{"logic","tool","logic","First-order rule query"}`
  （`registerDefaultAgiActions`），因此 EFE 规划器可以像选 `math`/`search` 一样
  选它，无需给 base 模型写任何提示词（遵守 "框架适配模型" 的硬约束）。
- 网关路由：`POST /api/logic/query`（116）。

## 5. 测试

`tests/gtest/unit/logic/test_logic_engine.cpp`：

1. 前向链传递闭包（`ancestor`）推导 3 条新事实；
2. 后向链返回**全部**解（不是第一个解）；
3. 内建比较门控规则（`age >= 18`）；
4. 循环规则库在界内终止；
5. 非 ground 事实被拒绝（fail closed）；
6. 对象形式 literal；
7. JSON 数字参数不被退化成字符串。

## 6. 文献

- Euclid-MCP：把 Prolog 确定性推理作为 MCP 工具暴露给 LLM（本设计的工具化路线）。
  <https://browse-export.arxiv.org/pdf/2607.21412>
- Prolog + 本体用于 NPC 决策（规则库组织方式）。
  <https://hal.science/hal-04333537v1/document>

# SparkArray 作用域化（scopes）：可选、可多选

**目标（用户需求）**：SparkArray 是随机森林式集成投票，理论上对**广泛范围的模型**都有效。当前它只挂在聊天管线上（`spark.gnnScheduler.enabled` 单开关）。目标是把"哪些功能域启用 SparkArray"变成**可选且可多选**：用户可选择一个作用域、多个作用域共同生效、或整体禁用。

前期基础已就绪：绝大多数学习模块已完成异步化与微服务化（异步任务系统、模块挂载工厂、控制器池/分片管理器），SparkArray 本身就是"池 + 分片"之上的无状态投票层，天然可复用到多个域。

---

## 1. 作用域分类（scope taxonomy）

| scope | 含义 | 现有调用点 | 阶段 |
|---|---|---|---|
| `chat` | 主聊天管线：gnnScheduler 集成投票 → 注解注入 graphContext | 116_section_tail.inc（/api/chat 与 /api/transformer/chat 两处，由 spark.gnnScheduler.enabled 门控） | **已接入门（v8.0）** |
| `consensus` | dispatchBig 多轮共识（bigRounds） | 116_section_tail.inc 数组派发端点（bigRounds>1 走 dispatchBig） | **已接入门（v8.0）** |
| `voter` | llamaVoter 输出重投 | 顶层 `llama_voter.*` 配置组（v8.0 已删除未读的 `spark.llamaVoter.*` 重复组；voter 非 SparkArray 本体，走独立开关） | 阶段 2 |
| `evaluate` | 候选/置信评估（sparkTransformerCandidates）+ 数组派发端点本身 | 116_section_tail.inc（sparkTransformerCandidates 收集 + dispatch） | **已接入门（v8.0）** |

## 2. 配置语义（可选 + 多选 + 禁用）

```json
"spark": {
  "scopes": ["all"],       // v8.3 默认：同时作用于 gnn 与整个项目（主体交流模式）
  // v8.0 曾是 ["none"]：chat 域的 RF 注解垃圾污染 graphContext；v8.3 交流模式重写 annotation 后已默认全开
  // "scopes": [],                       // 遗留行为（沿用各功能自己的开关，向后兼容逃生口）
  // "scopes": ["chat", "consensus"],   // 多选：并列生效
  // "scopes": ["chat"],                // 单选
  // "scopes": ["all"],                 // 全部
  // "scopes": ["none"],                // 整体禁用
  "gnnScheduler": { "enabled": true, "...": "..." }
}
```

判定规则（已实现于 116_section_tail.inc 的 `sparkScopeEnabled(scope)`）：

1. `scopes` 缺省/为空数组 → 遗留行为（仅各功能自己的开关决定，向后兼容）；
   **v8.0 提交的默认值是 `["none"]`**（质量优先：chat 域的 RF 注解垃圾曾污染 graphContext 导致模型答非所问；实测见 AGENTS.md "RDK UI on :5081" 同轮记录）；
2. 非空 → **白名单语义**：列表包含该 scope 或 `"all"` 才启用；`["none"]` 或未列出 → 关闭；
3. 配置解析异常 → 回退遗留行为（fail-open，不因配置错误瘫痪聊天）。

## 3. 为什么随机森林对广泛模型有效（理论依据）

- **误差-分歧分解**（Krogh & Vedelsby 1995）：集成泛化误差 = 平均个体误差 − 平均分歧度。分歧来自"多个**独立/异质**预测器"；SparkArray 的控制器池天然提供异质个体（不同权重/组/扰动变体），与具体后端无关——无论 ollama、llama-server、bitnet 还是 native transformer，投票框架都成立。
- **模型无关性**：SparkArray 只消费 `dispatch → {reply}` 与文本嵌入亲和度，不触碰后端内部；作用域化正是把这一"后端无关投票层"复用到 chat/consensus/voter/evaluate 各域。
- 与 llamaVoter 的区别：llamaVoter 是**同一模型内**的掩码扰动投票（温度/topK/掩码）；SparkArray 是**跨控制器**集成。两者互补，分属不同 scope。

## 4. 实施计划

- **阶段 1（已完成，v8.0）**：`spark.scopes` 配置 + `sparkScopeEnabled` 门 + 接入 `chat` 域（两个聊天处理器）；
- **阶段 2（已完成，v8.0）**：`consensus`（bigRounds 路径加门，未启用时回退单轮 dispatch 并返回 consensusSkipped）、`evaluate`（数组派发端点整体门控，关闭时 503 + 明确错误）；`voter`（llamaVoter 独立路径，非 SparkArray 本体）保持既有开关；
- **阶段 3（已完成，v8.0）**：`/api/array/layers` 响应新增 `scopes` 字段（当前配置视图）；共享 `sparkArrayScopeEnabled()`（094_section_before_sparkarray.inc）统一判定，chat 域两个处理器改用共享实现；
- **测试**：每域开关组合（[]/单选/多选/all/none）× 行为断言进 testing_methodology §13；
- **迁移文档**：与 migration_backlog.md 联动（A8 中"SparkArray 活跃推理"在作用域化完成后重新评估）。

## 5. 当前状态与遗留

- **全部四个域已接入门**（chat/consensus/evaluate + layers 视图）；`voter`（llamaVoter）非 SparkArray 本体、保持既有开关。
- dispatchBig 与 llamaVoter 的完整调用点盘点 = 阶段 2 的第一步（116_section_tail.inc 中 `spark_` 的其余消费点）。

## 6. v8.3 主体交流模式（exchange）+ 综合输出

**背景**：v8.0 的 SparkArray 是"独立投票"——每个 controller 独立应答，`aggregateResults`
只做 token 投票（删词）并返回空 `reply`，annotation（`gnn_transformer_rf_stage3|rf=...`）
污染了 graphContext。

**v8.3 改造**（`095_class_sparkarray.inc`）：

1. **交流轮**（`exchangeRounds`，默认 2；设为 1 恢复旧行为）：第 1 轮各主体独立应答后，
   把亲和度最高的 3 个回复合成"黑板"（≤1200 字符）分享给**全部**主体，每个主体带着
   黑板再综合应答一次——低层主体（gnn 控制器层）之间、以及与主推理模块候选
   （`sparkTransformerCandidates`）之间有了真正的信息交流；
2. **综合输出**：`aggregate.reply` 不再为空——取交流轮中亲和度最高的综合回复；
   chat 管线把它作为 `sparkReply` 注入 `graphContext`（综合文本先于 annotation）；
3. **干净的 annotation**：`spark-exchange|rounds=N|agents=M|conf=...|chosen=...|top:...`
   （人类可读，不再是无意义 token 串）。

**配置**：`spark.scopes` 默认 `["all"]`（同时作用于 gnn 与整个项目）；
`spark.gnnScheduler.exchangeRounds`（默认 2）。成本：每 chat 请求的 controller 调用数
×2（7 控制器 → 14 次），RDK 实测由测试者记录；如延迟超标可设 `exchangeRounds=1` 回退。

## 7. 参考文献

- Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *NIPS*.
- Breiman, L. (2001). Random forests. *Machine Learning*.（投票/集成的模型无关性）

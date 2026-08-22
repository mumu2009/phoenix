# Context 隔离与跨 context 记忆（v8.x 设计）

**问题（用户）**：同时跑多个任务 + 对话时会不会串 context？
**要求**：每个任务与对话（所有对话视为一个任务）context 独立；底层允许跨 context
记忆模块（并拓展跨 session 记忆）。

## 1. 分层规则

| 层 | 内容 | 隔离方式 |
|---|---|---|
| **Context 内（严格隔离）** | 工作区文件/deliverable/plan/tool-feedback | `sanitizeScope` 按 missionId / children/<id> 分目录 ✓ |
| | 即时感知（sensation/instinct 评估） | `activeFor(contextTag)`：source 无 `:` = 全局可见；有前缀只对本 context 可见（v8.x 新增） |
| | 认知迭代输入（pain/novelty/surprise） | source 带 `mission:<id>:` / `chat:<session>:` 前缀（v8.x 新增） |
| | 输出缓存/循环守卫 | `workspaceCache` 按 scope 键 ✓ |
| **跨 context 记忆（唯一交流点）** | GNN 图（outline/经验回灌） | 共享（长期不变图）✓ |
| | mission_experience（任务级经验） | 共享，检索注入 ✓ |
| | **cross_context_memory（会话级，v8.x 新增）** | 共享：chat 每轮沉积摘要、mission 完成沉积摘要；任何 context 用 ccmRecall 检索注入 |
| | world model store | 共享（RAG 证据库）✓ |
| | AGI 学习器 / episodic / subconscious 人格 | **声明为跨 context 记忆层**：学习到的策略与人格全局共享（不是串扰，是设计） |

## 2. 已验证的串扰点与修复

1. **mission 压力污染 chat 情绪**（已修）：压力 source `mission-pressure` →
   `mission:<missionId>:pressure`；iterate 评估用 `activeFor(ctxTag)` 过滤——
   chat 迭代再也看不到任务压力，mission 迭代看不到 chat 噪声。
2. **novelty/surprise 信号混用**（已修）：`world-uncertainty` →
   `<ctxTag>:novelty`；AGI 前向模型惊喜 → `<ctxTag>:surprise`。
3. **chat 与 mission 共用 iterate 状态**：AGI 学习器/episodic 保留共享（见上表
   声明为跨 context 层——策略学习全局受益）；即时评估已隔离。
4. **deliberator prompt 与 chat graphContext**：各自独立组装（mission 用
   goal/workspace/outline/经验；chat 用 graphContext 管线），互不读取对方活状态 ✓。

## 3. cross_context_memory（跨 session 记忆模块）

- 文件：`cross_context_memory.hpp`（header-only，与 mission_experience 同风格）；
- 存储：`runtime_store/cross_context_memory.json`（≤500 条，LRU）；
- 接口：`ccmRemember(path, sourceTag, text)` / `ccmRecall(path, query, k)`（词袋重叠检索）；
- 写入：chat 每轮（`chat:<session>`：用户文本⇒回复摘要 ≤400 字符）、mission 完成
  （`mission:<id>`：结构摘要）；
- 读取：chat graphContext 注入 `[cross-context memory]`（top-3）、mission deliberator
  静态前缀注入（top-2）——任务与对话都在"系统已学到的知识"上起步，但活状态永不互渗。

## 4. 多任务并存（已实现）

- **MissionLifecycle 容器化**：`missions_` map 按 missionId 存独立生命周期（状态机/交付物/
  子盒/lineage 各归各）；`/api/mission/assign` 相同 id = 重启该任务，新 id = 并行新任务，
  互不覆盖；`defaultMissionId_` = 最近 assign（旧单任务调用无需传 id 自动兼容）。
- **心跳轮转**：loopRun 每 tick 快照全部 Running missions 各自推进（父代先行 + 各自子盒），
  `autonomyLoop.maxMissionsPerTick`（默认 0=全部）限流；自暂停按 mission 分槽
  （missionPauseTicks_ map）。
- **压力隔离**：每个任务的 pain 带自己的 id 前缀注入（`mission:<id>:pressure`），
  各 context 只评估自己的压力；元心跳（空 contextTag）看全部压力作为跨任务调度信号。
- **API**：status 返回 `missions` 数组 + `defaultMissionId`；report/replicate/file/lineage
  接受 `missionId`（缺省 = 默认任务）；deliberator 从 scope 解析 missionId 并注入工具上下文。
- **共享（跨 context 层）**：AGI 学习器、genome 基线、GNN 图、经验库、lineage 审计。

## 5. 验收要点（测试者）

- 开一个 mission（压力增长中）同时聊天：chat 的 emotionTensor/inferenceOptions
  不应随压力变化（对比 mission 前后）；
- 两个不同 session 的 chat 情绪互不影响；
- mission 完成后：chat 提问相近主题能 recall 到 `[cross-context memory]` 注入；
- 新 mission 的静态前缀含 `Cross-context memory` 段；
- `cross_context_memory.json` 增长且 ≤500 条。

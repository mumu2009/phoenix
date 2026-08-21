# 任务层（Meeseeks 盒）——目标域化与时间紧迫性

本文回答两个问题：

1. **AI 的角色不必然是“服务用户”**：当一个实例的角色是“在自身生命周期内解决一个目标”（例如“在 T 秒内给出答案”），把“用户反馈 → 感受”作为唯一驱动就不合理。任务层把**估值来源从外部反馈移到内部目标完成判定**：目标完成 = 疼痛终结，估值完全在系统内部闭合。
2. **“Meeseeks 盒”是否可以落地**：可以，而且**不需要独立的运行时外层**。它是一个**进程内的生命周期状态机 + 可遗传基因组对象**（`mission_lifecycle.hpp` / `mission_lifecycle.cpp`）：出生绑定单一任务 → 疼痛随时间线性增长 → 只有目标被判定完成疼痛才停止 → 完成后按完成时间做选择压力，产生“繁殖/遗传/变异”。

所有功能**可选**、默认关闭（`config/phoenix.json` 的 `mission.enabled = false`）。

---

## 1. 形式化：时间递增疼痛 = 稳态紧迫性（allostatic urgency）

**定义**：一个任务（Mission）绑定单一目标（goal）。任务运行期间，疼痛强度按**用户可选的增长模式**（配置 `mission.pressureMode`，默认 `asymptotic`）增长：

$
p(t) = \begin{cases}
P_{\max}\,\tanh(t/\tau), & \text{asymptotic (default)},\\[2mm]
\min\{ g \cdot t,\; P_{\max} \}, & \text{linear},\\[2mm]
P_{\max} \cdot \dfrac{\ln(1+t)}{\ln(1+H)}, & \text{logarithmic},\\[2mm]
\mathrm{clamp}_{[0,P_{\max})}\big(f(t)\big), & \text{expression (user formula)}.
\end{cases}
$

其中 t 是任务开始后的经过秒数，g 是线性增长率（`mission.painGainPerSec`），H 是对数模式视界（`mission.pressureHorizonSec`，默认 3600），τ 是渐近模式时间常数（`mission.pressureTauSec`，默认 1800），P_max 是疼痛上界（`mission.maxPain`）。`expression` 模式用 `mission.pressureExpr` 求值（变量 `t/H/g/Pmax/tau`，支持三角/反三角/双曲/反双曲/多项式/指数/对数等初等函数）；结果钳制在 `[0, P_max)`，永不到顶。

**为什么默认渐近 tanh**：`tanh(t/τ)` 随时简趋近 P_max 但有限时间内永远到不了——长任务始终保留可用梯度，避免压力打满后模型幻觉/忘目标。线性/对数仍保留；表达式模式允许用户自定义任意初等公式。

**稳态（allostasis）诠释（Sterling 1988）**：疼痛是“目标未完成”这一稳态偏差的**信号强度**，g 是该偏差的**精度（precision）**。稳态偏差不随时间缩小（任务未完成），而系统的稳态代价随偏差持续积分，于是“越拖越痛”——这正是紧迫感的可计算定义。它与饥饿/疲劳同类：都是偏差驱动、可被趋利避害层最小化的原生感受，而不是一条硬编码的规则。

---

## 2. 定理：总疼痛 = g·T²/2，最小化总疼痛 ⟺ 最小化完成时间

**命题**：设疼痛率按 p(t)=min(g·t, P_max) 增长，任务在时刻 T 完成（t ≥ T 时 p=0），T₀ = P_max/g 为饱和时刻。则任务期间累积总疼痛

$$
P(T) = \int_0^T p(t)\,dt =
\begin{cases}
\dfrac{g}{2} T^2, & T \le T_0,\\[2mm]
\dfrac{g}{2} T_0^2 + P_{\max}(T - T_0), & T > T_0
\end{cases}
$$

并且 P(T) 在 T ≥ 0 上**严格递增**，因此“最小化累积疼痛”的策略与“最小化完成时间”的策略等价。

**证明**：

1. T ≤ T₀：P(T) = ∫₀ᵀ g·t dt = gT²/2，dP/dT = gT > 0（T>0）。
2. T > T₀：P(T) = ∫₀^{T₀} g·t dt + ∫_{T₀}^T P_max dt = gT₀²/2 + P_max(T−T₀)，dP/dT = P_max > 0。
3. 两段在 T=T₀ 处连续：gT₀²/2 = g(P_max/g)²/2 = P_max²/(2g)，右侧同值。

故 P 是 T 的严格增函数，argmin_T P(T) = argmin_T T。∎

**对数模式的闭式总疼痛**（默认模式）：设 p(t) = k·ln(1+t)，k = P_max/ln(1+H)，任务在 T 完成，则

$
P_{\log}(T) = \int_0^T k \ln(1+t)\, dt = k\,[(T+1)\ln(T+1) - T]
$

dP/dT = k·ln(1+T) > 0（T>0），且 p(T) 在 T ≤ H 段严格递增、T > H 段 p=P_max 恒定，故 P_log(T) 依然**严格递增**——等价性保持，只是闭式形式从 gT²/2 变为上式的对数积分。∎

**为什么这足以“强迫”最快完成**：Phoenix 已有的趋利避害/主动推理循环本来就最小化感受层的稳态代价（`homeostaticCost = Σ gain·|intensity − setpoint|`，Pain 的 setpoint=0）。把 p(t) 注入为 Pain 感受后，疼痛贡献的稳态代价对 T 单调递增，于是**原有闭环不需要任何新优化器**就会把“尽快完成任务”内化为偏好。这继承主动推理的 pragmatic 项：任务完成是终止状态的**先验偏好**（prior preference for terminal state, Friston），疼痛是该偏好未满足时的负对数似然 −ln p(o|C) 的实例。

---

## 3. 诚实的边界：这不是无条件的等价（奖励塑形警示）

Ng, Harada & Russell (1999) 证明：只有**势函数塑形**（potential-based shaping，Φ(s) − γΦ(s′) 形式）才不改变最优策略。时间递增疼痛是**时间依赖**信号，不属于势函数塑形——在 γ<1 的折扣收益下，逐步“痛”与总疼痛 gT²/2 的等价性会被折扣打破，理论上可能诱导策略偏离“真正的最短完成”。因此本节明确声明边界并给出缓解：

1. **等价性的成立条件**：任务层压力的作用通道是稳态驱动（allostatic drive，γ 不起作用的部分），而不是替换任务本身的成功判定；pain 只进入 `homeostaticCost` 与本能评估，不直接写入奖励。
2. **完成判定永远由显式门控**：`reportMissionOutcome({goalAchieved:bool})` 是疼痛停止的唯一途径。疼痛强度再高也不会“自动宣告完成”——防止走捷径（例如宣称完成而不交付结果）的唯一办法是让完成判定独立于疼痛。
3. **参数约束**：g 取小值（默认 0.01/s，300 秒任务下末端疼痛 ≈ 3，被 P_max=1 截断），避免疼痛压过探索（探索本身需要时间，见下节权衡）。
4. **适用域**：仅用于**目标域化实例**（goal-scoped instance）。面向开放对话的服务实例不应开启任务层——这也正是“模块可选、默认关闭”原则的含义。

**探索-利用权衡**：疼痛单调增加会系统性压低 epistemic 探索。缓解方式：EFE 的 epistemic 项权重（`agi.epistemicWeight`）与 `adaptiveExploration` 保持独立；压力只作为稳态项参与 `driveCost`，不与探索项直接竞争增益。这正是“任务层不替换、只包裹”的设计。

---

## 4. 繁殖 / 遗传 / 变异：复制是实例的自由能力（使命必达，非自然选择）

**定位（2026-08 修订三）**：**复制的权力交给实例本身**。`replicate` 是一个普通的规划器动作（与 math/search/computer 同级），实例经由自己的 EFE 规划**自由决定何时复制**——没有固定触发条件（不由压力阈值触发）、没有接班过程。这正对应 Meeseeks 盒：任务未完成前，压力只增不减、实例不会中途结束；它唯一"召唤帮手"的理由就是它自己判断"多一个我为同一目标工作"有利。

- **不是人来做**：`executeAgiAction(category="replicate")` 在 `iterate()` 的规划-执行环内由实例自己选择；
- **不是自然选择**：没有适应度比较、没有淘汰、没有最优父本保留——`MissionGenome::mutate`（高斯扰动 + clamp）只为多个并行尝试提供多样性；
- **唯一护栏**：`mission.maxReplicas`（默认 4）封顶复制数，防无限繁殖风暴；`mission.enabled=false` 仍是总开关。

**基因组（heritable genome）**：`MissionGenome` = 可遗传参数集合：

- `SubconsciousProfile`：PAD 气质基线、temperamentStrength、riskAversion、anticipatoryGain、每感受 {gain, halfLifeSec, setpoint}、自定义本能表；
- `learningRate`：价值学习器（TD(0)）的学习率。

**复制（replicate）**：`MissionLifecycle::replicate(rate)` 变异自身基因组，记录后继 `MissionChild{id, genome, goal, bornMs, generation}`，并在 `sessions_` 中创建绑定同一目标的新会话（携带子基因组）。父与子并行工作同一目标；父完成即目标完成。

**变异**：对每个可遗传标量 x′ = clamp(x + σ·N(0,1), lo, hi)，σ = rate（Beyer & Schwefel 2002 变异算子，仅作多样性来源）。

**监督者信号**：`stats()` 暴露 `completionTimeMs`、`children`、`maxReplicas`；进程内**没有** fitness 字段、没有按完成时间的自动选择——选择（是否停用、是否采纳某变体）永远属于人。原"不受控 (1+λ)-ES 自进化"设想仍封存于 `doc/v7.0/archive/uncontrolled_evolution.md`。

**为什么不是跨运行时层**：后继是同一进程内的新会话 + 一个 `MissionChild` 记录；`replicate` 是 O(基因组维数) 纯计算，无第二进程、无协议、无翻译损耗。
## 5. 实现映射与用法

| 层 | 位置 |
|---|---|
| 状态机 / 基因组 | `mission_lifecycle.hpp`、`mission_lifecycle.cpp`（`phoenix::mission`） |
| 接入自主栈 | `autonomy_stack.hpp/.cpp`：`assignMission` / `missionStatus` / `reportMissionOutcome` / `spawnMissionChild`；`iterate()` 在本能评估前注入 Pain（`source="mission-pressure"`，valence=−1），并在结果中暴露 `mission` 字段 |
| 网关配置 | `main_hub_parts/111_class_gatewayserver.inc` 加载 `mission.*`，`enabled && goal 非空` 时出生即绑定任务 |
| 配置 | `config/phoenix.json`：`mission.{enabled, painGainPerSec, maxPain, defaultDeadlineSec, mutationRate, id, goal}` |

配置示例：

```json
"mission": { "enabled": true, "goal": "在本实例生命周期内解出该问题并给出最短答案",
             "maxReplicas": 4,
             "painGainPerSec": 0.01, "maxPain": 1.0, "defaultDeadlineSec": 300,
             "mutationRate": 0.05 }
```

API（C++ 层，与 `configureAgi` 同模式）：

```cpp
cognition.assignMission({{"enabled", true}, {"goal", "..."}, {"painGainPerSec", 0.01}});
// ... 任务执行期间，压力自动注入 Pain 感受并抬高稳态代价 ...
cognition.reportMissionOutcome({{"goalAchieved", true}});   // 疼痛源停止
auto child = cognition.spawnMissionChild({{"mutationRate", 0.05}}); // 子代基因组
```

HTTP API（v8.0 起，人工监督控制台——"生命周期开始时设立问题"的用法）：

| 路由 | 方法 | 语义 |
|---|---|---|
| `/api/mission/status` | GET | `{enabled, stats{mission, pressure, generations, spawns, completions, completionTimeMs, children}, genome}` |
| `/api/mission/assign` | POST | body `{goal, deadlineSec?, painGainPerSec?, maxPain?, mutationRate?, maxReplicas?, pressureMode?, pressureHorizonSec?, pressureTauSec?, pressureExpr?}`；goal 必填，自动置 `enabled=true` 并**自动启动自主心跳**。压力模式（§1）：`asymptotic`（默认，Pmax·tanh(t/τ)，永不宣称饱和）/ `linear` / `logarithmic` / `expression`（自定公式，失败回退 asymptotic） |
| `/api/mission/report` | POST | body `{goalAchieved}`：进程外判定门（完成/失败），`completionTimeMs` 是给人类监督者的选择信号 |
| `/api/mission/replicate` | POST | body `{mutationRate?, subgoal?, parentId?}`：召唤一个子任务盒子（§5.6）；空/相同目标自动降级为 "assist with: <父目标>"；`parentId` 为空=根级，否则挂在指定盒子下（任务树，深度默认无限、可用 `mission.maxReplicaDepth` 限制） |
| `/api/mission/file` | POST/GET | body `{action: list\|read\|write\|append\|delete, path, content?}`：任务工作区的沙箱文件操作（§5.5） |

配套：`/api/cognition/autonomy/{status,interject,loop}`（插话/amendGoal/心跳循环）、
`/api/safety/estop[/status]`（急停）。前端 Mission 控制台即上述路由的消费方
（079project_frontend/src/components/MissionPanel.js）。

## 5.5 文件 IO 工作通道与工具调用协议（v8.1）

**为什么**：纯"只追加"的交付物循环必然退化——模型每次都带着前文续写，没有能力重写，
输出变成无限重复。文件 IO 让任务 worker 变成**迭代编辑**：先写一个极简版本，再读取→
重写→扩充；工具调用让插件（math/search/research/web/replicate/mcp.*）真正可被 agent 使用。

**工作区**：`mission.workspaceDir`（默认 `./runtime_store/mission_workspace`，env
`AI_MISSION_WORKSPACE`），每个任务一个子目录 `<missionId>/`。所有操作经
`phoenix::mission::workspaceExecute`（`mission_workspace.hpp`）沙箱化：拒绝绝对路径与
`..`，单文件上限 256KB。路由：`POST /api/mission/file` body
`{action: list|read|write|append|delete, path, content?}`（控制台与外部测试亦可直接使用）。

**每 tick 的协议**（任务 worker 的 deliberator，`111_class_gatewayserver.inc`）：
   prompt 携带 目标 + 工作区文件列表 + **滑动窗口上下文包**（钉住的摘要 / 可选
   GNN 摘要 + 近期全文，总预算对齐 `mission.ctxSize` 或 `llama_server.ctx_size`，
   默认 4096，可选 16384）+ 工具清单；
   用户在 Mission 面板选择 `contextPack=summary|full_and_summary` 与
   `includeGnnSummary`。
模型回复必须是以下三选一：

1. **纯文本** → 追加到 `deliverable.md`（用于第一次极简草稿）；
2. **JSON 文件操作** `{"action":"write"|"append"|"read"|"list","path":"...","content":"..."}` → 执行沙箱文件操作；
3. **JSON 工具调用** `{"tool":"<action-name>","arguments":{"userPrompt":"..."}}` → 经
   `executeAgiActionByName` 走已注册的 AGI 动作执行器（math/search/research/web/replicate/mcp.*）。

工具/文件操作的结果以 `[tool:...]` / `[file:...]` 记入交付物，下一 tick 的 prompt 会看到——
这是 agent 获得"行动-反馈"闭环、从而有逻辑地迭代（而非循环）的关键。

## 5.6 子代 = 使命必达盒子工具（v8.2）

**理念**：父代把子代看作**有界任务帮手**（Meeseeks box），而不是继承者。父代通过
工具调用 `{"tool":"replicate","arguments":{"userPrompt":"<子任务>"}}` 召唤一个盒子；
盒子在自己的工作区（`<workspaceDir>/<missionId>/children/<childId>/`）完成子任务、
产出自己的 `deliverable.md`；父代用文件工具读取（`children/<childId>/deliverable.md`）
并把最佳段落合并进自己的交付物。

**限制（父代不得全权甩包）**——三条机械约束：

1. **子任务有界**：`recordChild` 把空请求或"与父目标完全相同"的请求改写为
   `"assist with: <父目标>"`——父代永远无法把自己的整个目标原样交给子代；
2. **父代永远先干活，子代可同时推进**：心跳每 tick 先跑父代 deliberation，再推进
   **全部**活跃子盒子（顺序调用 LLM；RDK 上默认 `--parallel 2`，仍由
   `llamaCallMu_` 串行化网关侧调用以避免互相 cancel）。可用
   `autonomyLoop.maxChildrenPerTick`（默认 0=不封顶）限制每 tick 推进的子盒数量；
   子盒 token 预算为 `childDeliberateMaxTokens`（默认 256）。父代可同时存在多个
   帮手（上限仅 `mission.maxReplicas`）；
3. **深度由用户决定**：盒子可以召唤自己的盒子（深层任务树）。默认 `mission.maxReplicaDepth = 0`
   （**无限递归**）；用户可设置 >0 的上限（env `AI_MISSION_MAX_REPLICA_DEPTH`）。
   每条复制记录 `parentId`（谁召唤的）+ `depth`（树深），子孙各自独立工作区
   （`children/<id>/deliverable.md`），同 tick 内一并推进。

判定标的始终是**父代的 `deliverable.md`**：`completionTimeMs` 与人工判定只看父代产出；
所有盒子都是建议来源（多样草稿），每一层父级负责取舍与最终质量。

实现映射：`MissionChild{subgoal, goal, parentId, depth}`、
`MissionLifecycle::recordChild(parent, rate, subgoal, parentId)` + `maxReplicaDepth` 访问器、
`executeAgiAction` replicate 分支读取 `context.userPrompt`（子任务）+ `context.parentId`（血统）、
`loopRun` 子代轮转、deliberator 在子代模式向工具上下文注入 `parentId`、`mission_workspace.hpp`
`sanitizeScope`（按 `/` 分段沙箱化）、`spawnMissionChild` 接受 `subgoal/userPrompt/parentId`。
## 6. 参考文献

- Sterling, P. (1988). Allostasis: a model of predictive regulation. *Physiology & Behavior*.
- Friston, K. (2013). Life as we know it. *Journal of the Royal Society Interface*.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations. *ICML*.
- Beyer, H.-G., & Schwefel, H.-P. (2002). Evolution strategies: a comprehensive introduction. *Natural Computing*.
- 与既有模块的衔接证明见 `active_inference.md`（EFE/MPC）与 `subconscious.md`（稳态调谐）。

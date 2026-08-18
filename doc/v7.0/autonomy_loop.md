# 长期自主循环 / 自主推理 / 自主进化 + 人类插话（heartbeat + persistence + interjection）

本文把"自主循环、自主推理、自主进化"从"有外部消息才动一下"升级为**长期化**：一个内部心跳让计划-行动-观察-学习闭环在没有用户消息时持续运行，并把学到的状态持久化到磁盘（重启不丢）。同时补齐**插话**能力：生命周期一开始就设定目标之后，人类可以在中途注入新指示、甚至改写目标。

所有能力**可选、默认关闭**（`autonomyLoop.enabled=false`）；人类保有总开关。

---

## 1. 心跳 = 真正的自主循环（长期化）

`CognitionAutonomyManager` 内置一个心跳线程（`loopRun`），每 `intervalSec` 秒执行一个 tick：

1. `ensureHeartbeatSession()`：保证存在一个 `__autonomy_heartbeat__` 会话（shouldIterate=true），使 `iterate()` 全流程可用；
2. 最多 `maxStepsPerTick` 次调用 `iterate({}, {})`——每一次都是完整的"本能评估 → 感受衰减/注入 → 主动推理规划（EFE/MPC）→ 执行所选工具（math/search/computer/goal_advance/replicate/MCP）→ observeRewarded（TD(0) 价值学习 + 前向模型在线更新）→ 自适应探索"；
3. 每 `persistEveryTicks` 个 tick 把 `exportState()` 写入 `persistPath`；
4. `startAutonomyLoop()` 启动前若存在持久化文件则先 `importState()` 恢复。

这就是"自主循环"的实现语义：循环不需要外部消息驱动；`maxStepsPerTick` 限制每 tick 的推理预算（有界、可控）。

---

## 2. 持久化 = 自主进化的长期化

`exportState/importState` 现在覆盖**全部可学习状态**（此前只存会话表，进化是"假的长期化"）：

- `ActiveInferenceController` 全量序列化（`toJson/fromJson`）：价值头 w、前向模型 A/B/b、情景记忆（上限 1024 条）、surprise EMA、风险态度、探索状态、动作空间；
- 感受引擎（`PrimalSensationEngine::toJson/fromJson`）、任务生命周期（`MissionLifecycle::toJson/fromJson`：mission + genome + children + maxReplicas）、潜意识剖面、goals 队列、agiLatentState/lastAgiAction。

理论依据：TD(0) 价值学习与在线前向模型学习都是**增量在线算法**（Sutton & Barto 1988；Tsitsiklis & Van Roy 1997）——它们的状态就是"进化结果"，序列化即长期化；情景记忆重放（consolidate）跨重启延续（睡眠式巩固的持久化）。

---

## 3. 自主推理的闭环语义（诚实边界）

每个 tick 的推理 = EFE 规划器在当前动作空间（工具 + replicate + MCP 工具）上的单步最优选择 + 真实执行 + 观察回灌。多步推理由心跳的连续 tick 天然构成（上一步的工具结果进入下一步的上下文与状态）。

**边界（不吹）**：
1. 学习质量取决于反馈质量：没有用户消息/工具结果时，netUtility 接近中性，TD 更新幅度小——循环在跑，但"进化素材"来自真实交互（这正是搜索插件的定位：实例可自主检索为自己找材料）。
2. 心跳默认关闭；开启后每个 tick 的 CPU 开销受 `maxStepsPerTick` 与 EFE 动作枚举复杂度约束（O(动作数 × 时域 × dim²)）。
3. 单进程内的心跳是单实例循环；"多实例并行"由 replicate 召唤的子会话在同一进程内共享调度。

---

## 4. 插话（interjection）：中途提示与目标改写

对应需求："生命周期一开始就设定目标之后，能不能中途提示？"——能，两个层级：

1. **注入指示**：`interject({text})` 入队（上限 64 条），下一个 tick/回合被 `iterate()` 消费：作为该回合的 userPrompt 并把全部未消费插话以 `[human interjections]` 块注入 cognitionModulation——中途提示不打断任务、不改目标；
2. **改写目标**：`interject({text, amendGoal})` 调用 `MissionLifecycle::amendGoal()`——任务保持 Running、**startMs 不变**（压力继续增长），只是目标被重定向（任务被重定向而非重启）。

网关路由：`POST /api/cognition/autonomy/interject`、`POST /api/cognition/autonomy/loop`（configure/start/stop）、`GET /api/cognition/autonomy/status`。

配置：

```json
"autonomyLoop": { "enabled": false, "intervalSec": 10, "maxStepsPerTick": 8,
                  "persistEveryTicks": 5, "persistPath": "runtime_store/autonomy_state.json" }
```

---

## 5. 测试协议

见 `testing_methodology.md` §11：心跳无外部消息自增 iteration、插话消费一次、目标改写保持 Running 且 startMs 不变、进化状态导出/导入往返一致、控制器全量序列化往返。

---

## 6. 参考文献

- Sutton & Barto (1988/2018), *Reinforcement Learning*: TD(0) 在线增量学习；
- Tsitsiklis & Van Roy (1997): TD(λ) 收敛性（线性函数逼近下的在线学习正当性）；
- Friston et al. (2015), Active inference and epistemic value（EFE 规划的长期循环语义）；
- 与项目内既有推导的衔接见 `active_inference.md`、`mission_layer.md`。

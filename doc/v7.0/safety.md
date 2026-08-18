# 系统级急停（E-stop）：多实例安全层

单实例安全已有 memebarrier（图谱安全巡检与可疑节点隔离，日常常禁用）。本文新增**系统级**安全层：一个闩锁式急停按钮——按下即杀死全部已登记实例，然后关闭自身，防止 AI 不受控。

---

## 1. 语义（工业急停的四个性质）

1. **闩锁（latching）**：`press()` 永久置位，代码无法解除（只能进程重启清除；`resetForTesting` 仅测试用）。能被"被保护对象自己"解除的急停不保护任何东西——fail-safe 要求开关单向。
2. **最高优先级**：闩锁位在 `iterate()`/`interject()`/`startAutonomyLoop()` 入口与心跳每 tick 之前检查；一旦置位，任何实例拒绝新工作。
3. **杀全部实例**：实例在**生命周期开始时登记**（`InstanceRegistry::registerInstance`：自主循环启动、任务生命周期开始时登记，析构注销）并携带自己的停堆处理函数（停心跳、停 MCP 子进程等）。`press()` 依次调用所有处理函数（每函数异常被捕获、逐项报告）。
4. **自我关闭**：停止全部实例后调用关停处理函数——网关安装的处理函数会先尽力持久化进化状态（`runtime_store/estop_state.json`）再 `drogon::app().quit()`，进程退出，系统不可能继续不受控运行。

分层（defense in depth）：memebarrier（单实例、图安全）→ 本模块（多实例、进程级）→ 进程退出（最后的物理边界）。这与工业功能安全（IEC 61508）中"急停回路必须独立于被控系统的正常控制路径、且为最高优先级联锁"的设计同构。

---

## 2. 实现映射

| 组件 | 位置 |
|---|---|
| 实例登记表 | `instance_registry.{hpp,cpp}`（`phoenix::safety::InstanceRegistry`，进程级单例，登记/注销/快照/stopAll） |
| 急停闩锁 | `emergency_stop.{hpp,cpp}`（`phoenix::safety::EmergencyStop`：press/status/latched/setShutdownHandler） |
| 实例接入 | `autonomy_stack`：`startAutonomyLoop()` 与任务 `assignMission` 时登记；停堆处理函数=停心跳+停 MCP+（非循环线程上 join）；析构注销 |
| 网关 | 关停处理函数（持久化→quit）；路由 `POST /api/safety/estop`（{reason}）、`GET /api/safety/estop/status`；配置 `safety.estop.enabled`（**默认 true**） |

**默认开启的理由（对"模块默认关闭"原则的唯一例外）**：安全装置默认关闭等于没有安全装置；它不产生任何推理/进化行为，只提供一条人握在手里的停止通路。

---

## 3. 诚实边界

- 作用域是本进程与本机：登记表与闩锁都是进程内状态；跨主机"杀死所有实例"需要基础设施层（消息总线/电源级控制），不在本文范围。
- 处理函数是协作式的：失控的死循环线程（无检查点）无法被协作停止——因此心跳每 tick、iterate/start/interject 入口都检查闩锁，把"不合作窗口"压到最小；进程退出是最终保证。
- `press()` 幂等：第二次按下返回原报告，不重复执行停堆/关停。

---

## 4. 测试协议

见 `testing_methodology.md` §12：登记/注销/stopAll、闩锁一次性、关停处理函数恰好一次、闩锁阻断 iterate/interject/start、急停杀死运行中心跳（1.5s 内 running=false）。

## 5. 参考文献

- IEC 61508（功能安全）：安全联锁独立于正常控制路径、最高优先级；
- 急停电路惯例（NFPA 79 / ISO 13850）：闩锁、单向、人工复位；
- 与 memebarrier 的分层关系见 `algorithm.md` §7/§23。

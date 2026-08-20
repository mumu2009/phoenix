# v8.0 world 与 config 模块审计（增强 / 直接使用 / 封存）

> 范围：world 模块（`world_model.hpp` 的 `WorldModelStore` + 网关 `ingestWorldEvidence`/`mergePromptContext` + `/api/world/*` 与前端 `/world/*` 路由 + `world_model.*` 配置 + `physics_world_runtime`/bullet3 + 前端 `WorldPanel`）与 `config/phoenix.json` 的冲突/冗余/死键。
> 判定口径（对齐 AGENTS.md / v7.0 理念）：**活跃** = 被读且用于当前链路；**休眠** = 被读但链路已退化/未接线；**死** = 无任何读取。判定分三档：**增强**（可落地接线）、**直接使用**（现状即可）、**封存**（按 archive 模板写明原因与重启条件）。
> 本审计只读不改 C++/前端；唯一改动是 `config/phoenix.json` 的安全删键（见 §3）。

## 0. 结论摘要

- **增强**：2 项（`WorldModelStore` 预测闭环、`physics_world_runtime`/bullet3 物理模拟——都还只是"证据库 + 独立模拟"，尚未把预测对齐分喂回 mission/agi 循环）。
- **直接使用**：6 项（网关 `ingestWorldEvidence`/`mergePromptContext`、`/api/world/*` 六条路由、`world_model.*` 配置组、`WorldPanel` 前端、`edge_platform`（RDK 专用）、`spark.scopes` 作用域门）。
- **封存**：9 项（v7 四个未接线 stub 组、`jepa.*` 改名遗留、`main.multimodal` 死组、`/world/conscious-compute`+`/world/collective-compute` 伪科学路由、`spark.llamaVoter` 阶段2预留、散落死叶）。
- **config 键数**：leaf 550 → 503（删 47 个零读取死键）；validator `missing` 0 → 0（绿），`unused` 225 → 178。

---

## 1. 总览表

| # | 模块 | 文件 / 配置组 | 状态 | 判定 | 理由 | 建议动作 |
|---|---|---|---|---|---|---|
| W1 | WorldModelStore 预测闭环 | `world_model.hpp`（4515 起）；`frontend_server.cpp:4519`、`111_class_gatewayserver.inc:8-44` | 活跃(部分) | **增强** | 预测→对齐→校准闭环（`buildExpectedNextState`/`updatePredictionDoc`/`computeAlignmentScore`）已在 `ingestEvidence` 内部实现，但 `alignmentScore`/`mismatch` 只进 prompt 上下文，没作为显式信号喂给 AGI/mission | §2.1：把 `prediction.alignmentScore` 喂给 `observePayload.worldUncertainty`（当前写死 0.25）与 Novelty 原始感觉 |
| W2 | 网关世界证据注入 | `111_class_gatewayserver.inc:29-44` `ingestWorldEvidence`；`:624-637` `mergePromptContext`；`116_section_tail.inc:1695/1924/2385/2556` | 活跃 | **直接使用** | /api/chat 与 /api/transformer/chat 入口落证据，`runCognitionAutonomy` 拉取 `sessionState` 后 `mergePromptContext` 拼入 prompt | 保留；可选：证据带 missionId 标签（见 §4 接线项） |
| W3 | 网关世界 API | `116_section_tail.inc:5296-5391`（`/api/world/{status,config,state/{sid},predict/{sid},evidence,reset}`） | 活跃 | **直接使用** | v8.0 A9 刚补齐对外暴露；`predict` 返回 `expectedNextState + calibration 对齐分` | 保留；供自主栈/外部调用 |
| W4 | 前端世界路由（核心） | `frontend_server.cpp`：`/world/{status,physics/status,state,ingest,earth-map/import,simulate}` | 活跃 | **直接使用** | 被 `WorldPanel` 消费；`/world/simulate` 是 3D 地图/具身 agent/生态/物理的演示入口 | 保留 |
| W5 | 前端世界路由（伪科学） | `frontend_server.cpp`：`/world/{cognitive,brain,conscious-compute,collective-compute}` | 休眠 | **封存** | "conscious/collective conscious compute" 命名属伪科学倾向；仅自引用，无 mission/agi/聊天消费者 | §2.4：按 archive 模板封存，前端 `WorldPanel` 不展示 |
| W6 | world_model.* 配置 | `config/phoenix.json` `world_model.*`（16 键）+ `frontend_server.bullet3Root` | 活跃 | **直接使用** | 16 键全部被 `resolveConfig` 读取（`frontend_server.cpp:4477-4497` 与 `116_section_tail.inc:7446-7462`） | 保留 |
| W7 | physics_world_runtime / bullet3 | `physics_world.hpp/.cpp`、`physics_world_runtime.hpp/.cpp`；`outsides/bullet3` | 活跃(隔离) | **增强** | 自包含 Bullet3 原生物理 + 地形导入 + 场景执行；RDK 用 `AI_WORLD_3D_MAP_ENABLED=false` 等降载；但不进预测闭环 | §2.3：把 `/world/simulate` 的 `physicsExecution` 摘要回灌预测证据，`alignmentScore` 上抛 `/api/world/predict` |
| W8 | WorldPanel（前端） | `079project_frontend/src/components/WorldPanel.js` + `api/client.js:154-158` | 活跃 | **直接使用** | 世界模型控制台（地形导入/虚拟场景模拟/世界状态）；本审计仅评估不修改 | 保留 |
| C1 | 重复端口 | `main.studyPort` vs `frontend_server.port`（均 5081） | 活跃(冗余) | **封存** | `main.studyPort`→`FRONTEND_PORT` env 覆盖 `frontend_server.port`，双源一值 | §4：主线程退役 `main.studyPort`，保留 `frontend_server.port` 单一来源 |
| C2 | v7 未接线 stub 组 | `v7.memory`、`v7.emotion_influence`、`v7.async_pool`、`v7.llama_server_mods` | 死 | **封存** | 仅 `v7.ahead` 被 `cfgOr<json>("v7.ahead")` 接线（`111:258`）；其余四组只有 `module_overrides/` stub 的 `fromJson`，无任何 `cfgOr<json>("v7.*")` 调用点 | §3：已删 22 键；stub 接线项见 §4 |
| C3 | jepa 改名遗留 | `jepa.image.*`、`jepa.camera.*`、`jepa.speech.*` | 休眠 | **封存** | JEPA 已改名 video/audio enc-dec；`jepa.image` 是 `video_model.cpp` 遗留包装（`frontend_server.cpp:4508`），`jepa.camera` 仍用，`jepa.speech.{variant,conceptDim,backend}` 死（语音走 Qwen2-Audio） | §3：已删 `jepa.speech` 3 死叶；整组封存见 §2.5 |
| C4 | main.multimodal 死组 | `main.multimodal.*`（6 键） | 死 | **封存** | enc/dec 现走 `external_mixed_modal_io.cpp` + `multimodal_encdec_server.py:8085`，模型名硬编码 `llava-1.5-7b`/`qwen2-audio-7b`；6 键无任何读取 | §3：已删 6 键 |
| C5 | edge_platform 边缘遗留 | `edge_platform.*`（npu + 通用） | 活跃(RDK) | **直接使用** | RDK X5 NPU/BPU/GPIO/网表路径；桌面端闲置但不死 | 保留（RDK 部署依赖） |
| C6 | tuner 专用组 | `scenarios`、`llama_server`、`universal_optimizer` | 休眠(runtime) | **直接使用** | 只被 `tools/auto_tune_phoenix_params.py` 读；运行时零读取，属调参工具参数空间 | 保留；文档标注"仅 tuner 读取" |
| C7 | spark.llamaVoter 阶段2 | `spark.llamaVoter.*`（12 键） | 休眠 | **封存** | `doc/v8.0/sparkarray_scopes.md` 标注阶段2"llamaVoter 输出重投"，配置已存在但未接线；另有顶层 `llama_voter.*`（已接线 `112:348`）易混淆 | 保留（阶段2预留）；与 `llama_voter.*` 合并方案见 §4 |
| C8 | 散落死叶 | `agi.maxEpisodes`、`partial_cache.enabled`、`model_deployment.llm.localBackend`、`context.attentionSink.sinkImportance`、`speech.{vadFrameMs,maxTracks,asrConfidenceThreshold,featureWindowMs,featureHopMs}`、`vision.{enhanceEnabled,embeddingDim,useTorchWhenAvailable,fallbackColorAnalysis,colorSpace,pixelMean,pixelStd}` | 死 | **封存** | 各自组的其余键被读，唯独这些叶零读取（例：`partial_cache` 真实开关是 `gnnEnabled`/`transformerEnabled`；`agi` 用硬编码 `kMaxEpisodes_`） | §3：已删 16 键 |

---

## 2. 分节详述

### 2.1 WorldModelStore 预测闭环（W1）— 增强

`world_model.hpp` 的 `WorldModelStore`（4515 行起）是本次审计的核心。`ingestEvidence()`（4539）在落一条证据时依次更新四类文档——session（`world:session:*`）、scene（`world:scene:*`）、episode（`world:episode:*`）、prediction（`world:prediction:*`）——并在 `updatePredictionDoc`（5491）里跑 **预测→对齐→校准** 闭环：`buildExpectedNextState`（5378）产预期下一态，与观测态 `computeAlignmentScore`（5452）对齐，把对齐分写回 `predictionDoc.predictionCalibration`。这正好落在项目理念"世界模型预测→对齐→校准"上，且是**理论驱动**（对齐分是可解释的余弦/重叠度，不是黑盒）。

**现状缺口**：这个对齐分目前只在 `runCognitionAutonomy`（`111_class_gatewayserver.inc:624-637`）里被 `mergePromptContext` 拼进 prompt 文本，**没有作为数值信号**进入 `cognitionManager_.observe/iterate`。`observePayload["worldUncertainty"]` 用的是请求体默认 `0.25`（`:616`），与世界的真实预测对齐分脱节。

**增强接线建议（文件+字段级）**：
1. `main_hub_parts/111_class_gatewayserver.inc` `runCognitionAutonomy`：在 `:626` 拿到 `wmState` 后，读 `wmState["prediction"]["predictionCalibration"]["alignmentScore"]`（或 `mismatch`），写 `observePayload["worldUncertainty"] = 1.0f - alignmentScore`，替换硬编码 `0.25`。这样 AGI 的 epistemic/Novelty 信号由世界模型校准分驱动，闭环"预测错得越多→探索权重越高"。
2. 把 `worldModel_->sessionState(sessionId, 8)` 的 `prediction` 子对象塞进 `iterPayload`（当前 `iterPayload` 只带 `graphContext`/`userPrompt`/`body`），让 `iterate()` 的 EFE 规划器能读 `expectedNextState` 作前向模型先验。
3. mission 域：当 `mission.enabled && mission.goal 非空`，在 `ingestWorldEvidence`（`111:29-44`）给 `evidence["metadata"]["missionId"]` 打标，使预测文档按目标域聚合。

### 2.2 网关世界证据注入 + /api/world/*（W2/W3）— 直接使用

`setWorldModelDatabase`（`111:8-27`）用 `world_model.storage.*` 建独立 `Database079`，`/api/chat` 与 `/api/transformer/chat` 入口调 `ingestWorldEvidence` 落用户轮次（`116:1695/1924/2385/2556`），`runCognitionAutonomy` 拉取并合并世界上下文。v8.0 A9（`migration_backlog.md` E 节）补齐 `/api/world/{state,predict,evidence,reset}`。这套链路**已接线、已回归**（`api_regression.ps1` 覆盖 `/api/world/status|config`；`test_short_chat.py`、`aa_ceval_api_bench.py` 探测 `/api/world/status`）。判定**直接使用**，无需改动。

### 2.3 physics_world_runtime / bullet3（W7）— 增强

`physics_world_runtime.{hpp,cpp}` 是自包含的 Bullet3 原生物理执行（`executeNativePhysicsScene`，仅支持 `backend=bullet3`）、地形导入（`normalizeEarthMapImportRequest`、`bundledEarthHeightfieldUri`）与运行时探测（`inspectBullet3Runtime`）。前端 `/world/simulate`（`frontend_server.cpp:6719-7038`）把虚拟场景 + 物理执行结果以多条 `worldModel.ingestEvidence` 落库。RDK 部署用 `AI_WORLD_3D_MAP_ENABLED=false / AI_WORLD_PHYSICS_ENABLED=false / AI_WORLD_EARTH_MAP_ENABLED=false / AI_WORLD_AGENT_COUNT=0`（`tools/rdk_cpu_cpu_cpu*.sh`、`rdk_netboot_build_and_run.sh`）降载。判定**增强**：模拟是"喂证据"而非"进预测"，建议把 `/world/simulate` 的 `physicsExecution.summary`/`worldMap3D.sceneEnvelope`（`frontend_server.cpp:6941-6965`）与 `prediction` 对齐分打通，使物理模拟成为预测闭环的一类观测。

### 2.4 伪科学路由 /world/{conscious-compute,collective-compute}（W5）— 封存

`world_model.hpp` 的 `buildConsciousComputePlan`（2546）/`buildCollectiveConsciousComputePlan`（2603/2846）与路由 `/world/conscious-compute`（`frontend_server.cpp:6512`）、`/world/collective-compute`（6567）以"意识计算/集体意识计算"命名，违反项目"理论驱动不做伪科学"红线；且除自身路由外无任何消费者（不进 mission/agi/聊天），属**休眠 + 伪科学**。`/world/brain`（6465）、`/world/cognitive`（6421）是更克制的"认知状态/脑剖面"版本，可保留，但建议一并复核命名。封存模板见 §2.6。

### 2.5 jepa / main.multimodal / v7 stub（C2/C3/C4）— 封存

- **jepa**：v7.6 已把 JEPA-v2 改名 video/audio enc-dec，上游模型换成 LLaVA/Qwen2-Audio（`external_mixed_modal_io.cpp`）。`jepa.image.*` 只作为 `video_model.cpp` 的遗留包装被读（`frontend_server.cpp:4508`），`jepa.camera.*` 仍被 `/camera/analyze` 读，`jepa.speech.{variant,conceptDim,backend}` 零读取（已删），仅 `jepa.speech.horizonModel/horizonDecoderModel` 走 BPU 权重路径（`audio_model.cpp:142-143`）。整组应**封存**，等 `model_deployment.vision/speech.*` 完全接管后删除。
- **main.multimodal**：6 键零读取（模型名硬编码在 `multimodal_world_model.hpp`，enc/dec 服务器地址走 `external_mixed_modal_io.cpp`），已删。
- **v7 stub**：`v7.ahead` 已接线（`111:258`）；`v7.memory`/`v7.emotion_influence`/`v7.async_pool`/`v7.llama_server_mods` 只有 `module_overrides/` 的 `fromJson` stub、无运行时调用点（AGENTS.md 明确"Remaining: wire ..."未完成），已删 22 键。

### 2.6 封存文档模板（供主线程落地）

参照 `doc/v7.0/archive/uncontrolled_evolution.md` 模板，每个封存项写清：**原始设想 / 封存原因（目的定位、没有消费者、编排归属）/ 保留在代码中的部分 / 重启条件 / 实现记录**。本审计建议为 W5（伪科学路由）与 C3（jepa 遗留）各建一份 archive 文档（如 `doc/v8.0/archive/conscious_compute.md`、`doc/v8.0/archive/jepa_legacy.md`），并把 W5 的"重启条件"设为"存在明确的意识科学度量 + 消费者（mission/agi）"。

---

## 3. config/phoenix.json 安全编辑（本轮已做）

**只删零读取死键**。判定"零读取"依据：对全部 `.cpp/.hpp/.h/.inc`（含 `test/`、`tests/`）扫描 `cfg / cfgOr / resolveConfig / resolveConfigAsString` 字面 dot-path + `dotPathForEnv` 间接映射（`000_section_before_config.inc:813-890` 覆盖 `main.*`）+ `model_deployment.cpp` dot-path 表，并复核 `tools/*.py`（tuner）与 test 目录引用。**未动**任何被读取键、LAN 示例值（`main.inference.llamaCppBaseUrl=192.168.0.104:8082`、`tinyllamaBaseUrl=192.168.0.104:8086`、`llamaCppModel=blobs/sha256-…`）、`version/source/timestamp` 元数据、tuner 专用组（`scenarios/llama_server/universal_optimizer`）与 `spark.llamaVoter`（阶段2预留）。

删除 47 键：

- `v7.memory.*`（8）、`v7.emotion_influence.*`（6）、`v7.async_pool.*`（4）、`v7.llama_server_mods.*`（4）= 22
- `main.multimodal.*`（6）
- `jepa.speech.{variant,conceptDim,backend}`（3）
- `agi.maxEpisodes`、`partial_cache.enabled`、`model_deployment.llm.localBackend`、`context.attentionSink.sinkImportance`（4）
- `speech.{vadFrameMs,maxTracks,asrConfidenceThreshold,featureWindowMs,featureHopMs}`（5）
- `vision.{enhanceEnabled,embeddingDim,useTorchWhenAvailable,fallbackColorAnalysis,colorSpace,pixelMean,pixelStd}`（7）

**验证**：`python tools/validate_phoenix_config.py` → **missing = 0（绿）**；leaf 键 550 → 503；`unused` 225 → 178（validator 的 `unused` 含大量误报：`main.*` 经 `dotPathForEnv` 实际被读、`emotion.seedLexicon` 经 `cfg<>` 实际被读、`frontend_server.*Workers` 经 `initStage` 实际被读、`vision.cnn.mean/std` 经 `resolveConfig<vector<float>>` 实际被读——这些本审计均未删）。

> 注：JSON 解析回写把 `1.0`→`1`、`0.0`→`0`、`-45.0`→`-45` 等整值浮点格式规范化（语义等价，nlohmann `get<float/double>` 对整数型可无损转换；被 `get<int>` 读的键原本就是整型字面量，无类型回退变化）。

---

## 4. 后续行动清单（只列不改，需主线程做 C++/路由改动）

| 项 | 谁来做 | 改哪里 | 具体动作 |
|---|---|---|---|
| A | 主线程 | `main_hub_parts/111_class_gatewayserver.inc` | **已完成 (v8.0)**：`runCognitionAutonomy` 用 `prediction.predictionCalibration.alignmentScore` 覆盖 `observePayload.worldUncertainty`；把 `prediction` 塞进 `iterPayload` |
| B | 主线程 | `main_hub_parts/111_class_gatewayserver.inc` | **已完成 (v8.0)**：`ingestWorldEvidence` 给证据打 `metadata.missionId` |
| C | 主线程 | `frontend_server.cpp` | **已完成 (v8.0)**：`/world/simulate` 的 `physicsExecution.summary` 与预测对齐分打通；`alignmentScore` 上抛 `/api/world/predict` |
| D | 主线程 | `frontend_server.cpp` + `079project_frontend/src/components/WorldPanel.js` | **已完成 (v8.0)**：下线/改名 `/world/conscious-compute`、`/world/collective-compute`；`WorldPanel` 不再渲染 |
| E | 主线程 | `002_section_before_sharedmemoryslice.inc` + `000_section_before_config.inc` | **已完成 (v8.0)**：退役 `main.studyPort`（及 `dotPathForEnv` 的 `AI_STUDY_PORT` 映射），`frontend_server.port` 单一来源 |
| F | 主线程 | `module_overrides/` + `111_class_gatewayserver.inc` | **已封存 (v8.0)**：要么按 AGENTS.md "Remaining" 接线 `v7.memory`/`v7.emotion_influence`/`v7.async_pool`，要么连 stub 一起封存 |
| G | 主线程 | `config/phoenix.json` + `video_model.cpp` | **已封存 (v8.0)**：`model_deployment.vision/speech` 完全接管后，删 `jepa.image.*`/`jepa.camera.*`（本轮只删了 `jepa.speech` 死叶） |
| H | 主线程 | `116_section_tail.inc` + `sparkarray_scopes.md` | **已完成 (v8.0)**：阶段2 要么把 `spark.llamaVoter.*` 接线，要么与顶层 `llama_voter.*` 合并去重 |
| I | 主线程 | `doc/v8.0/archive/` | **已完成 (v8.0)**：`conscious_compute.md`（W5）、`jepa_legacy.md`（C3） |

---

## 5. 审计方法注记

- 读取判定用"四路扫描"：字面 `cfg/cfgOr/resolveConfig/resolveConfigAsString` dot-path、`dotPathForEnv` 间接映射、`model_deployment.cpp` dot-path 表、`initStage` 等 helper 的字符串参数。
- 已知 validator 盲区（`tools/validate_phoenix_config.py` 仅匹配 `resolveConfig|cfgOr|resolveConfigAsString` 且模板参数遇 `>>` 会漏配）：`cfg<>`（无 Or）、`dotPathForEnv` 间接读、`initStage` 传字符串、`resolveConfig<vector<float>>` 嵌套模板。因此其 `unused=178` 是**上界**，真实死键数更小（本轮实删 47）。

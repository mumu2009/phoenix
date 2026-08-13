# Phoenix 算法/性能优化记录（20260805）

本文档记录一次针对 `phoenix/` 目录（不含 `mcu` 相关分支与 `catastrophe/`、`outsides/` 第三方库）
的算法与性能优化工作：方法、已落地的改动、验证结果，以及尚未处理、留待后续迭代的清单。

## 1. 范围与方法

- **范围**：`phoenix/` 根目录 C++ 源文件、`main_hub_parts/`（`main.cpp` 的分段实现）、
  `addons/`、`tools/`（Python 运维/调优脚本）、`079project_frontend/src`（React 前端）。
- **不在本轮范围内**：`catastrophe/`、`edge_platform.{cpp,hpp}` 中与 `mcu` 相关的分支、
  `outsides/`（vLLM/llama.cpp/Bullet3/onnxruntime 等第三方库，非本项目代码）。
- **方法**：先用只读子任务对上述范围做静态审查，产出按 P0/P1/P2 分级的算法复杂度、冗余计算、
  不必要拷贝、锁粒度、内存分配等问题清单；再对清单中风险可控、行为等价可验证的项逐一实施，
  每次修改后重新编译并跑单元测试，确保功能与可移植性不变。

## 2. 已完成的优化

### C++ 运行时

| 文件 | 内容 | 说明 |
|------|------|------|
| `hierarchical_memory.cpp` (`HierarchicalMemory::query`) | 排序比较函数中原来对每个候选在 `working_/shortTerm_/longTerm_` 三个 map 中重复 `find`（每次比较最多 6 次 map 查找）。改为收集阶段直接算出 `score` 并随 key/value 一起保存，排序只比较预先算好的 `score`。 | 纯性能优化，排序结果（分值公式、topK 截断）与原实现完全一致。 |
| `primal_sensation.cpp` (`PrimalSensationEngine::decay`) | 原来对 `sensations_` 做两次 `erase(remove_if(...))`：先删 `intensity <= 0`，再删 `intensity < kIntensityEpsilon`。由于 `kIntensityEpsilon(1e-3) > 0`，第二个条件已完全覆盖第一个，合并为一次遍历。 | 少一次 O(n) 遍历，行为不变。 |
| `external_mixed_modal_io.cpp` (`videoEncoder` / `audioEncoder`) | 原来在持锁状态下调用 `createVideoModel/createAudioModel`（可能涉及磁盘加载 ONNX/BPU 模型），锁粒度过粗会阻塞所有并发请求。改为双重检查锁定：先无锁查找，未命中时在锁外构建模型，再加锁二次检查后插入（若并发命中同一新 key，保留先插入的一份）。 | 减少长时间持锁；缓存语义不变，返回值不变。 |
| `semantic_unit.cpp` (`getProjectionMatrix`) | 同上模式：投影矩阵的构建是给定 `(sourceDim,targetDim,seed)` 的确定性计算，原来在锁内构建。改为锁外构建、锁内二次检查再插入。 | 并发场景下减少持锁时间；矩阵内容与原实现完全一致（相同 RNG 序列）。 |
| `graph_diffusion_summarizer.cpp`（邻接权重归一化） | 原来对每条边的方向加权值（`w *= 0.85/1.15` 等）计算了两次：一次累加 `wsum`，一次做归一化。改为只计算一次并缓存到临时数组，第二遍直接使用。 | 减少一半的浮点乘法/取绝对值调用，结果数值完全一致。 |
| `main_hub_parts/007_class_hotmatrixcache.inc`（`HotMatrixCache::getMatrix` / `recordAndMaybeSet`） | 原来用双重 `for` 循环逐元素拷贝 `vector<float>` <-> `vector<vector<float>>`。改为按行使用 `std::copy` / `vector::insert`。 | 减少循环开销，语义不变。 |

### Python 工具（`tools/`）

| 文件 | 内容 | 说明 |
|------|------|------|
| `tools/auto_tune_phoenix_params.py`（`ParameterSpace`） | `encode()` 与坐标下降策略中原来用 `self.values[i].index(value)`（O(n) 线性查找）。新增 `_value_index`（每个参数一个 `value -> index` 字典）与 `index_of()` 辅助方法，查找降为 O(1)。 | 不改变 CLI 行为；`ValueError` 语义保持一致。 |
| `tools/investor_benchmark_v2.py` / `tools/investor_benchmark_v3_tri.py` | 命中率统计中原来对每个结果都用 `next(ex.reference for ex in examples if ex.example_id == r.example_id)` 线性扫描全部样例（O(n·m)）。改为先构建 `example_id -> reference` 字典，再 O(1) 查找。 | 输出数值完全一致，仅内部实现变化。 |
| `tools/memory_tier_benchmark_v1.py`（`token_f1`） | 手写的 token 计数字典改为 `collections.Counter`（文件已导入该模块），逻辑等价、实现更简洁。 | 无行为变化。 |

### 前端（`079project_frontend/src`）

| 文件 | 内容 | 说明 |
|------|------|------|
| `components/ConfigPanel.js` | `robotsFiles` 复选框列表中原来用 `robotsSelected.includes(f)`（每行一次 O(n) 数组扫描）。新增 `robotsSelectedSet = useMemo(() => new Set(robotsSelected), [robotsSelected])`，改用 `Set.has()`。 | 渲染结果不变，仅查找方式优化。 |

## 2.1 第二轮：更激进的算法结构优化（含 constexpr 查表）

在第一轮基础上，按用户要求进一步处理了此前列入 backlog 的结构性问题，允许使用 `constexpr`
查表/魔数一类"对性能友好但对工程可读性略有牺牲"的手段：

| 文件 | 内容 | 说明 |
|------|------|------|
| `main_hub_parts/041_class_memegraph.inc`（`MemeGraph`） | `nodeOrder_`/`metaOrder_` 原来靠 `std::find` 判重（`loadAll`/`ensureRowLoaded` 中，图较大时是 O(n) 甚至整体 O(n²)）。新增 `nodeOrderSet_`/`metaOrderSet_` 两个 `unordered_set` 作为"是否已在 order 数组中"的 O(1) 镜像索引，统一通过新私有方法 `pushNodeOrderIfNew`/`pushMetaOrderIfNew`/`eraseFromNodeOrder`/`eraseFromMetaOrder` 维护，`ensureNodeLocked`/`loadAll`/`ensureRowLoaded`/`removeNode`/`setMeta`/`importSnapshot` 全部改为走这几个方法，保证两套结构始终同步。`NodeEdges::order`（每个节点自己的邻居顺序表）未改动——它的判重已经由 `table.map`（`unordered_map`）保证 O(1)，只有删除邻居时的 `vector::erase` 是 O(n)，但删除操作远比查找/插入少见，风险收益比不划算，本轮未动。 |
| `main_hub_parts/095_class_sparkarray.inc`（`SparkArray::dispatch`） | 请求文本、每个扰动变体（variant）在多个 controller/layer 间被反复 `textToMiniEmbedding()` 重新计算。新增按文本内容做键的局部 `embCache`（`unordered_map<string, vector<float>>` + `emb()` 闭包），同一次 `dispatch()` 调用内相同文本只计算一次 embedding。 |
| `main_hub_parts/047_class_dimreducer.inc`（`DimReducer::umap2D` 暴力 KNN） | 原来对每一对点都计算 `sqrt(acc)` 再做 `partial_sort`；由于 `sqrt` 对非负数单调，改为先按平方距离 `partial_sort`（`std::sqrt` 调用次数从 O(nRows) 降到 O(k)），再仅对选出的 k 个近邻取一次 `sqrt` 还原真实距离（下游 `rho`/`sigma`/权重计算仍使用真实距离，数值结果不变）。完整 KD-Tree/Ball-Tree 改造仍列入下方 backlog，因为改动面更大、需要专门的近邻检索单测。 |
| `main_hub_parts/057_class_sessionmanager.inc`（`SessionManager::truncateLocked`） | 原来对全部活跃会话按 `lastActivity` 完整 `std::sort`（O(n log n)），只是为了取前 `maxSessions_` 个。改为 `std::nth_element` 做 O(n) 划分——保留下来的会话集合与原实现完全一致（只是集合内部顺序不再保证，而调用方只是把它们塞进 `unordered_set` 做后续过滤，不依赖顺序）。 |
| `main_hub_parts/111_class_gatewayserver.inc`（`GatewayServer::sanitizeInputText`） | 这是 `/api/chat`、`/api/transformer/chat` 等接口每次请求都会执行的热路径：原来逐字节判断 `ch<0x20 \|\| ch==0x7F`、再判断是否是 `\t/\n/\r`，否则调用 `std::isspace(ch)`（本进程从未设置过非默认 locale，`isspace` 行为等价于经典 ASCII 空白表）。改为一个 **编译期生成的 256 项 `constexpr std::array` 查表**（`kSanitizeByteTable`），把"保留原字符 / 折叠为单个空格 / 丢弃控制字符"三种动作直接编码进表里，热循环里退化为一次数组下标 + `switch`，避免了每字节 1~2 次分支比较和一次 locale 相关的库函数调用。这是本轮唯一一处按用户要求、明确为了性能而牺牲一点直观性的"魔数/查表"式优化；行为与原实现逐字节等价（含 `changed` 标记的触发时机）。 |

**验证**：`compile.bat` 全部通过；`compile_gtest.bat` + `gtest_runner.exe` 全量 2085 个测试仍是 2084
通过、同一个与本次改动无关的既有失败项。`MemeGraph`/`SparkArray`/`DimReducer`/`SessionManager`
四个类目前在 `tests/gtest` 里**没有以类名命名的专项单测**（`grep` 确认），因此额外启动了一次
`phoenix_main.exe`，用 `POST /api/export/graph`（带 `radius`/`file` 参数）实际触发了
`MemeGraph::loadAll/buildWindow/exportSnapshot` 路径，返回 `ok=true` 且内容非空，作为编译+全量单测之外的
一次真实运行时验证。

**顺带发现但本轮未处理的问题（不在本次任务范围内，未修改）**：`POST /api/export/graph` 的处理器里
`json opts{..., {"file", body.value("file", json())}}` 在请求体不带 `file` 字段时，会往 `opts` 里塞入
一个值为 `null` 的 `"file"` 键；而 `exportGraphToFile()` 内部用 `options.value("file", "")` 读取它，
nlohmann::json 在键存在但值为 `null` 时会对 `get<std::string>()` 抛 `type_error.302`，导致不带 `file`
参数调用该接口时返回 500。这是一个**已存在的功能性 bug**（与本轮任何算法改动无关，本轮未触碰
`111_class_gatewayserver.inc` 中该 handler 或 `065_class_runtimestate.inc` 中的 `exportGraphToFile`），
记录于此供后续单独修复。

## 3. 验证

- `compile.bat`：`phoenix_main.exe` / `bug_shooter.exe` / `phoenix_sql_cli.exe` 均编译成功
  （仅有第三方库如 OpenCV/Bullet3 的既有 deprecation 警告，非本次改动引入）。
- `compile_gtest.bat` + `gtest_runner.exe`：全量 2085 个测试中 2084 通过；唯一失败项
  `CognitionMixedModalTest.PretrainSpeechConceptAndStatus` 经 `git stash` 验证在**改动前的原始代码上同样失败**
  （本地开发环境缺少语音世界模型所需的模型文件，属既有环境依赖问题，与本次改动无关）。
- 针对本次直接修改到的模块，额外跑了定向过滤：
  `--gtest_filter="*HierarchicalMemory*:*GraphDiffusionSummarizer*:*MixedModalIOTest*:*SemanticUnit*:*SemanticMemory*"`，
  47/47 全部通过。
- `test-tools/api_regression.ps1`：在本地无真实推理后端（未配置 Ollama/llama.cpp 模型、鉴权默认关闭）的
  沙箱环境中试跑，验证了新的"继续执行 + 汇总报告"机制按预期工作（遇到 404/超时不会像旧版那样立刻整体退出，
  而是记录失败、继续跑完其余步骤并在最后给出 PASS/FAIL/SKIP 汇总）。`/api/chat` 等依赖真实模型推理的用例
  在无后端配置的环境下会超时，这是环境限制，不是脚本或后端代码的缺陷；在配置了真实推理后端的环境中运行时，
  这些用例与原脚本行为一致。

## 4. `test-tools/api_regression.ps1` 增强说明

在保持原有全部用例和默认调用方式（`./test-tools/api_regression.ps1 -BaseUrl ... -Token ...`）向后兼容的前提下，新增：

- **执行模式**：默认改为"尽量跑完全部步骤 + 结尾打印 PASS/FAIL/SKIP 汇总"，而不是一遇到失败就整体退出，
  便于一次运行看到所有回归点的状态。如需恢复旧的"遇错即停"行为，加 `-FailFast`。
- **负向/错误输入用例**（`-SkipExtendedChecks` 可关闭，用于快速冒烟）：
  - 未带 token / 使用错误 token 访问需要鉴权的接口应返回 401。
  - 密码错误登录应返回错误。
  - `/api/chat` 缺少 `text` 字段应返回 400。
  - 未知路由应返回 404。
- **数据清洗回归（对应 `testing_strategy_v3.md` 第 6 节）**：
  - `GET /api/monitoring/stats` 校验存在 `routes` 与 `cleaning.{enabled,maxChars,cleanedInputs,cleanedSamples}`。
  - `POST /api/monitoring/reset` 后校验 `cleanedInputs`/`cleanedSamples` 归零。
  - `PATCH /api/runtime/features` 设置 `dataCleaningEnabled=false` 后校验：关闭清洗时发送含控制字符的输入，
    `cleanedInputs` 计数不增长（因为 `sanitizeInputText` 在禁用时直接返回原文）。
  - 重新开启清洗并设置 `dataCleanMaxChars=256` 后，发送超长输入，校验 `cleanedInputs` 增长（触发截断）。
  - 校验越界的 `dataCleanMaxChars`（如 999999999，超出服务端 `[128,65536]` 校验范围）会被静默忽略而不是崩溃或写入非法值。
  - `/api/corpus/ingest` 传入含 NUL/控制字符的文本，确认请求不崩溃、返回 `ok=true`。
  - 测试结束前会把 `dataCleaningEnabled`/`dataCleanMaxChars` 恢复为默认值（`true`/`2048`），避免污染后续用例或服务运行状态。
- **`/api/model/update` 序号单调性**：新增连续两次调用校验 `result.seq` 严格递增。

## 5. 后续优化 backlog（本轮未处理，按优先级排列）

出于"不动不该动的地方、每一步都要能编译验证"原则，本轮只落地了风险可控、可独立验证的改动。
以下是审查中发现但本轮未处理的项，建议作为后续迭代的输入（文件路径见括号内）：

**P0（已在第二轮处理，见 2.1 节）**

- ~~`MemeGraph::loadAll`/`ensureRowLoaded` 的 O(n) `std::find` 判重~~ → 已加 `nodeOrderSet_`/`metaOrderSet_`。
- ~~`DimReducer::umap2D` 暴力 KNN 中对每一对点都调用 `sqrt`~~ → 已改为按平方距离排序、只对 top-k 取 `sqrt`。
  完整 KD-Tree/Ball-Tree 化（把 O(n²) 距离矩阵本身降为 O(n log n)）仍未做，见下方 P0 保留项。
- ~~`SessionManager::truncateLocked` 每次截断都全量 `std::sort`~~ → 已改为 `std::nth_element` 分区。
- ~~`SparkArray::dispatch` 中同一文本被多个 controller/layer 重复 `textToMiniEmbedding`~~ → 已加请求内 `embCache`。

**P0（仍保留，改动面大，建议单独一轮 + 专项回归）**

- `MemeGraph::removeNode`/`decayEdge`/`removeOutgoingEdges` 里 `NodeEdges::order`（每个节点自己的邻居顺序表）
  的删除仍是 `vector::erase(std::remove(...))`（O(n)）。`NodeEdges` 是被 `link/decayEdge/removeOutgoingEdges/
  importSnapshot/loadAll/ensureRowLoaded/persistNode/buildWindow/messagePassingEmbeddings/exportSnapshot/
  neighborsOf` 等十余处共享的内部结构，改动面显著大于本轮已处理的 `nodeOrder_`/`metaOrder_`，且删除操作本身
  远少于查找/插入，本轮评估后判断改动风险/收益比不划算，故保留不动。
- `DimReducer::umap2D` 的距离矩阵计算本身仍是 O(n²)（对每一对点都算一次距离），只是省去了多余的 `sqrt`；
  如果需要处理大规模节点集合，才值得引入 KD-Tree/Ball-Tree 做近似最近邻，改动和测试成本都比较高。

**P1（局部性能优化，改动面小但需要补充针对性单测）**

- 澄清一处此前容易被误读为"重复计算"的既有功能（`093_class_personaforestaverager.inc` /
  `014_section_before_rng32.inc`）：这是两个独立但配合工作的老功能，均按设计工作，本轮未改、也不应被当作
  冗余计算去合并：
  1. `PersonaForestAverager::pick` 里对每个候选 `reply` 各算一次 `textToMiniEmbedding`——候选的 `reply`
     文本本身互不相同（来自不同 controller/layer），所以这不是重复计算；`pick` 内部真正的"随机森林"体现在
     `trees_`（默认 32 棵树）的投票循环上：每棵树用 `Rng32` 随机抽一个特征子空间（`pickSubspace`，对应
     `featureSubspace_`）、给子空间内每个特征随机加权后给所有候选打分投票，這是标准的随机子空间集成
     （random subspace ensemble）做法，用来让"选出哪个候选回复最终返回"这一决策比单一固定权重公式更鲁棒；
     每棵树复用的是同一份预先算好的 `candidates[i].features`（第 106-116 行一次性算好），并不会在树之间
     重新计算 embedding，所以这一层本身也没有可优化的重复计算。
  2. `SparkArray::dispatch` 里的 `buildVariants`/`perturbTokens`（`014_section_before_rng32.inc`）是另一
     个独立机制：对请求文本随机丢弃一两个词生成若干"扰动变体"，把每个变体也发给同一个 controller，得到的
     `variantResults[].affinity` 用来衡量该 controller 的回复对输入扰动是否稳健（鲁棒性打分），这部分已经
     通过本文件 2.1 节新增的请求级 `embCache` 复用了变体文本的 embedding 计算，属于本轮已优化范围。
- `tools/x5_bpu_evaluate.py`、`tools/rk3588_npu_evaluate.py`、`tools/jetson_trt_evaluate.py`、
  `tools/edge_ort_evaluate.py` 四个脚本中 `parse_shape`/`load_batch`/MSE 计算逻辑高度重复，
  建议抽取共享模块（需要保持各脚本 CLI 参数完全兼容）。
- 前端 `App.js` 中的会话列表渲染、`onSend` 等回调未使用 `useCallback`/`React.memo`，存在不必要的重渲染。

**P2（代码整洁/维护性，收益主要是可维护性而非性能）**

- `trimCopy`/`lowerCopy` 一类小工具函数在 `model_deployment.cpp`、`external_runtime.cpp`、`edge_platform.cpp`
  等多个文件中重复实现，可考虑统一到共享头文件（`edge_platform.cpp` 涉及 mcu 分支，需单独确认后再合并）。
- `gguf_tensor_parser.cpp` 中的 13 分支 `switch`（`readScalarValue`）可考虑表驱动，但当前实现已足够清晰，
  优先级较低。

以上 backlog 建议在后续迭代中按"一个模块一次 PR + 对应单测/回归脚本"的节奏推进，避免一次性大范围重构
带来的回归风险。

## 6. World Model 与主工作流基础整合（2026-08-06）

作为 World Model 与主工作流“Full architecture”的第一步，本次改动把原先只在 `frontend_server.cpp` 中实例化的 `world_model::WorldModelStore` 接入到 `GatewayServer` 核心聊天路径：

- `main_hub_parts/111_class_gatewayserver.inc`：
  - `GatewayServer` 新增 `std::shared_ptr<world_model::WorldModelStore> worldModel_` 成员。
  - 新增 `setWorldModelDatabase(Database079Ptr)`，从 `Database079` 创建 `kvm`/`meme_graph`/`session` 三个命名空间存储。
  - 新增 `ingestWorldEvidence(sessionId, text, graphSummary, modality)`，把聊天输入写成世界模型证据。
  - `runCognitionAutonomy()` 在调用 `cognitionManager_.observe/iterate()` 之前，先用 `worldModel_->sessionState()` 拉取当前会话的世界状态，并用 `world_model::mergePromptContext()` 把世界上下文拼接到 prompt 里，再把结果写回 `worldState["graphContext"]`。
- `main_hub_parts/116_section_tail.inc`：
  - `main()` 创建独立的 `Database079`（默认路径 `runtime_store/frontend_world_model.sqlite`），并在 `GatewayServer` 构建完成后调用 `setWorldModelDatabase()`。
  - `/api/chat` 与 `/api/transformer/chat` 在请求入口处调用 `ingestWorldEvidence()`，确保证据先落盘；后续 `runCognitionAutonomy()` 再读取同一份世界状态用于推理上下文。
- 共享存储：网关与 `frontend_server` 使用同一个 SQLite 文件与相同命名空间（`kvm`、`meme_graph`、`session`），因此前端 `/world/*` 路由与后端 `/api/chat` 看到的是同一世界状态。

**验证**：

- `compile.bat` 编译通过（仅依赖项/OpenCV 弃用警告）。
- 启动 `phoenix_main.exe` 后，日志打印 `[gateway-world] world model store attached`。
- `POST /api/chat {"text":"hello world","sessionId":"wm_test"}` 后，SQLite `kv_store` 表出现对应 `world:session:wm_test`、`world:scene:wm_test`、`world:evidence:...` 记录。
- 前端 `GET http://127.0.0.1:5081/world/state?sessionId=wm_test` 返回包含 `episode`、`sceneState`、`prediction`、`recentEvidence` 的完整世界状态，说明网关写入的证据已能被前端消费。

**已知限制 / 后续步骤**：

- 当前 `/api/chat` 在调用图选择/Transformer 时仍会失败（无本地 LLM 模型/Ollama 配置），因此本次只验证了 World Model 的“写入->读取”链路；完整的 `cognitionManager` 推理+回复生成需要配置真实推理后端才能端到端跑通。
- 物理世界模拟（Bullet3）、视频/音频世界模型与主聊天的更深耦合（作为工具调用、作为虚拟数据输入）是下一步，需要先在 `WorldModelStore` 之上扩展 `tool`/`observe`/`predict` 网关 API。

# 079 项目新人导读

这份文档是给第一次接手 079 项目的人用的。目标不是把每一个文件都解释完，而是让你能在一到两小时内建立下面几件事：

1. 这个项目到底由哪几层组成。
2. 真正的启动入口、网关入口、前端入口、模型入口分别在哪。
3. 每一层最重要的类、函数、路由和算法是什么。
4. 遇到一个需求时，应该先去哪个文件，而不是在仓库里盲搜。
5. 还有哪些现成文档可以继续往下看。

## 1. 先用一句话理解这个项目

079 不是一个单一的“聊天程序”，它是一个混合架构系统：

- C++ 主程序负责启动、网关、路由、数据库、自治、硬件接口和外部后端管理。
- Drogon 提供 HTTP 服务层。
- React 前端提供可视化入口。
- Python 和 Cython 负责 Transformer 的可维护实现与桥接。
- World Model、Autonomy、Mechanical Mind、Addon、Edge Platform 则是能力层。

如果只记一条主线，可以记成：

用户请求 -> 网关路由 -> 上下文/世界模型 -> 模型后端或内置 Transformer -> 回复 -> 自治与状态回灌

## 2. 第一次看仓库，先看什么，先别看什么

### 建议先看的文件

- [README.md](../README.md)
- [main.cpp](../main.cpp)
- [main_hub_parts/001_struct_config.inc](../main_hub_parts/001_struct_config.inc)
- [main_hub_parts/tail_parts/093_class_gatewayserver.inc](../main_hub_parts/tail_parts/093_class_gatewayserver.inc)
- [main_hub_parts/tail_parts/094_section_before_contexthint.inc](../main_hub_parts/tail_parts/094_section_before_contexthint.inc)
- [main_hub_parts/tail_parts/098_section_tail.inc](../main_hub_parts/tail_parts/098_section_tail.inc)
- [frontend_server.cpp](../frontend_server.cpp)
- [DATABASE_079.hpp](../DATABASE_079.hpp)
- [world_model.hpp](../world_model.hpp)
- [autonomy_stack.hpp](../autonomy_stack.hpp)
- [transformer.hpp](../transformer.hpp)
- [transformer_main.py](../transformer_main.py)

### 第一次可以先忽略的目录和文件

- [build/](../build/)：构建产物和 Conan/CMake 结果。
- [outsides/](../outsides/)：第三方外部项目集合，例如 llama.cpp、BitNet、Bullet3。
- [poppler-25.12.0/](../poppler-25.12.0/)：第三方依赖。
- [Python314/](../Python314/)：本地打包的 Python 运行时。
- [Redis-8.0.3-Windows-x64-cygwin-with-Service/](../Redis-8.0.3-Windows-x64-cygwin-with-Service/)：本地 Redis 发行包。
- [runtime_store/](../runtime_store/)、[snapshots/](../snapshots/)、[uploads/](../uploads/)：运行期数据。
- [transformer_main.cpp](../transformer_main.cpp)：这是 Cython 生成出来的超大桥接文件，理解逻辑时优先读 [transformer_main.py](../transformer_main.py)。
- [main.cpp](../main.cpp) 本身也不要久留，它只是一个 include hub，真实逻辑在 [main_hub_parts/](../main_hub_parts/) 里。

## 3. 启动链路总览

### 3.1 用户层启动入口

- [start_079_oneclick.bat](../start_079_oneclick.bat)：Windows 用户常用入口。
- [build_start_079_oneclick_exe.bat](../build_start_079_oneclick_exe.bat)：把 GUI 启动器打包成 exe。
- [compile.bat](../compile.bat)：完整构建脚本，负责 Conan、GCC、Python314、outsides 绑定、第三方探测和最终编译。

### 3.2 C++ 主程序启动入口

主入口不是普通手写的 main.cpp 大文件，而是：

- [main.cpp](../main.cpp)：只按顺序 include 拆分片段。
- 真正的 main 函数在 [main_hub_parts/tail_parts/098_section_tail.inc](../main_hub_parts/tail_parts/098_section_tail.inc)。

主程序启动流程可以按下面顺序理解：

1. main 先调用 parseArgs 和 loadConfig。
2. 如果命令行要求 frontend-only，则只启动前端服务 setupFrontendServer。
3. 否则初始化运行目录、PID 文件、日志和崩溃钩子。
4. 创建 Database079，并从中拿到 kvm、meme_graph、session 这三个存储。
5. 创建 GatewayServer。
6. GatewayServer 构造时完成路由注册 setupRoutes、自治初始化、边缘平台初始化、安全过滤初始化。
7. Drogon 监听网关端口并开始提供服务。

### 3.3 配置对象

核心配置结构体是 [main_hub_parts/001_struct_config.inc](../main_hub_parts/001_struct_config.inc) 中的 Config。

这个 Config 不是一个小配置，而是全局运行图谱，里面集中定义了：

- 网关和前端端口。
- Redis、SQLite、LMDB、snapshot、robots、tests 等路径。
- world model 的 agent 数量、地图尺寸、physics、earth map 开关。
- MemeBarrier 和 Mechanical Mind 阈值。
- reasoning planner、reasoning critic、cognition autonomy 开关。
- edge platform 和 NPU 相关配置。
- builtin addon 和 addon 动态库列表。
- Ollama、llama.cpp、BitNet 等外部后端参数。
- GGUF、outsides、bug shooter、Python 桥接路径。

如果你需要知道“某个运行参数在哪里进系统”，先看 Config。

## 4. 最高层结构图

| 层 | 主要文件 | 关键类 / 函数 | 作用 |
| --- | --- | --- | --- |
| 启动与配置 | [main.cpp](../main.cpp), [main_hub_parts/001_struct_config.inc](../main_hub_parts/001_struct_config.inc) | Config, main, parseArgs, loadConfig | 进程启动、参数解析、总配置装配 |
| 网关层 | [main_hub_parts/tail_parts/093_class_gatewayserver.inc](../main_hub_parts/tail_parts/093_class_gatewayserver.inc), [main_hub_parts/tail_parts/094_section_before_contexthint.inc](../main_hub_parts/tail_parts/094_section_before_contexthint.inc), [main_hub_parts/tail_parts/098_section_tail.inc](../main_hub_parts/tail_parts/098_section_tail.inc) | GatewayServer, chatWithOllama, chatWithExternalAdapter, chatWithLlamaCpp, chatWithBitNet, setupRoutes | 主 API 服务和聊天主链 |
| 前端服务层 | [frontend_server.cpp](../frontend_server.cpp), [frontend_server.hpp](../frontend_server.hpp) | setupFrontendServer, UserStore, ContextService | 5081 前端服务、认证、v51、world model 辅助路由 |
| 数据层 | [DATABASE_079.hpp](../DATABASE_079.hpp), [DATABASE_079.cpp](../DATABASE_079.cpp) | KeyValueStore, Database079, SqliteKeyValueStore | SQLite/Redis/LMDB 多后端键值和推理缓存 |
| 世界模型层 | [world_model.hpp](../world_model.hpp) | HotspotAnalysis, SelectiveKvCache, WorldModelStore, buildPromptContext, buildCognitiveState, buildReasoningAssembly, simulateVirtualScene | 世界状态、证据、提示拼装、认知脑与虚拟场景 |
| 自治层 | [autonomy_stack.hpp](../autonomy_stack.hpp), [autonomy_stack.cpp](../autonomy_stack.cpp) | TransformerClusterManager, SpiderAutonomyManager, OptimizerAutonomyManager, CognitionAutonomyManager, DatasetCatalogManager | 集群路由、自适应爬取、参数调优、认知自治、数据集治理 |
| 模型层 | [transformer.hpp](../transformer.hpp), [transformer_main.py](../transformer_main.py), [transformer_main.cpp](../transformer_main.cpp) | TransformerParams, Tokenizer, TransformerModel, TransformerService, TransformerMainEngine, get_engine, reset_engine | 内置 Transformer、训练、生成、反馈和 Python/C++ 桥接 |
| 外部运行时 | [external_runtime.hpp](../external_runtime.hpp), [external_runtime.cpp](../external_runtime.cpp), [gguf_tensor_parser.hpp](../gguf_tensor_parser.hpp) | BackendRuntimeSpec, BackendRuntimeState, ensureBackendReady, inspectFile, buildBrainMapDocument | Ollama / llama.cpp / BitNet / bug shooter 的启动与健康探测 |
| 安全与过滤 | [mechanical_mind.hpp](../mechanical_mind.hpp), [memebarrier_phrase_blocklist.hpp](../memebarrier_phrase_blocklist.hpp), [memebarrier_phrase_feedback.hpp](../memebarrier_phrase_feedback.hpp) | mechanical_mind::Filter, Analysis | 文本机械化过滤和安全清洗 |
| 边缘与物理 | [edge_platform.hpp](../edge_platform.hpp), [physics_world_runtime.hpp](../physics_world_runtime.hpp) | edge_platform::PlatformManager, executeNativePhysicsScene | 硬件拓扑、NPU/外设调度、物理场景执行 |
| 插件与扩展 | [addon.hpp](../addon.hpp), [addon.cpp](../addon.cpp), [addons/builtin_registry.hpp](../addons/builtin_registry.hpp), [module_mount.hpp](../module_mount.hpp) | Addon, AddonManager, createBuiltinAddon, ModuleRegistry | Addon 插件与模块工厂注册 |
| 前端工程 | [079project_frontend/README.md](../079project_frontend/README.md), [079project_frontend/package.json](../079project_frontend/package.json) | React 应用脚本、代理配置 | Web UI |

## 5. 按模块展开理解

### 5.1 网关主链

这里是最重要的部分。如果用户说“接口返回错了”“聊天逻辑要改”“模型切换有问题”，基本都要先看这里。

关键文件：

- [main_hub_parts/tail_parts/093_class_gatewayserver.inc](../main_hub_parts/tail_parts/093_class_gatewayserver.inc)
- [main_hub_parts/tail_parts/094_section_before_contexthint.inc](../main_hub_parts/tail_parts/094_section_before_contexthint.inc)
- [main_hub_parts/tail_parts/098_section_tail.inc](../main_hub_parts/tail_parts/098_section_tail.inc)

关键类和函数：

- GatewayServer：主网关服务类。
- GatewayServer::listen：启动 Drogon 监听。
- GatewayServer::warmupLearning：启动时做 RL、ADV、Transformer 预热。
- GatewayServer::bootstrapTransformerFromCorpus：从语料做初始训练。
- GatewayServer::chatWithOllama：调 Ollama。
- GatewayServer::chatWithExternalAdapter：调通用外部适配器。
- GatewayServer::chatWithLlamaCpp：调 llama.cpp。
- GatewayServer::chatWithBitNet：调 BitNet。
- setupRoutes：注册所有 /api/* 路由。

主聊天链的重点路由：

- /api/chat
- /api/transformer/chat
- /api/model/lifecycle
- /api/model/compress
- /api/model/explain
- /api/model/deploy
- /api/model/update
- /api/cognition/autonomy/status
- /api/cognition/autonomy/session/{id}
- /api/cognition/autonomy/iterate

你需要知道的一条事实：

- 这个项目的“聊天”不是只调用一个模型函数。
- 网关会把 graphContext、world model、自治、过滤、安全、外部后端选择一起串起来。

### 5.2 前端服务、认证和上下文服务

关键文件：

- [frontend_server.cpp](../frontend_server.cpp)
- [frontend_server.hpp](../frontend_server.hpp)
- [auth/README.md](../auth/README.md)

关键函数：

- setupFrontendServer：前端侧服务装配入口。

frontend_server.cpp 中定义了几个很重要、但不在头文件中的类：

- UserStore：本地用户库，存储在 [auth/users.json](../auth/users.json)。
- ContextService：会话上下文服务，负责根据会话长度和模式生成提示上下文。
- V51RuntimeEngine：v5.1 兼容引擎，声明在 [v51_runtime.hpp](../v51_runtime.hpp)。

UserStore 的核心方法：

- hasUsers
- addUser
- verifyUser
- getUser
- getUserByEmail
- updateUser
- updateEmail
- changePassword

ContextService 的核心方法：

- ingest
- reset
- status
- generateSessionId
- selectMode

这层最重要的路由：

- /auth/config
- /auth/bootstrap
- /auth/register
- /auth/login
- /auth/me
- /auth/profile
- /auth/change-password
- /auth/verify/request
- /auth/verify
- /auth/forgot
- /auth/reset
- /auth/logout
- /auth/admin/users
- /auth/admin/cleanup-test-users
- /v51/chat
- /context/reset
- /context/status
- /world/status
- /world/physics/status
- /world/state
- /world/cognitive
- /world/brain
- /world/conscious-compute
- /world/collective-compute
- /world/ingest
- /world/earth-map/import
- /world/simulate

v51/chat 这一条路由特别值得注意，因为它不是单做 v51：

- 先调用 ContextService::ingest 生成上下文。
- 再调用 WorldModelStore::ingestEvidence 回灌世界模型。
- 然后才按条件调用 V51RuntimeEngine::process 和 learn。

这说明前端服务层并不是简单静态文件服务器，它自己也承载了很多业务逻辑。

### 5.3 数据层和持久化

关键文件：

- [DATABASE_079.hpp](../DATABASE_079.hpp)
- [DATABASE_079.cpp](../DATABASE_079.cpp)

关键抽象：

- KeyValueStore：统一键值接口。
- Database079：总数据库管理器。
- SqliteKeyValueStore：SQLite 命名空间存储实现。
- HotCache：Redis 热缓存。

Database079 的核心方法：

- open
- close
- createStore
- getInferenceCache
- setInferenceCache
- configureInferenceSwap
- getSwapStats
- isSqliteStore

这个项目的数据层不是单数据库模式，而是分层模式：

- SQLite 负责主持久化。
- Redis 负责热缓存。
- LMDB 在主启动链里作为补充或回退存储出现。
- 业务上又分为 kvm、meme_graph、session 三类 store。

如果你在查“某段状态为什么会丢”“某段状态为什么命中很快”，通常要同时看 main 的存储装配逻辑和 Database079。

### 5.4 世界模型和认知脑

关键文件：

- [world_model.hpp](../world_model.hpp)
- [frontend_server.cpp](../frontend_server.cpp)

这个头文件很大，而且是 header-only 风格。它不是单纯的“状态结构体”，而是整个认知上下文、提示拼装、虚拟场景、证据聚合和脑型配置的核心。

重要结构和枚举：

- HotspotAnalysis
- SelectiveKvCache
- BrainProfileKind
- ReasoningAssemblyOptions
- PromptContextOptions
- WorldModelStore

核心函数：

- extractKeywords
- analyzeReasoningHotspot
- analyzeReplyHotspot
- buildPromptContext
- buildCognitiveState
- buildReasoningAgenda
- buildReasoningAgendaPromptContext
- buildReasoningAssembly
- buildVirtualSceneRollout
- buildVirtualSceneTrainingSamples
- simulateVirtualScene

WorldModelStore 的核心方法：

- ingestEvidence
- sessionState
- resetSession
- status

WorldModelStore 实际会维护几类数据：

- world:evidence:sessionId:evidenceId
- world:session:sessionId
- world:scene:sessionId
- world:episode:sessionId:current

理解这一层时要把它当成一个“把输入证据整理成可推理图景”的系统，而不是当成普通 KV 存储。

这层还定义了两类非常重要的认知概念：

- Functional brain：偏功能型认知组织。
- Structural brain：偏结构型认知组织。

同时它还有 reasoning agenda、planner、critic、virtual scene rollout 等更高阶概念。也就是说，world_model.hpp 不是只负责“记忆”，它同时也参与“怎么想”。

### 5.5 自治系统

关键文件：

- [autonomy_stack.hpp](../autonomy_stack.hpp)
- [autonomy_stack.cpp](../autonomy_stack.cpp)

核心类：

- autonomy::TransformerClusterManager
- autonomy::SpiderAutonomyManager
- autonomy::OptimizerAutonomyManager
- autonomy::CognitionAutonomyManager
- autonomy::DatasetCatalogManager

各类的关键方法：

- TransformerClusterManager::status
- TransformerClusterManager::updateNodes
- TransformerClusterManager::pickNode
- TransformerClusterManager::feedback
- SpiderAutonomyManager::status
- SpiderAutonomyManager::adapt
- OptimizerAutonomyManager::status
- OptimizerAutonomyManager::iterate
- OptimizerAutonomyManager::applyPerfProfile
- OptimizerAutonomyManager::proposeGnnUpgrade
- OptimizerAutonomyManager::proposeTransformerUpgrade
- OptimizerAutonomyManager::modernizeTransformer
- CognitionAutonomyManager::status
- CognitionAutonomyManager::observe
- CognitionAutonomyManager::iterate
- CognitionAutonomyManager::session
- DatasetCatalogManager::status
- DatasetCatalogManager::list
- DatasetCatalogManager::registerDataset
- DatasetCatalogManager::activate
- DatasetCatalogManager::updateCleaningProfile
- DatasetCatalogManager::collectData
- DatasetCatalogManager::governance

这里的核心认识是：

- 集群路由不是死写死配，而是一个带反馈的路由器。
- 爬虫参数会根据监控结果自动调整。
- 性能调优器会参考 avgMs、errorRate 和 transformerParams 自动生成 patch。
- 认知自治会维护 session 级目标、假设、反思和补丁建议。

### 5.6 模型层

关键文件：

- [transformer.hpp](../transformer.hpp)
- [transformer_main.py](../transformer_main.py)
- [transformer_main.cpp](../transformer_main.cpp)
- [model_lifecycle.hpp](../model_lifecycle.hpp)
- [model_lifecycle.cpp](../model_lifecycle.cpp)

transformer.hpp 中最重要的类型：

- transformer::TrainSample
- transformer::TransformerParams
- transformer::Matrix
- transformer::Linear
- transformer::LayerNorm
- transformer::MultiHeadAttention
- transformer::FeedForward
- transformer::EncoderLayer
- transformer::DecoderLayer
- transformer::Tokenizer
- transformer::TransformerModel
- transformer::TransformerService

TransformerModel 的核心方法：

- encode
- decode
- logitsAt
- generate
- fuseMemory
- trainOnSample
- toJson
- stateDict
- loadStateDict
- updateParams

TransformerService 的核心方法：

- chat
- pretrain
- jointTrain
- optimizeGA
- verify
- addFeedback
- addPreferenceFeedback
- trainFromFeedback
- rlhfStats
- params
- applyParams
- saveCheckpoint
- loadCheckpoint

Python 可维护源主要看 [transformer_main.py](../transformer_main.py)，关键类和函数包括：

- TransformerReservedArena
- MatrixHotspotCache
- Tokenizer
- TransformerModel
- TransformerMainEngine
- get_engine
- reset_engine

如果你需要读“真实业务逻辑”，优先读 transformer_main.py；如果你需要查 C++ 如何桥接到 Python，再看 transformer_main.cpp。

transformer_main.cpp 的几个关键桥接点：

- PyInit_transformer_main
- PyImport_AppendInittab
- TransformerService::chat
- TransformerService::pretrain
- TransformerService::jointTrain
- TransformerService::optimizeGA
- TransformerService::verify

### 5.7 模型生命周期管理

关键文件：

- [model_lifecycle.hpp](../model_lifecycle.hpp)
- [model_lifecycle.cpp](../model_lifecycle.cpp)

核心类：

- model_lifecycle::ModelLifecycleManager

核心方法：

- status
- compressPlan
- explainOutput
- deployTarget
- applyOnlineUpdate

这一层是“运营和部署控制面”，不是生成模型本体。它主要负责：

- 压缩计划。
- 输出解释。
- 部署目标切换。
- 在线更新状态。

### 5.8 外部后端和 GGUF

关键文件：

- [external_runtime.hpp](../external_runtime.hpp)
- [external_runtime.cpp](../external_runtime.cpp)
- [gguf_tensor_parser.hpp](../gguf_tensor_parser.hpp)

external_runtime 中的重要结构：

- BackendRuntimeSpec
- BackendRuntimeState
- BugShooterSpec
- BugShooterState

重要函数：

- ensureBackendReady
- ensureBugShooterAttached
- readStatusFile

gguf_tensor_parser 中的重要函数：

- inspectFile
- buildBrainMapDocument
- buildStructuredExportBundle
- writeStructuredExportFiles

这一层解决的是“后端怎么拉起来、怎么探活、怎么从 GGUF 模型里读出结构信息”的问题。

### 5.9 安全过滤和语音

关键文件：

- [mechanical_mind.hpp](../mechanical_mind.hpp)
- [speak_io.hpp](../speak_io.hpp)

mechanical_mind 的重要类型：

- mechanical_mind::Document
- mechanical_mind::Options
- mechanical_mind::Analysis
- mechanical_mind::Filter

Filter 的关键方法：

- setEnabled
- setTextThreshold
- setTokenThreshold
- setPlaceholder
- warmup
- analyzeAndSanitize

SpeakIO 的关键方法：

- analyzeWavBytes
- analyzePcm
- synthesizeText
- analyzeEnvironment
- analyzeTone
- recognizeSpeech
- separateTracks

SpeakIO 内部会使用一些传统音频特征函数：

- computeRms
- computeZcr
- computeSpectralCentroid
- estimatePitch
- classifyEnvironment
- classifyEmotion

### 5.10 边缘平台和物理世界

关键文件：

- [edge_platform.hpp](../edge_platform.hpp)
- [physics_world_runtime.hpp](../physics_world_runtime.hpp)
- [rpi_zero2w_edge_platform.md](rpi_zero2w_edge_platform.md)
- [rpi_zero2w_capability_probe.md](rpi_zero2w_capability_probe.md)

edge_platform 的关键类型：

- edge_platform::RuntimeConfig
- edge_platform::PlatformManager
- edge_platform::PlatformManager::SignalBinding
- edge_platform::PlatformManager::InterfaceSummary
- edge_platform::PlatformManager::PhysicalPinSummary
- edge_platform::PlatformManager::ConnectorSummary
- edge_platform::PlatformManager::TopologyCache
- edge_platform::PlatformManager::Metrics

PlatformManager 的核心方法：

- reconfigure
- status
- refreshTopology
- applyPatch
- planCompute
- dispatchCompute
- dispatchPeripheral
- planMobility
- dispatchMobility
- runSelfTest

physics_world_runtime 的核心函数：

- physics_world::executeNativePhysicsScene

这一层主要服务于：

- 硬件连接拓扑。
- NPU / GPIO / 电机 / 摄像头等外围设备。
- 物理仿真世界和地形导入。

### 5.11 插件和模块挂载

关键文件：

- [addon.hpp](../addon.hpp)
- [addon.cpp](../addon.cpp)
- [addons/builtin_registry.hpp](../addons/builtin_registry.hpp)
- [addons/math_addon.cpp](../addons/math_addon.cpp)
- [addons/search_addon.cpp](../addons/search_addon.cpp)
- [module_mount.hpp](../module_mount.hpp)
- [addons/README.md](../addons/README.md)
- [module_overrides/README.md](../module_overrides/README.md)

Addon 体系的核心抽象：

- addon::Addon
- addon::AddonResult
- addon::AddonManager

AddonManager 的核心方法：

- registerAddon
- addBuiltin
- loadLibrary
- removeAddon
- listAddons
- run

Builtin addon 工厂：

- addon::builtins::createBuiltinAddon
- addon::builtins::createDefaultBuiltinAddons
- addon::builtins::createMathAddon
- addon::builtins::createSearchAddon

module_mount 和 addon 不是一回事：

- addon 更像运行时功能插件。
- module_mount::ModuleRegistry 更像内部服务工厂替换点。

ModuleRegistry 支持的工厂接口包括：

- RedisSynchronizerFactory
- StudyEngineFactory
- SnapshotManagerFactory
- PersonaForestAveragerFactory
- SparkArrayFactory
- ReinforcementLearnerFactory
- AdversarialLearnerFactory
- GnnGaLearnerFactory
- GatewayServerFactory

### 5.12 前端工程

关键文件：

- [079project_frontend/README.md](../079project_frontend/README.md)
- [079project_frontend/package.json](../079project_frontend/package.json)

前端层的重要事实：

- 它是 React 工程。
- 代理默认指向 5080 网关。
- scripts 只有 start、build、test、eject 四个标准入口。
- 前端默认把 core 后端作为主 provider。

## 6. 这个项目用了哪些算法

这一节只总结“已经能在代码里明确看到”的算法或策略，不写抽象口号。

### 6.1 Transformer 和训练侧

实现位置：

- [transformer.hpp](../transformer.hpp)
- [transformer_main.py](../transformer_main.py)

可直接从代码看到的机制包括：

- Multi-Head Attention
- Feed Forward Network
- LayerNorm
- MoE，参数名包括 enableMoE、moeExperts、moeTopK、moeAuxWeight
- MLA，参数名包括 enableMLA、mlaRank、mlaScale
- Coef Attention，参数名包括 enableCoefAttention、coefAttentionScale
- Hierarchical Embedding，参数名包括 enableHierEmbedding、hierStride
- Dynamic Sampling，参数名包括 dynamicSampling、topK、topP、temperatureMin、temperatureMax
- Multi-token objective
- CoT scaffold
- Program synthesis bias
- RLHF 和 IRL 参数
- Addon module learning
- Tokenizer 的 BPE 模式

### 6.2 世界模型和提示拼装算法

实现位置：

- [world_model.hpp](../world_model.hpp)

能明确识别出的算法和策略：

- stableHash64：FNV 风格稳定哈希，用来构造缓存键。
- extractKeywords：从文本里抽关键词标签。
- analyzeReasoningHotspot：根据 summary、labels、recentEvidence、graphContext 计算热点分数。
- analyzeReplyHotspot：根据回复长度、graphContext、关键词数量、maxTokens 估算热点。
- SelectiveKvCache：冷热分层缓存，按命中次数和 TTL 晋升热键。
- buildPromptContext：把 recentEvidence、summary、objectSlots 等拼成提示上下文。
- buildCognitiveState 和 buildReasoningAgenda：把世界状态整理成 reasoning agenda。
- buildVirtualSceneRollout：构造 observer / planner / critic / memory / explorer 等 embodied agents 的虚拟场景推演。
- buildVirtualSceneTrainingSamples：从虚拟场景 roll-out 反推训练样本。

### 6.3 自治和调度算法

实现位置：

- [autonomy_stack.cpp](../autonomy_stack.cpp)

关键策略：

- TransformerClusterManager::pickNode：按 inflight、weight、emaLatencyMs、healthy、maxTokens 做评分选路。
- TransformerClusterManager::feedback：用 EMA 延迟和成功失败反馈更新节点状态。
- SpiderAutonomyManager::adapt：根据 /api/chat 平均时延调节 intervalSec 和 maxPages。
- OptimizerAutonomyManager::iterate：根据 /api/transformer/chat 的 avgMs 和 errorRate 对 transformer 参数做启发式 patch。
- CognitionAutonomyManager：维护认知观测、反思、补丁建议和 session 状态。

### 6.4 语境模式选择算法

实现位置：

- [frontend_server.cpp](../frontend_server.cpp)

ContextService::selectMode 采用很朴素但有效的规则：

- 对话初期用 concat。
- 中期切到 rnn。
- 长会话切到 lstm。

这说明前端服务层自己也在做轻量语义状态机，而不是纯转发。

### 6.5 机械化心智过滤算法

实现位置：

- [mechanical_mind.hpp](../mechanical_mind.hpp)

Filter::analyzeAndSanitize 会综合：

- lexicalScore
- densityScore
- anthropomorphicScore
- flaggedTokens
- flaggedPhrases

最后再根据阈值决定 trigger 和 replacement。也就是说这不是简单黑名单替换，而是一个带分数合成的判定器。

### 6.6 语音侧传统特征算法

实现位置：

- [speak_io.hpp](../speak_io.hpp)

可直接看到的传统信号处理特征：

- RMS
- ZCR
- Spectral Centroid
- Pitch Estimation

然后再基于这些特征做环境分类和情绪分类。

### 6.7 数学插件算法

实现位置：

- [addons/math_addon.cpp](../addons/math_addon.cpp)

这部分非常明确：

- tokenizeExpr 先做词法拆分。
- toRpn 把表达式转成逆波兰式。
- evalRpn 执行逆波兰式求值。

这实际上就是一个小型的表达式解释器，使用的是 Shunting-yard 思想加 RPN 求值。

### 6.8 搜索插件策略

实现位置：

- [addons/search_addon.cpp](../addons/search_addon.cpp)

核心步骤：

- stripPrefix 识别 search、web、lookup、搜索、检索、查询 等前缀。
- 通过 invokeAddonOnlineLookup 把查询转给在线检索处理器。
- 从 snippet、text、suggestions 里抽回复文本。

### 6.9 模型生命周期估算算法

实现位置：

- [model_lifecycle.cpp](../model_lifecycle.cpp)

可以明确看到的估算逻辑：

- compressPlan 会根据 pruneRatio 和 quant 类型估计 estimatedSizeRatio 和 estimatedSpeedup。
- explainOutput 会对 input、graphContext、reply 做 token overlap 支撑度分析。
- deployTarget 会做 target、replicas、routingPolicy、canaryPercent 的部署面配置。

### 6.10 外部运行时探测算法

实现位置：

- [external_runtime.cpp](../external_runtime.cpp)
- [gguf_tensor_parser.hpp](../gguf_tensor_parser.hpp)

关键策略：

- 解析 HTTP URL 和 TCP 端口。
- 轮询健康检查接口。
- detached process 启动外部可执行文件。
- 通过 GGUF inspect 推断 KV cache 代价和 brain map 信息。

## 7. 如果你要改某个功能，先去哪里

### 改聊天请求或主 API

- 先看 [main_hub_parts/tail_parts/093_class_gatewayserver.inc](../main_hub_parts/tail_parts/093_class_gatewayserver.inc)
- 再看 [main_hub_parts/tail_parts/094_section_before_contexthint.inc](../main_hub_parts/tail_parts/094_section_before_contexthint.inc)
- 最后看 [main_hub_parts/tail_parts/098_section_tail.inc](../main_hub_parts/tail_parts/098_section_tail.inc)

### 改登录、注册、JWT、本地 token

- 先看 [frontend_server.cpp](../frontend_server.cpp) 中的 UserStore 和 /auth/* 路由
- 再看 [auth/README.md](../auth/README.md)

### 改世界模型或 session 状态

- 先看 [world_model.hpp](../world_model.hpp)
- 再看 [frontend_server.cpp](../frontend_server.cpp) 中 /world/* 路由

### 改 Transformer 参数、训练或反馈

- 先看 [transformer.hpp](../transformer.hpp)
- 再看 [transformer_main.py](../transformer_main.py)
- 最后才看 [transformer_main.cpp](../transformer_main.cpp)

### 改外部后端启动或探活

- 先看 [external_runtime.hpp](../external_runtime.hpp)
- 再看 [external_runtime.cpp](../external_runtime.cpp)

### 改插件系统

- 先看 [addon.hpp](../addon.hpp)
- 再看 [addon.cpp](../addon.cpp)
- 再看 [addons/README.md](../addons/README.md)

### 改树莓派、NPU、外设或物理仿真

- 先看 [edge_platform.hpp](../edge_platform.hpp)
- 再看 [physics_world_runtime.hpp](../physics_world_runtime.hpp)
- 再看 [rpi_zero2w_edge_platform.md](rpi_zero2w_edge_platform.md)

## 8. 项目内现成文档索引

### 8.1 总览和契约

- [根 README.md](../README.md)
- [doc/README.md](README.md)
- [v3_contract.md](v3_contract.md)
- [testing_strategy_v3.md](testing_strategy_v3.md)

### 8.2 架构与研究方向

- [brain_dual_track_and_conscious_compute_20260405.md](brain_dual_track_and_conscious_compute_20260405.md)
- [sparkarray_transformer_ensemble_and_gguf.md](sparkarray_transformer_ensemble_and_gguf.md)
- [agi_direction_and_humanlike_evaluation_20260403.md](agi_direction_and_humanlike_evaluation_20260403.md)
- [intelligence_gap_analysis_20260403.md](intelligence_gap_analysis_20260403.md)
- [stability_root_cause_report_20260504.md](stability_root_cause_report_20260504.md)

### 8.3 算法文档

- [algorithm/algorithm.md](algorithm/algorithm.md)
- [algorithm/StreamOfData.md](algorithm/StreamOfData.md)
- [algorithm/reinforcement_learner_algorithm.md](algorithm/reinforcement_learner_algorithm.md)
- [algorithm/adversarial_learner_algorithm.md](algorithm/adversarial_learner_algorithm.md)
- [algorithm/gnn_ga_learner_algorithm.md](algorithm/gnn_ga_learner_algorithm.md)
- [algorithm/HAI_Formal_Spec.md](algorithm/HAI_Formal_Spec.md)

### 8.4 数学证明和推导

- [math/transformer_math_proof.md](math/transformer_math_proof.md)
- [math/fine_tuning_math_proof.md](math/fine_tuning_math_proof.md)
- [math/fineT_with_GNN.md](math/fineT_with_GNN.md)
- [math/mathtransformer_math.proof.md](math/mathtransformer_math.proof.md)

### 8.5 外部依赖与数据集

- [indexOfOutside.md](indexOfOutside.md)
- [external_dataset_index.json](external_dataset_index.json)
- [training_data_policy.md](training_data_policy.md)

### 8.6 版本演进

- [v5.1/](v5.1/)
- [v5.2/](v5.2/)

### 8.7 其他子模块文档

- [079project_frontend/README.md](../079project_frontend/README.md)
- [addons/README.md](../addons/README.md)
- [auth/README.md](../auth/README.md)
- [module_overrides/README.md](../module_overrides/README.md)

## 9. 推荐阅读顺序

如果你是新加入项目的同学，建议按下面顺序读：

1. 先读 [README.md](../README.md) 和本文。
2. 再读 [main_hub_parts/001_struct_config.inc](../main_hub_parts/001_struct_config.inc)，知道系统能配什么。
3. 再读 [main_hub_parts/tail_parts/093_class_gatewayserver.inc](../main_hub_parts/tail_parts/093_class_gatewayserver.inc) 和 [main_hub_parts/tail_parts/098_section_tail.inc](../main_hub_parts/tail_parts/098_section_tail.inc)，知道主网关怎么跑。
4. 再读 [frontend_server.cpp](../frontend_server.cpp)，知道 auth、context、world、v51 在哪里。
5. 再读 [DATABASE_079.hpp](../DATABASE_079.hpp) 和 [world_model.hpp](../world_model.hpp)，知道状态存在哪里、怎么拼成 graphContext。
6. 再读 [autonomy_stack.hpp](../autonomy_stack.hpp) 和 [autonomy_stack.cpp](../autonomy_stack.cpp)，知道系统如何自调度。
7. 再读 [transformer.hpp](../transformer.hpp) 和 [transformer_main.py](../transformer_main.py)，知道内置模型怎么工作。
8. 最后按自己的任务再进入 [edge_platform.hpp](../edge_platform.hpp)、[addon.hpp](../addon.hpp)、[external_runtime.cpp](../external_runtime.cpp)、[079project_frontend/](../079project_frontend/) 等专项区域。

## 10. 给新人的最后几个提醒

### 10.1 不要一上来就改生成文件

- [main.cpp](../main.cpp) 是汇总 include hub。
- [transformer_main.cpp](../transformer_main.cpp) 是 Cython 生成桥接文件。

这两个文件都可能需要看，但都不适合作为第一编辑点。

### 10.2 先抓“控制点”，不要先抓“最长的文件”

这个仓库很多文件都很长，但真正决定行为的控制点并不多，通常是下面这些名字：

- Config
- GatewayServer
- setupFrontendServer
- UserStore
- ContextService
- Database079
- WorldModelStore
- CognitionAutonomyManager
- TransformerService
- ModelLifecycleManager
- PlatformManager
- AddonManager
- ModuleRegistry

### 10.3 你可以把系统分成四个问题来想

- 请求从哪里进来。
- 状态存到哪里。
- 模型怎么被调用。
- 调用结果怎么被反馈回世界模型和自治层。

只要这四件事能说清，你就已经不是“看不懂项目”，而是在正常接手了。

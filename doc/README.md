# doc

本目录收纳仓库内的设计、协议、测试策略与版本文档。

- `product_ops.md`：登录、权限、监控、数据库与常见故障（给用户/部署者）。
- `DATABASE_079.md`：SQLite/LMDB 路径、备份与管理 API。

## 主要内容

- `v3_contract.md`：接口与契约说明。
- `testing_strategy_v3.md`：测试策略与验证范围。
- `brain_dual_track_and_conscious_compute_20260405.md`：双版本类脑架构、研究脑反推与人机协同计算协议。
- `training_data_policy.md`：训练数据与外部数据集策略。
- `external_dataset_index.json`：外部数据集索引元数据。
- `indexOfOutside.md`：外部依赖与外部模块索引。
- `algorithm/`：算法说明与数据流文档。
- `math/`：数学推导与证明材料。
- `v5.1/`、`v5.2/`：版本演进文档。
- `v8.3/self_mind.md`：自我层（对话强化学习、无提问的自动思考、镜子/图灵探针）。校准语料 `v8.3/self_mind_sample.json` 是合成的。
- `v8.0/`：v8.0 "Lancelot" 文档集：
  - `migration_backlog.md`：ollama/GNN 时代功能 → llama-server 迁移积压（A1–A8 状态表 + 优先级 + 不迁移清单）；
  - `sparkarray_scopes.md`：SparkArray 作用域化设计（可选/多选/禁用，chat 域已接入）；
  - `rdk_netboot.md`：RDK X5 网络引导部署（netboot，含 :5081 前端子进程与 UI 分发）；
  - `config_world_audit.md`：world/config 模块审计（增强/直接使用/封存，A-I 行动清单）；
  - `archive/conscious_compute.md`、`archive/jepa_legacy.md`、`archive/v7_stub_configs.md`：封存档案（伪科学路由、jepa 过渡层、v7 未接线配置组）。
- `v7.0/`：v7.0 "Arthur" 文档集：
  - `v7.0.md`：目标架构总纲；`workflow.md`：数据流；`algorithm.md`：算法（§1–§23，含全部数学证明）；
  - `active_inference.md`：EFE/MPC 与 TD(0) 自进化；`subconscious.md`：PAD 气质/稳态调谐；
  - `mission_layer.md`：Meeseeks 任务层（g·T²/2 定理、自由复制）；`plugins.md`：插件生态（cli-json/MCP/数学/搜索）；
  - `autonomy_loop.md`：长期自主心跳 + 持久化 + 插话；`safety.md`：系统级急停（E-stop）；
  - `model_deployment.md` / `complexity_bounds.md` / `llamacpp_optimization.md` / `testing_methodology.md`（§1–§12 验证协议） / `testing_plan.md`；
  - `archive/uncontrolled_evolution.md`：封存的不受控进化设想（原因与重启条件）。
- `v10.1/`：v10.1 "Gareth" 文档集：
  - `artifact_contract.md`：产物契约（模式/版本/兼容/拆分/锁）；
  - `plan_round1.md`：首轮计划；`implementation_2026-11-07.md`：实现记录（情绪矩阵、大规模测试、增长、并行数据竞争）；
  - `crash_forensics_2026-11-15.md`：`0xC0000374` 堆损坏取证（共享状态审计表、四处竞争点、修复与验证方法）；
  - `round_2026-11-15.md`：本轮总结（GPU 空转根因＝`callLlamaCppBackend` 递归占满 slot、异步训练、配置/状态分离、实测数据、残留缺陷）；
  - `logic_engine.md`：符号推理层（Prolog/Datalog）设计与复杂度；
  - `reasoning_research_plan.md`：符号推理的**决策门**（Q1 需求频率 / Q2 选择率 / Q3 A-B 效果，含通过阈值与最小集成方式）。
- `architecture_v10.2.html`：**交互式架构图（Canvas）**。10 层 / 58 模块 / 327 文件条目，
  覆盖从进程部署到基础设施的全部子系统；支持拖拽平移、滚轮缩放、点击查看模块的文件与职责、
  按名称/文件/职责搜索、按层开关、全量文件索引。用浏览器直接打开即可（无外部依赖）。

## 适用场景

1. 理解架构与历史演进。
2. 确认测试与接口契约。
3. 查询外部数据集和外部模块的索引信息。

## 说明

`external_dataset_index.json` 只保存索引与元数据，不直接包含可评测样本。
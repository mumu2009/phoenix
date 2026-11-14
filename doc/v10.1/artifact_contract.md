# v10.1 训练产物契约（framework 化）

目标：**改架构与长期训练可以并行推进，架构变更不会让已训练成果作废。**

## 1. 产物清单与稳定契约

| 产物 | 路径 | 形状 | 版本字段 |
| --- | --- | --- | --- |
| 自主状态 | `runtime_store/autonomy_state.json` | 对象；`sessions` 必须存在 | `schemaVersion`（当前 2）、`schemaTag` |
| 专家权重 | `runtime_store/experts/study_tiny/weights.json` | TinyExpert.toJson | `paramCount`/`vocab`/`hidden`/`classes` |
| 情绪调控矩阵 | `runtime_store/emotion_modulation_matrix.json` | `{"rows":[[8x8]]}` | 行数=8 自校验 |
| 语料预训练戳 | `runtime_store/corpus_pretrain_stamp.json` | 已灌入单元数 | 时间戳 |
| MMLU 历史 | `runtime_store/long_run/mmlu_history.jsonl` | 每行一条 | 时间戳 |

## 2. 兼容规则（已实现）

- **导出**：`autonomy_stack.cpp:exportState` 写入 `schemaVersion=2` 与 `schemaTag="v10.1"`。
- **导入**（`importState`）：
  - `schemaVersion` 缺失 = v10.0 旧文件，**仍然接受**；
  - `schemaVersion > 2`（未来版本）：打印告警，**读取它认识的字段**，不丢弃；
  - 所有字段读取都带默认值（`state.value("x", 当前值)`），新增字段被忽略，删除字段回退默认；
  - `sessions` 缺失才拒绝（这是唯一的结构必需项）。
- 专家权重：`fromJson` 按 `vocab/hidden/classes` 重建矩阵，尺寸不符时以权重文件为准重建，不崩溃。
- 情绪矩阵：加载时校验 64 个元素，不足则退回单位阵（旧行为）。

## 3. 架构变更的安全流程

1. **只增不改**：新字段用新 key，旧 key 保留至少一个版本；
2. **改语义必须升 `schemaVersion`**，并在本文件记录迁移说明；
3. 训练产物永远只依赖"路径 + 形状"，不依赖代码里的类布局；
4. 改架构前先备份：`runtime_store/autonomy_state.pre-<变更名>-<时间>.json`；
5. 构建后用 `gtest` 的 persist/restore 测试确认旧文件仍可导入。

## 4. 领域专家：训练集 / 留出集

专家训练改为**两集分离**（v10.1，`tiny_expert.cpp:trainProjectTree`）：

- 训练集：4/5 样本（`i % 5 != 4`）；
- 留出集：1/5 样本（`i % 5 == 4`），**从不参与拟合**；
- 输出同时报告 `trainAccuracy` 与 `heldoutAccuracy`；
- 应用（打分/预测）对两边都生效——留出集只用于"诚实评估"，不用于训练。

## 5. 锁的分层

- **细粒度锁**（各资源自有）：`llamaCallMu_`、`outlineMu_`、`graphMu_`、`gWsCacheMu`、`experienceAdd` 静态锁；
- **粗粒度兜底**：`missionDelibMu_`，由 `autonomyLoop.missionDelibLock` 控制（默认 true）。
  关闭它即进入"仅细锁"模式；只有在连续 2 小时并行任务 0 崩溃后才可默认关闭。

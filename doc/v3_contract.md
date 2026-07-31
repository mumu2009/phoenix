# v3.0 Contract: GNN + Transformer

## 1) 不变约束

- 锁定组合思想：使用 GNN 的长链结构特征增强 Transformer 的长距离语料适应能力。
- 允许替换引擎：GNN/Transformer 的具体实现可演进，但接口语义保持一致。

## 2) 主链接口约束

- 核心请求入口：`/api/chat`。
- Transformer 能力入口：`/api/transformer/chat` 及相关训练/验证接口。
- 响应最小契约：`{ ok, result: { reply } }`，可选字段包含 `latency`、`verify`、`graph`、`addon`。

## 3) 图增强契约

- 图上下文可通过文本上下文注入。
- 图嵌入可通过桥接层注入 Transformer 推理过程。
- 当图增强开关关闭时，系统可回落到纯 Transformer，但默认保持图增强可用。

## 4) 训练集策略

采用混合模式：

- 仓内小样本：用于快速回归和功能验证。
- 外部大样本：用于训练规模化，仓库仅保存索引、校验和说明。

数据治理要求：

- 明确数据来源、版本、许可边界。
- 保留最小可复现实验配置。

## 5) 实施优先级

1. 语法与构建稳定（compile/conan 可重复）。
2. 功能完整（接口契约与并行旁路可用）。
3. 性能优化（并发、缓存、资源利用）。
4. 代码组织优化（模块边界和可维护性）。

## 6) 运行时监控与清洗契约

- 监控接口：
	- `GET /api/monitoring/stats`
	- `POST /api/monitoring/reset`
- 监控指标最小字段：
	- `uptimeSec`
	- `cleaning.enabled / cleaning.maxChars / cleaning.cleanedInputs / cleaning.cleanedSamples`
	- `routes[route].calls/errors/lastMs/maxMs/avgMs`
- 数据清洗入口：
	- 在线请求：`/api/chat`、`/api/transformer/chat`
	- 训练请求样本：`/api/transformer/pretrain`、`/api/transformer/joint_train`、`/api/transformer/ga_optimize`
- 运行时开关（`PATCH /api/runtime/features`）：
	- `dataCleaningEnabled: boolean`
	- `dataCleanMaxChars: integer`

## 7) 模型生命周期契约

- 生命周期状态：
	- `GET /api/model/lifecycle`
- 压缩计划（剪枝/量化）：
	- `POST /api/model/compress`
- 可解释输出（支持度 + verify）：
	- `POST /api/model/explain`
- 部署目标与版本管理：
	- `POST /api/model/deploy`
- 在线增量更新：
	- `POST /api/model/update`

最小字段约定（新增）：

- `GET /api/model/lifecycle` 返回：
	- `servingCluster.replicas / routingPolicy / fallback`
	- `updateSeq`（在线更新序列号）
- `POST /api/model/compress` 返回：
	- `result.estimatedSizeRatio`
	- `result.estimatedSpeedup`
- `POST /api/model/deploy` 请求可选：
	- `replicas`、`routingPolicy`、`canaryPercent`
- `POST /api/model/update` 请求可选：
	- `activateVersion`、`warmupBatches`
	- 返回 `result.seq / result.beforeVersion / result.activeVersion`

约束：

- 以上接口只定义生命周期编排，不改变 GNN+Transformer 主链语义。
- 生命周期清单持久化到 `runtime_store/model_lifecycle/manifest.json`。

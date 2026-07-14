# SparkArray 联合加权与 GGUF 启动校验说明

## 1. GGUF 模型目录与启动前约束

主程序在 `transformer-mode` 为 `llamacpp` 或 `bitnet` 时，改为强制使用本地 `.gguf` 模型文件：

- 默认模型目录：`GGUF_models/`
- 可覆盖参数：`--gguf-models-dir`（或环境变量 `AI_GGUF_MODELS_DIR`）
- 自动检测逻辑：
  - 目录中仅 1 个 `.gguf`：自动绑定到对应后端模型参数
  - 目录中 0 个或 >1 个 `.gguf`：启动失败并要求显式指定模型
- 显式指定时：
  - 支持绝对路径
  - 支持相对路径（优先按 `GGUF_models/` 解析）
  - 文件必须真实存在且后缀为 `.gguf`

### 相关参数

- `--transformer-mode=llamacpp|bitnet`
- `--llamacpp-model`
- `--bitnet-model`
- `--gguf-models-dir`

---

## 2. SparkArray：从 GNN 单通道加权升级为 GNN + Transformer 联合加权

`SparkArray::aggregateResults` 现在会融合两类信号：

1. **GNN 控制器回复**（原有信号）
2. **Transformer NLP 候选回复**（新增信号）

新增输入字段：

- `sparkTransformerCandidates`（数组）
  - 每项结构示例：
    - `source`: 后端标识（`ollama` / `llamacpp` / `bitnet` / `native`）
    - `reply`: 文本回复
    - `latency`: 延迟（毫秒）
    - `affinity`: 与请求 embedding 的相似度（可选，缺省时在聚合内部计算）

融合方式：

- 统一按 `affinity + latency + overlap` 计算评分，映射为 token vote 权重。
- 对 Transformer 候选使用略高的 score boost，形成与 GNN 的联合投票。
- 输出字段新增：
  - `voteBudget`
  - `gnnSignals`
  - `transformerSignals`
- 聚合方法标识更新为：`gnn-transformer-rf-ensemble`

---

## 3. /api/array/chat 的联动行为

网关在处理 `/api/array/chat` 时会根据当前 `transformer-mode` 自动尝试获取 1 条 Transformer 候选回复，并注入到 `sparkTransformerCandidates`：

- 默认开启：`options.includeTransformerNlp=true`
- 可关闭：`options.includeTransformerNlp=false`

当 Transformer 回复为空或请求失败时，会自动降级回仅 GNN 信号，不阻断 SparkArray 主流程。

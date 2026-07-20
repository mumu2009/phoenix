# SparkArray 旧实验数据弃用提醒

由于本阶段已完成以下集成，**在此之前的 SparkArray 实验数据已不具备可比性，建议全部丢弃**：

1. **SparkArray I** 已接入 `/api/chat` 主链路，作为 GNN 调度器使用（`spark.gnnScheduler.enabled`）。
2. **SparkArray II** 已在 `llama-server` 层实现随机森林投票 + 随机微型掩码（`llama_voter.enabled`）。
3. 语音 IO 与图像输入已接入 `config/phoenix.json` 运行时超参，并新增情感反馈链路。
4. 四阶段 `EmotionVocabWeightTable` 已集成到 `frontend_server` 并在 gateway 的 `/api/chat` 流程中进行 `logit_bias` 与 `context` 调制。

## 建议清理的目录/文件

- 旧的 `runtime_store/spark_*` 或 `sparkarray_history*` 实验日志
- 旧的 `runtime_store/gnn_*` 与 `emotion_*` 缓存（情感状态 DB 可保留，因为结构未变）
- 任何在 `outsides/` 下以 `sparkarray` 命名的旧测试产物

> 若后续需要重新标定 SparkArray，请在新构建上重新采集至少 50~100 轮对话数据后再分析。

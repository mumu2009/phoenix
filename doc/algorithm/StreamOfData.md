# Stream Of Data（v3.0）

```mermaid
flowchart LR
    U[User Input] --> C0[/api/chat or /api/transformer/chat/]
    C0 --> S1[Input Sanitize]
    S1 --> G1[MemeGraph Activation]
    G1 --> G2[Graph Context + Embeddings]
    G2 --> T1[Transformer Inference]
    T1 --> V1[/api/transformer/verify]
    V1 --> R1[Normalized Response reply/latency/graph/addon]
    R1 --> F1[Frontend Provider Adapter]

    D0[tests + robots samples] --> D1[Data Cleaning]
    D1 --> D2[TrainSample Contract]
    D2 --> T2[/api/transformer/pretrain & joint_train]

    M0[Route Metrics] --> M1[/api/monitoring/stats]
    M2[Reset Metrics] --> M3[/api/monitoring/reset]

    L0[Model Lifecycle] --> L1[/api/model/compress]
    L0 --> L2[/api/model/explain]
    L0 --> L3[/api/model/deploy]
    L0 --> L4[/api/model/update]
```

## 1) 在线数据流

- 请求入口先做输入清洗（控制字符/空白折叠/长度上限）。
- 图增强阶段输出 graphContext 与 graphEmbeddings。
- Transformer 阶段生成 reply，并保留延迟与可选诊断信息。
- verify 阶段计算一致性分数，用于解释与质量闸门。
- 前端统一消费归一化结构，保障 provider 切换不破坏 UI。

## 2) 训练数据流

- 仓内样本来自 tests、robots，可审计且可回放。
- 外部样本仅通过索引+校验进入流程，不直接入库原文。
- 样本统一映射为 TrainSample：input/target/graph。
- pretrain 与 joint_train 保持同一清洗策略，避免线上线下偏差。

## 3) 监控数据流

- 每个核心路由记录 calls/errors/lastMs/maxMs/avgMs。
- 清洗计数记录 cleanedInputs/cleanedSamples。
- 通过 /api/monitoring/stats 拉取，通过 /api/monitoring/reset 复位。

## 4) 生命周期数据流

- compress：记录压缩计划（剪枝/量化），可选同步到 Transformer 参数。
- explain：对输出进行 token 支撑度分析并附加 verify。
- deploy：记录部署目标与版本，支持滚动/金丝雀信息。
- update：记录增量包、checksum、策略并入清单。

所有生命周期事件落盘到：

- runtime_store/model_lifecycle/manifest.json

## 5) 回归闸门

每次并行改动都必须通过：

1. compile.bat 构建成功
2. /api/chat、/api/transformer/chat、/api/transformer/verify、/api/tests/list 可回归
3. 前端 npm test 通过
4. 文档契约（README + algorithm + StreamOfData）字段一致
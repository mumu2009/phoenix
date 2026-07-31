# 算法总览（v3.0）

## 1. 不变核心

本项目固定采用“GNN 增强 Transformer”组合范式：

- GNN 负责图结构检索、关系激活、语义线索凝聚。
- Transformer 负责序列推理、回复生成、可验证输出。
- 两者之间通过图上下文文本与图嵌入桥接，不允许退化为纯替换关系。

## 2. 推理主链

1) 输入文本进入图侧（MemeGraph + message passing）获取激活子图。  
2) 图侧输出两类信息：
- 结构化上下文文本（graphContext）
- 图嵌入（graphEmbeddings，可选）
3) Transformer 在保持原始用户语序语义的同时接收图增强信息。  
4) 输出 reply，并可选返回 verify/latency/graph/addon 等可观测字段。

该链路对应后端主接口：

- POST /api/chat
- POST /api/transformer/chat
- POST /api/transformer/verify

## 3. 形式化表达

设输入序列为 $x$，图为 $G=(V,E)$，图增强模块输出为 $h_G$，Transformer 输出为 $h_T$。

图侧传播：

$$
H^{(l+1)} = \sigma\left(\hat{A}H^{(l)}W^{(l)}\right)
$$

融合表示：

$$
z = f(h_T, h_G; \alpha)
$$

其中 $\alpha$ 为图增强权重（运行时可调），但默认保持开启。

最终目标：

$$
\min_{\theta_G,\theta_T} \mathcal{L}_{task}(z, y) + \lambda \mathcal{L}_{verify}
$$

其中 $\mathcal{L}_{verify}$ 用于约束回复与输入/图上下文的一致性，降低语义偏离。

## 4. 训练与数据策略

训练集采用混合模式：

- 仓内小样本（tests + robots）：用于快速回归与小规模预训练。
- 外部大样本：仅在仓库保存索引、版本、哈希，不提交大体积原文。

数据治理要求：

- 数据来源可追溯（owner/reviewer/version）。
- 样本清洗规则固定（非法字符过滤、长度上限、空样本剔除）。
- 每次训练保存模型、数据、配置版本映射。

## 5. 生命周期分层（todo #12/#13/#14/#17）

v3.0 先定义统一生命周期接口，再渐进增强：

- 压缩层（compress）：剪枝率、量化位宽、计划化变更。
- 解释层（explain）：输出支持度、显著 token、verify 对齐结果。
- 部署层（deploy）：目标、版本、金丝雀比例、滚动策略。
- 在线更新层（update）：包索引、校验、增量策略。

对应 API：

- GET /api/model/lifecycle
- POST /api/model/compress
- POST /api/model/explain
- POST /api/model/deploy
- POST /api/model/update

## 6. 并行升级原则

在功能并行推进时，必须遵守闸门顺序：

1. 可编译（compile.bat + conan install）
2. 契约稳定（/api/chat 与 /api/transformer/* 返回兼容）
3. 回归可执行（前端单测 + 后端接口回归脚本）
4. 性能优化（延迟/吞吐/锁等待/内存）


## 8. 结论

v3.0 的算法演进目标不是“替换主链”，而是在主链稳定的前提下增强可观测性、可治理性、可部署性，确保 GNN+Transformer 组合长期可维护。
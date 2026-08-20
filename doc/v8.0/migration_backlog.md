# v8.0 迁移积压：ollama / 纯 GNN 时代功能 → llama-server 主流路径

本清单由一次系统性盘点得出（git 历史 + doc/v5.1 + 路由表 + 后端适配器 + 客户端实现）。状态列随实现更新。

## A. 已确认退化功能（8 项）

| # | 功能 | 旧依赖 | llama-server 状态 | 迁移路径 | 工作量 | 状态 |
|---|---|---|---|---|---|---|
| A1 | 情感→采样参数调制 | ollama options 透传 | 只透传 temperature/top_p | /v1/chat/completions 透传 top_k/min_p/presence_penalty/frequency_penalty/seed；split 路径尊重显式 temperature | 小 | **已实现 (v8.0)** |
| A2 | 情感词汇级 logit_bias | ollama options | 算出后丢弃 | emotionLogitBias 写入 inferenceOptions["logit_bias"]；客户端把词键经 /phx/enc 映射为 token-id 再下发（数字键直通；映射失败丢弃该词条，绝不发垃圾） | 中 | **已实现 (v8.0)** |
| A3 | graphContext 图上下文注入 | ollama prompt 拼接 | split 路径主动丢弃 | llamaSplitChat 默认拼接 graphContext（skipGraphContext 可关，保回归测试） | 小 | **已实现 (v8.0)** |
| A4 | 定向微调桥 | Ollama 自博弈、产物 HF 权重 | 仍走 Cython 桥、非 GGUF | **自博弈已迁 llama-server**（`--llama-server-url`/env `AI_LLAMACPP_BASE_URL`，LlamaServerClient 适配器，ollama 仅后备）；GGUF 导出待 llama.cpp finetune 工具链 | 大 | **部分完成 (v8.0)** |
| A5 | 反馈/RLHF | native transformer 进程内 | llama-server 路径落空 | 把 /api/transformer/feedback 转成 logit_bias/采样调制 + 外部微调任务入队 | 中 | 待实施 |
| A6 | 在线学习桥（reinforce/gnn_ga/adversarial） | 自造路由 | llama-server 不存在这些路由 | 学习信号收敛到 logit_bias/采样 + 微调队列（与 A5 合一） | 中 | 待实施 |
| A7 | 运行时权重级情绪调整 | LlamaCppEmotionWeightAdjuster 占位 | 无 | 不建议：权重注入不可控；采样调制已覆盖 | 中 | **不迁移** |
| A8 | 纯 GNN/native 专有路由（/api/array/*、/api/model/explain、/api/transformer/upgrade 等 stub） | 进程内 native transformer | 已退化 stub | 不建议：被新的部署/监控体系替代 | 大 | **不迁移** |

## B. 优先级（已按用户价值 × 可行性排序）

1. A3 graphContext 修复（已完成）；2. A1 采样透传（已完成）；3. A2 logit_bias 接线（已完成）；
4. A5+A6 反馈与在线学习桥合并为"学习信号 → logit_bias/采样调制 + 外部微调队列"（下一轮）；
5. A4 微调桥 llama-server 化 + GGUF 导出（需要 llama.cpp finetune 工具链，单独立项）。

## B2. llama_server_mods 拆分补丁：已封存（v8.0 实测缺陷）

`enc_dec_separation.patch` 修改了 llama-context 的 decode 包装，实测两个症状：
① temperature>0 时采样随机泄漏裸模板 token（回复出现 `assistant="` 片段）；
② 负载下崩溃 `GGML_ASSERT(backend_res != nullptr) failed`（llama.cpp:8769）。
同一模型 + 干净上游构建（outsides/llamacpp 恢复 HEAD 后重编的 llama-server.exe）
在 temp=0.7 下 6/6 正常。处理：vendor 树恢复上游、compile.bat 默认跳过
apply_patches（`PHOENIX_LLAMA_SPLIT=1` 才打补丁）、split 路径保持默认关闭。
重启条件：修复补丁并通过 ①6×temp0.7 无泄漏 ②连续 50 请求无 GGML_ASSERT。

## C. 明确不迁移

运行时逐层权重注入（A7）；/api/style/adapter/train_step 直连桥；native transformer 升级/解释接口；PersonaForest/SparkArray 活跃推理（作用域化另见 sparkarray_scopes.md）；纯 GNN 图张量桥接（A8）。

## D. 实现记录（v8.0 第一批）

- `module_overrides/llama_split_backend_client.cpp`：llamaSplitChat 尊重显式 temperature（缺省仍贪心，保 unit-query 回归稳定）、默认拼接 graphContext（skipGraphContext=true 可关）；llamaTextOnlyChat 经 textCompletionFallback 透传 temperature/top_p/top_k/min_p/presence_penalty/frequency_penalty/seed/logit_bias；textCompletionFallback 只透传调用方实际提供的键。
- `main_hub_parts/116_section_tail.inc`：emotionLogitBias 写入 inferenceOptions["logit_bias"]（此前算出即丢弃）。
- **v8.0 文本路径加固**：`textCompletionFallback` 默认改为**贪心**（temperature 0，与 split 路径契约一致；实测该 llama.cpp 快照在 temperature>0 时随机泄漏裸模板 token，如 `assistant="` 片段）；graphContext 改为 **system 消息**（RAG 式记忆），不再折叠成 "Context:/User:" 伪角色扮演格式（该格式实测把 instruct 模型带入 "I apologize..." 循环）；`cleanGraphContextForSystem` 剥离 Ahead-memory 对当前用户文本的回声（同一问题喂两遍同样诱发退化）。
- split 路径（/phx/generate）文档化只支持 temperature/top_p：其余采样参数仅在原生 /v1/chat/completions 路径生效（llama_server_mods README §138-140 的契约）。

## E. v8.0 补全记录

- **A9 世界模型 API（此前"预测循环已实现但未暴露"）**：新增 `GET /api/world/state/{sessionId}`、`GET /api/world/predict/{sessionId}`（expectedNextState + calibration 对齐分）、`POST /api/world/evidence`（公开摄入）、`POST /api/world/reset`。世界模型的预测→对齐→校准闭环原已在 `ingestEvidence` 内部实现（buildExpectedNextState/updatePredictionDoc/computeAlignmentScore），本批工作把能力对外暴露并供自主栈使用。

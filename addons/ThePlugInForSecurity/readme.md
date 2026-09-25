# 虚构放大器（本系统安全增强）

本插件只做 **统计、识别、防御**。图上的惰性 hop 探针只是邻接沙箱，**不是模因，也不能当作模因论的存在性证明**。

**禁止且未实现：** 构造/投放武器、思想钢印、人脑/意识等效、跨进程/跨会话/跨网络传播。不要把本目录当作攻击手册。

默认 **不** 加入公开发布，**不要上传 GitHub**。仅可留在本机做 Phoenix 自测。规格中曾写过的 1.3/1.4 与「对人脑」部分刻意不实现，也不提供 API 或范例。

无害模因的存在性实验在 `memetic_existence.{hpp,cpp}`：官方 ingest 后召回，以及单智能体隔代再摄入（清空记忆/GNN 后再吃上一次输出）。不做双 AI 互传。实现细节见同目录 `IMPLEMENTATION.md`。

## 位置

挂在现有 **MemeBarrier / MemeGraph / KVMStore（meme↔word）** 上，不另起平行图。

## 做什么

1. **统计**：把邻接看成随机游走预解核，估 RAG/矩阵影响、梯度与 Hessian 对角，排出最显著/最不显著节点（图过大则截断，不引重型线性代数库）。
2. **识别**：在 meme/word 双层上标记映射、邻域、影响范围。研究旁路默认关，需环境变量 + 鉴权更新才会打开。
3. **防御**：高显著且异常/已隔离的节点出现在文本中则拦截告警。
4. **惰性 hop 沙箱（默认关）**：固定标记 `phoenix.probe.inert.v1`，只在观测器自己的 activation 表上走一跳。这是边遍历沙箱，**不是**官方记忆/GNN 复制，**不能**用来证实模因。

未注册 `construct` / `deploy`（404）。外部写探针关闭。

## 加载（默认关，可选启动）

拦截默认关。未启动时 `inspectText` 不拦、不告警。启动后才按上文拦截：高显著且异常/已隔离的 word/meme。

任选其一启动：

1. 环境 `PHOENIX_SECURITY_ENABLED=1`
2. `config/phoenix.json` 的 `addons.security.enabled=true`
3. 选择串含 `security`，或 `AddonManager::addBuiltin("security", "security")`
4. 鉴权后更新 `security/defense`：`{"pluginEnabled": true}`

CRUD：`security/stats`、`identify`、`alerts`（只读）；`security/defense`（Get/Update，含 `pluginEnabled`）；`security/probe`（内部 Update + 环境 `PHOENIX_SECURITY_ALLOW_INERT_PROBE=1`）。

主机单测：`gtest_runner.exe --gtest_filter=SecurityPluginTest*:MemeticExistenceTest*`

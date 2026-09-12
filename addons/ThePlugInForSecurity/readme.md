# 虚构放大器（本系统安全增强）

本插件只做 **统计、识别、防御**，以及 **本进程 AI 图内的无害惰性存在性探针**。

**禁止且未实现：** 构造/投放武器、思想钢印、人脑/意识等效、跨进程/跨会话/跨网络传播。不要把本目录当作攻击手册。

默认 **不** 加入公开发布，**不要上传 GitHub**。仅可留在本机做 Phoenix 自测。规格中曾写过的 1.3/1.4 与「对人脑」部分刻意不实现，也不提供 API 或范例。

实现细节见同目录 `IMPLEMENTATION.md`。

## 位置

挂在现有 **MemeBarrier / MemeGraph / KVMStore（meme↔word）** 上，不另起平行图。

## 做什么

1. **统计**：把邻接看成随机游走预解核，估 RAG/矩阵影响、梯度与 Hessian 对角，排出最显著/最不显著节点（图过大则截断，不引重型线性代数库）。
2. **识别**：在 meme/word 双层上标记映射、邻域、影响范围。研究旁路默认关，需环境变量 + 鉴权更新才会打开。
3. **防御**：高显著且异常/已隔离的节点出现在文本中则拦截告警。
4. **惰性探针（默认关）**：固定标记 `phoenix.probe.inert.v1`，只在本进程图上走一跳，证明边上传导存在。不是武器，不改权重，不关防御，不外泄。

未注册 `construct` / `deploy`（404）。外部写探针关闭。

## 加载

不进 `createDefaultBuiltinAddons`。显式：`AddonManager::addBuiltin("security", "security")`，或选择串含 `security`。

CRUD：`security/stats`、`identify`、`alerts`（只读）；`security/defense`（Get/Update）；`security/probe`（内部 Update + 环境 `PHOENIX_SECURITY_ALLOW_INERT_PROBE=1`）。

主机单测：`gtest_runner.exe --gtest_filter=SecurityPluginTest*`

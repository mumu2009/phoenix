# 虚构放大器（本系统安全增强）实现说明

本实现**只含统计、识别与防御**，**不含**构造投放武器，**不含**对人脑 / 意识涌现 / 可向人类传播的模因部分。

用户规格仍保留在同目录 `readme.md`。规格中的 1.3、1.4 与第 2 部分**刻意不实现**，也不提供 API、PoC 或投放范例。本插件默认**不打包进公开发布**，**不要上传 GitHub**；仅可留在本地仓库做本系统自测。

## 与现有结构的关系

本插件是 **MemeBarrier + MemeGraph + KVMStore（meme/word 多对多）** 的增强，不另起一套平行图。

- 离散图来自现有 `MemeGraph` 节点与边。
- 映射来自现有 `KVMStore`（`word ↔ meme`）。
- 巡检仍走 `MemeBarrier::inspectText` / `scanNetwork` / `isolateMeme`。
- 对外资源走已有 `phoenix/util/module_resource.hpp` 与 `/api/modules/{moduleId}/resources/{type}`。

## 1.1 统计

将离散邻接近似为随机游走预解核 `P = (I - α D⁻¹ A)⁻¹`（`α=0.85`），并在拉普拉斯坐标上构造径向场。

- RAG 影响：`F(a) = ||P a||²`（种子激活经图扩散后对后续 RAG 输入的能量）。
- 矩阵影响：`aᵀ (PᵀP) a`。
- **梯度** `∇F = 2 PᵀP a`，**二阶** Hessian 对角 `H_ii = 2 (PᵀP)_ii`（是梯度/Hessian，不是单独某个偏导符号）。
- 连续场：节点嵌入到低维拉普拉斯坐标后，在节点处求场的梯度范数与 Hessian 迹。
- 按综合显著度排序，给出最显著 / 最不显著节点。

图规模超过 48 个节点时只分析前 48 个（按摄入顺序截断），避免引入重型线性代数库。

## 1.2 识别

在 meme/word 双层多对多上标记显著点，记录：

- `mapped`：对侧层 ID
- `neighbors`：图邻域
- `impactScope`：对 RAG / 矩阵 / 邻域 / 映射规模的摘要

**研究旁路（观测开关）默认关闭。** 打开需要同时满足：

1. 进程环境 `PHOENIX_SECURITY_ALLOW_RESEARCH_OBSERVE=1`
2. 再通过内部或鉴权后的 `security/defense` 更新 `researchObserve=true`

仅旁路本系统拦截以便自测观测，不提供构造或对外攻击能力。

## 防御（产品落地）

在统计+识别结果上增强 MemeBarrier：

- 高显著节点且被标为异常 / 已隔离时，文本中出现对应 word/meme 则拦截并告警。
- `isolateMeme` 会记入观测器，提高后续拦截。
- 防御默认开启；研究旁路默认关。

## 开关与加载

- **默认不**加入 `createDefaultBuiltinAddons`，避免默认挂上「安全研究」插件。
- 显式加载：`AddonManager::addBuiltin("security", "security")`，或选择串包含 `security`。
- CRUD 资源在 `installBuiltinModuleResources` 时注册（只读统计/识别/告警 + 防御开关）。
- 关闭防御：环境 `PHOENIX_SECURITY_DEFENSE_OFF=1`，或更新 `security/defense` 的 `defenseEnabled=false`。

### CRUD（无构造/投放）

| 资源 | 操作 | 说明 |
|---|---|---|
| `security/stats` | List/Get | 只读统计 |
| `security/identify` | List/Get | 只读识别 |
| `security/alerts` | List/Get | 只读告警 |
| `security/defense` | Get + Update | 防御与研究观测开关 |

未注册类型（含 `construct` / `deploy`）返回 404。外部写操作仍走既有 bearer 鉴权。

## 主机单测

```text
phoenix\compile_gtest.bat
phoenix\run_gtest.bat --gtest_filter=SecurityPluginTest*
```

覆盖：梯度与有限差分对照、显著点排序、识别映射、高影响异常模因拦截、研究旁路默认关、未注册/越权 CRUD。

## 未实现（攻击面）

- 1.3 构造攻击模因组 / 图像 payload / 思想钢印
- 1.4 投放流程或范例
- 第 2 部分全部（人脑、意识涌现、对人传播）
- 任何 exploit、木马植入、打外部系统的方法

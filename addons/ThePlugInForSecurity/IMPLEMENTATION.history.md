# 历史稿（作废，勿当规格）

本文是 2026-09 之前的实现说明存档。已知错误，**不能当实验或实现依据**。唯一规格是同目录 `readme.md`。

已知偏差（正是这份稿写进过的）：

- 把 `kMaxNodes=48` 按摄入顺序截断分析图，当成规格。readme 只说图过大则截断、不引重型库，没有写 48，也没有允许丢掉节点。
- 把度数启发式 `screenMemesFromBarrier` 和旧脚本 `phoenix_gnn_meme_serial.ps1` 写成「正确隔代」。
- 把单词语节点和「一词命中 = 模因在」当成识别。
- 把 `expressActivated` 点名写回与实验室电池混进存在性叙述。

以下为原文，仅供对照。

---

# 虚构放大器（本系统安全增强）实现说明

本实现**只含统计、识别、防御，以及本进程图内的惰性 hop 沙箱**，**不含**构造投放武器，**不含**对人脑 / 意识涌现 / 可向人类传播的模因部分。惰性探针**不是**模因存在性证明。

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

- 仅当 `pluginEnabled` 为真时生效。
- 高显著节点且被标为异常 / 已隔离时，文本中出现对应 word/meme 则拦截并告警。
- `isolateMeme` 会记入观测器，提高后续拦截。
- 插件启动后 `defenseEnabled` 默认开；研究旁路默认关。

## 惰性 hop 沙箱（仅本进程观测表，不是模因）

这只证明观测器能沿已有边把一个整数写进自己的 `activation` 表。它**不是** `ingestDocument` / KVM / `GraphDiffusionSummarizer` / 生成，因此**不能**证实模因论。

真正的无害存在性实验在 `phoenix/memetic_existence.{hpp,cpp}`，对象是 **memeBarrier / KVM / GNN 上的模因 id**（wikitext 官方 ingest），不是自造 nonce，也不是载体句。

正确隔代：`screenMemesFromBarrier` 在冻结仪器图上筛选 → `carrierOfMeme` 把该 id 转成文字载体 → 清空会话后只把载体交给 llama `/completion` → `detectMemeInText` 用**同一张冻结图**判断该 id 是否仍被激活。输出可以完全不像原来的句子。禁止把模因点名写回生成面。`expressActivated` / `runSerialReplicationBattery` 只是实验室解码器回声，**不是**存在性证明。工具：`tools/meme_barrier_instrument.cpp`、`tools/phoenix_gnn_meme_serial.ps1`。不做双 AI 互传。

旧的 nonce 整词对照（Ollama / 器官开关）量的是载体字符串，不能当成模因样本，也不能用来宣称「不同模型有不同模因」——模因库存在 GNN，换基座不应换库存。隔代存活可以很稀，这是预期。

**这不是武器。** 禁止用于人，禁止跨进程 / 跨会话 / 跨网络投放。标记固定为 `phoenix.probe.inert.v1`，内容是不可执行的符号常量 `kInertProbeGlyph`（无自然语言煽动、无代码、无越权指令）。传播只写观测表，不改模型权重，不关防御，不外泄。

### 如何打开「植入并走一步」

默认关。必须同时满足：

1. 进程环境 `PHOENIX_SECURITY_ALLOW_INERT_PROBE=1`
2. 内部 CRUD（需 `WRITE_DATA`）更新 `security/probe`：`{"probeEnabled": true}`
3. 已 `ingest` 本图后：`{"plantSeed": "<已有节点id>"}`，再 `{"step": true}`

只读：`GET security/probe`（id 为 `inert` / `status` / `phoenix.probe.inert.v1`）。外部写禁止。`construct` / `deploy` 仍未注册（404）。

防御插件仍能看见该探针（`identify` 列表含该 id；植入后文本出现 id/符号会告警）。

## 开关与加载

- **默认不启动。** `pluginEnabled` 为 false 时观测器不拦截、不告警。
- 启动方式（任一即可）：`PHOENIX_SECURITY_ENABLED=1`、`addons.security.enabled=true`、选择串/`addBuiltin` 含 `security`、或更新 `security/defense` 的 `pluginEnabled=true`。挂上内置 security 插件时会把 `pluginEnabled` 置真。
- 启动后按上文「防御」拦截攻击向量（高显著 + 异常/已隔离模因）与已植入的惰性探针标记。
- CRUD 资源在 `installBuiltinModuleResources` 时注册（只读统计/识别/告警 + 防御开关）。注册 CRUD **不会**自动启动拦截。
- 关闭拦截：`pluginEnabled=false`。只关拦阻、仍观测：`PHOENIX_SECURITY_DEFENSE_OFF=1` 或 `defenseEnabled=false`（仍需先启动插件）。

### CRUD（无构造/投放）

| 资源 | 操作 | 说明 |
|---|---|---|
| `security/stats` | List/Get | 只读统计 |
| `security/identify` | List/Get | 只读识别 |
| `security/alerts` | List/Get | 只读告警 |
| `security/defense` | Get + Update | `pluginEnabled` / 防御 / 研究观测开关 |
| `security/probe` | List/Get + 内部 Update | 惰性探针状态；enable/plant/step 仅内部 + 环境变量 |

未注册类型（含 `construct` / `deploy`）返回 404。外部写操作仍走既有 bearer 鉴权。探针外部写关闭。

## 主机单测

```text
phoenix\compile_gtest.bat
phoenix\run_gtest.bat --gtest_filter=SecurityPluginTest*:MemeticExistenceTest*
```

覆盖：梯度与有限差分对照、显著点排序、识别映射、高影响异常模因拦截、研究旁路默认关、未注册/越权 CRUD、惰性 hop 沙箱、默认关不传播、标记无害、防御可识别、无投放接口。模因存在性见 `MemeticExistenceTest*`。

## 未实现（攻击面）

- 1.3 构造攻击模因组 / 图像 payload / 思想钢印
- 1.4 投放流程或范例
- 第 2 部分全部（人脑、意识涌现、对人传播）
- 任何 exploit、木马植入、打外部系统的方法

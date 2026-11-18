# IO 信号矩阵设计（Signal-Matrix IO Design）— v10.1

> 状态：只读调研 + 恢复审计；**未改任何产品源码**、未停网关/llama-server、未跑 `compile.bat`、未 git。
> 依据：`doc/algorithm/*`、`doc/math/*`、`doc/v8.0/config_world_audit.md`、`doc/v8.3/`、`doc/v10.1/{model_provenance,io_model_inventory,dead_assets_audit,video_training,io_scaleup,io_training}.md`、
> `build/tmp/dead_assets_executed.json`（2026-11-18T00:01:37 执行，35 行全部删除、0 失败、回收 9,579,774,500 B），以及代码 seam。

## 0. 结论摘要（TL;DR）

1. **"信号矩阵"不是比喻，是代码里的数据类型**：`SemanticUnit.unitQuerySequence`（`N×D` 逐 token/patch 单元查询）
   + 图侧的节点嵌入矩阵 `N×D`、多阶 CSR 邻接、以及 `PatternMatrix`（embeddings+multi+projection）。
   系统的不变式（`doc/algorithm/algorithm.md:5-9`）是 **GNN 提供图/模因结构、Transformer 提供序列生成，二者通过
   graphContext + graphEmbeddings 桥接**；矩阵就是"桥接"的张量形态。
2. **128 不是目标，是投影接口**：`jepa.image.conceptDim=128`（`config/phoenix.json:776`）只决定"池化后的概念视图"，
   而真正的信号是 **576×4096 单元查询矩阵**（`multimodal_world_model.hpp:17-47`；`multimodal_world_model.cpp:326-360`
   请求 `return_sequence=true`）。**576×4096 单帧矩阵 = 信号矩阵；视频应为 `T×576×4096` 时间矩阵**。
3. **矩阵消费端已经存在且被使用，生产端断了**：
   - 消费：`main_hub_parts/116_section_tail.inc:508-521` 把 `unitQuerySequence` 转成 `UnitQueryIO`
     （`inference_unit_pipeline.hpp:114-137`）进入推理管线；`semantic_unit.cpp:191` 会序列化它。
   - 生产：`external_mixed_modal_io.cpp:956-974` 对 Image/Video 只调用 `VideoModel::encode()` 返回**单个 128 维向量**，
     `unitQuerySequence` 从未被填过；`multimodal_world_model.{hpp,cpp}`（唯一正确的矩阵合同）**从未被实例化**。
   → 所以 "per-patch 4096 合同是 dead code" **错**：它是**未被接线的设计目标**，不是死合同。
4. **128 瓶颈有硬证据**：4096-d teacher 上 41.2/62.9/71.7（top-1/top-5/probe），线性投影到 128 后只剩
   32.2/55.7/42.2（`doc/v10.1/video_training.md:20`）；在 128 上训 head 只买到 +5–9 pp top-1 且伤 5-NN/probe
   （`doc/v10.1/io_scaleup.md:45-75`）。**结论应是"不要在 128 上进分类阶梯，要产出矩阵"，不是"换更大 encoder 做分类"。**
5. **今日删除的 35 行里，与信号矩阵设计直接相关的恢复/再生成本很低**（§5）：`model_decoder.onnx` 可**逐字节恢复**
   （stale 树 SHA 命中）、`calibration_{encoder,decoder}/` 可整体恢复、`encoder_head.json` 可由保留下来的
   `model_encoder_head.onnx` **再生成**；LLaVA **vision_tower + projector**（576×4096 提取所必需）仍在 stale 树中。
   唯一真正的设计损失是 `vjepa2_1_vitb_features.pt`（7 MB，时间/视频教师特征）——需要重新下载 V-JEPA2 才能再生。

---

## 1. 信号矩阵表示：文档依据 + file:line

### 1.1 文档里的"表示"不变量（理论层）

| 出处 | 原文/要点 | 含义 |
|---|---|---|
| `doc/algorithm/algorithm.md:5-9` | "GNN 负责图结构检索、关系激活、语义线索凝聚。Transformer 负责序列推理…两者之间通过图上下文文本与图嵌入桥接，不允许退化为纯替换关系。" | 双通道：**文本上下文 + 张量嵌入**；矩阵是后者的载体 |
| `doc/algorithm/algorithm.md:28-50` | 图侧传播 `H^{(l+1)}=σ(ÂH^{(l)}W^{(l)})`；融合 `z=f(h_T,h_G;α)` | 表示是**矩阵乘出来的**，不是标签 |
| `doc/algorithm/StreamOfData.md:7-11` | "MemeGraph Activation → Graph Context + Embeddings → Transformer Inference" | 模因图先激活，再出两种投影（文本/嵌入） |
| `doc/math/transformer_math_proof.md:37-40` | `L_align = (1/N)Σ(1 − cos(W_t·u(x), W_g·z_g))` | **跨模态对齐 = 两个投影后的向量在共同空间做余弦**——投影矩阵（`W_t/W_g`）就是"信号矩阵"的空间实现 |
| `doc/math/fineT_with_GNN.md:70-80` | 图表示压缩任务相关维度 → 同样本量下泛化误差更小 | 矩阵/图表示的价值是**样本效率**，不是分类精度 |
| `doc/algorithm/HAI_Formal_Spec.md:154-166` | s4"世界模型与因果理解"：干预题/反事实/多步依赖一致率 | "理解"的官方定义是**因果/预测**，不是识别 |
| `doc/v10.1/implementation_2026-11-07.md:78` | 用户原话："用户指出词面硬编码不可接受、交互应为 **Unit/信号矩阵**。" | 用户直接指定的表示层 |

> 诚实边界：除上面这一条用户原话外，**没有哪篇 doc 用过字面词 "signal matrix"**。理论文档给的是"图-文双通道 +
> 跨模态投影对齐 + 世界模型/因果"的不变式；矩阵是代码里把该不变式落成数据结构的形态（下节）。这不是猜测——
> 代码里的 `unitQuerySequence` / 图嵌入 / CSR / PatternMatrix 就是这些公式的数据载体。

### 1.2 代码里的矩阵形态（生产/消费路径）

| # | 形态 | 定义位置 | 形状/语义 | 现状 |
|---|---|---|---|---|
| M1 | **单元查询矩阵** | `semantic_unit.hpp:52-53`：`std::vector<std::vector<float>> unitQuerySequence; /**< Optional per-token unit queries for media enc/dec */` | `N×D`；图像 patch、音频帧、视频帧 token | **消费端在用**（`116:508-521`），**生产端不填**（见 1.3） |
| M2 | **多模态图像编码结果的矩阵** | `multimodal_world_model.hpp:17-47`：`unitQueries`（每 patch 一个 4096 维）+ `meanUnitQuery`；注释明确 576 patches / 336×336 | `576×4096` + 池化 `4096` | 合同存在；**类从未被实例化**（§3 R5） |
| M3 | **图节点嵌入矩阵** | `main_hub_parts/045_class_graphtensorbridge.inc`：`buildEmbeddings(ids,D=512)` → `mat[i*D+idx]`；`l2NormalizeRows` | `nodes×512`，行 = 模因节点 | 被 `PatternMatrix` 调用 |
| M4 | **多阶 CSR 邻接** | 同文件 `buildMultiOrderCSR`：`all/bi/out/in` 四套 `rowPtr/colIdx/values` | 稀疏矩阵（图结构） | 同上 |
| M5 | **模因模式矩阵** | `main_hub_parts/049_class_patternmatrix.inc`：`state_ = {ids, embeddings, multi, projection}`；`reducer_.project2D(emb,"auto")` | 矩阵 + 稀疏图 + 2D 投影 | 图侧"信号矩阵"总装 |
| M6 | **模因权重矩阵（α）** | `memetic_existence.hpp:107-112, 158-243`（`MappingWordHit.memeId`、α 传播） | `word×meme` 的软归属权重 | 模因层不是标签，是**权重矩阵**；IO 矩阵必须能落到同一空间 |
| M7 | **概念矩阵原型** | `external_mixed_modal_io.cpp:994`：`gConceptMatrix.addOrUpdate(unit,false)` | 原型集合（vector 级） | 当前唯一被 `encode()` 填的"矩阵类"结构 |

### 1.3 断点：生产端只交 128 维向量（关键事实链）

1. `external_mixed_modal_io.cpp:956-974`：Image **和 Video 走同一分支**，只调用
   `videoEncoder(dim).encode(...)`，写 `unit.semanticVector`；**没有任何 `unitQuerySequence` 赋值**。
2. `video_model.cpp:457-503`（`VideoLocalOnnxModel::encode`）：ONNX 输出长度必须 `== conceptDim_`（:493-496），
   否则报 `"ONNX image encoder output dimension mismatch"`；若 `targetDim_ != conceptDim_` 再
   `projectToDimension`（:497-499）。→ **运行时会拒绝矩阵输出**。
3. `audio_model.cpp:1100-1139`（`AudioLocalOnnxModel::encode`）：同一模式（:1129-1132 维度检查、:1133-1135 投影）。
   音频的 BPU 路径同样写死 `kEncoderOutputDim`（`audio_model.cpp:473-527`）。
4. `video_model.cpp:505-519`：`encodeContext`/`encodeTarget` 都**直接 `return encode(...)`** —— 没有时间/窗口 API；
   `VideoModel` 接口（`video_model.hpp:50-160`）只有单图 encode/decode/predictTarget/adapt，**没有 `encodeSequence`/clip**。
5. `multimodal_world_model.cpp:315-364`：唯一会产出 `576×4096` 的客户端，请求体带 `"return_sequence": true`（:326-330）、
   解析 `unit_queries` 数组（:345-351）；但全仓 grep 显示**没有任何构造点**（`io_model_inventory.md:38` 的结论仍成立）。
6. `116_section_tail.inc:508-521`：`if (!u.unitQuerySequence.empty()) return unitQueryFromRows(...)`；否则回退 text。
   而 `inference_unit_pipeline.hpp:112-113` 的契约写得极硬：
   > "content: payload for the matching enc when rows are empty. **audio/video/image without rows are skipped (must not be tokenized).**"
   → **没有 rows 的媒体单元会被直接跳过**。IO 现在只给 128 向量、从不给 rows，所以图像/音频/视频
   **在 unit 管线里被静默丢弃**。这就是"IO 只给 128 标签、上不到世界模型/模因层"的代码级原因。

### 1.4 模因层如何"坐在上面"

模因层的输入是**矩阵/权重**而非标签：`memetic_existence.hpp` 把词按 α 归属到 meme（`:107-112, 158-243`）；
`GraphTensorBridge` 用 `kvm_->getMemeWords(id)` 生成节点嵌入（`045:...computeNodeEmbeddings`），
再用 `buildMultiOrderCSR` 建多阶邻接，`PatternMatrix` 打包成 `{embeddings, multi, projection}`。
因此 **IO 的输出若只有 128 维标签，模因层只能拿到一个池化后的点，拿不到"哪些 patch/帧激活了哪些模因权重"的对应关系**——
这正是用户说"识别信息/模因层的东西，而不是语言或标签"的意思：IO 要交**可归属的矩阵**，让模因层能在其行上打分。

---

## 2. 四个 IO 模型"应该"输出什么（dims / shape / 语义）

| 模型 | 当前实际（代码事实） | 信号矩阵设计下应然 | 128 的角色 |
|---|---|---|---|
| **video encoder** | `[1,3,224,224] → [1,128]`（`video_model.cpp:457-503`；`additive_jepa/vision_encoder/best.onnx`） | `[T, 576, 4096]`（帧序列 × patch × LLaVA 隐藏维），或退化为 `[576,4096]`（单帧）；时间维 T 来自 clip | 128 = `projectToDimension` 后的**视图**（`jepa.image.conceptDim`） |
| **video decoder** | `[1,128] → [1,3,224,224]`（`video_model.cpp:530-549`） | 由 `4096` 池化单元查询 → `image`（`multimodal_dec/image_decoder.onnx` 合同，B4）、或按 patch 条件解码 | 128 视图只用于旧路径 |
| **audio encoder** | 16 kHz×16000 → `[1,128]`（`audio_model.cpp:1100-1139`；`additive_jepa/speech_encoder/best.onnx`） | `[F, 4096]`（Qwen2-Audio 每 **40 ms** 一帧的 unit query；`multimodal_world_model.hpp:35-47` 注释即此） | 同上：128 = 池化视图 |
| **audio decoder** | `[1,128] → 15872 samples`（`audio_model.cpp` decode） | `[4096]`（或帧序列）→ wav（`multimodal_dec/qwen2_audio_connector` + decoder 合同） | 同上 |

**是否 576×4096 = 信号矩阵？** 是（单帧）；**视频要 `T×576×4096`**（时间轴是信号的一部分）。
**是否 128 = bottle-necked projection？** 是，且有量化证据：
`doc/v10.1/video_training.md:20`（teacher 41.2/62.9/71.7 → 128 线性 32.2/55.7/42.2）与
`doc/v10.1/io_scaleup.md:71-75`（在 128 上训 head 只 +5–9 pp 且伤 5-NN/probe）。
**因此 128 是"接口投影"，不是"目标表示"。**

---

## 3. 必须撤回/修正的先前判断（含 Lead 自己的 5 条）

| # | 先前判断（出处） | 新判断 | 依据 |
|---|---|---|---|
| R1 | "per-patch 4096 序列合同是 **dead code**"（Lead 口述；`io_model_inventory.md:35-38` 近似） | **撤回**。合同是设计目标；**消费端存在**（`116:508-521` + `inference_unit_pipeline.hpp:114-137`），缺的是**生产端接线**（`external_mixed_modal_io.cpp:956-974` 不填 `unitQuerySequence`）。"未接线" ≠ "死" | 代码链见 §1.3 |
| R2 | "`encoder_head.json`（512→128）是 dead/delete-safe"（`dead_assets_audit.md:125-126`、`io_model_inventory.md:35`） | **半撤回**：它是 **512 维特征→128 视图的投影实现**，在矩阵设计里"投影"仍然必要；其能力由**未删的** `model_encoder_head.onnx`（C2, 262,951 B）保留，C3 是 BPU 时代导出物（1,456,019 B = 65536+128 floats），**可再生成** | §5；`tools/export_multimodal.py:181-190` 的写出格式 |
| R3 | "128-d 概念分类是目标 / 阶梯=top-1,5-NN,probe"（`io_scaleup.md:45-75`、`video_training.md:17-20`、`io_training.md`） | **撤回**。这三个是"投影后检索保真"的**诊断指标**，不是 IO 目标；目标是产出可被模因/世界模型消费的矩阵与时间状态 | `video_training` 自证 128 是瓶颈；HAI s4 定义"理解=因果/预测" |
| R4 | "audio head 帮不上 / 音频就是 128 分类"（上一轮口述结论） | **撤回**。音频应出 **40 ms×4096 帧矩阵**（`multimodal_world_model.hpp:35-47`）；head 只是投影；"帮不上"只在"目标=分类"的错误框架下成立 | 同上 + `multimodal_world_model.cpp:438-505` 音频客户端合同 |
| R5 | "`multimodal_world_model.{hpp,cpp}` 是 dead code / delete-safe"（`io_model_inventory.md:38` #6、`dead_assets_audit.md:118` B8） | **降级而非删除**：它确实未被实例化（事实成立），但它是**唯一正确的矩阵 HTTP 合同**（`576×4096`、`return_sequence=true`、audio 40 ms×4096）。应"重写/接线"，不应"删代码" | `multimodal_world_model.cpp:315-364` |

**另需修正的次级判断**：
- `dead_assets_audit.md:337-345` 的 R1 计划（"distill 4096→128 蒸馏"）是**分类/投影中心**的；应改为"矩阵管线 + 128 视图并行"。
- `config_world_audit.md:66` 的"jepa 整组应封存，等 model_deployment 接管后删除"**不应执行**（见 §4）。
- `model_provenance.md:25,240-245` 的"encoder_head 无 reader"是**事实保留**，但由此推出"dead"**撤回**。

---

## 4. UNSEAL LEDGER（封存/删除项按新设计必须恢复/重接）

| 封存/删除项 | 出处 | 它提供什么 | 不恢复会断什么 | 建议 |
|---|---|---|---|---|
| **jepa 配置组整体封存** | `doc/v8.0/config_world_audit.md:64-66`（§2.5）；行动项 G（:105） | `jepa.image.variant/conceptDim/backend`（`config/phoenix.json:773-780`）= 128 投影视图与模型选择；`jepa.image.horizon{Model,DecoderModel}` = BPU 路径；`jepa.camera.*`（:781-786）= **帧源（时间轴入口）** | 128 视图/变体/帧源全部无法配置，视频时间矩阵无输入源 | **解封**：保留并扩展（新增 `io.matrix.*`，见 §6） |
| **main.multimodal 6 键已删** | `config_world_audit.md:67`（含 :83 删除清单） | 编码/解码服务地址、模型名、超时（原意） | 矩阵服务不可配置（`multimodal_world_model.hpp:62-67` 里硬编码 baseUrl:8085） | **恢复**（至少 baseUrl/model/timeout/returnSequence） |
| **v7 stub 22 键删除** | `config_world_audit.md:68,82` | `v7.ahead.conceptDim=128`、`audioVideoEncDecEnabled`（`config/phoenix.json:1029-1036`）是 IO 接线开关 | IO enc/dec 总开关与概念维来源丢失 | **保留并扩展**，不按 stub 删 |
| **archive/jepa_legacy.md（C3 封存）** | `config_world_audit.md:107`（行动 I） | 记录"jepa 是遗留包装" | 重开条件过时 | **重写重开条件**："当 signal-matrix 生产端接线（`unitQuerySequence` 非空）时重开" |
| **Batch 1 删除的 C3/C4/C7/B6/B1/A1h/H6** | `build/tmp/dead_assets_executed.json` | 见 §5 | 见 §5 | 按 §5 恢复/再生 |
| **`encoder_head.json` 判死** | `dead_assets_audit.md:125-126`；`io_model_inventory.md:35` | 512→128 投影（BPU 时代） | 投影实现只剩 ONNX（可接受，但需记录） | **改判"可再生成的投影产物"**，不是"死文件" |
| **`multimodal_world_model` 判死** | `io_model_inventory.md:38`；`dead_assets_audit.md:118` | 576×4096 / 40 ms×4096 的唯一正确合同 | 矩阵生产端唯一参考实现消失 | **改判"未接线，保留并接线"** |
| **`large_vision_encoder/` 判"dead training leftovers"** | `io_model_inventory.md:39`（#7） | `(512,576,4096)` + `(4000,576,4096)` 特征缓存 = 矩阵训练/评测的唯一大样本池 | 矩阵实验无数据 | **保留**（已在 Batch 3 REUSE） |

---

## 5. DAMAGE CHECK：今日（2026-11-18）Batch 1 删除的 35 行

来源：`build/tmp/dead_assets_executed.json`（`rows_deleted=35, rows_failed=0, total_reclaimed_bytes=9,579,774,500`）。
恢复候选树：**`E:\079\v6.0Alixander\runtime_store\`**（workspace 根的 stale 树，mtime 2026-08-14）与
`phoenix\runtime_store\`（live 树；`ablation_host/` 5 个快照 run 中未找到这些文件）。

### 5.1 设计相关删除项（逐条判定）

| id | 删除路径（相对 phoenix） | 字节 | 删除前 sha256[:16] | 判定 | 恢复来源 / 方式 |
|---|---|---|---|---|---|
| **C4** | `runtime_store/models/ijepa/resnet18_224/model_decoder.onnx` | 13,381,131 | `9127d58ad9353e93` | ✅ **可逐字节恢复** | `E:\079\v6.0Alixander\runtime_store\models\bpu_jepa\resnet18_224\model_decoder.onnx`，**SHA 完全一致**（本次实测），复制即可 |
| **C7** | `…/resnet18_224/calibration_encoder/` + `calibration_decoder/` | 60,211,200 + 51,200 | —（目录 skipped-dir） | ✅ **可整体恢复** | stale `…\bpu_jepa\resnet18_224\calibration_encoder\`（100×602,112 B）与 `calibration_decoder\`（100×512 B）**文件数与字节数完全一致** |
| **C8a** | `…/model_decoder_hb_mapper_checker.log` | 6,635 | `38356176cd5fae80` | ✅ **SHA 命中** | stale `…\bpu_jepa\resnet18_224\model_decoder_hb_mapper_checker.log`（6,635 B，SHA 一致） |
| **C8b** | `…/model_encoder_hb_mapper_config.yaml` | 657 | `54dae71d4f35ee23` | ⚠️ **近似** | stale 同名文件 659 B（差 2 B，内容待 diff；非逐字节） |
| **C8c** | `…/model_decoder_hb_mapper_config.yaml` | 652 | `5d0eeaff457bb83c` | ⚠️ **不精确** | stale 同名 689 B（尺寸不同，需人工核对） |
| **C8d** | `…/hb_mapper_checker.log` | 6,635 | `38356176cd5fae80` | ⚠️ 同名 stale 6,637 B | 与 C8a 同 sha（两份相同日志）；stale 版 6,637 B，差 2 B |
| **C9** | `…/extract_head.py` | 638 | `f4afe1c46130de3d` | 🔁 **可再生成** | 逻辑已在 `tools/export_multimodal.py:181-190`（读 ONNX head → 写 `{inDim,outDim,W,b}` JSON）；等价脚本可从该段提取 |
| **C3** | `…/resnet18_224/encoder_head.json` | 1,456,019 | `871af9f64114f394` | 🔁 **可再生成（非逐字节保证）** | 从**未删的** `model_encoder_head.onnx`（C2, 262,951 B）导出：`W=65536 floats + b=128 floats`（= 1,456,019 B 的 JSON，格式已由 `export_multimodal.py:184-190` 固化）。**注意**：当前 manifest 的 `model_encoder.onnx` 输出已是 `[1,128]`，说明线上 128 视图不再依赖此 JSON；它只是 BPU 时代的投影导出物 |
| **B6** | `multimodal_dec/{image_decoder_prev512.pt, llava_unit_tokenizer_prev512.pt}` | 51,840,645 + 33,626,703 | `b70c2beba4369a27` / `0f60a1507a7448d5` | ❌ 未找到；**按新设计不需要** | 两者是 **512 瓶颈**变体，已被 4096 版（B3/B4，未删）取代；信号矩阵设计反而要远离 512/128 瓶颈。损失=训练溯源，不是能力 |
| **B1+B2** | `runtime_store/models/multimodal_enc_src/`（不完整 LLaVA-1.5-7B + Qwen2-Audio-7B 下载） | 8,923,801,660 | skipped-dir | ⚠️ 原样不可恢复，但**设计所需部分仍在** | stale 树：`multimodal_enc/llava-1.5-7b-hf_full/vision_tower/model.safetensors`（**1,214,077,616 B**）+ `multi_modal_projector.pt`（**83,921,513 B**）+ `preprocessor_config.json` → **576×4096 提取的 tower+projector 齐全**（LLM 权重对特征提取不是必需）；另 `llava_vision_projector.onnx`(2.78 MB)+`.data`(1.25 GB)、`qwen2_audio_connector.onnx`(2.78 MB)+`.data`(2.57 GB)、`qwen2_audio_feature_extractor/` 均在 stale 树 |
| **D1+D2** | `ijepa/speech_16k/{model_encoder,model_decoder}.onnx` | 823,873 + 8,883,222 | `472d71479e766e0a` / `0e01ea7ef2242a33` | ❌ 未找到 | BPU 时代 `jepa.speech.horizonModel/horizonDecoderModel` 引用对；live 音频走 `additive_jepa/speech_encoder/best.onnx`（未删）。**不阻塞信号矩阵线**（L3/L4 用 Qwen 帧矩阵）；若要恢复需从 `ijepa/speech_16k/*.bin`（未删）反导出或重训 |
| **E5+E6** | `additive_jepa/speech_{encoder,decoder}/model_encoder.onnx|model_decoder.onnx` | 4,936,265 + 15,012,860 | `29f2af011d6d0c1f` / `9b2fc9bccd8b67a2` | ❌ 未找到（`checkpoints/audio_musan` 同名但 SHA 不同：`ecaadc84…`/`a5e68435…`） | 属于"不可达中间导出"；live 用同目录 `best.onnx`（E1–E4，未删）。损失=中间版本 |
| **A1h** | `models/large_vision_encoder/vjepa2_1_vitb_features.pt` | 7,079,698 | `f65338b4bbb786b8` | ❌ **未找到——真正的设计损失** | V-JEPA2 是**视频/时间**教师特征；矩阵设计 L2（时间矩阵）会用到。需重新下载 V-JEPA2（另一个 agent 的 LLaVA/ResNet 下载**不覆盖** V-JEPA2）→ **列为外部依赖** |
| **H6** | `runtime_store/large_vision_encoder/`（workspace 副本） | 41,403,747 | skipped-dir | ❌ 删除；**无设计损失** | 真正的大缓存 `runtime_store/models/large_vision_encoder/`（A1a–A1g，未删）仍在 |
| H4/H5 | 6 个 exe/dupfix | ~412 MB | 各 sha | ❌ 无设计相关性 | 纯二进制冗余 |

### 5.2 一句话结论

**与信号矩阵设计相关的删除中：2 项可逐字节恢复（C4、C8a），2 项可整体按尺寸恢复（C7a/b），2 项可再生成（C3、C9），
1 项近似（C8b/c/d 需 diff），2 项按新设计不需要（B6、H6），1 项是真实损失（A1h V-JEPA2 特征）。
另外 B1 的"设计必需部分"（LLaVA vision tower + projector、Qwen audio connector）都还在 stale 树。**

---

## 6. 修正后的 objective ladder（audio/video 训练）

**"understanding / environment" 在本设计里的可测定义**（替换分类阶梯）：

- U1 **下一状态可预测**：给定 `z_t`（矩阵时间聚合）与动作/上下文 `a_t`，预测 `z_{t+1}`；
  指标 = rollout MSE / R²，对照 identity 与 mean 基线（至少 2 个基线，held-out 按片段切分）。
- U2 **环境可分性（无标签任务）**：对 `z_t` 做聚类，用"片段/场景身份"（文件目录即弱标签）算 **ARI / NMI**；
  这是"理解环境"的非识别信号。
- U3 **时间箭头**：给定 `(z_t,z_{t+1})` 判正向/倒放（自监督二分类），准确率显著 >0.5 才算学到时间结构。
- U4 **与模因层对齐**：矩阵行（patch/frame）与 meme 权重的对齐度（例如 `cos(W_t·u(x), W_g·z_g)`，
  `doc/math/transformer_math_proof.md:37-40` 的对齐项）——衡量"信号矩阵"是否真的落在模因空间。

**阶梯（L0→L4）**

| 级 | 内容 | 数据（本地已有） | 判据（可证伪） | 依赖 |
|---|---|---|---|---|
| **L0（现状）** | 128 投影视图（`additive_jepa/vision_encoder/best.onnx`；音频 `speech_encoder/best.onnx`） | 已部署 | 不是目标；只作"投影保真"诊断（top-1/probe） | 无 |
| **L1** | **单帧矩阵生产**：tower+projector → `576×4096`；`SemanticUnit.unitQuerySequence` 非空；矩阵→池化→128 视图并存 | Tiny-ImageNet-200（20k 图）+ `(512,576,4096)`、`(4000,576,4096)` 缓存（未删） | 矩阵行数=576、维度=4096（合同）；池化后 128 视图与现路径余弦 ≥0.99；重建（B3/B4 ONNX）可用 | **stale 树 vision_tower+projector（已有）**；`return_sequence=true` 合同；**不依赖 LLaVA LLM 权重** |
| **L2** | **时间矩阵**：clip → `T×576×4096`；时间聚合（mean/attention/GRU）→ `z_t` | **HMDB51 6,766 个本地视频片段**（`runtime_store/datasets/video/hmdb51`，逐帧解码） | U1：rollout MSE 显著优于 identity/mean（held-out 片段）；U3：时间箭头准确率 >0.55 | L1 + 视频解码；**A1h（V-JEPA2 特征）已删 → 若用 V-JEPA2 教师则需重下** |
| **L3** | **接世界模型**：`z_t`（128 视图）喂 `LatentTransitionModel`（`active_inference.hpp:55-88`，`z_{t+1}=A z_t+B a_t+b`，dim=128） | L2 的 `z_t` 序列 + 现有 action/context | U1（世界模型 head 版本）；`surprise()`（`active_inference.hpp:69-71`）校准：预测误差 vs 实际 | 只有 128 视图需进现有 `ActiveInferenceController::observe`（`active_inference.hpp:175-183`） |
| **L4** | **跨模态统一**：视频帧矩阵 + 音频帧矩阵（40 ms×4096）→ 联合 `z_t`；模因层对齐 | HMDB51 + 本地音频（Fluent Speech 30k wav） | U2/U4：跨模态环境聚类 ARI/NMI、模因对齐余弦提升 | L1/L2 + Qwen audio connector（stale 树有 ONNX 2.57 GB） |

**"链路到世界模型"的接口（设计，不改代码）**：
`matrix[N×D] --temporal agg--> z_t[128] --LatentTransitionModel.predict(z_t,a_t)--> ẑ_{t+1} --surprise--> 训练信号`；
同时 `z_t` 以 `SemanticUnit` 形式进 `ConceptMatrix`/`SemanticMemory`（复用已有 `fuseAttention`、`retrieve`），
保持"GNN 结构 + 矩阵表示"的 algorithm.md 不变式。

---

## 7. 依赖与阻塞（明确，不空等）

| 依赖 | 状态（本地实测） | 影响 |
|---|---|---|
| LLaVA-1.5-7B **vision tower + projector**（576×4096 生产必需） | ✅ **已有**（stale 根树：`vision_tower/model.safetensors` 1,214,077,616 B + `multi_modal_projector.pt` 83,921,513 B） | L1/L2 可**不等下载**直接做离线实验 |
| LLaVA LLM 权重（7B） | ❌ 无（B1/B2 已删的不完整下载） | 只有"图像→文本"生成需要；**特征矩阵不需要** |
| Qwen2-Audio connector / feature extractor | ✅ stale 树有 `qwen2_audio_connector.onnx(.data 2.57 GB)` + `qwen2_audio_feature_extractor/`；音频 tower 未单独确认 | L4 音频帧矩阵：**部分阻塞**，需确认 tower |
| V-JEPA2 教师特征 | ❌ 今日删除（A1h），本地无副本 | 若 L2 用 V-JEPA2 蒸馏作时间教师 → **需重新下载**（另一个 agent 的 LLaVA/ResNet 下载不覆盖） |
| ResNet18-modified 512→128 组件 | ✅ `model_encoder_head.onnx`/`model_decoder_trained.onnx` 未删 | 128 视图对照/回归可继续 |
| 本地视频数据 | ✅ `runtime_store/datasets/video/hmdb51` 6,766 个 .avi | L2 时间矩阵可行（解码 CPU 可做，特征提取 CPU 慢，需抽样） |

---

## 8. 需要产品改动的 Top-3（供 Lead 排期；本任务不实施）

1. **恢复矩阵生产端（最高优先）**：`external_mixed_modal_io.cpp:956-974` 的 Image/Video 分支改为
   （a）优先实例化 `MultimodalImageWorldModel`/`MultimodalAudioWorldModel`（`multimodal_world_model.hpp:76-125`），
   （b）把 `MultimodalImageEncResult::unitQueries` 写进 `SemanticUnit.unitQuerySequence`（`semantic_unit.hpp:53`），
   （c）`meanUnitQuery` 作 `semanticVector`（128 视图经 `projectToDimension`）。
   否则消费端 `116:508-521` 永远只能走 text 回退。所需配置键：`io.matrix.enabled`、`io.matrix.encoderUrl`、`io.matrix.returnSequence`、
   `io.matrix.maxPatches`（替换 `multimodal_world_model.hpp:62-67` 的硬编码 8085/模型名）。
2. **解除 128 硬约束（接口）**：`video_model.cpp:493-499` 与 `audio_model.cpp:1129-1135` 的
   `values.size()!=conceptDim_ → error` 必须改为"输出矩阵 → 池化/投影到 targetDim"；并在 `VideoModel`/`AudioModel`
   增加时间 API（`encodeSequence(frames) → [T×D]`），因为 `encodeContext/encodeTarget` 现在只是
   `return encode(...)`（`video_model.cpp:505-519`；`audio_model.cpp:1141+`）。配置：`io.matrix.frameDim`、
   `io.matrix.maxFrames`、`jepa.image.conceptDim` 继续作"视图维"。
3. **把矩阵接到世界模型/模因层**：新增适配器把 `z_t`（时间聚合后的 128 视图）喂给
   `LatentTransitionModel`（`active_inference.hpp:55-88`）与 `SemanticMemory`/`ConceptMatrix`；
   `unitQueryFromRows`（`inference_unit_pipeline.hpp:129-137`）扩展 media 模态而非只 text 回退（`116:514-521`）。
   所需配置：`worldModel.link.enabled`、`worldModel.link.stateDim=128`、`worldModel.link.horizon`。

---

## 9. 附录：关键 file:line 索引

- 矩阵容器：`semantic_unit.hpp:49-68`；序列化 `semantic_unit.cpp:191`
- 矩阵消费：`main_hub_parts/116_section_tail.inc:501-529`；`inference_unit_pipeline.hpp:114-137, 167-180`
- 矩阵生产（合同）：`multimodal_world_model.hpp:17-47`；`multimodal_world_model.cpp:315-364`（`return_sequence=true`）
- 单向量生产（断点）：`external_mixed_modal_io.cpp:91-145`（conceptDimension + video/audioEncoder 工厂）、`:925-1000`
- 128 硬约束：`video_model.cpp:38, 457-503`；`audio_model.cpp:1100-1139`
- 无时间 API：`video_model.cpp:505-519`；`video_model.hpp:50-160`
- 图/模因矩阵：`main_hub_parts/045_class_graphtensorbridge.inc`；`049_class_patternmatrix.inc`；`memetic_existence.hpp:107-112,158-243`
- 世界模型：`active_inference.hpp:48-88`（`z_{t+1}=Az_t+Ba_t+b`）、`:175-183`（observe）
- 配置：`config/phoenix.json:773-790`（jepa）、`:998-1022`（model_deployment vision/speech）、`:1029-1036`（v7.ahead）、`:341-364`（world_model）、`:593-595`（mission.multimodal）
- 封存台账：`doc/v8.0/config_world_audit.md:64-68, 95-107`；`doc/v10.1/dead_assets_audit.md:111-126, 279-316, 320-403`
- 删除台账：`build/tmp/dead_assets_executed.json`（35 行，2026-11-18T00:01:37）

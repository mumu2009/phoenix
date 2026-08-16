# Phoenix v7.0 "Arthur" 算法说明

本文件依据 `doc/v7.0/v7.0.md` 与代码实现，对 v7.0 核心算法进行精确描述。

---

## 1. 基本记号

- $v \in \mathbb{R}^{d}$：一个语义向量。
- $\hat{v}$：归一化向量，$\hat{v}=v/\|v\|$，若 $\|v\|=0$ 则 $\hat{v}=v$。
- $\operatorname{sim}(a,b)=a\cdot b /(\|a\|\|b\|)$：余弦相似度。
- $\operatorname{softmax}_i(x)=e^{x_i}/\sum_j e^{x_j}$。

---

## 2. 语义单元（Semantic Unit）与多模态真连接

### 2.1 语义单元结构

定义在 `semantic_unit.hpp` 的 `phoenix::multimodal::SemanticUnit`：

```text
id              : 16 位十六进制稳定 id（generateSemanticId）
modality        : text / image / audio / video / sensor / structured
semanticVector  : 统一语义向量（浮点）
content         : 可选原始内容或摘要
confidence      : [0, 1]
timestampMs     : UTC 毫秒
metadata        : 键值对（如 videoVariant、audioVariant、conceptEncoder 等）
associationIds  : 关联单元 id
modalWeights    : 各模态在融合中的权重
```

### 2.2 向量投影 `projectToDimension`

实现：`semantic_unit.cpp`

目标：让不同模态、不同维度的向量映射到同一目标维度，使跨模态融合不必重训大模型。

伪代码：

```text
projectToDimension(v, targetDim, seed):
    if targetDim == 0 or targetDim == len(v):
        return v
    M = getCachedMatrix(sourceDim=len(v), targetDim, seed)
    // M ~ N(0, sqrt(2/(sourceDim+targetDim)))
    return M * v    // Eigen 矩阵乘法
```

关键性质：
- 投影矩阵由 `(sourceDim, targetDim, seed)` 唯一确定，保证可复现。
- 默认 `seed = 0x61727468U`；图像/语音/文本编码器可指定不同 seed 以区分语义子空间。

### 2.3 归一化与相似度

```text
cosineSimilarity(a, b):
    if len(a) != len(b) or empty: return 0
    dot = Σ a_i * b_i
    sumA = Σ a_i^2
    sumB = Σ b_i^2
    if sumA == 0 or sumB == 0: return 0
    return dot / sqrt(sumA * sumB)
```

`normalizeVector` 为 L2 归一化，若范数为 0 则返回原向量。

### 2.4 融合算子

定义在 `semantic_unit.hpp`：

- `fuseAdd(a, b, targetDim)`：将两者投影到 `targetDim` 后逐元素相加，再归一化。
- `fuseMultiply(a, b, targetDim)`：投影后逐元素相乘（Hadamard）。
- `fuseAttention(query, units, targetDim)`：
  1. 对 `units` 中每个 `u` 计算 $\operatorname{sim}(\hat{q},\hat{u})$。
  2. `softmax` 得到权重。
  3. 加权求和得到融合向量。
  4. 返回权重最大输入单元对应的 modality。

### 2.5 多模态损失（v7.0 设计目标）

```text
L_total = α * L_contrastive + β * L_reconstruction + γ * L_semantic_consistency
```

- `L_contrastive`：拉近同义不同模态样本，推远无关样本。
- `L_reconstruction`：世界模型从概念向量重建原始模态特征。
- `L_semantic_consistency`：语义单元在关联链上保持一致性。

---

## 3. I-JEPA / 视频世界模型

### 3.1 官方变体

定义在 `video_model.hpp`：

| id | arch | repo | patchSize | resolution | params | dataset |
|---|---|---|---|---|---|---|
| ijepa_vith14_1k | vit_h14 | facebook/ijepa_vith14_1k | 14 | 224 | 632M | IN1K |
| ijepa_vith16_448 | vit_h16 | facebook/ijepa_vith16_448 | 16 | 448 | 632M | IN1K |
| ijepa_vith14_22k | vit_h14 | facebook/ijepa_vith14_22k | 14 | 224 | 632M | IN22K |
| ijepa_vitg16_22k | vit_g16 | facebook/ijepa_vitg16_22k | 16 | 224 | 1B | IN22K |

本地权重路径（下载后）：

```text
runtime_store/models/ijepa/<id>/model.safetensors
```

由 `videoModelExpectedWeightsPath(cfg)` 计算。

### 3.2 接口语义

`VideoModel` 接口：`encode`、`encodeContext`、`encodeTarget`、`predictTarget`、`adapt`、`decode`、`status`。

工厂根据部署配置选择后端：

```text
local(cpu|gpu)  -> VideoLocalOnnxModel  运行 additive_jepa/.../best.onnx
local(bpu)     -> VideoHbdnnModel      加载 best.bin / model_encoder.bin
remote         -> VideoRemoteModel      POST JSON 到 remote url
server-client  -> VideoServerClientModel 仅接受客户端发来的 concept vector
missing        -> VideoUnavailableModel  status() 报错，encode/decode 返回空
```

`VideoLocalOnnxModel` 在 x86_64 上通过 `tools/local_onnx_runner.py` 调用 ONNX Runtime；`VideoHbdnnModel` 在 RDK X5 上调用 `rdk_x5_bpu::execute`。不再有确定性统计 fallback。

### 3.3 JEPA 预测流程

```text
1. 将图像划分为 patch 网格（resolution / patchSize）^2。
2. 随机采样若干目标 patch block 作为 target。
3. encodeContext：编码可见 patch。
4. encodeTarget：编码目标 patch。
5. predictTarget(contextRepr, targetPos)：用 predictor head 预测目标表示。
6. 损失为预测与目标表示的某种距离（MSE / 对比）。
7. adapt：反向更新 predictor/encoder（真实后端实现）。
```

---

## 4. 1D 音频世界模型

实现：`audio_model.{hpp,cpp}`

### 4.1 编码

- 输入：一维音频样本序列（默认 16kHz）。
- 分帧：窗口大小 `frameSamples`， hop `hopSamples`，生成 mel-like 频带能量。
- 时间 patch：将频带能量序列切分为 `temporalPatches`。
- 对每个 patch 计算均值/标准差，拼接后投影到 `targetDim`。

### 4.2 对比预训练

`contrastiveAdapt(audioSamples, textEmbedding, lr)`：

```text
1. audioConcept = encode(audioSamples)
2. alignmentDelta = (audioConcept - textEmbedding) * lr
3. 将 alignmentDelta 持久化到 SpeechConceptModel 的 meanAlignment
4. 保存到 runtime_store/speech_concept_model.json
```

`MixedModalConceptBridge::pretrainSpeech(audio, transcript, targetDim)`：
1. 用 `transformer::TransformerModel` 编码 transcript 得到 textEmbedding。
2. 调用 `speechModel.adapt(audio.payload, sampleRate, mimeType, ...)` 做一次 JEPA 自监督适应。
3. 调用 `speechModel.contrastiveAdapt(audio.payload, sampleRate, mimeType, textEmbedding, ...)` 做语音-文本对比对齐。
4. 用新对齐重新编码 audio 得到 acoustic；更新 `SpeechConceptModel.meanAlignment` 并持久化到 `runtime_store/speech_concept_model.json`。
5. 将 audio 和 text 构建为 `SemanticUnit`（共享 correlationId），并调用 `PersistentConceptMatrix::addOrUpdate(unit, true)` 加入/更新共享 ConceptMatrix。

---

## 5. Transformer 文本编码器

实现：`transformer.hpp`、`external_mixed_modal_io.cpp` 中 `TransformerTextEncoder`。

- 检查点：`runtime_store/transformer_text_encoder.json`。
- JSON 包含 `params`（vocabSize, dModel, nHeads, nLayers, dFF, maxLen, maxTokens, tokenizerMode）和可选 `stateDict`。
- `TransformerModel::encode(tokens)` 返回隐藏状态向量序列。
- 文本编码取最后一层隐藏状态的平均或最后一个 token，投影到目标维度。

---

## 6. 外部混合模态 I/O

实现：`external_mixed_modal_io.{hpp,cpp}`

### 6.1 `MixedModalPacket`

```text
id, modality, payload（字节）, mimeType, source, timestampMs, metadata
```

### 6.2 `MixedModalConceptBridge::encode`

```text
encode(packet, targetDim, contentHint):
    modality = packet.modality
    if text:
        semanticVector = textEncoder.encode(packet.payload 解码为 UTF-8)
    else if image or video:
        variant = metadata["jepaVariant"] or "ijepa_vith14_1k"
        semanticVector = videoEncoder(variant, targetDim).encode(payload)
        if empty: semanticVector = mediaConcept(payload, targetDim, 0x494D4147U)
    else if audio:
        variant = metadata["audioVariant"] or "audio-16k"
        semanticVector = audioEncoder(variant, targetDim).encode(payload)
        if empty: semanticVector = mediaConcept(payload, targetDim, 0x41554449U)
        if SpeechConceptModel.meanAlignment 存在且同维度：
            semanticVector += meanAlignment
            semanticVector = normalize(semanticVector)
    else:
        semanticVector = mediaConcept(payload, targetDim, modalitySeed)
    unit.semanticVector = semanticVector
    result = PersistentConceptMatrix::addOrUpdate(unit, false)
    unit.semanticVector = result.unit.semanticVector  // 可能是最邻近原型修正
    unit.metadata["conceptMatrixResidual"] = result.residual
    unit.metadata["conceptMatrixAction"] = result.action
    返回 unit
```

`mediaConcept` 对未知模态使用基于字节哈希的确定性特征映射。

audio 路径在得到 world-model 向量后叠加 `SpeechConceptModel.meanAlignment` 进行对比对齐修正；随后 `PersistentConceptMatrix` 执行 `addOrUpdate(unit, false)`，根据与已有概念的距离决定新增、更新或返回原型，实现编码时的残差记忆。

### 6.3 `MixedModalConceptBridge::decode`

```text
decode(unit, target, source):
    if target == Text:
        packet.payload = unit.content 的 UTF-8 字节
    else if target == Image or Video:
        packet.payload = videoDecoder.decode(unit.semanticVector, "image/png")
        if empty: 使用 1x1 PNG 占位图并标记 imageDecodeFallback
    else if target == Audio:
        packet.payload = audioDecoder.decode(unit.semanticVector, "audio/pcm")
        if empty: packet.payload = JSON({semanticVector, sourceModality, lengthHint})
    else:
        packet.payload = JSON({semanticVector, sourceModality, content})
        metadata["requiresModalityDecoder"] = true
```

### 6.4 `PersistentConceptMatrix`

实现：`external_mixed_modal_io.cpp` 中的 `PersistentConceptMatrix`。

- 持久化文件：`runtime_store/concept_matrix.json`。
- 最大容量 4096 条；使用余弦相似度与残差阈值（`addThreshold_=0.5`，`learnThreshold_=0.2`）判断新概念或更新原型。
- `addOrUpdate(query, pretrain)`:
  1. 加载持久化 entries。
  2. 在现有条目中寻找最相似原型；计算 `cosineSimilarity(query, prototype)`。
  3. 若相似度低于 `addThreshold`，新增条目；否则按残差对原型做残差校正并增加访问计数。
  4. 若 `pretrain==true`（或推理次数累计到 32 次）则保存到 `runtime_store/concept_matrix.json`。
  5. 返回 `Result{action, residual, refinedUnit}`，其中 `action` 为 `"added"`、`"updated"` 或 `"retrieved"`。
- `status()` 报告 entries 数量、add/learn 阈值、持久化路径、加载状态。
- `reset()` 清空矩阵并删除持久化文件。

### 6.5 `MixedModalConceptBridge::pretrainImage`

```text
pretrainImage(image, caption, targetDim):
    if 不是 image/video 或 payload 空：返回 false
    dim = conceptDimension(targetDim)
    variant = image.metadata["jepaVariant"] or "ijepa_vith14_1k"
    model = videoEncoder(variant, dim)
    model.adapt(image.payload, width, height, mimeType, steps=1, lr=1e-3)
    visual = model.encode(image.payload, width, height, mimeType)
    if visual 空: visual = mediaConcept(image.payload, dim, 0x494D4147U)
    if visual.size != dim: 返回 false
    imageUnit = SemanticUnit(..., visual, conceptEncoder="video-encoder")
    PersistentConceptMatrix::addOrUpdate(imageUnit, true)
    if caption 非空:
        textConcept = transformerTextEncoderConcept(caption, dim)
        if textConcept 非空且维度匹配:
            textUnit = SemanticUnit(..., textConcept, conceptEncoder="transformer-text-encoder")
            textUnit.associationIds.push_back(correlationId)
            PersistentConceptMatrix::addOrUpdate(textUnit, true)
    返回 true
```

### 6.6 `MixedModalConceptBridge::reset`

```text
reset():
    在 gSpeechModelMutex 保护下重置 SpeechConceptModel 为默认值
    删除 runtime_store/speech_concept_model.json
    gConceptMatrix.clear()  // 清空并删除 runtime_store/concept_matrix.json
```

### 6.7 `MixedModalConceptBridge::status`

`status()` 返回一个 JSON，包括：
- `audioEncoder`：persistent、dimension、samples。
- `videoEncoder`：当前图像 world model 的 `status()`。
- `audioDecoder`：当前语音 world model 的 `status()`。
- `textEncoder`：`TransformerTextEncoder` 的加载/错误状态。
- `conceptMatrix`：`PersistentConceptMatrix::status()` 的结果。

---

## 7. 原生感受、野性与趋利避害

### 7.1 `PrimalSensation`

定义：`primal_sensation.hpp`

- `SensationType`：Pain, Pleasure, Hunger, Temperature, Fatigue, Threat, SocialIsolation, Novelty。
- 字段：`type`, `intensity [0,1]`, `valence [-1,1]`, `durationSec`, `source`, `timestampMs`。
- `decay(halfLifeSec, dtSec)`：按指数衰减强度，低于阈值则移除。

`PrimalSensationEngine`：

- `add(s)`：追加感受。
- `decay(halfLife, dt)`：衰减并清理。
- `netValence()`：按强度加权的平均 valence。
- `netArousal()`：最大 intensity。
- `dominant()`：强度最大的感受。

### 7.2 `Instinct`

定义：`instinct.hpp`

- `InstinctType`：Survival, Exploration, Avoidance, Affiliation, Curiosity。
- 字段：`type`, `activation [0,1]`, `benefitWeight`, `harmWeight`, `targetSensation`, `actionBias`。

默认引擎（`InstinctEngine::defaultEngine`）：

```text
Survival    -> Threat           -> protect
Exploration -> Novelty          -> explore
Avoidance   -> Pain             -> avoid
Affiliation -> SocialIsolation  -> connect
Curiosity   -> Novelty          -> investigate
```

### 7.3 感受评估（目标契合度 appraisal）与动态激活

**目标契合度评估**（goal-conduciveness appraisal，Smith & Lazarus 1990）把感受映射为对某野性的利/害贡献：

```text
affinity = typeAffinity(s.type, instinct.type)        // 软亲和度 ∈ [0,1]
drive    = s.intensity * affinity
benefit  = drive * max(0, s.valence) * benefitWeight  // 正效价 -> 利
harm     = drive * max(0, -s.valence) * harmWeight    // 负效价 -> 害
```

`typeAffinity` 由 v7.0 设计的"野性-原生感受绑定表"导出为软亲和度（主绑定=1.0，次绑定=0.3~0.8，无关=0.1 下限），取代原先的硬 0/1 匹配——这样 Survival 对 Pain/Fatigue 也有响应，Avoidance 对 Threat 也有响应。

`InstinctEngine::update(sensations, dtSec)`：

```text
for each instinct i:
    drive_i = Σ_s (benefit_i(s) + harm_i(s))
    decay   = 0.5^(dt / activationDecayHalfLife_)   // 默认半衰期 60s
    current_i = clamp(activation_i * (1 + drive_i) * decay, 0, 1)
```

### 7.4 趋利避害评估（效用形式）

`InstinctEngine::evaluate(sensations)`：

```text
B = Σ_i act_i * Σ_s benefit_i(s)     // 总利
H = Σ_i act_i * Σ_s harm_i(s)        // 总害
denom = B + H + ε
benefitScore = clamp(B / denom, 0, 1)
harmScore    = clamp(H / denom, 0, 1)
netUtility U = clamp((B - H) / denom, -1, 1)

recommendedAction = argmax_a Σ_{i: actionBias_i = a} act_i * (benefit_i - harm_i)
```

**定理 7.1（有界性）**：U ∈ [-1, 1]。

证明：|B − H| ≤ B + H < B + H + ε，故 |U| = |B−H|/(B+H+ε) < 1。∎

**定理 7.2（单调性）**：U 对 B 单调不减、对 H 单调不增。

证明：∂U/∂B = 2H/(B+H+ε)² ≥ 0；∂U/∂H = −2B/(B+H+ε)² ≤ 0。∎

**定理 7.3（软最大化 / 温度）**：动作选择的 softmax 策略 p(a) ∝ exp(u_a/T) 在 T→0⁺ 时收敛到 argmax（贪心），在 T→∞ 时收敛到均匀分布（最大熵，探索）。

证明：softmax 是严格单调变换，故 argmax_a p(a) = argmax_a u_a 对任意 T 成立；当 T→∞ 时 p_a → 1/K（K 为动作数），熵 H(p) = −Σ p_a ln p_a → ln K 为最大熵。∎（当前实现取确定性 argmax，temperature 保留给未来随机采样。）

### 7.5 8 维驱动向量（规范映射）

`BenefitHarmResult.driveVector` 现在由 `emotion::fromAppraisal(benefitScore, harmScore)` 产生（见 §15.1），取代原先的固定 8×5 矩阵。二者都落在 [-1,1]，且与情感模块共享同一规范函数，保证代码/文档/工作流一致。

---

## 8. Prompt 双分架构

实现：`prompt_split.{hpp,cpp}`

### 8.1 `SystemPrompt`

不可变：identity、version、constraints、coreDirective。

默认 `SystemPrompt::arthurDefault`：

- identity: "You are Phoenix, an autonomous cognitive assistant codenamed Arthur."
- version: "Phoenix v7.0 Arthur"
- constraints: 诚实、安全、隐私等硬规则。
- coreDirective: 协助用户、从上下文学习、保护系统、平衡探索与回避。

### 8.2 `MemoryPrompt`

动态：summary、relevantFacts、activeGoals、emotionalTone、benefitHarmBias。

v7.0 要求：`benefitHarmBias` 不再硬编码为 "approach/avoid/wait" 等词语，而是使用 `BenefitHarmResult.driveVector` 的数值权值。

### 8.3 `PromptComposer::compose`

```text
identity
Constraints: ...
Mission: ...
---
[Memory] summary
Relevant facts:
- ...
Active goals:
- ...
Tone: ...
Directive: <driveVector 的 JSON 字符串或数值权值>
---
User: userPrompt
```

`composeMessages` 返回 OpenAI 风格 chat messages 数组。

### 8.4 从上下文/语义记忆生成记忆

- `fromContext(contextEntries, maxFacts)`：取最近条目内容作为 relevantFacts。
- `fromSemanticMemory(semanticMemory, query, topK)`：按余弦相似度检索语义单元。

---

## 9. 自主决策管理器

实现：`autonomy_stack.{hpp,cpp}` 的 `autonomy::CognitionAutonomyManager`。

### 9.1 组成

- `PrimalSensationEngine sensationEngine_`
- `InstinctEngine instinctEngine_`（默认引擎）
- `PromptComposer promptComposer_`
- `MixedModalInputBuffer inputBuffer_`
- `MixedModalOutputQueue outputQueue_`
- `MixedModalChannelRegistry channelRegistry_`
- sessions_：每个 session 的 JSON 状态。

### 9.2 `iterate()` 主循环

```text
iterate(payload, worldState):
    if not enabled: return status()
    确定 targetSessions
    for each session:
        收集 actions / subgoals / simulationRequests / reflectionTasks
        对 heads 排序（cognitionHeadScore = 0.65*weight + 0.35*priority/100）
        生成 scheduledHeads
        生成 worldEvidence

    // v7.0 本能/趋利避害更新
    dtSec = nowMs() - lastIterAtMs_（至少 1s）
    instinctEngine_.update(sensationEngine_.active(), dtSec)
    bh = instinctEngine_.evaluate(sensationEngine_.active())
    lastBenefitHarmBias_ = json(bh.driveVector).dump()

    // 更新记忆 prompt
    memory.benefitHarmBias = driveVector JSON
    memory.summary = "Benefit=... Harm=... Net=..."
    promptComposer_.setMemory(memory)

    // 生成最终 prompt
    if payload 包含 userPrompt:
        composedPrompt = promptComposer_.compose(userPrompt, true)

    return {iteration, sessions, scheduledHeads, worldEvidence,
            runtimeFeaturePatch, benefitHarm, composedPrompt}
```

### 9.3 `observe()` 观察接口

- 接收 `enabled`、`backgroundEnabled`、阈值等配置。
- 若 payload 含 `sensation`，调用 `sensationEngine_.add`。
- 若 payload 含 `mixedModalPacket`，反序列化并 `inputBuffer_.push`，同时 `channelRegistry_.registerSource`。

### 9.4 混合模态 API

- `ingestMixedModalPacket(payload)`：`MixedModalPacket::fromJson` -> `MixedModalConceptBridge::encode` -> `inputBuffer_.push`。
- `pretrainSpeechConcept(payload)`：音频包 + transcript -> `MixedModalConceptBridge::pretrainSpeech`。
- `emitMixedModalOutput(payload)`：`SemanticUnit::fromJson` -> `MixedModalConceptBridge::decode` -> `outputQueue_.push`。
- `drainMixedModalOutputs(payload)`：`outputQueue_.drain(max)` -> JSON 数组。

---

## 10. 现代上下文系统

实现：`modern_context_system.hpp`

- `ContextEntry`：content、role、timestamp、turnNumber、importance、embedding、semanticUnits。
- `ContextWindowConfig`：maxTokens、reservedSystemTokens、importanceThreshold、strategy、enableSemanticSearch、enableAttentionSink、enableHierarchical、enableMultimodal、multimodalEmbeddingDim、semanticChunkSize、similarityThreshold。
- `ModernContextManager`：addEntry、addSemanticUnit、getContext、semanticSearch、pruneContext。
- `AttentionSinkManager`：保留少量低重要性 token 作为 attention sink，辅助长上下文。

上下文修剪策略包括滑动窗口、attention sink、层级、语义分块、混合。

---

## 11. 关键持久化文件

| 文件 | 用途 |
|---|---|
| `runtime_store/speech_concept_model.json` | 语音概念模型 meanAlignment、维度、样本数 |
| `runtime_store/transformer_text_encoder.json` | 文本编码器参数与 stateDict |
| `runtime_store/models/ijepa/<variant>/model.safetensors` | I-JEPA 图像世界模型权重 |

---

## 12. 数据流总结

```text
外部输入 (HTTP/WebSocket/语音/图像)
    |
    v
frontend_server.cpp 路由处理
    |
    +---> 文本  --> ModernContextManager / TransformerService
    +---> 多模态包 --> CognitionAutonomyManager::ingestMixedModalPacket
                               |
                               v
                    MixedModalConceptBridge::encode
                               |
                               +---> 文本  --> TransformerTextEncoder
                               +---> 图像  --> VideoModel
                               +---> 音频  --> AudioModel
                               +---> 其他  --> mediaConcept
                               |
                               v
                         SemanticUnit
                               |
          +------------------+------------------+
          v                  v                  v
   SemanticMemory   ModernContextManager   InstinctEngine/PrimalSensationEngine
          |                  |                       |
          +------------------+-----------------------+
                             v
                   CognitionAutonomyManager::iterate
                             |
            +----------------+----------------+
            v                v                v
      PromptComposer   BenefitHarmResult   ModernContextManager
            |                |                     |
            +----------------+---------------------+
                             v
                  TransformerService::chat/generate
                             |
                             v
                  MixedModalConceptBridge::decode (可选)
                             |
                             v
                       输出到用户/执行器
```

---

## 13. aheadModule：多模态前置编码与并行摘要算法（v7.0 目标）

> `aheadModule` 是 GNN 之前所有预处理逻辑的**概念名**（详见 `workflow.md` 文件头与 §0），本节给出其内部算法。当前代码中对应 `external_mixed_modal_io`、`video_model`、`audio_model`、`transformer::TransformerTextEncoder`、`TorchTextModels`、`PrimalSensationEngine`/`InstinctEngine` 的组合调用。

### 13.1 输入编码调度

```text
aheadModule.process(rawInputs: {text?, image?, audio?, video?, sensor?}) -> AheadOutput:
    units = []
    if rawInputs.text present:
        // 文本编码器可直接复用 llama3.1 8b 自带的 embedding/encoder 部分，
        // 避免为文本单独训练一个语义空间；仅在需要跨模态对齐时才投影到统一维度。
        textVec = llama3_1_embedding_lookup(tokenizer.encode(rawInputs.text))
        units.append(SemanticUnit(modality=Text, vector=projectToDimension(textVec, D)))
    if rawInputs.image or rawInputs.video present:
        units.append(MixedModalConceptBridge.encode(image/video packet, D))   // VideoModel
    if rawInputs.audio present:
        units.append(MixedModalConceptBridge.encode(audio packet, D))        // AudioModel
    if rawInputs.sensor present:
        units.append(mediaConcept(sensor payload, D, sensorSeed))

    // 情感观测（与编码并行，不阻塞编码结果）
    submit_async(EmotionQueue, task=observe(rawInputs, internalState))

    memorySummary = MemoryModule.summarize(units, priorContext)   // 见 §13.2
    return AheadOutput(units, memorySummary)
```

要点：

- 文本路径优先复用 llama3.1 8b 自带 tokenizer/embedding，只有当需要与图像/音频等其它模态做 `fuseAttention`/`fuseAdd` 融合时，才调用 `projectToDimension` 对齐到统一的跨模态维度 `D`（沿用 §2.2 的投影矩阵机制），这样不需要为文本重新训练单独的编码器。
- 图像/音频编码器（`VideoModel` / `AudioModel`）保持现状：由 aheadModule 内部持有和调度，对上层（GNN、Backend）只暴露 `SemanticUnit`。
- 情感观测被显式建模为**异步子任务**，提交到 §16 的 Emotion 队列，不在编码主路径上同步等待。

### 13.2 记忆模块：RNN/LSTM/Transformer 摘要 + 并行双分支

```text
MemoryModule.summarize(units, priorContext) -> Summary:
    // 三种摘要器按配置择一或级联：RNN/LSTM（TorchTextModels）用于长历史压缩，
    // Transformer（TinyLlama/SummaryModel）用于短程语义摘要
    hiddenState = RNNLSTM.encodeSequence(priorContext.tokens + units.textLike())
    summaryText = TransformerSummary.generate(hiddenState, units, maxTokens)
    summary = Summary(text=summaryText, vector=hiddenState.pooled(), ts=now())

    // 关键：摘要只计算一次，随后 fan-out 到两个独立、只读的下游任务，
    // 二者互不等待（对应 workflow.md §2.1 的 par...and...end）
    submit_async(BackendQueue,  task=deliverToBackend(summary.readOnlyView()))
    submit_async(GnnQueue,      task=deliverToGnnMemeBarrier(summary.readOnlyView()))

    return summary
```

`deliverToBackend` 将摘要拼入 `MemoryPrompt.summary`（复用 `PromptComposer`）；`deliverToGnnMemeBarrier` 将摘要向量作为 `MemeGraph::query` 的检索 key。两个任务在提交时即产生各自的 `summary.readOnlyView()`（不可变引用/深拷贝），从算法层面保证互不阻塞、无共享可变状态。

### 13.3 句子生成模块（TinyLlama 等）输出的双副本分发

```text
SentenceGenerator.onGenerated(text) -> void:
    view = ImmutableTextView(text)   // 只读视图，两分支共享同一份底层缓冲，仅持有权不同
    submit_async(BackendQueue, task=appendToNextPrompt(view))
    submit_async(GnnQueue,     task=memeBarrierScanAndMaybeWriteback(view))
    // 不在此处 join/wait；两个 task 各自独立完成，互不感知对方进度
```

这与 v6.0 现有的“TinyLlama 摘要 -> 拼入 prompt -> 调主 LLM”**串行**链路（见 `workflow.md` §2 步骤 8-9）是不同的目标状态：v7.0 要求生成文本对 Backend（下一轮 prompt）和对 MemeBarrier（安全检查/写回 GNN）是两条并行任务，而不是前者完成后才触发后者。

---

## 14. Backend：llama3.1 8b 的 enc / inference / dec 拆分算法

对应 `workflow.md` §0.4、§15。

### 14.1 编译期：tracked patch 应用

```text
build_backend():
    fetch llama.cpp submodule/vendor source
    apply_patch("patches/llama_server_phoenix_v7.patch")   // 版本化、可追踪，随仓库提交
        // patch 内容示例（目标，非最终 API）：
        //   - 暴露 forward 过程中的逐层 hidden state 回调（emotion 钩子的前置条件）
        //   - 暴露采样前 logits 数组，供 logit-bias 注入
        //   - 暴露可配置的 embedding 输入接口（供 enc 段直接注入语义向量而非仅 token id）
    cmake --build . --target llama-server
    // 产物：patched llama-server 二进制，供 inference 段以子进程/库形式调用
```

### 14.2 运行期三段调用

```text
Backend.run(request):
    // enc
    if request.hasPrecomputedEmbedding:
        embedding = request.embedding                 // 来自 aheadModule 的跨模态对齐向量
    else:
        tokenIds = tokenizer.encode(request.text)
        embedding = enc.lookup(tokenIds)                // token embedding 查表

    // inference（patched llama-server 主体）
    emotionPromptFragment = Emotion.renderPromptModulation()
    hiddenState = inference.forward(embedding, prompt=systemPrompt+memoryPrompt+emotionPromptFragment+userPrompt)

    // dec
    if request.modality == Text:
        logits = dec.lmHead(hiddenState)
        logits = logits + Emotion.logitBias(request)    // 见 §15
        token = sample(logits, temperature=Emotion.temperature(request), top_p=Emotion.topP(request))
        return TextOutput(token)
    else:  // audio / video
        conceptVector = dec_av.projectFromHidden(hiddenState)    // 运行期代码，非 .gguf 权重
        payload = dec_av.decode(conceptVector, targetMimeType)   // 见 video/audio world_model::decode
        return MediaOutput(payload)
```

要点：

- `dec_av`（音视频 `dec`）不依赖 `.gguf`：其权重来自 `runtime_store/models/{ijepa,additive_jepa}/...`，由 `video_model.cpp` / `audio_model.cpp` 的 `decode()` 实现，属于 Phoenix 自己训练/维护的解码器，不与主 LLM 权重一起分发，因此文档与代码都不能假设它随 `.gguf` 一起提交。
- `enc` 支持"预计算 embedding 直接注入"是为了让 aheadModule 产出的跨模态语义向量可以绕过 tokenizer，直接进入 llama 的隐藏空间，减少多模态到文本的重复转换。

---

## 15. 情感对 Backend 的影响算法：Prompt 调制 / Logit Bias / Temperature

对应 `workflow.md` §0.5。三种机制均**不需要**修改或乘以 Backend 的权重矩阵，只作用于 prompt 文本、采样前 logits、采样超参数三个天然可控的接口。

### 15.1 情感表示：PAD 核心与评估派生（规范映射）

8 维 `EmotionTensor` 不再由各模块各自维护魔数矩阵生成，统一由两个规范函数产生（`emotion_system.hpp`）：

**PAD → 8 维**（`padToTensor(V, A, D)`；Mehrabian & Russell 1974；Warriner et al. 2013）：

```text
valence   = V
arousal   = A
dominance = D
trust     = (V + D) / 2
joy       = (V + A) / 2
fear      = (-V - D) / 2
anger     = (-V + D) / 2
surprise  = A * (1 - |D|)
```

**利/害评估 → 8 维**（`fromAppraisal(B, H)`，供 `InstinctEngine::evaluate` 使用）：

```text
U = (B - H) / (B + H + ε)      // 净效用 ∈ [-1,1]
A = 2 * min(B + H, 1) - 1      // 唤醒度
D = U                          // 本能层无独立 coping/blame 信号，支配度 ≡ 效价
return padToTensor(U, A, U)
```

**定理 15.1（有界性）**：`padToTensor` 与 `fromAppraisal` 的 8 维输出均 ∈ [-1,1]。

证明：trust/joy/fear/anger 是 [-1,1] 内两数之和的一半；surprise = A(1−|D|)，因 |A| ≤ 1 且 0 ≤ 1−|D| ≤ 1，故 ∈ [0,1]；其余维度为直接拷贝。∎

**定理 15.2（Lipschitz 连续性）**：`padToTensor` 对 (V,A,D) Lipschitz 连续（系数 ≤ 1），`fromAppraisal` 在 B+H > 0 处连续。故相邻时间步的情感状态不会跳变。∎

### 15.2 情感动力学（稳态回归 / 对手过程）

`EmotionSystem::processMessage` 采用一阶稳态更新（对手过程理论，Solomon & Corbit 1974）：

```text
E' = lerp(E, E_new, 1 - emotionDecayRate)   // 刺激耦合（EMA）
E  = lerp(E', baseline, homeostasisRate)    // 向基线回归
```

对应连续时间 ODE：dE/dt = α(E_new − E) + β(baseline − E)，α = 1−emotionDecayRate，β = homeostasisRate。长期无刺激时 E → baseline（稳态），避免情感漂移。

### 15.3 文本情感分析（VAD 词表）

`RuleBasedEmotionAnalyzer` 使用紧凑 VAD 词表（词 → (valence, arousal, dominance)）并做**全体 token 均值池化**：

```text
(V, A, D) = (1/N) * Σ_{t ∈ tokens} lexicon(t)     // N = 总 token 数，未命中贡献 0
emotion   = padToTensor(V, A, D)
```

均值池化使强度与命中密度成正比、被 N 归一化，消除原关键词法"重叠类别重复计数、单关键词即饱和"的问题。`analyzeAudio` 将 ≥3 维声学特征解释为 (V,A,D)、单维特征解释为唤醒度；`combineEmotions` 为置信度加权平均。

### 15.4 情感状态到三通道参数的映射

```text
EmotionState = {valence, arousal, dominance, trust, joy, fear, anger, surprise}   // 复用现有 driveVector 8 维定义

Emotion.renderPromptModulation(state) -> string:
    // 参考 CTRL 式控制码前缀，将情感状态压缩为简短自然语言标签，拼入 MemoryPrompt
    label = classifyToneLabel(state)          // 例如 valence>0.3 & arousal>0.3 -> "engaged/positive"
    return "[Tone:" + label + " driveVector=" + json(state) + "]"

Emotion.logitBias(state, vocab) -> map<tokenId, float>:
    // 参考 PPLM / DExperts 的解码期引导：为情感相关词簇加/减一个与强度成比例的偏置
    bias = {}
    for group, direction in EMOTION_TOKEN_GROUPS(state):     // 例如 fear 高 -> 谨慎用语词簇 +bias
        strength = clamp(state[direction.dim] * direction.sign, -1, 1)
        for tokenId in group.tokenIds:
            bias[tokenId] = direction.scale * strength
    return bias

Emotion.temperature(state) -> float:
    // arousal 越高，采样越发散；dominance 越高，越倾向确定性表达（略降 temperature）
    base = config.sampler.baseTemperature   // 默认沿用现有采样默认值
    return clamp(base + 0.5 * state.arousal - 0.2 * state.dominance, 0.1, 1.5)

Emotion.topP(state) -> float:
    return clamp(0.9 + 0.05 * state.arousal, 0.5, 1.0)
```

### 15.5 依据说明（避免"发明 API"，标注证据来源）

| 通道 | 依据 | 说明 |
|---|---|---|
| Prompt 调制 | CTRL（Keskar et al., 2019, "CTRL: A Conditional Transformer Language Model for Controllable Generation"） | 用可读的控制标签/前缀条件化生成风格，是目前最常见、零侵入的可控生成方式；已被 `PromptComposer`/`MemoryPrompt` 部分实现（`benefitHarmBias`）。 |
| Logit Bias | PPLM（Dathathri et al., 2020）、DExperts（Liu et al., 2021） | 在采样前对 logits 做外部引导修正，不需要重训或反向传播进主模型权重；llama.cpp/llama-server 原生支持 `logit_bias` 请求参数，可直接复用。 |
| Temperature/Top-p | 标准自回归采样超参数（Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration"） | 温度/核采样对输出的"发散度"有良好实证支持，用唤醒度映射温度是对现有采样接口的复用，零新增推理开销。 |

### 15.6 端到端算法

```text
onGenerateRequest(request):
    state = Emotion.observe(request.rawInput, InternalState.current())
    state = Emotion.applySentimentModulation(state, aheadModule.sentimentFeatures(request))

    promptFragment = Emotion.renderPromptModulation(state)
    request.prompt = compose(systemPrompt, memoryPrompt, promptFragment, userPrompt)

    logitsBiasMap = Emotion.logitBias(state, vocab)
    samplerParams = { temperature: Emotion.temperature(state), top_p: Emotion.topP(state), logit_bias: logitsBiasMap }

    return Backend.run(request, samplerParams)   // 见 §14.2
```

`Emotion.applySentimentModulation` 对应需求中 "sentiment/aheadModule 对情感进行调制"：aheadModule 抽取的语义/情绪线索（如文本情感极性、语调特征）作为额外调制项叠加到情感状态上，而不是替代 `PrimalSensationEngine`/`InstinctEngine` 的既有计算（见 §7）。

---

## 16. 异步任务系统：per-module 优先级队列 + work-stealing 线程池

对应 `workflow.md` §0.6、§16。

### 16.1 数据结构

```text
struct Task {
    ModuleId module;         // AheadModule | Gnn | MemeBarrier | Backend | OutputDecoder | AsyncSidecar
    int priority;            // 数值越小优先级越高；MemeBarrier 默认最高，AsyncSidecar 默认最低
    uint64_t submitTimeMs;
    function<void()> fn;
}

struct ModuleQueue {
    ModuleId owner;
    priority_queue<Task> local;     // 按 (priority, submitTimeMs) 排序的本地队列
    mutex localMutex;               // 仅在跨线程 push/steal 时短暂持锁
}

struct WorkerThread {
    int id;
    ModuleQueue* affinity;          // 优先服务的队列（可为空 = 无固定归属，纯 stealer）
}
```

### 16.2 调度循环

```text
worker_loop(worker):
    while not shuttingDown:
        task = try_pop(worker.affinity)              // 先服务自己归属的队列
        if task is None:
            task = steal_from_any_queue_tail(exclude=worker.affinity)   // 从其他队列"尾部"偷取，
                                                                          // 避免和该队列自己的 pop（从"头部"）竞争同一端
        if task is not None:
            execute(task)
        else:
            exponential_backoff_park(maxMs=cfg.asyncPool.parkMaxMs)      // 短暂休眠，避免忙等
```

### 16.3 提交 API（供各模块调用）

```text
submit_async(queueId, fn, priority=default_priority_of(queueId)):
    task = Task(module=queueId, priority=priority, submitTimeMs=now(), fn=fn)
    ModuleQueue[queueId].push(task)     // O(log n) 入堆
    wake_one_idle_worker()               // 避免所有 worker 都在 park 时任务无人处理
```

### 16.4 默认优先级建议（数值越小越先执行）

| 模块队列 | 默认优先级 | 理由 |
|---|---|---|
| MemeBarrier | 0（最高） | 安全过滤属于关键路径，且执行时间通常很短，应尽快完成，避免不安全内容意外被下游消费。 |
| Backend（enc/inference/dec） | 1 | 用户可感知的主路径延迟来源，需要稳定的调度优先级，但允许被 MemeBarrier 抢占。 |
| GNN | 1 | 与 Backend 同级：图检索通常快于 LLM 推理，实际等待时间由任务本身耗时决定，不需要额外降级。 |
| aheadModule（编码/摘要/情感） | 2 | 属于请求路径的前置步骤，但相对 Backend/GNN 结果的最终呈现不是瓶颈时可适度降级。 |
| 输出/音视频解码 | 2 | 生成结果确定后才触发，允许与前置编码共享中等优先级资源。 |
| 异步旁路（学习/工具/持久化） | 3（最低） | 允许被无限期推迟，只要不造成无界内存增长（见 16.5）。 |

### 16.5 背压与公平性

```text
// 防止低优先级队列（AsyncSidecar）无限堆积
if ModuleQueue[AsyncSidecar].size() > cfg.asyncPool.maxBacklog:
    drop_oldest_or_coalesce(ModuleQueue[AsyncSidecar])   // 例如合并连续的"重新计算摘要"任务

// 防止某个高优先级队列长期饿死其它队列的 steal 机会
every N scheduling rounds:
    force_round_robin_steal_attempt()   // 定期强制尝试从每个队列偷一次，避免饿死
```

### 16.6 与 GNN-GA / RL / ADV 等既有异步学习管线的关系

现有 `ReinforcementLearner`、`AdversarialLearner`、`GNNGALearner`（见 `workflow.md` §8）天然属于 §0.6 图中的“异步旁路（学习/工具/持久化）”队列，只需在 `onDialogCompleted` 等触发点改为 `submit_async(AsyncSidecar, ...)`，即可复用同一套 work-stealing 线程池，无需为学习管线单独维护调度逻辑。


## 17. 效率瓶颈与时间复杂度分析

对应任务"找出当前效率瓶颈并进行时间复杂度分析"。按"调用频率高 × 单次成本高"排序。记 D 为语义向量维度（当前 additive-JEPA 概念为 128 维；LLaVA/Qwen2-Audio unit-query 为 4096 维），N 为存储条目数，K 为 top-K，S 为感受数，I 为野性数（≈5），T 为 token 数，steps 为传播步数。

### 17.1 复杂度总表

| 函数 | 位置 | 时间复杂度 | 瓶颈评级 |
|---|---|---|---|
| `projectToDimension` | `semantic_unit.cpp` | O(D_in × D_out)，投影矩阵首次生成 O(D_in × D_out) | 中 |
| `fuseAttention` | `semantic_unit.cpp` | O(K × D_in × D_out)（每个 unit 投影两次） | **高** |
| `PersistentConceptMatrix::findNearestLocked` | `external_mixed_modal_io.cpp` | O(N × D)（线性扫描，N ≤ 4096） | **高** |
| `SemanticMemory::retrieve` | `semantic_unit.cpp` | O(N × D + N log N) | 中 |
| `ConceptMatrix::propagate` | `concept_matrix.cpp` | O(steps × N × 25 × D) | 中 |
| `ConceptMatrix::topConcepts` | `concept_matrix.cpp` | O(N log N) | 低 |
| `InstinctEngine::evaluate` | `instinct.cpp` | O(I × S) | 低 |
| `PrimalSensationEngine::netValence` | `primal_sensation.cpp` | O(S) | 低 |
| `EmotionVocabWeightTable::computeTokenBias` | `emotion_system.cpp` | O(T × 4) | 低 |
| `cosineSimilarity` | `semantic_unit.cpp` | O(D)（一次点积 + 两次平方和） | 基准 |

### 17.2 主要瓶颈与优化建议

1. **`fuseAttention` 的重复投影（高）**：当前对每个 unit 先投影求相似度，再在加权求和循环里对每个 unit 再投影一次，共约 2K 次矩阵-向量乘 O(K × D_in × D_out)。当 D=4096 时单次投影约 16.7M 乘加，K 个 unit 即成瓶颈。建议：一次投影缓存每个 unit 的结果复用；或统一先投影到目标维度后只做点积，降到 O(K × D_out)。

2. **`PersistentConceptMatrix::findNearestLocked` 线性扫描（高）**：每编码一个 packet 做一次 O(N × D) 全扫描（N 最多 4096；D=128 时约 524K 乘加/次，D=4096 时约 16.7M/次）。建议：引入近似最近邻（HNSW / IVF / 乘积量化），或按维度分桶；短期可用"最近一次查询缓存 + 惰性全量刷新"。

3. **`ConceptMatrix::propagate` 的 5×5 邻域（中）**：每步对每个活跃单元扫 24 个邻域并各算一次余弦，O(steps × N × 25 × D)。建议：空间哈希只对"同位置附近且实际存在"的单元计算，或预计算邻接表。

4. **`projectToDimension` 大矩阵（中）**：D=4096 时单次 16.7M 乘加。建议：高维 unit-query 直接使用其原生维度（已是 llama3.1 8b 隐空间），仅在必须跨模态融合时才投影；或改用结构化随机投影（Sparse JL / FJLT）把成本降到 O(D log D)。

5. **低复杂度项**：本能/感受/情感词表均为 O(常数 × S 或 T)，不在热路径上；无需优化。

### 17.3 复杂度推导要点

- 余弦相似度 O(D)：dot 与 sum2 各一趟线性扫描。
- 投影 O(D_in × D_out)：稠密矩阵-向量乘；矩阵由 (D_in, D_out, seed) 唯一缓存。
- 线性扫描最近邻 O(N × D)：对 N 个条目各算一次余弦。
- 注意力融合 O(K × D)：softmax 一次 max + 一次 Σexp + 一次归一化，加权重和。
- 传播 O(steps × N × 25 × D)：每个单元 × 每步 × 24 邻域 × 余弦。

### 17.4 与既有性能工作的关系

本表为**复杂度视角的静态分析**，与 `doc/algorithm/performance_optimization_2026.md`（已完成的热点清理清单）互补：前者记录已落地的改动，本文给出剩余的结构性瓶颈（ANN 索引、投影缓存、邻接表）的量化依据与优先级。

### 17.5 已实现优化（本次会话）

按用户提示的"仿 JIT 热点加速 / 查 hash 表 / 事件循环多进程"思路落地，均为**确定性正确**的优化（不改变语义，可安全编译）：

| 优化 | 位置 | 手法 | 复杂度变化 |
|---|---|---|---|
| fuseAttention 投影复用 | semantic_unit.cpp | JIT 式单次物化：每个 unit 的投影+归一化只算一次，聚合循环复用缓存向量 | 投影 2K 次 → K 次（省一半 O(K·D_in·D_out)） |
| SemanticMemory::retrieve top-K | semantic_unit.cpp | std::partial_sort 只排前 K 个 | O(N·D + N log N) → O(N·D + N log K) |
| findNearestLocked 提前终止 | external_mixed_modal_io.cpp | 余弦 ≤ 1，命中近精确原型即 break | 最坏 O(N·D) 不变，常见命中显著缩短 |
| GraphDiffusionSummarizer 收敛早停 | graph_diffusion_summarizer.cpp | PageRank 幂迭代几何收敛，L1 步长 < ε 即停 | 最坏 O(rounds·E) 不变，稀疏图提前终止 |

### 17.6 推荐的结构性优化（待更便宜模型验证后落地）

1. **LSH 近似最近邻 + SQLite 持久化（已落地）**：`PersistentConceptMatrix` 已完成三项重构：
   - **稳定索引**：用 `std::unordered_map<int64_t, Entry>` 替代 `std::vector<Entry>`，`evictOldestLocked` 按稳定 id 删除，避免索引漂移。
   - **SQLite 后端**：用 `runtime_store/concept_matrix.db` 替代原来的 `runtime_store/concept_matrix.json`，表 `concept_matrix(id, unit_json, count, residual, timestamp_ms)` 加 `timestamp_ms` 索引，持久化只写 dirty 行；加载时一次性读入内存，构建 LSH 采用惰性按维度触发。
   - **随机超平面 LSH（Charikar 2002）**：按维度缓存 `L=8` 张 hash 表、每表 `b=16` 位签名，超平面为单位高斯向量。查询时先取 LSH 桶内候选并精确计算余弦，若最佳候选相似度 `< 0.95` 或桶为空，则回退到完整线性扫描，确保 `addOrUpdate` 不会静默选错原型。小集合（`N ≤ 256` 或 `D < 32`）直接走精确扫描。

   **复杂度**：索引构建 `O(N·L·b·D)`（一次性/维度），查询 `O(L·b·D + C·D)`（C 为桶内候选数），最坏仍 `O(N·D)`（由完整扫描兜底）。

   **召回率下界**：对夹角 θ 的两个单位向量，单个超平面碰撞概率 P = 1 − θ/π；L 张表（每表 b 位）的召回率 ≥ 1 − (1 − (1−θ/π)^b)^L。取 L=8、b=16 时，对 θ=π/6 的理论召回 ≥ 1 − (1 − 0.833^16)^8 ≈ 1 − 6×10⁻⁵，近似误差由精确回退兜底。

2. **并行化 + 异步任务（已落地）**：复用项目已有的 `phoenix::v7::AsyncTaskSystem`（per-module 优先级队列 + work-stealing 线程池），把 `fuseAttention` 与 `ConceptMatrix::propagate` 中相互独立的 per-unit 计算提交到后台线程：
   - `fuseAttention`：把 `units` 分块，每块并行完成 `projectToDimension + normalize + cosine相似度`，主线程收集 `weights`/`projected` 后做 softmax 与加权聚合。`units.size() > 8` 才并行，否则顺序执行。
   - `ConceptMatrix::propagate`：把外层单元循环分块，每个工作线程维护私有的 `boosts` map，最后合并到全局 map 并顺序 apply。`units_.size() > 16` 才并行。
   - 提交前调用 `AsyncTaskSystem::global().start()` 保证工作线程已启动；若提交失败（队列满等）自动回退顺序执行，永不破坏语义。

   **复杂度变化**：两次的最坏时间复杂度不变，但 wall-clock 在多核下随块数近似线性加速，I/O 密集的 `ConceptMatrix::propagate` 尤为明显。

3. **高维投影稀疏化（已落地）**：`projectToDimension` 在 `sourceDim * targetDim > 4096` 时自动切换为 Sparse JL（Achlioptas-style）实现，每列固定 `s = max(3, targetDim / 3)` 个非零项，取值为 `±1/√s`，将单次投影成本从 `O(D_in · D_out)` 降到 `O(D_in · s)`；同时暴露 `projectToDimensionSparse` 接口供热路径显式调用。该矩阵在缓存中按 `(sourceDim, targetDim, seed, nonZeros)` 键复用，对 one-hot 输入精确保持单位 L2 范数，对单位随机向量以高概率保持成对距离。相应单元测试覆盖尺寸、范围、范数保持与确定性。小维度仍走原稠密高斯路径以兼容现有测试与快速路径。

## 18. 主动推理 / 模型预测控制（可选闭环，agi.*）

新增可选子系统（默认关闭，config `agi.enabled=false`），把趋利避害扩展为完整的最优决策闭环（`active_inference.hpp` 声明 + `active_inference.cpp` 实现，已入构建）：

- `active_inference.hpp`：期望自由能（EFE）三项（pragmatic/intrinsic/epistemic）+ 前向 latent 模型 `LatentTransitionModel` + 滚动时域规划器 `ActiveInferenceController`（含情景记忆）。
- 已接线：`AutonomyStack` 通过 `configureAgi/agiPlan/ingestAgiTransition` 接入 `ActiveInferenceController`（`iterate()` 在模型有真实转移后覆盖本能 argmax）。`active_inference.cpp` 已加入 `compile.bat` 的 `COMMON_SOURCES`。
- 自我进化闭环（§18.1）：`observeRewarded`（TD(0)，奖励 = bh.netUtility）、`bootstrapPreferences`（一次播种）、`consolidate`（每 K 回合重放）、`explorationMultiplier`（VDBE 自适应探索）。测试方法见 `doc/v7.0/testing_methodology.md`。

完整推导（EFE 分解、MPC↔主动推理等价性证明、前向模型 surprise、复杂度）见 `doc/v7.0/active_inference.md`。

## 19. 潜意识可定制层与 llama.cpp 优化（可选，默认关闭）

- `subconscious_profile.{hpp,cpp}`：全稳态剖面（气质/敏感度/稳态设定点/风险态度/自定义野性表），接 `configureSubconscious`；理论见 `doc/v7.0/subconscious.md`。
- `tools/llama_prune_analyzer.py`：GGUF 幅度剪枝（逐层自适应阈值 + 原始备份 manifest + keep-mask），只看矩阵不看输出。
- `sparse_block_matmul.{hpp,cpp}`：块稀疏矩阵乘法（τ=0 精确、τ>0 带 Frobenius 界）；ggml 集成规格与上游更新核查见 `doc/v7.0/llamacpp_optimization.md`。


## 20. 任务层（Meeseeks 盒，可选，mission.*）

新增可选子系统（默认关闭，config `mission.enabled=false`）：把一个实例包装为**目标域化（goal-scoped）**生命周期——出生绑定单一目标，疼痛随时间线性增长，只有目标被判定完成（`reportMissionOutcome({goalAchieved:true})`）疼痛才停止。这回答了“AI 的角色不必是服务用户”的问题：估值来源从外部反馈移到内部目标完成判定，闭环完全在系统内部。

- **形式化**：疼痛 p(t)=min(g·t, P_max)，g=`mission.painGainPerSec`，P_max=`mission.maxPain`。**定理**：任务在 T 时刻完成时累积总疼痛 P(T)=∫₀ᵀ p(t)dt = gT²/2（未饱和段），且 P(T) 对 T 严格递增——最小化累积疼痛 ⟺ 最小化完成时间（证明见 `doc/v7.0/mission_layer.md` §2）。已有趋利避害/主动推理闭环本来就最小化稳态代价（Pain 的 setpoint=0），因此**无需新优化器**，压力自动内化为“尽快完成”的偏好（pragmatic 项的终止状态先验偏好，Friston）。
- **实现**：`mission_lifecycle.{hpp,cpp}`（状态机 Idle/Running/Completed/Failed + 基因组 `MissionGenome`，互斥量保护）；`autonomy_stack` 接入 `assignMission/missionStatus/reportMissionOutcome/spawnMissionChild`，`iterate()` 在本能评估前把 pressure 注入为 Pain 感受（source="mission-pressure"）并在结果暴露 `mission` 字段；网关 `111_class_gatewayserver.inc` 加载 `mission.*` 配置。已入 `compile.bat` / `compile_gtest.bat` 的 COMMON_SOURCES 与 `tools/build_rdk_x5.sh`。
- **繁殖/遗传/变异**：`spawnChild(parent, rate)` 对可遗传参数（SubconsciousProfile + learningRate）做高斯扰动（clamp），完成时间作适应度 fitness=−T，构成 (1+λ)-ES；子代基因组可注入新实例/新会话。**无独立运行时外层**：外层 = 进程内状态机 + 基因组对象，`iterate()` 内 `pressureNow()` 为 O(1)，无跨层翻译成本。
- **诚实的边界**：时间依赖疼痛不是势函数塑形（Ng, Harada & Russell 1999），γ<1 时与总疼痛 gT²/2 的等价性会被折扣打破；因此压力只走稳态通道、完成判定由显式门控（防走捷径）、仅适用于目标域化实例。完整推导、探索-利用权衡与参考文献见 `doc/v7.0/mission_layer.md`。

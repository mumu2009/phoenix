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
metadata        : 键值对（如 jpeaVariant、jpeaBackend、conceptEncoder 等）
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

## 3. I-JEPA / JPEA-v2 图像世界模型

### 3.1 官方变体

定义在 `jpea_v2_image_world_model.hpp`：

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

由 `jpeaV2ImageExpectedWeightsPath(cfg)` 计算。

### 3.2 接口语义

`JpeaV2ImageWorldModel` 接口：`encode`、`encodeContext`、`encodeTarget`、`predictTarget`、`adapt`、`decode`、`status`。

当前实现 `JpeaV2ImageFallbackModel` 为确定性 fallback：

```text
encode(imageBytes, width, height, mimeType):
    grid = preprocess(imageBytes, width, height, mimeType)  // 224x224 灰度浮点
    return patchStatsToConcept(grid, {})                    // 全 patch 统计

patchStatsToConcept(grid, mask):
    for each patch (px, py):
        if mask 非空且 mask[patchIdx] == false: skip
        mean = average(patch pixels)
        std = sqrt(E[x^2] - mean^2)
        stats.append(mean, std)
    return projectToDimension(stats, targetDim, 0x1DEA)
```

真实后端（PyTorch/HuggingFace）待接入；fallback 确保语义接口可运行、可测试。

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

## 4. 1D JPEA 语音世界模型

实现：`jpea_v2_speech_world_model.{hpp,cpp}`

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
        variant = metadata["jpeaVariant"] or "ijepa_vith14_1k"
        semanticVector = imageWorldModel(variant, targetDim).encode(payload)
        if empty: semanticVector = mediaConcept(payload, targetDim, 0x494D4147U)
    else if audio:
        variant = metadata["jpeaSpeechVariant"] or "jpea_v2_speech_16k"
        semanticVector = speechWorldModel(variant, targetDim).encode(payload)
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
        packet.payload = imageWorldModel.decode(unit.semanticVector, "image/png")
        if empty: 使用 1x1 PNG 占位图并标记 imageDecodeFallback
    else if target == Audio:
        packet.payload = speechWorldModel.decode(unit.semanticVector, "audio/pcm")
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
    variant = image.metadata["jpeaVariant"] or "ijepa_vith14_1k"
    model = imageWorldModel(variant, dim)
    model.adapt(image.payload, width, height, mimeType, steps=1, lr=1e-3)
    visual = model.encode(image.payload, width, height, mimeType)
    if visual 空: visual = mediaConcept(image.payload, dim, 0x494D4147U)
    if visual.size != dim: 返回 false
    imageUnit = SemanticUnit(..., visual, conceptEncoder="jpea-v2-image-world-model")
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
- `speechWorldModel`：persistent、dimension、samples。
- `imageWorldModel`：当前图像 world model 的 `status()`。
- `jpeaSpeechWorldModel`：当前语音 world model 的 `status()`。
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

### 7.3 动态激活更新

`InstinctEngine::update(sensations, dtSec)`：

```text
for each instinct i:
    score = Σ sensationInstinctScore(s, instinct_i)
    decay = 0.5^(dt / activationDecayHalfLife_)    // 默认半衰期 60s
    current_i = activation_i * (1 + score) * decay
    current_i = clamp(current_i, 0, 1)
```

`sensationInstinctScore(s, instinct)`：

```text
typeMatch = (s.type == instinct.targetSensation) ? 1 : 0
valenceMatch = 1 - |s.valence - (benefitWeight - harmWeight)|
return typeMatch * s.intensity * (0.5 * valenceMatch + 0.5)
```

### 7.4 趋利避害评估

`InstinctEngine::evaluate(sensations, temperature)`：

```text
for each instinct i:
    score_i = (Σ sensationScore(s, instinct_i)) * currentActivation_i
    if temperature > 0:
        score_i = (score_i + 1e-6)^(1/temperature)
    benefit += score_i * benefitWeight_i
    harm    += score_i * harmWeight_i
    total   += score_i
    actionScores[actionBias_i] += score_i

result.benefitScore = clamp(benefit / total, 0, 1)
result.harmScore    = clamp(harm / total, 0, 1)
result.netUtility   = clamp((benefit - harm) / total, -1, 1)
result.recommendedAction = argmax(actionScores)
```

### 7.5 8 维驱动向量

`BenefitHarmResult.driveVector` 由固定线性矩阵产生，映射输入 `[benefit, harm, netUtility, |netUtility|, activationNorm]` 到 `[valence, arousal, dominance, trust, joy, fear, anger, surprise]`：

```text
M = [[ 0.00,  0.00,  1.00,  0.00,  0.00],
     [ 0.30,  0.30,  0.00,  0.00,  0.40],
     [ 0.00,  0.00,  0.00,  1.00,  0.00],
     [ 0.70, -0.20,  0.30,  0.00,  0.00],
     [ 0.60, -0.30,  0.40,  0.00,  0.10],
     [-0.20,  0.80,  0.00,  0.00,  0.20],
     [ 0.00,  0.60, -0.60,  0.00,  0.10],
     [ 0.10,  0.10,  0.00,  0.30,  0.50]]

driveVector = clamp(M * input, -1, 1)
```

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
                               +---> 图像  --> JpeaV2ImageWorldModel
                               +---> 音频  --> JpeaV2SpeechWorldModel
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

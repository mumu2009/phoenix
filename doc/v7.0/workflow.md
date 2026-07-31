# Phoenix v7.0 "Arthur" 系统流程图

> 完整架构与数据流总览见 [`dataflow.md`](./dataflow.md)；本文件按 **模块、类、函数** 三种粒度，使用 Mermaid 绘制完整系统流程。节点命名尽量与源码一致。

---

## 1. 模块粒度系统流程

```mermaid
flowchart TD
    subgraph EXT["外部输入"]
        USER[用户/浏览器/传感器]
        WEB[079project_frontend<br/>React UI]
        MIC[麦克风/摄像头/文件]
    end

    subgraph HTTP["HTTP/WebSocket 服务"]
        FS[frontend_server.cpp<br/>Drogon 路由与处理器]
    end

    subgraph CORE["自主认知核心"]
        AUT[autonomy_stack<br/>CognitionAutonomyManager]
        PRI[primal_sensation<br/>PrimalSensationEngine]
        INST[instinct<br/>InstinctEngine]
        PROM[prompt_split<br/>PromptComposer]
    end

    subgraph MM["多模态 I/O 与世界模型"]
        EMMIO[external_mixed_modal_io<br/>MixedModalConceptBridge]
        IMG[jpea_v2_image_world_model<br/>JpeaV2ImageWorldModel]
        SPK[jpea_v2_speech_world_model<br/>JpeaV2SpeechWorldModel]
        TXT[transformer<br/>TransformerTextEncoder/Service]
    end

    subgraph MEM["上下文与记忆"]
        SEMU[semantic_unit<br/>SemanticUnit / SemanticMemory]
        CTX[modern_context_system<br/>ModernContextManager]
    end

    subgraph RUN["运行时与执行"]
        RUNT[runtime_tuned_config<br/>JSON 配置覆盖]
        EDGE[edge_platform<br/>NPU / GPIO 执行]
        OUT[输出层<br/>文本/语音/图像/执行器]
    end

    subgraph PER["持久化"]
        RS[(runtime_store)]
    end

    USER -->|输入消息| WEB
    MIC -->|原始字节| FS
    WEB -->|HTTP/WebSocket 请求| FS

    FS -->|文本路径| CTX
    FS -->|多模态包| AUT
    FS -->|配置加载| RUNT

    AUT -->|ingestSensation| PRI
    AUT -->|update/evaluate| INST
    AUT -->|composePrompt| PROM
    AUT <-->|encode/decode| EMMIO

    EMMIO -->|图像/视频| IMG
    EMMIO -->|音频| SPK
    EMMIO -->|文本| TXT
    EMMIO -->|语义单元| SEMU

    SEMU -->|检索/融合| CTX
    SEMU -->|语义记忆| MEM

    PROM -->|系统+记忆+用户| CTX
    PROM -->|最终 prompt| TXT

    INST -->|BenefitHarmResult<br/>driveVector| PROM
    CTX -->|相关上下文| PROM

    TXT -->|生成结果| FS
    FS -->|需要硬件执行| EDGE
    FS -->|多模态输出| EMMIO
    EMMIO -->|decode| OUT

    IMG -.->|读取权重| RS
    SPK -.->|speech_concept_model.json| RS
    TXT -.->|transformer_text_encoder.json| RS
    CTX -.->|attention sink / chunk config| RS
    RUNT -.->|config/phoenix_tuned.json| RS
```

---

## 2. 类粒度系统流程

```mermaid
flowchart TD
    subgraph UI["前端层"]
        APP[079project_frontend<br/>App.js]
        HANDLER[frontend_server.cpp<br/>HTTP 处理器]
    end

    subgraph AUTONOMY["自主栈"]
        CAM["autonomy::CognitionAutonomyManager"]
        PSE["phoenix::primal::PrimalSensationEngine"]
        PS["phoenix::primal::PrimalSensation"]
        IENG["phoenix::instinct::InstinctEngine"]
        INSTOBJ["phoenix::instinct::Instinct"]
        BHR["phoenix::instinct::BenefitHarmResult"]
    end

    subgraph PROMPT["Prompt 双分"]
        SYS["phoenix::prompt::SystemPrompt"]
        MEMPR["phoenix::prompt::MemoryPrompt"]
        PC["phoenix::prompt::PromptComposer"]
    end

    subgraph MMIO["混合模态桥"]
        MMP["phoenix::io::MixedModalPacket"]
        BRIDGE["phoenix::io::MixedModalConceptBridge"]
        INBUF["phoenix::io::MixedModalInputBuffer"]
        OUTQ["phoenix::io::MixedModalOutputQueue"]
        REG["phoenix::io::MixedModalChannelRegistry"]
    end

    subgraph WM["世界模型"]
        IMGIF["phoenix::io::JpeaV2ImageWorldModel"]
        IMGF["phoenix::io::JpeaV2ImageFallbackModel"]
        SPKIF["phoenix::io::JpeaV2SpeechWorldModel"]
        SPKF["phoenix::io::JpeaV2SpeechFallbackModel"]
    end

    subgraph TXT["文本/Transformer"]
        TS["transformer::TransformerService"]
        TM["transformer::TransformerModel"]
        TOK["transformer::Tokenizer"]
        TTE["phoenix::io::TransformerTextEncoder"]
    end

    subgraph SEMU["语义单元与记忆"]
        SU["phoenix::multimodal::SemanticUnit"]
        SM["phoenix::multimodal::SemanticMemory"]
        CTXM["phoenix::context::ModernContextManager"]
        CTXE["phoenix::context::ContextEntry"]
        ASM["phoenix::context::AttentionSinkManager"]
    end

    subgraph CFG["配置"]
        RTC["phoenix::tuned::loadTunedConfig"]
        PCC["phoenix::Config"]
    end

    APP -->|HTTP 请求| HANDLER
    HANDLER -->|observe/iterate| CAM
    HANDLER -->|ingestMixedModalPacket| CAM

    CAM -->|add| PSE
    PSE -->|active| IENG
    PSE -->|PrimalSensation| PS
    IENG -->|Instinct| INSTOBJ
    IENG -->|BenefitHarmResult| BHR
    BHR -->|driveVector| MEMPR

    CAM -->|setMemory| PC
    SYS -->|system_| PC
    MEMPR -->|memory_| PC
    PC -->|compose/composeMessages| TS

    CAM -->|encode| BRIDGE
    CAM -->|decode| BRIDGE
    CAM -->|pretrainSpeech| BRIDGE
    BRIDGE -->|textEncoder| TTE
    BRIDGE -->|imageWorldModel| IMGIF
    BRIDGE -->|speechWorldModel| SPKIF
    IMGIF -->|fallback| IMGF
    SPKIF -->|fallback| SPKF
    BRIDGE <-->|SemanticUnit| SU

    SU -->|retrieve/fuse| SM
    SU -->|addEntry/addSemanticUnit| CTXM
    CTXM -->|ContextEntry| CTXE
    CTXM -->|AttentionSinkManager| ASM

    TS -->|model_| TM
    TM -->|tokenizer_| TOK
    TTE -->|model_| TM

    RTC -->|覆盖配置| PCC
    PCC -->|RuntimeConfig| TS
    PCC -->|RuntimeConfig| EDGE

    CAM -->|inputBuffer_| INBUF
    CAM -->|outputQueue_| OUTQ
    CAM -->|channelRegistry_| REG

    HANDLER -->|输出响应| APP
    BRIDGE -->|输出包| OUTQ
```

---

## 3. 函数粒度系统流程

```mermaid
flowchart TD
    subgraph IN["用户输入入口"]
        UI["App.js 发送 HTTP"]
        FS_R_Text["frontend_server.cpp<br/>处理 /chat 或 /api/autonomy/observe"]
        FS_R_MM["frontend_server.cpp<br/>处理 /api/autonomy/ingestMixedModalPacket"]
    end

    subgraph CAM_F["CognitionAutonomyManager 函数"]
        CAM_observe["observe(payload, worldState)"]
        CAM_iterate["iterate(payload, worldState)"]
        CAM_ingest["ingestMixedModalPacket(payload)"]
        CAM_pretrain["pretrainSpeechConcept(payload)"]
        CAM_emit["emitMixedModalOutput(payload)"]
        CAM_drain["drainMixedModalOutputs(payload)"]
        CAM_compose["composePrompt(payload)"]
        CAM_eval["evaluateInstincts()"]
        CAM_sens["ingestSensation(payload)"]
    end

    subgraph PRIMAL_F["原生感受函数"]
        PSE_add["PrimalSensationEngine::add(s)"]
        PSE_active["PrimalSensationEngine::active()"]
        PSE_decay["PrimalSensationEngine::decay(halfLife, dt)"]
        PSE_val["PrimalSensationEngine::netValence()"]
        PSE_arous["PrimalSensationEngine::netArousal()"]
        PS_from["PrimalSensation::fromJson(j)"]
    end

    subgraph INST_F["本能函数"]
        IENG_update["InstinctEngine::update(sensations, dtSec)"]
        IENG_eval["InstinctEngine::evaluate(sensations, temperature)"]
        IENG_drive2emo["InstinctEngine::driveToEmotion(result)"]
        IENG_evalAction["InstinctEngine::evaluateAction(action, sensations)"]
    end

    subgraph PROMPT_F["Prompt 函数"]
        SYS_default["SystemPrompt::arthurDefault()"]
        MEM_empty["MemoryPrompt::empty()"]
        PC_compose["PromptComposer::compose(userPrompt, includeMemory)"]
        PC_msgs["PromptComposer::composeMessages(userPrompt, includeMemory)"]
        PC_fromCtx["PromptComposer::fromContext(context, maxFacts)"]
        PC_fromSemMem["PromptComposer::fromSemanticMemory(memory, query, topK)"]
    end

    subgraph BRIDGE_F["混合模态桥函数"]
        BRIDGE_encode["MixedModalConceptBridge::encode(packet, targetDim, hint)"]
        BRIDGE_decode["MixedModalConceptBridge::decode(unit, target, source)"]
        BRIDGE_pretrain["MixedModalConceptBridge::pretrainSpeech(audio, transcript, targetDim)"]
        BRIDGE_status["MixedModalConceptBridge::status()"]
        MMP_toSU["MixedModalPacket::toSemanticUnit(targetDim, hint)"]
        MMP_fromJson["MixedModalPacket::fromJson(j)"]
        MMP_toJson["MixedModalPacket::toJson()"]
    end

    subgraph WM_F["世界模型函数"]
        CREATE_IMG["createJpeaV2ImageWorldModel(variant, targetDim, backend)"]
        IMG_encode["JpeaV2ImageFallbackModel::encode(...)"]
        IMG_decode["JpeaV2ImageFallbackModel::decode(...)"]
        IMG_status["JpeaV2ImageWorldModel::status()"]
        CREATE_SPK["createJpeaV2SpeechWorldModel(variant, targetDim)"]
        SPK_encode["JpeaV2SpeechFallbackModel::encode(...)"]
        SPK_adapt["JpeaV2SpeechFallbackModel::contrastiveAdapt(...)"]
    end

    subgraph TXT_F["文本编码/生成函数"]
        TTE_load["TransformerTextEncoder::load()"]
        TM_encode["TransformerModel::encode(tokens)"]
        TM_gen["TransformerModel::generate(...)"]
        TS_chat["TransformerService::chat(text, graphContext, maxTokens)"]
        TOK_encode["Tokenizer::encode(text)"]
    end

    subgraph SEM_F["语义单元/上下文函数"]
        SU_toJson["SemanticUnit::toJson()"]
        SU_fromJson["SemanticUnit::fromJson(j)"]
        PROJ["projectToDimension(v, targetDim, seed)"]
        FUSE_ATT["fuseAttention(query, units, targetDim)"]
        SM_add["SemanticMemory::addUnit(unit)"]
        SM_ret["SemanticMemory::retrieve(query, topK)"]
        CTX_addU["ModernContextManager::addSemanticUnit(unit, role)"]
        CTX_get["ModernContextManager::getContext(maxTokens)"]
        CTX_search["ModernContextManager::semanticSearch(query, topK)"]
        CTX_prune["ModernContextManager::pruneContext()"]
    end

    subgraph OUT["输出与持久化"]
        WRITE_RSCM["保存 speech_concept_model.json"]
        WRITE_TTE["保存 transformer_text_encoder.json"]
        OUT_HTTP["HTTP 响应回前端"]
        OUT_ACT["执行器 / 语音 / 图像输出"]
    end

    UI --> FS_R_Text
    UI --> FS_R_MM

    FS_R_Text --> CAM_observe
    FS_R_Text --> CAM_iterate
    FS_R_MM --> CAM_ingest

    CAM_observe -->|sensation| CAM_sens
    CAM_observe -->|mixedModalPacket| CAM_ingest
    CAM_sens --> PSE_add
    PSE_add --> PSE_active
    PSE_active --> PSE_decay

    CAM_ingest --> MMP_fromJson
    MMP_fromJson --> MMP_toSU
    MMP_toSU --> BRIDGE_encode

    BRIDGE_encode -->|text| TOK_encode
    TOK_encode --> TM_encode
    BRIDGE_encode -->|image| CREATE_IMG
    CREATE_IMG --> IMG_encode
    BRIDGE_encode -->|audio| CREATE_SPK
    CREATE_SPK --> SPK_encode
    BRIDGE_encode -->|other| PROJ

    IMG_encode --> PROJ
    SPK_encode --> PROJ
    TM_encode --> PROJ

    PROJ --> SU_toJson
    SU_fromJson --> BRIDGE_decode

    BRIDGE_encode --> SM_add
    SM_add --> SM_ret
    SM_ret --> CTX_addU
    CTX_addU --> CTX_get
    CTX_get --> CTX_prune
    CTX_search --> PC_fromCtx
    PC_fromCtx --> MEM_empty

    CAM_iterate -->|dtSec| IENG_update
    IENG_update --> IENG_eval
    IENG_eval --> IENG_drive2emo
    IENG_drive2emo --> PC_compose
    IENG_eval -->|action 评估| IENG_evalAction

    SYS_default --> PC_compose
    MEM_empty --> PC_compose
    PC_compose -->|最终 prompt| TS_chat
    PC_msgs --> TS_chat
    TS_chat --> TM_gen
    TM_gen --> OUT_HTTP

    CAM_pretrain --> BRIDGE_pretrain
    BRIDGE_pretrain -->|文本编码| TM_encode
    BRIDGE_pretrain -->|语音编码| SPK_encode
    BRIDGE_pretrain --> SPK_adapt
    SPK_adapt --> WRITE_RSCM

    TTE_load -->|读取| TM_encode
    TM_encode --> WRITE_TTE

    CAM_emit --> SU_fromJson
    SU_fromJson --> BRIDGE_decode
    BRIDGE_decode -->|image| IMG_decode
    BRIDGE_decode -->|text| OUT_HTTP
    IMG_decode --> OUT_ACT

    CAM_drain -->|packets| OUT_HTTP

    BRIDGE_status --> IMG_status
    BRIDGE_status -->|读取| WRITE_RSCM
    BRIDGE_status -->|读取| WRITE_TTE

    FS_R_Text -->|输出| OUT_HTTP
    FS_R_MM -->|输出| OUT_HTTP
```

---

## 说明

- **模块粒度** 展示系统主要模块与数据流向。
- **类粒度** 展示核心类之间的协作关系。
- **函数粒度** 展示一次典型多模态输入到自主响应的完整调用链。
- 图中 `fallback` 路径表示当前尚未接入真实 PyTorch/HuggingFace 后端时的确定性实现，真实权重下载后可在 `createJpeaV2ImageWorldModel` / `createJpeaV2SpeechWorldModel` 中选择后端。

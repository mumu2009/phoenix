# Phoenix v7.0 "Arthur" 数据流与架构文档

本文件基于 `v7.0.md`、`workflow.md`、`algorithm.md` 以及 Phoenix v7.0 实际源码（`main_hub_parts`、`autonomy_stack.{hpp,cpp}`、`external_mixed_modal_io.{hpp,cpp}`、`semantic_unit.{hpp,cpp}`、`prompt_split.{hpp,cpp}`、`edge_platform.{hpp,cpp}`、`frontend_server.cpp`、`transformer.hpp`、`modern_context_system.hpp`、`rdk_x5_bpu.{hpp,cpp}` 等）绘制完整数据流图。所有图使用 Mermaid 语法，便于后续导入可视化工具或本仓库的 `tools/mermaid_flow_editor/`。

---

## 1. 整体架构（按子系统分层）

```mermaid
flowchart TB
    subgraph EXT["外部层"]
        UI["079project_frontend<br/>React UI / App.js"]
        CLI["CLI / curl / SDK"]
        SENS["传感器 / 摄像头 / 麦克风"]
        FILE["上传文件 / 语料库 robots/"]
    end

    subgraph PROXY["前端代理层 :5081"]
        FS["frontend_server.cpp<br/>Drogon 静态资源 + /api/* 反向代理"]
    end

    subgraph GATE["网关核心层 :5080"]
        GW["GatewayServer<br/>(main_hub_parts/111)"]
        AUTH["JWT / local-token 鉴权"]
        ROUTE["路由注册<br/>/api/chat /api/learn/* ..."]
    end

    subgraph MEME["记忆与图网络"]
        MG["MemeGraph (GNN)"]
        KVM["KVMStore / LMDB"]
        MM["MemeBarrier<br/>TextCNN + RNN/LSTM"]
    end

    subgraph CTX["上下文与语义"]
        MCS["ModernContextManager"]
        ASM["AttentionSinkManager"]
        SU["SemanticUnit / SemanticMemory"]
        PC["PromptComposer<br/>SystemPrompt + MemoryPrompt"]
    end

    subgraph COG["自主认知层 v7.0"]
        CAM["CognitionAutonomyManager"]
        PRI["PrimalSensationEngine"]
        INS["InstinctEngine<br/>趋利避害评估"]
        BR["MixedModalConceptBridge"]
        IB["MixedModalInputBuffer"]
        OQ["MixedModalOutputQueue"]
        REG["MixedModalChannelRegistry"]
    end

    subgraph MOD["世界模型 / 编码器"]
        TXT["TransformerTextEncoder<br/>Tokenizer / TransformerModel"]
        IMG["JepaV2ImageWorldModel<br/>+ Local-ONNX / BPU / Remote / ServerClient"]
        SPK["JepaV2SpeechWorldModel<br/>+ Local-ONNX / BPU / Remote / ServerClient"]
    end

    subgraph INF["推理后端"]
        OLL["Ollama :11434"]
        LCPP["llama.cpp / llama-server :8080"]
        BIT["BitNet :8090"]
        NAT["Native built-in Transformer"]
    end

    subgraph EDGE["边缘执行层"]
        EP["edge_platform::PlatformManager"]
        NPU["模拟 NPU<br/>GPIO / SPI / 权重换页"]
        MCU["MCU target<br/>GD32H759ZMT6 / MIMXRT1021"]
    end

    subgraph STORE["持久化"]
        RS["runtime_store/"]
        SS["snapshots/"]
        LMDB["lmdb/"]
        SQL["ai_store.sqlite"]
        REDIS["Redis :6379"]
    end

    UI -->|HTTP/WebSocket| FS
    CLI -->|HTTP| FS
    SENS -->|原始字节流| FS
    FILE -->|语料| GW

    FS -->|/api/* 转发| GW
    GW -->|鉴权| AUTH
    AUTH -->|路由| ROUTE

    ROUTE -->|/api/chat| MG
    ROUTE -->|多模态包| CAM
    ROUTE -->|/api/learn/*| LL

    MG <-->|meme 存取| KVM
    MG -->|异常扫描| MM
    MG -->|top-k memes + keywords| CTX

    MCS -->|相关上下文| PC
    SU -->|语义检索 / 融合| MCS
    ASM -->|长上下文 sink| MCS

    CAM -->|observe/iterate| PRI
    CAM -->|evaluate| INS
    CAM <-->|encode/decode| BR
    CAM -->|缓冲| IB
    CAM -->|输出| OQ
    CAM -->|通道注册| REG

    BR -->|text| TXT
    BR -->|image| IMG
    BR -->|audio| SPK
    BR <-->|SemanticUnit| SU

    PC -->|系统+记忆+用户 prompt| INF
    MCS -->|context| INF
    MG -->|graphContext| INF

    INF -->|生成结果| GW
    GW -->|需要硬件执行| EP
    EP -->|NPU 调度| NPU
    EP -->|MCU 烧录/控制| MCU
    NPU -.->|权重/拓扑| RS

    GW -->|快照| SS
    KVM -->|持久图数据| LMDB
    MCS -->|context config| RS
    CAM -->|speech_concept_model.json| RS
    TXT -->|transformer_text_encoder.json| RS
    IMG -->|ijepa weights| RS
    REDIS -->|session cache / pubsub| GW
    SQL -->|结构化数据| GW

    subgraph LL["在线学习"]
        RL["ReinforcementLearner"]
        ADV["AdversarialLearner"]
        GNNGA["GNNGALearner"]
    end

    LL -->|更新权重| MG
    LL -->|fine-tuning adapter| INF
```

**说明**：外部请求统一经过 `frontend_server.cpp` 反向代理到达网关 `GatewayServer`；网关负责鉴权、路由、记忆图（MemeGraph）、上下文、自主认知层以及推理后端调度；边缘层负责将部分算子或权重下放到 NPU/MCU 执行；持久化层为所有子系统提供状态保存与恢复能力。

---

## 2. `/api/chat` 请求生命周期

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户/UI
    participant F as frontend_server :5081
    participant G as GatewayServer :5080
    participant A as Auth
    participant M as MemeGraph
    participant C as ModernContextManager
    participant S as SemanticMemory
    participant I as Inference Backend
    participant D as DB / Redis

    U->>F: POST /api/chat {text, imageContext?, vision?}
    F->>G: 反向代理 /api/chat
    G->>A: Bearer token 校验
    A-->>G: ok / 401

    alt 未禁用图
        G->>M: graph_selector_start
        M->>M: 根据输入 token / imageEmbedding 检索 top-k memes
        M->>M: Jaccard alignment 生成 graphContext
        M-->>G: graphContext + graphEmbeddings
    end

    G->>C: injectImageContext / 组装上下文
    C->>S: semanticSearch(SemanticUnit / text)
    S-->>C: 相关语义单元
    C->>C: pruneContext / attention sink / 分块
    C-->>G: contextEntries + prompt

    G->>I: chat(text, graphContext, maxTokens)
    alt provider=ollama
        I->>I: HTTP POST /api/generate
    else provider=llamacpp
        I->>I: HTTP POST /completion
    else provider=bitnet
        I->>I: HTTP POST /completion
    else provider=transformer
        I->>I: in-process generate
    end
    I-->>G: reply JSON

    G->>G: isLikelyGibberishReply / verify
    G->>D: recordRouteMetric + session cache
    G-->>F: response
    F-->>U: HTTP response

    opt 对话完成后
        G->>G: onDialogCompleted
        G->>RL: 触发 RL（turn counter）
        G->>ADV: 触发 ADV
        G->>GNNGA: 触发 GNN-GA
    end
```

**说明**：

1. `frontend_server.cpp` 仅做代理，不处理业务逻辑。
2. 鉴权支持 JWT 与 `local-{user}-{ts}-{seq}` 本地 token。
3. `/api/chat` 默认启用 MemeGraph selector，可通过 `disableGraphSelector` 关闭。
4. 图像上下文通过 `injectImageContext` 注入，可能进入 `jepa_v2_image_world_model` 编码为语义单元。
5. 生成完成后会进行 `isLikelyGibberishReply` 与可选的 verify，并异步触发在线学习。

---

## 3. 多模态真连接数据流

```mermaid
flowchart LR
    subgraph IN["外部输入"]
        TEXT["文本 UTF-8"]
        IMG_IN["图像 / 视频 bytes"]
        AUD["音频 bytes"]
        SNS["传感器 / JSON"]
        VID["视频 bytes"]
    end

    subgraph PACK["MixedModalPacket"]
        P["id, modality, payload,<br/>mimeType, source, timestampMs, metadata"]
    end

    subgraph BRIDGE["MixedModalConceptBridge"]
        ENC["encode(packet, targetDim)"]
        DEC["decode(unit, target)"]
        PRE["pretrainSpeech(audio, transcript)"]
    end

    subgraph ENC_LAYER["模态编码器"]
        TE["TransformerTextEncoder"]
        JW["JepaV2ImageWorldModel<br/>Local-ONNX / BPU / Remote / ServerClient"]
        SW["JepaV2SpeechWorldModel<br/>Local-ONNX / BPU / Remote / ServerClient"]
        MC["mediaConcept<br/>未知模态"]
    end

    subgraph SEM["语义空间"]
        SU["SemanticUnit<br/>semanticVector + metadata"]
        SM["SemanticMemory<br/>retrieve / fuse"]
        FUSE["fuseAdd / fuseMultiply / fuseAttention"]
    end

    subgraph OUT["输出侧"]
        OT["文本输出"]
        OI["图像/视频解码"]
        OO["概念载荷<br/>requiresModalityDecoder"]
    end

    TEXT --> P
    IMG_IN --> P
    AUD --> P
    SNS --> P
    VID --> P

    P --> ENC

    ENC -->|text| TE
    ENC -->|image/video| JW
    ENC -->|audio| SW
    ENC -->|other| MC

    TE -->|projectToDimension| SU
    JW -->|projectToDimension| SU
    SW -->|projectToDimension| SU
    MC -->|projectToDimension| SU

    SU -->|addUnit| SM
    SU -->|fuse| FUSE
    SM -->|retrieve topK| FUSE
    FUSE -->|融合语义向量| SU

    SU --> DEC
    DEC -->|target=Text| OT
    DEC -->|target=Image/Video| OI
    DEC -->|target=Other| OO

    AUD --> PRE
    TEXT -->|transcript embedding| PRE
    PRE -->|alignmentDelta| SM
    SM -.->|持久化| RS[(runtime_store/<br/>speech_concept_model.json)]
```

**说明**：

- `MixedModalPacket` 是统一的外部输入/输出容器；`MixedModalConceptBridge` 负责将不同模态编码到同一语义空间。
- 文本走 `TransformerTextEncoder`；图像/视频优先使用 `JepaV2ImageWorldModel`（真实后端可切换为 RDK X5 BPU 或 PyTorch）；音频使用 `JepaV2SpeechWorldModel`，并可通过 `pretrainSpeech` 与文本对齐。
- 所有语义向量通过 `projectToDimension` 对齐到目标维度，实现不重新训练 LLM 的跨模态融合。
- 解码时，非文本目标输出概念载荷并标记 `requiresModalityDecoder`，由外部解码器实体化。

---

## 4. 自主认知层（Autonomy Stack）内部循环

```mermaid
flowchart TB
    subgraph INPUT["观测输入"]
        OBS["POST /api/autonomy/observe"]
        MMS["POST /api/autonomy/ingestMixedModalPacket"]
        PRE_TRAIN["POST /api/autonomy/pretrainSpeechConcept"]
    end

    subgraph CAM["CognitionAutonomyManager"]
        O["observe(payload, worldState)"]
        I["iterate(payload, worldState)"]
        S["ingestSensation(payload)"]
        E["evaluateInstincts()"]
        CP["composePrompt(payload)"]
        EM["emitMixedModalOutput(payload)"]
        DR["drainMixedModalOutputs(payload)"]
    end

    subgraph AFFECT["情感/本能层"]
        PSE["PrimalSensationEngine<br/>add / decay / netValence"]
        INST["InstinctEngine<br/>update / evaluate / driveToEmotion"]
        BHR["BenefitHarmResult<br/>benefitScore / harmScore / driveVector"]
    end

    subgraph PROMPT["Prompt 双分"]
        SYS["SystemPrompt<br/>arthurDefault (immutable)"]
        MEM["MemoryPrompt<br/>summary / relevantFacts /<br/>activeGoals / benefitHarmBias"]
        PC["PromptComposer<br/>compose / composeMessages"]
    end

    subgraph BRIDGE["多模态桥"]
        ENC2["MixedModalConceptBridge::encode"]
        DEC2["MixedModalConceptBridge::decode"]
    end

    OBS --> O
    MMS --> I
    PRE_TRAIN -->|audio + transcript| ENC2

    O -->|sensation| S
    S -->|add| PSE
    PSE -->|active sensations| INST
    I -->|dtSec| INST
    INST -->|evaluate| BHR
    BHR -->|driveVector JSON| MEM

    MEM --> PC
    SYS --> PC
    PC --> CP
    CP -->|userPrompt + includeMemory| PC
    PC -->|最终 prompt| OUT["TransformerService / LLM"]

    I -->|mixedModalPacket| ENC2
    ENC2 -->|SemanticUnit| SU2["SemanticUnit"]
    SU2 -->|inputBuffer_.push| IB["MixedModalInputBuffer"]

    EM -->|SemanticUnit from JSON| DEC2
    DEC2 -->|packet| OQ["MixedModalOutputQueue"]
    DR --> OQ
    OQ -->|JSON array| RESPONSE["HTTP response"]

    style SYS fill:#e1f5e1
    style MEM fill:#fff2cc
    style BHR fill:#f8cecc
```

**说明**：

- `observe()` 接收外部观测，可包含原生感受（sensation）或多模态包；`iterate()` 是主循环，每次计算时间差 `dtSec` 并更新本能强度。
- `InstinctEngine::update()` 根据感受匹配分和时间半衰期更新当前激活度；`evaluate()` 计算 benefit/harm 并输出 8 维 `driveVector`。
- `MemoryPrompt.benefitHarmBias` 不再硬编码为 `approach/avoid/wait`，而是直接写入 `driveVector` 数值权值，作为下游矩阵（如 logit-bias）的潜在信号。
- `SystemPrompt` 不可变，`MemoryPrompt` 每轮根据上下文和趋利避害结果重建。

---

## 5. 在线学习与数据生命周期

```mermaid
flowchart LR
    subgraph DATA["数据来源"]
        ROBOTS["robots/ 语料"]
        UPLOAD["/api/corpus/ingest"]
        CRAWL["/api/corpus/crawl"]
        DIALOG["对话结果"]
        FEED["/api/transformer/feedback"]
    end

    subgraph STUDY["学习管线"]
        SE["StudyEngine"]
        RL["ReinforcementLearner"]
        ADV["AdversarialLearner"]
        GNNGA["GNNGALearner"]
    end

    subgraph MODEL["模型更新"]
        MG2["MemeGraph 权重/边"]
        TRANS["Transformer / Adapter"]
        LORA["LoRA adapter files"]
    end

    subgraph STORE2["持久化"]
        LMDB2["lmdb/"]
        SNAP["snapshots/"]
        CKPT["checkpoints/"]
    end

    ROBOTS -->|warmup| SE
    UPLOAD --> SE
    CRAWL --> SE
    DIALOG -->|turn counter| RL
    FEED -->|preference pair| RL

    SE -->|batch| RL
    SE -->|attack/defend samples| ADV
    SE -->|graph evolution| GNNGA

    RL -->|update| TRANS
    ADV -->|update| TRANS
    GNNGA -->|evolve| MG2

    MG2 -->|持久图| LMDB2
    TRANS -->|checkpoint| CKPT
    MG2 -->|snapshot| SNAP
    TRANS -->|LoRA| LORA
    LORA -->|llamacpp-lora-files| INF2["llama.cpp"]
```

**说明**：

- 语料库 `robots/` 在启动时可选自动加载；外部文档通过 `/api/corpus/ingest` 或 `/api/corpus/crawl` 进入 `StudyEngine` 队列。
- 每完成一定轮数对话后自动触发 `RL / ADV / GNN-GA`；用户反馈通过 `/api/transformer/feedback` 收集并进入 RLHF replay buffer。
- GNN-GA 演化 MemeGraph 边权重；RL/ADV 更新 Transformer 参数或生成 LoRA adapter。

---

## 6. 边缘部署与 RDK X5 / NPU 数据流

```mermaid
flowchart TB
    subgraph HOST["主机侧"]
        HOST_LLAMA["llama-server :8082<br/>7B GGUF"]
        CONV["convert_ijepa_to_hbdnn.sh<br/>ONNX -> Horizon .bin"]
        CALIB["calibration_capture.py<br/>calibration frames"]
    end

    subgraph BOARD["RDK X5 板端"]
        X5_PHOENIX["phoenix_main (aarch64)"]
        BPU_BRIDGE["rdk_x5_bpu_bridge.py<br/>hobot_dnn HTTP server"]
        CAMERA["V4L2 camera"]
    end

    subgraph HW["自研硬件 / MCU"]
        MCU2["MCU target<br/>MIMXRT1021 / GD32H759"]
        NPU2["模拟 NPU<br/>R-2R DAC + 比较器阵列"]
        EP2["edge_platform::PlatformManager"]
    end

    CONV -->|model.bin| BPU_BRIDGE
    CALIB -->|golden set| BPU_BRIDGE

    HOST_LLAMA -->|HTTP /completion| X5_PHOENIX
    X5_PHOENIX -->|POST /infer| BPU_BRIDGE
    BPU_BRIDGE -->|BPU| CAMERA
    BPU_BRIDGE -->|concept vector| X5_PHOENIX

    X5_PHOENIX -->|需要硬件执行| EP2
    EP2 -->|SPI/GPIO| MCU2
    MCU2 -->|NPU_SAMPLE_SYNC / DAC| NPU2
    NPU2 -->|比较器读回| MCU2
    MCU2 -->|结果| EP2
    EP2 -->|weight virtualization| SD[(SD card weights)]
```

**说明**：

- 当前部署策略：LLM 推理放在主机 `llama-server`，X5 负责 BPU 视觉/编码器模型，C++ 网关通过 HTTP 与本地 Python BPU bridge 通信。
- `rdk_x5_bpu.cpp` 是 HTTP 客户端，`rdk_x5_bpu_bridge.py` 加载 `hobot_dnn` 编译后的 `.bin` 运行推理；任何错误 fail-closed，返回空向量并在 status JSON 中报告错误。
- 自研模拟 NPU 通过 `edge_platform` 控制 MCU（GD32H759/MIMXRT1021）进行 GPIO/SPI 调度、权重换页与算子分发。

---

## 7. 持久化与存储矩阵

```mermaid
flowchart LR
    subgraph RUNTIME["运行时状态"]
        RS1["runtime_store/ai_store.sqlite"]
        RS2["runtime_store/phoenix_tuned.json"]
        RS3["runtime_store/speech_concept_model.json"]
        RS4["runtime_store/transformer_text_encoder.json"]
        RS5["runtime_store/models/ijepa/"]
        RS6["runtime_store/sdcard_weights/"]
    end

    subgraph CACHE["缓存/消息"]
        RED["Redis :6379<br/>session / pubsub / hot matrix"]
    end

    GRAPH["lmdb/<br/>MemeGraph/KVMStore"] -->|图数据| MG3
    SNAP2["snapshots/<br/>crash recovery"] -->|restore| GW2

    MG3["MemeGraph"] -->|读/写| GRAPH
    GW2["GatewayServer"] -->|session| RED
    CAM2["CognitionAutonomyManager"] -->|speech concept| RS3
    TXT2["TransformerTextEncoder"] -->|params/state| RS4
    IMG2["JepaV2ImageWorldModel"] -->|safetensors| RS5
    EP3["edge_platform"] -->|cold weights| RS6

    GW2 -->|structured entities| RS1
    GW2 -->|tuned config| RS2
```

---

## 8. 核心类协作关系

```mermaid
classDiagram
    class GatewayServer {
        +respond()
        +chatWithOllama()
        +chatWithLlamaCpp()
        +chatWithBitNet()
        +onDialogCompleted()
        +recordRouteMetric()
    }

    class CognitionAutonomyManager {
        +observe()
        +iterate()
        +ingestSensation()
        +evaluateInstincts()
        +composePrompt()
        +ingestMixedModalPacket()
        +pretrainSpeechConcept()
        +emitMixedModalOutput()
        +drainMixedModalOutputs()
        -PrimalSensationEngine sensationEngine_
        -InstinctEngine instinctEngine_
        -PromptComposer promptComposer_
        -MixedModalInputBuffer inputBuffer_
        -MixedModalOutputQueue outputQueue_
    }

    class MixedModalConceptBridge {
        +encode(packet, targetDim)
        +decode(unit, target)
        +pretrainSpeech(audio, transcript)
        +status()
    }

    class SemanticUnit {
        +string id
        +Modality modality
        +vector semanticVector
        +string content
        +float confidence
        +uint64_t timestampMs
        +map metadata
    }

    class PromptComposer {
        +compose(userPrompt)
        +composeMessages(userPrompt)
        +fromContext()
        +fromSemanticMemory()
        -SystemPrompt system_
        -MemoryPrompt memory_
    }

    class ModernContextManager {
        +addEntry()
        +addSemanticUnit()
        +getContext()
        +semanticSearch()
        +pruneContext()
    }

    class MemeGraph {
        +query()
        +link()
        +decay()
        +exportGraph()
    }

    class TransformerService {
        +chat()
        +pretrain()
        +jointTrain()
        +verify()
        +saveCheckpoint()
    }

    class PlatformManager {
        +status()
        +planCompute()
        +dispatchCompute()
        +applyPatch()
    }

    GatewayServer --> CognitionAutonomyManager : routes
    GatewayServer --> ModernContextManager : context
    GatewayServer --> MemeGraph : graph selector
    GatewayServer --> TransformerService : inference
    GatewayServer --> PlatformManager : edge dispatch

    CognitionAutonomyManager --> MixedModalConceptBridge : encode/decode
    CognitionAutonomyManager --> PromptComposer : compose
    MixedModalConceptBridge --> SemanticUnit : produces/consumes
    PromptComposer --> ModernContextManager : fromContext
    PromptComposer --> SemanticUnit : fromSemanticMemory
    ModernContextManager --> SemanticUnit : addSemanticUnit
```

---

## 9. 配置加载与启动数据流

```mermaid
flowchart TB
    START["main.cpp 启动"] --> PARSE["解析 CLI / 环境变量"]
    PARSE --> CONFIG["Config 结构体"]
    CONFIG --> TUNED["runtime_tuned_config.hpp<br/>phoenix_tuned.json 覆盖"]
    CONFIG --> ENV["applyFrontendEnvFromConfig"]

    TUNED --> CTX_CFG["ContextWindowConfig"]
    TUNED --> MB_CFG["MemeBarrier 参数"]
    TUNED --> SM_CFG["SummaryModel 参数"]

    CONFIG --> INIT["初始化子系统"]
    INIT --> LMDB["LMDB store"]
    INIT --> REDIS["Redis synchronizer"]
    INIT --> SQLITE["SQLite ai_store"]
    INIT --> MEMEG["MemeGraph"]
    INIT --> MEMEB["MemeBarrier 后台线程"]
    INIT --> CONTROLLER["ControllerPool"]
    INIT --> AUTONOMY["CognitionAutonomyManager"]
    INIT --> FRONTEND["frontend_server :5081"]

    FRONTEND --> ROUTES["Drogon 路由注册 :5080"]
    ROUTES --> READY["服务就绪"]
```

---

## 10. 版本与约定

- 本文件随 Phoenix v7.0 "Arthur" 同步维护。
- 所有节点命名尽量与源码中的类/函数/文件保持一致。
- 虚线箭头（`-.->`）表示持久化、配置读取或可选/低频路径；实线箭头（`-->`）表示主数据流。
- 子系统分组对应实际源码目录与编译单元：`main_hub_parts`、`autonomy_stack`、`external_mixed_modal_io`、`semantic_unit`、`prompt_split`、`modern_context_system`、`edge_platform`、`rdk_x5_bpu`。

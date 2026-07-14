# 双轨类脑架构与 Conscious Compute 协议

## 1. 目标

本轮实现把类脑模型明确分成两条长期路线：

- 应用路线：functional application brain
  - 目标是用较低成本获得更接近人类工作流的感知、注意、评估、动作与记忆协同。
  - 默认用于聊天上下文、世界模型训练和场景仿真。
- 研究路线：structural research brain
  - 目标是尽量把处理链映射到更像人脑的结构分区与通路上。
  - 默认用于思维反推、人机协同计算协议和研究分析。

## 2. 应用脑：functional application brain

functional 脑由 world_model.hpp 中的 buildBrainProfile(functional) 生成，核心字段包括：

- corticalSystems
  - visualCortex：视觉观测、对象与空间线索
  - auditoryCortex：语音/听觉观测与指令线索
  - languageNetwork：语义词元与可 verbalize 的概念痕迹
  - multimodalAssociation：视觉与听觉绑定结果
  - somatosensoryCortex：体感/交互目标与身体图式
  - motorPlanner：动作意图与候选动作
  - episodicMemory：情景记忆回放痕迹
- functionalRuntime
  - perceptionLoop：当前视觉/听觉/多模态绑定缓冲
  - attentionController：显著性目标和当前注意焦点
  - valueSystem：奖励驱动与不确定性压力
  - executiveController：perceive -> attend -> bind -> evaluate -> act -> replay 六阶段循环
  - actionBuffer：动作候选与 selectedAction
  - replayLoop：当前 consolidationTarget

### 应用侧落地点

- /api/chat 两条主 graphContext 路径已经注入 functional 脑 prompt 上下文。
- dialog jointTrain 除 grounded sensory samples 与 virtual scene samples 外，新增 functional brain samples：
  - 下一步动作
  - 当前应固化的记忆痕迹
  - 当前应关注的感知目标

## 3. 研究脑：structural research brain

structural 脑由 buildBrainProfile(structural) 生成，核心字段包括：

- corticalMap
  - occipitalCortex
  - auditoryCortex
  - wernickeArea
  - brocaArea
  - parietalAssociationCortex
  - somatosensoryCortex
  - motorCortex
  - prefrontalCortex
  - hippocampus
  - thalamus
  - basalGanglia
  - cerebellum
- pathways
  - visual-spatial relay
  - auditory-language relay
  - speech planning relay
  - salience gating
  - episodic recall
  - action selection
  - timing correction

### 研究附加输出

- humanThoughtModel
  - sensory_registration
  - salience_gating
  - associative_recall
  - counterfactual_branching
  - action_commitment
  - conscious_report
- consciousComputePlan
  - machinePrecompute
  - humanStages
  - machinePostprocess
  - estimatedSavings

## 4. 人类思维反推与 Conscious Compute

这里的“利用人的意识减少计算量”在实现上被定义为：

- 机器先完成多模态压缩、候选分支生成与排序。
- 结构脑把当前状态反推为较接近人的思维阶段。
- 人类只在少量高价值节点给出显式、主动、短文本判断。
- 机器再把这些判断作为高层先验，而不是替代验证。

这比暴力枚举更像人类的求解方式：

- 先抓显著线索
- 再回忆最像的经验原型
- 再剪枝少数分支
- 最后做 satisficing action selection

### 协议约束

- 参与必须是明确且自愿的。
- 人的输出只作为高层先验，不替代多模态一致性验证。
- 协议目标是减少搜索和 token 消耗，不是读取隐藏心智状态。
- 禁止把人作为长期、隐蔽、不可撤销的数据存储器。

## 5. 集体认知计算网格

为了回应“大量自愿参与者可否替代部分算力”的设想，本轮新增 collective semantic compute 协议。

### 核心思想

- 机器先把高成本任务压缩成可审计的小分片。
- 每个分片不要求精确数值计算，而要求参与者在有限候选语义关系中做选择。
- 机器对这些选择做冗余投票、共识聚合和多模态校验。
- 目标不是复刻精确线性代数，而是在可接受误差范围内获得稀疏、高价值先验。

### 当前实现边界

- 支持矩阵乘法类任务的分片协议描述。
- 支持把“输入词 A,B 近似输出 C”表述成 participant-signature-lookup，也就是寻找更适合该输入对的志愿者，而不是把 C 存到某个人脑中。
- 支持候选关系词、参与者槽位、冗余因子和误差窗口。
- 支持 mnemonic hints，但只允许 ephemeral、可撤销、短会话级提示。
- 明确禁止 persistent human storage，也就是不支持把原始数据长期存入人的记忆中作为正式存储层。

### 协议输出

- collectiveMesh
  - participantCount
  - minimumParticipants
  - shardCount
  - redundancyFactor
  - humanShards
  - mnemonicHints
- routingDirectory
  - storageMode=participant-signature-lookup
  - entries
  - routeQueries
- aggregation
  - redundant semantic voting + machine verification
- ethics
  - voluntaryOnly
  - auditableTasks
  - forbidPersistentHumanStorage
  - forbidSensitivePayloads
  - revocableParticipation

### 为什么不支持长期脑内存储

这里需要区分两种完全不同的“存储”：

- 允许的：participant-signature-lookup
  - 系统保存的是“谁更擅长处理 A,B 这类输入并给出近似于 C 的输出”这一类志愿者能力签名。
  - 本质上这是人路由目录，类似算子调度表，不是脑内容存储。
- 不允许的：persistent human storage
  - 系统试图把原始数据长期压进某个人的记忆，并把该人当作长期存储介质。

原因不是保守，而是工程上和伦理上都不稳：

- 无法可靠审计记忆保持与遗忘过程
- 无法证明数据被真正删除
- 无法保证参与者长期理解其负担和风险
- 容易把“自愿参与”滑向“事实上的不可退出依赖”

因此当前系统只允许短期、可退出、低敏感度的 mnemonic hints。

## 6. 接口

### 读取类脑状态

- GET /world/cognitive?sessionId=...&profile=functional|structural|dual
- GET /world/brain?sessionId=...&profile=functional|structural|dual

### 读取 conscious-compute 协议

- GET /world/conscious-compute?sessionId=...

### 读取集体认知计算协议

- POST /world/collective-compute
  - body: sessionId, computeTask, participantCount, shardCount, redundancyFactor, acceptableRelativeError, allowMnemonicEncoding 等

### 结构脑驱动的仿真

- POST /world/simulate
  - 可带 brainProfile=functional|structural

## 7. 测试

当前已覆盖：

- test/world_model_scene_brain_tests.cpp
  - 双版本 brain profile
  - functional runtime
  - structural simulation mode
  - conscious compute plan
  - collective semantic compute plan
- test/world_model_prompt_tests.cpp
- test/world_model_learning_tests.cpp
- compile.bat

## 8. 当前边界

当前实现是“人脑风格近似系统”，不是生物学完备仿真。

它已经完成：

- 多中枢显式建模
- 双轨功能/结构分化
- 人类思维阶段反推
- 人机协同计算协议化
- 多人自愿语义计算网格化

但仍未覆盖：

- 生物神经元电活动级别仿真
- 长时程突触可塑性微分方程
- 完整皮层柱与白质连接组重建
- 真正安全可证的脑内长期存储

这些属于下一阶段研究扩展，而不是当前 v6.0 可交付边界。
# v10.1 计划（第一轮：无破坏性实验 + v10.0.0 机制核查 + 文献；第二轮实施）

2026-11-07。本文件 v2：补上了对 v10.0.0 现有机制的核查与文献思路。
v10.0.0 已经建立的机制（同事完成，方案直接在其上增强，不重建）：
`noteFieldOpportunity` 的现场训练链（`unitLearner_.learnEpisode` +
`selfMind_.observeUtterance` + `tinyExpert_.trainProjectTree`，`[field-train]`
日志的 expert/acc/params/trainLmax 字段）与自主进化链（`MissionGenome` 的
tokensPerTick 变异 + 子盒 + `adoptEliteLocked` 的最短 workMs 替换 +
`lineage`）。前三个方向的方案都挂在这两条链上。

## 1. 情绪模块增强

**实验（无破坏）**：对项目自己的语料（`build/tmp/*.out.log` +
`runtime_store/mission_workspace/*/deliverable.md`，共 1,728,413 字符）测了
现有 VAD 词典（`emotion_system.hpp` 的 54 词）的覆盖率：

- 词典词在语料中出现 20/46；
- **工程情绪词 29/32 出现但全部不在词典里**：error×69、correct×72、
  done×434、fix×177、lost×35、leak×28、bug×30、success×28、invalid×14、
  blocked×8、crash×9、retry×8、wait×7、slow×5、fast×7、race×1、dead×5、
  missing×6、empty×11、wrong×15、pass×12、failed×19、abort×2、broken×2、
  safe×50、works×17、ok×11、passed×2。

**文献**：VAD 词典无法区分同价情绪（如 alarm 与 urgency），已有做法是
在词典之上加**任务化情绪分类器**（LLM 金融代理论文，
[paper.pdf](https://lope.linguistics.ntu.edu.tw/publications/how-llm-financial-agents-respond-to-stated-affect-depends-on-their-assigned-role/paper.pdf)）。
**方案**：(a) 词典加工程情绪层（29 词，VAD 三元组与现有词的语义对齐，
每个新词进测试断言防漂移）；(b) `surfaceAffect` 的未命中词降级到词干
（去 ed/s/ing）匹配；(c) 任务化层不新开模型：把工程情绪词的 VAD 命中
作为 `fromAppraisal` 的一个加权通道（`emotion.taskAffectWeight` 配置，
默认 0.5），保持 mean-pool 结构；(d) 增强不重建 `evaluateInstinctsFor`
的既有统一管线，只增加输入信息。

## 2. 自护进化模块效能

**机制核查（v10.0.0 已有）**：现场训练链（unitLearner/selfMind/tinyExpert）
与进化链（genome/子盒/adoptElite/lineage）都在；8 代日志显示每代 3 个子代
workMs 52k–74k ms **串行**执行（~160 s/代仅子代）。慢不在选择规则，在
串行 λ 评估与无退火。
**文献**：LLM 进化器的常见加速是并行候选评估 + 变异退火（darwinian-evolver，
[NousResearch/hermes-agent#12719](https://github.com/NousResearch/hermes-agent/pull/12719)）。
**方案**：(a) 同一 tick 的 λ 个子盒 LLM 调用走 `n_parallel` 批量（先测 λ=3
的加速比与质量不变性）；(b) `effectiveMutationRate` 的 0.5–2.0× 范围随代数
退火（`mission.evolution.annealPerGen`，默认 0.15）；(c) 连续两代 workMs
无改进即停该支（早停不改 adoptElite 的规则）；(d) 全部走 `mission.evolution.*`
门，默认关。

## 3. 热点加速的隐藏条件利用（动态）

**实验**：87 条 `[mission-deliberate]` 行：`causalStaticLen` 只有 5 个变体
（1–143），静态前缀**随 session 的上下文变化**，单槽前缀缓存跨变体全 miss；
`recentChars` 48/87 非零。
**文献**：前缀缓存的命中率本质是**保留策略**（hit rate is a retention
policy，[zenodo](https://zenodo.org/records/23519028/files/a-hit-rate-is-a-retention-policy.pdf)），
异构负载需要按工作负载变体统一驱逐（UniCache，
[dlnext.acm.org](https://dlnext.acm.org/doi/pdf/10.1145/3801489.3806861)）。
**方案**：(a) 前缀按变体指纹（staticLen + 情绪张量量化值 + 工具集哈希）
分槽缓存（≤8 槽 LRU）；(b) 指纹运行时动态计算（session 间变化自动换槽）；
(c) 保留策略 = 变体级 LRU + 命中率驱动的槽大小分配（热变体占多槽）；
(d) `[prefix-cache]` 日志与 `stats()` 输出；(e) `research.hotPath.*` 门。

## 4. 模型融合（格式塔/单 base-MoE）—— 只做大规模无破坏实验，不实现

**实验（无破坏）**：qwen coder 1.5B 与 Llama 3.1 8B 的 `/tokenize` 交集：
普通英语 1/9、领域词 0/7、代码句 9/16——tokenizer 不对齐。
**文献**：模型融合已有词汇/结构失配的专门解法（MergeME 的同/异质 MoE
合并，[arxiv 2502.00997](https://arxiv-org.ezproxy.obspm.fr/html/2502.00997v3)；
词汇失配的投影与重对齐综述，[sciencedirect](https://www.sciencedirect.com/science/article/pii/S295016012500049X)）。
**第二轮的大规模实验（可回滚，不实现）**：(a) 三模型两两 tokenize 交集
矩阵 + 项目语料 1000 句的 OOV 覆盖率（选共享 tokenizer 候选）；(b) 词汇
投影层的往返损失（共享 tokenizer → 专家 detokenize → retokenize 的编辑
距离/膨胀比）；(c) 提示级融合（base 生成 + coder 修订放进一个 KV 缓存）
的单进程延迟测量；(d) 全部只在探针与测量脚本里进行，服务器/权重文件
不动，实验后回滚任何临时文件。

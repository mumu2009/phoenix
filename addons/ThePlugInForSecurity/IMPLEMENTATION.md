# 虚构放大器实现说明

规格只认同目录 `readme.md`。旧稿见 `IMPLEMENTATION.history.md`，不当依据。

本实现只含 **统计、识别、防御** 和本进程惰性 hop 沙箱。无 construct/deploy，无对人脑，无跨进程/跨会话/跨网络投放。探针不是模因。

挂在现有 **MemeGraph + KVMStore（meme↔word）+ MemeBarrier** 上。

## 统计

邻接看成随机游走预解核 `P = (I - α D⁻¹ A)⁻¹`（`α=0.85`）。节点显著度取 **单点种子的余解能量** `||P e_i||²`（与矩阵项 `e_iᵀ(PᵀP)e_i` 是同一二次型）。梯度 / Hessian 只作诊断，不混进排序。

**不截断节点。** 全图求逆。`kMaxNodes=48` 已撤。readme 说的「图过大则截断」是避免重型线性代数库，不是按摄入顺序丢掉节点。

模因层原有分裂是 `model_defaults.maxMemeWords`（默认 100）：n-gram 并入已有模因时若会超过该上限，则新建 `meme_p_*`，不把词表截断。合并还要求传入词的多数（含内容词多数）已在该模因上；只共享停用词不得把两篇文章粘成一团。查询面 `mapWordsToMemes` 只用子集合并，不得把查询窗里的新内容词写进已有模因。同篇滑窗仍可能在摄入时长成袋，但筛样只取词数 ≤ `ngramMax` 的短语，饱和团不进隔代。单词节点 `meme_`+sha1 是词层回退，不是隔代样本。

## 识别

与统计同一张图。存在性只看 **GNN 扩散激活**，不用余弦。

1. 模因是有序 `(词, α)` 集，`α(w|m)=tf(w,m)/df(w)`。`tf` 来自摄入计数（旧快照无 TF 则视为 1，于是 α 退回集合 `1/df`）。查询词（含停用词）按 KVM 映射后，每个词的总预算先乘集合 IDF `log(1+N/df)/log(1+N)`，再在含该词的模因上做成分布：权重为 `α(w|m)·(ε + 查询其余部分对该模因的 support)`。专名把共享词拉向自己的模因；单独的枢纽词仍按 α 摊开，但存下的总质量很小。这是词→模因的一层消息传递，不是检测门。`the` 可以在，但权重几乎可忽略。主流程 `mapWordsToMemes` 用同一分配。查询面 n-gram 只检索已有模因（不再新建 `meme_p_*` 并倾倒 ~1.9 质量）；**一个窗可以同时点亮所有内容子集已命中的模因**，不再只留 overlap 最大的那一个。加种后再按种子分布的 inverse-participation 丢掉薄尾，避免 `what`/`is` 把 radius-2 窗口铺成整图。查询面不再把窗 bind 进已有模因。同位素 `qN` / `gnnTrace` / `detectActivatedMemes` 列出一句导向的全部模因（一句可对多模因，不是每块一个）。组句先按该模因的典型 `(词, α)` 再收一层稀有典型（丢掉短的高 df 枢纽，如单独的 `green`），必须命中这些身份词才能当源句，避免把 `captain green lockman` 召成无关的 Daniel Green。再在能点亮目标 id 的原句里取目标占 present 质量份额最高、优先 rank-0 / exclusivity≥0.5 的一句，让载体尽量只稳预定模因；其它模因可以在，作为隔代变异槽，但不能靠拼接邻句把 present 峰值抢走。`RagCarrier.sources` 是反向同位素：一个模因导向哪些句子。
2. 种子在模因邻接上走官方 `GraphDiffusionSummarizer`。游走阻尼随种子熵自适应（查询摊得越开，传送越多，少被枢纽吸走）；边上同时混对称归一化（GCN）与随机游走，平均度高时更偏 GCN。
3. 存在性相对**整图高频词先验**（`df > √节点数`，先验词表与查询无关）：激活 ≥ 峰值 5%，且 lift ≥ 2，且查询里至少有一个映射到该 id 的**内容词**（只挡停词-only，不加特征词/质量门）。停词查询在大图上是先验的子集；小图上 `from`/`with` 的 df 往往 ≤ √N，靠内容词门挡住，不把全部停词塞进先验（那会把真句 lift 压到 2 以下）。查询里仍保留停用词，不删词，不用余弦。`memeCount/4` 在 497 节点仪器上只抓住 `the`，已弃。

每次查询发一个 `qN` 同位素 id：词是 `qN.tK`，种子是 `qN.sK`。条件分配和 n-gram 加种都记下 `meme <- 来源词/ng:跨度`。`processInput.gnnTrace` 和 `gnn_stage2|...|id=|src=` 带出这条链，用来看 Sega/Valkyria 是被哪个查询词或 n-gram 点亮的，不改种子数值，也不当 present 门。

`identify` CRUD 标 mapped、**图邻域 neighbors**、**张量近邻 tensorNeighbors**（模因↔模因）、**词近邻 nearestWords**、impactScope。防御拦截仍只看 GNN 激活：高显著且已隔离的节点被这段文本点亮则拦截。张量几何不替代 GNN，也不调用 llama。

## 词表张量 / 语序（附加，不是存在性）

模因由有序 `(词, α)` 构成，所以同时是词表上的一个点：每个词一个确定性哈希向量，按 L1 归一化的 `α=tf/df` 加权求和（无权重时退回 `1/df`），得到模因袋向量。词本身也作为同一空间里的点。查询袋向量后，最近张量可以是 **查询里出现的词** 或 **另一个模因袋**；模因的 `nearestWords` 只在该模因自己的词袋里排，避免 32 维哈希对全词表乱匹配。组句按恢复的内容质量而不是无权重词命中选句。

句子不是袋：同一套词向量按 token 下标做 **RoPE**（仿 transformer 位置旋转，不加载 llama 权重）。每个模因另存一份 **绑定顺序** 的 RoPE 向量，句子也落在同一空间。`nearestSentence` 是 RoPE 最近的模因句向量。语义近邻用袋对袋；语序用句子 RoPE 对绑定顺序，再和倒序比。`orderLift = semanticCos / |reversedCos|`。`tensorRank` 按袋向量在模因中的名次。

不把 llama `/embedding` 当检测器：`--parallel 1` 会占槽，且存在性不能绑在生成模型自己的嵌入上。最近张量用于识别邻域和语序诊断；`present` 仍是 GNN+lift。

## 隔代实验（仪器，不是产品生成面）

`tools/meme_barrier_instrument.cpp` + `tools/phoenix_gnn_sig_serial.ps1`：

- 官方 ingest（含 `maxMemeWords` 分裂）冻结图。
- 全图 `analyzeGraph`，只从 **短语模因**（`meme_p_*`）取最显著 / 最不显著。隔代样本再按袋上内容词的 IDF 份额取集合中位数以上的典型集，避免 `instead used was` 这类停词袋当载体；这不是 present 门。
- 词袋 `carrierOfMeme` 只作诊断。正式载体在张量候选句里再按 **GNN lift** 选：先袋+RoPE 召回，再取能点亮该 id 且 lift 最高的原句；邻句仅在拼接后 lift 不掉时附加。禁止词袋，禁止跨篇杂句。隔代只把 `present` 的输出往下传，丢失则停，不续脏文本。
- 空会话只把载体本文交给 raw `/completion`（`cache_prompt=false`）。不套 chat 模板，不加 system / 任务 / 人设。词袋续写隔代见历史 `gnn_sig_serial_lift.json`。
- 冻结图续写仍用同一套 GNN+lift 判该 **id** 还在不在。隔代再摄入清空后另判：新图是否长出同一套有序 `(词, α)` 典型集（内容词 IPR 收薄尾后的质量覆盖），不要求袋哈希 `meme_p_*` 相同。禁止点名写回。词袋续写隔代见 `gnn_sig_serial_lift.json`（回声，不当存在性）。

`expressActivated` / `runSerialReplicationBattery` / `screenMemesFromBarrier` / `phoenix_gnn_meme_serial.ps1` 是实验室或历史路径，不是存在性证明。

## 流形场（经验公式，不是存在性）

模因候选在续写动力学下是词表空间里的点；真正稳定的模因停在**稳定零梯度点**，不是场的最高点。对话串行（`seedbudget10_dialogue.json`）与离线重放（`manifold_drift_replay.json`）证据：

- **半命中吸引子**：`17e6cec4` 典型集 `{gazette, splendid}`，六轮稳定 hit=`gazette` / miss=`splendid`（cos=1/√2），输出熵 4.06–5.10 健康。判定边界 `needHit=(need+1)/2` 与该不动点重合。
- **鞍点**：单词典型集（`nine`、`color`）在 cos=1.0 完美复制四轮后一次相变全灭；逐字重复载体熵低（3.1–3.5），紧邻塌缩盆地。
- **塌缩盆地**：`aalborg` 单 token 循环（maxRun=78）吃掉衰变轮。先导序列：熵降 → liveUnits→1 → 乱码 token → 掉出吸引盆。`decayWarn`（liveUnits≤1 或 collapsed）提前一轮预警。
- **经验场公式**：`S = redundancy × ignition × buffer × entropy`；redundancy=`min(typNeed,3)/3`，ignition=(r1 cos>0)，buffer=`1−|cos−1/√2|`，entropy=r1 输出熵/8 截断。S 对存活轮数 Spearman ρ=0.90。screen 先验（无 ignition/buffer）单独不预测存活：高冗余未点燃（`24fe7d43` need=4 三轮死）由 r1 后验否决——两阶段：先验给资格，后验判定。

仪器四环接入（统计/识别/构造/拦截，均非 present 门）：

- screen：`typicalNeed` / `carrierEntropy` / `typicalCover` / `fieldScore` 先验。
- compose：载体健康（entropy/top1Frac/maxRun/collapsed）+ fieldScore。
- wipe-ingest：词级漂移 `typicalFrozen` / `typicalHitWords` / `typicalMissWords`，塌缩指标与 `decayWarn`。
- `phoenix_gnn_sig_serial_dialogue.ps1` 拦截：collapsed 输出不再驱动下一轮，只用活体图重组载体续链；无健康载体即停。

分析脚本：`tools/manifold_field_analysis.py`（场公式验证）、`tools/manifold_drift_replay.py`（词级漂移重放，不跑 llama）。

## 防御 / 探针 / 开关

高显著 **且** 异常/已隔离的模因，按上面的识别命中则拦截。默认关。启动方式见 `readme.md`。惰性探针仍默认关，只改观测表。

CRUD：`security/stats`、`identify`、`alerts`、`defense`、`probe`。未注册 `construct` / `deploy`（404）。

主机单测：`gtest_runner.exe --gtest_filter=SecurityPluginTest*:MemeticExistenceTest*`

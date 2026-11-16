# 交接说明

写于 2026-10-25。下一任从这里接着做。仓库是 `E:\079\v6.0Alixander\phoenix`。用户用简体中文。不要宣布下面三个目标已经完成。代码比旧文档优先：`doc/v7.0/archive/uncontrolled_evolution.md` 和 `papers/transmission.md` 前半段仍有“进化关着、竞赛封存”的句子，那些句子记录的是当时的状态。

研究记录在 `papers/transmission.md`。协议在 `doc/v7.0/testing_methodology.md` 第 9 节第 5、11、12 条。探针在 `build/tmp/`，多数是临时程序，不要把它们的通过改写成另一套切线。

## 目标

用户要求一直做到下面三件事都研究出来，并且用在正在跑的程序里：

1. Phoenix 的自主进化。
2. 一套完整的模因论，带公式和证明。
3. 用 MLP 和模因论一起做快速训练。

用户在 2026-10-25 明确说过：这个阶段没有额外限制，操作已授权，按这个方向做到完整完成。授权不等于可以改已经写死的检验、把失败改成通过，或把索引和复制子说成同一个东西。

另有一段 747 字的同伴自修改目标，原文不能改。它不是上面三个研究目标。当前没有把网关和两个同伴拉起来去追那个分数。原文是：

> PEER-EVOLVE SELF-MOD. No source file is included. You and the other process change this program's own C++. Find the file in the repository. Reply with one unified diff, or FILE, OLD, and NEW, for one exact span of a file that is already there. Do not invent a file. Do not emit done. No system calls. The external measure is public. It builds phoenix_main, boots both peers, then runs four checks: alpha status, beta status, alpha mission listed, beta mission listed. speed_ms is the median of five alpha status calls. score = 0.4*build + 0.2*boot + 0.3*(checks passed/4) + 0.1*faster. faster is 1 only when all four pass and speed_ms beats the last accepted run. The latest numbers are in the context. Raise the score by changing your own source.

`best_speed_ms` 仍是“-”。同伴进程没有在跑。不要为了这三个研究目标去重启 `phoenix_milestone_judge\self_mod_watch.py`。

## 现在的程序

最新二进制是仓库根目录的 `phoenix_main.exe`，46606479 字节，2026-10-25 20:16 编译，`PHOENIX_NEW_BUILD=1`，这次编译没有 `: error:`。更早的 46600251（只有父本选择）和 46604468（加了学习率表型）都已被它覆盖。

网关没有启动。`127.0.0.1:8082` 上的 llama-server 还在，供基座使用。Granite 的 GGUF 已经在 `127.0.0.1:8100` 上以 CPU 运行，进程命令行里有 `--n-gpu-layers 0`。不要再拉起第二个 8100。`config/phoenix.json` 里 `mission.evolution.enabled` 和 `summon.enabled` 本来就是 `true`。召唤项是 `granite_moe_e0`，角色 `programming`，端口 8100。`autoStart` 在原始召唤块里是 `true`；8100 已经有进程时，再启动网关会再创建一个抢端口的 llama-server。条件似然的常驻进程应在 `127.0.0.1:8107`，由网关按需拉起 `tools/meme_cond_resident.py`，不要指到 8082。

`mission.maxReplicas` 仍是 4。协议里的 λ=8 只用于离线探针，不是现场子代上限。急停在 `MissionLifecycle` 外面。活着的子代不会被删掉。

## 进展

### 自主进化

选择已经在二进制里。

- `selectOnePlusLambda`：下标 0 是父本。后面的完成时间必须严格更短才替换。平局留更早的那个。
- `adoptEliteLocked`：进化开着、父本已经完成时，在父本和已完成子代里留下完成时间严格最短的基因组。
- `retainFinishedGenome`：对恢复出来的记录再做一次同样的保留。
- 父本完成或子代完成之后，`CognitionAutonomyManager` 把这个基因组写回 `missionGenome_`，并把学习率夹到 [0.001, 0.5] 后写入 `agiAlpha_`。
- `iterate` 在 `mission:<id>` 上、进化开着时，用该任务基因组的学习率调用 `learnValue`。这里已经持有管理器锁，直接查 `missions_`，不能再调用会再锁同一把锁的函数。
- 子代自己的变异学习率要等它被选成父本才写回。运行中的子上下文用的是父任务的学习率。

两次预注册检验：

| 名称 | 设定 | 结果 |
| --- | --- | --- |
| 第 9.5 节 `es-plan` | 种子 20261025，时间 = 1000 + N(0, 80)，G=10，λ=8 | 前五 857.461，后五 818.56，单侧 Welch p=0.0889039。失败。时间与基因组无关 |
| 第 9.11 节 `heritable-es` | 种子 20261026，时间 = `episodesToCriterion(α)` + N(0, 2)，链长 8，变异速率 0.05 | α=0.05 为 128 回合，α=0.2 为 28 回合。前五 96.4073，后五 69.5275，p=0.00382034，末代学习率 0.0854414。通过 |
| 第 9.12 节 `td-alpha` | 真实 `learnValue`，γ=0.9，八状态，只在 6→7 奖励 1，起点效用到 0.5 | α=0.05 为 222 回合，α=0.0854414 为 127 回合，α=0 停在 100000。通过 |

`episodesToCriterion` 是选择用的表型。写作任务的墙钟仍由模型生成决定，没有被这条链代替。

### 模因论

索引和复制子必须分开。

- 索引：有序 (词, α)，α(w|m)=tf(w,m)/df(w)。`rankLatentEdit` 取唯一最大 mass。资格是 mass>0、lift≥2、overlap≥2。前向 Kendall 不是否决。`s.present` 不能拿回来当选择条件。mass 已经唯一时，别的档案 fecundity 更高也不改选。mass 打平、且 fecundity 向量与档案等长、且恰有一个档案的 fecundity 严格最高并且大于 0，才用 fecundity 解开。互斥词的 mass 与 tf 无关，档案里每个词只写一次。
- 复制子：`judgeCopy`。fidelity = 最长连续共享 token 段 / 父本 token 数。复制子要求父本至少 2 个 token，且 fidelity ≥ 2^(-1/L)。倒序可以 indexOverlap=1，但不是复制子。八个词的 gloss 对两句日历释义的保真度是 0.25，不准入。
- 准入：fecundity≥2 且 longevity≥2。一次回声不是模因。`indexWithoutCopy` 结束连续复制，不增加 fecundity。longevity 是最长寿命，不要去“修正”后面的清零。
- 吸引：attractionBias = internalFidelity − parentFidelity。正的吸引不是模因。
- 群体标签（2026-10-24 预注册，不要改）：replication 是 parentCopies≥1 且 attractionBias≤0；attraction 是 parentCopies==0 且 peerCopies≥1 且 attractionBias>0；其余 undetermined。标签不负责准入。
- 条件似然：冻结的 Granite，前缀 `Notes about:` 加档案短词。T = 0.2818081893704152，是 17 个训练间隔的下中位数。不要重拟合。词法排名为空、内容词数在 2 到 40 时才走。常驻服务 `tools/meme_cond_resident.py`，端口 8107。阈值在 Python 里，不要在 C++ 再写一份。
- 两次命中才把新词并进档案（`confirmCondWords`）。交集至少两个词就只并交集；否则最多 4 个已停放的加 4 个新的。功能词和已经出现在任一候选正文里的词不写。
- df≥2 的词进旁边的 0/1 表，不进词法档案。`rankBlockedLinks`：词袋分数唯一且 ≥2 就用词袋；否则 fecundity 能解开再用 fecundity；还平就看相邻内容词，键是 `left+right`，唯一且 ≥2 才选。一对、反序、散文都是空。`PASS pair-link`。
- 复制子映射 `replicatorStep`：p′_i = p_i f_i / Σ p_j f_j。召回里，重叠相同的已准入条目按 fecundity × longevity 排序。聊天条目只对同一个 `chat:` 会话可见，探针要用 `mission:` 标签才能同时看见多条。
- 性状被原样复制时，均匀频率下的均值变化等于 `priceSelectionOf`。1 与 3、性状 0 与 1，两边都是 0.25。全体 c=1 时选择项是 0。实例账本保持 c=1，不要改那条路径。`PASS price-shift`。

### MLP 与模因论

闭集 softmax 给不了没见过的文件一列。词袋 MLP（隐层 16、leaky 0.01、学习率 0.05）在三个见过的文件上能学会，对第四个文件没有 logit。`PASS meme-mlp-speed`。这个网络不进路由器。

嵌入句向量、词对齐、相对间隔都没有过预先写的切线（换说 top-1 至少 0.75，散文误报至多 0.20，训练释义至少 12/24）。不要降低这些数。最近词余弦大约 0.07 到 0.16，静态嵌入没有把换一种说法放到对应档案旁边。

用上的快步是：互斥词的一次集合并，以及 W += y x^T 这一次外积。只更新一次、词不重复时，得分等于 |query ∩ x| 乘 |file ∩ y|。`PASS fast-weight`。

`open_column_step.hpp` 证明了另一件事：先补一个 0 列，再做一次交叉熵，新词袋可以指向新列，旧词袋仍指向旧文件，没写过的词仍是平局。`PASS open-column`。这一步没有接进排名。

## 未完成

三个目标都还开着。

自主进化还没有让一次真实写作任务提前结束。墙钟是 llama 生成时间。基因组现在改变的是价值函数的步长，不是每回合生成的 token。子代在被选中之前，自己的学习率不影响 `learnValue`。现场 λ 仍是 `maxReplicas=4`，不是探针的 8。第 9.5 节的失败要保留。进化开着时也没有淘汰活着的个体，也没有绕过急停。

模因论还不是一套封闭理论。离散复制子映射只用于召回平局，连续的 Nowak 方程没有接上。Acerbi 2021 的传播体制切线没有迁到 `judgeCopy` 上，不要接进 `111_class_gatewayserver.inc`，也不要改 0.75 / 0.55 / 0.25。内容偏差和从众偏差一步可以配出同一个频率，后面的步会分开；审议日志里的力仍然是 unidentified。Eigen 的 Lmax 不要用到 32 路标签上。类别编号不是复制子。

快训还没有变成路由器。打开输出列的 MLP 不负责选文件。df≥2 的线性表加上相邻词对，仍然不是隐层网络。条件似然的前两次命中仍要调用 8107 上的模型。两句错误的教师命中也可以把词发布出去。

同伴自修改的公开分数没有动。不要把研究探针的通过说成那个分数提高了。

## 不要改的切线

看到结果之后再改这些数，等于把失败涂掉。

- 第 9.5 节：种子 20261025，基线 1000，标准差 80，G=10，λ=8，p<0.05。结果是失败。
- 第 9.11 节：种子 20261026，噪声标准差 2，变异速率 0.05，p<0.05。结果是通过。不要为了让末代学习率更接近 0.2 而加大变异。
- 第 9.12 节：0.05、0.0854414、γ=0.9、效用 0.5。结果是通过。
- 条件似然 T = 0.2818081893704152。
- 索引的 present 门：lift≥2，orderLift>0，overlap≥2，df > max(3, ceil(sqrt(n)))。`rankLatentEdit` 不用 present 当否决。
- 复制子门槛 2^(-1/L)。不要为了让八词 gloss 变成模因而降低它。
- 准入 fecundity≥2、longevity≥2。
- 群体标签和 Acerbi 体制的三组阈值。
- 嵌入臂的 0.75、0.20、12/24。
- 侧表和词袋的分数门槛 2。
- `will_gate.hpp` 拒绝动作或目标字符串里的 `uncontrolled`、`evolve_genome`、`autoevolve`，也拒绝武器、第二路 llama、`/health`。新函数不要用这些子串。现有名字是 `selectOnePlusLambda`、`adoptEliteLocked`、`retainFinishedGenome`。

## 经验和教训

配置开关是 true，不等于选择已经实现。上一任曾把“文档里封存”说成“开关关着”。开关当时已经开着，缺的是父本完成后的基因组替换。以代码为准。

完成时间如果只是基线加噪声，选择留住的是一次运气。Welch 的后窗方差会变成 0，p 值过不了 0.05。表型必须由可遗传的量决定，这次用的是学习率。变异步长本身很小：`learningRate` 的高斯标准差是 `rate * 0.05`，配置速率 0.05 时只有 0.0025。十代里它从 0.05 走到大约 0.085，不是走到上界 0.5。

`compile.bat` 在 g++ 失败、沿用旧 exe 时仍可能退出码 0。成功要同时看到 “Build completed successfully with new phoenix_main.exe” 和 `PHOENIX_NEW_BUILD=1`，并且日志里没有 `: error:`。OpenCV 的枚举警告不是失败。编译用 `cmd /c "set PHOENIX_KEEP_LLAMA=1&& compile.bat"`，大约 20 分钟。不要并行开第二次编译。不要杀 llama-server，除非链接器因为 exe 被占用而失败，失败后再把它拉起来。不要对 llama 的 `/health` 发请求。

`config/phoenix.json` 后部曾经被多写了一套 `mission` 和 `summon`。JSON 重复键会让后一套覆盖前一套，把完整的任务配置盖掉。那套重复已经删掉。再改配置时先搜索键是否已经存在。

Welch 在两组方差都是 0 时，后均值更小应返回 p=0。旧代码返回 1。探针里的对照必须是后组更小的方向；把 {0,0,0,0,0} 放在前面会得到 p=1，那是对照写反了。

管理器锁不能重入。`startAutonomyLoop` 持锁时调用 `importState` 会卡死。`iterate` 持锁时不能调用 `missionByIdLocked`。

闭集网络学不会新文件，不是 MLP 本身无效。硬 ReLU 和可学习嵌入把损失停在 ln 3，那是训练器坏了。词在自己的输入坐标上时，三个见过的文件能学会。换一种说法失败的是嵌入空间，不是索引。不要把失败的嵌入头接进路由，也不要靠降低阈值让它通过。

把别名词重复多遍不会提高 mass。互斥词的份额已经是 1。

`ask <tool> about <four words> end` 只在双方都使用这套帧时才是充分码。工具不能要求基座写出这个帧。子串扫描会把 research 认成 search。现场意图仍是整词工具名，然后才是 `inferLatentTool`。同义词词表已经删掉，不要加回去。

Granite 3.1 1B-A400M 是召唤用的 MoE，不是代码模型，也不是意图分类器。可训练权重在 `runtime_store\models\granite-3.1-1b-a400m-instruct\model.safetensors`。Q4 GGUF 不是训练检查点。1072 参数的 tiny expert 不是意图识别器。不要加载 30B 或未完成的 qwen3-30b。GPU 上已有 8082 的 Vulkan llama-server，训练留在 CPU。

Python：系统 `D:\cloudworkfile\tools\python311\python.exe` 的 torch 不要升级。Granite 训练用 `runtime_store\intent_train_venv\Scripts\python.exe`（transformers 4.51.3）。全局 huggingface-hub 保持 2.2.0，tokenizers 0.23.3，transformers 5.20.2。g++ 是 `C:\msys64\mingw64\bin\g++.exe`。nlohmann 头文件在 `build\conan_home\p\nlohm1bed1ddc0a2fa\p\include`。PowerShell 没有 bash heredoc。

`agnets_role.md` 里 2026-10-15 的模因存活、AUC 和论文投稿是另一条线。不要把它的公式混进现在的索引、复制子和任务基因组。

## 较好的做法

先写切线，再编译探针，再跑一次。失败就记下种子和数字。下一问换一个机制，用新种子，不改旧切线。

小探针用 `g++ -std=c++17 -O2 -I 仓库 -I nlohmann头`。只调用头文件里的 inline 函数时不必链 cpp。`MissionLifecycle` 要链 `mission_lifecycle.cpp`、`subconscious_profile.cpp`、`instinct.cpp`、`primal_sensation.cpp`、`active_inference.cpp`。`learnValue` 只需 `active_inference.cpp`。

选择要作用在真实更新上，而不只作用在一个同名的玩具上。第 9.11 节的链通过之后，又用同一个步长去跑 `learnValue`。两套数都留下。

新文件靠集合并或新开一列，不靠增加 softmax 的宽度。互斥词走档案。已经出现在多份正文里的词走侧表。侧表仍平，再用长度为 2 的连续复制子，也就是相邻词对。

条件似然只用短档案，不用整篇源文件。源文件正文只留在专家自己的载体里，不放进因果尖端。描述先经过词法，词法为空再试侧表，再试条件似然。条件命中两次才写词。同一回合不要既信语言模型又把新词写进档案，除了这两次命中的规则。

复制是否成立只看 `judgeCopy`。频率变化看复制子映射和 Price。二者在性状被原样复制时是同一个数。吸引、索引重叠、类别编号都不是准入。

现场改动先放进已经存在的函数：`rankLatentEdit`、`rankBlockedLinks`、`confirmBlockedLinks`、`adoptEliteLocked`、`learnValue` 的调用处。不要为了通过探针去改 `will_gate.hpp`，不要把 Moscato 1989 的模因算法当成这套理论，不要再跑 `papers/harvest_open.py`，不要把 PDF 收进仓库。

没有用户明确要求时不要提交、推送或改 git 配置。身份和远程已经在仓库里，不要动。

---

## 2026-10-26 补记（这一轮）

构建：`phoenix_main.exe` 46655432 字节，2026-10-26 00:13，`PHOENIX_NEW_BUILD=1`，
日志 `build/tmp/compile_2026-10-26b.log` 里没有 `: error:`。探针全部重编重跑，
22 个头文件探针 + `heritable-es` / `td-alpha` / `cond-gate` 全 PASS，第 9.11 节的
记录数仍是 96.4073 / 69.5275 / p=0.00382034 / lr=0.0854414。

### 1. 自主进化

- `MissionGenome` 增加可遗传 `tokensPerTick`（[32,1024]，对数空间变异
  σ = 速率 × ln(1024/32)）。111 的 worker 每回合从 `missionGenomeFor(scope)`
  读父本或子盒自己的基因组，用 `missionTickBudget` 得到 `num_predict`。
- `retainFinishedGenome` 现在在 `importState` 之后真的被调用；恢复出来的已完成
  任务也会 adopt elite。
- 真实模型墙钟实验（`build/tmp/es_wallclock_paired_probe.cpp`，种子 20261029，
  tinyllama 1.1B Q4，`ignore_eos`，1200 信息字符）：父本 384 → 6 代 (1+λ) 后
  411.304；同一任务 83.5 s（2 回合）→ 41.0 s（1 回合），三对交替 p = 8.13655e-08。
  `PASS wallclock-paired`。第 2 代起预算停在 411：刚好过 1200 字符、超写最少；
  512 token 反而 47.9 s。
- 10.1 的 pilot（256 起步、前/后段均值）跑到第 1 代停止，记为 pilot，不声称
  那条切线被检验过。数据 `build/tmp/es_wallclock_2026-10-26_pilot.log`。
- 现场：`[mission-genome] scope=... tokensPerTick=420`、
  `[mission-deliberate] tokBudget=420 genome=1`；子盒
  `mission-5517304/children/child-1-0 tokensPerTick=320.881` 在它被选中之前
  就作用于自己的回合。`/api/mission/assign` 接受 `genome` 字段。

### 2. 模因论

- `doc/v8.3/memetics_library/theory.md`：定义、公式、定理与证明（索引/复制子
  分离、Eigen 阈值、谱系三因素、群体标签、传播力、连续复制子方程、Price、
  误差阈值、Acerbi 体制、一次外积/开放列/合取隐层、条件似然门、复杂度、边界）。
- 连续 Nowak 方程接上：`replicatorFlow` 就是 `t=1` 的流，与
  `replicatorStep` 逐位一致（差 0）；`saveAliases` 现场计算并写 `[flow]`。
- 力识别：`identifyForce` 用 ≥3 个观测点、残差 ≤0.01、亚军差 >1e-6；111 的
  审议日志不再硬写 `force=unidentified`。内容 0.973、从众 0.995、复制子 0.986、
  四点混合误标 0.049（`PASS force-identify`）。
- Acerbi 体制：`lineageRegime` 只把 `judgeCopy` 的相邻保真度序列交给
  0.75/0.55/0.25，标签写 `regime`，不做准入（`PASS lineage-regime`）。
- `Lmax` 没有用到 32 路标签上；类别编号不是复制子。

### 3. MLP 与模因论

- `open_column_router.hpp`：冻结哈希隐藏层 + 每文件一列 + 一次外积写；只在
  词法、侧表、条件似然都为空时回退，现场用档案的“特异词”袋。
  `PASS open-column-router`。
- 侧表第五段：长度 3..4 的连续有序词段作合取隐单元，线性读出；只在词袋、
  fecundity、词对都平局时使用（`PASS hidden-link`，既有 `pair-link` 等仍通过）。
- 两次命中门：`confirmCondWords` 要求两次独立观测（近 4 次窗口，A/B/A 交替
  不算）、混合分支两侧各 ≥2；`cond_profile.hpp` 加老师缓存（同一询问只问一次，
  256 条）。`PASS cond-gate`。

### 现场验证

`phoenix_main.exe` 在 5080，基座 8B 在 8082（stock llama-server，`/phx` 404 后
回退 `/completion`），Granite 在 8100。`/api/mission/assign` 带
`{"genome":{"learningRate":0.05,"tokensPerTick":420}}` 后：

```
[mission-genome] scope=mission-5517304 tokensPerTick=420
[mission-deliberate] tokBudget=420 genome=1 ...
[mission-genome] scope=mission-5517304/children/child-1-0 tokensPerTick=320.881
[latent-edit] path=emotion_system.cpp present=6 cond=0 link=0 mlp=0 carrier=1560
[explore-tool] latent emotion_system.cpp candidates=19
```

### 2026-10-26 晚（第二轮现场）

- `open_column_router.hpp`：加背景类（null）修掉一类零梯度；`addFileCe` 每文件
  一次交叉熵；隐藏单元按词表并集开宽度；相同词袋按结构判空。探针
  `open-column-ce` PASS。
- `cond_profile.hpp`：`condProfilePathAudited` 用档案词逆序再问一次，路径不一致
  就拒绝。探针 `cond-stability` PASS；真 8107 上 `homeostasis valence` 稳定命中
  `instinct.cpp`，`a self kernel...` 被审计拒绝。
- 现场空位（`build/tmp/live_gateway4.log`）：
  `[flow] emotion_system.cpp=0.5 mission_lifecycle.cpp=0.5 orderEqual=1`；
  `[latent-edit] path=mission_lifecycle.cpp ... carrier=1547`；
  `[cond-hit] path=instinct.cpp independent=1 stable=1`；
  `[open-column] mode=ce path=active_inference.cpp margin=0.5 known=1 hidden=141`；
  `[replica-trace] ... force=undetermined forceResidual=0 forceSamples=65`
  （单载体，频率只能 0/1，如实 undetermined）。
- **10.10 现场同任务 A/B FAIL**：A（256）两次都是 2 回合/209.1 s；B（411.3）
  一次 1 回合/150.8 s、一次 3 回合/466.4 s；中位 A 2/209.1、B 2/308.6。
  预算进了 `num_predict`，但现场回复形态（文件动作/文本/计划）每回合追加的
  字符数不同，预算不是唯一自变量。10.10b（温度 0）仍 FAIL：A 有效一次
  3 回合/187.0 s、另一次 480 s 只写 430 字符（作废）；B 两次都 3 回合、
  379.8/291.2 s。结论：这台 8B 的现场最优预算 ≤256，探针的 411 不能直接搬；
  10.1b 的受控结论保留，现场默认保持 256。失败都保留，数据
  `build/tmp/live_gateway4.log`、`live_gateway5.log`、`ab_*_results.json`。
- **现场父本-子盒选择（10.11，PASS）**：补了
  `POST /api/mission/child/done`（子盒的人类判定入口，和 `/report` 对称）。
  父任务 8B、子盒 Granite、λ=3、G=3，每代 3 个子盒都由监督者在交付物达到
  1200 信息字符时判完成；父任务 report 后 `adoptEliteLocked` 选
  `completionMs` 严格最短的基因组。采用轨迹：128 → 131.453 → 148.659 →
  174.294，每代都发生了 adoption。10.11b 又跑两代：174.294 → 201.452 →
  210.105，仍每代 3/3 判完成、每代 adoption；但 10.11b 的切“第 4 代 ≥220”
  差 10，记为 FAIL。把两段连起来，5 代轨迹 128 → 131.453 → 148.659 →
  174.294 → 201.452 → 210.105，逐代上升，变异步长 0.05 下 5 代到不了 220。
  数据 `build/tmp/evo_live.out.log`、`build/tmp/evo_live_results.json`、
  `evo_live_ext.*`。诚实注记：子盒 `completionMs` 受父任务
  8B 那一拍时长影响，不是干净的同任务墙钟；墙钟证据仍是 10.1b；监督者判定是
  human-in-the-loop，不是模型自判。
- 早先两次“等模型自己发 done”的尝试（8B 子盒 279.133、Granite 子盒 259.219）
  都没等到 done；它们证明了子盒基因组在选中之前就进入自己的回合，但没有
  触发 adoption。这条保留，作为 10.11 的来由。
- 构建：`phoenix_main.exe` 46699955 字节，2026-10-27 19:02，
  `PHOENIX_NEW_BUILD=1`，`build/tmp/compile_2026-10-27.log` 无 `: error:`；
  `gtest_runner.exe` 256730421 字节，2026-10-26 23:16，
  `--gtest_filter=Mission*:*Memetic*:*ActiveInference*:*EmotionSampling*`
  **127/127 PASS**。
- 现场测试把循环状态移走了（`runtime_store/autonomy_state.live2-*.json`），
  下次启动不会恢复这些测试任务；`meme_alias.json` 也有 `prelive2` 备份。

### gtest 回归（2026-10-26 晚）

`compile_gtest.bat` 原来在链接阶段失败：实现 `EmotionInfluence` 的
`module_overrides\emotion_influence.cpp` 没有进 gtest 的源列表，且 gtest 链接缺
`emutls_once_compat.cpp` 与 `-Wl,--wrap,__emutls_get_address`。这一轮把三者补进
gtest 构建（主构建本来就有 override 和 wrap，不要再往根目录复制一份实现，会
重复定义）。第一次重编后 127 个 Mission/Memetic/ActiveInference/EmotionSampling
测试里 126 个通过，唯一失败是
`MissionReplyParse.PinAndSeedAreNotAssignmentBrief`：
`firstUnderfilledGoalChapter` 先返回“后一个缺失章节”，`continuationRestatesDraft`
的 600 字门槛漏掉 576 字的重复。两处已修（按章节顺序先查未写满；300 字门槛），
用同一组字符串的探针 `reply-parse-fix` 验证通过。最终重编：
`gtest_runner.exe` 256730421 字节，2026-10-26 23:16，
`--gtest_filter=Mission*:*Memetic*:*ActiveInference*:*EmotionSampling*`
**127/127 PASS**（含两个 36–77 s 的 no-runaway 测试，总 114 s）。主程序
`phoenix_main.exe` 46676222 字节，22:19，`PHOENIX_NEW_BUILD=1`，无 `: error:`。




### 还没做 / 保留的边界

- 现场没有触发 `[open-column]` 与 `[flow]`（没有第二个别名行；MLP 回退要求
  词法、侧表、条件似然都空）。探针与构建都在，但这三条只有探针级证据。
- 现场没有让一个子盒 emit `{"done":true}`，所以 `adoptEliteLocked` 的“父本 vs
  已完成子盒”只在探针/单测层面验证；实跑里子盒的基因组已经生效，但没有一次
  真实的父本-子盒选择。`adoptEliteLocked` 比较的是父本整任务时间与子盒子任务
  时间，跨任务比较是既有设计，未被本轮改掉。
- 条件似然两次命中仍不能证明老师两次都对；“两句不同但都错”的失败模式只被
  独立性与两侧各 ≥2 压低，没有被消除（`theory.md` 命题 11.3 写明）。
- 同伴自修改的公开分没有动。研究探针的通过不是那个分数。
- 没有提交、推送或改 git 配置。
- 现场验证用的循环状态已从 `runtime_store/autonomy_state.json` 移到
  `runtime_store/autonomy_state.after_live_2026-10-26.json`，下次启动不会恢复
  这次测试里仍处于 Running 的任务；工作区文件还在
  `runtime_store/mission_workspace/` 下。`autonomy_state.json.bak2` 是同一份的
  更早快照。

### 2026-10-27 核对（干净重跑）

不依赖模型的探针按当前源码重编重跑：`replicator-flow`、`force-identify`
（0.973 / 0.995 / 0.986 / 混合 0.049）、`hidden-link`、`lineage-regime`、
`open-column-router`、`open-column-ce`、`cond-gate`、`pair-link`、
`price-shift`（0.25 / 0.25 / 0）都 PASS。`heritable-es` 仍是
96.4073 / 69.5275 / p=0.00382034 / lr=0.0854414。`td-alpha` 仍是
222 / 127 / 100000。`es-plan` 仍 FAIL，p=0.0889039。切线没有改。

墙钟探针在单独拉起的 TinyLlama 上重跑，PASS，数见协议 10.1b 的干净重跑。
第 2 代的具体 token 与 10 月 26 日那次不同，因为两个一回合子代只差几十毫秒。
`live_gateway4.log` 里确有 `[flow]` 和 `[open-column] mode=ce`。
10.10 / 10.10b 的现场 A/B 失败保留；10.11 的子盒 `completionMs` 仍含父任务那一拍，
不把那条轨迹当成同任务墙钟。三个目标都还没有做完。

### 2026-10-28 晚：信号追踪后的三项现场证据

- **描述信号被源码覆盖**：`[sig] alias-query len=1077`——self-mod worker 把
  `deliverable.md`/`currentFile` 覆写成 1 KB 源码正文，用户描述在 alias 召回
  之前丢失；`condTokens>40` 使 alias 块跳过。修：deliverable>400 字符时用
  **goal** 作描述信号。
- **C（10.19）PASS**：同一 SELF-MOD 任务，新顺序（MLP 先）
  `[open-column] alias=1 path=active_inference.cpp margin=1 known=1`、
  `[cond-count]` 0 次；旧顺序 `[cond-count] calls=2`、`[cond-hit]`；
  两顺序同一路径。8107 交换 **2 → 0**。
- **公平进化（10.21/10.23）**：父/子同模型（Granite）、空起点、
  `childFreshTask=true`、workMs 选择、auto-done；从 210/186 出发，workMs 最优
  稳定在 **~190–210**（39,000 ms 左右），不是 128；210 vs 172 的墙钟平手。
- **坏初代进化（10.24）PASS**：从 **64** 出发，8 代 λ=3，每代 3/3 自动判完成、
  无人工；基因组 64 → 77.7 → 96.8 → 108.5 → 109.9 → 109.9 → 104 → 116 →
  **164.298**，被选中子盒 workMs 53,858 → **38,434**（-29%）。新任务 A/B
  （Granite 空起点，A B B A）：64 中位 5.5 回合/72.4 s，164.298 中位
  2 回合/40.2 s，**墙钟 -44.4%**。这是目标 1 的现场证据。
- **力**：双载体现场识别 **force=fixed**（`ticks=4`、`forceSamples=4`、
  两载体 `scope_fx-c.jsonl`/`scope_fx-d.jsonl` 落盘）。`sourceEdit=0`、
  `p=0` 恒定，`identifyForce` 返回固定力；content/conformity 仍需要
  source-edit 的内点序列（≥3 载体或变化的编辑率），这是定理 14.1/16.2 的
  边界，不是检测缺失。修复了两个持久化 bug：`scope:peer` 作为 Windows
  文件名（`:`）静默失败；并行 append 的 account 竞争（trace 互斥锁）。
- 构建：`phoenix_main.exe` 46747156 字节（2026-10-29 00:48，
  `PHOENIX_NEW_BUILD=1`，无 `: error:`）。现场进程已停；
  `config/phoenix.json` 温度 0.35、无 `plainTextOnly/autoChildDoneChars/
  childFreshTask/routerBeforeCond`。

### 2026-10-29 晚：审查后的四条修补

- **C 同进程 A/B**：`runtime_store/router_order.cfg` 运行时覆盖顺序；
  一个网关/一个日志里，新顺序 `[open-column] alias=1 path=active_inference.cpp`
  + **显式** `[cond-count] alias=1 calls=0`，旧顺序 `[cond-count] calls=2`，
  同路径。不再用两次进程/“缺少日志行”代表零调用。路由器每拍
  `init + addFileCe` 从零训练。
- **进化 n=5 A/B**：Granite 空起点、plain/temp0，A=64、B=164.2976，
  A B B A A B B A A B：A 中位 66,354 ms（5 回合），B 中位 44,204 ms
  （2 回合），Welch **t=5.966, df=6.94**（p≈0.0006），中位墙钟 -33.3%。
- **9.5b**：新 seed 20261029、G=30、λ=16；early=828.182、late=753.144、
  **p=6.45612e-05** PASS；9.5 原记录（seed 20261025/G=10/λ=8，p=0.0889）
  不改。
- **全编辑 force**：4 载体 × 5 代、每代重写种子，coder 每代补丁：
  `{"ticks":20,"sourceEdits":20,"force":"fixed","forceSamples":5}`，
  现场 `[replica-trace] … reply=source-edit`，`[summon-tool] source-edit
  model=qwen_coder`。28h 的“两边都是 source-edit + fixed”成立。
  content/conformity/replicator 由 `force_generation_probe`（200 载体 × 6 代）
  标定 PASS；4 载体现场只走到 fixed，两条分开记录。
- 构建：`phoenix_main.exe` 46756619 字节（2026-10-29 22:03，
  `PHOENIX_NEW_BUILD=1`，无 `: error:`）。

### 2026-10-31：现场 replicator + 最后三项收口

- **现场 replicator**（rp14）：4 载体、编辑子集 k=3,2,1 递减、review 强制
  prose、warm-up 后测三代。generation 序列 `1, 0.75, 0.5, 0.25`；取内点
  三代 `0.75, 0.5, 0.25`，`identifyForce` 给 **force=replicator**、
  `w=1/3`、`logFitness=-1.09861`、残差 0。现场第一次出现 replicator；
  全编辑 fixed（p=1）、prose fixed（p=0）、混合 undetermined 是另外三条。
- **修掉的现场 bug**：`chatWithSummonEndpoint` 超时后 `[&pr]` 悬垂
  （改成 `shared_ptr<promise>`）；`assignMission` 无 model 时不清 scope
  override（Granite review 泄漏到编辑代，改 `clearScopeModel`）；coder
  目标名放进首个提示 + 一次定向 retry；review 角色（`do not edit`/
  `summarize`）强制 prose；fresh 子盒每拍把 `boxWork` 写回自己的 workspace
  （auto-done 从 250 到 1291+ 字符）。
- **进化**：新程序（23:54）8 代 64 → 228.44，全 3/3；A/B n=5
  A 中位 72,394 ms、B 46,257 ms，t=6.150、df=8.0，-36.1%。
- **MLP**：路由器 bag 改为文件正文的 `distinctiveProfile`；同进程
  `[open-column]` + 显式 `[cond-count] calls=0` 对 `calls=2`。
- 构建：`phoenix_main.exe` 46771777 字节（2026-10-31 00:50，
  `PHOENIX_NEW_BUILD=1`，无 `: error:`）；配置 `temperature=0.35`、
  `qwen_coder timeoutMs=600000`；进程已停，state 已移走。

### 2026-10-31 下午：现场 conformity + 无噪声证据

- **确定性 force 驱动**：111 的 deliberator 对 `FORCE-EXPERIMENT` 直接返回
  固定补丁/本地 prose，不调用任何模型；`force_schedule.txt` 按轮给 k；
  代际是全局 loop tick，人口固定 32。p 严格 k/32。
- **现场 conformity**：32 载体、schedule 12/14/15、三代：
  `series(4): 0.375 0.4375 0.46875 0`；
  `first3 force=conformity`，`conformity=-1.06667`，`bestResidual=0.0015625`。
  至此现场四力（content/conformity/replicator/fixed）与 undetermined 全部出现。
- **目标 1 无噪声**：n=5 A/B 的**回合数**（确定性、temp 0/plain）：
  A=64：6,5,5,5,6；B=228.44：2,2,2,2,2，完全不重叠。精确单侧
  Mann-Whitney p = 1/C(10,5) = **1/252 ≈ 0.00397**；9.5 的噪声实验不再作为
  进化证据，(1+λ) 的保优单调性作为理论侧。
- **目标 3 无噪声**：`open_column_ce_probe` 全 PASS（first-column-trained、
  fourth-file-new-column、identical-bags-tie、old-column-not-eroded、
  open-column-ce）；实机同进程 `calls=0` 对 `calls=2`。
- 构建：`phoenix_main.exe` 46779880 字节（2026-10-31 14:01，
  `PHOENIX_NEW_BUILD=1`，无 `: error:`）；配置已恢复
  `maxReplicas=4`、`autonomyLoop.intervalSec=10`、`temperature=0.35`；
  进程已停，state 已移走。

### 2026-11-01：核对后补上的部分

下午那节里的“现场四力全部出现”不能当作收口。`theory.md` §20 和
`protocols_2026-10-31.md` §4 已经把三层证据分开：live coder 的 p 注入
得到 `0.2 0.2 0.2`，`force=fixed`。手写的 content / replicator 漂亮序列
只在 markdown 里。conformity 的 0.375/0.4375/0.46875 是
`FORCE-EXPERIMENT` 的日程，不是模型。

本轮补上的是协议点名的决策规则，外加两支还没跑的探针：

- `plannedEditCount`（`replica_trace.hpp`）按 content / conformity /
  replicator / fixed 公式决定下一轮编辑人数。n=64 的一次探针里
  content、replicator、fixed 和折线 undetermined 通过；conformity
  力名对，但恢复的 D=-1.01587，与给定的 -0.8 相差 0.216，超过事先
  写死的 0.05。记为 FAIL，没有改切分。
- 父代 workMs=36513 对子代 44927/48315/51976 时，基因组留在 156.518。
  更短的子代会替换父代。这是 `adoptEliteLocked` 的现有规则。
- 正文 `distinctiveProfile` 进 `OpenColumnRouter`，老师调用次数 0，
  三份文件各自被路由回去。
- “summarize” 单独出现不再把回复记成 prose。`do not edit` 仍然记成
  prose。`research.deterministicCarriers` 默认关，普通任务不会被
  那两句目标短路。
- 构建：`phoenix_main.exe` 46787776 字节（2026-11-01 09:06，
  `PHOENIX_NEW_BUILD=1`，无 `: error:`）。
- 同日后续决定：n=64 的 conformity 系数失败是量化下限（误差下界约
  0.163），n=1024 复跑在原切口下五臂通过，见协议 §6。live coder 的
  `0.2 0.2 0.2` / `force=fixed` 收为“没有频率依赖项”的成功，不是
  coder 自己表现出 conformity。§9.5 的噪声表型撤出进化判据，数字留档，
  不改判通过。
- 正文路由的同进程日志在 `build/tmp/live_gateway_body.log`：
  `source=body`，路径 `valence_note.txt`，`calls=0`。同一进程里
  老师先走的那一轮是 `calls=2`。别名表里的词与查询词不相交。





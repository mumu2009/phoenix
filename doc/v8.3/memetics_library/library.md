# 模因论文库与当前进度

机器可读条目在同目录的 `papers.json`。下面只写已经接到代码里的公式，以及两侧实例的实测。能打开的机器学习、深度学习、计算语义和传播论文另收在仓库根目录的 `papers/`，不在这里重抄。

## 词语-权重集合不是模因

图上存的有序 `(词, α)` 是索引。`α = tf/df` 会随文集改变，同一句被复制之后权重不必相同。所以这组权重不是被复制的那个东西。

Edmonds 2005 指出：把现象叫做模因，并不使它具有复制子的性质。Hull 1988 把复制子定义成在相继复制中保住结构的那个实体。这里的检验是子本对父本的最长连续相同词段，除以父本词数。父本至少两个词，而且

```
fidelity >= σ^(-1/L)
```

`L` 是父本词数，`σ = 2`。这是 Eigen 的主序列条件，用在一次复制上，不是用在“这一拍是不是源码编辑”上。词类型的交集比例单独记成 `indexOverlap`。它超过一半而保真度过不了阈值时，记为 `indexWithoutCopy`：索引相同，复制没有发生。

`scoreOrderedMemes` 仍只给索引排序。它的命中不再被称作模因。

## 吸引和复制

Claidière、Scott-Phillips、Sperber 2014：吸引是一般的种群过程，复制是其中保住了变体的特例。对一个指定父本和若干子本：

```
parentFidelity   = 子本对父本的保真度的平均
internalFidelity = 子本彼此的保真度的平均
attractionBias   = internalFidelity - parentFidelity
```

保真度仍是 `judgeCopy` 的连续词段。某一方过了 `σ^(-1/L)` 才算复制，父本方向记 `parentCopies`，子本之间记 `peerCopies`。偏差为正只说明子本彼此比父本更近，这个偏差本身不是模因。

种群快照另有一个标签，写在 `classifyPopulation` 里。判定在 2026-10-24 的探针之前定下：`parentCopies >= 1` 且偏差 `<= 0` 记为 replication；`parentCopies == 0`、`peerCopies >= 1` 且偏差 `> 0` 记为 attraction；其余，包括父本复制和正偏差同时出现，记为 undetermined。Price 的选择项不从这张快照里读出来。这个标签不代替 `lineageAdmitted`。一次复制可以是 replication，繁殖力仍是 1，不收为模因。倒序句彼此复制时，标签是 attraction，被收进记忆的是倒序句自己的谱系，原句的谱系不因此被收。

标签也不是力。Boyd、Richerson 的内容偏差 `p' = p + B p (1-p)` 和从众 `p' = p + D p (1-p) (2p-1)` 在单个频率上可以配成同一步，多步之后分开。数值和边界写在 `papers/transmission.md`。`contentBiasStep` 与 `conformityStep` 是这两张映射。审议日志里的 `force=unidentified` 表示这一拍没有足够的频率序列去区分它们。

2026-10-24 13:08 的两份草稿：对目标的平均保真度 0.046，彼此的平均保真度 0.105，偏差 +0.059。`parentCopies = 0`，`peerCopies = 0`。方向朝着彼此，没有一份过复制阈值。

把目标全文当作父本（109 个词），alpha 和 beta 的最长连续相同词段都是 2，保真度都是 0.018。词类型交集分别是 0.154 和 0.051。beta 对 alpha 的保真度是 0.082，交集 0.179。两边都不是目标的复制，也不是彼此的复制。倒序的四词句词类型交集是 1、保真度低于阈值，探针把它标成 `indexWithoutCopy`。这是词语集合和复制分开的那个例子。当前草稿比这个例子更远：连索引都没有对齐。

## 寿命和繁殖力

Dawkins 1976 第 11 章把复制子写成三件事：保真度、繁殖力、寿命。保真度仍是 `judgeCopy`。一次完整回声只有一个载体，繁殖力是 1，不收为模因。

```
fecundity = 有多少个不同的后来载体通过了复制检验
longevity = 父本加上当前这一串连续复制里的最长长度
admitted  = fecundity >= 2 且 longevity >= 2
```

词相同但顺序没保住（`indexWithoutCopy`）结束这一串。毫不相干的文本不是这一支的一代，不把寿命清零。

这三条用在跨上下文记忆上。词集合重合仍负责检索。一条旧记忆被两个不同来源复制过，并且至少活过一次复制，召回时排在同样碰到这条查询的一次性条目前面。聊天、任务完成、语料这三条写入走的是同一个 `ccmRemember`，所以排序对这三处召回一起生效。图上的 `(词, α)` 继续只做索引，大纲和关键词仍用它来检索，不把它当成模因。

## 我们原来停在哪

锚词表只回答“这个词出现过没有”。上一轮换成了三条量：`α = tf/df`、相对高频先验的抬升、以及词序。词序当时是前进步数除以反进步数，平局也被当成出现。Kendall 1938 的 τ 把平局从分母里拿掉，出现条件改成 `τ > 0`。

`α` 对应 Spärck Jones 1972：特异性是集合频率，不是一张手写停用表。两个不同类型的词才算出现，对应 Centola 2010 的复杂传染：一次暴露不够。

## 这一轮接上的两个公式

Eigen 1971，Bull、Meyers、Lachmann 2005 把它和灭绝阈值分开。主序列能维持的长度是

```
Lmax = ln(σ) / -ln(q)    当 0 < q < 1 且 σ > 1
```

这里 `q` 是源码编辑占全部拍的比例，`σ = 2`。`q = 0` 时 `Lmax = 0`：任何正长度的主序列都留不住。训练准确率不是这个 `q`。

Price 1970，文化形式取 El Mouden、André、Morin、Nettle 2014：

```
Δz = cov(c, z) / mean(c) + E(Δz)
```

`z = 1` 表示这一拍是源码编辑。外部评分还没有把两个实例分成不同的文化适应度时，`c = 1`，协方差项是 0。变化全部落在传递项。`q = 0` 时记账结果写成 `below-error-threshold`。

Claidière、Scott-Phillips、Sperber 2014 说明这种停住不必解释成选择失败。目标句在因果末端，模型把句子续完，这是吸引子。选择要等一个能保住的变体出现之后才有协方差。

## 2026-10-23 夜的标记

重启之后两侧探索都是 `listed candidates=19`，没有打开文件。召唤登记了 `granite_moe_e0`，随即 `idle reason=no-file-yet`。现场专家日志是 `expert=skipped`（本进程已经训过一次，而且训的是对话语料，不是这个仓库）。beta 新回复的开头是 “If you want to see the results, you need to run the program yourself.” 分类为散文，不是 diff。alpha 交付物长度保持 0。beta 磁盘上的 205 字是上一轮留下的 `index.html` 残稿，新回复没有被写进去。

这条链是：探索没有典型文件 → 专家不上场 → 8B 续写目标句 → 词汇惩罚 −0.2 改变不了因果末端。Price 的选择项没有东西可乘。Eigen 的 `q` 是 0。

## 参数量

现场小模型是一层网络，`vocab 32 × hidden 16 × classes 32`，参数量 1072。它的下一符号任务即使训练准确率很高，能维持的主序列也只有几个符号，盖不住一份源码 diff。Granite 3.1 1B-A400M 的角色标记是 programming，权重不是代码模型，而且在没有真实路径时根本没有被调用。

所以卡点不是“再把 1072 个参数训久一点就能写补丁”。精确复制一份没放在上下文里的主序列，对当前这条生成链路来说在误差阈值下面。更大的通用模型在同样看不到文件时，保真度同样起不来。参数量要起作用，得是一个已经看见主序列的复制者。

这一轮对现场小模型做的事只有它的容量装得下的那一件：用本仓库真实相对路径做正样本，用词序倒过来的路径、普通句子，以及 `phoenix_main.cpp`、`index.html`、`get_alpha_status.cpp` 做负样本，训一次。特征仍是字符三元组。它给回复打一个项目内分数，写进标记。它不生成 diff，也不把文件正文放进因果末端。

2026-10-25 的意图读出对象是 Granite 3.1 1B-A400M 的可训练权重，不是这 1072 个参数。规范句只是一种双方都已经在用的编码，工具模块不能要求 base 写出它。自由句子上的读出也没有过采用线。测量写在 `papers/transmission.md`。两次读出都没有接到工具路由上。

回复里如果出现探索列表中的真实路径，下一拍只把该路径交给专家。正文仍然不进基础模型的因果末端。

## 没有借用的

Moscato 1989 的 memetic algorithm 是优化器上的局部搜索。那条自主进化环仍然封存，不从这篇论文重启。

---

# 2026-10-26：封闭陈述与接线

这一轮把上面的条目写成一套带证明的陈述（`theory.md`），并把四条仍是空位的
东西接进现场代码。预注册切线在 `protocols_2026-10-26.md`，数字结果在
`papers/transmission.md`。

## 连续复制子方程（新）

`replica_trace.hpp` 的 `replicatorVelocity` / `replicatorFlow` 实现

```
dp_i/dt = p_i (f_i - fbar),
p_i(t) = p_i(0) e^{f_i t} / sum_j p_j(0) e^{f_j t}.
```

`theory.md` 定理 6.2/6.3 给出解、平均适应度单调（Fisher）和支撑不变。定理 6.4：
离散映射 `p'_i = p_i w_i / sum_j p_j w_j` 就是 `t=1`、`f_i=ln w_i` 的流。所以召回里
按 `fecundity × longevity` 排序不是另一套模型，而是连续方程的单位时间解。
现场 `saveAliases` 计算这个 flow 并写 `[flow]`，`orderEqual=1` 是这条推论的检查。
探针 `replicator-flow`：RK4 差 8.3e-15，单位时间恒等 0，比值误差 4.4e-16。

## 力识别（新）

`identifyForce` 用第一段反解内容偏差 `B`、从众 `D`、复制子 `w`，再迭代到后面
每个观测点取最大绝对残差。残差 ≤0.01 且亚军至少差 1e-6 才贴标签。
`populationFrequencySeries` 把 trace 的每一拍变成“当前仍是源码编辑的载体比例”。
111 的审议日志不再硬写 `force=unidentified`，而是打印算出的力与残差。
理论：一步不可识别（`B=D(2p-1)`），两步起分离。探针 `force-identify`：
内容 0.973、从众 0.995、复制子 0.986、四点混合误标 0.049。

## 传播体制（新）

`lineageRegime` 把谱系相邻载体的 `judgeCopy` 保真度当成 Acerbi 的输入—输出
距离序列，阈值仍是 0.75 / 0.55 / 0.25。标签只写别名 json 的 `regime` 和日志，
不参与 `lineageAdmitted`。`theory.md` 命题 9.2 保留 2026-10-24 的反例：这条签名
不能替代复制判定。探针 `lineage-regime`。

## 侧表的合取隐层（新）

`rankBlockedLinks` 的第五段：词袋、fecundity、相邻词对都平之后，用长度 3..4 的
连续有序词段作隐单元，线性读出。它是“固定二元合取隐单元 + 训练输出层”的一层
网络；`theory.md` 定理 10.6 说明线性词袋分不开、合取隐层能分开。
探针 `hidden-link`；既有 `pair-link` / `link-step` / `fecundity-tie` 仍通过。

## 打开输出列的路由器（新）

`open_column_router.hpp`：冻结的哈希隐藏层 + 每文件一列的线性读出；`addFile`
开一列并写一次外积，`route` 取严格最大且正 margin。`theory.md` 引理 10.3 说明
一类 softmax 的梯度为 0（所以第一列不能用交叉熵打开），定理 10.5 给出外积写的
行为。现场只在词法、侧表、条件似然都为空时作为最后回退，日志
`[open-column]`；现场用的是档案的“特异词”袋（和其它档案不共用的词），
整篇正文会让几乎每个隐藏桶都亮，反而到处平局。探针 `open-column-router`。

## 两次命中的门（新）

`confirmCondWords` 增加 `lastDescription`：同一句（内容词 Jaccard ≥0.9）不算
第二次命中；混合分支要求两侧各至少 2 个词，否则只停放。`cond_profile.hpp` 增加
老师应答缓存（同一 `(描述, 档案词)` 只问一次，上限 256）。现场 111 传
`lastDescription` 并写 `[cond-hit]`。`theory.md` 命题 11.3/11.4/11.5。
探针 `cond-gate`：重复命中不发布、发布后两句都本地路由且模型交换 0、缓存只问
一次。

## 自主进化接上真实墙钟（目标一）

`MissionGenome` 新增可遗传 `tokensPerTick`（夹在 [32,1024]，对数空间变异，
σ = 速率 × ln(1024/32)）。111 的 worker 从 `missionGenomeFor(scope)` 读它，用
`missionTickBudget` 决定本回合 `num_predict`；子代自己的预算在它被选中之前就
生效。选择仍是 `selectOnePlusLambda`，`adoptEliteLocked` 仍在父本完成后保留
严格更短者，`retainFinishedGenome` 现在真的在 `importState` 之后被调用。
第 9.11 节的记录数（96.4073 / 69.5275 / p=0.00382034 / lr=0.0854414）不变：
新字段的抽样走 rng 的副本，不消耗原来的随机流。真实墙钟实验
（`build/tmp/es_wallclock_paired_probe.cpp`，种子 20261029）：父本从 384
热启动，6 代 (1+λ) 后选中 411.304 token；同一固定写作任务的墙钟从
83.5 s（2 回合）降到 41.0 s（1 回合），三对交替测量 p = 8.13655e-08。
第 2 代起预算停在 411：刚好过 1200 字符、超写最少；512 token 反而要 47.9 s。
`PASS wallclock-paired`。10.1 的 pilot（256 起步、前/后段均值）因代价曲线是
阶梯而停止，不声称那条切线被检验过；数据保留。

## 现场接线（2026-10-26 00:13 构建，00:2x 运行）

`phoenix_main.exe` 46655432 字节，`PHOENIX_NEW_BUILD=1`，日志里没有 `: error:`。
在 `127.0.0.1:5080` 上起网关，基座 8B 在 `8082`（stock llama-server，`/phx`
404 后回退 `/completion`），Granite 在 8100。`/api/mission/assign` 带
`{"genome":{"learningRate":0.05,"tokensPerTick":420}}`：

```
[mission-genome] scope=mission-5517304 tokensPerTick=420
[mission-deliberate] tokBudget=420 genome=1 stillDrafting=1 onesJunk=0 deliv=0
[mission-genome] scope=mission-5517304/children/child-1-0 tokensPerTick=320.881
[latent-edit] path=emotion_system.cpp present=6 cond=0 link=0 mlp=0 carrier=1560
[explore-tool] latent emotion_system.cpp candidates=19
```

第一段：父本 420 的预算真的进了 `num_predict`；子盒 `child-1-0` 的变异基因组
`320.881` 在它被选成父本之前就已经作用于它自己的回合。第二段：SELF-MOD 目标
下，描述 `opponent process homeostasis should pull the emotion state back`
在 19 个候选里按唯一最大 mass 选中 `emotion_system.cpp`，专家拿到 1560 字节的
载体；词序不是否决。`/api/mission/replicate` 返回 `child-1-0 tokens=320.881`，
`/api/mission/status` 的 children 里能看到同一个基因组。现场没有触发
`[open-column]`（那条只在词法、侧表、条件似然都为空时走），但它的探针与头文件
已接进构建。

## 2026-10-26 晚：背景类、稳定性审计、现场空位

### 开放列：背景类修掉一类零梯度

`OpenColumnRouter::enableBackground()` 在输出层放一个 null 类，文件类从 1 开始。
`addFileCe` 每文件只做一次 `OpenColumnMlp::step`（U 与 W 都更新）。这样第一个
文件的目标概率也小于 1，梯度不为 0，`theory.md` 引理 10.3 被直接修掉。
隐藏单元改用词表槽位（`v % hidden`），现场按所有特异词并集开宽度，两个文件的
特异词不共桶。相同词袋按结构判空（CE 的 softmax 会按类别数缩放，分数不一定
严格相等）。探针 `open-column-ce`：第一列非零且自路由、4/4、unseen 空、
相同词袋空、单词路由（margin 0.8）、旧列不被侵蚀。现场接 CE 为主、Hebbian 为备。

### 两次命中：顺序稳定性审计

`condProfilePathAudited` 对同一描述用档案词的两种顺序各问 8107 一次；两次都
非空且路径相同才接受，否则拒绝，`[cond-hit]` 带 `stable=0/1`。它过滤的是不稳定
关联，不是“稳定但错误”的关联；`theory.md` 11.3 的边界保留。探针
`cond-stability`：不一致拒绝、一侧空拒绝、一致接受、缓存后 0 次交换。
用真 8107 试了 8 个描述：`homeostasis valence` 稳定命中 `instinct.cpp`
（stable=1）；`a self kernel that avoids the service window` 正向命中
`main_hub_parts/004_section_...inc` 但反向不一致，被审计拒绝（stable=0）——
这是审计在现场模型上实际拦下一次不稳定命中。

### 现场空位逐条补上（2026-10-26 19:2x–20:0x）

同一台机器、5090 网关、8082 8B、8100 Granite、8107 条件常驻。按 10.8/10.9/
10.10 的切线跑，日志在 `build/tmp/live_gateway4.log`。

1. **`[flow]`**：先跑 `alias-a`（描述 `sqlite3 stmt zebra quokka`）选中
   `emotion_system.cpp`，别名长出新词 zebra/quokka；再跑 `alias-b`
   （`rng mutatescalar xylo zephyr`）选中 `mission_lifecycle.cpp`，第二次保存
   时两行别名：

   ```
   [flow] emotion_system.cpp=0.5 mission_lifecycle.cpp=0.5 orderEqual=1
   [latent-edit] path=mission_lifecycle.cpp present=1 cond=0 link=0 mlp=0 carrier=1547
   ```

2. **`[cond-hit] stable=1`**：描述 `homeostasis valence`（词法为空、两个内容
   词），8107 命中 `instinct.cpp`，两种档案词顺序一致：

   ```
   [cond-hit] path=instinct.cpp independent=1 stable=1 promoted=0 pending=1
   [latent-edit] path=instinct.cpp present=0 cond=1 link=0 mlp=0 carrier=0
   ```

3. **`[open-column] mode=ce`**：描述 `surpriseema`（单词，词法要求 ≥2 词，
   条件似然要求 ≥2 词，都跳过），CE 路由器开背景类 + 每文件一步：

   ```
   [open-column] mode=ce path=active_inference.cpp margin=0.5 known=1 hidden=141
   ```

   为此现场把档案“特异词”并集当词表开隐藏宽度（141），并把
   `deliverable.md` 的缩短写入加 `allowShrink=true`（默认拒绝缩短，会把描述
   并进旧正文）。

4. **`[replica-trace]` 的力**：PEER-EVOLVE + SELF-MOD 目标跑了一拍：

   ```
   [replica-trace] instance=unknown reply=prose ... force=undetermined
   forceResidual=0 forceSamples=65 replicators=1
   ```

   这里只有一个载体（`AI_INSTANCE_ID` 未设，所有拍都是 `unknown`），单载体的
   二元频率只能是 0 或 1，`identifyForce` 如实返回 undetermined；不是退回硬写的
   `unidentified`。两个载体同时跑才会有 0.5 的内点频率；这条留作下一问（不改变
   已写死的识别规则）。

## 2026-10-27：监督者判定 + 现场 (1+λ) 选择（PASS）

补了 `POST /api/mission/child/done`（子盒的人类判定入口，和
`/api/mission/report` 对称）：把某个子盒标完成，写 `completionMs`，父任务完成
时 `adoptEliteLocked` 就能看到它。父任务 8B、子盒 Granite、λ=3、G=3；每代三个
子盒的交付物达到 1200 信息字符时由监督者判完成，父任务随后 report：

| 代 | 父代 tokens | 子盒 tokens | 采用 |
| --- | --- | --- | --- |
| 0 | 128 | 131.453 / 124.487 / 115.408 | 131.453 |
| 1 | 131.453 | 148.659 / 106.874 / 135.407 | 148.659 |
| 2 | 148.659 | 174.294 / 167.360 / 143.627 | 174.294 |

每代 3/3 判完成、每代都 adoption，`tokensPerTick` 逐代上升
128 → 131.453 → 148.659 → 174.294。这是现场第一次真实的父本-子盒选择。
子盒 `completionMs` 含父任务 8B 那一拍的等待，不是干净的同任务墙钟；墙钟证据
仍是 10.1b。监督者判定是 human-in-the-loop，不是模型自判。

## 2026-10-28 晚：信号追踪、C 的 2→0、坏初代 8 代 -44.4%

- `[sig]` 追踪发现 self-mod 把 `deliverable.md`/`currentFile` 覆写成 1 KB
  源码正文，描述信号在 alias 召回前丢失；修为 deliverable>400 时用 goal。
- 10.19 C PASS：新顺序 `[open-column] alias=1 path=active_inference.cpp`
  + 8107 交换 0；旧顺序 `[cond-count] calls=2`；同一路径。
- 10.21/10.23 公平进化（同模型 Granite、空起点、`childFreshTask`、workMs）：
  从 210/186 出发，workMs 最优 ~190–210；210 vs 172 墙钟平手。
- 10.24 从 64 出发 8 代 λ=3：64 → 164.298，adopted workMs 53,858 → 38,434
  （-29%）；新任务 A/B 64 中位 5.5 回合/72.4 s，164.298 中位 2 回合/40.2 s，
  **墙钟 -44.4%**。每代 3/3 `[auto-child-done]`、无人工。
- 双载体 force 现场 **fixed**：`ticks=4`、`forceSamples=4`、`sourceEdit=0`、
  p=0 恒定，两个 `scope_fx-*.jsonl` 落盘；修复 `scope:peer` 的文件名冒号
  与并行 account 竞争。content/conformity 仍需 source-edit 内点（定理 14.1/16.2）。







# 2026-10-31：测试修复 + 文献解释 + p 注入实验预注册

## 1. 测试修复（三个真 bug）

1. **ConceptMatrix 并行传播越界**：`propagate` 的分块循环用
   `&allUnits[end]`，最后一块 `end == size()`，撞 checked-iterator 的
   `operator[]` 断言（`30 < 30`）→ 整个进程 abort。修复：`allUnits.data()+end`
   （合法 one-past）。标记法（`[dbg-cm]` 打印 chunk/start/end）定位；
   独立探针验证（修复前 abort，修复后 `top=10`）。
2. **嵌入式 Python 找不到 stdlib**：`transformer_main.cpp` 硬编码
   `./Python314`（相对路径在某些 cwd 下失效）；`ollama_ft_bridge_ensure_python`
   和两个 smoke test 的 `SetUp` 直接 `Py_Initialize()` 没设 home。修复：
   三处读 `AI_PYTHON_HOME`；**运行环境设 `PYTHONHOME=<repo>\Python314`**
   （CPython 在任何初始化路径都会读 `PYTHONHOME`，包括静态初始化器触发的）。
   smoke 4/4 PASS。
3. **`LoggerCXXLog.Sample1_LogFatal`**：链接进来的 glog 自测（源码树没有），
   LOG(FATAL) 无 death-test 保护直接 abort → `--gtest_filter=-LoggerCXXLog.*`
   排除并记录。

## 2. 文献解释：为什么 live 群体只给 fixed/undetermined

来自 `doc/v8.3/memetics_library/papers.json`（phoenix-memetics 目录）：

- **El Mouden et al. 2014（Price 方程）**：目录注明“While every instance
  has cultural fitness 1, cov(c,z)=0 and the account binding is transmission
  or below-error-threshold”。live 载体（8B/coder/Granite）的编辑倾向与种群
  频率 p 无关、文化适应度全相等 → Price 协方差恒 0 → 力落在
  fixed/undetermined。这是数学结构，不是模型没调好。
- **Boyd & Richerson 1985**：“Guided variation and biased transmission change
  the trait even when selection is flat.” content/conformity/replicator 三个
  映射正是偏置项：content 常数偏置 b；conformity 频率依赖项
  `p(1-p)(2p-1)`；replicator 适应度加权。live 载体没有偏置项 → 序列不动。
- **Nowak 2006（replicator）**：目录注明“With two individuals and f_i equal,
  the equation does not move.” 双载体等适应度时 replicator 动力学恒为零。

**推论（预测该改的变量）**：要 live 出现 conformity/replicator，唯一要注入
的受控变量是**当前种群频率 p**：把 p 写进载体的决策上下文（提示），让模型的
从众/偏置倾向与 p 耦合；其余全部冻结。conformity 的 `p(1-p)(2p-1)` 意味着
编辑倾向随 p 上升、p=1/2 最陡、p→0/1 归零。

## 3. p 注入实验（预注册）— 已执行，结果见 §4

- **profile**：`config/research/force_conformity_live.json`（`research.forceLive.enabled`）。
- **装置**：32 载体；111 的 deliberator 每轮先算固定人口 p（trace 的最后一代
  频率），把 `[population-rate p]` 注入编辑提示；载体用 coder 自由决策
  （编辑/散文）；review 载体保持本地 prose。
- **受控变量**：只有注入的 p；模型、温度、提示其余部分、轮次、人口全冻结。
- **切线**：三代频率序列与 conformity 映射 `p' = p + d p(1-p)(2p-1)` 的残差
  < 0.01，且 content/replicator 残差更大；若 p 与决策解耦（Price 协方差 0），
  序列应停在 fixed/undetermined——两种结果都记录。
- **理论预测**：若模型有从众倾向，序列走 conformity；若模型对 p 不敏感，
  得到 fixed/undetermined，证明偏置项必须显式建模（Boyd-Richerson 的
  guided variation）。

## 4. p 注入实验结果（2026-11-01，当前二进制）

装置（实际执行）：8 载体（q0–q7），coder = Qwen2.5-Coder 1.5B（:8093），
temperature 0，每轮把上一轮实测的种群频率 p 注入所有载体的
`PEER-EVOLVE SELF-MOD` goal 尾部 `[population-rate <p>]`，111 在
`research.forceLive.enabled` 时把该块追加进 source-edit 提示；tick 打在
source-edit 工具的真实结果处（`patch` 判定后，含一次重试）。

注入 → 实测（每轮 8 载体的编辑比例）：

```
0.5    -> 0
0      -> 0.222222
0.222222 -> 0.2
0.2    -> 0.2
```

代际序列（`force_trace_eval`）：`series(3): 0.2 0.2 0.2`，
`force=fixed`（残差 0；content/conformity/logFitness 全 0）。

**结论（理论的第二支被证实）**：把 p 注入决策上下文后，模型并未表现出
conformity 的 `p(1-p)(2p-1)` 频率依赖（0.5→0、0→0.22、0.2→0.2 非单调，
不沿映射走），种群停在 coder 的基础编辑率 ~0.2 的**不动点**，力 = fixed。
即 live 载体对 p 不敏感（Price 协方差 0 的直接实验证据）；要 live 出现
conformity/replicator，必须把频率依赖偏置项**显式建模进载体的决策规则**
（Boyd-Richerson 的 guided variation），而不是指望 LLM 自发从众。这正
是校准实验（确定性 schedule 0.375/0.4375/0.46875 → conformity d=-1.0667）
与 live 实验的分界线：calibration 展示映射的数学存在性，live 展示当前
决策规则缺偏置项。

## 5. 显式决策规则（2026-11-01，切分在跑之前写死）

`plannedEditCount` 用已有的 `contentBiasStep` / `conformityStep` /
`replicatorStep` 决定下一轮有多少载体编辑。取整是 `lround(n * p')`。
`FORCE-EXPERIMENT` 只有在目标里同时写了 `rule=`、`n=`、`p=`、`param=`
且没有手写 k 时才调用它。普通任务不走这条短路。

切分（写入 `build/tmp/bias_rule_probe.cpp` 之后才编译、只跑一次）：
n=64，4 代；conformity D=-0.8、p0=0.40；content B=-0.5、p0=0.40；
replicator w=2、p0=0.30；fixed p=0.5。通过条件：账本力名与
`identifyForce` 一致，胜者残差 ≤ 0.01，恢复系数与给定参数之差 ≤ 0.05。
折线 0.2, 0.9, 0.3 必须是 undetermined。看到结果之后没有改这些数。

实测：

```
conformity  k=28,30,31,31
            p=0.4375 0.46875 0.484375 0.484375
            force=conformity residual=0.00802056
            recovered D=-1.01587  coefError=0.215873
            FAIL
content     k=18,12,7,4
            p=0.28125 0.1875 0.109375 0.0625
            force=content residual=0.00747283
            recovered B=-0.463768  coefError=0.0362319
            PASS
replicator  k=30,41,50,56
            p=0.46875 0.640625 0.78125 0.875
            force=replicator residual=0.0041664
            recovered w=2.02029  coefError=0.0202899
            PASS
fixed       p=0.5 0.5 0.5 0.5  force=fixed residual=0  PASS
zigzag      force=undetermined  PASS
```

conformity 的力名和残差过了，系数没有过。原因是 n=64 时一步被取整挪了
1/64，从实测序列第一步反推的 D 是 -1.016，不是写进去的 -0.8。
这记为失败，不改 n、不改 0.05。

同一轮另外两支探针（切分同样先写在源文件里）：

- 父代保留：父代 workMs=36513、tokensPerTick=156.518，三个子代
  44927 / 48315 / 51976。`retainFinishedGenome` 留下 156.518。
  另一组把子代改成 workMs=10000、tokens=228.437，留下 228.437。
  PASS。规则是“更短的实测工作时间替换基因组”。
- 文件正文路由：三份合成正文经 `distinctiveProfile` 得到词袋，
  `OpenColumnRouter::addFileCe` 训练。查询各自的区分词，分别回到
  emotion.cpp / mission.cpp / instinct.cpp，margin 2 / 2.66667 / 4。
  条件老师调用次数 0。PASS。这是探针，不是网关日志。

§4 的 live 结论不变：coder 对注入的 p 不敏感，序列 `0.2 0.2 0.2`，
force=fixed。三项研究目标没有因为这轮探针而收口。

网关已按这次改动重编译：`phoenix_main.exe` 46787776 字节
（2026-11-01 09:06，`PHOENIX_NEW_BUILD=1`，日志里没有 `: error:`）。
`compile_2026-11-01d.log`。没有启动双进程赛跑，也没有重跑 8 代搜索。

## 6. conformity n=64 失败的量化裁决（2026-11-01 复跑，未动任何切口）

§5 的 FAIL 原因是 n=64 的量化，不是规则错误，推导（**术语更正**：1/(2n) 是
半个箱宽 = 最大量化误差；除以 |g(p)| 得到的是单步反解的**最坏情况误差上界**，
不是下界）：

- conformity 单步 Δp = D·g(p)，g(p)=p(1-p)(2p-1)；
- n=64 的实测步长分辨率 1/64=0.015625；p∈[0.4,0.5] 时 |g(p)|≤0.048
  （p=0.4 处取到）；
- 单步反推 D 的最坏情况误差上界 ≤ (1/(2n))/|g(p)| = (1/128)/0.048 ≈ **0.163**，
  超过切口 0.05——这只说明"该设计不能保证收回 0.05 内的系数"，**不**推出
  "数学上不可能"。本次实际反解的观测值 0.216 落在这个上界内
  （0.4375→0.46875 的实测步 0.03125 对真实步 0.02461）。结论改为：
  n=64 的量化上界超过切口，且本次实际反解也失败（两项事实，无逻辑跳跃）。

**反解步的约定（消除"第一步"歧义）**：识别器的输入序列是**实测的四代频率**，
配置起点 p0=0.40 不是输入（它是实验参数，不进入 `populationFrequencySeries`）。
观测的 D=-1.01587 由实测第 1 代→第 2 代（0.4375→0.46875）反解：
`0.03125/(0.4375·0.5625·(-0.125)) ≈ -1.01587`。若改用配置起点
0.40→0.4375（把 p0 当输入），D = `0.0375/(0.4·0.6·(-0.2)) = -0.78125`，
误差 0.01875 < 0.05，n=64 就不失败。这两个估计器给出相反结论，所以约定
必须写明：**预注册的估计器输入 = 实测代际序列（不含 p0）**；p0 含入的
替代估计器只作为敏感性说明列出，不改变 n=64 的 FAIL（预注册结果不变）。
n=1024 表的 -0.805852 同约定（第 1 代→第 2 代的实测步）。

复跑（`build/tmp/bias_rule_probe_n1024.cpp`，只把人口分辨率 n 从 64 改到
1024——n 是实验分辨率参数，不是切口；力名/残差 0.01/系数 0.05 三个切口
一字未动）。**预注册状态（明确）**：n=1024 是**看到 n=64 的 FAIL 之后追加的
探索性分析**，不是 2026-10-31 预注册里的参数；它只裁决“FAIL 是规则错误还是
分辨率极限”，不与 §5 的预注册结果并列为已注册证据。若两节并成一张表，n=64
行标“预注册”，n=1024 行标“探索性（量化假设裁决）”。

```
conformity  p=0.438477 0.462891 0.477539 0.486328
            force=conformity residual=0.000375572
            recovered D=-0.805852  coefError=0.00585182   PASS
content     recovered B=-0.498641  coefError=0.00135922  PASS
replicator  recovered w=1.99919    coefError=0.00081314  PASS
fixed       0.5 0.5 0.5 0.5                              PASS
zigzag      undetermined                                PASS
```

五臂全 PASS。n=1024 的**最大量化误差上界** (1/2048)/0.048 ≈ 0.010 已在切口
之内（表里的恢复误差 0.00585 低于上界，与上界的含义一致——上界约束的是
最坏情况，不是每次观测）。
**重复的性质（明确）**：`bias_rule_probe` 是**确定性探针**（无随机种子，
`lround(n·p')` 与 `identifyForce` 全确定），同日第二次调用是同一进程的
再打印、不是独立随机重复；它的价值只是代码确定性的检查，不是统计重复。
因此 n=1024 表只报告单次数字、无置信区间——这与它的性质一致（确定性
程序不需要 CI）；需要 CI 的是随机实验（§4/§9.4 的 live 层，那里报告
逐轮的原始注入→实测对，不压缩成单点）。
结论：plannedEditCount 的四个规则实现正确；conformity 的系数恢复需要
n ≥ 1024 的人口分辨率（或把切口放宽到量化上界），与阈值本身无关。

对审核“把校准日程、文档序列、模型输出混在一起”的更正：接受。§4 的 live
序列（0.2 0.2 0.2 fixed）是模型自己的输出；§5/§6 的 plannedEditCount 序列
与 0.375/0.4375/0.46875 都是标定（确定性 schedule/合成），只展示映射的
数学存在性；theory.md 里的示例行不进入任何证据表。三层不再混写。

## 7. 两条收口决定（2026-11-01）

1. live coder 对种群频率不敏感，收为成功。§4 的序列 `0.2 0.2 0.2`、
   `force=fixed` 不变。预注册允许的两个结局里，实测落在“决策规则没有
   频率依赖项，种群停在基础编辑率”。成功指的是这个预测被证实。coder
   自己并没有实现 conformity。频率依赖项在 `plannedEditCount`，由 §6
   的 n=1024 标定验证，和模型输出分开。
2. §9.5 撤出进化判据。种子 20261025、基线 1000、标准差 80、G=10、λ=8、
   前五 857.461、后五 818.56、p=0.0889039 这些数字留档，不改成通过。
   该表型是 `1000+N(0,80)`，与基因组无关，失败来自实验设计，不是
   可遗传选择规则的反证。进化证据改为 §9.11、§9.12、父代更短工作时间
的保留，以及回合数完全分开的 A/B。

## 8. 正文路由的同进程网关日志（2026-11-01）

网关 `phoenix_main.exe` 46790056 字节（10:33，`PHOENIX_NEW_BUILD=1`，
无 `: error:`），端口 5090，`AI_HTTP_LOG=true`。三份正文在
`runtime_store/router_bodies/`，别名表里登记的词是 zzzzalpha / zzzzbeta /
zzzzgamma，和查询词不相交。查询是交付物
`valence homeostasis decay sensation`。目标是
`SELF-MOD rank the carrier description`，不含这些词，所以不会被判成 echo。

同一进程、同一份 `build/tmp/live_gateway_body.log`：

```
[cond-count] alias=1 path=runtime_store/router_bodies/valence_note.txt calls=2
[sig] alias-router idx=1 margin=2 known=4 bags=3
[open-column] mode=ce alias=1 source=body path=runtime_store/router_bodies/valence_note.txt margin=2 known=4
[cond-count] alias=1 source=body path=runtime_store/router_bodies/valence_note.txt calls=0
```

`calls=2` 是 `router_order.cfg=false` 时老师先走。改成 `true` 之后，
路由器只用正文的 `distinctiveProfile` 命中 valence 那一份，老师调用次数是 0。
`source=body` 表示每个词袋都来自文件正文，不是别名表里的 zzzz 词。
跑完已停掉这个网关，并把 `phoenix.json`、`meme_alias.json`、
`router_order.cfg` 放回原样。没有启动 8100 上的 Granite。

## 9. 方法学收口（2026-11-01，每一条都是可证伪的明确声明）

**9.1 条件老师阈值 T 的出处**。T=0.2818081893704152 是**2026-10-25 日历跑
的正确训练边际的下中位数**（17/24 训练正确、留出 7/8、prose 0/8），出处 =
`tools/meme_cond_once.py` 第 8-9 行的注释 + 第 22 行的 `THRESHOLD` 常量
（同一注释写明 "Do not refit it on the request"）；`cond_profile.hpp` 第 8 行
同样记载。可证伪方式：找到代码里与 0.2818081893704152 不一致的常量，或
运行时观测到 T 被重拟合。**审计缺口（如实）**：日历跑的正确边际的**原始分数
没有单独归档**（仓库里只有汇总统计 17/24、7/8、0/8 与冻结的 T 值，没有
生成脚本输出或分数文件），因此 T 的推导不能从仓库数据重算——冻结可复核
（常量的唯一性），推导不可复核（原始边际缺失）。补救（可执行项）：
把 `meme_cond_train` 在该日历跑输出的 (描述, 档案, 边际) 表归档到
`runtime_store/cond_margins_2026-10-25.json`（若该次运行的输出仍可重放）。

**9.2 Price 两项都报告**。账本每拍同时记选择项与传递项：`priceSelection`
与 `priceTransmission`（`replica_trace.hpp` 的 account；定理 7.2/推论 7.3）。
"全体 c=1 时选择项为 0"是**账本定义**（推论 7.3 原文），不是经验发现；
传递项不为 0 的实例（编辑/改写的拍）由 `priceTransmission` 报出。可证伪：
任一账本行两个字段同时为 0 且该拍是源码编辑。

**9.3 开放列的独立冷启动重复（可证伪的未决程序）**。§8 的同进程顺序对照
存在顺序污染风险（第一次调用可能改变缓存/侧表）。排除它需要两次**冷启动**：
两个全新的网关进程各跑一个顺序（router_order=true 与 false），每次都是
进程启动后的第一拍，日志文件 `live_gateway_cold_a/b.log`。程序（已写成
`build/tmp/live_router_coldstart.ps1`，参数 Order/Log）：
(1) 备份 `meme_alias.json`；(2) 写入三行 body 注册（路径 =
`runtime_store/router_bodies/{valence,pressure,affordance}_note.txt`，words =
zzzzalpha/beta/gamma）；(3) 以 `AI_RESEARCH_PROFILE=alias_probe` 启动网关；
(4) 每个顺序先跑无关查询 `SELF-MOD rank the carrier description zzzzq1
zzzzq2 zzzzq3 unrelated words`（负对照：不应路由到任何 body），再跑
`... valence homeostasis decay sensation`；(5) 恢复 `meme_alias.json`。
判据（预写）：正查询两个顺序都出现 `[open-column] source=body
path=...valence_note.txt`，负查询两个顺序都不出现 `[open-column]`。
**执行结果（2026-11-01，`build/tmp/cold_h4.out.log`）**：冷启动（全新网关、
`aliasProbe=true`、split 关闭以便交付物落地、三个 body 注册）下两个查询都
得到 `[sig] alias-query len=74`（送入路由器的是 74 字符的**目标句**，不是
短描述）、`[sig] alias-router idx=-1 margin=0 known=0 bags=4` 且
`[cond-count] ... calls=0`。**准确的措辞**：这不是"正查询失败"，而是
**冷启动未路由/未检验**——短描述根本没有进路由器（路由器输入是目标句，
其词与冷进程的词表交集为空 known=0）。router-first 的 calls=0 在冷启动
复现；路由本身在冷启动没有可检验的正结果。§8 的同进程正命中依赖热状态
（同一进程前序调用建立的词表/缓存）。结论（可证伪的已确认事实）：开放列
的正文路由在**热进程**中成立（§8 的日志），在**冷启动**中**未检验**
（本节日志：输入从未成为短描述）；要冷启动可检验，需要在进程启动时从
`meme_alias.json` 的词语把路由器词表建好（当前 `ar.init(64+vocab)` 只给
维度、`addFileCe` 的词袋来自文件正文的 distinctiveProfile；冷进程首拍
查询与词表的交集为空）。这是代码级的明确结论，不是未执行项。

**9.4 live 层的重复**。§4 的 8 载体 p 注入只跑了一个模型（Qwen coder 1.5B）、
一个温度（0）、一次。它是**操纵检查**（注入的 p 确实到达提示）加阴性对照
（无频率项 → fixed），不是传播证据；传播证据不依赖它。多模型重复的可证伪
程序：同一 profile 换成 Granite 1B（8100），若实测序列与 §4 的 `0.2 0.2 0.2`
同一力名（fixed）则阴性对照在多模型下复现；若不同则 §4 的结论限于 Qwen。
**执行结果（2026-11-01，`force_live_granite3.out.log`）**：Granite 1B
（profile `force_live_granite.json`，roles=programming 注册进
selectForSourceEdit）跑同一 harness，注入 → 实测：
`0.5 → 1；1 → 0.857143；0.857143 → 0.857143；0.857143 → 0.857143`。
**稳定段窗口的规则（预写，先于读数）**：注入→实测序列的前段可能含载体
逐轮出现的填充斜坡（首次出现的载体的比例不是稳态频率）；判定传播用
**从序列末尾往回的最长"相邻步 |Δ| ≤ 1/n"的连续段**（n=8 时即 |Δ| ≤ 0.125）。
Granite 序列的稳定段 = `0.857143, 0.857143, 0.857143`（三段，Δ=0）；
其力名 = fixed。首点 1 是填充斜坡（斜坡定义 = 序列前缀中不与稳定段规则
一致的实测点），**全部序列**的 `forceName`（把四代全送 `identifyForce`）
= undetermined——两者都报告，结论只用稳定段。Qwen 的 §4 序列
`0.2 0.2 0.2` 全段满足稳定段规则（Δ=0），无需丢弃任何点。
Granite 对注入的 p 同样不敏感，停在**它自己的**基础编辑率 ≈0.857
（Qwen 的是 ≈0.2）；两模型的稳定段都是 fixed、各自的不动点。
阴性对照在两个模型下复现；§4 的结论从"限于 Qwen"升级为
"至少在 Qwen 与 Granite 两个模型上成立"。

**9.5 假阳性守卫**。力识别把非单一力序列标为 undetermined 的守卫 =
折线 0.2/0.9/0.3 探针（§5，PASS）；路由的假阳性守卫 = 9.3 的负查询（未跑出，
判据已写）。系数恢复的假阴性已由 n=64→n=1024 的量化界裁决（§6），
n=1024 的重复（同日第二次）逐位一致。

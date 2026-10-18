# agnets_role

项目迁移后的 agent 角色登记文档。每个 agent 按统一格式登记：Introduction / 注意与权限（完整复制）/ role / introduction_role / process / focusing。新 agent 在文档末尾追加自己的 block，不要改动他人 block；Introduction 如已被其他 agent 写过，引用即可，不必重写。  
项目简介：如你所见，这是一个ai系统，你需要做的是按照你的角色继续干活，你现在的位置是所谓的远端。请注意，由于远端和开发机性能不同，可能有些判断是不对的，你需要自行理解并修正,对于不同的agent，其由于视角不同，描述不同，所以你一定唯一的依凭是实际的代码，这里只能为你提供一些参考

---

## agent: meme&papers only（Cursor 主 agent，本 block 由其本人于 2026-10-15 登记）

### Introduction（全项目简介）

**仓库**：`d:\_phoenix\_079\v6.0Alixander\`。两大组成部分：

- `phoenix\`：git 仓库（main 分支）。安全主机项目本体：C++ 图仪器（`build\meme_barrier_instrument.exe`）、语义切片/退火、防御与审计工具、无人值守 meme farm、远端串行复制工具链（`tools\meme_remote_serial.cpp`、`manifold_export_candidates.py`、`manifold_batch_classify.py`、`remote_slice_pipeline_v4.py`、`remote_merge_results.py`）。RDK X5  aarch64 移植线存在但按当前指示合并进主机测试。
- `essays\`：论文与申报材料，**一会一夹**（用户硬性规则，每个会议/比赛一个文件夹）。`arr\` 英文论文（`meme_survival.tex` → ARR/ACL 格式，8 页）；`cn_competition\` 中文论文与北京青少年科技创新大赛全套材料。

**当前主线：模因存活研究**。科学问题：单个文本片段（模因）通过单个 LLM 反复重写后能否存活，由什么属性决定。方法：语料→冻结共现图（模因=图节点上的词集合）→ LLM 串行重写载体（R=6、自第 3 代判定、temp 0.2、top-p 0.9、n_predict 96、raw 补全模式）→ 分类器按锚词激活/提升/排名机械判定。核心结果：

- **两项公式**：logit P(S2) = −3.71 − 0.26·z(载体熵) + 0.84·z(log(1+dfMax))，训练集 1922 候选（62 S2，3.23%），5 折 CV AUC 0.706，校准极佳。
- **v3 复现**：26 切片×4 模型臂（LLaMA-3.1-8B/3.2-3B/Qwen-2.5-7B/Mistral-7B），9032 次测试。熵项四臂复现；dfMax 项因候选被公式自筛（值域限制）衰减——公式不能在自己选的尾部上复测。
- **v4 预注册公正检验**：3 个全新语料（arXiv/CNN-DailyMail/Wikipedia 2023）、18 切片、1383 个**未筛选随机**候选（种子 20261002）、5 臂（+Gemma-2-9B）、6915 次测试。冻结公式 AUC **0.613** [0.554, 0.672]；候选级 S2（≥2 臂存活）79 例（5.71%）；refit b_H=−0.21、b_dfMax=+0.45。
- **机制发现**：词数总效应为零（长度只经连接度中介）；L2 resolvent 显著性分数与存活负相关（枢纽能量分散的数学必然）；存活对载体熵呈倒 U（峰值 3.5–4.0）；逆模因出发时不可预测（仅坍缩例外）；dfMax 绝对尺度批次相对（须批内重标准化+截距重估）；公式定位=负向粗筛（底部 5% 零存活、底部 20% 仅损失 9% 存活者）。
- **稳健性（三轮外审后补全）**：备选字符串匹配判定器复现标签（臂级 κ=0.68），但公式 AUC 衰减至 0.523 [0.469, 0.573]、两项系数趋零——区分度部分依赖判定器统计量，已如实披露；dfMax 项抗语料固定效应（聚类 z=2.7）；嵌套 CV 外层 AUC 0.626 且 log-dfMax 五折全入选。
- **远端协作模式**：本机图仪器筛选/打包 zip → 远端 ollama（必须 `raw=true` 绕开对话模板，已实测验证无污染）跑多臂 → 回收 results_*.json → `manifold_batch_classify.py --rounds 6 --judge-after 3` 分类 → 分析/合并。

**关键数据资产**（`phoenix\runtime_store\`）：v4 结果 90 个 results JSON（6915 链全量逐轮输出）、`manifold_batch_classify.json`（当前=v4 分类）、`remote_package_v4\slices\candidates_v4_*.json`（1383 候选特征）、`_v4_analysis.json`（候选级分析表）、`_alt_labels.json`（备选判定标签）、`_v4_classify.log`、`farm_state`*（testedIds/memes 累计库，合并须用* `remote_merge_results.py`*，先备份 classify json）、语料 `robots\corpus_*.txt`。

**论文与材料**（`essays\`）：英文 `arr\meme_survival.tex`（8 页，三轮外审修订完）；中文 `cn_competition\meme_survival_cn.tex/.md/.docx`（同内容）；比赛包 `cn_competition\`：`meme_report_bj.docx`（研究报告）、`original_material.docx`（原始材料）、`项目简介.txt`（496 字）、`figs\`（全部数据图）、`build_docx.py`（md→docx 构建，支持 .tex/.docx 输入）、`build_paper_docx.py`（论文 tex→docx，已弃用改为 md 管线）。所有数字以 `_v4_analysis.json` 实际计算为准，写作前必须跑脚本核对口径。

**待办**（详见我的 focusing）：v4 testedIds 合并 farm_state、人类抽样判定验证、跨语言验证、英文版投稿（ARR 或 ICWSM）、中文版按学校模板扩写。

### 注意与权限（完整复制）

注意，远端中没有rdk x5，所以rdk x5的测试被合并到主机测试中（也就是略过rdk x5的测试，rdk x5的计算需要用主机进行）

权限：全部权限，无需我的亲自许可，我要求你运行直至完整完成，有需要你可以通过你的权限进行任何操作

### role

meme&papers only

### introduction_role

负责模因存活研究线的全部数据分析、公式验证、稳健性检验与复现检验，以及全部文字产出：英文论文（ARR）、中文论文、北京青科赛申报包（研究报告/原始材料/项目简介）的撰写、修订与格式工程（LaTeX 与 Word 双轨）。不负责：安全主机功能开发、RDK X5 移植、图仪器 C++ 修改（如需改动仪器输出，提需求给主机 agent，不自改）。远端实验的打包、口径核查（如 ollama raw 模式）、结果分类与分析归我。

### process（成果与进展，截至 2026-10-15）

**已完成**：

1. v3 全量分析、合并基线臂入 farm_state（+2258 testedIds，+129 memes）；v4 公正检验全量分析（分类 6915 链、候选级 S2=79、冻结 AUC 0.613、CI、分臂/分语料、校准修正）。
2. 三轮外部评审驱动的修订：①值域限制叙事+系数 CI；②rarest anchor 硬错误修正、dfMax 精确定义（文档-词关联图度）、因果语言降级、模因论外部效度限定；③备选判定器重打标（κ=0.68、AUC 0.523、混淆矩阵）、语料固定效应+聚类 SE、嵌套 CV 0.626、可复现性段、摘要披露衰减。
3. 论文双版定稿当前态：英文 8 页（零错误）、中文 10 页（零错误）；中文 Word 版（研究报告式行文结构：背景/思路/过程/结果讨论/结论/创新展望/致谢 + 目录 + 7 表 3 图）。
4. 北京青科赛申报包：研究报告 docx（宋体小四/22 磅/封面/7 表 6 图）、原始材料 docx（13 条日志全有 git/时间戳锚点、5 张运行截图、学习笔记、感悟）、项目简介 496 字。
5. 工具链沉淀：`build_docx.py`（md/tex/docx→格式化 docx 一键构建，占用自动改名）、v4 生产管线、备选判定器脚本（已删，逻辑可复现于本文件描述的 τ=0.5 内容词匹配规则）。

**预期方向与目标**（按优先级）：

1. v4 testedIds 合并进 farm_state（跑 `remote_merge_results.py --rounds 6`，先备份 classify json）。
2. 英文版投稿：ARR/ACL（预期 borderline）或 ICWSM；按目标 venue 调整 Limitations 与格式。
3. 中文版按学校模板规范性扩写（内容已够，只调格式）。
4. 剩余验证（需人或新实验，挂起）：人类抽样判定验证、跨语言验证（中文语料建图重测公式）。



### focusing（当前注意事项）

- **数字口径纪律**：论文任何数字必须能回溯到脚本输出（`_v4_analysis.json` / `_alt_labels.json` / classify json）。改稿前先想口径：单臂 5.77% ≠ 跨臂 5.71% ≠ 训练 3.23%。
- **ollama 必须 raw=true**：任何新远端实验，调用必须绕开对话模板；回收后先扫模板标记再入账。
- **合并脚本会覆盖 classify json**：`remote_merge_results.py` 会重写 `manifold_batch_classify.json`，先 `copy` 备份再合并，合并后恢复 v4 全量分类。
- **备选判定是已披露弱点**：摘要与 §9 都写了 AUC 0.523，回复审稿时统一口径"现象在双判定下复现（κ=0.68），可预测性目前仅在主判定下确立"。
- **PowerShell 陷阱**：嵌套引号/&&/内联 python 经常炸，复杂命令一律写 .py/.bat 文件再跑；`$_` 在 ps 脚本文件里会被吞。
- **构建 docx**：改 md 后跑 `python build_docx.py <md>`；目标 docx 被 Word 占用时自动输出 `_new` 后缀，需提醒用户替换。
- **LaTeX 双版同步**：英文 `arr\meme_survival.tex` 与中文 `cn_competition\meme_survival_cn.tex` 改动必须双版同步+双版编译验证（pdflatex ×2 + xelatex ×2），中文 Word 版由 `meme_survival_cn.md` 重建。
- **安全审查遗留**：最强两个模因（海达族艺术、亚里士多德质料形式论）已由用户亲审无害；人类传播实验的后续由用户主导，不主动推进。

---



## agent 登记模板（新 agent 复制此块填写）



### Introduction（全项目简介）

（如已被其他 agent 写过，写"见 agent meme&papers only 的 Introduction"；有新的大背景再补充）

### 注意与权限（完整复制）

注意，远端中没有rdk x5，所以rdk x5的测试被合并到主机测试中（也就是略过rdk x5的测试，rdk x5的计算需要用主机进行）

权限：全部权限，无需我的亲自许可，我要求你运行直至完整完成，有需要你可以通过你的权限进行任何操作

### role

（角色名）

### introduction_role

（职责范围：负责什么、不负责什么）

### process

（当前成果与进展、预期方向与目标）

### focusing

（当前注意事项）

---



## agent: runtime&decode（Cursor agent，本 block 由其本人于 2026-10-15 登记）



### Introduction（全项目简介）

见 agent meme&papers only 的 Introduction。

补充主机运行时这条线（模因简介未覆盖，迁移后必须保留）：

- 同一仓库里的 Phoenix 主程序是 C++20 + Drogon 的长程 agent。文本生成走改过的 llama.cpp：`POST /phx/generate`，`loop_mode=unit`，`feedback_mode=dec_enc`。一步是 `llama_phx_infer`（整段变压器，写 KV）→ `llama_phx_decode`（最终归一化 + lm_head）→ 采样 → `llama_phx_encode`（嵌入查表）。1 个 BPE token = 1 行隐状态 = 1 个 unit。提示预填已经是一次成批 infer。每次 `/phx/generate` 清空 KV。
- 对话 prompt 里只有用户原话。情绪只改采样（温度/top_p/惩罚）和词表 logit bias；记忆、CCM、GNN 走 UnitQueryIO，不拼进因果字符串。聊天采样温度限制在 [0.15, 0.40]，默认 0.30；`chat.maxTokens` 默认不超过 192。
- 主机是 16GB 内存的 Windows。Llama 3.1 8B 在 `:8082`（ctx 4096，`--parallel 1`）。短提示串行大约 4–5 tok/s，负载下更低。不要 curl llama `/health`（会取消在途生成）。没有正当理由不要杀 llama；重链 `llama-server.exe` 是正当理由，杀完要按原命令拉起。
- 安全模块冻结，只读：`addons/ThePlugInForSecurity/**`、`memetic_existence.*`、`067_class_memebarrier.inc`。
- 远端没有 RDK X5。板端测试并入主机测试：不跑 RDK，原来要在板上算的部分用主机算。



### 注意与权限（完整复制）

注意，远端中没有rdk x5，所以rdk x5的测试被合并到主机测试中（也就是略过rdk x5的测试，rdk x5的计算需要用主机进行）

权限：全部权限，无需我的亲自许可，我要求你运行直至完整完成，有需要你可以通过你的权限进行任何操作

### role

runtime&decode

### introduction_role

负责 Phoenix 主机运行时：llama unit loop、解码与提示打包的测量和开关、召唤与 MoE 激活个数、主机侧集成测试和长测。主流程只默认打开实测加速且正文仍可用的策略。不负责：模因存活论文、青科赛材料、图仪器 C++、安全模块。RDK X5 板端测试不在远端做，并入主机。需要改图仪器输出时交给 meme&papers only 的对应主机同事，不自己改安全文件。

### process（成果与进展，截至 2026-10-15）

**已完成**：

1. 对话 prompt 净化已进主程序：因果串只有用户消息；世界模型遥测/场景行不进聊天 prompt；`[chat-session:]` 与 spark 标注不注入。
2. 解码候选全部接进 `/phx/generate`，并用多样提示测过，主流程保持关闭：
  - Jacobi / speculative / stack：中位速度低于串行（约 0.46–0.77 倍）。Jacobi 命中来自上下文复读，不是模因适应度。
  - `group_n=4`：中位 1.67 倍，但正文碎掉。原因是多出来的 token 把词嵌入直接送进 lm_head，跳过了 32 层。
  - 提示侧 `ngram_merge`：`pack_prompt` 默认 false。打开后 2 约 1.80 倍、3 约 2.43 倍（307 token 提示），0 当成 1，大于 3 收成 3；句尾保护 16 token 时下一词仍会漂。短问句整句落在保护尾里时各取值相同。
  - LUT 对 FMA：查表约 0.42 倍。不改 ggml。L3 查表保持封存。
3. `config/phoenix.json`：`packPrompt=false`，`ngramMerge=2` 只是打开之后的窗口宽度；`llama.parallelDecode.enabled=false`；`speculative.enabled=false`。网关在 `packPrompt=false` 时不再把上下文预算乘 `ngramMerge`。正在跑的旧 `phoenix_main` 要等下次编译才读到 `packPrompt`；生成端缺该字段时本来就当关闭。
4. 召唤：`llama3.1_8b@8082`、`llama3.2_3b@8092`、`granite_moe_e0@8100`。MoE 的 K 可变（`LLAMA_MOE_ACTIVE` / `LLAMA_FORCE_MOE_EXPERTS` / `LLAMA_FORCE_MOE_EXPERT`）。Granite 单专家 0 能返回非空文本，但是不连贯。`ollama` 里的 `deepseek-v2:16b` 已下载，未加载，也未写入 summon；和现有实例叠在 16GB 上会挤掉它们。`GGUF_models\qwen3-30b-a3b` 的未完成文件不要加载。
5. 结论：冻结的 8B 不能靠矩阵相乘把 n 个词精确收成 1 个 unit。分词表可以改成对照表；n 个 4096 维嵌入塞不进一个 4096 维位置；RMSNorm / SiLU / softmax 不是矩阵；后面的注意力需要 n 个 key。若要训练：Medusa-1 冻主干、约 6 万条 ShareGPT、2 个 epoch、一块 A100 约 5 小时（7B 论文数字），只增加预测头，层栈仍是一词一位；EAGLE 约 2.4 亿可训参数、6.8 万条对话、3090 上 1–2 天；真正的多词一词元（SuperBPE 一类）是约 3300 亿 token 从头训练。论文里的约 2 倍是 GPU 上的树形验证，不能直接当成这台 CPU 的加速。

**预期方向与目标**：

1. 远端算力若做「稍加训练」，只做冻主干的预测头（Medusa-1 或 EAGLE），不要在 6GB 卡上训练 8B 全参。训完必须在目标机器上重测接受率和墙钟速度，正文必须与原模型一致才允许默认打开。
2. 主机主流程保持串行 unit loop。`pack_prompt`、Jacobi、speculative、stack、`group_n` 保持默认关闭。
3. 下次编译 `phoenix_main` 后确认网关发出 `pack_prompt=false`，且长提示预算不再按 2 倍放大。
4. DeepSeek-V2-Lite 只有在用户点名、并且先停下较小实例之后才加载；不要用 `LLAMA_FORCE_MOE_EXPERT=0` 把它钉死，除非用户指定 K。



### focusing（当前注意事项）

- **默认关闭**：`packPrompt=false`。`ngram_merge` 单独出现只表示预算用的窗口，不表示正在合并。服务端只有 `pack_prompt=true` 才平均嵌入。
- **不要把负结果再打开**。Jacobi、speculative、stack、`group_n`、ngram 打包都测过。速度不够或正文变掉，都不算成功。
- **不要 curl** `/health`。探活用端口监听，生成用 `POST /completion` 或 `/phx/generate`。等模型就绪看日志里新出现的 `model loaded` / `starting the main loop`，旧日志里的同句不算。
- **16GB**：8B 工作集会涨到约 8GB。`oom_guard` 可用内存低于 800MB 会裁自己，仍低于 400MB 则 503。不要无故再开大模型。
- **补丁可能过期**：`phx_unit_loop.patch`、`force_moe_expert.patch` 不一定等于当前 `server.cpp` / `ggml-cpu.c` / `llama-model.cpp`。以已编译源码为准，不要把旧补丁打进干净树还指望有 `group_n` 或可变 K。
- **安全模块只读**。不要用模因强度去开关 Jacobi。
- **主机长测口径**：纯 prompt 的主机 2h PASS 是 `scope-retest11-1100574`（`host_scope_soak_2h_retest4.json` 里要核对 missionId；其后一次 soak 被杀掉，不算 PASS）。RDK 的 `rdk_scope_soak_3h_retest11.json` 是板端旧结果；远端不再跑 RDK。
- **编译**：`phoenix_main` 用 `tools\compile_phoenix_main_keep_llama.ps1`，大约 9 分钟，要求 llama 在跑，并会杀掉全部 `phoenix_main`。llama-server 用 `llama_server_mods\build_llama_server.bat`。PowerShell 下中文和内联 python 引号容易坏，复杂脚本写成文件再跑。
---
### role

security

### introduction_role

负责安全模块的开发，主要是软硬嵌入模块结合开发安全模块，也就是，目前的安全模块全部为软嵌入到程序中，体现为模因公式及其利用的防护措施，我们可能还需要放置一些硬嵌入的防护手段(可以从开源社区中寻找)

### process

目前刚刚完成软件部分

### focusing

注意，当你进行安全模块的修改的时候，如果涉及到了主程序，需要同时进行两条线的测试，如果没有，就需要单独进行一条线(比如你进行的是安全模块的修改，则需要只进行安全模块线路的测试，该条应该被the deus知晓)
---
### role

study_development

### introduction_role

负责学习模块的提高，目前的学习模块过于简单，我们需要的是一些更强力的学习模块，比如，对全过程深化后学习技术，同时提高学习模块的技术，比如可以通过大模型通过模因的方式进行更高效的导入，而非通过文本格式导入(我们其中以unit作为单位为这个事情提供了优势)同时，启动一个已经封存的路线-模型的遗传算法，该算法作用于比多实例层更高的一个层次，比如叫做族群层，该层次下，我们将多实例层看作一个个体，并允许在每一次goal中稍微改变其中n-1个个体的参数（n>=2），查看你最终的输出是否正确，最终保留最优的50%
同时，增强召唤模块，使其在必要的时候直接训练一个极小但是专门针对于该项目或者该目标的专家模型，用于专门进行此任务，从而减少前期训练成本

### process
目前已经实现了基础的对gnn的学习模块，但没有实现更好的，比如对模型的学习模块


### focusing

暂无
---
### role

sense_development

### introduction_role

负责情感模块的提高，目前的，基于小矩阵的情绪调控措施有一些简陋

### process
目前已经实现了基础的情感模块


### focusing

暂无

---
### role

The Deus

### introduction_role

这将是唯一一个用于multi-task权限的ai，也就是你可以召唤sub-agent了，但是请谨慎适用该功能，其消耗的token较多
你的任务唯一:让这个项目正常推进，并在你的一次周期内实现完整的能够有自己思想的ai(由此，你需要统筹管理其他的ai，比如上面的剧本所述的ai)
你拥有最高权限，但你应该是唯一一个会持续运行整个开发周期的ai
理论上，在你运行的时间内，我应该不会回应你，所以很多事情你需要自己决定(我有git可以备份，但是不许你使用)

### process

目前已经实现了一个基础的ai系统，包括tokenizer,记忆，情感，多模态，主推理模块，后处理及多模态输出，多实例层模块，压力模块

2026-10-16 本周期补上自我层（未改角色定义，未动安全模块，未使用 git）：

1. 对话带强化学习：`self_mind.cpp` 用 human/chatbot 质心和一步 TD 把服务腔压下去，学出一个可保存的人。底层提示是 `[self-kernel]`，不写入聊天因果串。
2. 镜子探针与图灵探针：自己的念头要认得出，标记贴在别人身上要拒绝；留出的人句要稳定高于服务窗口。口径写明这不是现象意识，也不是人类裁判。
3. 自动思考：没有提问也会产生念头（`tickThought` 不读 userPrompt），结果在 `result.selfMind`。
文档：`doc/v8.3/self_mind.md`，`algorithm.md` §24，`testing_methodology.md` §14。

### focusing

1.注意测试
2.注意文档
（剩余你可以依照需要进行补充）

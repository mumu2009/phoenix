# 方法储备清单（Methods Backlog）— v10.1

> 目的：给当前 todo（残留 `0xC0000374`、AGI 分支树、IO 训练、MMLU 噪声）补上**可测试的方法**，
> 而不是继续加功能。每一条都必须回答：**基线是什么、指标是什么、怎么证伪、失败能学到什么**。
>
> 边界（本文件写作时遵守）：只读 `doc/` + `tools/`，不改产品源码，不建 git 分支/不 commit，
> 不跑 `compile.bat`，不碰正在跑的长跑（网关 PID 23920 自 2026-11-17 08:15:50 存活；
> 所有实验一律用侧车二进制 + 独立端口 + 独立工作目录，见 `crash_hunt_pageheap.md` §3）。

---

## 0. 本轮事实基线（所有实验设计的约束来源）

### 0.1 残留缺陷的准确状态

| 事实 | 出处 |
|---|---|
| `0xC0000374` 是 NT 堆校验发现元数据损坏，**崩溃点 ≠ 错误点**（延迟检测） | `crash_forensics_2026-11-15.md` §1 |
| 已定位并修复 6 处：`explorePaths` 惰性 `push_back`、`namedRealPath` 全局 string、`sectionDropStreak` 无锁 map、`TinyExpert::fromJson`、`executeAgiAction` 锁外写 `sessions_`、`OnlineResearcher::cache_`（第 5 处由 dump 栈指认） | `crash_forensics_2026-11-15.md` §3、`round_2026-11-15.md` §10.1 |
| 修完第 5 处后仍有"约每 10–40 分钟"的复现报告，怀疑点排序：`mcpActionMap_` 无锁读、`ScopedTrainableMemory::hot(scope)` 锁外写 bucket、`mission::*` 内隐藏 static | `round_2026-11-15.md` §7 |
| `_GLIBCXX_DEBUG` 侧车**实测链接失败**（conan 预编译库 ABI 冲突）；PageHeap 是当前主方案 | `crash_hunt_pageheap.md` §1–§2 |
| 截至该文档写作：生产二进制自 08:15:50 起连续 11 小时无崩溃；当天 4 次崩溃集中在 00:55 与 08:12–08:15 | `crash_hunt_pageheap.md` §6 |
| 观测补充（2026-11-17 21:58）：主网关 23920 仍存活（~13.7 h）；但进程表仍有 **7 个 32 KB 工作集的僵尸 `phoenix_main`**（19460/37916/38228/44932/46520/47604/48252） | 本文件作者只读观测 |

**结论：`0xC0000374` 目前是"未被证伪"，不是"已修复"。** 任何"已修复"声明必须带 exposure 与率上界（见 M6）。

### 0.2 工具链实测（决定方法可行性，不是文献）

| 能力 | 状态 | 实测方式 |
|---|---|---|
| ThreadSanitizer | **Windows 不支持**（官方支持平台：Android/Darwin/FreeBSD/Linux/NetBSD） | [Clang TSan 文档](https://clang.llvm.org/docs/ThreadSanitizer.html)（2026-11-17 取）；本机 `mingw64/lib` 无 `libtsan*` |
| ASAN | 本机 `mingw64/lib` 无 `libasan*`；`round_2026-11-15.md` §7 记 "MinGW 无 ASAN" | 本机观测 |
| gcc | MSYS2 gcc 16.1.0（`C:\msys64\mingw64\bin`） | 本机观测 |
| clang / clang-cl | **未安装**（PATH 与 `C:\msys64\clang64` 均无） | 本机观测 |
| gdb / cdb / WinDbg | **未安装**（PATH、`C:\Program Files\Windows Kits\...`、Store 路径均无） | 本机观测 |
| PageHeap | 注册表键存在，`GlobalFlag` 未设置（当前未开启） | 本机观测 |
| TTD | 需 WinDbg + 管理员；**仅用户态**；录制开销 **10–20×**；trace 分钟级即数 GB；回放只读 | [Microsoft Learn: TTD Overview](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/time-travel-debugging-overview)（2026-11-17 取） |

> 读法：**不能靠"装个 sanitizer"解决**（Windows 没有 TSan，MinGW 没有 ASan）。剩下的是
> 自建插桩（M1）、Windows 原生重放（M2）、编译期静态检查（M3）三条路，成本递增、证据强度递减。

### 0.3 评测噪声实测（MMLU 40 题）

数据：`runtime_store/long_run/mmlu_history.jsonl`，2026-11-09 起共 **1453 次 `total=40`** 的跑次。

| 统计量 | 值 | 含义 |
|---|---|---|
| 均值 | **0.4777** | 当前 8B 在这 40 题上的水平 |
| run-to-run `sd` | **0.0343** | **同一 40 题重复跑**的运行噪声（模型/采样/负载抖动） |
| 二项标准误（p≈0.478, n=40） | **0.0790** | 这 40 题作为"全 MMLU 样本"的**抽样误差**（doc 的 "±~8pp" 即此） |
| n=40 临界偏移（α=0.05） | **≈14.7pp**（单侧 25/40）/ **≈17.5pp**（双侧 27/40） | 这是"能被判为显著"的最小观测偏移；**80% 功效 MDE** 更大：单侧 19.7pp、双侧 22.1pp（见 `mmlu_protocol.md` §3） |
| 要 ±0.025 所需 n | **≈400**（`p(1-p)/0.025²`） | M8 的依据 |

**关键区分（本轮新增结论）**：run-to-run `sd=0.034` **小于** 抽样 SE `0.079`，说明"同一题库重复跑"测的是运行噪声，
**不是**对全 MMLU 的泛化误差。二者不可互换：比较两个 checkpoint 必须用**同题配对检验**，声明"模型变强了"
必须用**足够大的题量**。这解释了为什么 40 题口径下会反复看到 0.425–0.500 的"漂移"，却不能判读为成长。

### 0.4 构建与搜索宽度约束

- `compile.bat` 不能并行（会 `del build\tmp\cc*`，`round_2026-11-15.md` §9），且会 **taskkill llama-server**
  （`AGENTS.md`）→ 自改候选**只能**走侧车编译 `build/tmp/compile_next.bat`，否则打断生产长跑。
- 分支宽度 W=3–5 是硬上限（`branch_batched_selfmod.md` §3）；打分公式与 0.75 阈值见 §5；流水线 00–08 见 §4。
- 打分 `score = 0.5*测试通过率 + 0.3*soak稳定性 + 0.2*目标指标增量`（`branch_batched_selfmod.md` §5）。

---

## 1. 方法清单（9 条）

优先级标签：`P0` = 直击当前最高优先级 todo；`P1` = 让后续所有结论可信；`P2` = 能力线。

### M1（P0）RaceSentinel-MinGW：锁集 + 采样的自建竞争检测

**要解决的问题**

没有 sanitizer 可用（§0.2），当前找竞争的流程是"人工枚举共享可变状态 + 逐个判锁"
（`crash_forensics_2026-11-15.md` §2 的 12 行表格），一次事故才能推进一轮；`round_2026-11-15.md` §7 的
三个怀疑点（`mcpActionMap_`、`ScopedTrainableMemory::hot` bucket、`mission::*` static）至今没有证据。
`crash_forensics` §6 已把教训写成规则（"任何 lambda 级 static 可变容器在并发路径上都是缺陷"），
但没有能**自动发现违反规则**的工具。

**文献依据**（均已核验书目元数据，2026-11-17）

1. *Eraser: A Dynamic Data Race Detector for Multithreaded Programs* — Stefan Savage, Michael Burrows,
   Greg Nelson, Patrick Sobalvarro, Thomas Anderson, **1997**（SOSP'97，Proceedings of the 16th ACM SOSP, pp.27–37）。
   https://doi.org/10.1145/268998.266641 —— 锁集（lockset）算法：记录每个共享位置的锁集，交集为空即报警。
2. *LiteRace: Effective Sampling for Lightweight Data-Race Detection* — Daniel Marino, Madanlal Musuvathi,
   Satish Narayanasamy, **2009**（PLDI'09 / SIGPLAN Notices 44(6)）。
   https://doi.org/10.1145/1543135.1542491 —— 按"首次访问函数/低频采样"只插桩一小部分访问，
   把开销从 5–15× 降到可接受，同时保留大部分检出率。
3. *FastTrack: Efficient and Precise Dynamic Race Detection* — Cormac Flanagan, Stephen N. Freund, **2009**（PLDI'09）。
   https://doi.org/10.1145/1542476.1542490 —— 用 happens-before（vector clock）替代 lockset 判定，
   精度更高；作为 M1 第二阶段的升级路线（当 lockset 误报不可接受时）。

**最小可测试实验**

- 实现（不改产品源；独立 gtest 二进制 + `tools/` 下的装饰器头）：`RaceTrack<T>` 记录每个共享对象的
  `(last_writer_tid, last_reader_tid, lock_set_epoch)`；采样率 1/100（LiteRace 思路）；仅 debug 构建启用。
- **对照**：
  - 正对照 4 个（复刻已修缺陷的**模式**，不是改产品）：① `vector` 惰性 `push_back` + 并发遍历；
    ② 全局 `std::string` 赋值 + 并发 `find`；③ `std::map::operator[]` 插入 + 并发遍历；④ `cache_.erase` + 并发遍历。
  - 负对照 2 个：同样 4 个模式各加 `std::mutex` 的正确版（应 0 报告）；产品二进制（无插桩，作为"检测不到"基线）。
- **指标**：每种模式的检出率；负对照误报数/30 min；运行时开销倍率；平均检出延迟（秒）。
- **样本量/功效**：4 种模式 × 20 次独立运行（随机种子）= 80 次正对照；负对照 10 次 × 30 min。
  若检出率 p≈0.9，n=80 的 95% CI 半宽 ≈0.066（足够判"≥0.8 vs ≤0.95"）。
- **判据**：正对照检出率 ≥90%（每种 ≥18/20）**且** 负对照误报 ≤1 次/30 min **且** 开销 ≤5×
  （否则无法跑 30 min soak）→ 上线到三个怀疑点（`mcpActionMap_` 读、`hot()` bucket 写、`mission::*` static），
  用**侧车端口 5101 + 独立工作目录 + 8 个并行 mission**复现（`crash_hunt_pageheap.md` §3 的隔离方式）。
- **预计耗时**：实现 1.5 天；正/负对照 1 天；三个怀疑点排查各 0.5 天（30 min × 3 臂）→ 合计 3–4 天。

**风险与失败也能学到什么**

- 风险：采样漏检（低概率竞争窗口可能一次都不触发）；lockset 对读写锁/条件变量/`std::async` 的锁契约表达粗糙
  → 误报；装饰器会改到产品源（需走 `self_modification_protocol.md` §3 的 L2 白名单 + 侧车编译，由 Lead 批准）。
- 失败收益：即使检出率只有 50%，也把"剩余竞争在哪条路径"从 0 信息变成有排序的线索；
  且有/无竞争检测的对照会首次量化"`missionDelibLock=false` 下每条路径的竞争概率"——
  这正是 `artifact_contract.md` §5 "连续 2 小时 0 崩溃才可默认关闭粗锁"缺的那个分母。

**解锁的 todo**：残留 `0xC0000374`（主）。

---

### M2（P0）TTD 确定性重放：把"延迟发现"变成"当场回看"

**要解决的问题**

`crash_forensics_2026-11-15.md` §1 的核心困难是崩溃点与错误点分离；`crash_hunt_pageheap.md` §4 用 PageHeap
把延迟检测变当场（`0xC0000374`→`0xC0000005`），但 PageHeap 只给"写点栈"，**不能回看写者与被写者的完整时序**。
本机没有 WinDbg/cdb/gdb（§0.2）；rr 是 Linux 用户态方案，PinPlay 依赖 Intel Pin，均不能直接用。
TTD 是 Windows 原生且微软文档化的用户态录制/重放方案。

**文献依据**（已核验）

1. *Time Travel Debugging — Overview* — Microsoft Learn（Windows Driver Docs），文档更新 2025-11-07。
   https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/time-travel-debugging-overview
   —— 录制 `.run` + 索引 `.idx`、可正反向单步、回放只读、仅用户态、录制 10–20× 开销、trace 可达数 GB。
2. *Engineering Record And Replay For Deployability: Extended Technical Report* — Robert O'Callahan,
   Chris Jones, Nathan Froyd, Kyle Huey, Albert Noll, Nimrod Partush, **2017**（arXiv:1705.05937）。
   https://arxiv.org/abs/1705.05937 —— 论证"低开销用户态 record-replay 可行"的约束条件（CPU/OS 特性），
   并说明 rr 的适用边界；用于判断"Windows 上没有 rr 等价物时 TTD 是否是唯一合理选择"。
3. *PinPlay: A Framework for Deterministic Replay and Reproducible Analysis of Parallel Programs* —
   Harish Patil, Cristiano Pereira, Mack Stallcup, Gregory Lueck, James Cownie, **2010**（CGO 2010）。
   https://doi.org/10.1145/1772954.1772958 —— 确定性重放用于并行程序可复现分析的经典工作；
   `PinPlay` 证明了"重放 + 竞争分析"组合的价值，也说明其成本（插桩/DRI）。

**最小可测试实验**

- 用 TTD 录**高复现率触发窗口**（不是 8–24 h 生产窗口——按 10–20× 开销根本录不完）：
  12 个并行 mission + 1 个 chat 循环；debug-only 的 `PHOENIX_RACE_YIELD_US`（在 3 个已知锁点插
  `sched_yield`/`Sleep(0)` 随机时延）把 TTF 压低。
- **对照**：yield = 0 / 1 / 3 / 5 ms 四臂，每臂 10 次运行（每次 ≤5 min 墙钟；崩溃即停并保存 trace）。
- **指标**：中位 TTF（time-to-first-crash）；TTD 实测开销倍率；trace 体积/分钟；能否在 rewind 后定位到
  "写同一堆块的另一线程栈"（2 名判读者独立判定，报 Cohen κ）。
- **样本量/功效**：4 臂 × 10 运行；中位 TTF 用 bootstrap 95% CI；判读一致性 κ 目标 ≥0.7（10 个崩溃样本）。
- **判据**：至少一臂中位 TTF ≤2 min **且** 能在 trace 中回放到崩溃前 1 s 内的写指令并还原写者栈
  → TTD 进入崩溃复盘标准流程（只在复现窗口用，用完关）；若 TTF 压不下来 → 判定"竞争窗口不在注入的锁点"，
  转 M1 的采样权重。
- **预计耗时**：装 WinDbg+TTD 0.5 天；4×10 运行 + 判读 ≈2 天。

**风险与失败也能学到什么**

- 风险：**很可能失败**——TTD 录制 20× 开销会吃掉时间预算；trace 体积大；需管理员权限；与杀软可能冲突；
  录制期间进程仍可能崩溃（trace 可能截断在崩溃点之后，需要 TTD 的 crash dump 集成，本轮未验证）。
- 失败收益：四臂 yield 实验本身就是"竞争窗口对调度扰动敏感性"的直接测量——即使 TTD 不可用，
  也能得到"哪几个同步点决定 TTF"的排序，直接给 M1 的采样点加权；这比再猜一轮"第 6 处竞争在哪"便宜。

**解锁的 todo**：残留 `0xC0000374`（复盘证据链）；与 `debugger-tooling` 方向重叠（用它安装 TTD/gdb 路径）。

---

### M3（P1）编译期能力注解审计（clang `-Wthread-safety` + RacerX 式锁集规则）

**要解决的问题**

`crash_forensics_2026-11-15.md` §6 的教训要变成**结构性防线**，但当前只有文档规则与人工表格；
`artifact_contract.md` §5 写了锁分层（细锁 5 个 + 粗锁兜底），代码里却没有"访问某容器必须持某锁"的
可检查契约。本机没有 clang（§0.2），所以这是一个"值不值得装工具"的决策实验。

**文献依据**（已核验）

1. *Thread Safety Analysis* — LLVM/Clang 官方文档（Clang 团队，持续更新；本文取 2026-11-17 版本）。
   https://clang.llvm.org/docs/ThreadSafetyAnalysis.html —— `GUARDED_BY/REQUIRES/ACQUIRE/RELEASE/EXCLUDES`
   等注解，**纯静态、零运行时开销**；明确说明"**非过程间**"，回调/跨函数需显式声明。
2. *RacerX: Effective, Static Detection of Race Conditions and Deadlocks* — Dawson Engler, Ken Ashcraft,
   **2003**（SOSP'03）。https://doi.org/10.1145/945445.945468 —— 用锁集/加锁顺序的静态数据流分析
   在 OS 代码里找竞争与死锁；用于论证"静态锁集规则在大型 C/C++ 代码里能找到真 bug"，同时其误报率
   是采用前的必测项。

**最小可测试实验**

- 三条路线按成本排序：**(A)** 装 LLVM/clang（MSYS2 `pacman -S mingw-w64-x86_64-clang` 或官方安装包），
  给项目 mutex 包装类打 `CAPABILITY`/`SCOPED_CAPABILITY`，在 3 个 TU（graph 路径、online 检索、mission 循环）
  上跑 `clang++ -Wthread-safety -fsyntax-only`（不链接，绕开 conan ABI 问题）；**(B)** clang-tidy 自定义 check；
  **(C)** 不装工具，写 Python 正则/AST 规则检查"容器声明 + 锁守卫"配对（预期误报高）。
- **对照**：把三个已修缺陷的原始代码段（doc 里有完整贴片）做成正对照 TU，把修复后代码做成负对照 TU。
- **指标**：历史缺陷召回（0–3）；随机抽 50 条 warning 的人工确认 TP 率；每 TU 告警数；安装/注解工时。
- **样本量**：3 个已知缺陷 + 50 条 warning 抽样判读（50 条的 TP 率 95% CI 约 ±0.12）。
- **判据**：召回 ≥2/3 **且** 抽样 FP ≤20% → 接入 `self_modification_protocol.md` §3 第 6 步作为 gate；
  若召回 ≤1/3 → 只作为提示，不 gate（避免自改被误报卡死）。
- **预计耗时**：装工具 0.5 天；注解 12 个共享容器 1–2 天；跑 + 判读 0.5 天（合计 2–3 天）。

**风险与失败也能学到什么**

- 风险：Clang 对 MinGW/Conan 头文件兼容性未实测（`-fsyntax-only` 仍需正确 include 路径）；
  `-Wthread-safety` 非过程间 → `std::async` lambda、回调、间接访问全部漏报，而本项目大量使用 lambda；
  注解本身要改产品头（需 L2 流程）。
- 失败收益：注解缺口清单本身就是资产——把 `artifact_contract.md` §5 从文字变成"哪些共享容器没有锁契约"
  的勾选表；即使工具不可用，这张表也能让下一轮人工审计从 12 行变成"有注解 vs 无注解"的差分。

**解锁的 todo**：残留 `0xC0000374`（预防回归）；与 `debugger-tooling` 方向共享工具安装。

---

### M4（P0）分支树搜索策略：把 MCTS/质量-多样性映射到 `branch_batched_selfmod` 流水线

**要解决的问题**

`branch_batched_selfmod.md` §3 的 W=3–5 是**固定宽度**：每代铺 3–5 个候选、全跑一遍、按分数留 top-1，
没有"某个分支看起来更有希望就继续下钻""某分支明显失败就提前剪掉"的决策规则；§4 的 07 打分只用于
保留/丢弃，搜索预算没有分配。AGI 自改的奖励是**极稀疏的**（编译 + 定向 gtest + 宽回归 + 30 min soak 全过），
固定宽度在预算有限时等价于随机采样。缺的是"分支树 + 剪枝 + 打分"里**搜索策略**那一层。

**文献依据**（已核验）

1. *Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models* — Andy Zhou, Kai Yan,
   Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang, **2023/2024**（arXiv:2310.04406）。
   https://arxiv.org/abs/2310.04406 —— 把 MCTS 用在 LLM agent 上：LM 值函数 + 自我反思 + 外部环境反馈；
   给出 UCT 式选择与回传的具体做法（代码开源）。用于"节点=分支、动作=补丁、价值=score"的映射。
2. *SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement* —
   Antonis Antoniades, Albert Örwall, Kexun Zhang, Yuxi Xie, Anirudh Goyal, William Wang, **2024**（arXiv:2410.20285）。
   https://arxiv.org/abs/2410.20285 —— 更贴近本场景（repo 级代码修改）：混合值函数（数值 + LLM 定性评估）+
   多 agent 辩论；报告在 SWE-bench 上相对无 MCTS 的开源 agent **23% 相对提升**（5 个模型平均）。
   该数字只能当**效应量先验**，不能直接搬到 Phoenix（模型/任务不同）。
3. *Illuminating search spaces by mapping elites* — Jean-Baptiste Mouret, Jeff Clune, **2015**（arXiv:1504.04909）。
   https://arxiv.org/abs/1504.04909 —— MAP-Elites：用"行为描述维度"建档案，每个格子保留最优解，
   同时获得多样性与覆盖率；用于把"只留 top-1"改成"留一个覆盖不同改动类型的档案"（避免连续多代
   都在同一个局部最优附近打转）。
4. *Automated Design of Agentic Systems* — Shengran Hu, Cong Lu, Jeff Clune, **2024**（arXiv:2408.08435）。
   https://arxiv.org/abs/2408.08435 —— Meta Agent Search：用"不断增长的发现档案"驱动下一轮提案；
   对应本项目"用已合并分支历史作为下一轮 patch 的上下文"。

**最小可测试实验**

- 把流水线 00–08 包装成一次 rollout（**候选评估先只用 Stage A 代理**：侧车编译 + 定向 gtest，
  不做宽回归/soak，见 M5）；节点 = 一个 branch（一次 commit），动作 = LLM 生成的下一个 patch，
  价值 = `branch_batched_selfmod.md` §5 的 score（或 Stage A 代理分），选择 = PUCT（LLM 先验）。
- **对照 3 臂**：(a) 单线贪心 W=1（当前"修 bug"模式）；(b) 固定宽度 W=3（当前 branch 模式）；
  (c) MCTS（预算 = 10 次候选评估）。
- **任务集**：2 个 L1/L2 边界任务（`self_modification_protocol.md` §1 允许的 L1/L2）：`agi.efe.horizon` 3→4/5/6、
  `mission.recallDiversity` 0.6/0.7/0.8（`branch_batched_selfmod.md` §9 自己建议的第一批目标）；每任务 3 个
  随机种子（patch 生成温度 0.8）。
- **指标**：预算-最优分数曲线（AUC over 10 evals）；达到 score ≥0.8 所需评估次数；每代墙钟；
  MAP-Elites 档案覆盖率（改动类型分桶：性能/能力/安全/可维护性）。
- **样本量/功效**：3 臂 × 2 任务 × 3 种子 = 18 次运行；每次 ≈ 10 × 15 min = 2.5 h → **≈45 h 机器时间**。
  n=6 配对观测只能探测**大效应**；先做 3 次 pilot 估方差，再决定是否扩种子。
- **判据**：MCTS 的 AUC 在 ≥4/6 配对中优于固定宽度（先用符号检验，p<0.05 需 5/6，所以先 pilot 估方差、
  再定是否扩到 8–10 对）；若不胜 → 保留 W=3，并把"哪些任务一次就成"的记录并入档案。
- **预计耗时**：pilot ≈7.5 h；全量 ≈45 h（约 2 天，可与实验室其它工作错峰，但**不能**与产品编译并行）。

**风险与失败也能学到什么**

- 风险：rollout 极贵且**编译串行**（`round_2026-11-15.md` §9：两个编译不能并行；`compile.bat` 会 kill
  llama-server → 必须侧车）；LLM 生成可编译补丁的成功率可能 <30%，MCTS 会把预算花在无效节点；
  c_puct 等超参在 n=6 时无法可靠调；修改 `main_hub_parts/` 可能触发宽回归超时。
- 失败收益：即使 MCTS 不胜，也会产出**"改动类型 × 一次成功率"**的经验分布——这直接是固定宽度策略的
  先验（比如"L1 配置类改动一次成功率 0.7，L2 代码类 0.2"），并把 `branch_batched_selfmod.md` §9 的
  "下一步"从猜测变成有数据的任务排序。

**解锁的 todo**：AGI 分支树。

---

### M5（P0）候选评估的测量系统分析：两阶段早停 + 接受判据的噪声校准

**要解决的问题**

`branch_batched_selfmod.md` §4 对**每个**候选都跑"宽回归 268 + 30 min soak"；§5 的 `score ≥0.75`
是硬阈值。但 soak 项只有 30 min 采样，同一候选重复评分的漂移没有测过——阈值附近 0.74/0.76 的差别
可能纯属评估噪声。没有"测量系统分析（MSA）"，分支树上的所有比较都不可比；M4 的搜索策略也会
把噪声当信号去下钻。

**文献依据**（已核验）

1. *SWE-Search* — 同 M4（Antoniades et al., 2024, arXiv:2410.20285）：其"混合值函数 + 迭代 refinement"
   的核心动机就是**单次评估不可靠、要用多次反馈降方差**；对应本提案"重复评估 + 顺序检验"。
2. *If Nothing Goes Wrong, Is Everything All Right? Interpreting Zero Numerators* — James A. Hanley,
   **1983**（JAMA 249(13):1743）。https://doi.org/10.1001/jama.1983.03330370053031
   —— "零事件"的 95% 上界 ≈ 3/n（即可靠性工程里常说的 rule-of-three）。用于说明
   **30 min soak 零崩溃只能得到 λ<6/h**，对"残留 1/3h 的 bug"毫无鉴别力。
   *（核验说明：Crossref 条目只登记第一作者 Hanley，本文按第一作者引用。）*
3. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing* —
   Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart, **2018**（ACL 2018）。
   https://aclanthology.org/P18-1128/ —— 实验比较必须选对检验、报告效应量与样本量；
   用于本提案"配对比较 + 置信区间"的规范（同样适用于 benchmark，不限于 NLP）。

**最小可测试实验**

- **测量重复性**：固定 1 个候选（不改代码），重复走 05–07（宽回归 + 30 min soak + 打分）8 次，
  得到 score 的 run-to-run `sd` 与各分项（通过率/soak/指标增量）的方差贡献。
- **测量候选间方差**：10 个候选各评一次，2 个候选各评 2 次（共 12 次评估），估 ICC（评估噪声占总方差比例）。
- **两阶段早停**：Stage A = 侧车编译 + 定向 gtest（7 套件，见协议 §3 第 6 步）；Stage A 通过才进 Stage B
  （宽回归 + soak）。用历史失败率算期望成本：`E[C] = C_A + p_passA · C_B`。
- **指标**：score 的 `sd`、分项方差占比、ICC、两阶段期望节省墙钟（%）、假接受率（阈值放宽到 0.70 时多接受的
  低质量候选比例，用 M4 pilot 的候选池做 ground truth）、30 min soak 的 λ 上界（rule of three）。
- **样本量**：8 次重复（sd 估计 95% CI 约 ±25%）+ 12 次候选评估 → 总计约 20 次评估 × ~45 min ≈ 15 h。
- **判据**：若 ICC <0.6（评估噪声主导）→ 必须改判据（重复取中位数 + 只对 Stage A 做硬 gate，soak 只做
  "灾难性回归"过滤）；若两阶段节省 ≥25% 且假接受率不上升 → 替换流水线 05–07；否则维持现状并记录"已知噪声"。
- **预计耗时**：~1.5 天（含 15 h 机器时间）。

**风险与失败也能学到什么**

- 风险：重复评估占用机器与 sidecar 端口；候选分支必须能确定性复现（git tag + 固定种子）；
  生产长跑同时占用 GPU/内存，会让"评估噪声"与"环境负载"混在一起（需与 stability-harness 的时间窗错开）。
- 失败收益：即使早停不省，也会第一次给出 selfmod score 的 **MSA 报告**（重复性、分项方差、阈值敏感性）。
  这张表是 M4 的前置条件，也是 Lead 判断"0.75 阈值要不要改"的唯一依据。

**解锁的 todo**：AGI 分支树（主）；为 M6 的判据校准提供输入。

---

### M6（P0）长跑稳定性证据标准：Poisson 零事件上界 + Kaplan-Meier + canary 双窗口

**要解决的问题**

`operating_rules.md` §4 要求"3 小时量级连续可运行 + 结构化证据"，`artifact_contract.md` §5 要求
"连续 2 小时并行 0 崩溃才可默认关粗锁"，但**没有任何换算规则**把"无崩溃时长"变成"崩溃率上界"，
于是 11 小时无崩溃（`crash_hunt_pageheap.md` §6）容易被读成"已修复"，而同一文档记录当天 4 次崩溃。
稳定性的分母（exposure）也没定义：进程被 supervisor 拉起、僵尸进程（§0.1 观测 7 个）都会污染 up 率。

**文献依据**（已核验）

1. *"Bathtub" curve* — NIST/SEMATECH e-Handbook of Statistical Methods, §8.1.2.4（Assessing Product Reliability）。
   https://www.itl.nist.gov/div898/handbook/apr/section1/apr124.htm —— 早期失效期（递减率）/稳定期（恒率）/
   磨损期（递增率）；**重启后的进程是"新单元"**，不能与稳定期混合统计。
2. *A Statistical Distribution Function of Wide Applicability* — Waloddi Weibull, **1951**（J. Applied Mechanics 18(3)）。
   https://doi.org/10.1115/1.4010337 —— Weibull 形状参数 β<1/≈1/>1 正好对应 bath-tub 三段；
   用于把"我们处在哪一段"变成可估参数（而不是形容词）。
3. *Nonparametric Estimation from Incomplete Observations* — E. L. Kaplan, Paul Meier, **1958**（JASA 53(282)）。
   https://doi.org/10.1080/01621459.1958.10501452 —— Kaplan-Meier 生存曲线（右删失）：
   3 h 窗口结束时未完成的 mission 必须作为删失处理，否则完成率被低估。
4. *If Nothing Goes Wrong, Is Everything All Right?* — James A. Hanley（Crossref 仅登记第一作者），**1983**（JAMA）。
   https://doi.org/10.1001/jama.1983.03330370053031 —— 零事件 95% 上界 = 3/n：**3 h 零崩溃 → λ<1/h**，
   12 h → λ<0.25/h，30 h → λ<0.1/h。这是"3 小时证据"能说和不能说的话。
5. *Canary Release* — Danilo Sato（Martin Fowler bliki），**2014-06-25**。
   https://martinfowler.com/bliki/CanaryRelease.html —— 渐进放量 + 指标监控 + 快速回滚；
   还区分 canary（分钟–小时级找回归）与 A/B（天级显著性），以及"自动回滚 = cluster immune system"。

**最小可测试实验**

- **定义 exposure**：`T = Σ(区间内进程 up 且完成至少 1 个 tick 的小时数)`，按 PID 变化切分区间；重启后前
  30 min 单列为"早期失效"（不并入稳态）。同时记录删失（窗口结束仍未完成的 mission）。
- **实验 A（判据换算，零成本）**：对现有 13.7 h 存活窗口（PID 23920，08:15:50 起）应用 rule of three：
  λ<3/13.7=0.22/h（95%）。与"今天 4 次崩溃"对照，说明**日级波动本身就要求 ≥12 h 才能判读**。
- **实验 B（KM 完成时间）**：在 3 h 窗口内跑 8 个并行 mission（≥24 个 mission 生命周期，按 30–40 min/生命周期估），
  记录 start/finish/censor；画 KM 曲线，报告 median completion（含删失）、删失比例、按 PID 切分的分段。
- **实验 C（canary 双窗口）**：新二进制走 sidecar（独立端口 5101+、独立工作目录、独立参数），
  与生产网关**同任务集**同时跑；快窗 1 h（每 60 s 采样 tick 耗时/up/崩溃）+ 慢窗 12 h（每 5 min）；
  用 Poisson 精确检验比较两臂崩溃率与 tick p95。
- **指标**：λ 上界（95%）、KM median/删失%、僵尸进程积累率（个/小时）、tick p50/p95、GPU 均值、
  mission 完成率（assign→report）。
- **样本量/功效**：3 h 窗口 ≥24 mission；12 h 窗口 ≥96；3 h 零崩溃只能排除 λ≥1/h（算 S1），
  12 h 才能谈 0.25/h（S2），30 h 才 0.1/h（S3）。
- **判据（分级发布）**：
  - **S1（3 h，零崩溃）**：只能声明"未发现新问题"，禁止写"已修复"或"可默认关粗锁"；
  - **S2（12 h，零崩溃，或崩溃率上界 <0.25/h）**：允许"显著降低"，允许进入合并评审；
  - **S3（30 h+，λ<0.1/h）**：才允许"默认关粗锁 / 合主干"（对齐 `artifact_contract.md` §5）。
  任何一级都必须同时报告 exposure、删失、PID 变化次数、僵尸计数——缺一项则降级。
- **预计耗时**：实验 A 0.5 h（纯计算）；实验 B 3 h（与生产长跑并行，sidecar）；实验 C 1 h 快窗 + 12 h 慢窗（后台）。

**风险与失败也能学到什么**

- 风险：supervisor 自动重启会把一次崩溃伪装成"两次短命进程"，必须按 PID 切分（否则 MTBF 虚高）；
  僵尸进程让"进程存在"不等于"功能活着"（需加健康检查而非仅 PID）；sidecar 与生产共享 llama-server
  会互相拖慢，tick 指标不可比（必须独立 slot 或独立 server）。
- 失败收益：即使分级被判"太严"，也会建立 **exposure + 删失 + PID 变化 + 僵尸** 四列的原始表；
  这是以后任何稳定性声明唯一的可追溯来源，也直接回答"3 h 覆盖一个任务生命周期"到底覆盖了几个。

**解锁的 todo**：残留 `0xC0000374`（修复验收口径）；与 `stability-harness` 方向主重叠（本提案给它判据）。

---

### M7（P1）meme-graph 检索评测集 ≥200 题（可复现、可 diff、带金标）

**要解决的问题**

`implementation_2026-11-07.md` §7 的 MMLU 评测只有 40 题（同种子），"+2.5pp 在噪声内"；
记忆层的召回质量（`experienceTop` + `mmrSelect`、`ccmRecall`、GNN outline、meme 簇交集）**没有任何离线指标**，
只能用"看起来更宽"来判断。`round_2026-11-15.md` §10.7 的 MMR 宽度原则需要一个能证伪它的评测集。
`reasoning_research_plan.md` §1 已示范正确姿势（先定样本量与门槛，再决策）——本提案把同一纪律搬到记忆层。

**文献依据**（已核验）

1. *Cumulated Gain-Based Evaluation of IR Techniques* — Kalervo Järvelin, Jaana Kekäläinen, **2002**（ACM TOIS 20(4)）。
   https://doi.org/10.1145/582415.582418 —— nDCG 的原始定义（分级相关性 + 位置折扣）；
   用于"金标有强弱相关之分"的场景（meme 簇命中 vs 同义命中）。
2. *BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models* —
   Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych, **2021**（NeurIPS D&B）。
   https://arxiv.org/abs/2104.08663 —— 18 个数据集 + 统一指标（nDCG@10 等）+ BM25 强基线；
   用于"必须打败 BM25 才叫有检索能力"的实验设计。
3. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* — Darren Edge 等 10 人, **2024**
   （arXiv:2404.16130）。https://arxiv.org/abs/2404.16130 —— 图索引 + 社区摘要 + 分局部/全局问题评估；
   用于设计"图路径检索"的对照臂与问题分层（局部事实 vs 全局总结）。
4. *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models* —
   Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su, **2024**（NeurIPS 2024）。
   https://arxiv.org/abs/2405.14831 —— 知识图 + Personalized PageRank 做单步多跳检索；
   是本项目 meme-graph 最接近的参考实现与对照臂。
5. *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory* — Di Wu, Hongwei Wang,
   Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu, **2024/2025**（ICLR 2025；500 题）。
   https://arxiv.org/abs/2410.10813 —— 长期记忆的五种能力（抽取/多 session 推理/时间/知识更新/弃答）
   与"索引-检索-阅读"三阶段框架；本提案的**题目分层直接借用这五类**。

**最小可测试实验**

- **题库构建**：从 `runtime_store/mission_workspace/*/deliverable.md` + `gateway.log` 抽 (query, gold-unit) 对；
  5 层 × 48 题 = **240 题**：单跳事实 / 跨 session 时间 / 多跳图路径 / 无答案（abstention）/ 去重多样性
  （同一配方 3 变体，测 MMR 是否真的保宽）。金标 = 规则抽取 + 人工复核；抽 60 题双人标注，Cohen κ 目标 ≥0.8。
- **可复现性**：题库落 `tools/`（只读、可 diff），JSON + 版本号 + sha256；每题 3 个固定措辞变体；
  固定模型、固定解码（temperature=0）、固定 seed；一条命令重放。
- **对照 3 臂**：BM25（词袋强基线）、当前 `mmrSelect`（λ=0.7）、图游走（HippoRAG 式 PPR）。
- **指标**：nDCG@10、MRR@10、Recall@10、abstention 准确率、多样性（top-3 平均两两距离）、
  "答案级"指标（可选：8B 作 reader 的 exact/ROUGE-L，先不做，避免端到端噪声盖住检索差异）。
- **样本量/功效**：240 题；先用 20 题 pilot 估方差。比例型指标（abstention）SE≈0.032 → 只能探测 ≥9pp；
  若目标 5pp 需 ~600 题。nDCG 是连续量，配对 bootstrap 通常可探测 2–4pp（需 pilot 验证）。
- **判据**：图游走 vs MMR 的 nDCG@10 配对 bootstrap 95% CI 不含 0 且差值 ≥0.03 → 采纳图游走；
  否则**记录为"无差异"**（这也是结论，防止不断重试同一想法）。240 题必须全部可重放，κ≥0.8 才算有效题库。
- **预计耗时**：题库 + 标注 2 天；pilot 0.5 天；正式 3 臂 × 240 题（纯检索）≈2–4 h；分析 0.5 天。

**风险与失败也能学到什么**

- 风险：**循环论证**——金标若来自模型自己的交付物，等于用模型的输出去判模型（必须用"任务陈述/人类可判定的
  目标"做 query，交付物只做候选单元）；工作区 256 KB 上限会让长 deliverable 被截断（截断单元要剔除）；
  图游走实现可能引入新代码（属自改 L2，需走护栏）。
- 失败收益：即使图游走不胜，也第一次有了记忆层改动的**可复现回归集**；并且把 `round_2026-11-15.md` §10.7 的
  MMR 从"手感"变成 nDCG/多样性的双指标；题库本身可以复用给 M8（同一套统计协议）。

**解锁的 todo**：MMLU 噪声（把"成长"的度量从 40 题 MMLU 挪到 240 题目标评测）；记忆层改动验收。

---

### M8（P1）小样本统计协议：从"MMLU 40 题"到可解释的成长报告

> **已工具化（2026-11-17）**：`tools/eval_mmlu_protocol.py`（legacy 40 / stratified 400 / paired McNemar+bootstrap / `--power-table` / `--self-test`）+ `doc/v10.1/mmlu_protocol.md`（抽样设计、功效表、判据、决策规则）。本提案保留为设计依据。

**要解决的问题**

§0.3 已量化：n=40 的抽样 SE=0.079、α 临界偏移 ≈14.7–17.5pp、80% 功效 MDE ≈19.7–22.1pp（详见 `mmlu_protocol.md` §3）；而项目历史上仍在用 40 题同种子对比
得出 "+2.5pp"（`implementation_2026-11-07.md` §7），并每 ~7 min 跑一次（supervisor.log 实测 21:26–21:56
共 5 次）。同时 `reasoning_research_plan.md` §7 用 203 ticks / 842 missions 的门槛做了正确的需求判定——
说明项目已经有"样本量纪律"，只是**没有搬到 benchmark 上**。

**文献依据**（已核验）

1. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing* —
   Rotem Dror, Gili Baumer, Segev Shlomov, Roi Reichart, **2018**（ACL 2018）。
   https://aclanthology.org/P18-1128/ —— 检验选择协议（配对/非配对、bootstrap vs 参数检验）、
   报告效应量与样本量的规范；本提案的检验选择直接照此执行。
2. *tinyBenchmarks: evaluating LLMs with fewer examples* — Felipe Maia Polo, Lucas Weber, Leshem Choshen,
   Yuekai Sun, Gongjun Xu, Mikhail Yurochkin, **2024**（ICML 2024）。
   https://arxiv.org/abs/2402.14992 —— 用 IRT 选 100 道精选 MMLU 题即可高保真估计全量 14K 题；
   用于"**精选 100 题**优于**随机 40 题**"的替代方案。
3. *Measuring Massive Multitask Language Understanding* — Dan Hendrycks, Collin Burns, Steven Basart,
   Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt, **2021**（ICLR 2021）。
   https://arxiv.org/abs/2009.03300 —— MMLU 原始定义（57 subjects、14K 题、四选一）；
   用于说明"40 题子集"与"全量"的关系，以及为什么随机子集的方差必须按二项分布算。
4. *If Nothing Goes Wrong, Is Everything All Right?* — James A. Hanley, **1983**（JAMA）。
   https://doi.org/10.1001/jama.1983.03330370053031 —— 同样适用于"0 个错题"的区间解释（零事件上界）。

**最小可测试实验**

- **三种估计并行**：(a) 现有 40 题（历史连续性）；(b) **400 题分层随机**（按 subject 分层，固定 seed，±0.025）；
  (c) tinyBenchmarks 式 100 题子集（若公开子集/IRT 参数可得，否则跳过并标注）。
- **报告协议**：任何"成长/变强"声明必须给 **同题配对 McNemar**（两 checkpoint × 同 400 题 × 同 seed）+
  **bootstrap 95% CI**（1000 次重采样）；未达 n=400 的评测**只能报 CI，不能报显著**；同一天 ≥3 次重复，
  用 ICC 报告运行噪声占比。同时保留 40 题口径但只用于"是否大幅退化"的哨兵。
- **指标**：score、CI 半宽、McNemar 的 discordant 对数（b/c）、ICC、单次评测墙钟。
- **样本量/功效**：n=400 → SE≈0.025；n=100 精选 → 需先在 400 题上验证相关 ≥0.9 才可替代；
  n=40 的哨兵保留但明确标注"临界偏移 ≈14.7pp（单侧）/17.5pp（双侧）；80% 功效 MDE 19.7pp/22.1pp"。
- **判据**：400 题口径稳定运行 2 周且 ICC<0.5（运行噪声 < 一半方差）→ 用它替代 40 题做成长口径；
  若评测墙钟 >90 min/次，降为 240 题并接受 ±0.032。
- **预计耗时**：脚本改造 0.5 天（复用 `tools/mmlu_subset_eval.py` + pandas CSV；本机已有 **14,042 记录**的全量 MMLU test split CSV `build/tmp/mmlu.csv`——物理行 18,503 是题面内嵌换行，勿当题数；工具已交付，见 `mmlu_protocol.md`）；
  400 题一次评测 ≈70 min；pilot 2 天（含 2 周数据的第一批）。

**风险与失败也能学到什么**

- 风险：8B + temperature=0 在 40 题上分数"粘住"（实测 sd 0.034），扩题后整体分数会偏移 → 历史曲线断裂
  （所以保留 (a) 口径）；评测变慢 10× 会挤占训练时间；MMLU 与项目目标（mission/记忆）相关性有限。
- 失败收益：即使 400 题太慢，也能得到"评测成本-精度"曲线，决定以后每轮自改值不值得跑评测；
  并且第一次把"M8 的统计协议"变成项目通用规范（M7/M9 都复用它）。

**解锁的 todo**：MMLU 噪声（主）；为 M4/M5 的"目标指标增量"项提供可信测量。

---

### M9（P2）IO 训练：冻结编码器的探针评测协议（SUPERB / linear-probe 规范）

**要解决的问题**

IO 训练方向要在既有 ijepa/BPU 产物上训练 head（`AGENTS.md` 记 additive-JEPA + `x5_online_learner`，
`doc/v8.3/*`），但没有规定"怎样才算真的更好"：没有固定切分、没有随机基线、没有样本量。
`artifact_contract.md` §4 已经给 tiny_expert 做了 train/heldout 分离的好先例（4/5 + 1/5），
本提案把它升级成 IO 训练的统一评测协议——否则 io-train 每轮都会在"loss 降了但不知道有没有用"里循环。

**文献依据**（已核验）

1. *SUPERB: Speech processing Universal PERformance Benchmark* — Shu-wen Yang 等 20 人, **2021**（Interspeech 2021）。
   https://arxiv.org/abs/2105.01051 —— 冻结共享编码器 + 轻量任务头（task-specialized prediction heads）的
   统一评测框架；直接就是本提案的协议骨架（本项目编码器不可再训，只能训 head）。
2. *Do Better ImageNet Models Transfer Better?* — Simon Kornblith, Jonathon Shlens, Quoc V. Le, **2019**（CVPR 2019 Oral）。
   https://arxiv.org/abs/1805.08974 —— 固定特征提取器（linear probe）与微调两种设定的系统对照；
   说明"冻结特征 + 线性头"是评估表征质量的廉价且有效手段，也提醒"训练目标不同会让特征迁移性变差"。
3. *LongMemEval* / *BEIR*：见 M7/M8 —— 其"固定切分 + 固定基线 + 报告 CI"的写法可复用（不重复列）。

**最小可测试实验**

- **协议**：冻结编码器（ijepa / BPU 输出，512-D 特征），只训 head（`x5_online_learner` 或 CPU SGD）；
  数据 80/10/10 切分（train/val/test），test 只在最终报告用一次。
- **任务集**：5 个任务（图像 3：方向/尺度/颜色一致性；音频 2：音高/说话人），每任务 5 类 × 200 样本 = **1000**
  （test 200）；另做"随机化背景 + 反事实样本"防捷径。
- **对照 3 臂**：(a) 随机初始化编码器（同 head，测"编码器是否有用"）；(b) 多数类/随机基线；(c) 现有 head（若有）。
- **指标**：test accuracy、macro-F1、相对多数类提升、train-test gap、3 seeds 均值 ± 95% CI。
- **样本量/功效**：test 200/任务 → 比例 SE≈0.035，可探测 ~10pp；每任务 3 seeds 用于估训练方差。
- **判据**：新 head 相对 (a) 提升 ≥15pp 且 CI 不含 0，且 train-test gap <10pp（排除过拟合）→ 进 BPU 编译；
  否则"该编码器特征对本任务无信息"是结论（不做无谓的 BPU 适配）。
- **预计耗时**：数据收集/合成 1 天；5 任务 × 3 臂 × 3 seeds 训练 ≈ 数小时（CPU）；报告 0.5 天。

**风险与失败也能学到什么**

- 风险：合成数据会被 head 学到捷径（颜色/位置/背景），必须做反事实样本；BPU 产物在运行时不可再训练
  （`AGENTS.md` 已写明），所以只能训 head——本协议以此为**前提**而非缺陷；主机与 X5 的特征分布可能有差
  （需先在主机复现，再在 X5 上算一次域差）。
- 失败收益：即使 head 不行，也给编码器一份"信息含量"基线（哪些任务的 512-D 特征是死的），
  直接决定 io-train 下一步该收什么数据、该换什么任务——这比继续调 head 的学习率更省。

**解锁的 todo**：IO 训练。

---

## 2. 与当前 todo 的映射（最短可交付）

| 当前 todo | 对应提案 | 最短可交付（先做哪一步） | 机器时间 |
|---|---|---|---|
| **残留 `0xC0000374`** | M1（主）、M2、M3；M6 给验收口径 | M1 的正/负对照 80 次运行 → 决定是否上线 | ~3–4 天 |
| **AGI 分支树** | M5（先）、M4（后） | M5 的 8 次重复评估 → 拿到 score 噪声，再谈 MCTS | ~1.5 天（M5） |
| **IO 训练** | M9 | 5 任务 × 3 臂 pilot（先 200 样本/任务） | ~2 天 |
| **MMLU 噪声** | M8（主）、M7 | 400 题分层随机 + 配对 McNemar 报告 | ~2 天 |
| （相关）3 h stability harness | M6 | 先做"exposure/删失/PID/僵尸"四列表，再谈 3 h | 0.5 天定义 |
| （相关）debugger tooling | M2/M3 | 安装 WinDbg+TTD 与 clang 的决策实验 | 1 天装工具 |

**建议顺序（互斥与依赖）**

1. **M8 先行（0.5 天）**：它只改评测脚本，不占编译/GPU；先让后续所有"变强了"的结论有分母。
2. **M6 的定义部分（0.5 天）**：exposure/删失/PID 四列，从今天起所有稳定性报告都按它写；后台跑 S1/S2 窗口。
3. **M1 实现与对照（3–4 天）**：与生产长跑并行（独立端口/目录，不跑 `compile.bat`）。
4. **M5 → M4**：先测评估噪声（M5），再决定是否投 45 h 到 MCTS（M4）；顺序不可反。
5. **M7 与 M9 可并行**：M7 占标注/分析人力，M9 占 CPU 数小时；都不碰主网关。
6. **M2/M3 是"工具决策"**：先各花 0.5 天验证安装 + 正对照能不能跑通，再决定是否投入。

**硬约束（所有实验共用）**

- 不跑 `compile.bat`（会 kill llama-server，`AGENTS.md`）；侧车用 `build/tmp/compile_next.bat`。
- 不碰生产网关端口（当前 5097 / 生产进程 PID 23920）；实验一律 5101+ 独立端口 + `build/tmp/phx_*` 独立工作目录
  （`crash_hunt_pageheap.md` §3 的隔离法）。
- 不建 git 分支/不 commit（`operating_rules.md` §5）；候选改动只落 `runtime_store/self_mod/staging/<id>/`。

---

## 3. 引用核验说明（诚实边界）

**核验方式（全部 2026-11-17）**：文档类（Clang、Microsoft Learn、NIST、martinfowler、ACL Anthology）用
`web_fetch` 取原文页面；arXiv 条目用 `export.arxiv.org/api/query` 取 Atom 元数据（标题/作者/摘要）；
ACM/ASME/JASA/JAMA 的 DOI 记录用 OpenAlex 或 Crossref API 核验（标题/作者/年份/venue/DOI）。
**没有任何一条引用是凭记忆写的**（每条的标题/作者/年份/URL 都有一次实际抓取）。但要区分两种核验强度：
(a) 抓了原文页面或 arXiv 摘要的条目，其方法描述**来自读到的文本**；(b) 只用 OpenAlex/Crossref 核验了书目元数据的条目
（Eraser、FastTrack、RacerX、nDCG、Weibull、Kaplan-Meier、PinPlay），对其"做了什么"的描述是领域常识
（教科书级），**不是从原文摘要核验的**——引用时未扩展具体数字/结论。两种强度在下面各自的引用里已按此约束使用。

**明确未能取得、因此未引用（避免臆造）**

| 想引的 | 状态 | 处理 |
|---|---|---|
| Google SRE 书（SLO/error budget/多窗口 burn-rate） | `sre.google` 两次 `fetch failed`；`queue.acm.org` 返回 403 Cloudflare | **不引用**；只在 M6 用 Poisson/rule-of-three 数学，SLO 只作为"术语"提及并标注未核实 |
| FunSearch（Nature 2024, Romera-Paredes et al.） | nature.com 重定向到 `idp.nature.com`，未取到原文 | **不引用**；用 AlphaEvolve（arXiv，已取）代表同类"LLM + 程序搜索 + 评估器"方法 |
| Data Collider（Erickson et al., OSDI 2010） | ACM DL 403；OpenAlex 标题检索未命中 | **不引用**；采样检测只引 LiteRace |
| ThreadSanitizer 论文（Serebryany & Iskhodzhanov 2009） | ACM DL 403 | **不引用论文**；改引 Clang 官方 TSan 文档（已取，含平台与开销） |

**已知不确定（标注为不确定，不作为判据）**

- MinGW 上"没有 ASAN/TSan"是项目文档事实 + 本机 `mingw64/lib` 无 `libtsan*/libasan*` 的观测；
  未尝试用 MSYS2 包管理器安装（需网络与时间，且 TSan 在 Windows 上的支持本身不存在）。
- `clang -Wthread-safety` 在 MinGW 头文件/Conan include 路径下能否 `-fsyntax-only` 通过，**未实测**。
- WinDbg/TTD 安装与录制需要管理员权限与网络；本机未安装，**未实测**；TTD 与崩溃 dump 的集成方式未验证。
- `1/100 采样率`、`开销 ≤5×`、`TTF ≤2 min` 等阈值是依据同类文献量级给出的**工程判据**，不是文献常数。
- M7 的"240 题可探测 9pp / nDCG 2–4pp"是 pilot 前估计，须用 20 题 pilot 复估。

---

## 4. 与其它方向的边界（避免重复劳动）

| 方向 | 本文件提供 | 本文件不做 |
|---|---|---|
| `crash-fix`（0xC0000374 根因） | M1 的检测工具与正/负对照、M2 的重放流程、M3 的静态规则 | 不改产品代码、不跑 PageHeap 复现（由 crash-fix 主跑，我只给判据） |
| `debugger-tooling`（gdb/cdb/TTD） | M2 的 TTD 决策实验与 4 臂 yield 设计 | 不负责实际安装/脚本（只给实验设计） |
| `stability-harness`（3 h） | M6 的 exposure/KM/Poisson 判据与 S1/S2/S3 分级 | 不接管长期采样（由 stability-harness 执行，我只定口径） |
| `agi-branch-tree`（分支树） | M4/M5：搜索策略映射、样本效率估算、评估噪声前提 | 不实现 harness（只给"先测 M5 再投 M4"的决策规则） |
| `io-train`（IO 训练） | M9 的冻结编码器探针协议与样本量 | 不训练模型、不碰 BPU 产物 |

---

## 5. 一句话总结

**当前最该补的不是新功能，而是三样"测量仪器"**：(1) 能在没有 sanitizer 的 Windows 上找竞争的检测器（M1）；
(2) 能把"零崩溃时长"换算成"崩溃率上界"的稳定性判据（M6）；(3) 能分辨"变强了"与"噪声"的评测协议（M8/M7）。
三样都便宜、都只读、都不打断长跑；而 M4 的搜索策略要等 M5 给出评估噪声之后才值得投 45 h 机器时间。

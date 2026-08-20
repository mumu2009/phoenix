# 测试方法（Testing Methodology）——自我进化·主动推理智能体

本文件只给**方法**（指标、协议、门槛），不写测试代码。分层：单元不变量 → 数学性质 → 闭环学习 → 系统回归 → 长期纵向。原则：一切自我进化结论必须用 **ablation 对照 + 多种子 + 效应量**支撑，不能只看单次分数。

---

## 1. 数学不变量（Property tests，随机输入模糊测试）

对每个模块用固定种子（mt19937）随机输入检查：

| 模块 | 不变量 | 依据 |
|---|---|---|
| emotion | padToTensor/fromAppraisal 输出 ∈ [-1,1]；fromAppraisal 对 B 单调增、对 H 单调减 | algorithm.md 定理 15.1/7.2 |
| instinct | netUtility ∈ [-1,1]；benefitScore+harmScore ≤ 1；动作 argmax 稳定 | 定理 7.1 |
| primal | homeostaticCost ≥ 0，且 = 0 当且仅当所有感受恰在设定点；gain 后 intensity ∈ [0,1] | subconscious.md §3 |
| active_inference | 任意 learnValue 序列后 w 各分量 ≤ valueClamp；surprise ≥ 0；explorationMultiplier ∈ [0.5,2.0] | 钳制不变量 |
| subconscious | applyTemperament 后张量 ∈ [-1,1] | clip 定理 |
| sparse_block_matmul | τ=0 时与 denseMatMul 逐位一致；τ>0 时误差在 Frobenius 界内 | llamacpp_optimization.md §3.2 |
| determinism | 同 seed 同输入 → projectToDimension/plan 输出逐位相同 | 确定性投影缓存 |

## 2. 自我进化的验证协议（核心）

### 2.1 稳态环境学习曲线
合成环境：固定真值 w*，奖励 r_t = w*·z_t + 噪声，跑 N 回合，断言（容差内、N≥10 种子）：
1. 前向模型 surprise 的 EMA 单调下降（动力学被学会）；
2. TD 误差绝对值下降（价值收敛）；
3. cos(w_learned, w*) 上升（偏好朝真值方向）；
4. 实际累积效用 > 冻结学习（α=0）的 ablation。

### 2.2 Ablation 对照
单变量切换：α ∈ {0, 0.05}；γ ∈ {0, 0.9}；consolidate 开/关；adaptiveExploration 开/关。每种组合 N=30 种子，报告 mean ± std 与 Cohen's d（效应量），不只看 p 值。

### 2.3 无失控（No-runaway）
- 恒定奖励 r=+1 连续 10⁵ 回合：w 始终被钳制、无 NaN/Inf、全稳态代价有界。
- 恶意奖励（随机符号 / 恒负）：同样有界。

### 2.4 探索/利用调度
- 早期（surprise 高）：explorationMultiplier > 1；
- 稳态后期：→ 1；
- 环境非平稳（奖励分布漂移）：< 1（转利用）。

### 2.5 全稳态调节
- 注入恰好处于设定点的感受 → homeostaticCost ≈ 0；
- 偏差 δ → 代价随 δ 线性增长（斜率 = gain）；
- 改变设定点 → 代价最小值跟随新设定点移动。

### 2.6 巩固有效性
- 每 K 回合 consolidate 后，重放集的平均 surprise < 重放前；
- 情景记忆规模 ≤ 4096（上界不变量）。

## 3. 主动推理的验证

### 3.1 合成 MDP 最优性
已知最优动作的合成网格 MDP：plan() 的 argmin-EFE 动作与最优动作一致率 ≥ 基线；MPC vs 贪心：H=3 在延迟奖励任务上的效用 ≥ H=1（非近视性）。

### 3.2 好奇行为（epistemic 项可证伪）
构造某动作通向未探索状态的环境：epistW 高时该动作被选中频率显著上升；用状态访问熵随 epistW 单调上升作为指标。

### 3.3 趋利避害衔接
注入 Pain → 规划器偏向 avoid 动作；Threat → protect；并核对 driveVector/benefitHarm/agiPlan 字段与刺激一致。

## 4. 系统回归（沿用既有设施）
- compile.bat 全量编译门；各模块 gtest 过滤（InstinctTest / Emotion* / MixedModalIOTest / CognitionMixedModalTest / DeploymentMatrixLoadAll）；每次改动后跑 test-tools/api_regression.ps1（42 项）。
- 配置审计：tools/validate_phoenix_config.py——每个 cfgOr/resolveConfig dot-path 在 config/phoenix.json 有默认值（新增 agi.*、subconscious.* 后必须复跑）。
- llama.cpp 路径：先用参考实现单测 τ=0 逐位一致 + τ>0 界内，再写 ggml 补丁；补丁后跑 perplexity 回归（剪枝模型 ppl 涨幅 ≤ 预算）。
- graphContext：只测长上下文召回（带/不带 graphContext 的长期记忆保持率）；不再引用被废弃的 graphContext 有害 / raw>aug 结论。

## 5. 长期纵向（多日自主）
每日记录：累积 netUtility、全稳态代价（调节质量）、前向模型 surprise（适应性）、TD 误差绝对值（价值拟合）、冻结提示集上的 perplexity（基座无退化）、算力预算遵守度。验收：连续 7 天无指标单调恶化，多种子方差受控。

## 6. 模块级验收门槛（ship gates）
- active_inference（自我进化）：2.1/2.3/2.4 在 ≥9/10 种子通过。
- subconscious：§1 不变量 + 2.5 全稳态协议通过。
- sparse_block_matmul：τ=0 模糊一致 + τ>0 界内。
- llama_prune_analyzer：backup manifest 的 SHA-256 与原件一致；mask 置零比例在目标 ±1% 内。

## 7. 工具建议（不实现）
- C++：gtest（现有）跑不变量；固定种子 mt19937 模糊器。
- Python：numpy 跑统计协议（沿用 memory_tier_benchmark 的 TUI 模式）；pytest 组织协议。
- 统计：N≥30 种子、配对比较、Cohen's d；确定性测试固定种子。
### 3.4 动作分派（action dispatch）
- 注册动作 → `listAgiActions` 可见；重复注册按 name 幂等覆盖。
- 规划器选中 tool 类动作且执行器已设置 → `agiPlan.execution` 非空且 `handled=true`；执行器未设置 → 不执行、只记录 `bestAction`（可验证的惰性行为）。
- 注入 `tool:math` 且构造 EFE 使 math 占优（把其它动作 embedding 置零对比）→ 执行器收到 `addonType=math` 且 `reply` 为合法结果。
- 执行器异常（addon 返回 handled=false）→ 系统不崩溃、`agiPlan.execution.ok=false`，其余字段完好。
---

## 8. 完整验证剧本（供廉价 AI 逐条执行）

每步给出**命令、判据、失败时修什么**。按顺序执行；任一步失败即停下报告，不要继续。

### 阶段 0：构建门
- 命令：`compile.bat`（生成 phoenix_main.exe / bug_shooter.exe / phoenix_sql_cli.exe）；`compile_gtest.bat`（生成 gtest_runner.exe）。
- 判据：两个命令 Exit code 0；日志无 `error:` 行（允许 OpenCV/依赖 deprecation warning）。
- 失败修什么：新增的 5 个 .cpp（active_inference / subconscious_profile / sparse_block_matmul / agi_action_registry + 各模块）的符号错误；先修 `active_inference.cpp`，再修 `subconscious_profile.cpp`。

### 阶段 1：单元不变量（gtest）
- 命令（按模块逐个跑，记录通过数）：
  - `gtest_runner.exe --gtest_filter=InstinctTest.*:PrimalSensationTest.*`（趋利避害/感受）
  - `gtest_runner.exe --gtest_filter=EmotionSystemTest.*:EmotionTensorTest.*:EmotionWeightTest.*`（情感）
  - `gtest_runner.exe --gtest_filter=MixedModalIOTest.*:CognitionMixedModalTest.*`（混合模态桥）
  - `gtest_runner.exe --gtest_filter=DeploymentMatrixLoadAll.*`（649 配置矩阵，~15 分钟）
  - `gtest_runner.exe --gtest_filter=SemanticUnitTest.*:ConceptMatrixTest.*`（若存在）
- 判据：全部 PASS（当前全量基线 2099/2099）。
- 失败修什么：对应模块；§1 不变量表是最小检查集。

### 阶段 2：自我进化协议（gtest）
- 命令：`gtest_runner.exe --gtest_filter=ActiveInferenceProtocol.*`。
- 判据：4/4 PASS（学习曲线 ≥9/10 种子、双无失控、探索调度）。
- 失败修什么：先看 §2.1 的种子通过率——若 cos(w,w*) 低，检查奖励定义（必须是**当前状态** r_t = w*·z_t，不是下一状态）；若 w 发散，查 valueClamp 是否被 setValueClamp 覆盖。

### 阶段 3：动作分派（新协议，先写后跑）
- 新增 gtest（照 §3.4 写）：注册 4 个 tool 动作 → `listAgiActions` 幂等；设置一个记录式 executor（lambda 把调用写入 vector）→ 构造 EFE 使 `tool:math` 占优 → iterate 一次 → 断言 executor 收到 `addonType=math`、`agiPlan.execution.ok=true`；executor 返回 handled=false → `execution.ok=false` 且其余字段完好；未设 executor → 不执行、`bestAction` 仍在。
- 判据：新 gtest 全 PASS。

### 阶段 4：系统回归
- 命令：`powershell -ExecutionPolicy Bypass -File test-tools\api_regression.ps1`；`python tools\validate_phoenix_config.py`；`test\apis\run_backend_matrix.bat`（需 llama-server 8082 + tinyllama 8086 先起）。
- 判据：api_regression 42/42 PASS；validate 报告 0 missing dot-path；backend_matrix 按 AGENTS.md 预期（ollama/bitnet/native 预期失败，llama 模式通过）。
- 失败修什么：新增 `agi.*`、`subconscious.*` 配置键若缺默认值，补 `config/phoenix.json`；网关 111 的新增 cfgOr 若拼错路径，对照 validate 输出。

### 阶段 5：性能/并发专项（本轮去锁与激进优化的验证）
1. **LSH 召回**：写一个 gtest（或临时 main）：向 PersistentConceptMatrix 灌入 ≥1000 个 128 维归一化随机原型，随机扰动出 1000 个查询（扰动角度 ≤18°），对每个查询比较 LSH 结果与暴力扫描的 argmax 是否一致；判据 recall@1 ≥ 0.95；低于则调 L/b 或关闭 LSH（改 `kLshL/kLshBits`）。
2. **save 去锁**：起 4 线程并发 `addOrUpdate`（不同语义向量）+ 主线程并发 `status()`，跑 10⁴ 次；判据无崩溃、无死锁（超时即失败）、SQLite 文件可重新加载且条数一致。
3. **投影缓存 shared_mutex**：8 线程并发 `projectToDimension`（同一 (D,s) 键）+ 期间首次触发新键生成；判据结果与单线程逐位一致、无崩溃。
4. **稀疏 JL 距离保持**：1000 个随机 4096 维向量对，稀疏投影到 128 维后，成对距离相对误差 ≤ 0.2 的比例 ≥ 95%；不达标则提高 `nonZerosPerColumn`（改 `projectToDimensionSparse` 默认 max(3, D_out/3)）。
5. **并行 propagate 一致性**：同一概念矩阵分别跑串行（禁并行）与并行 propagate，终态 confidence 逐位差 ≤ 1e-6；并行版耗时 ≤ 串行（大 N 下）。

### 阶段 6：llama.cpp 路径
1. `sparse_block_matmul` 单测：τ=0 与 denseMatMul 逐位一致（随机矩阵模糊 1000 次）；τ>0 误差在 Frobenius 界内。
2. `python tools\llama_prune_analyzer.py <model.gguf> --analyze --backup-dir build\prune_backup --sparsity 0.3 --mask build\mask.npz --report build\prune_report.json`：manifest SHA-256 与原件一致、mask 置零比例在 30%±1%。
3. 剪枝模型 perplexity 回归（对照原模型，涨幅 ≤ 预算）后才允许写 ggml 补丁（挂点见 llamacpp_optimization.md §4）。

### 阶段 7：长期纵向
按 §5 每日指标跑 7 天；判据：无指标单调恶化、多种子方差受控。此阶段由人工/定时任务执行，不属于廉价 AI 单次验证范围。

---

**执行顺序总纲**：0 → 1 → 2 → 3 → 4 → 5 →（通过后才）6 →（人工）7 → 8（任务层，默认关闭模块的独立门）。任何阶段的失败修复都要**重跑该阶段 + 阶段 4**（回归门），不要只修不回归。

## 9. 任务层（Meeseeks）验证协议

前置：`mission.enabled=false` 时一切行为与现状逐位一致（回归门阶段 4 覆盖）。以下全部针对 `phoenix::mission`（gtest）与 `CognitionAutonomyManager` 的 mission API。

1. **压力单调与截断（gtest）**：`Mission m{painGainPerSec=0.1, maxPain=1.0}`，Running 后按固定 nowMs 序列采样 `pressure()`：判据 ① 对 0≤t≤10 严格递增（0.1/秒）；② t≥10 后恒等于 1.0（饱和）；③ Idle/Completed/Failed 状态恒返回 0。
2. **markComplete 终结疼痛**：Running 中途 `markComplete()` 后 `pressureNow()==0` 且 `state==Completed`、`endMs≥startMs`；再次 `markComplete()` 幂等（状态不倒退）。`markFailed()` 同理且不可再 Complete。
3. **累积疼痛定理**：数值积分 g=0.1、P_max=1 的 p(t) 曲线，对照 P(T)=gT²/2（T≤10 段）与 P(T)=5+P_max(T−10)（T>10 段）；判据相对误差 ≤1e-6（验证 §2 定理的分段公式，防实现回归）。
4. **spawnChild 变异有界**：以 `SubconsciousProfile::defaults()` + learningRate=0.05 为父本，rate=0.05，spawn 100 个子代：判据 ① 所有标量都在文档声明的 clamp 界内（PAD∈[−1,1]、riskAversion∈[0.2,3]、gain∈[0,5]、halfLife∈[1,3600]、learningRate∈[0.001,0.5]）；② 变异前后均值差 ≤ 2σ（无漂移）；③ rate=0 时子代与父本逐位相等（纯遗传）。
5. **多代完成时间下降（协议层模拟）**：模拟 G=10 代、每代 λ=8 子代（同一合成任务、完成时间 = 基线 + 高斯噪声），按 (1+λ) 保留最优父本；判据后 5 代均值 < 前 5 代均值（单侧 t 检验 p<0.05 才记为通过；噪声任务下允许失败并记录）。注意：这是**协议层**验证"若编排多代重试，完成时间会下降"的性质；运行时**没有**进程内自动选择循环（见 `doc/v7.0/archive/uncontrolled_evolution.md`）。运行时繁殖 = **复制是实例的自由能力**：`replicate` 是默认规划器动作、无固定触发条件、无接班（gtest：ReplicateMutatesAndRecords / ReplicaLimitBoundsReproduction / ReplicateActionRegisteredByDefault / ExecuteReplicateSpawnsSuccessorSession / ReplicateRespectsMaxReplicas / ReportOutcomeFalseKeepsMissionFailed）。
6. **自主栈注入（gtest）**：`assignMission({enabled,goal,..})` 后调 `iterate`：判据 ① 返回 `result.mission.active==true` 且 `pressure>0`；② `sensationEngine_.active()` 中存在 `type=Pain, source="mission-pressure", valence=−1, intensity==pressure` 的感受；③ `reportMissionOutcome({goalAchieved:true})` 后再 iterate：`active==false`、`pressure==0`、不再注入新 Pain。**单信号与衰减（新）**：连续 3 次 iterate 后 status 中 mission-pressure Pain 恰为 **1 条**（刷新而非堆叠）；`decayAuto(150s)` 后强度 ≈0.707（缺省半衰期 300s），`decayAuto(1500s)` 后整条剪除、homeostaticCost==0；per-type 调谐（如 pain halfLife=60s）覆盖缺省值。
7. **空目标幂等**：`assignMission({enabled:true})`（goal 为空）→ 不进入 Running、压力恒 0、`missionStatus().enabled==true`（武装但空闲）；随后带 goal 的 assignMission 正常启动。
8. **配置门（阶段 8，系统级）**：`mission.enabled=true` 且配置 goal 非空时，网关启动后 `missionStatus` 返回 Running（出生即绑定）；`mission.enabled=false` 时任何 mission 字段都不出现在 iterate 结果里（`mission` 为空对象）。
9. **压力增长模式（v8.0，gtest + 手工）**：① 缺省 `pressureMode=="logarithmic"`；② 对数模式 p(H)=P_max（H=pressureHorizonSec）且在 [0,H] 单调不减；③ `pressureMode="linear"` 时与 §9.1-9.3 的线性断言逐位一致（既有测试已显式钉住 linear）；④ 手工冒烟（用户口径：每模式一测）：Mission 控制台设立任务后 1 分钟内 pressure 增长但**不应**在 1 分钟即饱和（对数默认），随后启动自主循环，交付物（deliverable）应开始累积、宿主 llama-server 出现显著负载。
10. **人工监督 HTTP API（v8.0，阶段 8 系统级）**：① `POST /api/mission/assign {goal:"t"}`（goal 必填，缺省 400；自动 enabled=true）→ 200 且 `result.mission.state==Running`，随后 `GET /api/mission/status` 的 `pressure` 随墙钟增长；② `POST /api/mission/report {goalAchieved:true}` → `state==Completed`、`pressure==0`、`completionTimeMs>=0`，之后 iterate 不再注入 Pain；③ `POST /api/mission/replicate {}` → 返回子代 `{id,generation,goal}` 且 status 的 `children` 增长 1；④ `POST /api/cognition/autonomy/interject {text:"x",amendGoal:"y"}` → 运行中任务目标被重定向（`startMs` 保留、压力继续增长）；⑤ 未认证（`auth.enabled=true` 时无 token）四条 mission 路由全部 401。

## 10. 插件生态验证协议

1. **数学精确性（gtest，test_math_addon_exact.cpp）**：0.1+0.2=="0.3"（exact）、1/3*3=="1"、2^100 与 100! 精确十进制、-17 mod 5=="3"（Python 语义）、floor(-7/2)=="-4"、a=2;b=3;a^b=="8"；sin(pi/6)≈0.5（浮点容差 1e-9）；1/0、sqrt(-1)、foo(1)、"1+" 全部 ok=false 且带 position。另对拍随机大整数（见 addons/math_exact.hpp 设计注释）：(a+b)-b==a、q*b+r==a、gcd 整除——独立测试程序已跑 4000 组通过。
2. **搜索引擎解析（gtest，离线无网络）**：urlEncode/htmlToText 单元断言；DDG Lite HTML fixture → {title,url,snippet}（uddg= 解包 + 实体解码顺序）；端点 JSON 两种形态（results 对象 / 顶层数组 + link/description 别名）。联网冒烟（可选，被墙区域跳过）：engine.search("Phoenix AI") 返回 count>0 且每条含 url。
3. **MCP（gtest，test_mcp_client.cpp）**：帧协议往返（id/method/params、响应/通知/坏版本/垃圾输入）；进程内匿名管道 + fake 服务器完整会话（initialize 握手→tools/list→tools/call 取值→isError→ping→shutdown 不悬挂）。真实服务器联调：`mcp.enabled=true` 配一个真实 MCP 服务器，判据 configureMcp 返回 tools 非空且 callMcpTool 返回内容。
4. **cli-json（gtest）**：白名单外工具被拒（fail-closed）；白名单内 echo 命令返回原文；注册表往返一致。进程安全判据：请求文本含 `; rm -rf` 时不得执行任何命令（无 shell 语义）。
5. **subprocess（gtest）**：echo 捕获 stdout、exitCode==0；不存在的命令 started==false 或 exitCode!=0；超时路径（写一个 sleep 命令，timeoutMs=200，判据 timedOut==true 且耗时 < 5s）。
6. **回归门**：以上全部通过后跑阶段 4 系统回归；数学/搜索插件改动不得影响 emotion/instinct/agi 既有 43 项进化协议测试。
## 11. 长期自主循环与插话验证协议

1. **心跳自驱动（gtest，AutonomyLoopTest.AutonomyLoopTicksWithoutExternalIterate）**：configure{enabled:true, intervalSec:1, maxStepsPerTick:2} + start → 睡眠 2.6s → 判据 ① status.tickCount ≥ 1；② status.iteration ≥ 1（无任何外部 iterate 调用）；③ stop 后 running=false。
2. **插话消费一次（InterjectQueuesAndIterateConsumes）**：interject{text} 后首次 iterate 结果 interjectionsConsumed==1 且 cognitionModulation 含原文；第二次 iterate 为 0。
3. **目标改写不重启（InterjectAmendGoalRedirectsRunningMission）**：interject{amendGoal} 后 mission goal 已改、state 仍 Running、startMs 不变（压力继续）；无任务时 goalAmended=false 且带 warning。
4. **进化状态往返（ExportImportRoundTripsEvolution）**：configureAgi+configureSubconscious+assignMission → 3 次 iterate → exportState 含 agi/sensations/mission → 新实例 importState → preferences 逐位一致、mission goal 一致、iteration>0。
5. **控制器全量序列化（ControllerPersistence.JsonRoundTripPreservesLearnedState）**：20 步 observeRewarded 后 toJson/fromJson → 价值头逐位一致、surpriseEma 一致、动作空间一致、episodes>0、prefsBootstrapped 保持、模型 dim/actionDim 一致。
6. **amendGoal 生命周期（AmendGoalRedirectsWithoutRestart）**：Running 时改目标成功且 startMs 保持；Completed 后返回 false 且目标不变。
7. **回归门**：以上全过后跑阶段 4；心跳默认关闭时所有既有行为逐位不变（配置门）。
## 12. 系统级急停验证协议

1. **登记表（RegistryRegisterUnregisterStopAll）**：登记 2 实例 → count==2、snapshot==2；注销 1 → count==1；stopAll → total==1、stopped==1、处理函数恰好一次；全注销 → count==0。
2. **闩锁一次性（PressLatchesOnceAndStopsAll）**：press 后 latched=true、reason 记录、report.stopped==2、各处理函数恰好一次；二次 press → alreadyLatched=true 且处理函数不再执行。
3. **关停处理函数恰好一次（PressRunsShutdownHandler）**：press 执行关停函数一次；再 press 不再执行。
4. **阻断（LatchBlocksIterateAndInterject）**：press 后 iterate/interject/startAutonomyLoop 全部返回 ok=false 且错误含 "emergency stop"。
5. **杀运行中心跳（EstopKillsRunningAutonomyLoop）**：intervalSec=1 的循环启动后 press → 1.5s 内 autonomyLoopStatus.running==false。
6. **回归门**：E-stop 未按下时全部既有行为逐位不变；以上测试后必须 resetForTesting/clearForTesting 以免污染同进程其它测试。
## 13. SparkArray 作用域验证协议（v8.0）

1. **遗留兼容**：`spark.scopes=[]` 时行为与既有完全一致（chat 域由 spark.gnnScheduler.enabled 单独决定）——回归门阶段 4 覆盖。
2. **单选**：`["chat"]` → /api/chat 与 /api/transformer/chat 的 gnnScheduler 投票生效；后续其它域接门后，未列出的域关闭。
3. **多选**：`["chat","consensus"]` → 两域并列生效（consensus 域接门后验证）。
4. **禁用**：`["none"]` → chat 域即便 gnnScheduler.enabled=true 也不发起 SparkArray 派发（sparkAnn 为空、graphContext 不受其影响）。
5. **all/异常**：`["all"]` 全开；配置损坏时 fail-open 回遗留行为。
6. **组合矩阵**：[]/chat/none/all × gnnScheduler.enabled∈{true,false} 共 8 组合逐一断言 sparkAnn 出现与否（阶段 2 起扩展到 consensus/voter/evaluate）。
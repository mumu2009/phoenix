# v10.1 实施记录（2026-11-07/08）

前三个方向的实现 + 第四个方向的无破坏实验。所有新行为配置门控、
默认关（或默认 = 旧行为），相关测试 822/822 PASS。

## 1. 情绪模块增强（已实现 + 测试）

- `emotion_system.hpp` 的 `vadLexicon()` 加 31 词工程情绪层：
  fail/failed/error/crash/fix/pass/passed/slow/fast/broken/timeout/retry/
  abort/success/correct/wrong/missing/empty/invalid/bug/works/ok/done/
  blocked/stuck/wait/lost/leak/race/dead/safe。VAD 三元组与既有词语义对齐
  （如 fail {-0.6,-0.3,-0.4} 对齐 sad；fix {0.35,0.2,0.4} 对齐 proud 的
  低强度版）。依据：项目语料 1.7M 字符中这些词 29/32 高频出现且原词典
  全盲（fix×177、done×434、error×69）。
- `surfaceAffect` 加词干回退：未命中 token 去 ed/es/s/ing/d 后缀再查
  （crashes → crash），CJK 子串路径不变。
- 测试（`test_emotion_analyzer.cpp` 的 EmotionEngineeringAffect 组）：
  正价文本 valence>0、负价 valence<0、词干等价（crashes ≡ crash），
  直接走真实 `surfaceAffect` 管线（不经 mock fixture）。

## 2. 进化效能（已实现 + 测试）

挂在 v10.0.0 已有的链上（genome 变异 + 子盒 + `adoptEliteLocked` +
lineage + 现场训练 unitLearner/selfMind/tinyExpert），新增：

- `MissionLifecycle::setEvolutionAnnealPerGen(perGen)`：`effectiveMutationRateLocked`
  的 0.5–2.0× 因子偏差按 `1 - perGen·generation` 收紧（perGen=0 精确等于
  旧行为）；config `mission.evolution.annealPerGen`（默认 0.0）。
- `MissionLifecycle::setEvolutionEarlyStop(on)` + `evolutionStalled()`：
  `adoptEliteLocked` 记 `lastBestWorkMs_`/`stallCount_`，连续两次采纳未
  缩短 workMs → stalled（复制动作据此停支）；config
  `mission.evolution.earlyStop`（默认 false）。
- `autonomy_stack` 解析两键并在 assign 时下发到各 mission。
- 测试（`test_mission_evolution_knobs.cpp`）：anneal gen0 与关=0 一致、
  早停初始不 stall、负数 perGen 钳制后速率仍 >0。

## 3. 热点隐藏条件动态加速（已实现，待现场验证）

- 111 的 deliberator 在 `research.hotPath.prefixSlots` 开启时把
  （staticLen、recentChars、记忆单元数、GNN 单元数、压力千分位）FNV-1a
  成 `opts["id_slot"] = "phx-<fp>"`，并打 `[prefix-cache]` 日志；指纹
  运行时动态计算（session 间隐藏条件变化自动换槽，服务器 KV 按变体
  复用）。默认关 = 旧单槽行为。
- 依据：87 条 `[mission-deliberate]` 行的 staticLen 仅 5 个变体且
  recentChars 48/87 非零——前缀随 session 上下文变化，单槽跨变体 miss。
- 现场验证项（下一步）：开 profile 跑两轮不同上下文的 mission，核对
  `[prefix-cache]` 的槽指纹随上下文变化且 llama-server 的 KV 命中率上升。
- **现场验证（2026-11-08）**：`AI_RESEARCH_PROFILE=hotpath_probe` 的网关
  跑 mission（`SELF-MOD rank the carrier description …`）打出
  `[prefix-cache] slot="phx-3145301889024154693" staticLen=74 recent=0
  mem=4 gnn=0 press=0.0011706`——指纹、id_slot、日志三件套在真网关生效；
  指纹的输入（staticLen/mem/gnn/press）随 session 上下文变化即自动换槽
  （代码级事实 + 单次现场观测）。

## 4. 模型融合（只实验，不实现）

无破坏探针（服务器只读、权重不动）：

- **三模型 tokenize 交集矩阵**（8 句项目语料）：
  qwen↔8B 共 34 个交集 token（全部是标点/数字/单字符）；qwen↔Granite 与
  8B↔Granite **0**。三个模型的词表两两不相交（除 ASCII 单字）。
- **词汇投影往返**（qwen 的 token ID 直接送 8B 的 `/detokenize`）：
  8 句全部内容性损毁（如 "surpriseEma_ = 0.0" → "ffectaramignoreV(using
  onCancel …"），字符长度比均值 1.004（长度幸存、内容 0/8 幸存）。
- **结论**：ID 直通式的词汇投影不可行；格式塔的"tokenizer-detokenizer
  对齐"需要共享词表（专家重分词）或嵌入级投影，不能靠 ID 映射。
  驱动/召唤体系保持不变；"输出驱动自身" = 现有 `[summon-tool]`/
  `[alias-router]` 的选择器（未改动）。

## 测试与构建

- gtest 重编（gtest_runner.exe 257,536,359，2026-11-07 22:26）后
  情绪/进化/使命/本能相关套件 **822/822 PASS**（含新增 6 测试）。
- 主程序重编（compile_v101_2026-11-07.log，PHOENIX_NEW_BUILD=1）。

## 5. 情绪模块矩阵化改造（2026-11-08，用户纠正后）

用户指出词面硬编码不可接受、交互应为 Unit/信号矩阵。改造：

- `EmotionSystem::processMessage` 改为信号路径：`const EmotionTensor *stimulus`
  （null = 中性刺激）；`RuleBasedEmotionAnalyzer::analyzeText` 不再进入
  运行时路径（词面层 `vadLexicon`/`surfaceAffect` 已从 processMessage 移除）；
- 8×8 训练矩阵 `modulationMatrix`（行主序 JSON `{"rows":[[8×8]]}`）：
  默认单位阵 = 旧行为；`loadModulationMatrix` + 配置
  `emotion.modulationMatrixPath`（116 接线）；
- 训练：`tools/train_emotion_modulation.py`（numpy 最小二乘，13 组自开发
  锚点：四象限 + alarm/urgency 区分 + 新奇/责备/控制角；均值绝对误差
  0.115）→ `runtime_store/emotion_modulation_matrix.json`；
- 测试：5 个词面断言改为信号路径（stimulus 传入），情绪套件 461/461 PASS；
  全量回归 4398/4398 PASS（v101b 之前的状态）。
- EmoBank 下载尝试两次（github raw/zip）因网络失败；锚点语料为自开发
  方案，外源语料的补充留待网络恢复。

## 6. 大规模测试（2026-11-08）

- simple-evals（OpenAI 的轻量评测库，GitHub zip 77 KB）下载并了解：
  MMLU/GPQA/MATH/HumanEval/MGSM/DROP/SimpleQA + HealthBench/BrowseComp，
  OpenAI 兼容端点；
- MMLU 数据集下载（6.7 MB）→ `tools/mmlu_subset_eval.py`（确定性种子）
  → **8B 基线 20/40 = 0.500**；
- 全组件 profile `config/research/full_components.json`：矩阵路径 +
  `mission.evolution`（enabled/anneal 0.15/earlyStop）+
  `research.hotPath.prefixSlots` + `aliasProbe` + split 关闭；
- 发现并记录：v10.0.0 的 `setMissionDeliverator` 没有任何调用点（mission
  工作环为空转），成长改走 chat 自修路径（field-train 挂在那条分支）；
- 成长循环 `build/tmp/growth_loop.ps1`（26 条消息：SELF-MOD + MMLU 题），
  受 CPU-only 8B 吞吐限制（每条数分钟），长跑中；完成后同种子重跑
  MMLU 对照 0.500。

## 7. 大规模测试结果（2026-11-08，GPU 版）

- 推理改走 Vulkan GPU（GTX 1660 SUPER，8B -ngl 16 + coder -ngl 99，`--parallel 3`）：
  3 路并发 3.8 秒，GPU 利用率 64%；
- 网关 5090 端口被死进程泄漏 socket 占住（netstat 3796 LISTENING 但进程不存在），
  换 5080 后正常；
- 成长循环 21 条消息（6 SELF-MOD + 15 MMLU 题）完成 7 条 HTTP 200（其余超时，
  受单请求 60 秒客户端超时与网关串行影响）；
- **同种子 MMLU 子集：成长前 20/40 = 0.500，成长后 21/40 = 0.525**（+1 题、
  +2.5pp）——方向为正但在 n=40 的噪声内（±~8pp），不能声称显著成长；
  更大 N 的对照需要更长跑。

## 8. 并行任务数据竞争：定位、修复与 2 小时验证（2026-11-11）

### 现象
`autonomyLoop.parallelMissions=true` 时，网关每 1–2 分钟崩溃一次：
Windows 事件日志 `异常代码 0xc0000374 (STATUS_HEAP_CORRUPTION)`，
故障模块 ntdll.dll。每次崩溃留下一个 32KB/0 线程的僵尸进程（句柄未释放），
24 小时累积 230+ 个；同时崩溃让 MMLU 评测拿到 0/0 的失败样本。

### 根因
`loopRun` 用 `std::async` 为每个 mission 起线程并发执行 deliberator lambda；
该 lambda 会触碰多个非线程安全的共享状态：
- llama 通道周边状态（`llamaCallMu_` 只串行化了调用本身）；
- 任务工作区文件 IO 与 L1 缓存；
- `missionOutlineCache_`、经验库、cross-context 记忆；
- `RuntimeState` 的 GNN 图（读写锁粒度不一致）。
并发写同一容器 → 堆元数据被破坏。

### 修复（有锁）
`GatewayServer` 新增 `mutable std::mutex missionDelibMu_`；在 mission
deliberator lambda 入口加 `std::lock_guard`，使并行调度保留但临界区串行化。
`config/phoenix.json` 恢复 `autonomyLoop.parallelMissions=true`，训练批次
恢复 5 条/轮。重编 `phoenix_main.exe` 46,780,457 字节（2026-11-11 00:48:30）。

### 2 小时验证（01:04–03:03，121 个采样）
- 崩溃事件：**0**；
- 网关健康：119/121 采样为 up（其余为启动窗口）；
- 网关 PID 变化：**0**（无重启）；
- 训练轮次持续推进（round 17 → 168），每轮 `assigned=5/5`；
- 结论：**并行任务在互斥保护下不再产生数据竞争，保留并行模式**。

### 遗留（可选优化）
当前为粗粒度锁，临界区串行执行，并行加速有限；若需真正并发提速，
应把锁细化到各资源（工作区/图/经验库分别加锁），需要先完整刻画访问路径。

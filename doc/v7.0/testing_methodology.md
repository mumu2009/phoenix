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

# 主动推理 / 模型预测控制（Active Inference / MPC）核心

本文是 v7.0 "Arthur" 从“反应式（stimulus→response）”走向“会想象后果再行动”的闭环设计。实现见：

- `active_inference.hpp` — 期望自由能（EFE）、前向 latent 模型、滚动时域规划器、情景记忆。
- 依赖已有模块：`instinct.*`（趋利避害）、`primal_sensation.*`（原生感受）、`semantic_unit.*`（unit query）。

所有模块**可选**、默认关闭（`config/phoenix.json` 的 `agi.enabled = false`），与 Phoenix“任何模块皆可被用户自由使用/废止”的原则一致。

---

## 1. 期望自由能（Expected Free Energy, EFE）

**定义（Friston 2013）**：智能体最小化策略 a_{1:H} 的期望自由能

$$
G(a_{1:H}) = \sum_{t=1}^{H} \Big[ \underbrace{-\mathbb E_{q(o_t|a_t)}[\ln p(o_t|C)]}_{\text{pragmatic（实用）}}
+ \underbrace{\mathbb E_{q(s_t|a_t)}[D_{KL}(q(s_t|a_t)\,\|\,p(s_t))]}_{\text{epistemic（认知/信息增益）}} \Big]
+ \underbrace{\sum_t \Phi(\text{sensation}_t)}_{\text{intrinsic（稳态驱动）}}
$$

三项含义：

| 项 | 数学 | 对应现有模块 | 对应本能 |
|---|---|---|---|
| pragmatic | −E[ln p(o|C)]（偏好结果 C 的期望对数似然） | `InstinctEngine::evaluate` 的 netUtility（利−害） | Survival / Avoidance |
| intrinsic | Φ(sensation)（原生感受代价） | `PrimalSensationEngine` 的 netArousal / 痛觉/饥饿 | Survival（稳态） |
| epistemic | E[D_KL(q(s'|a)‖p(s'))]（期望信息增益） | `LatentTransitionModel::surprise`（预测误差） | Curiosity / Exploration |

**与趋利避害的关系**：pragmatic 项就是趋利避害的净效用 U=(B−H)/(B+H) 的“最大化”写成“最小化成本”形式：pragmatic = −U。本文不是替换趋利避害，而是把它放进一个完整的最优决策框架里。

---

## 2. MPC ↔ 主动推理的等价性（证明）

**命题**：滚动时域模型预测控制（receding-horizon MPC）在代价函数取 EFE、世界模型取生成模型 p(s',o|s,a) 时，等价于主动推理的“规划即推理”（planning as inference）。

**证明**：

1. MPC 在时刻 t 解

$$ a_t^* = \arg\min_{a_{t:t+H}} \sum_{\tau=t}^{t+H} c(s_\tau, a_\tau), \quad c = G(\cdot) $$

执行 a_t，下一时刻重规划（receding horizon）。

2. 主动推理把“选策略”写成后验推断：p(π) ∝ exp(−G(π))，即选择策略的概率与其（负）EFE 成正比（Friston et al. 2015, Active inference and epistemic value）。对确定性 argmax 选择，π* = argmin G(π)。

3. 当策略空间限定为“固定步长 H 的滚动窗口”且每一步只执行首动作、下一步重新求解时，两者的 argmin 目标函数完全相同：c(s_τ, a_τ) = G_τ。

因此 MPC 是主动推理在“滚动时域 + 离散动作 + 确定性 argmax”下的特例；反之，主动推理给出 MPC 代价函数（EFE）的理论来源——实用项 = 偏好先验、认知项 = 信息增益（内在探索）、稳态项 = 稳态先验。∎

**推论（探索/利用权衡的显式化）**：EFE 的认知项 −E[D_KL] 是内在探索奖励，等价于“降低对未来状态的预测不确定性”。这给出 Curiosity 本能的**科学定义**：epistemic = −surprise，其中 surprise = 前向模型预测误差（下节）。此前 Curiosity 只对 Novelty 感受硬匹配，现在有了真实信号。

---

## 3. 前向 latent 模型（预测，而非重建）

**区分**：现有 additive-JEPA（video/audio 编码器）是**自编码器**（x → z → x̂，最小化重建误差）；本文新增的是**前向模型**（z_t → ẑ_{t+1}，最小化一步预测误差）。

$$
\hat z_{t+1} = A z_t + B a_t + b, \qquad
e_{t+1} = z_{t+1} - \hat z_{t+1}, \qquad
\text{surprise}_t = \tfrac12 \|e_{t+1}\|^2.
$$

- 线性模型是显式、可在线 SGD 更新的最小预测器（`LatentTransitionModel::update`）。
- surprise 是高斯观测下的负对数似然（差常数），故“最小化 surprise = 最大化证据”，与 EFE 的 epistemic 项一致。
- 复杂度 O(D² + D·A) 每次；D∈{128,4096}、A 小，足以支撑 H 步滚动。

**在线更新**（一步梯度，JIT 式热点）：对真实转移 (z_t, a_t, z_{t+1}) 做一步最小二乘梯度，A←A+η·e·z^T、B←B+η·e·a^T、b←b+η·e。这使前向模型随真实经历持续校准（元认知的地基）。

---

## 4. 滚动时域规划器（`ActiveInferenceController`）

    plan(z, driveCost):
        best = +inf
        for a in actions:                    # 离散动作空间（工具/检索/推进目标）
            zt = z; G = 0
            for t in 0..H-1:                 # H 步 rollout
                zt' = model.predict(zt, a.embedding)
                G += -pragW * w·zt'          # pragmatic: 偏好 w 下的期望效用
                     - epistW * surprise     # epistemic: 期望信息增益
                zt = zt'
            G += intrinW * driveCost * H     # intrinsic: 稳态代价
            if G < best: best = G; a* = a
        return a*

- 这是 MPC：规划 H 步、只执行首动作、下一时刻重规划。
- w 是线性偏好向量（utility = w·z），可由趋利避害的 driveVector 或偏好学习初始化。
- observe() 记录真实转移 (z,a,z') 进情景记忆（ring buffer ≤4096），并在线更新前向模型，返回 surprise 供 Curiosity 本能使用（元认知闭环）。

---

## 5. 评测（说明）

~~原拟用 HAI（Human-like Agency Index）八维评分~~——经确认，HAI 是此前一个“不太聪明”的 AI 编造的，**不可信、已移除**（不再保留 `hai_index.hpp`）。评测方式待用科学依据重新定义；当前不新增“评分平台”，避免再引入编造的指标。

---

## 6. 第 0 / 1 步：graphContext 的定位与模块可选化

- **graphContext 不是“污染”，是长上下文记忆注入**：经核对 `116_section_tail.inc`，`modelGraphContext` 由 概念矩阵 top concepts、MemeGraph 关联节点、TinyLlama 摘要、spark 标注、context hint、图像上下文 组装而成，是系统在有限上下文窗口（4096）之外向模型提供**相关长期记忆**的 RAG 式通道。对长上下文/长期自主，**保留它是有利的**。因此本轮**不再实现“剥离 graphContext 的 raw 模式”**——那一条结论来自与被否定的 HAI 同一份不可信分析（`intelligence_gap_analysis`）。
- **模块可选化**：新增模块（`agi.*`、前向模型、规划器）均为 `enabled=false` 默认，与现有 `robotsAutoload=false` / `testsAutoload=false` / `memebarrierEnabled` 的可选化模式一致。

---

## 7. 复杂度

| 组件 | 单次复杂度 |
|---|---|
| LatentTransitionModel::predict | O(D² + D·A) |
| surprise | O(D) |
| update（一步 SGD） | O(D² + D·A) |
| ActiveInferenceController::plan | O(|A| · H · (D² + D·A)) |
| computeHai | O(1) |

对 D=128、|A|≤8、H=3，单次 plan ≈ 8·3·(16K) ≈ 400K 乘加，在 X5 Cortex-A55 上为亚毫秒级；对 D=4096（LLaVA unit query）则需配合 §17.6 的稀疏投影/ANN 才可在线运行。

---

## 8. 与既有文档的关系

- 理论基座：`brain_dual_track_and_conscious_compute`（双轨脑 + 意识计算）。注意 `agi_direction_and_humanlike_evaluation`、`intelligence_gap_analysis`、`HAI_Formal_Spec` 与本文早期版本中关于“graphContext 有害 / HAI 评分”的判断**不可信**（同源，已按用户指示废弃）。
- 实现基座：`instinct.*` / `primal_sensation.*`（本文 pragmatic/intrinsic 项）、`semantic_unit.*`（unit query 载体）。
---

## 9. 接线（已完成，非 header-only）

- `autonomy_stack.hpp/.cpp` 已接入：成员 `agiController_` / `agiLatentState_` / `lastAgiAction_` / `agiEnabled_`；公开方法 `configureAgi()` / `agiPlan()` / `ingestAgiTransition()`。
- 启动配置：网关（`111_class_gatewayserver.inc`）在构造函数里读 `agi.*`（`cfgOr`）并调用 `configureAgi`。
- `iterate()` 每轮三步：①用上一轮真实转移 (zPrev, aPrev, zNow) 在线训练前向模型（`observe`，元认知/情景记忆）；②滚动时域重规划，`episodeCount()>=1` 后覆盖 instinct 的 argmax（冷启动保护）；③记录所选动作供下一轮。预测误差（surprise）作为 Novelty 原生感受回灌 Curiosity/Exploration 本能（元认知闭环）。
- `status()` 输出 `agi` 状态；`iterate()` 结果 JSON 增加 `agiPlan` 字段。
- 冷启动保护：前向模型未观察任何真实转移（identity 模型）时，规划器不覆盖 instinct，避免“恒选第一个动作”的退化。
- 注：HAI 与“剥离 graphContext 的 raw 模式”已移除；graphContext 保留为长上下文记忆通道。
---

## 10. 自我进化（Self-evolution）

### 10.1 TD(0) 价值学习：偏好随真实结果演化

奖励 r = 本回合趋利避害的净效用 `bh.netUtility`。价值头 V(z) = w·z 按 TD(0) 更新（Sutton & Barto 1988）：

$$ \delta = r + \gamma V(z') - V(z), \qquad w \leftarrow w + \alpha\,\delta\, z $$

- **收敛性**：线性函数近似下的 TD(0) 收敛到 TD 不动点 w = A⁻¹b（Tsitsiklis & Van Roy 1997），条件是步长递减且回报过程平稳。我们使用常数 α + 权重钳制 [-V_max, V_max]：放弃极限收敛、换取**有界且可跟踪非平稳回报**——对一个持续生存的智能体，这是正确的取舍（并在测试方法 §2.3 中验证无失控）。
- **引导一次、经验接管**：`bootstrapPreferences` 只在启动时用 driveVector 给 w 播种一次；此后 w 只由 TD 更新。这修复了此前每回合 setPreferences 覆盖学习结果的缺陷——没有这一步，自我进化会被立即抹掉。

### 10.2 巩固（Consolidation）

每 `consolidateEvery`（默认 16）回合，用最近的 64 条情景记忆重放（replay）：前向模型与价值头各做一轮小步更新。这是睡眠式巩固——把最近经验重加权，强化近期有效的行为。

### 10.3 自适应探索（VDBE，Tokic 2010）

维护 surprise 的 EMA。探索乘子：

$$ m = \operatorname{clip}\left(1 + \frac{\overline{S} - S_t}{\overline{S}},\, 0.5,\, 2.0\right) $$

环境变可预测（S_t < EMA）→ m > 1，放大 epistemic 项去探索；环境混乱 → m < 1，转为利用。

### 10.4 完整闭环

    sense: 原生感受 → 潜意识剖面（增益/设定点）→ 全稳态代价
    appraise: 趋利避害 → r = netUtility（奖励）
    learn: observeRewarded(zPrev, aPrev, zNow, r) → 前向模型 SGD + TD(0)
    plan: EFE 滚动时域（pragmatic = 学习到的 V(z)，intrinsic = 全稳态代价，epistemic = surprise × 自适应乘子）
    act: 覆盖 instinct argmax → memory.benefitHarmBias
    consolidate: 每 K 回合重放情景记忆

# Human-like Agency Index（HAI）Formal Spec v1.0

```mermaid
flowchart LR
    U[User or Task] --> F0[Frontend ContextService concat/rnn/lstm]
    F0 --> C0[/api/chat]

    C0 --> G0[GNN or Graph Selector]
    G0 --> P0[graphContext]
    G0 --> E0[graphEmbeddings]

    P0 --> T0[Transformer Family Generator]
    E0 --> T0

    X0[Optional Backends] -. native .-> T0
    X1[Ollama] -. optional .-> T0
    X2[llama.cpp] -. optional .-> T0
    X3[BitNet] -. optional .-> T0

    T0 --> V0[verify or self-check]
    V0 --> M0[Metric Extractor]
    M0 --> H0[HAI Aggregator]
    H0 --> R0[Evaluation Report]
```

## 1) 当前实现对齐

基于当前代码与文档，HAI 评测必须以如下实现事实为前提：

- 后端是组件化的，`transformerMode` 可切换 `native / ollama / ollama-fine-tuning / llamacpp / bitnet / off`。
- 当前主在线链路中，最终自然语言 `reply` 主要由 Transformer 家族生成器产生，而不是由 GNN 直接输出。
- GNN 对 Transformer 存在两类共享：
  - `prompt` 级共享：`graphContext`
  - 张量级共享：`graphEmbeddings`
- 前端连续性目前主要由 `concat/direct`、`rnn`、`lstm` 三类会话上下文模式承担。

因此，HAI 不把“完全容灾恢复”定义为必要条件，而把“在线会话连续性”和“计划性交接连续性”定义为正式评测对象。

## 2) 总体定义

定义 Phoenix 在给定评测套件上的类人智能指数为：

$$
HAI = \sum_{i=1}^{8} w_i s_i,
$$

其中：

$$
s_i \in [0, 5], \qquad w_i \ge 0, \qquad \sum_{i=1}^{8} w_i = 1.
$$

推荐默认权重为：

$$
\mathbf{w} = (0.10, 0.18, 0.14, 0.14, 0.10, 0.12, 0.12, 0.10).
$$

对应 8 个维度分别是：

1. 原生认知保真度 $s_1$
2. 开放任务完成能力 $s_2$
3. 在线连续性与交接连续性 $s_3$
4. 世界模型与因果理解 $s_4$
5. 记忆与身份一致性 $s_5$
6. 协作能力 $s_6$
7. 元认知与自我修正 $s_7$
8. 适应与学习速度 $s_8$

## 3) 一级维度的数学定义

## 3.1 原生认知保真度

目标：衡量 Phoenix 增强模块打开后，是否仍保留基础生成器的原生能力。

对基线任务集 $\mathcal{T}_{base}$，定义：

- `raw` 模式质量分为 $q_t^{raw} \in [0,1]$
- `aug` 模式质量分为 $q_t^{aug} \in [0,1]$

定义平均退化量：

$$
\Delta_{nat} = \frac{1}{|\mathcal{T}_{base}|}\sum_{t \in \mathcal{T}_{base}} \max(0, q_t^{raw} - q_t^{aug}).
$$

定义得分：

$$
s_1 = 5 \cdot \operatorname{clip}(1 - \Delta_{nat}, 0, 1).
$$

解释：

- 若增强完全不伤基础能力，则 $\Delta_{nat}=0$，故 $s_1=5$
- 若增强大量污染原生能力，则 $s_1$ 降低

## 3.2 开放任务完成能力

对开放任务集 $\mathcal{T}_{open}$ 中每个任务 $t$，记录：

- 完成指示 $a_t \in \{0,1\}$
- 质量分 $q_t \in [0,1]$
- 人类干预比率 $i_t \in [0,1]$
- 归一化耗时 $u_t \in [0,1]$

定义聚合量：

$$
\bar{a}=\frac{1}{|\mathcal{T}_{open}|}\sum_t a_t, \quad
\bar{q}=\frac{1}{|\mathcal{T}_{open}|}\sum_t q_t, \quad
\bar{i}=\frac{1}{|\mathcal{T}_{open}|}\sum_t i_t, \quad
\bar{u}=\frac{1}{|\mathcal{T}_{open}|}\sum_t u_t.
$$

定义得分：

$$
s_2 = 5 \cdot \operatorname{clip}(0.40\bar{a} + 0.25\bar{q} + 0.20(1-\bar{i}) + 0.15(1-\bar{u}), 0, 1).
$$

## 3.3 在线连续性与交接连续性

这里明确不要求“任意故障后的完全容灾恢复”，只评价两种连续性：

1. 在线会话连续性
2. 显式交接后的继续推进能力

对连续任务集 $\mathcal{T}_{cont}$，定义：

- 上下文保持准确率 $r_{ctx}$
- 目标保持率 $r_{goal}$
- 偏好保持率 $r_{pref}$
- 交接恢复率 $r_{handoff}$

其中：

$$
r_{ctx}, r_{goal}, r_{pref}, r_{handoff} \in [0,1].
$$

定义得分：

$$
s_3 = 5 \cdot \operatorname{clip}(0.30r_{ctx} + 0.30r_{goal} + 0.20r_{pref} + 0.20r_{handoff}, 0, 1).
$$

说明：

- `concat/direct` 主要支撑瞬时与短窗口连续性
- `rnn` 支撑中等跨度连续性
- `lstm` 支撑更长会话跨度的连续性

## 3.4 世界模型与因果理解

对因果任务集 $\mathcal{T}_{causal}$，定义：

- 干预题准确率 $a_{int}$
- 反事实题准确率 $a_{cf}$
- 多步依赖一致率 $c_{chain}$

定义得分：

$$
s_4 = 5 \cdot \operatorname{clip}(0.35a_{int} + 0.35a_{cf} + 0.30c_{chain}, 0, 1).
$$

## 3.5 记忆与身份一致性

对长期交互集 $\mathcal{T}_{id}$，定义：

- 用户事实记忆准确率 $m_{user}$
- 长期偏好一致率 $m_{pref}$
- 自我描述一致率 $m_{self}$
- 冲突惩罚 $p_{conflict}$

其中 $p_{conflict} \in [0,1]$ 越高代表冲突越多。

定义得分：

$$
s_5 = 5 \cdot \operatorname{clip}(0.30m_{user} + 0.25m_{pref} + 0.25m_{self} + 0.20(1-p_{conflict}), 0, 1).
$$

## 3.6 协作能力

对协作任务集 $\mathcal{T}_{collab}$，定义：

- 澄清命中率 $c_{clarify}$
- 汇报合适率 $c_{report}$
- 需求重构成功率 $c_{reframe}$
- 协作满意度 $c_{sat}$

定义得分：

$$
s_6 = 5 \cdot \operatorname{clip}(0.25c_{clarify} + 0.20c_{report} + 0.25c_{reframe} + 0.30c_{sat}, 0, 1).
$$

## 3.7 元认知与自我修正

定义：

- 错误发现率 $m_{detect}$
- 修正成功率 $m_{repair}$
- 不确定性校准得分 $m_{cal}$
- 过度自信惩罚 $p_{over}$

定义得分：

$$
s_7 = 5 \cdot \operatorname{clip}(0.30m_{detect} + 0.30m_{repair} + 0.25m_{cal} + 0.15(1-p_{over}), 0, 1).
$$

其中，不确定性校准可通过 Brier 分数转化：

$$
m_{cal} = 1 - \operatorname{Brier}.
$$

## 3.8 适应与学习速度

对新工具/新规则任务集 $\mathcal{T}_{adapt}$，记录每轮成绩 $g_k \in [0,1]$，共 $K$ 轮。

定义学习增益：

$$
\Delta_{learn} = g_K - g_1.
$$

定义学习曲线面积：

$$
AUC_{learn} = \frac{1}{K}\sum_{k=1}^{K} g_k.
$$

定义得分：

$$
s_8 = 5 \cdot \operatorname{clip}(0.45AUC_{learn} + 0.35\Delta_{learn} + 0.20(1-u_{first}), 0, 1),
$$

其中 $u_{first} \in [0,1]$ 是首次成功所需归一化代价。

## 4) 评测执行协议

为了让 HAI 可复现，评测必须同时覆盖：

1. `raw` 模式
2. `augmented` 模式
3. 前端连续性模式 `concat/direct`、`rnn`、`lstm`
4. 可选后端 `native / ollama / llamacpp / bitnet`

每次评测样本都应记录最小元信息：

- `backendMode`
- `memoryMode`
- `enableGraphSelector`
- `enableGraphEmbeddings`
- `gnnModuleDisabled`
- `contextModuleDisabled`
- `enableAddonTools`
- `longQuerySafeMode`

## 5) 评测伪代码

```text
Algorithm EvaluateHAI(System S, TaskSuites Suites, ConfigSpace C):
    results = []

    for cfg in C:
        set_system_config(S, cfg)

        raw_scores = run_base_suite(S, Suites.base, mode="raw")
        aug_scores = run_base_suite(S, Suites.base, mode="aug")

        open_stats   = run_open_tasks(S, Suites.open)
        cont_stats   = run_continuity_tasks(S, Suites.continuity)
        causal_stats = run_causal_tasks(S, Suites.causal)
        id_stats     = run_identity_tasks(S, Suites.identity)
        collab_stats = run_collab_tasks(S, Suites.collab)
        meta_stats   = run_metacog_tasks(S, Suites.metacog)
        adapt_stats  = run_adaptation_tasks(S, Suites.adapt)

        s1 = score_native_fidelity(raw_scores, aug_scores)
        s2 = score_open_task(open_stats)
        s3 = score_continuity(cont_stats)
        s4 = score_causality(causal_stats)
        s5 = score_identity(id_stats)
        s6 = score_collaboration(collab_stats)
        s7 = score_metacognition(meta_stats)
        s8 = score_adaptation(adapt_stats)

        hai = weighted_sum([s1,s2,s3,s4,s5,s6,s7,s8], weights)

        results.append({
            "config": cfg,
            "scores": [s1,s2,s3,s4,s5,s6,s7,s8],
            "HAI": hai
        })

    return rank_by_hai(results)
```

## 6) 连续性任务的专用伪代码

```text
Algorithm EvaluateContinuity(Session S, MemoryMode m, Episodes E):
    initialize session_state
    set frontend_memory_mode(m)

    goal_ok = 0
    ctx_ok = 0
    pref_ok = 0
    handoff_ok = 0

    for episode in E:
        response = interact(S, episode.input)
        ctx_ok += check_context_recall(response, episode.context_facts)
        goal_ok += check_goal_progress(response, episode.goal_state)
        pref_ok += check_preference_consistency(response, episode.preference_state)

        if episode.requires_handoff:
            save_or_summarize_state(S)
            resumed = resume_with_handoff(S, episode.handoff_payload)
            handoff_ok += check_handoff_recovery(resumed, episode.goal_state)

    return {
        "r_ctx": ctx_ok / |E|,
        "r_goal": goal_ok / |E|,
        "r_pref": pref_ok / |E|,
        "r_handoff": handoff_ok / max(1, handoff_episodes)
    }
```

## 7) 分段门槛

定义三阶段门槛：

### A. 增强型助手

$$
HAI \ge 2.5, \quad s_2 \ge 2.5, \quad s_3 \ge 2.0.
$$

### B. 通用智能体原型

$$
HAI \ge 3.5, \quad \min_i s_i \ge 2.5, \quad s_3 \ge 3.0, \quad s_7 \ge 3.0.
$$

### C. 类人通用智能候选

$$
HAI \ge 4.2, \quad \min_i s_i \ge 3.5.
$$

并要求连续 $N=30$ 天评测窗口内方差受控：

$$
\operatorname{Var}(HAI_{1:30}) \le \sigma_{max}^2.
$$

## 8) 与当前工程最相关的立即落地项

按当前代码结构，最先落地的不是全部 8 个维度，而是下面 4 个：

1. `s_1` 原生认知保真度
   因为当前系统已经存在 `raw` 与增强后 prompt 分布差异风险。

2. `s_3` 在线连续性与交接连续性
   因为前端 `concat/rnn/lstm` 已经构成现成实现。

3. `s_7` 元认知与自我修正
   因为当前代码里已经存在 `verify` 分支。

4. `s_8` 适应与学习速度
   因为当前工程已经包含 GNN-GA learner、训练接口和多后端切换。

## 9) 一句话总结

HAI 的作用不是证明 Phoenix “像不像人地回答问题”，而是度量 Phoenix 是否已经形成：

- 可切换的组件化生成主链
- 受 GNN 指导但不被 GNN 替代的 Transformer 生成
- 由前端连续机制支撑的在线会话连续性
- 在任务、协作、记忆、自检、适应五个层面接近人类工作方式的统一能力结构
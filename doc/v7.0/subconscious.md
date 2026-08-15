# 潜意识（Subconscious）可定制层：全稳态剖面

本文把“潜意识”定义为**脑干层可配置的先天参数**（`subconscious_profile.{hpp,cpp}`），在趋利避害之上增加“出生参数”——每个 Phoenix 实例可有不同的气质、敏感度、稳态设定点与风险态度。全部参数默认值 = 当前行为，`subconscious.enabled=false` 时零影响。

---

## 1. 参数与理论依据（全部可引用，不编造）

| 参数 | 理论 | 作用 |
|---|---|---|
| baselineValence/Arousal/Dominance + temperamentStrength | 气质 = PAD 倾向（Mehrabian & Russell 1974） | 情感张量向先天基线平移 |
| sensationTuning[type].gain / halfLifeSec / setpoint | 全稳态（Sterling 1988）与对手过程（Solomon & Corbit 1974） | 每类原生感受的精度、衰减半衰期、稳态设定点 |
| riskAversion | 前景理论（Kahneman & Tversky 1979） | 效用曲率 u' = sign(u)·|u|^γ |
| anticipatoryGain | 精度加权（Friston） | 前向模型预测误差（surprise）→ Curiosity 的增益 |
| instincts | 行为学（ethology） | 自定义野性表替代内置五本能 |

---

## 2. 气质（Temperament）：PAD 基线

气质是先天、稳定的 PAD 倾向。定义基线向量 b = (b_V, b_A, b_D) ∈ [-1,1]³ 与强度 s ≥ 0：

$$ E' = \operatorname{clip}\left(E + s \cdot b,\, -1,\, 1\right) $$

**性质（有界性）**：clip 保证 E' ∈ [-1,1]；s=0 时 E'=E（零影响）。∎

---

## 3. 全稳态代价（Allostatic cost）

每类感受 s 有设定点 set_s、增益 g_s：

$$ C = \sum_{s} g_s \cdot \left| i_s - \text{set}_s \right| $$

**性质**：C ≥ 0；C = 0 当且仅当所有感受恰好处于设定点；C 对任一偏差 |i_s − set_s| 单调不减（逐项绝对值）。∎

这是稳态调节（homeostasis）：如 pain 设定点 = 0（任何痛都是代价），novelty 设定点 ≈ 0.5（无聊与过载都代价）。`PrimalSensationEngine::homeostaticCost()` 实现之，并取代 `netArousal()` 作为主动推理的 intrinsic 项（仅当剖面启用）。

---

## 4. 风险态度（Prospect-theory curvature）

效用曲率：

$$ u' = \operatorname{sign}(u) \cdot |u|^{\gamma}, \quad \gamma > 0 $$

γ = 1 线性；γ < 1 凸（风险追求）；γ > 1 凹（风险厌恶）。当 u 限定在 [0,∞) 且 γ→1 时，由 L'Hôpital 可得其与 Bernoulli 对数效用 u ↦ ln u 同族（对数效用是 γ→1 的相对风险厌恶极限）。该曲率作用在 `ActiveInferenceController::utility()`（`setRiskAversion`）。

---

## 5. 预期性唤醒（Anticipatory arousal / 精度加权）

前向模型预测误差 surprise S（`LatentTransitionModel::surprise`）以增益 g_a 回灌 Novelty 原生感受：

$$ i_{novelty} = \operatorname{clip}\left(g_a \cdot \min(1, S),\, 0,\, 1\right) $$

g_a = 0 关闭元认知回灌；g_a > 1 放大好奇。这是精度加权：surprise 越可信（前向模型越准），好奇信号越强。

---

## 6. 自定义野性表

`instincts` 数组可整体替换内置五本能（`InstinctEngine::replaceAll`），每个野性含 type/activation/benefitWeight/harmWeight/targetSensation/actionBias——即“出生时不同的本能优先级”。

---

## 7. 接线与可选性

- `autonomy_stack`：`configureSubconscious()`；`iterate()` 应用气质、全稳态代价与预期性增益。
- 网关启动（`111_class_gatewayserver.inc`）读 `subconscious.*`（`cfgOr`）并调用 `configureSubconscious`。
- 全默认 = 当前行为，`enabled=false` 时剖面完全不生效（惰性）。

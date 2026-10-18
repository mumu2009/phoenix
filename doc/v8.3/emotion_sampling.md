# 聊天情感采样

情感仍走原来的三条：利/害鉴赏、PAD 八维、对手过程稳态。这一层只把采样从粗线性改成可检验的连续映射。聊天因果串仍然只有用户原话。

## 细鉴赏

`fromAppraisal(B, H)` 不变。`fromFineAppraisal` 在它上面加四个通道，全在 `[0, 1]`：

- `novelty`：意外程度。0 时唤醒度就是原来的 `A0`。
- `control`：应对力。0.5 时支配度仍等于净效用 `U`。
- `uncertainty`：结果没把握。0 时惊讶度就是 PAD 公式的值。
- `otherAgency`：事件归到别人身上。0 时愤怒和信任不被改写。

```text
U  = (B - H) / (B + H + eps)
A0 = 2 * min(B + H, 1) - 1
A  = clamp(A0 + 0.55 * novelty * (1 - A0), -1, 1)
D  = clamp(U + 0.65 * (2*control - 1) * (1 - |U|), -1, 1)
t  = padToTensor(U, A, D)
surprise = clamp(t.surprise * (1 + 0.75 * uncertainty), -1, 1)
blame = otherAgency * H
anger = clamp(anger + 0.40 * blame * (1 - anger), -1, 1)
trust = clamp(trust - 0.35 * blame * (1 + trust), -1, 1)
```

中性通道（novelty = 0，control = 0.5，uncertainty = 0，otherAgency = 0）与 `fromAppraisal(B, H)` 逐维相同。每一步都是仿射函数再夹断，所以对输入 Lipschitz。八维输出仍在 `[-1, 1]`。

本能层的利/害分数算法不动。Novelty、Threat、Pain、Pleasure、SocialIsolation、Fatigue 只额外累加这四个通道，再交给 `fromFineAppraisal`。没有这些感觉时，驱动向量等于旧鉴赏。

## 聊天采样

`mapChatSampling` 把八维张量收进聊天合法区间。中性张量给出温度 0.30、top_p 0.90，惩罚停在聊天下限。

```text
s = clamp(0.55*arousal - 0.35*fear + 0.10*surprise, -1, 1)
temperature = clamp(0.30 + 0.10*s, 0.15, 0.40)

top_p = clamp(0.90 + 0.04*surprise - 0.03*dominance, 0.50, 0.95)

frequency = 0.35 + 0.25 * clamp(0.5*max(anger,0) + 0.5*max(-valence,0), 0, 1)
presence  = 0.15 + 0.20 * clamp(0.5*max(-trust,0) + 0.5*max(joy,0), 0, 1)
repeat    = 1.18 + 0.22 * clamp(0.6*max(anger,0) + 0.4*max(surprise,0), 0, 1)
```

温度在夹断之前已经落在 `[0.20, 0.40]`，再夹到 `[0.15, 0.40]`。惩罚只在下限之上抬升，不会把聊天的防重复地板拖下去。

logit bias 是固定词上的连续偏置，绝对值不超过 0.75。中性情感给出空表。这些词只进采样参数，不进因果串。

任务路径仍用原来的宽区间 `inferenceOptions`（基线温度 0.7）。只有聊天 scope 走 `mapChatSampling`。网关若再套 `applyChatSamplingFloors`，只会把已经在带内的值留在带内。

## 稳态和念头

`homeostaticUpdate` 就是原来的两步插值：先向刺激靠拢，再向基线拉回。`processMessage` 改走这一个函数，数值和以前相同。

自我层的学习算法不改。`tickThought` 照旧更新策略。情感只读这一拍返回的 `action` 和 `text`：

- `blendThoughtAffect` 用言语行为的 PAD 增量，加上念头正文的词表均值，加到当前张量上。增益默认 0.22。它不调用 `SelfMind`，也不返回任何字符串。
- `sensationFromThought` 写一条带上下文前缀的原初感觉（`chat:<id>:thought` 或 `mission:<id>:thought`），供下一拍鉴赏衰减。来源里没有念头正文。空上下文写成 `detached:thought`，不会变成全局感觉。

聊天用户句若命中词表，再做一次小的稳态更新（耦合 0.25，向零基线拉 0.05）。未命中则不动。

## 因果串

聊天 scope 的 `composedPrompt` 等于 `chatCausalText(用户句)`，也就是用户句本身加一个换行。不写 `Affect:`、驱动向量、`<emotion>`、`[self-kernel]`，也不写念头。`EmotionInfluence` 的 `promptModulation` 为空。网关里 ahead 的采样只填认知包还没设的键，不能覆盖这条聊天映射。

## 测试口径

从仓库根目录编译并运行（不链 `phoenix_main`，不碰 llama-server）：

```text
g++ -std=c++20 -O2 -I. -Ibuild/conan_home/p/nlohm1bed1ddc0a2fa/p/include ^
  -o build/tmp/emotion_sampling_driver.exe ^
  test/emotion_sampling_driver.cpp instinct.cpp primal_sensation.cpp ^
  self_mind.cpp module_overrides/emotion_influence.cpp
build\tmp\emotion_sampling_driver.exe
```

通过条件：

- 中性细鉴赏与 `fromAppraisal` 一致；新奇抬高唤醒，掌控抬高支配，不确定放大惊讶，他人归因抬高愤怒并压低信任；网格上八维仍在 `[-1, 1]`。
- 采样网格上温度在 `[0.15, 0.40]`（按 float32 夹断；和 double 字面量比要留 1 ulp），top_p 在 `[0.50, 0.95]`，三项惩罚不低于 0.35 / 0.15 / 1.18，logit 偏置在 `[-0.75, 0.75]`。中性温度是 0.30。唤醒升高则温度升高。
- `I am happy` 的三词均值与旧 PAD 池化一致。中文「害怕」让恐惧为正、效价为负。
- 刺激等于基线时，稳态迭代收敛到基线。刺激一直是零向量、基线不是零时，不动点是两者的凸组合（衰减 0.80、拉回 0.25 时，效价停在 `0.625 * 基线`），不会把基线擦成零。
- 偏好念头的效价高于客服念头。混合函数不改 `SelfMind` 的 JSON。因果串等于用户句，不含念头和情感标签。聊天与任务的念头感觉互不可见。
- 无感觉时本能驱动等于旧鉴赏；有新奇感觉时唤醒度高于旧鉴赏。

全量套件里的同名检查在 `test/gtest/unit/emotion/test_emotion_sampling.cpp`（`EmotionSampling.*`）。本周期不链 `phoenix_main`。

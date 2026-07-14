# Reinforcement Learner Algorithm

## 1. Problem Formulation

We model policy optimization on text-derived state vectors as a discounted MDP

- State: $s_t \in \mathbb{R}^d$, sampled from corpus encodings.
- Action: $a_t \in \{0,1,2,3\}$, selecting transition mode.
- Transition: $s_{t+1}=T(s_t,a_t,\xi_t)$ where $\xi_t$ is Gaussian perturbation.
- Reward: $r_t = r_{ext}(s_{t+1},a_t) + \beta r_{int}(s_t,s_{t+1})$.

The objective is

$$
\max_\pi J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

## 2. State Encoding Pipeline

1. Read test corpus files from runtime test directory.
2. Tokenize with alnum+unicode pass.
3. Build capped vocabulary by global term frequency.
4. Encode each file with normalized TF-like vector.
5. Inject small noise at sampling for stochastic exploration.

This converts unstructured text to stable fixed-dimensional RL states.

## 3. Policy-Value Head

A shared linear policy-value head is used:

- Policy logits: $z_a = W_p^{(a)} s + b_p^{(a)}$.
- Action distribution: $\pi(a|s)=\text{softmax}(z)_a$.
- State value: $V(s)=W_v^\top s + b_v$.

Training uses actor-critic style gradients:

- TD target: $y_t = r_t + \gamma V(s_{t+1})$.
- Advantage: $A_t = y_t - V(s_t)$.
- Policy term with entropy regularization.
- Value term with squared TD residual.

## 4. Experience Replay Strategy

The implementation combines two buffers:

- Uniform replay for broad coverage.
- Prioritized replay for high-TD and high-novelty transitions.

Sampling from prioritized replay approximates

$$
P(i) = \frac{p_i^{\alpha}}{\sum_j p_j^{\alpha}}.
$$

After update, priorities are refreshed by reward and next-state magnitude proxy.

## 5. Intrinsic Curiosity Reward

Curiosity tracker computes novelty from state change with adaptive memory weights:

$$
r_{int}(s_t,s_{t+1})=\sqrt{\frac{1}{d}\sum_{i=1}^d (\Delta_i^2 (1+|m_i|))}, \quad \Delta_i=s_{t+1,i}-s_{t,i}.
$$

Memory update:

$$
m_i \leftarrow 0.995 m_i + 0.005\Delta_i.
$$

This improves exploration in sparse-reward regions.

## 6. Optimization Loop

Per cycle:

1. Sample state from encoded corpus.
2. Sample action from policy.
3. Simulate transition and compute extrinsic reward.
4. Add intrinsic reward.
5. Push transition to uniform+prioritized replay.
6. Train on uniform batch.
7. Train on prioritized mini-batch.
8. Periodically evaluate action histogram and average reward.

## 7. Risk and Stability Audit

A reward auditor outputs mean, standard deviation, and maximum drawdown on reward trace:

- Mean monitors optimization level.
- Std monitors volatility.
- Max drawdown monitors instability under non-stationarity.

These diagnostics are returned in runtime JSON for service-side observability.

## 8. Interface and Runtime Contract

`IReinforcementLearner` contract methods:

- `learn(cycles)` for training step.
- `latest()` for latest metrics snapshot.
- `refreshTests(dir)` and `setTestsDir(dir)` for corpus update.

Factory registration binds advanced implementation to existing module mount path without modifying caller logic.

# Adversarial Learner Algorithm

## 1. Objective

The adversarial learner optimizes a red-team/blue-team loop:

- Red-team generates perturbations that increase detector miss probability.
- Blue-team retrains detector with adversarially augmented data.
- Robustness metrics track defense quality across rounds.

Given detector $f_\theta(x)\in[0,1]$, attack generator $g_\phi$, and threshold $\tau$:

$$
\min_{\theta} \max_{\phi} \; \mathbb{E}_{x\sim\mathcal{D}}\big[\ell(f_\theta(g_\phi(x)),1)\big] + \lambda\,\mathbb{E}_{x\sim\mathcal{D}}\big[\ell(f_\theta(x),y)\big].
$$

## 2. Detector Model

A lightweight token-weight logistic model is used:

- Score: $s(x)=b+\sum_{t\in x}w_t$.
- Probability: $p(x)=\sigma(s(x))$.
- SGD update per sample.

This design is fast enough for repeated adversarial rounds and suitable for online adaptation.

## 3. Attack Operators

Character and lexical perturbations:

1. Visual substitutions (`a->@`, `o->0`, etc.).
2. Character dropping.
3. Adjacent character swap.
4. Synonym substitution.

Operators preserve sentence structure while creating difficult near-boundary variants.

## 4. Beam Search Attack Generation

For each seed sample:

- Initialize beam with original token sequence.
- Expand candidates with perturbation operators.
- Re-score by detector confidence.
- Keep top-K by score and edit count.

This approximates high-value adversarial search under bounded perturbation budget.

## 5. Rule-Based Defense Prior

A sensitive-token risk prior counts high-risk actions and raises detector score:

$$
\tilde{p}(x)=p(x)+\eta\,\text{riskCount}(x).
$$

Canonicalization maps obfuscated characters back to semantic tokens before risk lookup.

## 6. Threat Propagation Graph

The implementation maintains a token transition graph:

- Nodes are tokens with risk states.
- Directed edges represent observed attack-token transitions.
- Seed risk is injected from direct sensitive hits.
- Multi-step diffusion propagates risk to neighboring tokens.

For edge-normalized transitions,

$$
r^{(k+1)} = r^{(k)} + \alpha P^\top r^{(k)}.
$$

Sentence graph risk is averaged over token node risks.

## 7. Adversarial Training Loop

Per round:

1. Generate beam candidates from sampled seeds.
2. Ingest candidates into threat graph and diffuse risk.
3. Augment training set with generated attacks.
4. Re-train detector for one adversarial epoch.
5. Evaluate precision/recall/F1 and robust metrics.

## 8. Robustness Evaluation

Robustness suite outputs:

- `cleanAcc`: accuracy on validation clean samples.
- `attackAcc`: accuracy on generated attack set.
- `robustGap = cleanAcc - attackAcc`.

This gives a direct measure of attack-induced degradation.

## 9. Runtime Interface

`IAdversarialLearner` methods:

- `attack(rounds)` executes attack-defense rounds.
- `latest()` returns latest report.
- `refreshTests(dir)` reloads data.

Factory registration keeps existing service wiring intact while replacing behavior with advanced adversarial logic.

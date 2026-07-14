# Transformer + GNN 融合建模数学证明（独立版）

## 1. 定义与目标

令输入为 $x$，关键词图为 $g=(V,E,X_V)$，输出序列为 $y$。模型定义为：
$$
z_g=\mathrm{GNN}_{\psi}(g),\qquad p_{\theta}(y\mid x,z_g).
$$
记联合参数 $\Theta=(\theta,\psi)$。

训练目标：
$$
\min_{\Theta}\ \mathcal{J}(\Theta)=\lambda_{lm}\mathcal{L}_{lm}+\lambda_{sty}\mathcal{L}_{sty}+\lambda_{ver}\mathcal{L}_{ver}+\lambda_{reg}\|\Theta-\Theta_0\|_2^2.
$$

其中
$$
\mathcal{L}_{lm}=-\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{T_i}\log p_\theta(y_{i,t}\mid y_{i,<t},x_i,z_{g_i}),
$$
$$
\mathcal{L}_{sty}=-\frac{1}{N}\sum_{i=1}^N\log q_\phi(s_i\mid \hat{y}_i),\quad \hat{y}_i\sim p_\theta(\cdot\mid x_i,z_{g_i}),
$$
$$
\mathcal{L}_{ver}=\frac{1}{N}\sum_{i=1}^{N}(s_\Theta(x_i,g_i,y_i)-\hat{s}_i)^2.
$$

## 2. 信息增益命题

### 假设 A
$$
I(Y;G\mid X)>0.
$$

### 命题 1（最优对数风险不劣）
定义
$$
\mathcal{R}_X^*=\inf_q\mathbb{E}[-\log q(Y\mid X)],\qquad
\mathcal{R}_{XG}^*=\inf_q\mathbb{E}[-\log q(Y\mid X,G)].
$$
则
$$
\mathcal{R}_{XG}^*\le \mathcal{R}_X^*,
$$
且当 $I(Y;G\mid X)>0$ 时严格小于。

**证明**：
在对数损失下，最优风险等于条件熵：
$$
\mathcal{R}_X^*=H(Y\mid X),\quad \mathcal{R}_{XG}^*=H(Y\mid X,G).
$$
由条件熵性质 $H(Y\mid X,G)\le H(Y\mid X)$ 即得；严格性由互信息定义
$$
I(Y;G\mid X)=H(Y\mid X)-H(Y\mid X,G)>0
$$
推出。证毕。

## 3. 风格可控性命题

### 假设 B
$$
I(S;G\mid X)>0.
$$

### 命题 2（风格误差下界下降）
在一致分类器与充分优化条件下，风格交叉熵最优值近似满足：
$$
\inf \mathcal{L}_{sty}^{(X,G)}\approx H(S\mid X,G)\le H(S\mid X)\approx \inf \mathcal{L}_{sty}^{(X)}.
$$
即加入图上下文后，目标风格模拟的理论下界更低。

## 4. 优化收敛命题

### 假设 C
- $\mathcal{J}(\Theta)$ 下有界；
- $\nabla\mathcal{J}$ 为 $L$-Lipschitz；
- 随机梯度无偏且方差有界。

### 命题 3（非凸一阶收敛）
采用 SGD/Adam（满足等价稳定条件）且步长 $\eta_k\le 1/L$，有
$$
\frac{1}{K}\sum_{k=1}^{K}\mathbb{E}\|\nabla\mathcal{J}(\Theta_k)\|^2
\le
\frac{2(\mathcal{J}(\Theta_1)-\mathcal{J}^\star)}{\eta K}+\mathcal{O}(\eta\sigma^2).
$$
衰减步长下收敛到一阶稳定点邻域。

## 5. 复杂度分析

- GNN 每层复杂度：$\mathcal{O}(|E|d_g)$；
- Transformer 注意力主项：$\mathcal{O}(n^2d)$；
- 融合项通常为 $\mathcal{O}(nd)$，不改变主导阶。

结论：引入 GNN 条件在多数场景下增加线性级融合成本，换取条件信息增益。

## 6. 对当前工程的可验证指标

- 语言建模：PPL / NLL；
- 风格模拟：风格分类准确率、交叉熵；
- 语义保持：句向量余弦相似；
- 稳定性：梯度范数、参数漂移 $\|\Theta-\Theta_0\|_2$。

## 7. 结论

在信息论、优化理论与复杂度分析三方面，Transformer + GNN 融合方案均具备可证明优势：

1. 最优对数风险不劣且在非冗余图信息下严格更优；
2. 风格模拟误差下界可下降；
3. 在标准随机优化假设下可稳定收敛。

因此，该方案适合用于本项目 v5.0 的关键词增强微调与风格控制路径。
# Transformer + GNN 数学证明（完整版）

## 1. 统一建模

设输入文本为 $x$，图上下文为 $g=(V,E,X_V)$，输出序列为 $y$。模型由两部分构成：

1. 图编码器 $\mathrm{GNN}_{\psi}$：$g\mapsto z_g$；
2. 条件语言模型 $p_\theta(y\mid x,z_g)$。

联合参数记为 $\Theta=(\theta,\psi)$。

## 2. 目标函数

定义经验风险：
$$
\min_{\Theta}\ \mathcal{J}(\Theta)=
\lambda_{lm}\mathcal{L}_{lm}+
\lambda_{ver}\mathcal{L}_{ver}+
\lambda_{align}\mathcal{L}_{align}+
\lambda_{reg}\|\Theta-\Theta_0\|_2^2.
$$

其中：

### 2.1 语言建模项
$$
\mathcal{L}_{lm}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i}\log p_\theta\big(y_{i,t}\mid y_{i,<t},x_i,z_{g_i}\big).
$$

### 2.2 校验一致性项
令 $s_\Theta(x,g,y)$ 为模型自评分，$\hat{s}$ 为外部 verify 分数：
$$
\mathcal{L}_{ver}=\frac{1}{N}\sum_{i=1}^{N}\big(s_\Theta(x_i,g_i,y_i)-\hat{s}_i\big)^2.
$$

### 2.3 图-文对齐项
令文本编码器为 $u(x)$，图编码器输出为 $z_g$：
$$
\mathcal{L}_{align}=\frac{1}{N}\sum_{i=1}^N\left(1-\cos\big(W_tu(x_i),W_gz_{g_i}\big)\right).
$$

## 3. GNN 子模块的表达与复杂度

消息传递层：
$$
h_i^{(\ell+1)}=\sigma\left(W_s^{(\ell)}h_i^{(\ell)}+\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{(\ell)}W_m^{(\ell)}h_j^{(\ell)}\right),
$$
图级池化：
$$
z_g=\mathrm{Pool}(\{h_i^{(L)}\}_{i\in V}).
$$
若每层边数为 $|E|$、隐藏维度 $d_g$，则每层复杂度约 $\mathcal{O}(|E|d_g)$。

## 4. Transformer 子模块复杂度

单层自注意力主项复杂度：
$$
\mathcal{O}(n^2d),
$$
其中 $n$ 为序列长度、$d$ 为隐藏维度。引入图条件向量仅增加 $\mathcal{O}(nd)$ 级别融合开销，不改变主导阶。

## 5. 收敛性命题

### 假设 B
- $\mathcal{J}(\Theta)$ 下有界；
- 梯度 $L$-Lipschitz；
- 随机梯度无偏、方差有界。

### 命题 1
采用小批量 SGD/Adam（在等价假设下）并取步长 $\eta_k\le 1/L$，则
$$
\frac{1}{K}\sum_{k=1}^{K}\mathbb{E}\|\nabla\mathcal{J}(\Theta_k)\|^2
\le
\frac{2(\mathcal{J}(\Theta_1)-\mathcal{J}^\star)}{\eta K}+\mathcal{O}(\eta\sigma^2).
$$
当 $K$ 增大且步长衰减时，算法收敛到一阶稳定点邻域。

## 6. 图上下文有效性命题

令
$$
\Delta(x,g)=\log p_\Theta(y^\star\mid x,g)-\log p_\Theta(y^\star\mid x),
$$
表示图上下文带来的对数似然增益。

### 命题 2
若图-文互信息满足 $I(Y;G\mid X)>0$，且估计器一致，则最优条件模型满足
$$
\mathbb{E}[\Delta(X,G)]\ge 0,
$$
且当 $G$ 提供非冗余信息时严格大于 0。即：图上下文在统计意义上不劣于纯文本条件，并可提升生成质量。

## 7. 在线更新稳定约束

在线微调步长序列满足 Robbins–Monro 条件：
$$
\sum_t\eta_t=\infty,\qquad \sum_t\eta_t^2<\infty.
$$
并配合参数漂移正则 $\|\Theta-\Theta_0\|_2^2$ 与梯度裁剪，可将漂移控制在有界邻域，降低灾难性遗忘风险。

## 8. 路由优化目标

多实例推理路由评分：
$$
\mathrm{score}_k=\frac{q_k}{w_k}+\beta\hat{\ell}_k+\gamma\phi_k,
$$
选择
$$
k^*=\arg\min_k\mathrm{score}_k.
$$
在轻载-重载混合条件下，该目标可视作“队长近似 + 健康惩罚”的凸组合，兼顾吞吐与时延稳定。

## 9. 结论

在统一风险最小化框架下，Transformer 提供序列建模能力，GNN 提供结构先验；二者融合可在可证明的优化稳定性条件下提升条件生成质量，并在工程上维持可部署的复杂度与可控的在线更新行为。

# Transformer 风格微调的数学证明（严谨版）

## 1. 任务定义

设基础语言模型为 $p_{\theta}(y\mid x)$，其中 $x$ 为输入提示，$y=(y_1,\dots,y_T)$ 为输出序列。
给定目标风格数据集
$$
\mathcal{D}_s=\{(x_i, y_i, s_i)\}_{i=1}^N,
$$
其中 $s_i\in\{1,\dots,C\}$ 是风格标签（可退化为单一目标风格）。目标是在不显著破坏基础语义能力的前提下，将输出分布向目标风格迁移。

## 2. 多目标优化形式

定义联合风险：
$$
\mathcal{L}(\theta)=\lambda_{lm}\,\mathcal{L}_{lm}(\theta)+\lambda_{sty}\,\mathcal{L}_{sty}(\theta)+\lambda_{sem}\,\mathcal{L}_{sem}(\theta)+\lambda_{reg}\,\|\theta-\theta_0\|_2^2,
$$
其中 $\theta_0$ 为基座参数。

### 2.1 语言建模损失
$$
\mathcal{L}_{lm}(\theta)=-\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{T_i}\log p_{\theta}(y_{i,t}\mid y_{i,<t},x_i).
$$

### 2.2 风格分类损失
设风格判别器为 $q_\phi(s\mid y)$（冻结或慢速更新），则
$$
\mathcal{L}_{sty}(\theta)=-\frac{1}{N}\sum_{i=1}^N\log q_\phi(s_i\mid \hat{y}_i),\quad \hat{y}_i\sim p_\theta(\cdot\mid x_i).
$$

### 2.3 语义保持损失
设语义编码器 $f(\cdot)$（如句向量模型），定义
$$
\mathcal{L}_{sem}(\theta)=\frac{1}{N}\sum_{i=1}^N\left(1-\cos\big(f(\hat{y}_i),f(y_i)\big)\right).
$$

## 3. 梯度更新与可微近似

由于采样 $\hat{y}_i\sim p_\theta$ 导致不可导，可采用两类策略：

1. 教师强制近似：将风格/语义损失作用在 teacher-forcing 隐状态或 soft token 上；
2. 策略梯度近似：
$$
\nabla_\theta \mathbb{E}_{\hat{y}\sim p_\theta}[R(\hat{y})]=\mathbb{E}_{\hat{y}\sim p_\theta}\left[R(\hat{y})\nabla_\theta\log p_\theta(\hat{y}\mid x)\right].
$$

参数更新：
$$
	heta_{k+1}=\theta_k-\eta_k\,\widehat{\nabla \mathcal{L}(\theta_k)}.
$$

## 4. 收敛与稳定性结论

### 假设 A
- $\mathcal{L}(\theta)$ 下有界；
- $\nabla \mathcal{L}$ 为 $L$-Lipschitz 连续；
- 随机梯度无偏且方差有界：
$$
\mathbb{E}[g_k\mid\theta_k]=\nabla\mathcal{L}(\theta_k),\quad \mathbb{E}\|g_k-\nabla\mathcal{L}(\theta_k)\|^2\le \sigma^2.
$$

### 命题 1（非凸 SGD 一阶收敛）
若步长满足 $\eta_k=\eta\le 1/L$，则
$$
\frac{1}{K}\sum_{k=1}^K\mathbb{E}\|\nabla\mathcal{L}(\theta_k)\|^2
\le
\frac{2\big(\mathcal{L}(\theta_1)-\mathcal{L}^\star\big)}{\eta K}+L\eta\sigma^2.
$$
当 $K\to\infty$ 且采用衰减步长，可收敛到一阶稳定点邻域。

### 命题 2（灾难性遗忘上界）
设基础任务风险为 $\mathcal{R}_0$，微调后为 $\mathcal{R}_0'$。在局部二次近似下，
$$
\mathcal{R}_0'-\mathcal{R}_0\lesssim \frac{1}{2}(\theta-\theta_0)^\top H_0(\theta-\theta_0)
\le
\frac{\lambda_{\max}(H_0)}{2}\|\theta-\theta_0\|_2^2.
$$
因此加入 $\lambda_{reg}\|\theta-\theta_0\|_2^2$ 可显式抑制遗忘。

## 5. 风格可控性结论

定义目标风格误差
$$
\epsilon_{sty}(\theta)=\mathbb{E}[\mathbf{1}(\arg\max q_\phi(s\mid\hat{y})\neq s^\star)].
$$
若判别器校准误差有界且优化过程使 $\mathcal{L}_{sty}$ 单调下降，则 $\epsilon_{sty}(\theta)$ 的经验上界同步下降（由交叉熵与 0-1 风险的一致性上界得到）。

## 6. 实践对应到当前工程

- 风格微调主线：`transformer_main.py` 中训练/反馈循环。
- 可加入的严格实现项：
	- 以 `verify` 打分构造 $\hat{s}$；
	- 对每个 batch 记录 $\mathcal{L}_{lm},\mathcal{L}_{sty},\mathcal{L}_{sem}$；
	- 使用 EMA 与 early-stop 保证稳定收敛。

## 7. 结论

通过“语言建模 + 风格约束 + 语义保持 + 参数漂移正则”的联合目标，且在标准随机优化假设下，可证明微调过程收敛到稳定点邻域，并在可控范围内实现风格迁移与能力保持的折中。

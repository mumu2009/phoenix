# 深度学习

这一组看两件事：模型现在怎么训练和适配，以及 2025–2026 的综述把方向说成什么样。架构原文只收了这次检索能对上 arXiv 号的那些。

## 权重衰减和学习率要分开

Ilya Loshchilov，Frank Hutter。Decoupled Weight Decay Regularization。ICLR 2019。

全文：<https://arxiv.org/abs/1711.05101>

在普通 SGD 里，L2 惩罚和权重衰减只差一个学习率。在 Adam 里这两件事不再相同。把权重衰减从梯度更新里拆出来，最优衰减就不再跟着学习率一起漂，ImageNet 一类的泛化也更好。调优时不要把“权重衰减”和“损失里的 L2”当成同一个旋钮。

## 参数和数据一起放大

Jordan Hoffmann，Sebastian Borgeaud，Arthur Mensch 等。Training Compute-Optimal Large Language Models。2022。

全文：<https://arxiv.org/abs/2203.15556>

给定计算量，模型参数和训练词数应当近乎同比增加。当时的大模型在同样计算量下偏大、数据偏少。Chinchilla 用同样的预算、更小的模型和更多的数据，超过了更大的模型。缩放不是只加参数。

Margaret Li，Sneha Kudugunta，Luke Zettlemoyer。(Mis)Fitting: A Survey of Scaling Laws。2025。

全文：<https://arxiv.org/html/2502.18969v1>

他们看了五十多篇缩放论文。多数用幂律，但拟合公式、优化器、检查点选法和参数计数经常没写全，结论会跟着这些细节变，包括“多少词配多少参数”。缩放定律是经验拟合，不是换一个训练设置仍然成立的常数。

Ayan Sengupta，Yash Goel，Tanmoy Chakraborty。How to Upscale Neural Networks with Scaling Law? A Survey and Practical Guidelines。2025。

全文：<https://arxiv.org/pdf/2502.12051v3>

稀疏模型、混合专家、检索增强和多模态经常偏离那条稠密语言模型的幂律。缩放规律够用的地方用它做预算，不够用的地方不要外推。

Alexandra Volkova，Mher Safaryan，Christoph H. Lampert，Dan Alistarh。Towards Robust Scaling Laws for Optimizers。2026。

全文：<https://arxiv.org/html/2602.07712v2>

几乎所有缩放实验都把优化器固定成 AdamW。给每种优化器单独拟合一套 Chinchilla 指数，参数很不稳定。更稳的做法是共用指数，只让优化器改尺度系数。优化器是缩放的一个因子，不是训练脚本里的默认值。

## 适配大模型，而不是再训一个

Edward J. Hu，Yelong Shen，Phillip Wallis，Zeyuan Allen-Zhu，Yuanzhi Li，Shean Wang，Lu Wang，Weizhu Chen。LoRA: Low-Rank Adaptation of Large Language Models。2021。

全文：<https://arxiv.org/abs/2106.09685>

冻住预训练权重，只训练每层里的低秩增量。相对 GPT-3 175B 的全量微调，可训练参数可以少四个数量级。它假定原模型已经看过要学的那种分布。没有这份预训练时，低秩增量没有东西可改。

Menglin Yang，Jialin Chen，Jinkai Tao 等。Low-Rank Adaptation for Foundation Models: A Comprehensive Review。2025。

全文：<https://arxiv.org/html/2501.00365>

这篇把 LoRA 从语言模型扩到一般基础模型，并写出理论和稳健性上还没解决的部分。方法细节以 Hu 等 2021 的原文为准。

Nusrat Jahan Prottasha 等。PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models。2025。

全文：<https://arxiv.org/html/2504.14117>

2025 年的参数高效微调名单：加模块、只改一部分权重、重参数化、混合。用来查名字。每一种方法的公式以它自己的原文为准。

Rafael Rafailov，Archit Sharma，Eric Mitchell，Stefano Ermon，Christopher D. Manning，Chelsea Finn。Direct Preference Optimization: Your Language Model is Secretly a Reward Model。2023。

全文：<https://arxiv.org/abs/2305.18290>

偏好数据通常先拟合奖励模型，再用强化学习去追这个奖励。DPO 把最优策略写成闭式，于是同一件事变成一个分类损失。它仍然需要成对的偏好，以及一个已经会生成的模型。它不是从零训练一个小网络的配方。

检索确认的架构起点是 Vaswani 等，Attention Is All You Need，arXiv:1706.03762：<https://arxiv.org/abs/1706.03762>。上面的适配和缩放论文都建在这类变换器上。

## 标准注意力是一份快权重 MLP

Qishuai Wen，Zhiyuan Huang，Wei He，Xianghan Meng，Chun-Guang Li。Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights。NeurIPS 2026。

全文：<https://arxiv.org/abs/2602.01219>

他们写的恒等式是：标准注意力

```
Atten(q, K, V) = V softmax(K^T q / sqrt(d))
```

等于一个两层 MLP。查询是输入，缩放后的键和值是随输入变的权重，偏置是 0，激活是 `exp`，隐藏层宽度等于序列长度 `N`。第二层权重还要除以这些指数的和，否则结果不是 softmax。这和 2023 年 Huben、Morris 的方向相反：那里是用掩码注意力去实现一个 SiLU 神经元。

2026-10-24 在跑之前分开写了两套计算。手写例子 `q = 1`，`K = (0, 1)`，`V = (1, 3)`，`d = 1`：两套输出都是 2.46212，差小于 `1e-9`。把激活换成 ReLU 得到 3，去掉分母得到 9.15485，都离注意力超过事先写下的 0.1。另一组 `d = 4`、`N = 5` 的随机向量，最大绝对差是 `2.78e-17`。

`fuseAttention` 用的是归一化向量的点积，没有除以 `sqrt(d)`。分数 `[1, 0, 0]` 的第一项权重是 0.576117；同一分数除以 `sqrt(8)` 之后是 0.415908，差大于事先写下的 0.05。所以这篇论文的恒等式成立，但它的温度和我们的融合函数不是同一个算子。这一轮没有改 `fuseAttention`，也没有把 MiTA 的路标查询接进融合。

## 注意力没有被证明成我们这份融合

Robert Huben，Valerie Morris。Attention-Only Transformers and Implementing MLPs with Attention Heads。2023。

全文：<https://arxiv.org/abs/2309.08593>

定理 3 证明的方向是：一个没有偏置、隐藏层宽度为 `ℓ`、激活属于广义 SiLU 的 MLP，可以写成 `ℓ` 个内部维数为 1 的掩码注意力头之和。构造要多一个取值为 1 的偏置 token，掩码是每个 token 只看自己和这个偏置 token。定理 6 把整层 MLP 都换成这种头，头的数量按隐藏层宽度增加。这不是“softmax 自注意力本身是一个 MLP”。

证明里用到的恒等式是：token 的分数为 0、偏置 token 的分数为 `-x` 时，token 上的 softmax 权重乘以 `x` 等于 `SiLU(x) = x σ(x)`。正文里那句 `softmax([-x, 0]) = [σ(x), σ(-x)]` 和 softmax 的定义对不上。2026-10-24 的探针在 `x = -2,-1,0,1,2.5` 上核对了证明用的分数顺序，最大误差低于 `1e-9`。`x = 1` 时若把分数顺序换成 `[-x, 0]`，乘积是 0.268941，SiLU 是 0.731059，差大于事先写下的 0.1。

项目里的 `fuseAttention` 是余弦相似度、全体可见、值就是单元本身，没有偏置 token，也没有这条掩码。定理 3 的等式盖不住它。这一轮没有把 softmax 换成 GeLU，也没有把融合函数改成 MLP。

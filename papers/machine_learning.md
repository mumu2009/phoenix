# 机器学习

这一组收的是现在还能打开、并且直接关系到调优的机器学习论文。学习理论的课本没有放进来，因为这次没有打开对应的开放全文。

## 随机搜索是基线

James Bergstra，Yoshua Bengio。Random Search for Hyper-Parameter Optimization。Journal of Machine Learning Research 13:281–305，2012。

全文：<https://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>

多数数据集上只有少数超参数真正起作用，而且换一个数据集，起作用的是另外几个。网格把预算花在那些不起作用的轴上。随机搜索用同样的预算覆盖更多有效组合。后面的自适应调参要和这条基线比，而不是和手调的网格比。

## 评估很贵时用贝叶斯优化

Jasper Snoek，Hugo Larochelle，Ryan P. Adams。Practical Bayesian Optimization of Machine Learning Algorithms。2012。

全文：<https://arxiv.org/abs/1206.2944>

把泛化成绩看成高斯过程的一个样本。下一步试什么，由预测均值和不确定度一起决定。试验时间不一样、可以并行时，获取函数也要把成本算进去。它适合一次评估要跑很久的配置，不适合把每一次小改动都当成一次黑箱试验。

## 可以提前停止时用 Hyperband

Lisha Li，Kevin Jamieson，Giulia DeSalvo，Afshin Rostamizadeh，Ameet Talwalkar。Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization。Journal of Machine Learning Research 18:1–52，2018。

全文：<https://arxiv.org/abs/1603.06560>

先给很多配置很少的资源，只把还活着的配置留下来加资源。在深度网络和核方法上，它比当时的贝叶斯优化快一个数量级以上。前提是中途的成绩能够预测最后的成绩。做不到这一点时，提前停止会杀掉后来才变好的配置。

## 调参本身会过拟合验证集

Bernd Bischl，Martin Binder，Michel Lang，Tobias Pielok，Jakob Richter，Stefan Coors，Janek Thomas，Theresa Ullmann，Marc Becker，Anne-Laure Boulesteix，Difan Deng，Marius Lindauer。Hyperparameter Optimization: Foundations, Algorithms, Best Practices, and Open Challenges。

全文：<https://arxiv.org/pdf/2107.05847v3.pdf>
期刊页：<https://doi.org/10.1002/widm.1484>

这篇把随机搜索、贝叶斯优化、Hyperband、多点并行和嵌套验证放在一处。一个什么都不学的分类器，调得越久，越可能碰上一次碰巧好看的验证分。所以选择配置的那一层验证，不能同时又用来报告最终成绩。对我们来说，外部评分是外层。它只检测、只返回分数，不写进被调的那一层。

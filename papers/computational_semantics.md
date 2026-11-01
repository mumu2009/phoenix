# 计算语义

这一组看“意义”在计算里被定义成什么。分布假设、词向量、形式和意义的分界，都在这里。已经用在索引里的集合频率公式不重复写，见 `doc/v8.3/memetics_library/` 的 Spärck Jones 1972。

## 分布语义要有自上而下的目标

Guy Emerson。What are the Goals of Distributional Semantics? ACL 2020。

全文：<https://aclanthology.org/2020.acl-main.663.pdf>
同一篇：<https://arxiv.org/abs/2005.02982>

分布假设是：一个表达式出现的上下文告诉你它的意义。这条线从 Harris 1954、Firth 1957，到计数向量、Mikolov 等 2013 的词向量，再到上下文向量。Emerson 的论点是，只在下游数据集上加分，看不出这门学科要去哪。自上而下的目标是刻画一门语言里全部话语的意义，并且同时守住表达力和算得动。

Gemma Boleda。Distributional Semantics and Linguistic Theory。Annual Review of Linguistics 6:213–234，2020。

出版页：<https://doi.org/10.1146/annurev-linguistics-011619-030303>

本次打开的是摘要和出版页，不是开放全文。分布语义给出的是多维、有等级、从语料里估计出来的词表示。综述集中在语义变化、多义与组合、以及语义和句法、构词的界面。它说明计算语义已经能做分级相似性，理论语义仍在追真值条件。两边还没有合成一个模型。

## 全局共现，而不是只看窗口

Jeffrey Pennington，Richard Socher，Christopher D. Manning。GloVe: Global Vectors for Word Representation。EMNLP 2014。

全文：<https://nlp.stanford.edu/pubs/glove.pdf>
会议页：<https://aclanthology.org/D14-1162/>

词向量里的线性类比来自全局共现矩阵上的加权对数双线性回归，不是来自“把窗口里的词平均一下”。计数模型浪费在整个稀疏矩阵上，窗口模型又没用上全局统计。对我们已有的 `(词, α)` 索引，这篇的用处是提醒：特异性是文集上的量，一个窗口里的共现不能代替它。

## 只有形式时，不要把它叫做理解

Emily M. Bender，Alexander Koller。Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data。ACL 2020。

全文：<https://aclanthology.org/2020.acl-main.463/>

只在形式上训练的系统，先验地学不到意义。意义要连到形式之外的东西。大模型在任务上的成功，不等于它持有话语的意义。这和模因这边的区分同向：词和权重是索引，被复制的结构是另一件事。把索引得分高写成“理解了”或“这是一个模因”，都越过了证据。

## 上下文向量和词类型不是同一层

Marianna Apidianaki。From Words to Instances and Back: A Survey of Approaches to Meaning Representation and Interpretation。Computational Linguistics，2024-12-05。

出版页：<https://submissions.cljournal.org/index.php/cljournal/article/view/1946>

本次打开的是摘要。预训练模型给出的是词元位置上的表示。词的意义在更高的类型层。现在的工作在做两头：从词元向量收回类型向量，或者把静态词向量和知识注回词元向量。计算语义当前的进展是承认这两层，而不是用其中一层取代另一层。

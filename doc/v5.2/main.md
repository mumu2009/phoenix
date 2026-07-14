# v5.2 总览

## 1. 当前状态

v5.2 不是“全部算法目标均已完成”的终态版本，而是一个由三部分组成的混合状态版本：

1. 已落地并可运行的工程能力。
2. 已封装但仍需加强的实验性能力。
3. 尚未完成的研究型目标。

如果只看代码现状，v5.2 当前已经完成的重点是：

1. API、benchmark、智能评测三套测试链路。
2. GGUF 读取与本地后端监督器。
3. 数学桥接能力与 `outsides/_calculator` 接入。
4. 人格调度的基础控制与自治栈运行时。
5. 外部集全量 benchmark 和 PDF 汇总链路。

## 2. 已落地能力

### 2.1 测试与评估

1. 覆盖率口径与执行入口已整理，见 [coverage.md](coverage.md)。
2. 性能与质量联合 benchmark 已落地，见 [benchmark.md](benchmark.md)。
3. API 文档与接口契约已整理，见 [api.md](api.md)。

### 2.2 运行时与外部后端

1. 已具备 GGUF 解析能力与状态暴露能力。
2. 已具备基于外部工具的数学桥接能力。
3. 已具备人格调度相关的运行时基础设施。

## 3. 仍在进行中的研究目标

以下目标在 v5.2 中仍属于研究或部分实现状态，而不是已完成特性：

1. 词语到词元级语义拆分与合并算法。
2. Transformer 权重到神经动力学参数的完整映射。
3. 基于梯度下降的通用参数拟合模块。
4. 按需滑动窗口权重装载机制。
5. LoRA 转换限制的系统化评估与回退策略。
6. 完整的外部工具沙箱、权限边界与审计体系。

这些项的实时状态以 [todos](todos) 为准。

## 4. 配套文档

1. 算法过程：[algorithm.md](algorithm.md)
2. 数学论证：[mathproof.md](mathproof.md)
3. v5.1 残余任务闭环：[v5.1_carryover_closure.md](v5.1_carryover_closure.md)
4. 术语统一：[glossary.md](glossary.md)
5. 输入输出规范：[io_spec.md](io_spec.md)
6. API 文档：[api.md](api.md)
7. 覆盖率说明：[coverage.md](coverage.md)
8. 基准测试说明：[benchmark.md](benchmark.md)
9. 发布说明：[release.md](release.md)
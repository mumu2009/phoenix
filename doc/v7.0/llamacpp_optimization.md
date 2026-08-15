# llama.cpp 优化路线：上游更新、剪枝与块稀疏矩阵乘法

实现见：`tools/llama_prune_analyzer.py`（剪枝分析器）、`sparse_block_matmul.{hpp,cpp}`（块稀疏参考实现）。本文给出理论、复杂度与 ggml 集成规格；按“不编译不测试”约定，ggml 内核补丁留给便宜模型验证后落地。

---

## 1. 上游更新核查（2025-10 检索）

llama.cpp（ggml-org）持续活跃：近期 PR 包括 CUDA 对 FP32 KV cache 启用 Flash Attention（#16546）、Metal 新增 CONV_TRANSPOSE_2D（#16542）等。结论：**先 diff 仓库内 vendored 的 outsides/llamacpp 与上游 master**，把上游的量化/调度改进同步进来，再叠加本项目的剪枝/稀疏补丁（补丁方式沿用 llama_server_mods/*.patch 的既有机制）。

---

## 2. 权重剪枝：只看矩阵、不看输出（对应“最后一层矩阵”的要求）

**重要性定义（幅度剪枝，Han et al. 2015）**：

$$ \text{importance}(W_{ij}) = |W_{ij}| $$

**逐层自适应阈值**：每张张量（含最后一层矩阵）用**自身**幅度分布的 (100·s) 分位数作阈值，s 为目标稀疏率：

$$ \tau_t = Q_{|W_t|}(s) $$

幅度低于 τ_t 的权重被置 0（或抽离）。这与“从最后结果（最后一层矩阵）判断权重无意义”的直觉一致：**不需要任何前向传播**，直接按矩阵自身的统计判死权重；若未来可接受一次校准前向，可升级为 Wanda（Sun et al. 2023）的重要性 |W_ij|·‖x_j‖₂（权重 × 输入激活范数）或 SparseGPT（Frantar & Alistarh 2023）的逐层重建补偿，二者都需要少量校准数据。

**保留原始数据**：`llama_prune_analyzer.py --backup-dir` 复制原始 GGUF 并写 SHA-256 manifest（可回滚）。

---

## 3. 块稀疏矩阵乘法（跳过 ×0）

### 3.1 为什么是“块”而不是“逐元素”

- 逐元素判 0 需要在每次乘加前分支，破坏 SIMD 向量化，分支预测错误率高——实测往往比稠密更慢。
- 块级判断（如 32 行一块）的成本是 O(blockRows·K) 次比较，摊到 blockRows·K·N 次乘加上是 **1/N**，N 大时可忽略；跳过一个块省 blockRows·N 次乘加。

### 3.2 正确性与误差界

阈值 τ = 0（只跳过真零块）：**精确**，结果与稠密逐位一致。τ > 0（近似）：对跳过的行块，每一行 r 的误差满足（Cauchy–Schwarz）：

$$ |A_r \cdot B_{\cdot j}| \le \|A_r\|_2 \cdot \|B_{\cdot j}\|_2 \le \tau\sqrt{K}\, \|B_{\cdot j}\|_2 $$

聚合得到 Frobenius 界：

$$ \|C - C'\|_F \le \tau \cdot \sqrt{E_{\text{skip}}} \cdot \max_j \|B_{\cdot j}\|_2 $$

其中 E_skip 为被跳过条目的总数（见 `frobeniusErrorBound`）。

### 3.3 复杂度

- 稠密：O(M·N·K)；块稀疏：O((1−s)·M·N·K) + O(M·K) 判零，s 为块级稀疏率。
- 预期收益：剪枝 s=0.3 且块内全零时，理论加速 ≈ 1/(1−s) ≈ 1.43×（CPU 大 N 场景；N 小或向量化不充分的场景收益更低）。

---

## 4. ggml 集成规格（供便宜模型落地）

1. **稀疏标记**：剪枝工具输出 keep-mask（.npz）；写入 GGUF 时可在张量元数据加 `phoenix.sparse=1`，或直接用“全零块计数”在加载时识别。
2. **内核挂点**：`ggml_compute_forward_mul_mat`（ggml.c）中按 `ne11`（K 维）切行块；对 src0 的每 32 行块做一次 `|w| < τ` 检查，命中则跳过（C 该块置 0）。与现有线程分块（nth 维）正交。
3. **收益条件**：仅当 src0 的行块整体低于阈值时跳过；细粒度随机零不会形成空块，故剪枝时应优先**块结构化**（整块置零优先于散点置零）——可把 `llama_prune_analyzer.py` 的阈值改为“先按块均值排序，整块删除直到达到 s”。
4. **验证顺序**（便宜模型）：先用 `sparse_block_matmul` 参考实现单测“τ=0 精确一致 + τ>0 误差在界内”，再写 ggml 补丁，最后跑 perplexity 回归。

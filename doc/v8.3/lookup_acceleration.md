# 查表加速与多 token 批量解码（v8.x A3/A5 设计与可行性）

## 1. 用户假设与数学结论

**假设**：decoder-only 中层间图（权重矩阵）是静态的，可否查表直接得到下一层输出，
而不是矩阵计算。

**结论（诚实）**：
- 层间映射 h^{l+1} = f_W(h^l) 中 W 静态但 h 是高维连续向量——精确查表等价于
  存储 R^4096 → R^4096 的整个函数，表大小指数爆炸，**不可能**。
- 但量化把函数的取值域收缩到有限集：Q4_K_M 权重 ∈ 16 值，若激活也量化（Q8_0 ∈ 256 值），
  则单次乘法 a×w 的全部可能结果 = 256×16 = 4096 项，**可以枚举**（用户修正 3 成立）。
  预计算 4096 项 LUT（16KB，L1 常驻），每元素计算变为一次查表 + 累加。
- 风险：A55 的 L1 load ≈4 cycle，而整数 FMA 1 cycle/2 条——**查表未必更快**。
  `tools/layer_lookup_probe.cpp` 在 RDK 实测裁决（数据填入下方表格）。

## 2. 三档落地方案

### L1 输出级缓存（已实现，本版本）
- `mission_workspace.hpp`: `workspaceCacheGet/Put`——(scope, prompt) 4-gram Jaccard ≥
  `mission.cacheSim`(0.95) 判定『模型会重复输出』，直接跳过 LLM（省 ~2min/tick）；
- 连续 3 次命中 → 强制推进提示 + temperature 0.7 打破循环；
- 零质量风险（高阈值 + scope 隔离 + 每 scope 64 条 LRU）。

### L2 层间原型查表 + 量化枚举 LUT（实验，本版本交付工具与协议）
- **原型查表**：每层 hidden state K-means（每层 2048 原型）→ 最近原型输出 + 残差校正。
  误差界：输出误差 ≤ ‖残差‖·‖W‖（残差线性项，可证明）。
- **量化枚举 LUT**：如上 §1。等价性：整数域精确（FMA 与 LUT 结果逐位一致，probe 工具
  校验 exact=yes）。
- 基准裁决表（测试者填）：

| 配置 | FMA ns/elem | LUT ns/elem | 加速比 | 裁决 |
|---|---|---|---|---|
| RDK A55, D=4096, rows=4096 | 待填 | 待填 | 待填 | 待填 |

- 若 LUT 胜 → 规格书写 llama.cpp 内核补丁（Q4 权重量化激活枚举路径）；
  若 FMA 胜 → L2 封存，能量转向 L2'（见 §3 批量解码的权重复用才是 A55 的真正杠杆）。

### L3 完整层内查表（封印）
- 层内图（激活模式）随输入动态变化，记录层内动态图 = 记录所有可能输入，
  信息论上等价于训练集本身——文档封印，重开条件：出现可枚举的有限激活模式域。

## 3. 多 token 批量解码（用户核心洞察：『多个 token 当作一个 token 计算』）

**事实**：
- decode 每 token = 各层 GEMV（矩阵-向量）；prefill 多 token = GEMM（矩阵-矩阵）。
- A55 上 GEMM 每 token 成本显著低于 GEMV（权重复用，访存摊薄）。
- 自回归因果约束只要求位置 i 不看到位置 >i——**并行猜测多个位置**用上一轮上下文是合法的
  （Jacobi 迭代），猜测错的 token 验证后拒绝即可（无损）。
- 用户对齐要求：词表投影（detokenizer 矩阵）随多 token 批量计算（logits GEMM 批处理）；
  tokenizer/embedding 侧多 token 输入本就是 GEMM；多模态四部件的概念向量按同一批对齐。

**协议**（见 parallel_decode_spec.md）：请求 `n_parallel=N` + `parallel_mode=jacobi|speculative`；
服务器内部完成并行猜测 → 因果验证 → commit 接受前缀；响应仍是普通 message（客户端无感知）。
gateway 侧已透传（llama_split_backend_client.cpp + mission deliberator opts）。

**风险**：接受率 <100%，Jacobi 收敛轮数不定；实测接受率 <70% 时默认保持关闭（config
`llama.parallelDecode.enabled=false`）。


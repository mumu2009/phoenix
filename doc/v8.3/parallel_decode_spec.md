# 多 token 批量解码协议规格书（v8.x A5 - llama-server 侧，由廉价 AI 实现）

**目标**：在 llama-server（RDK A55 上 8B Q4_K_M）上把 decode 的 N 次 GEMV 合并为 GEMM，
即『多个 token 当作一个 token 计算』。客户端协议不变（响应仍是普通 message）。

## 1. 请求扩展（/v1/chat/completions 与 /completion 同样适用）
```json
{
  "n_parallel": 4,            // 1 = 经典单 token 自回归（默认，行为不变）
  "parallel_mode": "jacobi"   // jacobi | speculative（可选，默认 jacobi）
}
```
- gateway 已透传：`llama_split_backend_client.cpp` textCompletionFallback；mission deliberator
  通过 config `llama.parallelDecode.{enabled,N,mode}` 打开。

## 2. 服务器端算法（Jacobi 并行解码）

1. 维护窗口 W = min(n_parallel, 剩余 max_tokens)，候选 tokens t[0..W-1] 初始为上一轮猜测
   （首轮用单 token 采样填 t[0]，其余用 top-1 延伸）。
2. 每轮：以当前上下文 + 候选序列做一次**批量前向**（W 个位置并行，各层 GEMM）；
   按 causal mask 逐位置校验：t[i] 与 argmax/logits 一致则接受，否则从该位置截断。
3. commit 接受前缀（加入 KV cache，等价于经典 decode 的确定性结果——**无损**）；
   未满 W 且未到 max_tokens 则再猜一轮，直到窗口全接受或达到最大猜测轮数（默认 8）。
4. 终止：max_tokens 用尽或 EOS。响应 message.content = commit 后的完整续写文本。

**注意**：Jacobi 轮次内不采样（取 argmax），只在 commit 后按请求采样参数对下一位置采样，
保证与单 token 路径概率等价（拒绝采样语义）。

## 3. 投机解码（parallel_mode=speculative，第二阶段）
- 同一模型加一个小的并行猜测头（Medusa 式，多位置 logits head），主模型一次批量验证；
- 验证通过率由因果 mask + logits 一致性判定，拒绝的位置回退主模型单 token。
- 本阶段先交付 jacobi；speculative 等 jacobi 数据（接受率）决定是否需要。

## 4. 对齐要求（用户点名）
- **词表投影（detokenizer 矩阵）**：W 个位置的 logits 一次 GEMM（batch），不能逐位置 GEMV；
- **embedding**：多 token 输入本就是 GEMM，无需改动；
- **多模态四部件**（图像/语音/传感器/结构化编码器）：概念向量注入按同一批位置对齐——
  当前网关侧逐 tick 注入，多 token 模式下注入序列与 token 位置 1:1 对应（服务端保证）。

## 5. 实现位置建议
- llama.cpp `llama_decode` 已有 batch 前向能力（prefill 即 batch）——核心工作是把
  「猜测→批量前向→causal 校验→commit」循环做进 server 的生成循环（server-context /
  sampling 层），不动 ggml 内核。
- 保持 `--parallel` slot 语义不变；n_parallel 是同一 slot 内的窗口大小。

## 6. 验收数据（测试者在 RDK 填）
| n_parallel | 接受率 | decode s/token | 加速比 | 结论 |
|---|---|---|---|---|
| 1 (基线) | 1.0 | 待填 | 1.0x | |
| 2 | 待填 | 待填 | 待填 | |
| 4 | 待填 | 待填 | 待填 | |
| 8 | 待填 | 待填 | 待填 | |

接受率 <70% → config 默认保持 enabled=false；≥70% 且有 ≥1.5x 加速 → 默认开 N=接受率最优值。

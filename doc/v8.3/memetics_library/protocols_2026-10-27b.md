# 2026-10-27b：纯续写模式的现场墙钟 + 自动子盒判定

## 10.12 现场 8B 的同一任务墙钟（初代 128 vs 10.11 进化出的 210.105）

**改动**：`mission.plainTextOnly=true`（回复只当纯散文追加，跳过 JSON/工具/计划
解析）；`mission.temperature=0`。这样每回合追加多少文本由 `num_predict` 直接
决定，回复形态不再是自变量。

**装置**：8B 8082，网关 5090，同一 goal（1200 信息字符），A=128（10.11 的初代），
B=210.105（10.11 进化到第 5 代的基因组），A B B A，各两次；网关超时 480 s。

**切线**：

1. B 的回合数中位 < A 的回合数中位。
2. B 的墙钟中位 < A 的墙钟中位。
3. 任一次 480 s 内没到 1200 字符即作废，重跑该次；不把作废当 0 秒。
4. 不因为 B 是“进化出来的”就放宽；128/210 都是程序自己的基因组。

**失败处理**：若 1/2 不成立，记录两份墙钟、每回合字符数、是否命中 EOS，
不改成 256/411 重跑。

**结果（2026-10-27 22:47–23:01，A B B A）**：**FAIL**。

| run | tokens | ticks | ms | info | valid |
| --- | --- | --- | --- | --- | --- |
| plain-a0 | 128 | 3 | 148770 | 1624 | 是 |
| plain-b0 | 210.105 | 2 | 182820 | 1743 | 是 |
| plain-b1 | 210.105 | 7 | 480223 | 903 | 否（超时） |
| plain-a1 | 128 | 4 | 186837 | 1690 | 是 |

B 唯一有效的一次是 2 回合 / 182.8 s；A 两次有效 3 回合 / 148.8 s、4 回合 /
186.8 s（中位 167.8 s）。切 1（B 回合更少）成立，切 2（B 墙钟更短）不成立：
B 的 2 回合比 A 的中位还慢 15 s。plain 模式确实去掉了 JSON/工具分支，但 8B
在增长的提示词上仍会在 EOS 处停，128 token 已经够写 ~500 字符/回合；更大的
预算并不减少总解码，只把每回合拉长。B1 还出现 7 回合只写 903 字符的坏回复。
结论：这台 8B 的 plain worker 上，预算 128 已经接近每回合有效输出的上限；
10.11 在 Granite 子盒上选出的 210 不能直接搬到 8B 父任务。失败保留。


## 10.12b 无人判定的子盒完成（autoChildDoneChars）

**改动**：`mission.autoChildDoneChars=1200`：子盒工作区
`deliverableInformationChars >= 1200` 时，loop 直接
`markMissionChildDone`；不需要人调 `/api/mission/child/done`，也不需要模型发
`{"done":true}`。

**装置**：父任务 128，3 个 Granite 子盒（同 10.11），G=1；不调用
`/api/mission/child/done`；父任务 report。

**切线**：

1. 日志里至少 2 次 `[auto-child-done]`，且没有任何手工 `/child/done` POST。
2. 父任务 report 后 manager 的 `genome.tokensPerTick` 等于某个已完成子盒的
   基因组（adoption 发生）。
3. 子盒工作区达到 1200 信息字符才判完成；没到的子盒不被判完成。

**失败处理**：若 1 不成立，检查 loop 是否读到了子盒工作区/门槛；若 2 不成立，
记录父任务与子盒的 completionMs，不改 `adoptEliteLocked`。

**结果（2026-10-27 23:30–23:37）**：**PASS**（3 条切线都成立）。

- 3 个子盒都由 loop 自动判完成：
  `[auto-child-done] child-1-1 chars=1260`、`child-1-2 chars=1675`、
  `child-1-0 chars=1846`；没有调用 `/api/mission/child/done`，也没有模型
  `{"done":true}`。
- 父任务 report 后 `genome.tokensPerTick` 128.0 → **148.106**，adoption 发生
  （child-1-1，completionMs 122007）。
- 只有 ≥1200 信息字符的子盒被判完成。

诚实注记：`completionMs` 仍是“出生到判定”，child-1-0 在状态里还是
`done=False/completionMs=0`（它在同一拍稍后被判完成）；选择仍偏向先被处理的
子盒，不是工作耗时。这条 PASS 只证明“自动判定 + adoption 接线通了”，
不证明 148.106 比 128 快——10.12/10.13 恰好证明 128 更快。修工作耗时指标
是下一步（见 HANDOFF 第六节）。


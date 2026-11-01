# 2026-10-28：工作耗时选择的现场 (1+λ)

10.11/10.12b 的选择用 `completionMs = markChildDone - bornMs`，子盒在父任务
tick 之后顺序处理，等待时间被算进成绩；10.12/10.13 证明 128 比 210 快，而
选择却把基因组推到 210。这里把选择改成**工作耗时**：
`MissionChild.workMs`（子盒自己 deliberator 的累计毫秒）、
`Mission.workMs`（父任务自己 tick 的累计毫秒），`adoptEliteLocked` 在两者
都有测量时比较 `workMs`，否则退回墙钟。loop 在每次父/子 deliberator 调用
前后计时并累计；子盒的自动判定（`autoChildDoneChars`）沿用累计值。

**装置**：plainTextOnly=true，temperature=0，autoChildDoneChars=1200；
父任务 8B，子盒 Granite，λ=3，G=3；初代 210.105（10.11 末代）。每代子盒
交付物到 1200 信息字符时由 loop 自动判完成，父任务 report 后 adoption。

**切线**：

1. 每代都发生 adoption（manager genome ≠ 父代）。
2. 每代被选中的子盒的 `workMs` < 同代其它已完成子盒（严格最小）。
3. 第 3 代基因组 < 第 0 代 210.105（向 128 回落），或第 3 代被选中子盒的
   `workMs` < 第 0 代被选中子盒的 `workMs`。
4. 没有人工 `/api/mission/child/done`；`[auto-child-done]` 至少出现 2 次/代。

**失败处理**：记录每代所有子盒的 genome/workMs/completionMs；若工作耗时
选择仍不回落，检查 loop 的计时是否把父任务等待算进了子盒（不应），以及
auto-done 是否在累计之后触发。

**结果（2026-10-28 00:24–00:41，G=3，λ=3，每代 3/3 自动判完成）**：**PASS**。

| 代 | 父代 tokens | 3 个子盒 tokens | 被选中（workMs 最小） | 采用 |
| --- | --- | --- | --- | --- |
| 0 | 210.105 | 237.002 / 186.636 / 170.378 | 170.378（workMs 18995，另两个 25614/19933） | 170.378 |
| 1 | 170.378 | 138.802 / 190.019 / 184.578 | 138.802（35123，另两个 41807/38808） | 138.802 |
| 2 | 138.802 | 177.469 / 145.915 / 146.580 | 177.469（19321，另两个 34442/36319） | 177.469 |

切 1（每代 adoption）、切 2（每代 adopted 子盒 workMs 严格最小）、切 4
（每代 3 次 `[auto-child-done]`、无人工 `/child/done`）都成立；切 3（末代
基因组 < 210.105）成立（177.469 < 210.105）。**选择第一次按工作耗时而不是
出生顺序**：第 0 代选了最小基因组的子盒（workMs 18.995 s），第 1 代继续
落到 138.802；第 2 代最快的子盒是 177.469（workMs 19.321 s，写得更长），
所以轨迹不是单调下降，但每一次都是当次真实工作耗时最短。与 10.12/10.13
（128 比 210 快）保持一致：选择不再被父任务等待时间带偏。

数据 `build/tmp/worktime_evo.out.log`、`worktime_evo_results.json`、
`live_gateway13.log`。


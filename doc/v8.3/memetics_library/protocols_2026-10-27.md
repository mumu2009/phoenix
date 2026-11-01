# 2026-10-27：监督者判定 + 现场 (1+λ) 选择

前两轮把现场路径和探针都补齐了，但 `adoptEliteLocked` 的现场触发一直缺一次
“子盒自己宣布完成”：8B 和 Granite 的子盒都不按契约发 `{"done":true}`。
这里换一种利用方式：**监督者的完成判定**本来就是 mission_layer 设计的一部分
（`completionTimeMs` 是给人看的信号），但只有父任务的 `/api/mission/report`，
没有给子盒的判定入口。补一个 `POST /api/mission/child/done`，让监督者（本轮的
人）把某个子盒判为完成；`markMissionChildDone` 已经会写 `completionMs` 并在
父任务完成时触发 `adoptEliteLocked`。

**装置**：主程序加路由后重编；网关 5090；父任务用 8B（8082），子盒用
`model=granite_moe_e0`（8100，1B，快，四代内可控）；同一 goal；子盒用空
subgoal（`assist with: <父 goal>`，同一任务）；λ=3。

**流程（每代）**：

1. 父任务用上一代被选中的基因组（第一代用 128，故意低于现场最优 256）。
2. 复制 3 个同任务子盒（不同变异基因组），记录各自的 `tokensPerTick`。
3. 每个子盒的工作区 `deliverable.md` 达到 1200 个信息字符时，监督者调用
   `POST /api/mission/child/done` 判它完成；记录它的 `completionMs`。
4. 三个都判完后，`POST /api/mission/report {goalAchieved:true}` 完成父任务；
   `adoptEliteLocked` 在父任务和已完成子盒里保留 `completionMs` 严格最短的
   基因组；读 `result.genome.tokensPerTick`。
5. 下一代用这个基因组；重复 3 代。

**切线**：

1. 每代确实发生一次 adoption（父任务完成后 manager 的 genome 等于某个已完成
   子盒的 genome，`tokensPerTick` 与父代不同）。
2. 第 3 代被选中的 `tokensPerTick` > 第 0 代（128）；或者末代达到 1200 的
   墙钟 < 第 0 代。两条至少一条成立。
3. 子盒只判“交付物达到 1200 信息字符”，不按模型自评；判完不删任何活着的
   个体；`maxReplicas=4` 不破；急停不绕过。
4. 监督者判定不是“自动 done”；这条明确写：现场这一轮是 human-in-the-loop，
   不是模型自判。探针 10.1b 的自动选择仍是另一条证据。

**失败处理**：若 1 不成立（例如 completionMs 没写、adoption 没触发），先修
路由/接线再跑；若 2 不成立，记录基因组轨迹和每代墙钟，不为了通过改 256/411
或改 `adoptEliteLocked` 的比较规则。

**结果（2026-10-27 19:05–19:26，G=3，λ=3）**：**PASS**。

| 代 | 父代 tokens | 3 个子盒 tokens | 被选中的子盒 | 采用的 tokens |
| --- | --- | --- | --- | --- |
| 0 | 128 | 131.453 / 124.487 / 115.408 | child-1-0（completionMs 195849） | 131.453 |
| 1 | 131.453 | 148.659 / 106.874 / 135.407 | child-1-2（258949） | 148.659 |
| 2 | 148.659 | 174.294 / 167.360 / 143.627 | child-1-0（228921） | 174.294 |

每代 3 个子盒都由监督者判完成（`POST /api/mission/child/done`），父任务随后
`report`；`adoptEliteLocked` 每代都触发了，manager 的 `tokensPerTick` 逐代
上升 128 → 131.453 → 148.659 → 174.294。切 1（每代 adoption）和切 2（末代
基因组 > 128）都成立。数据 `build/tmp/evo_live.out.log`、
`build/tmp/evo_live_results.json`、`build/tmp/live_gateway8.log`。

诚实注记：子盒的 `completionMs` 受父任务 8B 那一拍的时长影响（每轮循环先跑
父任务，再跑子盒），所以它不是干净的“同一任务墙钟”；这条只证明**现场选择
机制与基因组逐代上升**，墙钟的受控证据仍是 10.1b 探针。监督者判定是
human-in-the-loop，不是模型自判，也不是自动进化。

### 10.11b 继续两代，看是否收敛到现场最优附近

**改动**：从 10.11 末代的 174.294 继续 G=2（第 3、4 代），λ=3，装置与判定
完全相同。10.11 的切 1 不重复检验；这一条只看轨迹。

**切线**：

1. 第 4 代被采用的 `tokensPerTick` ≥ 220（向现场最优 ~256 靠近）。
2. 被采用的 `tokensPerTick` ≤ 1024（不越过基因组夹子）。
3. 若某代没有 adoption，记录该代，不重跑 10.11。

**失败处理**：若轨迹停在 174–220 之间，记录“变异步长 0.05 下 5 代只能走到
这里”，不改 `adoptEliteLocked` 的比较规则，也不改 256/411。

**结果（2026-10-27 19:28–19:41）**：**FAIL**（阈值差 10）。

| 代 | 父代 tokens | 子盒 tokens | 采用 |
| --- | --- | --- | --- |
| 3（h0） | 174.294 | 201.452 / 131.158 / 198.098 | 201.452 |
| 4（h1） | 201.452 | 210.105 / 178.744 / 202.684 | 210.105 |

两代都 3/3 判完成、都 adoption；但第 4 代 210.105 < 220，切 1 不成立。
把 10.11 和 10.11b 连起来，5 代轨迹是
128 → 131.453 → 148.659 → 174.294 → 201.452 → 210.105：每代都上升，
但步长 0.05 下后段每次只加 ~4–15%，5 代到不了 220。按失败处理记录，
不改比较规则、不改 256/411。数据 `build/tmp/evo_live_ext.out.log`、
`build/tmp/evo_live_ext_results.json`。




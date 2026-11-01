# 2026-10-28d：独立 alias 召回的 8107 调用数对照

C 的前几次现场尝试都卡在 self-mod 的 explore 列表：`rankLatentEdit` 先选了
文件，`chosen` 非空，MLP/条件模型块不执行。这里给 111 加了**独立 alias
召回**：在 explore 排序之前，用持久 alias 表（`meme_alias.json` 的路径 +
学到的词）构造文件集，对 `currentFile`（2–40 个 cond token）先跑
“MLP → 条件模型”（或由 `mission.routerBeforeCond=false` 反序）。它不再依赖
self-mod 的 explore 列表。

**装置**：SELF-MOD 任务，种子 `deliverable.md = "surpriseema"`，alias 表包含
`active_inference.cpp`（词 `surpriseema`）；8B 8082，网关 5090；每次一拍。
新顺序默认；旧顺序把 `mission.routerBeforeCond` 设为 false 并重启网关。

**切线**：

1. 新顺序出现 `[open-column] mode=ce alias=1 path=active_inference.cpp`。
2. 新顺序的 `[cond-count] alias=1 calls=0`（MLP 命中，条件模型未调用）。
3. 旧顺序同一任务出现 `[cond-count] alias=1 calls>=1`（条件模型先跑）；
   两次选中的路径相同，或新顺序不更差。
4. 不因为“新顺序应该更好”就跳过旧顺序；失败保留。

**失败处理**：若新顺序没有 `[open-column]`，记录 query token 数、alias 表的
词、router 的 margin/known；若旧顺序没有 `[cond-count]`，检查 8107 是否在线
（`condProfileExchangeCount` 为 0 表示没发交换）。两种失败都保留。

**结果（2026-10-28 18:47–18:55）**：**PASS**。

信号追踪先找到了失败点：self-mod worker 把 `deliverable.md`/`currentFile`
覆写成 1077–1186 字符的源码正文，用户描述（19 字符种子）在 alias 召回之前
就丢了；`[sig] alias-query len=1077`，`condTokens > 40` 使 alias 块整段跳过。
修复：deliverable >400 字符时用 **goal** 作为描述信号（稳定的用户输入），
而不是被源码替换的 deliverable。之后同一任务：

| 顺序 | `[sig] alias-query` | 路由器 | 8107 交换 | 路径 |
| --- | --- | --- | --- | --- |
| 新（MLP 先） | len=63, q=8, rows=4 | `idx=3 margin=1 known=1` | **0**（无 `[cond-count]`） | active_inference.cpp |
| 旧（cond 先） | len=63, q=8, rows=4 | 未运行 | **2**（`[cond-count] calls=2`） | active_inference.cpp |

新顺序：`[open-column] mode=ce alias=1 path=active_inference.cpp margin=1
known=1`，`[latent-edit] ... cond=0 mlp=1`；旧顺序：`[cond-count] alias=1
path=active_inference.cpp calls=2`，`[cond-hit] ... stable=1`，`[latent-edit]
... cond=1 mlp=0`。切 1（新顺序 `[open-column]`）、切 2（新顺序 0 次、旧
顺序 2 次）、切 3（同一路径）都成立。这就是“MLP 前置让 8107 调用数下降”
的第一对干净现场数字。数据 `build/tmp/live_gateway22.log`（新）、
`live_gateway23.log`（旧）。


# 2026-10-28b：路由器前置 vs 条件模型前置的 8107 调用数对照

目标 C 的核心是“MLP 路由器放在条件模型之前，让 8107 调用数下降”。代码已经
把顺序改成 词法 → 侧表 → **开放列 MLP** → 条件模型；这里用同一个 SELF-MOD
种子（`deliverable.md = "surpriseema"`，goal `SELF-MOD: make one small
clarifying source edit.`）在新/旧两种顺序下各跑一拍，数
`[cond-count] calls=` 的和、`[open-column]` 是否命中、选中的路径。

**开关**：`mission.routerBeforeCond`（默认 true）。false 还原旧顺序
（条件模型先，MLP 只在条件模型空时兜底）。

**装置**：8B 8082，网关 5090；同一个 SELF-MOD goal 和 `surpriseema` 种子；
每次只跑一拍（路由器/条件模型在模型调用之前就决定）；新顺序默认，旧顺序
需要重启网关并把 `mission.routerBeforeCond` 设为 false。

**切线**：

1. 两种顺序都选中同一个路径，或新顺序选中 `active_inference.cpp`（种子词的
   档案）；新顺序不能更差。
2. 新顺序的 `[cond-count] calls=` 总和 < 旧顺序（严格更少）。
3. 新顺序出现 `[open-column]`，旧顺序在同一拍也要么不出现，要么出现前已经
   调用了 cond。
4. 不因为“新顺序应该更好”就跳过旧顺序的对照；失败保留。

**失败处理**：若种子不触发路由器（`[open-column]` 不出现），换一个档案的
特异词（如 `homeostasis valence`、`mission lifecycle genome`）重跑，两种
顺序用同一个种子；仍不触发则记录“该种子不在路由器覆盖内”。

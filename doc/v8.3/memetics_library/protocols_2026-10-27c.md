# 2026-10-27c：父任务与 (1+λ) 同模型（Granite）的预算 A/B

10.12 在 8B 父任务上失败，原因是 8B 每回合在 EOS 处停，128 token 已经够写
~500 字符，210 只把每回合拉长。10.11 的基因组是在 Granite 子盒上选的，所以
在 8B 父任务上测它是不公平的。这里给 `/api/mission/assign` 加 `model`，
让父任务本身走 Granite（与 10.11 的子盒同一模型），再测 128 vs 210.105。

**改动**：`assignMission` 读到 `model` 时
`SummonRegistry::setScopeModel(missionId, model)`；父任务整拍走该端点。

**装置**：Granite 8100，网关 5090；plainTextOnly=true，temperature=0；同一
goal（1200 信息字符）；A=128，B=210.105；A B B A，各两次；480 s。

**切线**：

1. 两次运行都真的走了 Granite（日志 `[mission-deliberate] summon scope=...`
   对应父任务 id）。
2. B 的回合数中位 < A；B 的墙钟中位 < A。
3. 任一 480 s 未到 1200 作废重跑该次。
4. 不把 A 的 128 改成 256；不因为 B 是进化值就放宽。

**失败处理**：记录两份墙钟和每回合字符数；若 Granite 的 EOS 也让预算不敏感，
写明“这个模型上 128 已够”，不改 10.11 的 adoption 规则。

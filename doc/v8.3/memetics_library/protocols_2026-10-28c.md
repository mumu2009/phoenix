# 2026-10-28c：双载体现场 force 识别

目标 B 的“力在双载体现场识别”。单载体时 `populationFrequencySeries` 只能是
0/1，`identifyForce` 返回 undetermined；两个并发载体让频率出现 0.5 的内点。

**装置**：两个并发的 `PEER-EVOLVE SELF-MOD` 任务（`peer-a`/`peer-b`），种子
`surpriseema` / `homeostasis valence`；父模型 8B；`[replica-trace]` 的
`mark.instance` 在 `AI_INSTANCE_ID` 未设时用 `scope:<missionId>`，两拍分别
落两个载体。trace 目录先移走，保证序列干净。

**切线**：

1. trace 里至少 2 个 instance，`forceSamples ≥ 3`。
2. 频率序列出现 0.5 或其它内点（0<p<1）。
3. `force` 是 content/conformity/replicator/fixed 之一，不是 undetermined。
4. force 只写日志，不参与准入（与命题 9.2 一致）。

**失败处理**：若 force=undetermined，记录序列、instance 数、每拍的
`sourceEdit` 标志；若序列全是 0/1，说明两个载体没有同时开/关，检查
`mark.instance` 的 scope 回退是否生效。若 force=fixed，如实记为“固定力”，
不硬说成 content/conformity。

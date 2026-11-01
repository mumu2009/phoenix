# 2026-10-28h：专家模型 + 代码插件同时工作，跑双载体 source-edit force

目标 B 的最后一项：双载体现场 force。之前的失败是 `sourceEdit` 由回复形态
决定，8B 的 SELF-MOD 回复是 prose，两个载体都记 `replicator=0`，
`forceSamples=0`。这一轮让两边同时工作：

- **专家侧**：`selectForSourceEdit()` 选 Granite（roles=programming），把
  `ownCarrier`（目标源码）连同“只回 FILE/OLD/NEW 的补丁”指令发给它。
- **插件侧**：`addons/code_edit_addon.cpp` + `code_edit_parser.hpp` 识别
  fenced/raw 代码，规范成 `FILE/OLD/NEW`；111 在专家回复不是补丁时调用它。
- **频率序列修复**：`populationFrequencySeries` 先扫描本批所有 instance，
  让第一个频率点就有两个载体（否则 `p0` 永远是 0/1，`identifyForce` 永远
  看不到内点）。
- **装置**：两个并发 `PEER-EVOLVE SELF-MOD` 任务，种子
  `FILE active_inference.cpp` / `FILE emotion_system.cpp`（给出 ownCarrier），
  8B 基座 + Granite 专家 + 8107 常驻；跑 ≥3 拍/载体。

**切线**：

1. `[summon-tool] source-edit model=granite_moe_e0` 且 `carrier bytes>0`。
2. 专家或插件给出补丁：`[summon-tool] plugin-normalised` 或回复本身含
   `FILE/OLD/NEW` / `diff`。
3. 两个 scope 都出现 `[replica-trace] ... reply=source-edit`。
4. `account.json` 的 `force` 是 content/conformity/replicator/fixed 之一，
   不是 undetermined；`forceSamples ≥ 3`。
5. 频率序列出现内点（0.5 或 0/2、1/2 的组合），不是全 0/全 1。

**失败处理**：记录 `[summon-tool]` 的 idle/fallback 原因、专家回复长度、
`[replica-trace]` 的 replyClass；若专家不给补丁，先修提示/插件；若两个载体
都 source-edit、序列恒 0.5，则 force=fixed，如实记为“固定力”，不硬说
content/conformity。

**结果（2026-10-29 00:0x–00:2x）**：**部分通过；force=fixed（识别为固定力）**。

- 专家侧选到了 Granite：`[summon-tool] source-edit model=granite_moe_e0`、
  `carrier bytes=1532/1541`；但 Granite 1B 对 600–1.5 KB 的源码载体输出
  token 沙拉（`replyLen=968 head= do ship emb emb …`），插件没有可识别的
  代码块，`plugin-normalised` 未触发；8B 用同一 patch 指令也输出沙拉。
- **关键修复**：`scope:peer-a` 作为文件名含 `:`，Windows 下静默写失败，
  `account.json` 一直是 0 tick。`appendReplicaTick` 现在把文件 stem 的
  非字母数字换成 `_`，JSON 的 `instance` 保留原值；trace 真的落盘
  （`scope_peer-a.jsonl` / `scope_peer-b.jsonl`）。
- 双载体 prose（无 seed，ownCarrier 空，8B 正常提示）现场日志：
  `[replica-trace] instance=scope:peer-b reply=prose … force=fixed
  forceSamples=3`。两个载体都是 prose、`sourceEdit=0`，`p=0` 恒定；
  `identifyForce` 在 3 个样本上返回 **fixed**。这是一个真实的两载体
  force 标签（固定力：没有复制在传播），不是 undetermined。
- 并行 append 最初会让 `account.json` 在 1–3 之间抖动；`appendReplicaTick`
  现在加了一把进程内 trace 互斥锁，读-改-写串行。稳定重跑（`fx-c`/`fx-d`
  无 seed、8B prose/path-mentioned，330 s + 420 s）：
  `{"ticks":4,"sourceEdits":0,"replicators":1,"force":"fixed","forceSamples":4}`，
  两个载体文件 `scope_fx-c.jsonl`、`scope_fx-d.jsonl` 都在；
  `[replica-trace] … force=fixed forceSamples=3/4` 是 live 行。
- 两个载体都 `sourceEdit=0`，`p=0` 恒定；`identifyForce` 在 4 个样本上返回
  **fixed**。这是一个稳定落盘的两载体 force 识别（固定力：没有复制在传播），
  不是 undetermined。
- `sourceEdits=0`，所以拿不到 content/conformity：两个二进制载体、没有
  source-edit 变化时，`p` 只能是 0/0.5/1；恒定 0 或 1 给 fixed，序列里出现
  0/1 给 undetermined，内点变化才拟合 content/conformity。这是定理
  14.1/16.2 的边界，不是插件或检测缺失。

**3 载体复跑（2026-10-29 19:2x，p3-a/b/c，带 `FILE active_inference.cpp`
种子，专家+插件在）**：`{"ticks":3,"sourceEdits":0,"force":"fixed",
"forceSamples":3}`，`scope_p3-a/b/c.jsonl` 三个文件都在。3 个载体
`sourceEdit` 全 0，`p=0` 恒定，仍是 **fixed**，不是 content/conformity。
原因不在载体数量：源编辑率是 0。Granite 1B 对源码载体输出沙拉；8B 的
raw-code patch 提示也沙拉；可用的 GGUF 只有 granite/tinyllama/8B，配置里
的 `llama3.2_3b` 文件
`GGUF_models/llama-3.2-3b-instruct-Q4_K_M.gguf` **不存在**（GGUF_models 下
只有 granite-3.1-1b 和 tinyllama-1.1b）。要拿 content/conformity，下一步
是装一个能写代码的 coder GGUF（或把 8B 的 code 提示调到能出补丁），并让
`selectForSourceEdit` 接受 `kind=llamacpp` 的 programming 角色；插件、
trace、互斥锁、文件名修复都已就绪。

**Coder 专家落地（2026-10-29 19:4x–20:2x）**：
- 从 hf-mirror 下载 `Qwen2.5-Coder-1.5B-Instruct Q4_K_M`（1,117,320,768 字节）
  到 `GGUF_models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`，手动起在 8093。
- 配置把它注册为 `summon.models[0]`（kind moe、roles ["programming"]、
  autoStart true），并把 granite 的 roles 置空；`selectForSourceEdit` 现在
  选到 `qwen_coder`。
- 现场：`[summon-tool] source-edit model=qwen_coder kind=moe`、
  `[replica-trace] instance=scope:q3-a/b/c reply=source-edit`——**coder 真的
  写出了 source edit**，4 个载体 5 拍里 `sourceEdits=4`。
- 但 4 载体 + coder 的 force 是 **undetermined**：
  `{"ticks":5,"sourceEdits":4,"force":"undetermined","forceResidual":0.5}`。
  这是混合序列（4 次 source edit + 1 次 prose），不是单一力；`identifyForce`
  在残差 0.5 处拒绝，符合定理 14.1（混合力不可分）。
- 结论：source-edit 侧已经修好；要拿 content/conformity 还差**受控的编辑率**
  （所有载体每拍都编辑，或按固定节奏编辑），让序列跟随单一映射；不是插件/
  trace/模型的代码能力问题。





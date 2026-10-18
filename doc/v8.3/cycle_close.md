# 本周期非训练线收口（2026-10-16）

只记录这一轮亲手核对过的事实。没有重跑 `remote_merge_results.py`，没有重写论文，没有连 RDK，没有重链 `phoenix_main`，没有下载 MoE 权重，没有改 `agnets_role.md`。训练锁 `runtime_store/gpu_train.lock` 当时存在，没有打开或改写。

## 二进制与进程

| 对象 | 核对结果 |
|---|---|
| `phoenix_main.exe` | 46,164,498 字节，mtime `2026-10-16 22:52:33` |
| 5080 | `127.0.0.1:5080` 由 PID 56156 监听。进程是这份 `phoenix_main.exe`，启动时间 `2026-10-16 22:53:51`，命令行 `--llamacpp-base-url=http://127.0.0.1:8082` |
| 8082 | `127.0.0.1:8082` 由 PID 38772 `llama-server.exe` 监听。命令行模型是 `GGUF_models\blobs\sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`，`--parallel 1`。本轮没有杀它，也没有请求它的 `/health` |
| 8100 | 收口开始和聊天之后各查一次，`netstat` 都没有 `:8100` |

二进制里能搜到字符串 `emergency-stop`（2 处）、`estopRequiresBlock`（1 处）、`pack_prompt=false`（1 处）、`[self-kernel]`（2 处）。抽查的源文件都不晚于这份二进制，因此没有重链：

- `main_hub_parts/067_class_memebarrier.inc` `2026-10-16 21:22:25`
- `main_hub_parts/111_class_gatewayserver.inc` `2026-10-16 22:08:02`
- `main_hub_parts/116_section_tail.inc` `2026-10-16 21:19:56`
- `module_overrides/llama_split_backend_client.cpp` `2026-10-16 21:13:58`
- `autonomy_stack.cpp` `2026-10-16 21:28:02`

没有做全库时间戳扫描。中途停掉了一次会走进训练目录的递归扫描。

## 1. 模因

`runtime_store/farm_state.json` 存在：358,897 字节，mtime `2026-10-16 21:12:36`。顶层键是 `testedIds`、`memes`、`antimemes`、`batches`。

- `testedIds` 6086 条
- `memes` 352 条
- `antimemes` 整数 5366
- `batches` 55 条

最后一批：`file=results_llama3.1_8b_v4_wikipedia2023_512.json`，`finishedAt=2026-10-16 21:12:28`，`rounds=6`，该批 `memes=4`，`antimemes=84`。本轮没有跑合并脚本。

论文：在 `doc/` 下递归查找 `*.tex`、`*.pdf`、`*.docx` 以及文件名含 paper / manuscript 的文件，结果为空。没有重写论文稿。

## 2. 情感

源码（`emotion_system.hpp`、`instinct.hpp`、`autonomy_stack.cpp`）：

- `fromFineAppraisal` 在 `fromAppraisal` 上加 `novelty` / `control` / `uncertainty` / `otherAgency`，中性通道与旧鉴赏一致。
- `mapChatSampling` 把温度夹在 `[0.15, 0.40]`，中性张量温度 `0.30`。
- `sensationFromThought` 的 `source` 是上下文标签加 `:thought`，空上下文写成 `detached:thought`，源里没有念头正文。
- 聊天 scope 的 `composedPrompt` 等于 `chatCausalText(用户句)`。念头写入 `result.selfMind`，不写入 `composedPrompt`。

`config/phoenix.json` 的 `emotion.enabled` 为 true。任务路径仍走原来的宽区间；只有聊天 scope 走 `mapChatSampling`。

驱动 `build/tmp/emotion_sampling_driver.exe`（mtime `2026-10-16 21:35:11`）比 `test/emotion_sampling_driver.cpp`、`instinct.cpp`、`self_mind.cpp`、`module_overrides/emotion_influence.cpp`、`emotion_system.hpp` 都新，没有重编。

本轮命令与结果：

```text
build\tmp\emotion_sampling_driver.exe
```

60 条 `PASS`，末行 `all passed`，退出码 0。

## 3. 安全

`addons/ThePlugInForSecurity/security_core.cpp` 以 `#include "hard_embed.cpp"` 编进主机。`067_class_memebarrier.inc` 的 `inspectText` 在急停闩上时先返回 `blocked=true`、`reason=emergency-stop`，然后才是私钥和词表拦截。`HardEmbedSoftBarrier.AnomalousMemeStillBlocksAndCleanDoesNot` 本轮通过，模因屏障没有放宽。

`config/phoenix.json`：`safety.estop.enabled=true`，`addons.security.enabled=false`。

安全线测试程序 `build/security_line_tests.exe` mtime `2026-10-16 21:31:07`。它链接的源（`test_hard_embed.cpp`、`security_core.cpp`、`hard_embed.cpp`、`meme_tensor.cpp`、`meme_semantic_slicer.cpp`、`meme_annealer.cpp`、`memetic_existence.cpp`、`graph_diffusion_summarizer.cpp`、`emergency_stop.cpp`、`instance_registry.cpp` 及对应头）都更旧，没有重编。

本轮命令与结果：

```text
build\security_line_tests.exe --gtest_filter=HardEmbed*
```

`[  PASSED  ] 12 tests.`，退出码 0。四组：`HardEmbedHash` 1、`HardEmbedTest` 9、`HardEmbedImage` 1、`HardEmbedSoftBarrier` 1。

## 4. 学习

`autonomy_stack.cpp` 启动时读配置：

| 键 | `phoenix.json` | 代码缺省 |
|---|---|---|
| `study.unitLearn.enabled` | true | `cfgOr` 缺省 true。类成员在 `configure` 之前是 false |
| `study.population.enabled` | false | `population_layer.hpp` 的 `enabled_` 缺省 false |
| `study.expert.enabled` | false | `tiny_expert.hpp` 的 `enabled_` 缺省 false |
| `mission.evolution.enabled` | false | 本轮没有把它改成启动 |

`iterate` 有用户文本时走 `unitLearner_.learnEpisode`，结果放在 `studyLearn`，并带 `writesCausalString: false`。族群层只在 enabled 且调用带 `populationGoal` 时跑。

召唤种类 `moe` 的字段在 `summon_registry.hpp`：`modelPath`（别名 `path` / `modelFile` / `gguf`）、`url`、`port`、`k`（别名 `K` / `activeExperts`）、`pinnedExpert`（别名 `forceExpert` / `expertId`）。`upsert` 不启动进程。

驱动 `build/study_development_driver.exe` mtime `2026-10-16 22:02:44`，比 `test/study_development_driver.cpp`、`study_unit_learn.cpp`、`population_layer.cpp`、`tiny_expert.cpp`、`summon_registry.hpp` 都新，没有重编。

本轮命令与结果：

```text
build\study_development_driver.exe
```

`STUDY_DRIVER passed=65 failed=0`，退出码 0。

这个驱动按自己的用例重写了 `runtime_store/experts/study_tiny/synthetic.json` 和 `weights.json`（mtime `2026-10-16 23:04:40`）。没有动 `gpu_train.lock`。

## 5. 运行时解码

`config/phoenix.json`：`main.inference.packPrompt=false`，`loopMode=unit`，`ngramMerge=2`，`llama.parallelDecode.enabled=false`（`mode` 仍写着 `jacobi`，开关是 false），`runtime_opt.speculative.enabled=false`。

`llama_split_backend_client.cpp` 的 `/phx/generate` 路径强制 `pack_prompt=false`、`loop_mode=unit`，并擦掉 `group_n`、`n_parallel`、`parallel_mode`。`causalBudgetNgram(false, ngram)` 在 pack 关闭时返回 1。聊天的 `selectCausalPrefix` 在 `skipGraphContext` 或非 `lowPriority` 时不把图上下文放进因果串。

驱动 `build/tmp/runtime_decode_driver.exe` mtime `2026-10-16 21:16:24`，比 `test/runtime_decode_driver.cpp` 和 `inference_unit_pipeline.hpp` 新，没有重编。

本轮命令与结果：

```text
build\tmp\runtime_decode_driver.exe
```

16 条 `PASS`，末行 `ALL PASS`，退出码 0。覆盖：pack 关闭时预算倍数为 1、长提示预算不按 2 倍放大、聊天因果串等于用户句且不含 `[self-kernel]` / `[Memory]` / GNN。

### 实机聊天

5080 仍是上面的 22:52 二进制。本轮：

```text
POST http://127.0.0.1:5080/api/chat
Authorization: Bearer local-cycle-close
{"text":"2+2?","sessionId":"cycle-close-causal","maxTokens":16}
```

HTTP 200。网关日志 `runtime_store/phoenix_runtime_decode_gateway.log` 新增：

```text
[gateway] /api/chat contextHint=empty, sessionId=cycle-close-causal
[gateway] /api/chat stage=model_start provider=llamacpp maxTokens=16
[llama] causal_prefix_chars=0 skip_graph=1 low_priority=0
[llama] pack_prompt=false budget_ngram=1 ngram_merge=2 loop_mode=unit causal_chars=4 self_kernel=0
[chat] llamacpp(split) ok=1 replyLen=26
[gateway] /api/chat stage=model_done provider=llamacpp elapsedMs=7373
[gateway] POST /api/chat -> 200
```

`causal_chars=4` 等于用户句 `2+2?` 的长度。前缀 0，`self_kernel=0`。日志行里没有 `group_n`。`ngram_merge=2` 只是配置窗口；`budget_ngram=1` 且 `pack_prompt=false`。

## 6. MoE

`config/phoenix.json` 的 `summon.moe` 与 `summon.models` 里都有 `granite_moe_e0`：`port=8100`，`k=1`，`pinnedExpert=0`（models 项用 `expertId=0`），`autoStart=false`，`moeEnv=LLAMA_FORCE_MOE_EXPERT`，`moeEnvValue=0`，`modelPath=GGUF_models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`。

该 GGUF 不在磁盘上。本轮没有下载。

正在跑的网关启动日志：

```text
[summon] moe name=granite_moe_e0 port=8100 k=1 pinnedExpert=0 env=LLAMA_FORCE_MOE_EXPERT=0 ggufPresent=0 autoStart=0 launch=0 reason=autoStart false
```

`111_class_gatewayserver.inc` 这条路径注释写明不 `CreateProcess` 第二个 llama-server。`launch` 固定打印 0。8100 没有被监听。权重没有加载。

驱动源在 `build/tmp/check_summon_moe.cpp`（mtime `2026-10-16 22:09:44`），exe mtime `2026-10-16 22:10:03`，比源、`summon_registry.hpp`、`config/phoenix.json` 都新，没有重编。

本轮命令与结果：

```text
build\tmp\check_summon_moe.exe
```

6 条 `PASS`（`three-models`、`upsert-no-start`、`granite-fields`、`kept-slots`、`autostart-off`、`env-switch`），末行 `ALL PASS`，退出码 0。它只读配置并登记，不听端口。

## 7. 自我层

`self_mind.hpp` / `self_mind.cpp` 仍在。`config/phoenix.json` 的 `selfMind.enabled=true`，`learnFromLiveTurns=false`，`corpusPath=./runtime_store/datasets/self_mind/sharegpt_zh_turns.jsonl`。代码缺省同样是 enabled true、live turns false。

`tickThought` 的结果进 `result.selfMind`。`autonomy_stack.cpp` 写明念头不写入 `composedPrompt`。实机日志 `self_kernel=0` 与此一致。

驱动 `build/tmp/self_mind_driver.exe` mtime `2026-10-16 21:16:36`，比 `test/self_mind_driver.cpp`、`self_mind.cpp`、`self_mind.hpp`、`doc/v8.3/self_mind_sample.json` 新，没有重编。

本轮命令与结果（仓库根目录）：

```text
build\tmp\self_mind_driver.exe
```

35 条 `PASS`，末行 `ALL PASS`，退出码 0。真实语料探针：`mirror.ownRecall=1.0`（6/6），`turing.accuracy=0.9921875`（127/128），两项 `pass=true`。

## 本轮测试一览

| 驱动 | 重编 | 结果 |
|---|---|---|
| `build\study_development_driver.exe` | 否 | passed=65 failed=0 |
| `build\tmp\emotion_sampling_driver.exe` | 否 | 60 PASS，`all passed` |
| `build\tmp\self_mind_driver.exe` | 否 | 35 PASS，`ALL PASS` |
| `build\security_line_tests.exe --gtest_filter=HardEmbed*` | 否 | 12 tests PASSED |
| `build\tmp\runtime_decode_driver.exe` | 否 | 16 PASS，`ALL PASS` |
| `build\tmp\check_summon_moe.exe` | 否 | 6 PASS，`ALL PASS` |
| `POST /api/chat` 文本 `2+2?` | 不适用 | HTTP 200；`pack_prompt=false` `budget_ngram=1` `loop_mode=unit` `causal_chars=4` `self_kernel=0` |

没有测试失败，所以没有改算法、没有改接线、没有改测试断言。

## 已知缺口

- 论文稿：`doc/` 下没有 tex/pdf/docx。本轮没有重写，也没有全盘搜索 `outsides/`、`GGUF_models/`、`runtime_store/`。
- MoE 权重 `GGUF_models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf` 不在磁盘上。8100 没有进程。不要把配置登记当成已加载。
- RDK 本轮没有连接，也没有在板上跑这些驱动。
- `llama.parallelDecode.mode` 配置文本仍是 `jacobi`，但 `enabled` 为 false。客户端发送前擦掉 `group_n`。本轮没有打开 Jacobi、speculative 或 `group_n`。

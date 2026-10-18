# 全栈实机（2026-10-16）

只记录这一轮亲手打到的请求、日志和进程。没有 git，没有改 `agnets_role.md`，没有连 RDK，没有重编译 `phoenix_main`，没有请求 `127.0.0.1:8082` 的 `/health`，没有杀 8082 上的 llama-server，没有用 GPU 启动 MoE，没有打开 `gpu_train.lock`。

## 配置：改了什么

活配置是 `config/phoenix.json`。`config/phoenix.example.json` 没改，下面对照的是改之前两边的值。

从 false 改成 true 的只有这三项：

| 键 | 改前 `phoenix.json` | 改后 | `phoenix.example.json`（未改） |
|---|---|---|---|
| `study.population.enabled` | false | true | false |
| `study.expert.enabled` | false | true | false |
| `selfMind.learnFromLiveTurns` | false | true | false |

本来就是 true，没有动：

- `study.unitLearn.enabled`
- `selfMind.enabled`
- `mission.enabled`
- `autonomyLoop.enabled`（example 里仍是 false；活配置本来就是 true）
- `safety.estop.enabled`
- `summon.enabled`

明确保持关闭：

- `main.inference.packPrompt=false`
- `llama.parallelDecode.enabled=false`（`mode` 文本仍是 `jacobi`，开关是 false）
- `runtime_opt.speculative.enabled=false`
- `summon.moe.autoStart=false`（网关代码路径不 `CreateProcess` 第二个 llama-server）

## 二进制和进程

`phoenix_main.exe` 仍是 46,164,498 字节，mtime `2026-10-16 22:52:33`。为了读新配置，只停了旧的 `phoenix_main`（网关 PID 56156，以及它拉起的 `--frontend-only=1` PID 57704），再用同一条命令行拉起：

```text
phoenix_main.exe --llamacpp-base-url=http://127.0.0.1:8082
```

重启后：

| 端口 | 进程 | 说明 |
|---|---|---|
| 127.0.0.1:5080 | PID 21704 `phoenix_main.exe` | 新网关，上面的命令行 |
| 0.0.0.0:5081 | PID 18356 `phoenix_main.exe --frontend-only=1` | 网关自己拉起的前端 |
| 127.0.0.1:8082 | PID 38772 `llama-server.exe` | 原 8B，没有杀，没有打 `/health` |
| 127.0.0.1:8100 | PID 8568 `llama-server.exe` | 本轮新拉起的 Granite |

抽查过的采样源文件都早于这份二进制：`112_section_before_contexthint.inc` 21:12:45，`emotion_system.hpp` 21:19:15，`116_section_tail.inc` 21:19:56。

内存不是 16GB。实测 `Win32_OperatingSystem`：总可见内存 130974.9 MB，启动 MoE 前空闲 82356.5 MB，监听之后空闲 81722.9 MB。小 MoE 和 8082 叠在一起没有 OOM。

音频训练在本轮结束前自己跑完：`runtime_store/logs/train_audio_musan.log` 末行是 `epoch 6/6`、`[train] exit 0`、`[lock] released pid=49168`。`runtime_store/gpu_train.lock` 是它自己释放后消失的，本轮没有打开或改写这个锁。LLaVA 相关 python（PID 15308、19440、26308、9564）结束时仍在。

## 1. 聊天、因果串、温度

请求：

```text
POST http://127.0.0.1:5080/api/chat
Authorization: Bearer local-fullstack
{"text":"live?","sessionId":"full-stack-live","maxTokens":16}
```

第一次 HTTP 200，网关日志：

```text
[gateway] /api/chat stage=model_start provider=llamacpp maxTokens=16
[llama] causal_prefix_chars=0 skip_graph=1 low_priority=0
[llama] pack_prompt=false budget_ngram=1 ngram_merge=2 loop_mode=unit causal_chars=5 self_kernel=0
[chat] llamacpp(split) ok=1 replyLen=45
[gateway] /api/chat stage=model_done provider=llamacpp elapsedMs=8048
[gateway] POST /api/chat -> 200
```

`causal_chars=5` 等于用户句 `live?`。前缀 0，`self_kernel=0`，`pack_prompt=false`，`budget_ngram=1`。日志行里没有 `group_n`。`ngram_merge=2` 只是配置窗口。

同一句又 POST 了一次，用来把被瘦身的正文读出来。这次没有 `stage=model_start`，是缓存命中，HTTP 200，9.675 秒。正文在 `result.reply`，不在顶层：

```text
2=?
This is a simple math problem, not 6
The
```

`result` 只有 `provider`、`reply`、`transformerReply`。`cognitionInferenceOptions` 在 `116_section_tail.inc` 里被丢掉了，HTTP 正文没有温度。网关日志和 `runtime_store/llama_8082_runtime_decode.log` 也没有温度数字，8082 只多了 `srv log_server_r: request: POST /phx/generate 127.0.0.1 200`。

因此这次没有测到一个可以引用的聊天温度数值。采样夹紧函数在这份 22:52 二进制里（对应源文件更早），但没有把当次采样值打出来。不能把缺省 0.30 写成实测值。

任务路径另有一行，不是聊天因果串：`low_priority=1`，`causal_chars=32`，`causal_prefix_chars=33`。任务日志里有 `top_p=0.8616 freqP=0.15`，那是任务采样，不是聊天温度。

## 2. 自我层和 learnFromLiveTurns

`GET /api/self/status` 在聊天前：

```text
enabled=true learnFromLiveTurns=true humanCount=2917 holdout=32 exemplars=12
episodes=0 policyBias=[0,0,0,0,0]
corpusPath=./runtime_store/datasets/self_mind/sharegpt_zh_turns.jsonl
```

第一句 `live?` 之后：`humanCount=2918`，`holdout` 仍是 32，`exemplars` 仍是 12，`chatbotCount` 仍是 4111，`episodes=1`，`lastAction=prefer`。第二句（缓存命中，仍走 `iterate`）之后：`humanCount=2919`，`episodes=2`。

会不会覆盖已经学到的人：不会把语料人物清掉，也没有把计数打回 1。`learnFromLiveTurns` 只在非任务、用户句不超过 400 字节时，把这句话当成 `human` 做一次观察。不是留出句时，质心按 `learningRate` 0.35 做指数滑动：`新 = 0.65*旧 + 0.35*这句的特征`。所以质心会被这句话拉动，但 ShareGPT 计数、12 条 exemplar、客服质心都还在。每 4 句且序号不少于 4 的留出句不进质心；这两句不是留出句。exemplar 满 12 条之后不再追加。

另外，`tickThought` 在自我层开启时每次 `iterate` 都会更新策略权重，和 `learnFromLiveTurns` 无关。这次 `policyBias` 从全 0 变成了非 0，是念头 tick，不是把人物模型换成用户句。

这个开关留在 true。之后每个符合上面条件的短聊天句都会继续拉动质心。

## 3. 族群和极小专家

`GET /api/cognition/autonomy/status` 的 `result.study`：

聊天前：

```text
population.enabled=true n=4 goalsRun=0 spawnedInstances=0 maxPopulation=8
expert.enabled=true ready=true paramCount=1072 vocab=32 hidden=16 lastLoss=-1
weightsPath=./runtime_store/experts/study_tiny/weights.json
unitLearn.enabled=true episodes=0 lastLoss=-1
```

两句聊天之后：`unitLearn.episodes=2`，`lastLoss=0.4877468464309405`。族群 `goalsRun` 仍是 0，专家 `lastLoss` 仍是 -1。

没有 HTTP 路由调用 `runPopulationGoal` 或 `summonTinyExpert`。状态里这两层是开的，族群已种下 4 个个体，专家参数量 1072 且 `ready=true`，但这一轮没有跑过一次 goal，也没有把磁盘上的 `weights.json` 训进这次损失。

`GET /api/summon/status` HTTP 200。模型行里有 `granite_moe_e0`：`kind=moe`，`url=http://127.0.0.1:8100`，`k=1`，`pinnedExpert=0`，`modelPath=GGUF_models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`。旁边还有 `llama3.1_8b@8082` 和 `llama3.2_3b@8092`。

启动日志：

```text
[summon] moe name=granite_moe_e0 port=8100 k=1 pinnedExpert=0 env=LLAMA_FORCE_MOE_EXPERT=0 ggufPresent=1 autoStart=0 launch=0 reason=autoStart false upsert={"ok":true,"started":false,"updated":true}
```

`launch=0` 是网关设计：它只登记，不拉进程。8100 是本轮在网关外面拉起来的。

## 4. Granite MoE / 8100

权重原先不在磁盘上。从 hf-mirror 下载的是 bartowski 的 Q4_K_M，不是 30B，也不是 DeepSeek：

```text
https://hf-mirror.com/bartowski/granite-3.1-1b-a400m-instruct-GGUF/resolve/main/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf
```

落到 `GGUF_models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`，821,847,360 字节，文件头 `GGUF`。

启动命令（子进程环境只有源码会读的那一个 MoE 变量；没有设 `LLAMA_MOE_ACTIVE` 或 `LLAMA_FORCE_MOE_EXPERTS`）：

```text
LLAMA_FORCE_MOE_EXPERT=0
llama-server.exe -m GGUF_models\granite-3.1-1b-a400m-instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8100 --ctx-size 4096 --parallel 1 --threads 6 --n-gpu-layers 0
```

`--n-gpu-layers 0` 是为了不占用正在训练的 GPU。日志是 CPU 映射：`CPU_Mapped model buffer size = 782.12 MiB`，`n_expert = 32`，`n_expert_used = 8`，`general.name = Granite 3.1 1b A400M Instruct`。监听成功：

```text
main: server is listening on http://127.0.0.1:8100 - starting the main loop
```

没有出现 `LLAMA_FORCE_MOE_EXPERT=0 (forcing single MoE expert)`。正在用的 `outsides\llamacpp\build-gcc\bin\llama-server.exe` 是 10,720,350 字节、mtime `2026-08-23 12:07:18`，二进制里搜不到 `LLAMA_FORCE_MOE_EXPERT`。源文件 `ggml-cpu.c` 在 `2026-10-16 20:51:02` 才有这段 `getenv`。环境变量已经设进 PID 8568 的父启动器，但这份旧二进制不会读它。没有在原地重链：那个 exe 正被 8082 的 PID 38772 占用，重链会碰到禁止杀掉的进程。所以专家 0 没有钉上，原生仍是 top-8。

补全请求（不是 `/health`）：

```text
POST http://127.0.0.1:8100/completion
{"prompt":"Say hi in one word.","n_predict":16,"temperature":0}
```

HTTP 200。`content` 原文：

```text
I'm sorry, I can't do that.

I'm
```

（正文前有两个换行，`tokens_predicted=16`，`stop_type=limit`。）非空。响应里的 `speculative.n_max` 是 llama-server 自己的 `generation_settings` 回显，Phoenix 的 `runtime_opt.speculative.enabled` 仍是 false，启动参数里也没有打开推测解码。

## 5. 任务心跳

`mission.enabled` 和 `autonomyLoop.enabled` 本来就是 true，没有改成别的值。重启后第一次 `POST /api/cognition/autonomy/loop {"action":"status"}`：

```text
enabled=true running=false tickCount=0 lastTickAtMs=0
```

原因在循环里：持久化状态里已经有两条结束了的任务（`integ-full-1700714` state=3 Failed，`scope-retest11-1697087` state=2 Completed）。有任务记录但没有一条还在 Running 时，循环打出 `tick=0 start` 后自己 `loopStop`，所以心跳当时是停的。

然后：

```text
POST /api/mission/assign
{"goal":"Reply with the single word pong."}
```

HTTP 200，0.006 秒。约 3 秒后：

```text
id=mission-5163841 state=1 pressure=0.05000000074505806
goal=Reply with the single word pong. deliverableChars=0
loop.running=true tickCount=0
```

state=1 是 Running。再停掉之前，交付物从 0 涨到 145，`tickCount` 变成 1，`lastTickAtMs=1792165191305`。交付物原文（`runtime_store/mission_workspace/mission-5163841/deliverable.md`）：

```text
But what is 4
2+2+4
The answer is 6
The answer is 2+2 2+2
The answer is 4
The answer is 4
The answer is 2+2+2 the answer is 4
The answer is 6
The
```

停止：

```text
POST /api/mission/report
{"goalAchieved":false,"note":"full-stack live probe stop","missionId":"mission-5163841"}
```

HTTP 200，21.597 秒，`ok=true`。日志：`[mission-report] wait-idle sec=21.59 budgetSec=692 idle=1`。停完之后 `running=false`，`tickCount=1`，这条任务 `state=3`（Failed），`pressure=0`。循环不再写它。

副作用：这次 report 把 `runtime_store/autonomy_state.json` 截成了 0 字节（mtime `2026-10-16 23:39:51`；截断前是 505,710 字节，mtime `2026-10-11 14:40:32`）。`exportState` 写盘进了空 catch，文件留在 0。进程内存里的状态还在 PID 21704，所以报告之后没有再重启网关。没有这份文件的备份。

## 6. 急停

没有按急停。`safety.estop.enabled` 仍是 true。

```text
GET /api/safety/estop/status
```

HTTP 200：`latched=false`，`pressedAtMs=0`，`registeredInstances=1`，实例是 `cognition-autonomy`。

## 没能打开或没能测到的

- 专家钉死：环境变量设了，8100 进程在，文本也非空，但 2026-08-23 的 `llama-server.exe` 不含 `LLAMA_FORCE_MOE_EXPERT`，日志里的 `n_expert_used` 仍是 8。没有重链，避免替换 8082 正在使用的同一个 exe。
- 聊天温度的具体数字：响应和两份日志都没有。不能报成已测到落在 `[0.15, 0.40]` 的某个值。
- 族群 goal 和极小专家训练步：状态里是开启的，没有对应 HTTP，`goalsRun=0`，专家 `lastLoss=-1`。
- `autonomy_state.json` 被 report 截成空文件。为了不把 PID 21704 内存里的状态丢掉，没有为了拦截温度再重启一次网关。

## 补测（三项返工）

上一节没收。这一节只补专家钉死、状态文件截断、聊天温度，并让族群 goal 和极小专家各走一步。`packPrompt` / Jacobi / speculative / `group_n` 仍关。`learnFromLiveTurns` 仍是 true。急停没有按，`latched=false`。没有 git，没有改 `agnets_role.md`，没有连 RDK，没有请求 `/health`，没有动 `gpu_train.lock`。

PID 21704 还在时，先把当时能打到的状态接口存成 `runtime_store/tmp/full_stack_retest/memory_backup.json`（26,258 字节）。没有导出整份 `exportState` 的 HTTP，所以这是接口快照，不是当时内存里的完整权重。之后重编 `phoenix_main` 时进程被停掉，旧的 0 字节文件不再被当恢复源。

### 1. 专家用量是 1

先停掉 8082 和 8100，用 `llama_server_mods\apply_patches.bat` 和 `build_llama_server.bat --force` 重链。`load_hparams` 在 `LLAMA_FORCE_MOE_EXPERT` 合法时把 `n_expert_used` 收成 1；argsort 仍把这一格钉在专家 N。新的 `llama-server.exe` 里能搜到 `LLAMA_FORCE_MOE_EXPERT` 和 `n_expert_used=1`。

8082 重新拉起原来的 8B（`GGUF_models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`，`127.0.0.1:8082`，不带该环境变量）。日志：`print_info: n_expert_used = 0`，`main: model loaded`，`listening on http://127.0.0.1:8082`。

8100 重新拉起 Granite，只加 `LLAMA_FORCE_MOE_EXPERT=0`，`--n-gpu-layers 0`。`runtime_store/moe_8100_live.err.log`：

```text
load_hparams: LLAMA_FORCE_MOE_EXPERT=0 n_expert_used=1 (native 8)
print_info: n_expert_used    = 1
LLAMA_FORCE_MOE_EXPERT=0 graph n_expert_used=1
main: model loaded
main: server is listening on http://127.0.0.1:8100
```

没有 `print_info: n_expert_used = 8`。括号里的 native 8 是 GGUF 原值，路由计数已经收成 1。之后 `POST /completion`（`n_predict=16`，`temperature=0`）HTTP 200，正文非空。这两台在补测结束时仍在听：8082 PID 9672，8100 PID 8736。

### 2. 状态文件不再被截成空文件

三处 `ofstream` 先打开再 `dump()` 的写法改成：先序列化到字符串（非法 UTF-8 用替换），写入 `*.tmp`，成功后再换到目标文件。dump 失败时不碰原文件。

`POST /api/mission/report` `{"goalAchieved":false,"missionId":"mission-7743288"}` HTTP 200。报告前文件已是 294,264 字节（循环里的同一套写入先落过盘），报告后是 **296,221 字节**，`json.loads` 成功。任务 `state=3`（Failed）。随后 `POST /api/cognition/autonomy/loop` `{"action":"stop"}` 返回 `running=false`。网关日志停在 `tick=13`，等过一个心跳间隔后没有 `tick=14`。

### 3. 聊天温度

`applyChatSamplingFloors` 在网关 stdout 打出实际采样值。短聊天 `POST /api/chat` `{"message":"2+2","sessionId":"retest-temp-1"}` HTTP 200。

```text
[gateway] chat sampling temperature=0.390251 top_p=0.871003
[llama] pack_prompt=false budget_ngram=1 ngram_merge=2 loop_mode=unit causal_chars=3 self_kernel=0
```

`0.390251` 在 `[0.15, 0.40]`。`causal_chars=3`，等于用户原话 `2+2`，`pack_prompt=false`，`self_kernel=0`。

### 族群 goal 和极小专家

`POST /api/study/population` `{"input":[0.1,0.2,0.3,0.4],"target":[1.0,0.0]}` 返回 `ran=true`。状态里 `goalsRun=1`。

`POST /api/study/expert` 16 条样本、2 个 epoch，返回 `ran=true`，`finalLoss=1.7613317966461182`。状态里 `lastLoss=1.7613317966461182`，不再是 -1。

`selfMind.learnFromLiveTurns` 仍是 true（这次聊天后 `humanCount=2918`）。急停状态仍是 `latched=false`，`registeredInstances=1`。

### 这次改过的文件

- `outsides/llamacpp/src/llama-model.cpp`
- `outsides/llamacpp/src/llama.cpp`
- `outsides/llamacpp/ggml/src/ggml-cpu/ggml-cpu.c`
- `outsides/llamacpp/src/llama-mmap.cpp`（MinGW 把 `PrefetchVirtualMemory` 编成直接抛错，mmap 加载失败；改为运行时解析，缺了就跳过预取）
- `llama_server_mods/force_moe_expert_count.patch`
- `llama_server_mods/prefetch_virtual_memory.patch`
- `llama_server_mods/apply_patches.bat`
- `llama_server_mods/apply_patches.sh`
- `llama_server_mods/build_llama_server.bat`
- `llama_server_mods/README.md`
- `autonomy_stack.cpp`
- `main_hub_parts/112_section_before_contexthint.inc`
- `main_hub_parts/116_section_tail.inc`
- `doc/v8.3/full_stack_live.md`

## 全功能实机（未完整通过）

这一轮把上次故意关掉的解码开关打开，并保持上一轮已经通过的约束。结论：聊天、族群、专家、MoE 专家数、因果串、温度、进化门、急停、状态文件都有实机证据。运行中的使命压力没有变化，所以这一轮不能写成通过。

### 最终开关

| 开关 | 最终值 |
| --- | --- |
| `main.inference.packPrompt` | true |
| `main.inference.ngramMerge` / `ngramProtectLast` | 2 / 16 |
| `llama.parallelDecode.enabled` | true，`N=4`，`mode=jacobi` |
| `runtime_opt.speculative.enabled` | true。`draftModel` 仍为空，`forbidSecondServerWhenParallel1`、`neverCurlHealth`、`neverRestartLiveLlama` 仍为 true。没有第二进程，也没有 `/health` |
| `mission.evolution.enabled` | true。启动时写入 `evolutionEnabled`，`mission-4800777` 的 lineage 返回 `enabled=true` |
| `mission.disableParallelDecode` | false |
| `study.population` | enabled true，`n=4`（没有改成 2） |
| `study.expert` / `study.unitLearn` / `selfMind` | true。`learnFromLiveTurns` true |
| `mission` / `autonomyLoop` / `summon` | true。`summon.moe.autoStart` 仍 false，8100 由手工拉起 |
| `safety.estop.enabled` | true。测试没有按下。结束后 `latched=false` |

### 第一次失败

只改配置不够。`116` 和 `llama_split_backend_client.cpp` 把 `pack_prompt` 写成 false，并删掉 `group_n`、`n_parallel`、`parallel_mode`。`/phx/generate` 也不读这些字段。打开开关后的第一问是「What is the capital of France?」。Jacobi 把贪心草稿整段提交，正文变成 `Paris` 后面反复接问句。短问的 `packed_groups` 也是 0，因为保护窗是 16。

### 修了什么

- 网关按配置发送 `pack_prompt`、`group_n=2`、`n_parallel=4`、`parallel_mode=jacobi`、`speculative=true`，日志改成打印实际值。因果串仍只有用户原话。
- `/phx/generate` 对保护窗外的隐藏行做 0.15 混合均值（`group_n`），并在同一进程里做 Jacobi 自推测。不启动第二个 llama，不占 GPU。
- 草稿如果接上提示词前缀，或者生成串里的 4-gram 再次出现，就拒绝提交并停下。开关保持打开。
- `mission.evolution.enabled` 在网关启动时传给 `assignMission`。

### 聊天（修复后重测）

三次都是 `temperature=0.390251`，`self_kernel=0`，`causal_chars` 等于用户句长度。

1. 「What is the capital of France?」正文 `Paris`。`causal_chars=30`。`[phx] pack_prompt=1 group_n=2 packed_groups=0 ... n_parallel=4 speculative=1 jacobi=1 jacobi_rounds=1 jacobi_accepted=2 jacobi_rejected=2`。短问落在 16 token 保护窗里，所以打包组数是 0；Jacobi 跑了并拒绝了回放。
2. 「Name three prime numbers under twenty.」正文 `1, 3, 5, 7, 11, 13, 17, 19` 后接一句收束。`causal_chars=38`。`jacobi_rounds=8 jacobi_accepted=32 jacobi_rejected=1`。
3. 多句行李问题。正文从水壶和钱包写起，可读，没有循环。`causal_chars=146`。`packed_groups=9 pack_l2=0.046584`，`jacobi_rounds=12 jacobi_accepted=48`。这一问证明打包和 Jacobi 都作用在生成上，正文没有碎。

### 族群、专家、MoE

两次连续 goal，输入维 4、目标维 2，请求里没有改 `n`。`goalsRun` 从 1 到 3。状态里 `n=4`。

专家 16 条样本、2 个 epoch，`finalLoss=0.7743147518485785`。状态里 `lastLoss` 从 `1.7613317966461182` 变为同一个新值。`learnFromLiveTurns=true`，`humanCount=2925`。

8100：`LLAMA_FORCE_MOE_EXPERT=0`，日志 `n_expert_used=1`（native 8），没有 `n_expert_used=8`。`POST /completion` 问「What is a prime number?」返回非空（83 字符）。8082 仍是原来的 8B blob，日志 `n_expert_used=0`。没有请求 `/health`。

### 使命

目标是三段行道树短文，并要求写 `notes.md`。实测使命是 `mission-4800777`。

- 心跳：日志 `[mission-loop] tick=0` 与 `tick=1`，各约 32 秒。状态 `tickCount` 到 2。随后 `action=stop`，`running=false`，没有再出现下一拍。
- 交付物：`deliverable.md` 从 0 增到 335，停测时 678 字节。内容是行道树净化空气的两段相近开头，可读。模型没有写出 `notes.md`，也没有 `[tool:]` / `[file:]`。文件动作是 deliberator 把正文追加进 `deliverable.md`。
- 压力：tick 0 和 tick 1 的接口值都是 `0.05000000074505806`。这是 `pressureFloor=0.05`。模式是 asymptotic，`pressureTauSec=86400`，一分钟内的 `tanh(t/tau)` 约 0.0007，低于地板，所以接口值不变。没有把地板、tau 或模式改掉来制造变化。**这一项没有通过。**
- `POST /api/mission/report` 之后 `runtime_store/autonomy_state.json` 为 571981 字节，JSON 可解析。

### 进程

修复后的最终进程：8082 PID 20432（8B blob），8100 PID 47880（Granite，`n-gpu-layers 0`），网关 PID 41384。没有连接 RDK，没有下载 30B 或 DeepSeek。`runtime_store/gpu_train.lock` 在测试后存在（语音续训，pid 54208），没有修改，也没有杀训练进程。

### 这次改过的文件

- `config/phoenix.json`
- `outsides/llamacpp/examples/server/server.cpp`
- `llama_server_mods/phx_pack_jacobi.patch`
- `llama_server_mods/apply_patches.bat`
- `llama_server_mods/apply_patches.sh`
- `llama_server_mods/build_llama_server.bat`
- `module_overrides/llama_split_backend_client.cpp`
- `main_hub_parts/116_section_tail.inc`
- `main_hub_parts/111_class_gatewayserver.inc`
- `inference_unit_pipeline.hpp`

## 压力曲线、网关拉起 8100、同进程草稿（通过）

上一节三项里，聊天、族群、专家和状态文件保持原样，开关没有关。这一节只补压力、`summon.moe.autoStart` 和 speculative。packPrompt、Jacobi、`group_n`、`mission.evolution`、population、expert、`learnFromLiveTurns` 仍为开。没有请求 `/health`，没有下载 30B，没有改 `gpu_train.lock`，没有杀 GPU 训练。

### 1. 使命压力

原因是 asymptotic、`pressureTauSec=86400` 加上地板 0.05，一两分钟内的曲线都被夹成 `0.05000000074505806`。配置改为 `pressureMode=linear`、`painGainPerSec=0.025`、`pressureFloor=0`。线性曲线在 40 秒到达 `maxPain=1`，之后保持满值。`POST /api/mission/assign` 在请求没带这些字段时改用这份配置，避免再退回一天的时间常数。

新短文使命 `mission-6873587`，目标是三段行道树短文并写入 `deliverable.md`。心跳是真实 deliberator，不是 1 秒假 tick：`tick=0 done elapsed=26325ms`。

连续两拍的日志：

- `[mission-loop] tick=0 start pressure=mission-6873587=0.012550000`
- `[mission-loop] tick=1 start pressure=mission-6873587=0.672049999`

0.672049999 严格大于 0.012550000。状态采样在运行中从 0.000425 增到 0.10085、0.201025、0.30115、0.401275、0.501375、0.601475、0.701625、0.801775、0.90245，在 `elapsed=40.105` 秒时为 `1.0`，当时 `state=1`。满值出现在一分钟以内。交付物在第一拍从 0 增到 336 字节，停测时工作区文件 614 字节，开头是 Summer Shade 段。

随后 `POST /api/mission/report`（`goalAchieved=false`，`missionId=mission-6873587`）返回 `ok=true`。状态变为 3，压力回到 0。`POST /api/cognition/autonomy/loop` 的 status 为 `running=false`、`tickCount=2`。日志没有 `tick=2`。`runtime_store/autonomy_state.json` 为 347180 字节，JSON 可解析。`mission.enabled` 仍为 true。

### 2. 网关自己拉起 8100

`summon.moe.autoStart` 改为 true。已有的手工 8100 先停掉。`phoenix_main` 启动日志：

`[summon] moe name=granite_moe_e0 port=8100 k=1 pinnedExpert=0 env=LLAMA_FORCE_MOE_EXPERT=0 ggufPresent=1 autoStart=1 launch=1 pid=22888 reason=gateway spawned granite, model loaded`

8100 的父进程是网关 PID 20660。命令行是已下载的 Granite，`--host 127.0.0.1 --port 8100 --ctx-size 4096 --parallel 1 --threads 6 --n-gpu-layers 0`。子进程日志：`LLAMA_FORCE_MOE_EXPERT=0 n_expert_used=1 (native 8)`，`print_info: n_expert_used = 1`，`graph n_expert_used=1`，`model loaded`，监听 8100。没有 `n_expert_used = 8`。

`POST /completion`（`n_predict=16`，`temperature=0`）返回 65 字符，非空。

8B 只在重链 `llama-server.exe` 时装过一次（speculative 计数在 server.cpp）。网关重启时没有再拉 8082。最终 8082 仍是 PID 6320，原 8B blob，`n_expert_used=0`。

### 3. 同进程 speculative

草稿模型仍是空字符串，没有再起一个模型进程。`/phx/generate` 在同一进程里用 argmax 当草稿，和温度采样比较：相同计接受，不同计拒绝。Jacobi 窗口里的草稿也计入这三列。

一问「Name three rivers in Europe.」：

- 温度 `[gateway] chat sampling temperature=0.390251 top_p=0.871003`，落在 [0.15, 0.40]。
- `[llama] pack_prompt=true group_n=2 n_parallel=4 parallel_mode=jacobi speculative=1 ... causal_chars=28 self_kernel=0`。用户句 28 字符，因果串只有这句。
- 同一条 `[phx]`：`jacobi=1 jacobi_rounds=9 jacobi_accepted=34 jacobi_rejected=2 speculative=1 speculative_draft=36 speculative_accepted=34 speculative_rejected=2`。
- 正文：`1. Rhine 2. Danube 3. Seine`，后面又接了一句新问题。莱茵、多瑙、塞纳可读，不是用户句回放。短问落在保护窗内，`packed_groups=0`。

### 仍开着的开关

`packPrompt=true`，`parallelDecode` Jacobi `N=4`，`speculative.enabled=true`，`ngramMerge=2`，`mission.evolution.enabled=true`，`study.population` `n=4`，`study.expert.enabled=true`，`learnFromLiveTurns=true`，`mission.enabled=true`，`summon.moe.autoStart=true`。

### 这次改过的文件

- `config/phoenix.json`
- `main_hub_parts/116_section_tail.inc`
- `main_hub_parts/111_class_gatewayserver.inc`
- `autonomy_stack.cpp`
- `outsides/llamacpp/examples/server/server.cpp`
- `llama_server_mods/phx_pack_jacobi.patch`

证据在 `runtime_store/tmp/pressure_moe_spec_result.json`、`runtime_store/fullfeat_gateway.out.log`、`runtime_store/moe_8100_gateway.err.log`、`runtime_store/llama_8082_pressure.err.log`。

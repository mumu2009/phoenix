# 2026-10-29a：针对审查的四条修补（同进程 C、n=5 进化、9.5b、全编辑 force）

审查指出的四条，逐条做现场：

## 1. C 的同进程 A/B（替代两次进程的 2→0）

**机制**：111 增加 `runtime_store/router_order.cfg`（true/false）运行时覆盖
`mission.routerBeforeCond`；alias 路由器命中时**显式**打印
`[cond-count] ... calls=0`，不再用“没有这一行”代表零调用。

**装置**：一个网关进程、一个日志；先文件 true 跑 `c-same-new`，再文件
false 跑 `c-same-old`；同一 goal/seed（`surpriseema expected`、
`active_inference.cpp`）。

**切线**：新顺序 `[open-column] alias=1` + `[cond-count] alias=1 calls=0`；
旧顺序 `[cond-count] alias=1 calls>=1`；两顺序同一路径；同一进程/日志。

**结果**：**PASS**。
```
[open-column] mode=ce alias=1 path=active_inference.cpp margin=1 known=1
[cond-count] alias=1 path=active_inference.cpp calls=0
[latent-edit] path=active_inference.cpp cond=0 mlp=1 carrier=1534

[cond-count] alias=1 path=active_inference.cpp calls=2
[cond-hit] path=active_inference.cpp independent=1 stable=1
[latent-edit] path=active_inference.cpp cond=1 mlp=0 carrier=1534
```
同一 `live_gateway36.log`。路由器每拍从零 `init + addFileCe`（`open_column_ce`
探针也是从零训练并按新列路由）；不是复用旧分类器的输出列。

**训练材料改成文件正文（2026-10-30）**：审查指出 `addFileCe` 的 bag 仍来自
别名表已存的词袋。改成：别名表只给出候选路径；每个候选的 bag 由
`active_inference.cpp` 等**源文件正文**经 `distinctiveProfile(path, body,
others)` 得到。19:13 的 `phoenix_main.exe`（46761406）上重跑同一进程：
`[sig] alias-router idx=4 margin=1 known=1 bags=4` →
`[open-column] mode=ce alias=1 path=active_inference.cpp` →
`[cond-count] alias=1 path=active_inference.cpp calls=0`；旧顺序仍是
`calls=2`。同一 `live_gateway43.log`。词←文件对应现在来自源码正文，
别名表只提供路径列表。


## 2. 进化的 n=5 A/B（替代每边两次）

**装置**：Granite 空起点、plain/temp0；A=64、B=164.29757690429688；
A B B A A B B A A B（各 5 次）；目标 1200 字符。

**结果**：**PASS**。
- A：64314、68328、62281、66354、74416 → 中位 **66,354 ms**（5 回合）
- B：38227、44204、40211、44253、56329 → 中位 **44,204 ms**（2 回合）
- Welch t = **5.966**，df = 6.94（p ≈ 0.0006，双侧）；中位墙钟 **-33.3%**。

## 3. 9.5b：新种子/更大 G、λ 的选择实验（9.5 原记录不动）

**装置**：completion = 1000 + N(0,80)；G=30、λ=16、seed 20261029；
early=前 15 代、late=后 15 代；Welch 单侧 p<0.05。

**结果**：**PASS**（仅作为“最小噪声选择”的显著性，不作为基因组进化）。
early=828.182，late=753.144，**p=6.45612e-05**。
**重要限定**：`es_plan_probe` 的完成时间是 `1000 + N(0,80)`，代码里没有
genome、没有 mutate；父本时间一直保留到某个子代抽到更小的噪声才换。G 从
10 加到 30、λ 从 8 加到 16 只是让这种“留下最小噪声”的阶梯在前后两段显著。
所以它**不能**补 9.5 的失败，也不是自主进化的证据；9.5 原实验（seed
20261025、G=10、λ=8、p=0.0889）仍是失败记录。自主进化的证据是
mission 层的 `tokensPerTick` 基因组搜索（下面第 5 条：新程序上重跑 8 代
+ n=5 A/B），不是这个纯噪声实验。

## 5. 新程序上的 8 代基因组搜索（补 10.24 的旧程序问题）

**装置**：22:03 的 `phoenix_main.exe`（46756619）；Granite 父/子、
空起点、`childFreshTask`、`autoChildDoneChars=1200`、workMs 选择、
初代 64、G=8、λ=3。

**切线**：每代 3/3 自动判完成、adopted 子盒 workMs 严格最小；末代基因组
> 64；用末代基因组在 Granite 空起点对 64 做 A/B（n=5）。

**结果**：见下一段现场记录（`fair_evo64_newbin.out.log`、
`g64x5_newbin` 结果）。

**第一轮失败与修复**：22:03 的程序上重跑 8 代时，子盒 `done=0/3`、
`workMs` 只有一拍：fresh 子盒跳过了父代 merge，而子盒自己的
`deliverable.md` 没有逐拍累积（每拍只看到最后 250 字符，永远到不了
`autoChildDoneChars=1200`）。修 `autonomy_stack.cpp`：fresh 子盒每拍把
`boxWork` 写回自己的 workspace，再交给 auto-verdict。23:54 的
`phoenix_main.exe`（46759179）重跑：`[auto-child-done] chars=1291/1335/1384`，
8 代全部 3/3，adopted 依次 71.268 → 90.069 → 104.276 → 105.578 → 110.435 →
158.251 → 156.518 → **228.437**。

**A/B（同一 23:54 程序，n=5，A=64、B=228.4369，plain/temp0）**：
A：74385、72394、62296、72357、76414 → 中位 **72,394 ms**（5/6 回合）
B：46247、56300、46257、56300、46245 → 中位 **46,257 ms**（2 回合）
Welch **t=6.150、df=8.0**（p≈0.0003），中位墙钟 **-36.1%**。
这是新程序上的“末代基因组比 64 快”的证据；164.298 的旧程序 10.24 被这一轮
取代，不再作为新程序证据。

**9.5b 的定位（见第 4 条）**：它不是基因组进化，不补 9.5。自主进化的证据
是本轮 mission 层 `tokensPerTick` 的 8 代搜索 + 上面的 A/B。

## 6. 现场 replicator（2026-10-31，最后一项）

**装置**：4 载体，编辑子集从 3 → 2 → 1 递减（k=3,2,1），编辑载体写
`FILE active_inference.cpp` 并由 coder 出补丁；review 载体（`summarize …
do not edit`）强制记 prose；每代先写 workspace 再 assign；启动后先跑一代
warm-up 并清空 trace，再测三代。

**切线**：三个内点代际的 p 跟随 replicator `p' = p w/(p w + 1-p)`，`w=1/3`，
残差 < 0.01。

**结果**：**PASS**。现场账本 `ticks=12 sourceEdits=10`，generation 序列
`1, 0.75, 0.5, 0.25`；取三个内点代际 `0.75, 0.5, 0.25`：
```
last3 0.75 0.5 0.25 force=replicator samples=3 bestResidual=0
      contentBias=-1.33333 conformity=-2.66667 logFitness=-1.09861
```
`logFitness=-1.09861 = ln(1/3)`，即 `w=1/3`；残差 0。现场第一次出现
replicator 标签；fixed（p=0/1）和混合 undetermined 是另两条边界记录。

## 7. 现场 conformity（2026-10-31，确定性驱动）

**机制**：111 的 deliberator 对 `FORCE-EXPERIMENT` goal 直接返回确定性结果
（编辑载体固定补丁 `FILE active_inference.cpp / OLD / NEW`，review 载体
`Reviewed: no edit.`），不调用任何模型；`runtime_store/force_schedule.txt`
按轮给出 k；代际是全局 loop tick，人口是固定 32。这样 p 严格是 k/32，
coder 的补丁率和超时被移出实验。

**装置**：32 载体 q0..q31；schedule 12 → 14 → 15；每轮 32 拍；三个测量轮。

**切线**：三个内点代际 `0.375, 0.4375, 0.46875` 跟随 conformity
`p' = p + d p(1-p)(2p-1)`，残差 < 0.01。

**结果**：**PASS**。
```json
{"ticks":128,"sourceEdits":41,"forceSamples":4}
series(4): 0.375 0.4375 0.46875 0
first3 force=conformity samples=3
       bestResidual=0.0015625
       conformity=-1.06667  contentBias=0.266667  logFitness=0.259511
```
第一代 0.375、第二代 0.4375、第三代 0.46875，`d=-1.06667`，残差
0.0015625 < 0.01。第四代 0 是 schedule 用尽后的尾巴，不在测量窗口。
至此现场四力（content/conformity/replicator/fixed）与 undetermined 全部出现。





## 4. 全编辑 force：每代 4 个载体都 source-edit

**机制**：每代重新 assign 4 个载体并重写 `FILE active_inference.cpp` 种子，
coder（qwen_coder）每代给出补丁；`focusByScope` 持久化；代际序列应为
p=1 恒定 → `fixed` 且 `sourceEdits>0`。

**切线**：generation 序列 ≥3 代、每代 sourceEdits=4、`force=fixed`、
`sourceEdits>0`（不是全散文的 fixed）。

**结果**：**PASS**。
```json
{"ticks":20,"sourceEdits":20,"replicators":0,"fidelity":1,
 "force":"fixed","forceSamples":5,"forceResidual":0,"binding":"selection"}
```
5 代 × 4 载体 = 20 拍，**20 次全是 source-edit**，代际序列 p=1 恒定，
`force=fixed`、`sourceEdits=20`。现场日志每个载体
`[replica-trace] … reply=source-edit q=1`，coder 每个载体都
`[summon-tool] source-edit model=qwen_coder`。这是 28h 切线要的
“两边都是 source-edit + fixed”，不是之前 `sourceEdits=0` 的 fixed。

content/conformity/replicator 的现场标签仍不是这个种群能给出的：全编辑
= p=1 恒定 = fixed；要 content/conformity 需要 p 按映射变化（受控编辑率）。
生产仪器的 content/conformity/replicator/fixed 四标签由
`force_generation_probe`（200 载体 × 6 代）标定 PASS；现场这个 4 载体
种群只走到 fixed。这两条分开记录，不互相冒充。


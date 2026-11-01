# Research profiles（研究参数组合）

这份文档是研究实验的**单一事实来源**。三个目标的实验不再靠手工改 config
或改代码，而是通过一个参数组合 .json 特化：`config/research/<profile>.json`
在启动时作为 **overlay 深合并** 到 `config/phoenix.json` 之上（`research.profile`
选择哪个；默认不加载，主流程路径不变）。

## 1. 规则（防止文档漂移）

1. 每个实验只用一个 profile；实验的所有参数都写在 profile 里。
2. 实验的原始输出只允许引用它自己写出的日志文件；协议/文档里的数字必须
   来自**原始日志的对应字段**，重放和标定必须单独标注层级。
3. profile 改动必须在这里同步记录；禁止把重放/标定写成 live。

## 2. profile 模式

```json
{
  "research": {
    "name": "<profile-name>",
    "profile": "<profile-name>",
    "description": "...",
    "force": { "enabled": false, "schedulePath": "...", "population": 32, "schedule": [] },
    "deterministicCarriers": false,
    "routerOrderSequence": [false]
  },
  "mission": { "...": "..." },
  "autonomyLoop": { "...": "..." }
}
```

- `research.force.enabled`：确定性 force 驱动（111 的 deliberator 对
  `FORCE-EXPERIMENT` goal 返回固定补丁/本地 prose，p = k/N）。
- `research.deterministicCarriers`：编辑/review 载体的确定性回复
  （`make one small ... edit` → 固定补丁；`summarize ... do not edit` →
  `Reviewed: no edit.`）。
- `research.routerOrderSequence`：同进程内 router-before-cond 的顺序序列
  （替代 `runtime_store/router_order.cfg`）。
- 其余键是 `config/phoenix.json` 的点路径，overlay 合并后 `cfgOr` 直接生效。

## 3. 三个 profile 与三个目标

### 3.1 `fair_evolution.json` — 目标 1（自主进化）

| 键 | 值 | 说明 |
| --- | --- | --- |
| mission.temperature | 0.0 | plain 路径确定性 |
| mission.plainTextOnly | true | 每拍输出长度只由 tokensPerTick 决定 |
| mission.autoChildDoneChars | 1200 | 子盒按字母数自动判完成 |
| mission.childFreshTask | true | 子盒空起点、公平对比 |
| mission.maxReplicas | 4 | 实验不禁止 replicate（8 代搜索无 trace 分母问题） |
| autonomyLoop.intervalSec | 10 | 默认轮转 |

配方：`powershell -File build\tmp\live_fair_evo64.ps1`
（8 代、初代 64、Granite），然后 `live_g64x5.ps1`（n=5 A/B）。
输出：`fair_evo64_current.out.log`（64→126.79，每代 3/3）、
`g64x5_126.out.log`（A 中位 66,406 ms / B 50,289 ms，t=5.566，df=4.30）、
`g64x5_current.out.log`（228.44 的 A/B：回合 5,5,6,5,6 对 2,2,2,2,2，
p=1/252≈0.00397，t=7.397）。

### 3.2 `force_conformity.json` — 目标 2（模因论现场力）

| 键 | 值 | 说明 |
| --- | --- | --- |
| research.force.enabled | true | 确定性 force 驱动 |
| research.force.schedulePath | runtime_store/force_schedule.txt | 每轮一个 k |
| research.force.schedule | [12, 14, 15] | 0.375 / 0.4375 / 0.46875 |
| research.force.population | 32 | 固定人口分母 |
| mission.maxReplicas | 0 | 子盒不进 trace 分母 |
| autonomyLoop.intervalSec | 60 | 每轮独立 |

配方：`AI_RESEARCH_PROFILE=force_conformity` 启动网关 +
`powershell -File build\tmp\live_force_det2.ps1`。
输出：`force_det7.out.log`（手动配置版）与 `force_profile.out.log`（profile 版，
二者一致：series `0.375 0.4375 0.46875 0`；first3 `force=conformity`，
`conformity=-1.06667`，残差 0.0015625；第 4 个 0 是 schedule 用尽后的尾巴，
不在测量窗口）。

### 3.3 `alias_probe.json` — 目标 3（MLP + 模因论快训）

| 键 | 值 | 说明 |
| --- | --- | --- |
| mission.aliasProbe | true | 描述-only 任务先走别名召回，不走 source-edit 工具 |
| research.routerOrderSequence | [true, false] | 同进程内新/旧顺序各一次 |
| mission.maxReplicas | 4 | 默认 |

配方：`AI_RESEARCH_PROFILE=alias_probe` 启动网关 +
`powershell -File build\tmp\live_router_sameproc.ps1`（描述-only goal）。
输出：`live_gateway70.log`（手动配置版）与 `live_gateway73.log`（profile 版，
二者一致：新顺序 `[open-column]` + 显式 `[cond-count] alias=1
path=active_inference.cpp calls=0`；旧顺序 `calls=2` + `[cond-hit]`）。

## 4. 力证据的三个层级（禁止混写）

| 层级 | 内容 | 来源 |
| --- | --- | --- |
| live | fixed（全编辑 20/20 p=1；prose p=0）、undetermined（rp14 原始日志 `1,0.75,0.666667,1`；content 原始日志 `1,0.5,0.6,1,1,1`） | 原始日志的 `force` 字段 |
| 重放 | content `0.5714,0.4286,0.2857`（b=-0.5833）、replicator 对应序列 | `force_trace_eval` 对 trace 的固定人口重放 |
| 标定 | conformity `0.375,0.4375,0.46875`（d=-1.0667）；`force_generation_probe` 四力合成标定 PASS | 确定性 schedule / 合成序列 |

## 5. 尚未闭合的项

- 9.5 的噪声表型（seed 20261025、G=10、λ=8、p=0.0889，时间与基因组无关）
  已撤出进化判据，数字留档，不改判通过。9.5b 仍不是进化证据。进化证据是
  第 9.11、9.12 节、父代更短工作时间的保留，以及 A/B 的回合数（p=1/252）。
- live coder 对注入频率不敏感（`0.2 0.2 0.2`，force=fixed）是预注册里
  “决策规则没有频率依赖项”的证实，记为成功。conformity 的
  0.375/0.4375/0.46875 仍只是标定。
- `open_column_ce_probe` 全 PASS 是目标 3 的无噪声证据，实机对照是
  `live_gateway70.log` 的同进程 0/2。

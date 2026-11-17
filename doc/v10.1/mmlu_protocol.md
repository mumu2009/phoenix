# MMLU 评测协议（M8 工具化）— v10.1

> 状态：`tools/eval_mmlu_protocol.py` 已实现，自检（mock，零外部负载）通过，**已在标注的共享端口上 dry run 成功**。
> 用途：把 `doc/v10.1/methods_backlog.md` 的 M8 从提案变成可执行口径——legacy 40 题哨兵（连续性）、
> 400 题分层主口径（±0.025）、配对 McNemar+bootstrap（成长判定）。
>
> 边界（写作时遵守）：**纯 HTTP 客户端**；不编译、不跑 `compile.bat`、不碰网关路由、不写 `runtime_store/`、
> 不改产品源码；产物只落 `build/tmp/mmlu_protocol/`。

---

## 0. 为什么需要它（问题与勘误）

| 问题 | 证据 |
|---|---|
| 40 题口径的 +2.5pp 不可判 | `implementation_2026-11-07.md` §7："20/40 → 21/40（+2.5pp）在 n=40 的噪声内（±~8pp）" |
| 40 题口径的重复性被误读 | 1453 次 `total=40` 历史：均值 0.4777，**run-to-run sd=0.0343**，而**抽样 SE=0.0790**（p≈0.478, n=40）——见 `methods_backlog.md` §0.3 |
| 没有配对检验 | 历史只有"两次单口径分数"，无同题配对、无 CI、无 McNemar |
| 评价脚本依赖第三方包 | `tools/mmlu_subset_eval.py` 需要 `openai`；本机 `python 3.11.9` **没有 openai**（pandas/numpy 有） |
| 历史脚本的抽样范围被忽略 | pandas 在**全部 14,042 行**上抽样；任何"先过滤再抽"的实现都会整体偏移（本轮 dry run 真实踩到，见 §7.1） |

**本轮两处勘误（对 `methods_backlog.md` 与上一份汇报）**

1. **题数**：早前说"18,502 题"是 `Get-Content` 的**物理行数**；CSV 里题面含内嵌换行（引号内），
   真实记录数是 **14,042**（MMLU test split；与 tinyBenchmarks 论文里的 "14K" 一致）。以记录数为准。
2. **"最小可检出差异 15pp"**：那是 **α=0.05 的临界偏移**（n=40 时 14.7pp），不是 **80% 功效的 MDE**。
   80% 功效下 n=40 的单侧 MDE 是 **19.7pp**（见 §3）。两个数都保留，但必须分开表述。

---

## 1. 数据集（精确路径 + 指纹）

| 项 | 值 |
|---|---|
| 路径 | `E:\079\v6.0Alixander\phoenix\build\tmp\mmlu.csv` |
| 字节 | 6,667,575 |
| SHA-256 | `15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb` |
| 记录数（CSV 记录） | **14,042** |
| 物理行数 | 18,503（含表头；题面内嵌换行导致，**不要用行数当题数**） |
| 学科数 | 57（每科 100–1534 题） |
| Answer 取值 | 仅 A/B/C/D（3776/3582/3462/3222） |
| 异常行 | **10 行有 1 个空选项**（第 1006/1090/1104/1117/1156/1429/1961/1988/2027/2037 行） |

**空选项行的处理（关键）**：`tools/mmlu_subset_eval.py` 用 pandas 读 CSV，空单元格变 `NaN`，
f-string 渲染成字面量 `"nan"`（例如 `A) nan`）。为了让 legacy 题集与提示词**逐字节一致**，本工具
**保留这些行在抽样框内**、渲染为 `"nan"`，并在 JSON 的 `dataset.rows_with_nan_option` 记录数量。
—— 这正是 §7.1 那次失败 dry run 的根因：先把 10 行剔除（14,032），抽样范围变了，40 题整体换了一批。

**数据集发现顺序**：`--csv` → 环境变量 `PHOENIX_MMLU_CSV` → `build/tmp/mmlu.csv` →
`tools/data/mmlu.csv` → `runtime_store/mmlu.csv` → `data/mmlu.csv` → `./mmlu.csv`。
找不到就**报错退出**（不会静默换数据）。每次运行的 JSON 都带 path/sha256/records/eligible/excluded，
换 CSV 会立刻在产物里可见。

---

## 2. 抽样设计

### 2.1 legacy（40 题哨兵，连续性口径）

- `n=40`，`seed=20261108`（与 `tools/mmlu_subset_eval.py` 默认一致）。
- 算法：`numpy.random.RandomState(seed).choice(N, n, replace=False)`，`N=len(records)=14,042`。
  本机实测（pandas 3.0.6）：`DataFrame.sample(n, random_state=seed)` 的**行位置与它完全相同**
  （pandas 整数 seed 走 RandomState 路径），因此历史 `mmlu_history.jsonl` 的分数**可直接对比**。
- 提示词：逐字节复刻 legacy（`f"{Question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nAnswer:"`；
  **不 strip** 题面/选项（pandas 不 strip，本轮验证过 2 个题面带前导空格的行会因此差 1 字节）。
  校验结果：40/40 题、40/40 提示词与 pandas 路径完全一致（§7.2）。
- 用途：只做"大幅退化"哨兵（>15pp 或同题配对方向为负）；**不做成长声明**。

### 2.2 stratified（400 题主口径）

- `n=400`，`seed=20261117`。
- 分配：`quota_s = n × size_s / N`，**Hamilton 最大余数法**；组内用
  `RandomState(sha256(f"{seed}|{subject}") 前 8 字节)` 抽样；输出按学科字典序排列（便于 diff）。
- 覆盖（实测）：
  - `n=400`：**57/57 学科全覆盖**，min=3（多个小学科各 3 题），max=44（professional_law）；
    前五：professional_law 44、moral_scenarios 25、miscellaneous 22、professional_psychology 17、high_school_psychology 15。
  - `n=200`：57/57 覆盖，min=1，max=22。
  - `n=40`：只覆盖 **32/57** 学科（8 个学科 0 题）→ n=40 分层**不能**做学科级声明（可用于 plumbing）。
- 精度：400 题在 p≈0.478 时二项 SE≈**0.025**；200 题 ≈0.035；40 题 ≈0.079。
- 复现：同 seed + 同 CSV ⇒ 同一组题；完整 allocation 写在每次 JSON 的 `sampling.allocation`。

---

## 3. 功效分析表（α=0.05, power=0.80, 精确二项/McNemar）

工具命令：`python tools/eval_mmlu_protocol.py --power-table`（不联网）。

**主表（baseline p0=0.4777，即当前 40 题哨兵历史均值）**

| n | 单臂单侧 MDE | 单臂双侧 MDE | 配对 MDE @10% 不一致率 | @20% | @30% |
|---|---|---|---|---|---|
| 40 | 19.7 pp | 22.1 pp | n/a | 20.0 pp | 25.0 pp |
| 200 | 9.0 pp | 10.0 pp | 6.0 pp | 9.0 pp | 12.0 pp |
| 400 | 6.5 pp | 7.2 pp | 4.5 pp | 7.0 pp | 8.0 pp |

**稳健性（baseline p0=0.5）**：n=40 → 19.9/22.2 pp；n=200 → 9.2/10.2；n=400 → 6.3/7.3；
配对列几乎不变（40: n/a、20.0、25.0；400: 4.5、7.0、8.0）。→ 结论对 p0 的小幅取值不敏感。

**怎么读这张表**

- **单臂** = 与固定参照 p0 比（回归哨兵、绝对分数）；**配对** = 同题前后比（McNemar 精确，成长判定）。
- n=40 的 10% 不一致率为什么是 `n/a`：40 题里只有 4 对不一致；即使 4 对**全部**翻转，
  精确双侧 p = 0.125 > 0.05 → 这个设计**永远无法拒绝**。不是"样本不够"，是结构上不可能。
- **α 临界偏移 vs 80% 功效 MDE**：n=40（p0=0.4777）时，能让 p≤0.05 的最小观测偏移是 14.7pp
  （25/40 vs 19.1/40），但只有约 50% 概率检出；要有 80% 概率检出，需 19.7pp。报告里写哪个都行，
  但必须写清楚是哪个。
- 计算是**精确离散检验**（`math.comb` 直接求和），不是正态近似；因此 n=40 的数字与"2.8×SE"公式
  （≈22pp）略有差别，以精确值为准。

---

## 4. 配对检验协议（成长/改进声明的唯一合法形式）

**触发**：任何"变强/更好/有成长"的声明（MMLU 绝对分、目标指标增量、自改候选对比）。

**设计**

1. 同一题集（同 CSV sha256）、同 seed、同解码参数（temperature=0，max_tokens=8）；
2. A/B 两臂 = 两个 build（侧车 vs 生产）或两个 checkpoint；
3. 顺序：默认 A→B（同一负载窗口内尽快跑完）；可选 `--interleave` 逐题随机 A/B 顺序，
   对并发负载漂移更稳（本机长跑环境下推荐）；
4. 统计：
   - `b` = A 对 B 错，`c` = B 对 A 错；
   - **精确双侧 McNemar**：`p = 2 × P(Bin(b+c, 0.5) ≤ min(b,c))`（b+c=0 时 p=1）；
   - `Δ = (A 正确数 − B 正确数)/n`；
   - **配对 bootstrap 95% CI**：按题重采样 1000 次，取 2.5/97.5 百分位。

**报告模板（强制字段）**

```
n, dataset_sha256, seed, 每臂 endpoint/model/时间戳,
b, c, 不一致率, Δpp, McNemar p, bootstrap CI, 
该 n 与不一致率对应的配对 MDE（查 §3）
```

**判据（预注册）**——三条同时满足才可写"有证据的改进"：

1. bootstrap 95% CI **不含 0**；
2. `|Δ| ≥` 该 n 与观测不一致率下的配对 MDE（§3 表；介于两档之间按下档更严的值）；
3. 独立重复 ≥3 次方向一致（≥3/3），或报告 ICC（运行噪声占比）且 ICC<0.5。

否则写：**"无差异（在 n=…、不一致率 … 下不可判）"**——这也是结论，必须落盘。

**禁止**

- 跨日/跨负载的单口径分数比较当成长；
- 只报 p 不报效应量与 n；
- A/B 用不同题集或不同 seed（`--a-json/--b-json` 会逐题校验 key 对齐，不一致直接报错退出）。

---

## 5. 决策规则：400 题何时取代 40 题哨兵

**预注册 gates（全部满足才切换）**

| Gate | 条件 | 数据来源 |
|---|---|---|
| G1 可复现 | 同 seed 连续两次 400 题运行：逐题一致率 ≥98%，且分数差 ≤0.01 | 本工具两次 JSON 的 `items` 对比 |
| G2 成本 | 单次 400 题墙钟 ≤30 min | 本机 dry run 外推：40 题在并发负载下 55–67 s → 400 题 ≈10–12 min（含长尾，取 30 min 上限） |
| G3 精度 | 连续 14 天、每天 ≥1 次：日分数 ICC<0.5，且 95% CI 半宽 ≤0.03 | 每日 JSON |
| G4 连续性 | 切换前 7 天两口径并行跑；两口径方向相反时人工复核（视为测量异常，不作结论） | 两侧 JSON |

**回退条件（任一触发即回退并记录）**：400 题单次 >30 min；错误率 >2%；
CSV sha256 变化；出现"两口径方向相反"且复核无法解释。

**切换后的分工**

- **40 题（legacy）**：降为**每周 1 次**的"是否大幅退化（>15pp）"哨兵，保留与历史曲线的连续性；
- **400 题（stratified）**：成为成长/回归主口径；所有成长声明走 §4 配对协议；
- **240 题**：若 G2 不满足，退而用 240 题（±0.032），规则不变。

**能力边界（必须诚实写进任何结论）**：即使 400 题，单口径只能支持 **±0.025 的绝对分数**；
要判 3pp 级成长，必须用配对设计（400 题 @10% 不一致率时配对 MDE≈4.5pp）或更大 n；
`Δ=2pp` 在任何现有 n 下都**不可判**。

---

## 6. 运行手册

```bash
# 自检：全部走进程内 mock，零外部负载（改代码后必跑）
python tools/eval_mmlu_protocol.py --self-test

# 功效表（不联网）
python tools/eval_mmlu_protocol.py --power-table

# legacy 40 题哨兵（默认 8082；会打印 WARNING）
python tools/eval_mmlu_protocol.py --mode legacy --n 40 --seed 20261108 \
    --base-url http://127.0.0.1:8082/v1 --tag sentinel

# 400 题主口径（建议 sidecar 端口；不带 --base-url 时默认 8082）
python tools/eval_mmlu_protocol.py --mode stratified --n 400 --seed 20261117 \
    --base-url http://127.0.0.1:5101/v1 --tag primary

# 配对：两个端点（A=侧车 build，B=生产）
python tools/eval_mmlu_protocol.py --mode paired --n 400 --seed 20261117 \
    --base-url-a http://127.0.0.1:5101/v1 --base-url-b http://127.0.0.1:8082/v1 \
    --interleave --tag ab

# 配对：两次已保存的运行（纯离线，不需要 CSV/网络）
python tools/eval_mmlu_protocol.py --mode paired \
    --a-json build/tmp/mmlu_protocol/A.json \
    --b-json build/tmp/mmlu_protocol/B.json
```

**产物**：默认 `build/tmp/mmlu_protocol/`，每次运行一对文件：

- `*.json`：schema/命令/时间/数据集指纹/抽样/allocation/端点/逐题结果/汇总；配对模式另含 arms/analysis；
- `*.csv`：逐题表（key/subject/gold/pred_legacy/pred_strict/correct/raw/latency_ms/error/arm）。

**对长跑的影响（实测）**：40 题 ≈55–70 s 的 slot 占用，与 supervisor 自己的 40 题基线同量级；
不触碰网关、不写 `runtime_store/`、不追加 `mmlu_history.jsonl`（只有 supervisor 写它）。
若要与 supervisor 的基线窗口错开：supervisor 约 7–9 min 一个"train 60 条 + MMLU 40 题"周期，
可在其 `mmlu baseline` 落日志后立即跑；A/B 比较尽量放在同一负载窗口内或使用 `--interleave`。

**推荐调度（与 §5 gates 对齐）**

- 每天 1 次 400 题（sidecar 端口）→ 记录 JSON，用于 G1/G3；
- 每次自改候选合并前 1 次配对（A=候选侧车，B=当前生产；`--interleave`）→ 按 §4 判据；
- 每周 1 次 legacy 40 题 → 与 `mmlu_history.jsonl` 历史曲线对照。

---

## 7. Dry run（本轮已执行，含一次失败复盘）

### 7.1 第一次 dry run：框架错了（已作废）

- 命令：legacy n=40 → 结果 **15/40 = 0.375**，耗时 133 s，0 error。
- 产物：`dryrun-legacy-legacy-n40-seed20261108-20261117T141233Z.{json,csv}`
  → 已重命名为 `INVALID-frame-*` 保留证据。
- 根因：当时实现把 10 个"空选项行"从抽样框剔除（14,032），而 pandas 历史路径是在 **14,042** 行上抽；
  `RandomState.choice` 的 `N` 变了 → 40 题整体换成另一批，与 supervisor 同时刻的 19/40 不可比。
- 教训（已写进代码注释与 `dataset.note`）：**抽样框长度是口径的一部分**；任何过滤都必须与历史路径一致，
  否则"legacy 连续性"不成立。修复后 40/40 题、40/40 提示词与 pandas 逐字节一致（§7.2）。

### 7.2 第二次 dry run：legacy 40 题（修正后）

| 项 | 值 |
|---|---|
| 命令 | `python tools/eval_mmlu_protocol.py --mode legacy --n 40 --seed 20261108 --base-url http://127.0.0.1:8082/v1 --tag dryrun-legacy` |
| 时间 | 2026-11-17 22:14:55（本地） |
| 结果 | **19/40 = 0.4750**，Wilson 95% CI [0.3294, 0.6250]，0 error，54.9 s |
| 延迟 | p50 1,171 ms，p95 3,330 ms（并发任务负载下） |
| strict 分 | 0.425（见 §8 解析器说明） |
| JSON | `build/tmp/mmlu_protocol/dryrun-legacy-legacy-n40-seed20261108-20261117T141455Z.json` |
| CSV | `build/tmp/mmlu_protocol/dryrun-legacy-legacy-n40-seed20261108-20261117T141455Z.csv` |

**一致性证据**：同一窗口内 supervisor 自己的 40 题基线（`supervisor.log` 22:12:31，约 2.5 分钟前）也是 **19/40 = 0.4750**
——题集与提示词复刻正确的最强证据（两者用同一个 CSV、同一个 seed、同一个端点、temperature=0）。

### 7.3 第二次 dry run：stratified n=40（plumbing）

| 项 | 值 |
|---|---|
| 命令 | `--mode stratified --n 40 --seed 20261117 --base-url ...8082... --tag dryrun-strat40` |
| 时间 | 2026-11-17 22:16:07（本地） |
| 结果 | **17/40 = 0.4250**，Wilson CI [0.2851, 0.5781]，0 error，67.2 s |
| JSON | `build/tmp/mmlu_protocol/dryrun-strat40-stratified-n40-seed20261117-20261117T141607Z.json` |
| CSV | `build/tmp/mmlu_protocol/dryrun-strat40-stratified-n40-seed20261117-20261117T141607Z.csv` |

说明：n=40 分层只覆盖 32/57 学科，**只用于证明 plumbing**；400 题主口径未在本轮跑（成本/长跑占用），
按 §5 的 daily 调度执行。

### 7.4 paired 已保存运行路径（离线，零网络）

- 同一 JSON 自比：`--mode paired --a-json <legacy.json> --b-json <legacy.json>` →
  `Δ=0.00pp`、`p=1`、CI `[0,0]`，输出
  `selftest-same-paired-n40-seed20261108-20261117T141651Z.{json,csv}`；
- 题集不一致（legacy vs stratified）→ 对齐校验直接报错退出（exit 2），不出错误结论。

### 7.5 长跑未被影响的证据

- 网关 PID **23920**（2026-11-17 08:15:50 起）在 dry run 前后均存活；
- `supervisor.log` 正常继续（22:12:31 的 `mmlu baseline` 与 22:14/22:16 的两次 dry run 并存）；
- 本工具**不写** `runtime_store/long_run/mmlu_history.jsonl`（该文件只由 supervisor 追加，本轮行数未因工具改变）；
- 未运行 `compile.bat`、未启动/停止任何 `phoenix_main`/`llama-server`。

---

## 8. 已知限制（写结论时必须带上）

1. **legacy 解析器是历史行为，不是好行为**：取输出中**第一个 A–D 字符**。8B 有时输出
   `## Step 1: Identify the pattern` → 取到 "Identify" 里的 `d` 当 D。本轮实测：legacy=19/40，
   strict（首个独立 token）=17/40，差 2 题。为连续性**保留 legacy**；同时记录 `pred_strict`/`strict_score`
   供诊断。**下一步建议**：若 strict 与 legacy 分歧率 >5%，评估是否改解析器并重跑历史基线（需在
   `mmlu_history.jsonl` 旁写明口径变更，不能悄悄改）。
2. **temperature=0 也不是逐位确定**：llama.cpp 的批处理组成/并发槽位会改变浮点归约顺序，
   近 tie 的 token 可能翻转。历史 1453 次同题重复的 sd=0.0343 就是证据；因此 §4 的配对设计 + ICC 是必需的。
   若两次"完全相同"的运行差 >4 题，先查负载/端点，再怀疑代码。
3. **只评 4 选 1 字母准确率**：不评推理质量、不做 reader/ROUGE、不做多选题的选项级诊断。
4. **400 题不是"免费"的**：约 10–12 min（并发下可能 20 min），不能塞进 7 min 的 supervisor 循环；
   G2 不满足时用 240 题。
5. **离线 paired 模式仍要求两臂同题**（key 级校验）；不同 seed/不同 n 的两次运行不能配对，
   必须先重跑。
6. **`--n 40` stratified 覆盖不全 57 学科**（32/57），只能当 plumbing/快速回归，不能当学科结论。

---

## 9. 与其它文档的关系

- 替代 `tools/mmlu_subset_eval.py` 作为评测入口（后者保留不动，作为历史脚本）；
- `doc/v10.1/methods_backlog.md` M8 = 本文件的方法论；M7 的记忆层题库可直接复用 §4 的配对协议与 §3 的表；
- `doc/v10.1/artifact_contract.md` 的产物契约不受影响：本工具只读 CSV、只写 `build/tmp/`；
- 与 `stability-harness` 的稳定性报告并列：MMLU 是"能力口径"，稳定性是"运行口径"，两者都必须带 n/exposure 与 CI。

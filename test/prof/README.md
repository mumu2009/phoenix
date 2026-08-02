# test/prof

本目录负责性能测试、速度与智能度综合 benchmark，以及结果导出。

## 主要文件

- `main.py`：当前主 benchmark 脚本。
- `main.cpp`：历史或兼容的 C++ benchmark 实现。
- `bench_prof.bat`：Windows 批处理入口。
- `offline_matrix.py`：离线矩阵编排器，会按预设依次拉起 `phoenix_main.exe` 并调用 `main.py`。
- `offline_matrix_plan.json`：离线矩阵计划文件，定义要跑的后端和组件预设。
- `component_presets/`：JSON/XML 组件预设样例。
- `export_pdf.py`：将 benchmark JSON 汇总为 PDF。
- `build.bat`：prof 相关构建脚本。
- `prompts*.txt`、`answer*.txt`：历史或兼容的样例数据。
- `reports/`：批处理输出目录。

## 当前能力

1. 对比 system API 与直连 Ollama。
2. 统计速度、智能度和综合分。
3. 支持标准公开 benchmark 自动拉取与本地缓存，默认内建 GSM8K、AI2 ARC Challenge、HellaSwag、WinoGrande。
4. 支持外部数据集导入、问答对加载、多问卷装载与多轮评测。
5. 支持自动发现本地 `tests/GPT4all/gpt4all.jsonl` 这类问答数据集，并按问答对数量而不是文件数计入样本量。
6. 支持断点续测，能够复用已有 JSON 快照并自动跳过已完成路由结果，例如只补跑 system 或跳过已完成的 Ollama 结果。
7. 支持增量落盘，默认每 100 条请求自动刷新 Markdown/JSON 报告，不必等到全部完成。
8. 支持 Ollama 自动管理、模型自动选择与 Phoenix 自动恢复。
9. 不再通过 REST API 覆盖 Ollama 线程数，避免因为线程参数不一致触发模型重载。
10. 支持独立配置 Ollama 预热超时，用于覆盖模型首次装载较慢的场景。
11. 在正式跑分前会先做 system/Ollama 预检，避免把服务异常直接记成无意义的 `N/A` 报告。
12. 默认对问卷、本地 QA 数据集和外部问答对执行随机抽样，不再按文件顺序截断前 N 条。
13. 支持基于结果收敛情况的稳定性早停，在样本结果趋于稳定时提前结束后续批次。
14. 支持把本地 GPT4all 这类 QA 数据集作为共享样本池，同时生成智能测试与速度问卷，并使用同一个上限共同约束两类样本。
15. 支持在现有 system/Ollama 结果上追加一条裸连 llama.cpp 路由，用于验证智能分差究竟来自模型还是测试链路。
16. 支持按组件预设矩阵自动重启 Phoenix，多轮比较功能脑、结构脑、最小组件集等不同运行组合。
17. 支持用 JSON/XML/CLI 三种方式描述 Phoenix 组件选择，并在离线矩阵里混合使用。
18. 对需要 GGUF 的 llama.cpp 预设做本地前置检查，缺少模型时自动跳过对应预设而不是整批失败。
19. 支持在离线矩阵中为每个 preset 额外生成一份 HAI baseline 报告，用来和 shared-local-qa 综合 benchmark 区分口径。
20. 离线矩阵总报告会显式列出每个 preset 的组件配置、综合 benchmark 质量源和 HAI 报告路径，避免把单个子报告误认为总结果。
21. 即使 preset 因缺少 GGUF 等前置条件而被跳过，总报告仍会保留该 preset 的目标组件组合与跳过原因，避免误判为“没配进去”。
22. 离线矩阵总报告现在会同时输出到 `build/offline_matrix/offline_matrix_summary.*` 和 `build/offline_matrix_summary.*`，便于直接在 `build/` 根目录查看。
23. 默认把“有报告但分数/门槛未通过”标记为 `regression`，仍然产出完整总表；只有真正的编排错误才会默认返回非零退出码。需要严格失败时可设置 `STRICT_EXIT=1`。

## 离线矩阵入口

如果目标机器不能联网、只能使用仓库内 Python，优先使用仓库根目录的 `offline_remote_benchmark.bat`。

它会：

1. 调用本地 `Python314\\python.exe`
2. 优先复用仓库根目录或 `build/` 下已有的 `phoenix_main.exe`，只有显式设置 `FORCE_COMPILE=1` 或本地根本没有可执行文件时才编译
3. 读取 `test/prof/offline_matrix_plan.json`
4. 逐个执行矩阵预设，并输出聚合报告

也可以直接运行：

```powershell
Python314\python.exe test/prof/offline_matrix.py --plan-file test/prof/offline_matrix_plan.json --output-dir build/offline_matrix --ollama-model llama3.1:8b
```

如果你希望离线矩阵在任意 preset 出现 `regression` 时也返回非零退出码，可追加：

```powershell
set STRICT_EXIT=1
offline_remote_benchmark.bat
```

计划文件中的单个 preset 常用字段：

- `backend`：`ollama` 或 `llamacpp`
- `componentConfig`：JSON/XML 组件配置文件路径
- `components`：命令行组件覆盖串
- `systemArgs`：直接透传给 `phoenix_main.exe` 的附加参数，例如 `--frontend-enabled=false`
- `benchmarkMode`：综合 benchmark 的质量集模式；当前默认计划使用 `shared-local-qa`
- `runHai`：是否在该 preset 上额外跑 `test/intelligence/main.py` 的 baseline HAI 评测
- `haiCasesFile`：HAI 评测用例文件，默认 `test/intelligence/cases.baseline.json`
- `reason`：若 preset 被跳过，总报告会展示跳过原因，例如缺少 GGUF 模型
- `enableDirectLlamacpp`：是否附带裸连 llama.cpp 路由
- `requireGguf`：是否要求本地存在 GGUF 模型

## 常用参数

- `--benchmark-presets`：标准公开 benchmark 列表，默认启用内建预设。
- `--benchmark-cache-dir`：标准 benchmark 的本地缓存目录。
- `--benchmark-limit-per-preset`：每个标准 benchmark 拉取的样本上限。
- `--dataset-server-url`：标准 benchmark 行数据源地址，默认是 Hugging Face datasets-server。
- `--benchmark-fetch-retries`：标准 benchmark 拉取失败后的重试次数。
- `--benchmark-cache-only`：只读取本地 benchmark 缓存，不访问远端数据源。
- `--refresh-benchmark-cache`：强制重新拉取标准 benchmark。
- `--no-standard-benchmarks`：关闭内建标准 benchmark。
- `--llamacpp-url`：裸连 llama.cpp 的 chat 接口地址，默认 `http://127.0.0.1:8082/v1/chat/completions`。
- `--auto-discover-tests-datasets`：自动发现本地 `tests` 目录下的已知 QA 数据集，默认开启。
- `--no-auto-discover-tests-datasets`：关闭本地 `tests` 数据集自动发现。
- `--shared-local-qa`：将 auto-discover 找到的本地 QA 数据集作为共享样本池，同时生成智能测试和速度问卷。
- `--llamacpp-model`：发送给裸连 llama.cpp 接口的模型名，默认 `llamacpp`。
- `--enable-llamacpp`：在原有 system/Ollama 比较之外，再追加一条裸连 llama.cpp 路由。
- `--tests-dataset-limit`：本地 `tests` QA 数据集的问答对装载上限，默认 `1000`。
- `--questionnaire-file`：单个速度问卷文件。
- `--questionnaire-files`：多个速度问卷文件，会自动合并去重。
- `--questionnaire-limit`：速度问卷总样本上限，默认 `1000`，`0` 表示不限制。
- `--rounds`：完整轮次数。
- `--random-seed`：随机采样种子，默认自动生成；配合 `--resume` 会优先复用已有快照里的种子。
- `--shuffle-cases` / `--no-shuffle-cases`：控制执行顺序是否随机，默认开启。
- `--stability-stop` / `--no-stability-stop`：控制是否启用稳定性早停，默认开启。
- `--stability-check-interval`：每累计多少组 system/Ollama 成对样本后检查一次是否趋于稳定。
- `--stability-min-samples`：触发稳定性早停前要求的最少成对样本数。
- `--stability-window`：最近多少个检查窗口都稳定时，才判定可以提前停止。
- `--stability-quality-delta`：质量均分允许波动阈值。
- `--stability-balanced-delta`：综合分允许波动阈值。
- `--stability-success-rate-delta`：成功率允许波动阈值。
- `--stability-latency-ratio` / `--stability-latency-delta-ms`：平均延迟允许波动阈值，按相对比例和绝对毫秒二者取较宽的一侧。
- `--resume`：从现有 `--json-output` 快照继续执行。
- `--checkpoint-every`：每隔多少条请求自动刷新一次快照。
- `--ollama-num-thread`：兼容旧命令的废弃参数，已忽略，不再写入 Ollama 请求。
- `--ollama-warmup-timeout`：Ollama 模型首次装载的预热超时。
- `--auto-manage-system`：自动拉起/恢复 `phoenix_main.exe`。
- `--auto-manage-ollama`：自动检测并拉起 Ollama。
- `--preferred-models`：模型偏好列表，脚本会优先匹配。

## 结果产物

默认输出 Markdown 与 JSON；运行中会持续刷新快照，可结合 `export_pdf.py` 再生成 PDF。

离线矩阵默认会产生这些聚合文件：

- `build/offline_matrix/offline_matrix_summary.md`
- `build/offline_matrix/offline_matrix_summary.json`
- `build/offline_matrix/offline_matrix_summary.csv`
- `build/offline_matrix_summary.md`
- `build/offline_matrix_summary.json`
- `build/offline_matrix_summary.csv`

## 推荐方式

- 如果你已经手动启动并调优过 Ollama，建议配合 `--no-auto-manage-ollama --ollama-warmup-timeout 180` 使用。
- 如果一次 benchmark 很长，建议固定 `--output` 和 `--json-output`，再配合 `--resume` 做断点续测。
- 如果你已经有旧的 system/Ollama 结果，只想补跑一条裸连 llama.cpp，对相同 `--output` / `--json-output` 加上 `--resume --enable-llamacpp` 即可；脚本会复用旧结果，只补跑 llama.cpp 缺失样本。
- 如果你沿用了旧命令里的 `http://127.0.0.1:8080/api/chat`，脚本现在会先探测本机标准 llama.cpp server；当发现仓库当前默认口 `8082` 可用时，会自动切到 `http://127.0.0.1:8082/v1/chat/completions`，并在启动日志里打印切换原因。
- 如果 `tests/GPT4all/gpt4all.jsonl` 存在，脚本会默认自动纳入质量测试；需要限制规模时用 `--tests-dataset-limit`，需要禁用时用 `--no-auto-discover-tests-datasets`。
- 如果你希望“智能 30 条 + 速度 30 条”都来自 GPT4all 的同一批随机样本，使用 `--shared-local-qa --tests-dataset-limit 30`；此时脚本会跳过默认 baseline、外部问答对和普通问卷文件。
- 如果当前只能回退到本地 `tests/*.txt` 问卷，脚本默认最多只会装载 1000 条；需要更少或更多时，直接改 `--questionnaire-limit`。
- 如果你希望固定一批可复现的随机样本，设置 `--random-seed`；否则脚本每次会自动生成新的随机种子。
- 如果大样本下指标已经明显收敛，可以保留默认的稳定性早停；如果你需要强制跑满全量样本，追加 `--no-stability-stop`。
- 如果标准 benchmark 拉取超时，可以先保留缓存目录不变，再调整 `--dataset-server-url` 或增大 `--benchmark-fetch-retries` 重新执行。
- 如果当前机器无法访问公开数据源，建议直接加 `--benchmark-cache-only`；脚本现在也会在首次连接超时后自动短路剩余预设，避免重复等待。
- 如果 Ollama 首次载入模型较慢，不要再通过 API 改线程数，直接增大 `--ollama-warmup-timeout` 即可。
# v5.2 基准测试说明

## 1. 目标

本文件定义 v5.2 的基准测试方法，用于同时衡量系统的性能、稳定性与回答质量，而不是只比较单次延迟。

## 2. 三类基准

### 2.1 API 回归基准

入口： [test-tools/api_regression.ps1](test-tools/api_regression.ps1)

关注指标：

1. 成功率
2. 单步骤超时
3. 字段断言通过率
4. 生命周期与自治接口可达性

### 2.2 智能效果基准

入口： [test/intelligence/main.py](test/intelligence/main.py)

关注指标：

1. 文本相似度分数
2. 关键词召回
3. 结构化任务命中率
4. 系统回答与直连 Ollama 回答的相对得分

### 2.3 性能基准

入口： [test/prof/main.py](test/prof/main.py)

关注指标：

1. 平均延迟
2. P50/P90/P95 延迟
3. 成功 QPS
4. 质量分平均值
5. 系统 API 与直连 Ollama 的差值

## 3. 推荐执行顺序

1. 编译：`compile.bat`
2. 启动：`phoenix_main.exe --port=5080 --study-port=5081 ...`
3. 接口回归：`test-tools/api_regression.ps1`
4. 智能评估：`test/intelligence/main.py`
5. prof 基准：`test/prof/main.py` 或 `dist/prof_bench.exe`

单次回归时可优先使用轻量样本：

1. [test/prof/prompts.quick.txt](test/prof/prompts.quick.txt)
2. [test/prof/answer.quick.txt](test/prof/answer.quick.txt)

## 4. 样本要求

1. prompts 不能全是同类型简单问答。
2. intelligence cases 至少覆盖理论解释、术语辨析、边界条件三类任务。
3. 对需要比较的系统与 Ollama，必须使用同一批 prompts 和 reference。

## 5. 输出产物

推荐统一输出到 build 目录：

1. `build/intelligence_eval_report.json`
2. `build/intelligence_eval_report.md`
3. `build/prof_report.md` 或等价输出
4. `build/automation_report.json`

## 6. 判读原则

1. 延迟更低但质量明显下降，不算整体更优。
2. 质量更高但超时率升高，也不算可发布。
3. 必须同时看成功率、延迟和质量三个维度。
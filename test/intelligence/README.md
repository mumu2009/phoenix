# Intelligence Eval

## 1. 目标

本目录用于承接 v5.2 的智能效果评估体系，覆盖回答质量、工具调用质量、人格切换一致性和转换前后等价性验证。

## 2. 评估维度

1. 语义正确性：回答是否覆盖参考答案的关键事实。
2. 任务完成度：是否满足指令中的格式、步骤和约束。
3. 工具调用有效性：是否选择了正确工具、参数是否完整、结果是否可用。
4. 一致性：同一输入在多次运行中的输出偏差是否受控。
5. 转换保真度：GGUF 原模型与 v5.2 转换模型在相同任务上的输出误差。

## 3. 用例格式

参考 [test/intelligence/cases.example.json](test/intelligence/cases.example.json)。

快速回归样例见 [test/intelligence/cases.quick.json](test/intelligence/cases.quick.json)。

每条用例建议包含：

1. id
2. taskType
3. input
4. reference
5. scoring
6. tags

## 4. 最小验收

1. 至少覆盖问答、检索、工具调用、人格切换四类任务。
2. 每类任务至少有 10 条样例。
3. 输出评分必须同时记录总分和子项分。

## 5. 执行入口

1. Python 直接运行：`d:/_phoenix/_079/v5.0Odin/.venv/Scripts/python.exe test/intelligence/main.py --system-url http://127.0.0.1:5080/api/chat --system-token local-dev`
2. 批处理入口：`test/intelligence/runtest.bat`
3. 默认输出：`build/intelligence_eval_report.json` 和 `build/intelligence_eval_report.md`
# v5.2 覆盖率说明

## 1. 目标

本文件用于把“几乎 100% 测试覆盖率”从口号改成可追溯口径，明确 v5.2 的覆盖对象、覆盖层级、执行入口和验收方式。

## 2. 覆盖率不是单一数字

v5.2 不把覆盖率限定为单一的语句覆盖率，而采用四层覆盖：

1. 路由覆盖：公开接口是否被实际请求过。
2. 行为覆盖：主链路与失败链路是否都被验证。
3. 数据覆盖：测试输入是否来自真实或可追溯样本，而不是空请求。
4. 回归覆盖：新增能力是否进入自动化执行链。

## 3. 当前执行入口

1. API 回归：[test/apis/main.py](test/apis/main.py)
2. 后端回归封装：[test-tools/api_regression.ps1](test-tools/api_regression.ps1)
3. 智能效果评估：[test/intelligence/main.py](test/intelligence/main.py)
4. 性能基准：[test/prof/main.py](test/prof/main.py)
5. 一键自动化：[test-tools/automation_suite.ps1](test-tools/automation_suite.ps1)

## 4. 覆盖对象

### 4.1 本地已注册接口

以 [test/apis/api_inventory.json](test/apis/api_inventory.json) 为准，要求：

1. 每个业务接口至少有一条成功路径。
2. 关键接口至少有一条失败路径。
3. 音频、图像、上下文、学习四类接口必须使用真实载荷。

### 4.2 README 公开接口

对 README 中声明的模型生命周期、集群自治、外部风格训练步等接口，至少要求：

1. 有契约级文档。
2. 有最小成功回归。
3. 关键字段有断言。

### 4.3 智能行为

智能评测不看 HTTP 200，而看：

1. 输出是否命中核心事实。
2. 输出是否保持主题一致。
3. 与参考答案的近似度是否达到阈值。

## 5. 覆盖率声明口径

“几乎 100% 覆盖率”仅在同时满足以下条件时成立：

1. API inventory 内公开接口全部进入回归范围。
2. 核心路径成功率达到 100%。
3. 失败路径与边界输入已覆盖。
4. intelligence 与 prof 两套非通断测试均已执行。

如果任一维度缺失，则只能称为“高覆盖率”，不能称“几乎 100%”。

## 6. 发布前检查清单

1. `compile.bat` 成功。
2. `phoenix_main.exe` 可正常启动。
3. API 回归通过。
4. intelligence 评估通过。
5. prof 基准已生成报告。
6. 报告文件已落入 build 目录。
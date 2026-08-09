# v3.0 自动化测试策略（最小可行）

## 1. 优先级

1. 语法与构建测试（最高）
2. 功能回归测试
3. 性能基线测试
4. 代码组织与静态检查

## 2. 后端回归入口

- 编译：`compile.bat`
- 接口回归脚本：`test-tools/api_regression.ps1`
- 接口回归建议覆盖：
  - `/api/chat`
  - `/api/transformer/chat`
  - `/api/transformer/verify`
  - `/api/tests/list`
  - `/api/model/lifecycle`
  - `/api/model/compress`
  - `/api/model/explain`
  - `/api/model/deploy`
  - `/api/model/update`

生命周期断言建议：

- `/api/model/lifecycle`：校验 `servingCluster` 与 `updateSeq` 存在。
- `/api/model/compress`：校验 `estimatedSizeRatio` 与 `estimatedSpeedup` 为数值。
- `/api/model/deploy`：传入 `replicas/routingPolicy` 后，返回 `result.cluster`。
- `/api/model/update`：校验 `result.seq` 递增，且 `activeVersion` 可更新。

## 3. 前端回归入口

- `npm test -- --watchAll=false`
- 重点验证：
  - 发送消息主流程不回归

## 4. 失败分级处理

- P0 阻塞：编译失败、主接口不可用。
- P1 阻塞：关键功能返回结构破坏。
- P2 可延后：性能回退但功能正确。

## 5. 持续集成建议

- 每次提交至少跑：`compile.bat` + 前端单测。
- 每日构建补充：接口回归 + 性能快照。
- 接口回归执行示例：

```powershell
./test-tools/api_regression.ps1 -BaseUrl http://127.0.0.1:5080 -Token local-dev
```

## 6. 接口回归脚本增强（2026）

`test-tools/api_regression.ps1` 已增强为默认"跑完全部步骤 + 结尾汇总 PASS/FAIL/SKIP"，并补充了
未授权/错误密码/缺字段/未知路由等负向用例，以及本节第 6 部分列出的数据清洗回归点的显式断言。
详见 `doc/algorithm/performance_optimization_2026.md` 第 4 节。旧的"遇错即停"行为可用 `-FailFast`
参数恢复；`-SkipExtendedChecks` 可跳过负向/清洗类用例做快速冒烟。

## 7. 新增回归点（监控/清洗）

- 监控接口：
  - `GET /api/monitoring/stats` 返回 `ok=true` 且包含 `routes` 与 `cleaning`。
  - `POST /api/monitoring/reset` 后 `cleanedInputs/cleanedSamples` 重置为 0。
- 数据清洗：
  - `PATCH /api/runtime/features` 设置 `dataCleaningEnabled=false/true` 可生效。
  - `PATCH /api/runtime/features` 设置 `dataCleanMaxChars` 后，长输入会被按阈值截断。
  - 训练接口传入含非法控制字符样本，能被清洗且请求不崩溃。

# test/apis

本目录用于 API 回归、接口探测与后端矩阵验证。

## 主要文件

- `main.py`：API 测试主入口。
- `api_inventory.json`：接口清单。
- `backend_matrix.py`：后端矩阵验证脚本。
- `memebarrier_phrase_feedback_smoke.py`：MemeBarrier 短语反馈与 runtime patch 烟测脚本。
- `runtest.bat`：批处理测试入口。
- `run_backend_matrix.bat`：后端矩阵批处理入口。
- `requirements.txt`：本测试目录所需 Python 依赖。
- `logs/`：测试日志输出。

## 覆盖范围

1. `/api/chat`、`/api/transformer/*` 等核心接口。
2. 鉴权与代理相关路径。
3. 多后端切换后的功能一致性。

## 推荐用途

修改网关、认证、模型路由或多后端调度后，优先执行这里的测试。
# Phoenix 进程拓扑与断点重启

本文描述产品运行时的分进程拓扑、监督器用法和断点恢复。产品名是 **Phoenix**。仓库里个别测试脚本仍带历史压测名（例如 Helios），那只是一次任务文本/压测标签，不是产品概念，也不要写进配置键或运行时标识。

## 进程拓扑

板上常见三块独立进程：

| 角色 | 默认进程 | 端口 | 探活 |
|---|---|---|---|
| 推理 | `llama-server` | 8082 | **pid 或端口是否在听**。禁止 `curl .../health` |
| 网关 | `phoenix_main`（无 `--frontend-only`） | 5080 | pid 文件 + `:5080` |
| 前端 | `phoenix_main --frontend-only=1` | 5081 | pid 文件 + `:5081` |

认知、概念矩阵、三份 LMDB 仍在网关进程内。它们和 `prepareChatContext` / concat-rnn-lstm / shortWindow / episodic 共用会话状态；拆成独立进程会改上下文合同，本轮不拆。监督器对网关做独立崩溃恢复和可选 RSS 上限，避免网关 OOM 拖死整机时误杀仍活着的 llama。

```
监督器 (phoenix_supervisor.py)
   ├─ 观察/拉起 gateway :5080
   ├─ 观察/拉起 frontend :5081
   └─ 观察 llama :8082（活着就留下；默认不自动拉起）
```

`main.frontendSubprocess=false` 时，旧逻辑仍会由网关 `spawnFrontendProcess()` 拉起前端。由监督器托管时请设置 `AI_SUPERVISOR_MANAGED=1`（监督器启动子进程时会自动带上），网关不再 spawn/退出时不再杀掉前端。

## 启动 / 停止

在仓库 `phoenix/` 下：

```bash
# 先保证 llama-server 已在 :8082 监听（用 ss/pid 确认，不要 curl /health）
export PHOENIX_GATEWAY_CMD='/path/to/phoenix_main --disable-watchdog=1'
export PHOENIX_FRONTEND_CMD='/path/to/phoenix_main --frontend-only=1'
# 可选：llama 已死且你明确要拉起时
# export PHOENIX_LLAMA_CMD='...'
# 并在 phoenix.json 把 main.services.llama.restartIfDead 设为 true

python tools/phoenix_supervisor.py
```

只巡检一轮（不进循环）：

```bash
python tools/phoenix_supervisor.py --once --dry-run
```

停止：对监督器发 SIGINT/SIGTERM。监督器**不会**杀仍活着的 llama。不要对 llama 做 `/health` 探活，也不要无故 `pkill llama`。

环境变量：

| 变量 | 作用 |
|---|---|
| `AI_SUPERVISOR_MANAGED` | 为 1 时网关不 spawn/不清理前端 |
| `AI_SUPERVISOR_PID_DIR` | 角色 pid 目录，默认 `runtime_store/pids` |
| `PHOENIX_GATEWAY_CMD` | 网关启动命令 |
| `PHOENIX_FRONTEND_CMD` | 前端启动命令 |
| `PHOENIX_LLAMA_CMD` | 仅当 llama 已死且允许自启时使用 |
| `AI_PID_FILE` | 网关单实例 pid 锁（原有） |

配置键（`config/phoenix.json` 的 `main.supervisor` / `main.services` / `main.checkpoint`）：

- `main.supervisor.enabled` 默认 `false`：网关**不会**自己拉起监督器。单进程/本机调试保持原样。托管模式请**手动**跑 `python tools/phoenix_supervisor.py`（脚本一启动即托管；不必改 `enabled`）。若以后要从网关自动拉起，再把 `enabled` 设为 `true`。
- `denyResumeMissions` 默认包含 `2678077`、`6477259`、`9374278`，断点重启不得把它们拉起来。
- `neverAssignMissionOnRestart` 恒为产品约定：监督器从不调用 `/api/mission/assign`。
- `llamaNeverCurlHealth`：运维约束，探活只用 pid/端口。
- **RSS 保险丝（不是泄漏修复）**：`main.services.gateway.maxRssMb`（默认配置 2048）超限时监督器只重启网关，**永不**因 RSS 动 llama（`llama.maxRssMb` 视为 0）。这是托管模式下的保险丝，不能代替查泄漏。

断点文件默认：`runtime_store/supervisor_checkpoint.json`（pid、角色、上次干净时间、脏 mission 名单）。`resumeMissions` 始终写 `false`。

## 运维约束

1. **不要 curl llama `/health`**。`llama-server --parallel 1` 时健康检查会 cancel 在途生成。
2. llama 活着就留下；只重启死掉的网关/前端。
3. 不要续脏 mission：`2678077`、`6477259`、`9374278`。
4. 不要跑会 `pkill llama-server` 的整包脚本（例如完整 `rdk_netboot_build_and_run.sh`）来做监督器验证。
5. 主机单测：
   - 立刻可跑：`python tools/test_phoenix_supervisor_policy.py`
   - 有 Conan/GTest 缓存后：`powershell -File tools/run_host_supervisor_unit_tests.ps1`
   - 或 `compile_gtest.bat` 之后 `gtest_runner.exe --gtest_filter=RuntimeSupervisor*`
   不要在板上跑这些单测。

# Phoenix 使用说明（登录、权限、监控、数据库）

面向最终用户与部署者。Helios 仅为历史压测名称，本文不使用该名。

默认前端地址：`http://<主机>:5081`。网关默认 `5080`。前端会把未单独实现的 `/api/*` 反代到网关。

## 登录与账号

1. 打开前端。板上若直接打 `/api/*` 出现 **401**，先登录拿 Bearer。无账号时 `GET /auth/config` 的 `allowBootstrap=true`，用「创建管理员」（或 `POST /auth/bootstrap`）完成首次初始化（用户名 ≥3 字符、邮箱、密码 ≥6 位）。之后 `POST /auth/login`。
2. 已有账号：输入用户名与密码点「登录」。令牌写入浏览器 `localStorage` 的 `phoenix_auth_token`，之后请求带 `Authorization: Bearer <token>`。
3. 若配置 `auth.requireEmailVerify=true`，注册/初始化后需完成邮箱验证（开发环境验证码可能回显在响应的 `verifyToken` 或 `auth.outboxDir`）。
4. 开放注册时（`auth.allowRegister=true`）可用「去注册」。忘记密码走邮箱重置码。
5. 侧栏「退出登录」调用 `POST /auth/logout` 并清除本地令牌。

### 角色

| 角色 | 能力 |
| --- | --- |
| `user` | 对话、查看监控与数据库位置 |
| `admin` | 以上 + 用户列表、改角色、启停**逻辑**模块、备份主库 |

管理员改角色：`POST /auth/admin/set-role`，JSON `{"username":"...","role":"user"|"admin"}`。

### 可读错误

接口同时返回机器码 `error` 与中文 `message`。常见含义：密码错误、邮箱未验证、未开放注册、仅管理员、模块禁止启停。

相关配置（`phoenix.json` 的 `auth`）：`allowRegister`、`requireEmailVerify`、`userDb`、`outboxDir`、`jwtSecret`、`allowLocalTokenFallback`。

## 控制模块

`GET /api/ops/modules` 列出逻辑开关（对话界面、世界、任务、学习、搜索、围栏、自主循环）。  
`POST /api/ops/modules`（管理员）`{"id":"search","enabled":false}`。

**不会**提供停止 llama / 监督器 / 插件进程的按钮。`inference-llama` 为只读。不要对推理进程调用 HTTP `/health`。

网关上已有的 `GET/PATCH /api/runtime/features` 仍可用于学习/围栏等既有开关，与上述逻辑开关独立。

## 监控

`GET /api/ops/monitor`（需登录）：

- 前端/网关/推理的 **host、port、pid（若配置）、TCP 端口是否连通**
- 本前端进程内存（RSS）
- 前端已处理请求计数

推理探活只用 **pid + TCP 端口**。可在 `phoenix.json` 的 `monitor.llamaPid` 或环境变量 `PHOENIX_LLAMA_PID` 填推理进程号。请求指标也可看网关 `GET /api/monitoring/stats`（需网关鉴权）。

分进程托管：默认 `main.supervisor.enabled=false`（网关不自动拉监督器）。要监控+断点巡检时，在仓库 `phoenix/` 下运行 `python tools/phoenix_supervisor.py`。网关 RSS 超 `main.services.gateway.maxRssMb` 时监督器只重启网关，不杀 llama。登录与监督器无关，401 先走上面的 bootstrap/登录。

`GET /api/health` 只表示前端 HTTP 自己还活着，不是推理健康检查。

## 数据库

引擎保持 **SQLite + LMDB（及可选 Redis 缓存）**，不要擅自换引擎。

| 项 | 默认 |
| --- | --- |
| 主库 | `main.dbPath` → `runtime_store/ai_store.sqlite` |
| LMDB | `main.lmdb.root` → `lmdb` |
| 前端世界库 | `frontend` 段下的 sqlite / legacyDir |
| 用户库 | `auth.userDb` → `./auth/users.json` |
| 备份目录 | `db.backupDir` → `runtime_store/db_backups` |

- `GET /api/ops/database`：路径、是否存在、是否可写、健康摘要  
- `POST /api/ops/database/backup`（管理员）：把主 SQLite 复制到备份目录，响应 `backupPath`

恢复：停止写入后，用备份文件覆盖 `dbPath` 指向的文件再启动。LMDB 目录请整目录复制，不要只拷单个数据文件。

## 常见故障

| 现象 | 处理 |
| --- | --- |
| 登录页一直转圈 | 确认 `:5081` 前端进程在跑；不要用 curl 打 llama `/health` 做检查 |
| 密码对但 403 | 邮箱未验证，到验证页贴验证码 |
| 注册 403 | `allowRegister` 为 false |
| 对话失败、侧栏 disconnected | 网关 `:5080` 未起来，或 Bearer 过期，重新登录 |
| 推理「未连通」 | 看配置的 llama 端口是否在听；填对 `monitor.llamaPid`；禁止用 `/health` |
| 备份 404 | 主库尚未创建（首次写入后才会有文件） |
| 管理员按钮灰色 | 当前账号 `role` 不是 `admin` |

## 前端自测说明

本环境无浏览器自动化。登录/运维 UI 用 Jest（`AuthGate`、`OpsPanel`、`App`）覆盖表单、中文错误、模块开关与导航。接口契约用 `client.test.js`。联调可用：

```
curl -s http://127.0.0.1:5081/auth/config
curl -s -H "Content-Type: application/json" -d "{\"username\":\"admin\",\"password\":\"...\"}" http://127.0.0.1:5081/auth/login
```

不要对 llama 发 `/health`。

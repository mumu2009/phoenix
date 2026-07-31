# 079 Frontend

## 角色定位

前端在 v6.0 中是现有主程序链的可视化入口，不单独定义后端架构。

- 默认面向官方推荐链路：`phoenix_main.exe` 提供的 `/api/*` 与 `/auth/*`。
- 前端允许切换 provider，但 provider 切换不等于替换后端主链设计。
- 前端依赖稳定优先，版本升级以可复现构建和自动化测试通过为前提。

## 开发命令

在 `079project_frontend/` 目录执行：

- `npm install`
- `npm start`
- `npm test`
- `npm run build`

## API 与提供者（Provider）

前端聊天默认走后端主链 `/api/chat`（即 GNN+Transformer 主链）。

可通过 `REACT_APP_API_BASE=` 配置前端网关基址。

统一返回结构由客户端归一化为 `{ ok, result: { reply, latency? } }`，避免 UI 改造扩散。

## 说明

- 系统默认使用 `core` 主链。
- 鉴权 token 由 `localStorage` 中的 `phoenix_auth_token` 管理。
- 推荐只把 `core` 视为正式发布态默认入口；其他 provider 视为并行旁路或兼容入口。
- 监控面板依赖后端 `GET /api/monitoring/stats` 与 `POST /api/monitoring/reset`。
- 运行时清洗开关依赖后端 `PATCH /api/runtime/features`。
- 生命周期能力由后端 API 提供：`/api/model/lifecycle`、`/api/model/compress`、`/api/model/explain`、`/api/model/deploy`、`/api/model/update`。

## 认证边界

- 登录态默认依赖后端返回的 bearer token。
- 前端应优先使用 `/auth/login`、`/auth/me`、`/auth/logout`、`/auth/change-password`、`/auth/profile` 这些稳定接口。
- 若启用邮件验证或找回密码流程，则继续使用 `/auth/verify*`、`/auth/forgot`、`/auth/reset`。
- 前端测试与本地联调建议配合隔离的测试用户库，避免污染仓库内默认 `auth/users.json`。

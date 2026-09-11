# DATABASE_079 使用说明

`DATABASE_079` 是 Phoenix 的持久化层：默认 **SQLite** 作冷存储，**LMDB** 作遗留/大映射目录，可选 **Redis** 作推理缓存。不要为性能引入新引擎。

## 位置

- 实现：`phoenix/DATABASE_079.hpp`、`phoenix/DATABASE_079.cpp`
- 主库路径：`phoenix.json` → `main.dbPath`（默认 `runtime_store/ai_store.sqlite`）
- LMDB：`main.lmdb.root`（默认 `lmdb`）
- 前端世界模型库：配置里 frontend 段的 sqlite + `legacyDir`
- 管理接口：`GET /api/ops/database`、`POST /api/ops/database/backup`

## 库里有什么

命名 store（`createStore(name)`）常见：`kvm`、`meme`/`meme_graph`、`session`。另有推理结果热/冷缓存（`getInferenceCache` / `setInferenceCache`）。

## 备份与健康

产品层只做文件级备份（复制 SQLite 文件到 `db.backupDir`）。健康判断：文件存在或父目录可写（允许首次启动尚未建库）。

LMDB 备份请复制整个 `lmdb` 目录。用户账号不在 SQLite 里，而在 `auth.userDb` 的 JSON。

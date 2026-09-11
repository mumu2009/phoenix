# 模块内资源双 CRUD

插件开发需要精确打到**各模块已声明的内部对象**，而不是任意内存。Phoenix 共用一份资源注册表，提供两种入口：

1. **外部 CRUD**：网关 HTTP，路径精确到模块内部资源。
2. **内部 CRUD**：插件 / util 进程内 API，同一资源模型，带 capability / ACL。

缺注册一律拒绝（`error=unregistered`）。权限用显式检查，不用 try/catch 当防护。

不暴露推理 / 上下文内部向量（embedding、RNN/LSTM 状态、shortWindow 等）。

---

## 1. 资源注册

模块在启动时向 `phoenix::util::ModuleResourceRegistry` 声明可 CRUD 的类型：

```cpp
#include "util/module_resource.hpp"

phoenix::util::ResourceSpec spec;
spec.moduleId = "my_module";
spec.type = "notes";
spec.description = "plugin-visible notes";
spec.acl.allowExternal = true;
spec.acl.allowInternal = true;
spec.acl.allowExternalWrite = false;
spec.acl.allowInternalWrite = true;
spec.handler = [](const phoenix::util::CrudCall &call) {
    /* 只读或明确允许的字段 */
    return phoenix::util::CrudReply{true, 200, "", nlohmann::json{{"id", call.id}}};
};
phoenix::util::ModuleResourceRegistry::instance().registerResource(spec);
```

内置挂载（`installBuiltinModuleResources`，网关 listen 时调用）：

| moduleId | type | 含义 | 写 |
|---|---|---|---|
| `session` | `meta` | 会话元数据：id / label / 时间戳，**无向量** | 仅内部可写 |
| `plugin` | `catalog` | PluginRegistry 工厂 + 已注册实例清单 | 只读 |
| `addon` | `mounted` | AddonManager 挂载清单 | 只读 |
| `config` | `public` | 用户可见配置（主机、端口、autoloadDir 等） | 只读 |
| `module_mount` | `factories` | 哪些模块工厂已安装（bool，不调用工厂） | 只读 |

`registerResource` 在 `moduleId` / `type` 为空或 handler 为空时返回 false。

---

## 2. 外部 CRUD（HTTP）

路径惯例：

```
/api/modules/{moduleId}/resources/{type}
/api/modules/{moduleId}/resources/{type}/{id}
```

| 方法 | 操作 |
|---|---|
| `GET` 无 id | list |
| `GET` 有 id | get |
| `POST` | create（id 可在路径或 JSON `id`） |
| `PUT` / `PATCH` | update |
| `DELETE` | delete |

鉴权：与网关其它 `/api/*` 相同，需要 `Authorization: Bearer …`。  
`local-*` 令牌在 `auth.acceptLocalToken` / `crud.http.acceptLocalToken` 为真时可用。  
也可配置 `crud.http.token` 作为专用静态令牌。

示例：

```http
GET /api/modules/plugin/resources/catalog
Authorization: Bearer local-dev
```

```http
GET /api/modules/config/resources/public/gatewayHost
Authorization: Bearer local-dev
```

处理函数是 `phoenix::util::handleExternalCrud(method, path, authorization, body)`，单元测试不启 drogon 即可打到同一逻辑。

---

## 3. 内部 CRUD（插件 / util）

```cpp
auto reply = phoenix::util::handleInternalCrud(
    "my-plugin",
    plugin->getCapabilities(),
    phoenix::util::CrudOp::Get,
    "session", "meta", "sess-1",
    nlohmann::json::object());
```

`PluginManager` 包装（能力取自已注册插件实例）：

- `crudList(actor, moduleId, type)`
- `crudGet(actor, moduleId, type, id)`
- `crudWrite(actor, moduleId, type, id, value, create)`
- `crudDelete(actor, moduleId, type, id)`

未注册的 actor 能力为空，读/写都会 `capability_denied`。

---

## 4. 权限模型（fail-closed）

对每次调用，按顺序：

1. 注册表里有没有 `(moduleId, type)`？没有 → `404 unregistered`
2. 通道开关：`allowExternal` / `allowInternal`
3. 写操作：`allowExternalWrite` / `allowInternalWrite`，否则 `405 method_not_allowed`
4. 外部：Bearer 无效 → `401 unauthorized`
5. 内部：
   - `allowedActors` 非空则 actor 必须在名单内
   - actor 必须同时具备 `requiredCaps` **以及** 该操作对应能力  
     （List/Get=`READ_DATA`，Create/Update=`WRITE_DATA`，Delete=`DELETE_DATA`；`FULL_ACCESS` 覆盖）

配置见 `config/phoenix.json` 的 `plugin` / `util` / `crud` 段。

---

## 5. 示例：插件写会话元数据（不碰 embedding）

```cpp
nlohmann::json body{{"id", "s1"}, {"label", "demo"}};
auto r = phoenix::util::handleInternalCrud(
    "annotator",
    {phoenix::plugin::PluginCapability::WRITE_DATA},
    phoenix::util::CrudOp::Create,
    "session", "meta", "s1", body);
```

外部只读同一条：

```http
GET /api/modules/session/resources/meta/s1
Authorization: Bearer local-dev
```

---

## 6. 主机单元测试

在 `phoenix/` 下编译并过滤：

```bat
compile_gtest.bat
gtest_runner.exe --gtest_filter=ModuleResourceCrud*
```

覆盖：注册、内部 CRUD、外部处理函数、越权拒绝、未注册拒绝。

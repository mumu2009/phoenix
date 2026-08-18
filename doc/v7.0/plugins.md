# 插件生态：CLI-Anything 桥、MCP 兼容与内置搜索/数学插件

本文回答"如何让 Phoenix 拥抱主流插件市场"与"为什么内置的搜索/数学插件必须大幅增强"。实现全部**可选、默认关闭/默认保守**，零外部 SDK 依赖。

---

## 1. 现状与三条扩展路径

| 路径 | 机制 | 适用 |
|---|---|---|
| 内置插件 | `addon.hpp` 的 `Addon` + `AddonManager`（内建 + 动态库加载） | 数学、搜索、computer-shell、**cli-json（新）** |
| 任意软件 → 插件 | `addons/cli_json_addon.*` + `phoenix/subprocess.*`：配置白名单里的命令模板直接 exec（不走 shell），stdout 若是 JSON 则结构化返回 | 结合 `outsides/CLI-Anything` 的"软件→CLI→插件"方法论 |
| 主流插件市场 | `phoenix/mcp_client.*`：MCP（Model Context Protocol）stdio 客户端，启动外部 MCP 服务器，工具进入 `AgiActionRegistry`（category="mcp"）供规划器选择 | 兼容市面绝大多数 MCP 服务器 |

---

## 2. CLI-Anything 桥：任何软件都能变成插件

`outsides/CLI-Anything`（HKUDS，MIT）的方法论：GUI 软件的后端引擎通常自带 CLI（`melt`、`convert`、`soffice`…）；为它写一个 JSON 输出的 CLI harness（SOP 见其 `HARNESS.md`），Agent 即可通过结构化 JSON 驱动该软件。

Phoenix 不需要复制它的 Python 栈：`cli_json_addon` 就是同一思想的**进程内等价物**——

1. 网关从配置 `cliTools.*` 读取白名单（名字 → 命令模板 + 固定参数 + 超时 + json 标记）；
2. 请求到达时，把文本按空白切分成 argv（**不经过 shell**，直接 `CreateProcess`/`fork+exec`，`phoenix::subprocess`）；
3. 输出尝试 `json::parse`，成功则结构化返回，失败则按文本截断返回；
4. 未注册的名字一律拒绝（fail-closed）——白名单是唯一的信任边界，不靠字符串引号。

示例配置：

```json
"cliTools": [
  { "name": "git-status", "command": "git", "args": ["status", "--porcelain"], "timeoutMs": 5000, "json": false },
  { "name": "soffice-convert", "command": "soffice", "args": ["--headless", "--convert-to", "pdf"], "timeoutMs": 30000, "json": false }
]
```

网关启动时调用 `addon::builtins::setCliJsonRegistry(...)` 安装白名单。

---

## 3. MCP 兼容（主流插件市场）

**范围（诚实声明）**：stdio 传输（MCP 规范最早且最可移植的传输；SSE/streamable-HTTP 不在本次范围）、客户端角色、JSON-RPC 2.0 严格按 id 匹配。

实现（`phoenix/mcp_client.{hpp,cpp}`）：

- 子进程启动（Windows `CreateProcess` 管道 / POSIX `fork+exec`）；
- `initialize` 握手 → `notifications/initialized`；
- `tools/list` / `tools/call` / `resources/read` / `prompts/get` / `ping` / `shutdown`；
- 一个读线程按行解析响应入 `id → response` 表，条件变量唤醒阻塞请求；通知消息安全忽略；
- 管理器 `McpManager` 多服务器聚合；`autonomy_stack` 的 `configureMcp` 把每个工具注册成规划器动作 `mcp.<server>.<tool>`（category="mcp"），`executeAgiAction` 直接派发。

配置：

```json
"mcp": {
  "enabled": true,
  "servers": [
    { "name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"], "timeoutMs": 10000 }
  ]
}
```

安全边界：MCP 服务器是**外部进程**，其能力等于该进程的权限；只在明确配置 `mcp.enabled=true` 时启动，默认关闭。

---

## 4. 数学插件：模型必须能信任算术

动机（你的原话）：模型如果足够笨，再进化也学不会算数——所以算术必须由**精确引擎**保证，而不是浮点近似。IEEE-754 会静默舍入（0.1+0.2≠0.3），价值学习器对着一个数值错误的计算器永远无法收敛。

增强后的 `addons/math_addon.*`：

- **精确模式**：`math_exact.hpp` 的大整数（符号 + 1e9 进制肢，加/减/乘/Knuth 长除/幂/gcd/阶乘）与有理数（p/q 约分）。整数与普通小数（如 `0.1`=1/10）的 + − × ÷ % ^ ! gcd lcm floor ceil round trunc abs min max 全程精确：`0.1+0.2 == 0.3`、`1/3*3 == 1`、`100!` 精确 158 位。非终止小数显示为 `p/q` 分式；终止小数显示为精确十进制。
- **浮点模式**：三角函数/对数/伽马/erf 等超越函数与 `pi`/`e`/`tau`/`phi` 落入 IEEE-754，结果最多 15 位有效数字，域错误（如 `sqrt(-1)`）报错而不是返回 NaN。
- **语句与变量**：`a=2; b=3; a^b`——每次请求独立作用域，赋值不外泄。
- **错误带位置**：解析/求值错误返回字符偏移，Agent 能据此修正表达式。
- 安全：解析器白名单，无 eval、无属性访问（Python 侧同理，AST 白名单）。

复杂度：大整数乘法 O(n·m)、除法 O(n²)（Knuth 归一化）；Agent 尺度（数百位）足够，非密码学库。

---

## 5. 搜索插件：进化的"材料来源"

增强后的 `addons/search_addon.*` + 新 `phoenix/web_search_engine.*`：

- **内置 DuckDuckGo Lite 后端**（`ddg_lite`）：纯 HTTP + 自研原始 socket HTTP/1.1 客户端（Windows/POSIX 同一套 shim，见 `web_search_engine.cpp`），**无需 API key、无外部 SDK**，开箱即用；
- **可配置 JSON 端点后端**（`endpoint`）：接私有 SearXNG/搜索 API；网关原有 `HAVE_CURL` 的 `OnlineResearcher` 路径（HTTPS、爬取、本地索引）保留为第一优先；
- **结构化多结果**：返回 top-N {title,url,snippet}（去重、排序、上限），回复给模型的是"编号标题+URL+摘要"，而不是一条扁平字符串——进化需要可引用的材料；
- **已知限制（诚实）**：`ddg_lite` 依赖网络可达性（部分区域被墙，本机实测百度可达而 DDG 不可达）；此类网络请配置 `search.endpoint` 指向可达的 HTTP JSON 搜索端点。HTTPS 端点走网关 HAVE_CURL 路径。

配置：

```json
"search": { "enabled": true, "backends": ["ddg_lite"], "endpoint": "",
            "timeoutMs": 5000, "maxResults": 8, "userAgent": "PhoenixWebSearch/1.0" }
```

---

## 6. 构建与测试

- 新源文件（已入 `compile.bat` / `compile_gtest.bat` / `tools/build_rdk_x5.sh` 及两个 compat 脚本）：`web_search_engine.cpp`、`mcp_client.cpp`、`subprocess.cpp`、`addons/cli_json_addon.cpp`、`addons/math_exact.hpp`（头）。
- gtest：`test_math_addon_exact.cpp`（精确算术 6 组）、`test_web_search_engine.cpp`（离线解析）、`test_mcp_client.cpp`（帧协议 + 进程内 stdio 会话）、`test_subprocess.cpp`（echo/失败路径）、`test_cli_json_addon.cpp`（白名单门）。验证协议见 `testing_methodology.md` §10。
- 数学/搜索/进程/MCP 各模块均在本机 g++ 13.2 独立编译并通过全部断言后才入树。

---

## 7. 参考文献

- CLI-Anything（HKUDS）：GUI-to-CLI agent harness 方法论（`outsides/CLI-Anything/HARNESS.md`，MIT）。
- Model Context Protocol specification：JSON-RPC 2.0 over stdio，tools/resources/prompts（Anthropic，2024-11-05 起）。
- Knuth, D. E., *The Art of Computer Programming*, Vol. 2, Algorithm D（长除法归一化）。
- IEEE 754-2019：浮点舍入与本文"精确优先"设计的对照。

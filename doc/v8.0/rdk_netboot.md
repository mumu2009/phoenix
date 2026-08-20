# v8.0 RDK X5 网络引导部署（netboot, cpu×3）

面向 192.168.0.107（RDK X5, aarch64）：宿主机（192.168.0.104）只做**一件事**：HTTP 提供文件（:8090）。RDK 自己在设备上
编译并运行**全部三层**（cpu×3）：llama-server（8b+tinyllama，mmap 加载 USB 上的
GGUF，CPU 推理）、phoenix_main 网关、前端子进程。模型只传输一次到 RDK USB，之后
推理完全离线于宿主机。

## 1. 组件分工

| 端口 | 位置 | 内容 |
|---|---|---|
| :8090 | 宿主机 | tools/rdk_netboot_serve.py，静态服务整个树（构建输入 + 预构建 UI + 模型文件 + llama 源码包）——宿主机**唯一**职责 |
| :8082/:8086 | **RDK** | llama-server（8b 主模型，mmap 自 USB）+ tinyllama-server（摘要），全部 CPU 推理 |
| :5080 | RDK | phoenix_main 网关（AI_GATEWAY_HOST=0.0.0.0，LLM 在本机 127.0.0.1:8082） |
| :5081 | RDK | phoenix_main 的前端子进程（--frontend-only=1），静态 UI + /api/* 反向代理 |

## 2. 清单与增量同步

- tools/rdk_netboot_manifest.py：只列构建输入（源码 + config + outsides/bullet3 + tools 白名单），
  流式 md5；INCLUDE_ALL_PREFIXES 支持多级目录，目前包含 079project_frontend/build/
  （预构建 React UI，RDK 无 node/npm 工具链）。
- tools/rdk_netboot_fetch.py：按 relpath|size|md5 差分下载，单文件流式写盘（exFAT 友好），校验失败不落盘。
- 流程：宿主机 `python tools/rdk_netboot_manifest.py > build/rdk_netboot_manifest.txt`，保持 serve 进程存活；
  RDK 上 `bash /tmp/rdk_netboot_v8.sh`（杀旧进程 → 引导工具 → 清单差分 → nohup 构建+运行，
  日志 /tmp/rdk_v8_build.log、/tmp/phoenix_v8.log）。

## 3. 前端子进程（5081 的真相）

网关 `/` 路由返回 404 正文 `UI moved to auth_frontend_server.cjs (default :5081)`。
该字符串是网关 404 的提示文案，不是磁盘上存在的文件——真正的 UI 由网关自身派生：

- 父进程在 `main.frontendEnabled=true`（env AI_FRONTEND_ENABLED）时 fork/exec 自身
  `--frontend-only=1`（116_section_tail.inc::spawnFrontendProcess，spawnDepth 上限防递归）。
- 子进程走 frontend-only 分支：setupFrontendServer() 以 FRONTEND_HOST:FRONTEND_PORT
  （默认 127.0.0.1:5081）addListener，静态服务 WEB_ROOT（默认 ./079project_frontend/build），
  并把 /api/*、/auth/* 反代到 chat.aiApiBase（默认 http://127.0.0.1:5080，即同机网关）。
- React 前端（079project_frontend）API_BASE 默认空串 = 同源相对路径，UI 页面经 5081 反代即可
  访问网关，无需构建期注入地址。

RDK 运行环境（tools/rdk_netboot_build_and_run.sh）导出：

```sh
export AI_FRONTEND_ENABLED=true
export FRONTEND_HOST=0.0.0.0
export FRONTEND_PORT=5081
export WEB_ROOT=./079project_frontend/build
```

子进程继承父进程的 AI_WORLD_*/AI_VISION_* 等降载开关，aarch64 上以确定性 CPU 回退运行
（无 BPU 模型时不启用 BPU），内存占用受控（reservedMemMb=96MB arena）。

v8.0 任务模式环境（同脚本）：`AI_ENABLE_COGNITION_AUTONOMY=true`（注意不是
AI_COGNITION_AUTONOMY_ENABLED——那个名字从未被网关读取，已从脚本移除）、
`AI_AGI_ENABLED=true`、`AI_MISSION_ENABLED=true`；不设 `AI_MISSION_GOAL`，目标由
Mission 控制台在生命周期开始时经 `POST /api/mission/assign` 设立
（111_section 已为 mission.enabled/goal、agi.enabled 增加 env 解析）。

注意：AI_API_BASE（反代上游 = chat.aiApiBase）由网关绑定地址推导；
applyFrontendEnvFromArgs / applyFrontendEnvFromConfig 会把 0.0.0.0、::、*、空串
归一化为 127.0.0.1——通配地址不是可连接目标，否则反代全部返回
`{"ok":false,"error":"disconnected"}`（v8.0 已修，见 000/002_section_*.inc）。

## 3.5 LLM 与模型（cpu×3 关键步骤）

- 源码：`tools/pack_llamacpp_src.py` 在宿主机打包 `outsides/llamacpp`（排除 build/.git）→
  `build/llamacpp_src.tar.gz`（~20MB，由 :8090 服务）。
- 构建：`tools/rdk_llama_setup.sh` 下载源码包 → cmake/ninja（examples ON，server target）→
  **复制到 /tmp/llama-server**（exFAT 挂载忽略 exec 位，必须从 ext4 的 /tmp 运行）。
- 模型：`tools/rdk_fetch_models.sh` 流式下载两个 GGUF（4.9GB 主模型 sha256 校验 + tinyllama），
  exFAT 安全（单文件、逐文件）。
- 运行：rdk_netboot_build_and_run.sh 先起 /tmp/llama-server（8082/8086，--host 127.0.0.1），
  等 health 后启动网关（AI_LLAMACPP_BASE_URL=http://127.0.0.1:8082）。
- 资源实测：8b 推理时 8 核 100%（正常，htop 中 8 个同名条目是同一进程的 8 个线程），
  内存 ~2.5/6.9G，swap 轻微。

## 4. 验证

```sh
# RDK 本机
curl -s http://127.0.0.1:5080/api/health                          # {"ok":true,...}
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5081/    # 200 (index.html)
curl -s http://127.0.0.1:5081/api/health                          # 200，反代到 5080

# 局域网 / 物理网口
curl -s http://192.168.0.107:5080/api/system/status | head -c 200
curl -s -o /dev/null -w "%{http_code}" http://192.168.0.107:5081/
```

宿主机 5080/5081/8082/8086 一律不监听：UI、网关、LLM 全在 RDK；宿主机只保留 :8090。

## 5. 灾难恢复（RDK 树被误删等）

netboot 设计保证 RDK 无唯一状态，误删 `phoenix/` 树后照此重建：

```sh
# RDK 上（/tmp 的引导脚本与二进制通常幸免）
bash /tmp/rdk_netboot_v8.sh            # 引导工具 + 清单全量拉取 + 构建 + 运行
# 若 /tmp/phoenix_main 尚存且版本正确，可跳过 45 分钟重编译：
ROOT=/media/KINGSTON/v6.0Alixander/phoenix
mkdir -p "$ROOT/tools"
for f in rdk_netboot_fetch.py rdk_netboot_build_and_run.sh rdk_llama_setup.sh \
         rdk_fetch_models.sh build_rdk_x5_compat.sh build_rdk_x5_env.sh rdk_x5_win_compat.h; do
  curl -fsSL -m 60 "http://192.168.0.104:8090/tools/$f" -o "$ROOT/tools/$f"
done
chmod +x "$ROOT"/tools/*.sh
python3 "$ROOT/tools/rdk_netboot_fetch.py" \
  "http://192.168.0.104:8090/build/rdk_netboot_manifest.txt" "$ROOT" \
  "http://192.168.0.104:8090"
SKIP_BUILD=1 bash "$ROOT/tools/rdk_netboot_build_and_run.sh"  # 复用 /tmp/phoenix_main
```

（模型 5.6GB 与 llama 构建产物被删时，上面的 run 脚本会自动重下/重建，全程幂等。）

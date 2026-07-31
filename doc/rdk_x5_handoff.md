# Phoenix 在 RDK X5 与主机上的部署、优化与演进总方案

## 1. 目标与生命周期分工

系统采用“主机负责大模型语言能力，X5 负责具身感知、推理执行与状态闭环”的分工。这里的“主机只做 llama-server”是**Phoenix 正式服务生命周期**的约束：主机日常不运行 Phoenix、不接摄像头、不运行 hbDNN/BPU、不保存世界模型状态，也不处理边缘外设。

| 组件 | 正式运行位置 | 职责 |
|---|---|---|
| `llama-server` + GGUF | 主机 | Llama 3.1 8B 权重加载、KV cache、文本推理、自然语言输出 |
| `phoenix_main` | RDK X5 | HTTP 服务、会话、工具编排、世界模型、GNN/GA、Redis/SQLite、外设控制 |
| USB 摄像头 | RDK X5 | 本地 V4L2 采集，固定规格输入 |
| JPEA 图像编码器 | RDK X5 BPU | 真实视觉 embedding 推理 |
| hbDNN / `libdnn.so` | RDK X5 | 加载和执行 Horizon `.bin` |
| 模型转换工具链 | 独立 x86_64 转换环境 | ONNX 检查、PTQ/QAT、量化、`hb_mapper` 编译 `.bin` |

主机与 X5 的在线通信仅为 X5 -> 主机的 HTTP LLM 请求。主机监听 LAN 地址的 `8082`；防火墙只放行 X5 IP，绝不暴露到公网。

## 2. 当前已经完成的改动

### 2.1 X5 摄像头直采

- 已检测到 USB UVC 摄像头。
- `/dev/video0` 是视频节点；`/dev/video1` 是 UVC 元数据节点，不能用于图像采集。
- 已实际验证 OpenCV 读取成功：`1920x1080`、三通道 BGR、`uint8`。
- 固定采集规格：`MJPG 1920x1080@30 FPS`。
- 新入口：`POST /camera/analyze`，请求体为 `{}` 或可选 `sessionId`。
- 旧的 `videoBase64`、`videoPath` 传输路径已被拒绝，防止主机承担视频链路、重复编码或引入格式漂移。
- 服务打开 V4L2 后主动请求固定 MJPG、宽高与帧率；最多丢弃 11 个预热帧，再使用第一张稳定帧。
- 服务锁定宽、高和 OpenCV 像素类型；偏离预期返回 HTTP `409`，而不是隐式转换或带着错误规格进入模型。

### 2.2 真实 JPEA 与 fail-closed 策略

图像 JPEA 只接受 Horizon hbDNN 推理：

- `JPEA_IMAGE_HORIZON_MODEL` 必须指向已编译的 `.bin`。
- X5 必须存在 BPU 设备节点和 `libdnn.so`。
- 摄像头帧直接作为连续 BGR 字节进入 JPEA 预处理，不再重新 JPEG 编码。
- 模型输出必须是 `JPEA_IMAGE_CONCEPT_DIM` 个浮点 embedding；默认维度为 `128`。
- 模型不存在、BPU 不可用、输入不符合编译规格、模型输出维度不符或推理失败时，接口明确返回失败，**绝不生成哈希、统计或伪造 embedding**。
- 语音 JPEA 在没有独立、真实的 X5 `.bin` 前保持 `unavailable`，不会回退到伪特征。

这种 fail-closed 策略是核心工程约束：世界模型中标记为 JPEA 的向量，必须确实来自 JPEA BPU 模型。

### 2.3 X5 构建与启动资产

- `tools/build_rdk_x5.sh`：检查 hbDNN 头文件与动态库，链接 `rdk_x5_bpu.cpp`、Phoenix 业务源及系统依赖，生成 `phoenix_main`。
- `tools/run_rdk_x5.sh`：检查可执行文件、JPEA `.bin` 与 BPU 设备节点；设置主机 LLM URL、JPEA 和摄像头环境变量后启动。
- `runtime_store/rdk_x5_launcher.json`：保存 X5 专用参数模板。
- `rdk_x5_bpu.cpp`：接入 Horizon `hbDNN` 执行路径。

## 3. 当前硬件与软件基线

### 3.1 X5 已验证能力

| 项目 | 状态 |
|---|---|
| 架构 | `aarch64` |
| 系统 | Ubuntu 22.04 |
| BPU 运行时 | 已有 `hobot-dnn`、`/usr/lib/libdnn.so`、`/usr/include/dnn/hb_dnn.h` |
| 视频工具 | 已有 OpenCV、FFmpeg、`v4l2-ctl` |
| 摄像头支持 | 已有 `tros-humble-hobot-usb-cam` |
| USB 摄像头 | 已识别为 `2K USB Camera` |
| 可用格式 | YUYV、MJPG、NV12；MJPG 支持 `2560x1440@30`、`1920x1080@30` 等 |

### 3.2 当前缺失的唯一关键模型资产

工程目录中存在 I-JEPA `.safetensors`，但不存在可直接运行的 Horizon JPEA `.bin`。这是启动真实 JPEA 的唯一硬阻塞。

> 不得将 `.safetensors`、ONNX、普通 PyTorch checkpoint 或任意示例检测模型伪装为 JPEA 模型。

## 4. 关键优化策略与工程经验

### 4.1 数据路径最短化

推荐路径：

```text
USB 摄像头 -> V4L2 MJPG -> OpenCV 解码 BGR -> JPEA 预处理 -> hbDNN/BPU -> embedding -> 世界模型/GNN -> LLM 上下文
```

避免路径：

```text
摄像头 -> 主机 -> 压缩视频 -> 网络 -> X5 -> 容器解码 -> JPEG 重编码 -> 模型
```

收益：降低网络抖动、编解码损耗、主机负担和时延方差；同时让感知与动作闭环保持在 X5。

### 4.2 固定规格而非“自动适配”

模型编译阶段已经决定输入颜色、布局、尺寸、量化和输出张量语义。运行时“自动适配”常常只是把错误静默传播到世界模型。必须固定并验证：

- 摄像头节点：`/dev/video0` 
- 摄像头编码：`MJPG` 
- 源规格：`1920x1080@30` 
- 解码帧：连续 `CV_8UC3` BGR
- JPEA 输入：按模型编译配置确定，例如 `224x224x3` BGR 或 RGB
- 输出：`128` 维 F32 embedding，或与 `JPEA_IMAGE_CONCEPT_DIM` 一致

### 4.3 BPU 使用原则

- BPU 负责批量矩阵计算密集的视觉编码器。
- GNN、GA、规则、会话、数据库和外设控制留在 X5 CPU。
- Llama 3.1 8B 留在主机 CPU/GPU，X5 只发送紧凑上下文和接收文本结果。
- 不将大视频流、原始视频文件或整段摄像头录像塞给 LLM；只传递结构化世界状态、事件和按需截取的视觉摘要。

### 4.4 并发与稳定性

- X5 边缘计算默认最多 `2` 个在途任务，外设默认最多 `4` 个在途任务；先稳定再提高。
- 主机 `llama-server` 先使用 `--parallel 1 --ctx-size 4096`，观察内存、首 token 延迟与总时延后再调大。
- 摄像头模型接口每次采一帧；后续如需连续感知，应引入单独的帧采集线程、有限长度环形缓冲区和最新帧优先策略，不能让多个 HTTP 请求同时竞争 `/dev/video0`。
- 对每个 BPU 任务记录：模型版本、输入规格、输入时间、输出维度、推理耗时、失败原因。这样才能定位规格漂移、BPU 异常和模型升级问题。

### 4.5 存储与世界模型

- 原始视频默认不持久化；仅保存经策略筛选的事件摘要、embedding、模型版本、时间戳与必要的引用。
- Redis 用于短期会话和队列；SQLite/LMDB 用于可恢复的世界状态与索引。
- embedding 写入前必须带 `encoder=transformer-text-encoder`、`encoder=world-model` 或 `encoder=jpea-horizon-hbdnn` 等来源标记；禁止混用不同空间的向量。

## 5. 已提出的创新点与后续方法

### 5.1 感知契约（Perception Contract）

将摄像头规格、BPU 模型 SHA-256、输入布局、颜色空间、量化版本、输出维度作为“感知契约”写入运行配置和每次世界模型 evidence。模型或摄像头一旦变化，必须生成新契约版本并拒绝与旧 embedding 混算。

### 5.2 双层表征

- **低层**：JPEA embedding，保留对视觉状态、变化和场景结构的压缩信息。
- **高层**：由 X5 世界模型/GNN 将 embedding 关联为对象、事件、时空关系和置信度。
- **语言层**：仅将高层摘要、关键 embedding 检索结果和任务意图送至主机 LLM。

这避免 LLM 直接处理连续视觉流，同时减少网络负载与上下文膨胀。

### 5.3 主动采样而非固定高频推理

连续采集不等于每帧跑 JPEA。建议后续实现：

1. 低成本帧差或运动检测在 CPU/NV12 路径进行。
2. 仅当运动、外设事件、任务请求或置信度下降时触发 JPEA。
3. 静态场景采用指数退避采样间隔。
4. 将关键帧 embedding 送入世界模型，非关键帧只更新心跳与时间信息。

这将明显降低 BPU 占用和功耗，并提升对关键事件的响应能力。

### 5.4 编译产物可复现性

每个 `.bin` 应配套保存：

```text
model.bin
model.manifest.json
calibration/             # 校准帧列表与来源说明
model.onnx.sha256
source-checkpoint.sha256
mapper-config.yaml
build.log
```

`model.manifest.json` 至少包括：模型名、输入名称、布局、颜色空间、宽高、量化类型、输出名称、输出形状、输出数据类型、OpenExplorer/mapper 版本、目标 BPU 架构与 SHA-256。

## 6. X5 正式部署步骤

### 6.1 检查摄像头与 BPU

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
ls -l /dev/bpu /dev/bpu_core0 2>/dev/null || true
ls -l /usr/lib/libdnn.so /usr/include/dnn/hb_dnn.h
```

期望摄像头出现 `/dev/video0`，并可看到 `MJPG 1920x1080@30`；BPU 节点和 hbDNN 文件存在。

### 6.2 准备模型

将经 x86_64 OpenExplorer 环境编译并验收的模型放至：

```text
runtime_store/models/ijepa/ijepa_vith14_1k/model.bin
```

模型必须匹配本项目当前适配：BGR/RGB 原始连续输入、目标 JPEA 分辨率和 `128` 维 F32 embedding 输出。若模型实际需要 NV12、量化输出或多个输入张量，必须先修改 `jpea_v2_image_world_model.cpp` 与模型 manifest，不能仅靠环境变量伪装兼容。

### 6.3 构建与启动

```bash
bash tools/build_rdk_x5.sh
HOST_LLAMA_SERVER_URL=http://<HOST_LAN_IP>:8082 bash tools/run_rdk_x5.sh
```

常用覆盖：

```bash
HOST_LLAMA_SERVER_URL=http://192.168.1.10:8082 \
JPEA_CAMERA_DEVICE=/dev/video0 \
JPEA_CAMERA_WIDTH=1920 \
JPEA_CAMERA_HEIGHT=1080 \
JPEA_CAMERA_FPS=30 \
bash tools/run_rdk_x5.sh
```

### 6.4 验收

```bash
curl -sS -X POST http://127.0.0.1:5080/camera/analyze \
  -H 'Content-Type: application/json' \
  -d '{}'
```

验收响应必须同时满足：

- `ok: true` 
- `backend: horizon-hbdnn` 
- `frameSpec.width: 1920` 
- `frameSpec.height: 1080` 
- `embeddingDim: 128` 

若返回 `503`，优先检查 `.bin`、BPU 节点和 hbDNN 错误信息。若返回 `409`，检查摄像头是否改为了另一种分辨率、帧格式或像素类型。

## 7. 主机端正式运行步骤

### 7.1 只启动 llama-server

Linux：

```bash
llama-server \
  -m '<ABSOLUTE_GGUF_PATH>' \
  --host 0.0.0.0 \
  --port 8082 \
  --ctx-size 4096 \
  --parallel 1 \
  --threads '<HOST_CPU_THREADS>'
```

Windows PowerShell：

```powershell
.\llama-server.exe -m '<ABSOLUTE_GGUF_PATH>' --host 0.0.0.0 --port 8082 --ctx-size 4096 --parallel 1 --threads <HOST_CPU_THREADS>
```

### 7.2 网络和安全验收

主机本机：

```bash
curl http://127.0.0.1:8082/health
curl http://127.0.0.1:8082/props
```

X5：

```bash
curl http://<HOST_LAN_IP>:8082/health
```

规则：

- 主机防火墙仅允许 X5 IP 访问 TCP `8082`。
- 不要将 `8082` 映射到公网。
- X5 使用 `AI_LLAMACPP_BASE_URL=http://<HOST_LAN_IP>:8082`。
- 主机在正式生命周期不启动 Phoenix、Docker 推理服务、摄像头采集或 BPU 服务。

## 8. 模型转换环境：独立 x86_64 主机或临时转换容器

`hb_mapper` 是 x86_64 开发工具，不是 aarch64 X5 运行时组件。应使用独立转换机；若只有主机可用，可在维护窗口短暂启动转换容器，生成 `.bin` 后立即退出。该维护行为不属于 Phoenix 正式服务生命周期。

标准流程：

```text
I-JEPA checkpoint -> 导出 ONNX -> hb_mapper checker -> 校准数据 -> PTQ/QAT -> hb_mapper makertbin -> .bin + manifest -> 复制到 X5 -> X5 验收
```

转换前必须确认：

1. 模型架构可导出为固定输入尺寸的 ONNX。
2. 目标图中没有 Horizon 不支持的算子；先运行 `hb_mapper checker`。
3. 校准集来自真实 X5 摄像头相同规格的代表性帧，覆盖室内、室外、暗光、运动、主体近远等场景。
4. 明确目标 BPU 架构和 OpenExplorer 版本与 X5 `hobot-dnn` 运行时兼容。
5. 不能以“输出存在”为验收；必须用固定 golden frames 对比浮点 ONNX 与 `.bin` 的余弦相似度、维度和稳定性。

## 9. 中国网络环境与镜像策略

### 9.1 基本原则

- 对**操作系统包、Python 包、Docker 基础镜像**优先使用稳定的国内镜像。
- 对**Horizon/D-Robotics OpenExplorer 专有包、BPU runtime、官方模型工具链**必须优先使用官方发布物；国内镜像仅做网络加速，不替换来源完整性。
- 下载后保存 SHA-256、版本号和下载来源；不要从网盘、未知 Git 仓库或二次打包站下载 BPU 编译器与 `.bin`。
- 账号、密码、Token 不写入脚本、Git、JSON 配置或日志。使用环境变量、凭据管理器或一次性登录会话。

### 9.2 Ubuntu APT 国内镜像

X5 为 Ubuntu 22.04。修改前备份：

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
```

可选镜像之一：

```text
https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/
https://mirrors.aliyun.com/ubuntu-ports/
https://mirrors.cloud.tencent.com/ubuntu-ports/
```

注意 aarch64 应使用 `ubuntu-ports`，不是普通的 `ubuntu` 路径。修改后：

```bash
sudo apt update
sudo apt install -y build-essential pkg-config v4l-utils ffmpeg
```

Horizon/D-Robotics apt 源保持系统镜像自带或官方配置，不要把 `hobot-*` 包替换成第三方来源。

### 9.3 Python / PyPI 镜像

临时使用清华镜像：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx onnxruntime numpy
```

持久配置示例：

```ini
# ~/.config/pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
```

安装模型转换依赖前固定版本并导出锁定文件：

```bash
python3 -m pip freeze > requirements-conversion.lock.txt
```

### 9.4 Docker 镜像加速

转换机若使用 Docker，可在 Docker daemon 配置中加入企业/区域可达的镜像加速地址。常见可用性会随地区和时间变化，应由运维选择并验证，例如 DaoCloud 的代理域名或云厂商提供的专属加速器。

对于公开镜像，显式写出代理前缀比“全局不透明替换”更容易审计：

```bash
# 示例：确认组织策略允许后使用。
docker pull docker.m.daocloud.io/library/ubuntu:22.04
```

OpenExplorer 专有镜像/离线包的首选方式：

1. 从 D-Robotics 官方下载页或官方账号渠道下载。
2. 在具备网络的 x86_64 转换机验证 SHA-256。
3. 使用 `docker load -i <official-offline-image>.tar.gz` 导入离线镜像，或按官方说明执行安装脚本。
4. 导入后执行 `hb_mapper --help`，记录版本。

不要假设任意国内 Docker 代理一定包含 D-Robotics 私有镜像；若拉取失败，使用官方离线包传输，而不是替换为未知同名镜像。

### 9.5 Git、Git LFS 与 Hugging Face

- Git 可使用公司内部 Git 镜像或 GitHub 加速服务，但只克隆官方上游仓库并校验提交/tag。
- 大 checkpoint 常经 Git LFS 或 Hugging Face 分发；国内网络中应预先下载到受控缓存盘，校验 SHA-256，再导入转换环境。
- 对需要授权的模型，仅使用用户已有的官方 Token；不得把 Token 提交至代码库或共享文档。

## 10. 故障矩阵

| 症状 | 首先检查 | 处理 |
|---|---|---|
| X5 看不到摄像头 | `v4l2-ctl --list-devices` | 检查 USB 供电、线缆、`uvcvideo`、重新插拔 |
| `/camera/analyze` 返回 503 | `.bin`、`/dev/bpu*`、`libdnn.so` | 按响应错误检查模型路径、BPU runtime 与模型 ABI |
| `/camera/analyze` 返回 409 | 摄像头实际规格 | 恢复 MJPG `1920x1080@30` 或同步修改并重新编译模型契约 |
| BPU 输出维度错误 | `.bin` manifest | 重新检查输出 tensor、F32/量化类型和 `JPEA_IMAGE_CONCEPT_DIM` |
| X5 不能调用 LLM | 主机 `/health`、路由、绑定地址、防火墙 | 主机监听 `0.0.0.0`，只对 X5 放行 `8082` |
| 主机内存不足或慢 | `--ctx-size`、`--parallel`、线程 | 先降到 `4096/1`，再测量调优 |
| 下载慢或失败 | 镜像地址、DNS、证书 | apt/PyPI 采用国内镜像；Horizon 包使用官方离线包和 SHA-256 |
| `hb_mapper` 不存在于 X5 | 架构角色 | 这是正常状态；在 x86_64 转换环境执行 |

## 11. 上线前检查清单

- [ ] 主机 `llama-server` 在 LAN `8082` 健康检查通过。
- [ ] X5 可访问主机 `/health`。
- [ ] `/dev/video0` 可采集 MJPG `1920x1080@30`。
- [ ] `hobot-dnn`、`libdnn.so`、BPU 节点存在。
- [ ] JPEA `.bin` 有完整 manifest、哈希和与运行时匹配的 ABI。
- [ ] `.bin` 输入/输出与 Phoenix JPEA 适配器一致。
- [ ] `POST /camera/analyze` 返回真实 `horizon-hbdnn` embedding。
- [ ] 世界模型记录 embedding 来源、模型版本和感知契约版本。
- [ ] 主机未运行 Phoenix、摄像头服务、BPU 服务或额外 LLM 生命周期进程。
- [ ] 备份 X5 运行配置、模型 manifest、构建日志与已验证版本。

## 12. 下一阶段优先级

1. 建立独立 x86_64 OpenExplorer 转换环境，产出第一版真实 JPEA `.bin`。
2. 以 X5 摄像头帧建立校准集和 golden-frame 回归集。
3. 部署 `.bin` 并完成 `/camera/analyze` 端到端验收。
4. 引入单采集线程 + 环形缓冲区 + 主动采样，支持稳定连续感知。
5. 增加模型 manifest 校验、BPU 延迟指标和 embedding 漂移监控。
6. 有真实语音模型后，再以相同 fail-closed 原则接入语音 JPEA。

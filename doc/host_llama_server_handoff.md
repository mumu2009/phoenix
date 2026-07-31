# 主机 Llama Server 交接说明

## 责任边界

- 主机负责 Llama 3.1 8B 的完整权重加载、长上下文、文本生成和最终自然语言输出。
- RDK X5 不运行 GGUF LLM 服务；它保留给 hbDNN/BPU 视觉编码、分类、轻量世界模型，以及 CPU 上的 GNN 控制与图优化。
- 板端通过 HTTP 调用主机的 llama-server；Linux 远端 TCP 和 `/health`、`/props`、`/v1/models` 健康检查已经在 `external_runtime.cpp` 实现。

## 模型与端口

- 模型文件：`GGUF_models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`
- 模型：Meta Llama 3.1 8B Instruct，Q4_K_M，约 4.58 GiB。
- 主机服务端口：`8082`。
- 监听地址：主机 LAN 地址对应的接口或 `0.0.0.0`。不得只绑定 `127.0.0.1`，否则 X5 无法连接。
- 防火墙：仅允许 X5 的 LAN 地址访问 TCP 8082；不要把端口暴露到公网。

## 主机启动

在模型目录可访问、且 `llama-server` 与主机架构匹配的前提下执行：

```powershell
.\llama-server.exe -m "<ABSOLUTE_GGUF_PATH>" --host 0.0.0.0 --port 8082 --ctx-size 4096 --parallel 1 --threads <HOST_CPU_THREADS>
```

Linux 主机等价命令：

```bash
llama-server -m "<ABSOLUTE_GGUF_PATH>" --host 0.0.0.0 --port 8082 --ctx-size 4096 --parallel 1 --threads "<HOST_CPU_THREADS>"
```

先以 `--parallel 1` 和 `--ctx-size 4096` 验证。只有在主机内存和延迟监控稳定时，才提高并发或上下文；每个额外并发槽位和更大的上下文都会增加 KV cache。

## 主机验收

在主机本机执行：

```bash
curl http://127.0.0.1:8082/health
curl http://127.0.0.1:8082/props
```

预期 `/health` 返回 HTTP 200 和 `{"status":"ok"}`。再从 X5 执行：

```bash
curl http://<HOST_LAN_ADDRESS>:8082/health
```

X5 返回 HTTP 200 后，编辑 `runtime_store/start_079_launcher.json`：

```json
"llamacpp_base_url": "http://<HOST_LAN_ADDRESS>:8082"
```

`external_auto_launch` 必须保持 `false`：板端不允许尝试启动 Windows `llama-server.exe`。

## 故障定位

- X5 状态为 `connection_failed`：检查主机 IP、主机防火墙、监听地址，以及 X5 到主机的 TCP 8082 路由。
- X5 状态为 `health_timeout`：TCP 已连接但模型仍在加载，或 llama-server 的 `/health` 未就绪。等待模型加载完成并检查主机日志。
- 主机内存压力高：降低 `--ctx-size`、保持 `--parallel 1`，或选择更小的 GGUF。
- 不要将 I-JEPA `.safetensors` 直接交给 hbDNN；必须先在主机转换、量化并编译为 Horizon `.bin`，再部署到 X5。

## X5 BPU 交接

`edge_platform.cpp` 已在请求提供 `bpuModelPath` 时调用 `rdk_x5_bpu::execute`。部署 Horizon 模型时，请一并提供输入文件布局、色彩格式、尺寸、量化参数和输出张量语义。

## X5 全量部署边界

- 主机只能运行 `llama-server`。不得在主机启动 Phoenix、视频解码、JPEA、hbDNN、GNN、Redis、SQLite 或边缘外设进程。
- X5 运行 `phoenix_main`、HTTP 前端、视频接收/解码、JPEA、hbDNN/BPU、GNN-GA、世界模型、数据库及所有边缘外设逻辑。
- X5 仅以 HTTP 调用主机的 `llama-server`；`AI_LLAMACPP_BASE_URL` 必须设置为主机 LAN URL。
- 使用 `runtime_store/rdk_x5_launcher.json` 作为 X5 配置模板，并以 `bash tools/build_rdk_x5.sh` 和 `HOST_LLAMA_SERVER_URL=http://<HOST_LAN_ADDRESS>:8082 bash tools/run_rdk_x5.sh` 构建、启动。

## X5 摄像头到 JPEA

- X5 已识别 UVC 摄像头：`/dev/video0` 是视频节点，`/dev/video1` 是 UVC 元数据节点。固定使用 `/dev/video0` 的 `MJPG 1920x1080@30`。
- 调用 `POST /camera/analyze`（空 JSON 请求体）直接读取 X5 摄像头。`videoBase64` 与 `videoPath` 被明确拒绝，主机不再传输视频。
- 服务打开 V4L2 设备后请求 MJPG、1920x1080、30 FPS，丢弃最多 11 个预热帧，再读取一帧。帧必须是连续 `CV_8UC3` BGR；宽、高或像素类型偏离锁定规格时返回 HTTP 409。
- JPEA 直接接收摄像头 BGR 字节，不经 JPEG 或视频容器重编码；随后缩放为 JPEA 模型分辨率并按 `JPEA_IMAGE_INPUT_COLOR` 生成 BGR 或 RGB BPU 输入张量。
- 生产运行仅接受 `JPEA_IMAGE_HORIZON_MODEL` 指向的已编译 Horizon `.bin`。没有该文件、没有 `/dev/bpu` 或 hbDNN、输入张量大小不匹配、输出不是目标维度的 F32 embedding 时，接口返回失败；不会产生伪 embedding。
- 工程当前只有 I-JEPA `.safetensors`，没有可部署的 JPEA `.bin`。OpenExplorer 的 `hb_mapper` 是 **x86_64 开发机或其 Docker 容器** 使用的转换工具，不应安装在 aarch64 X5。必须在该开发环境完成 checkpoint -> ONNX -> 定点量化 -> `.bin` 编译，再将 `.bin` 复制到 X5。

## 已移除的降级行为

- 图像 JPEA 不再返回确定性统计特征；只调用 hbDNN。
- 语音 JPEA 不再返回确定性统计特征；在没有独立的真实语音 `.bin` 前保持不可用。
- 模型缺失或 BPU 不可用时，世界模型请求必须失败并暴露错误，不能将输出标为 JPEA 结果。

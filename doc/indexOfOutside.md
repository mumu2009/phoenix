# outsides 核心文件级索引（v5.0 升级）

说明：本索引按 `outsides/` 下“每个独立克隆项目”列出核心业务逻辑文件，强调**文件路径、名称、作用**，而不是仅到文件夹级别。

---

## 1) BitNet（`outsides/BitNet`）

### 核心实现文件
- `src/ggml-bitnet-lut.cpp`
  - 作用：BitNet 的 LUT（查表）型低比特推理核心算子实现。
- `src/ggml-bitnet-mad.cpp`
  - 作用：BitNet 的 MAD/向量化低比特计算核心，实现 CPU 指令优化路径。
- `run_inference.py`
  - 作用：推理启动脚本，组织模型路径、上下文长度、温度、线程数并调用 `llama-cli`。
- `setup_env.py`
  - 作用：环境初始化与运行准备脚本。

### 配置/构建关键文件
- `CMakeLists.txt`：构建入口。
- `requirements.txt`：Python 侧依赖。

---

## 2) llama（`outsides/llama`）

### 核心实现文件
- `llama/model.py`
  - 作用：Transformer 主体结构定义（含模型参数结构与层实现）。
- `llama/generation.py`
  - 作用：自回归生成逻辑（采样、步进解码）。
- `llama/tokenizer.py`
  - 作用：SentencePiece 分词封装。
- `example.py`
  - 作用：端到端推理示例入口（多卡并行初始化 + 权重加载 + 生成）。

### 配置/下载关键文件
- `download.sh`：官方权重与 tokenizer 下载脚本。
- `setup.py`：Python 包安装入口。

---

## 3) llama.cpp（`outsides/llama.cpp`）

> 该目录是迁移到 AtomGit 的发行/打包分发项目，不含完整上游源码树。

### 核心文件
- `README.md`
  - 作用：项目定位、部署方式、命令行用法说明。
- `llama.cpp.spec`
  - 作用：RPM/打包规范，定义安装产物与构建规则。
- `Dockerfile-llama`
  - 作用：容器化运行镜像构建脚本。
- `backport-CVE-2025-49847.patch`
- `backport-CVE-2025-52566.patch`
- `backport-CVE-2025-53630.patch`
  - 作用：安全补丁回移文件。

---

## 4) llamacpp（`outsides/llamacpp`，上游主源码）

### 核心实现文件（C/C++）
- `include/llama.h`
  - 作用：公共 C API 头文件（跨语言调用基准接口）。
- `src/llama.cpp`
  - 作用：核心实现入口，包含模型加载、推理主路径与大部分 API 行为。
- `src/llama-model.cpp`
  - 作用：模型结构/权重与元信息处理核心。
- `src/llama-context.cpp`
  - 作用：上下文状态、KV cache 与推理会话管理。
- `src/llama-sampling.cpp`
  - 作用：采样策略实现（温度、top-k/top-p 等）。
- `src/llama-impl.cpp`
  - 作用：日志、计时、底层工具实现。

### 配套关键文件
- `convert_hf_to_gguf.py`
  - 作用：HuggingFace 权重转 GGUF。
- `convert_lora_to_gguf.py`
  - 作用：LoRA 转换。

---

## 5) xllamacpp（`outsides/xllamacpp`）

### 核心实现文件（Python/Cython/C++桥接）
- `src/xllamacpp/xllamacpp.pyx`
  - 作用：Cython 主封装层，对外暴露 Python API。
- `src/xllamacpp/llama_cpp.pxd`
  - 作用：Cython 到 `llama.cpp` C/C++ 类型与函数声明桥。
- `src/xllamacpp/server.cpp`
  - 作用：C++ 服务实现（OpenAI 风格接口相关能力）。
- `src/xllamacpp/server.h`
  - 作用：服务类声明（`xllamacpp::Server`）。
- `src/xllamacpp/server.pxd`
  - 作用：Cython 与 `server.h` 的绑定声明。
- `src/xllamacpp/__init__.py`
  - 作用：Python 包导出入口。

### 连接关系（精简）
- Python 调用 `xllamacpp.pyx`。
- `xllamacpp.pyx` 经 `.pxd` 声明调用 `server.cpp` 与 `llama.cpp` API。
- 底层实际推理由 `llamacpp`（C/C++）执行。

---

## 6) swanlab（`outsides/swanlab`）

### 核心实现文件
- `swanlab/__main__.py`
  - 作用：CLI 启动入口（调用 `swanlab.cli`）。
- `swanlab/cli/__init__.py`
  - 作用：CLI 导出入口。
- `swanlab/api/__init__.py`
  - 作用：OpenAPI 高层对象封装（Api/Project/User/Workspace）。
- `swanlab/core_python/client/__init__.py`
  - 作用：HTTP 客户端核心，实现会话、请求与鉴权流程。
- `swanlab/core_python/uploader/__init__.py`
  - 作用：上传线程与上传模型导出。
- `swanlab/core_python/__init__.py`
  - 作用：核心 Python 业务层汇总导出。

### 配置关键文件
- `pyproject.toml`：构建与发布配置。
- `requirements.txt`：运行依赖。

---

## 7) OpenClaw-2026.2.17（`outsides/OpenClaw-2026.2.17`）

> 当前为 macOS 应用包分发形态，非源码仓。

### 核心文件
- `OpenClaw.app/Contents/MacOS/OpenClaw`
  - 作用：主可执行二进制。
- `OpenClaw.app/Contents/Info.plist`
  - 作用：应用元信息（版本、Bundle ID、URL Scheme、系统要求）。
- `OpenClaw.app/Contents/Frameworks/`
  - 作用：运行时依赖框架。
- `OpenClaw.app/Contents/Resources/`
  - 作用：资源文件。

---

## 8) Redis-8.0.3-Windows-x64-cygwin-with-Service（`outsides/Redis-8.0.3-Windows-x64-cygwin-with-Service`）

### 核心文件
- `redis-server.exe`
  - 作用：Redis 服务主进程。
- `redis-cli.exe`
  - 作用：Redis 命令行客户端。
- `RedisService.exe`
  - 作用：Windows 服务包装进程。
- `redis.conf`
  - 作用：Redis 主配置文件。
- `redis-full.conf`
  - 作用：完整配置模板。
- `start.bat`
  - 作用：快捷启动脚本。
- `install_redis_service.bat`
  - 作用：安装为系统服务。
- `uninstall_redis_service.bat`
  - 作用：卸载系统服务。

---

## 9) DGL（`outsides/dgl`）

### 核心实现文件
- `python/dgl/__init__.py`
  - 作用：DGL Python 包总入口，聚合图结构、采样、变换、后端绑定等核心模块。
- `python/dgl/heterograph.py`
  - 作用：异构图核心数据结构 `DGLHeteroGraph` 的主要实现（节点/边类型、视图、操作接口）。
- `python/dgl/graph.py`
  - 作用：同构图与基础图操作实现。
- `src/`（C++ 核心目录）
  - 作用：图存储、算子调度、运行时等底层高性能实现（Python API 的底层执行支撑）。
- `CMakeLists.txt`
  - 作用：DGL 原生层构建入口。

---

## 10) GraphScope（`outsides/GraphScope`）

### 核心实现文件
- `python/graphscope/__init__.py`
  - 作用：GraphScope Python SDK 总入口，装配 analytical / interactive / learning / session 能力。
- `coordinator/gscoordinator/__init__.py`
  - 作用：协调器包入口（集群会话与任务调度相关）。
- `coordinator/gscoordinator/coordinator.py`
  - 作用：协调器主流程实现（实例管理、任务编排、状态控制）。
- `gsctl.py`
  - 作用：项目级运维/管理脚本入口。
- `python/setup.py`
  - 作用：Python 包构建与安装入口。

---

## 11) ONNX Runtime（`outsides/onnxruntime`）

> 备注：该仓库 README 顶部标注“停止维护”，集成时建议锁定可审计版本并结合安全扫描。

### 核心实现文件
- `onnxruntime/core/session/inference_session.cc`
  - 作用：推理会话核心实现（模型加载、执行图初始化、执行流程控制）。
- `onnxruntime/core/session/onnxruntime_c_api.cc`
  - 作用：C API 主入口实现（跨语言调用桥）。
- `onnxruntime/python/onnxruntime_inference_collection.py`
  - 作用：Python 高层推理接口封装（Provider 参数归一化、Session 调用管理）。
- `onnxruntime/python/onnxruntime_pybind_state.cc`
  - 作用：Python 绑定核心（PyBind 暴露原生能力）。
- `setup.py`
  - 作用：Python 打包/安装入口。

---

## 12) RKNN（`outsides/rknn`）

> 当前仓库主要为转换后的 RKNN 模型工件集合，不是完整训练/推理源码仓。

### 核心文件
- `RK3588/ppocrv4_det.rknn`
  - 作用：RK3588 平台 OCR 检测模型二进制。
- `RK3588/ppocrv4_rec.rknn`
  - 作用：RK3588 平台 OCR 识别模型二进制。
- `RK3576/ppocrv4_det.rknn`
  - 作用：RK3576 平台 OCR 检测模型二进制。
- `RK3576/ppocrv4_rec.rknn`
  - 作用：RK3576 平台 OCR 识别模型二进制。
- `README.md`
  - 作用：模型来源与用途说明。

---

## 13) SGLang（`outsides/sglang`）

### 核心实现文件
- `python/sglang/launch_server.py`
  - 作用：服务启动总入口（根据参数选择 HTTP / gRPC / encoder-only 模式）。
- `python/sglang/srt/entrypoints/http_server.py`
  - 作用：SRT 推理服务 HTTP API 主实现（FastAPI）。
- `python/sglang/srt/entrypoints/grpc_server.py`
  - 作用：SRT 推理服务 gRPC 入口实现。
- `python/sglang/srt/entrypoints/engine.py`
  - 作用：运行时引擎装配与调度入口。
- `python/sglang/cli/`
  - 作用：命令行工具实现。

---

## 14) SpikingJelly（`outsides/spikingjelly`）

### 核心实现文件
- `spikingjelly/activation_based/neuron.py`
  - 作用：脉冲神经元核心实现（膜电位更新、发放、重置与 surrogate 梯度路径）。
- `spikingjelly/activation_based/surrogate.py`
  - 作用：替代梯度函数实现。
- `spikingjelly/datasets/`
  - 作用：神经形态数据集加载与处理。
- `spikingjelly/__init__.py`
  - 作用：包入口与模块导出。
- `setup.py`
  - 作用：安装与打包入口。

---

## 15) vLLM（`outsides/vllm`）

### 核心实现文件
- `vllm/entrypoints/openai/api_server.py`
  - 作用：OpenAI 兼容 API 服务入口（FastAPI + engine client）。
- `vllm/v1/engine/llm_engine.py`
  - 作用：v1 引擎主实现（请求生命周期、采样与输出组织、并行调度协同）。
- `vllm/engine/llm_engine.py`
  - 作用：兼容层引擎入口（别名映射到 v1 引擎）。
- `vllm/model_executor/`
  - 作用：模型执行器核心（算子执行、设备调度相关）。
- `vllm/config/`
  - 作用：推理与并行配置定义。

---

## 建议（给 v5.0 集成）

- 需要“可二次开发源码”的优先对接：`llamacpp`、`xllamacpp`、`BitNet`、`swanlab`、`vllm`、`sglang`。
- 需要“图学习/图计算能力”优先对接：`dgl`、`GraphScope`（按场景选库，避免重复建设）。
- 需要“可直接部署运行”的优先对接：`Redis-8.0.3...`、`OpenClaw-2026.2.17`。
- `llama.cpp`（该目录）应视为分发/打包项目；若需深入改内核，应以 `llamacpp` 为主。
- `rknn` 当前以模型工件为主，建议补充对应推理脚本仓后再纳入“可开发模块”。

# Phoenix v7.0 "Arthur"

A set of facilities based on LLM, which can efficiently boost the speed and accuracy of LLM in long context, making it more like a person.

> This branch is the v7.0 "Arthur" upgrade of the Phoenix system, focused on true multimodal fusion, primal-sensation/instinct/emotion integration, and comprehensive test coverage.

---

## Features

### Core AI Engine
- **MemeGraph (GNN)** — associative memory graph storing learned concepts ("memes") as nodes with weighted edges; supports incremental learning, decay, n-gram indexing, and graph export
- **SparkArray ensemble** — multi-AI voting layer dispatching queries across a controller pool; supports `PersonaForestAverager` and pluggable factory modules; `bigRounds` for iterative consensus
- **Dual-track context** — independent GNN graph-context and semantic context system that feed into the Transformer together; GNN keywords harmonize with context hints via Jaccard alignment scoring
- **Attention-sink context window** — configurable `maxTokens`, `importanceThreshold`, `similarityThreshold`, `semanticChunkSize`, and attention-sink tokens via `config/phoenix_tuned.json`
- **True-multimodal semantic units** — `SemanticUnit`/`SemanticMemory` in `semantic_unit.{hpp,cpp}` with modality-aware fusion, projection, and cosine-similarity search; integrated into `ModernContextManager`, `ContextBuilder`, and `MemeGraph` for v7.0
- **Primal sensation / instinct layer** — biological interoceptive signals (`primal_sensation.{hpp,cpp}`) and innate drives (`instinct.{hpp,cpp}`) with benefit-harm (趋利避害) evaluation wired into `CognitionAutonomyManager`
- **Prompt split** — immutable `SystemPrompt` and dynamic `MemoryPrompt` composed by `PromptComposer` (`prompt_split.{hpp,cpp}`); memory portion is regenerated from context and affect signals each turn
- **External mixed-modal I/O** — `MixedModalPacket`, `MixedModalInputBuffer`, `MixedModalOutputQueue`, and `MixedModalChannelRegistry` (`external_mixed_modal_io.{hpp,cpp}`) translate external text/image/audio/video/sensor payloads into `SemanticUnit` objects
- **Emotion system** — pluggable emotion processing layer (`emotion_system.cpp`) integrated into the response pipeline
- **World model** — scene-level world representation (`world_model.hpp`) for environment-aware reasoning
- **Vision / multimodal input** — `/api/chat` and `/api/transformer/chat` accept `imageContext`, `imageEmbedding`, `imageEmbeddings`, and `vision` payloads; embedding chunks are injected into graph context

### Inference Backends
- **Ollama** — standard and fine-tuning adapter modes (`ollama`, `ollama-fine-tuning`)
- **llama.cpp / llama-server** — GGUF model server with LoRA adapter support (`llamacpp-lora-files`, `llamacpp-lora-init-without-apply`)
- **BitNet** — 1-bit quantized GGUF inference adapter
- **Native built-in Transformer** — self-contained transformer with checkpoint save/load, pre-training, joint GNN+Transformer training, and GA optimization
- **Auto-selection** — if only one `.gguf` is in `GGUF_models/` the backend picks it automatically; multiple candidates require explicit `--llamacpp-model`

### Online Learning
- **Reinforcement Learner (RL)** — learns from dialog outcomes; triggered every N dialogs (configurable `rlEvery`) or on-demand via `/api/learn/reinforce`
- **Adversarial Learner (ADV)** — attack-and-defend training on corpus samples; configurable `advEvery`
- **GNN Genetic Algorithm (GA)** — evolves graph edge weights with configurable `generations`, `population`, `mutationRate`, `mutationScale`, `residualWeight`
- **External style adapter** — fine-tuning from dialog style via `/api/external_style/train_step` and `/api/fine_tuning/run`; bridges learning results to external backends
- **Transformer feedback loop** — user feedback collected at `/api/transformer/feedback`, triggers background training at `/api/transformer/feedback/train`
- **Dialog-triggered auto-learning** — after each completed dialog, RL / ADV / GNN-GA fire automatically based on turn counters; counters reset via `/api/learn/dialog/reset`

### Safety & Monitoring
- **MemeBarrier** — background thread scanning the meme graph for anomalous growth patterns; TextCNN + RNN/LSTM Torch models score and isolate malicious nodes; threshold, scan interval, and all model hyperparameters are runtime-configurable via `config/phoenix_tuned.json`
- **Bug Shooter** — subprocess (`bug_shooter.exe`) monitoring process memory; soft and hard RSS limits with configurable thresholds
- **Optimizer Autonomy** — self-monitoring autonomy agent that proposes GNN and Transformer upgrades (`/api/optimizer/autonomy/iterate`, `/api/gnn/upgrade`, `/api/transformer/upgrade`)
- **Spider Autonomy** — adaptive web-crawl scheduling agent (`/api/spider/autonomy/adapt`)
- **Route metrics** — every API route records latency and success/failure counters; queryable at `/api/monitoring/stats`; training jobs tracked at `/api/monitoring/training`

### Data Pipeline
- **Robots corpus** — text files in `robots/` loaded at startup as the base knowledge corpus; configurable chunk size, warmup limit, shuffle, and autoload
- **LMDB store** — persistent key-value layer with configurable map size; used by MemeGraph and KVMStore for persistent meme and relationship storage
- **SQLite store** — `ai_store.sqlite` for structured entity and session data
- **Redis** — session caching, inter-process pub/sub (`AI-model-workspace` channel), and hot matrix cache (configurable DB index and key prefix)
- **Corpus ingest API** — `/api/corpus/ingest` (single doc), `/api/robots/retrain` (batch with offset/limit), `/api/corpus/online` (web research), `/api/corpus/crawl` (recursive crawl + optional ingest)
- **Study Engine** — asynchronous document queue (`/api/study/enqueue`) for background learning on ingested content
- **Dataset catalog** — register, activate, and govern external datasets; data cleaning profiles via `/api/data/*`
- **Graph export** — export active or per-group meme graphs to JSON files via `/api/export/graph`

### Service Architecture
- **Drogon HTTP gateway** (port 5080) — async C++20 HTTP/1.1 server; thread count auto-detected from CPU cores (4–16), overridable via `AI_HTTP_THREADS`
- **Frontend / Study proxy** (port 5081) — reverse proxy routing `/api/*` → 5080
- **Controller pool** — multiple AI controller instances arranged in named groups; supports single-proc, group-proc, and infer-MP execution modes
- **Shard manager** — query sharding across controller groups for horizontal scaling
- **Redis synchronizer** — cross-process state synchronization and rotation management
- **Snapshot manager** — periodic snapshots of runtime state to `snapshots/` for crash recovery
- **JWT + local-token auth** — all `/api/*` routes require `Authorization: Bearer <token>`; local tokens (`local-{user}-{ts}-{seq}`) or JWT; configurable via env `AI_AUTH_JWT_SECRET`

### Developer Tooling
- **Auto-tuning pipeline** — `tools/auto_tune_phoenix_params.py` runs grid/random search over context window, MemeBarrier, scenario thresholds, and llama-server knobs; writes `config/phoenix_tuned.json`
- **Runtime tuned config** — `runtime_tuned_config.hpp` provides `phoenix::tuned::value(dotPath, fallback)` for zero-rebuild JSON overrides at startup
- **Memory tier benchmark** — `tools/run_memory_tier_benchmark_tui.py` with TUI progress display; scenario turn thresholds loaded from `tools/tuned_scenario_thresholds.py`
- **Module override system** — `module_overrides/` lets external code replace `SparkArray`, `PersonaForestAverager`, and other factory-registered components at link time
- **Split main** — `tools/split_main_cpp.py` splits `main.cpp` into `main_hub_parts/*.inc` segments for parallel compilation and diff readability

---

## Model Deployment Topology (v7.0)

Phoenix can place the three heavy model roles on the local host or on separate
edge devices at startup.  Each role is configured independently:

- `llm` — text generation backend (Ollama / llama.cpp server / BitNet).
- `vision` — image encoder / JEPA world model.
- `speech` — audio / 1D JEPA world model.

Configuration comes from command-line arguments, environment variables, or a
JSON file; later sources override earlier ones.  Examples:

```bash
# 1) Host runs everything (default)
phoenix_main.exe

# 2) Edge LLM on another machine, local vision/speech
phoenix_main.exe \
  --llm-placement remote \
  --llm-remote-url http://192.168.1.10:11434 \
  --llm-remote-method ollama \
  --llm-remote-model llama3.1:8b

# 3) Three models on three devices via JSON config
phoenix_main.exe --model-deployment-config config/model_deployment.json
```

See `doc/v7.0/model_deployment.md` and `config/model_deployment.example.json`
for the full argument list, environment variables, and the HTTP/JSON protocol
used by remote vision and speech endpoints.

Helper tools:

```bash
# Generate a deployment JSON from the command line
python tools/generate_model_deployment_config.py \
  --llm remote --llm-url http://192.168.1.10:11434 --llm-method ollama --llm-model llama3.1:8b \
  --vision remote --vision-url http://192.168.1.11:5000/infer \
  --speech remote --speech-url http://192.168.1.12:5001/infer \
  -o config/model_deployment.json

# Example edge inference server for vision/speech (run on the edge devices)
python tools/model_deployment_edge_example.py --port 5000
```

---

## Quick Start

### Prerequisites

| Dependency | Notes |
|---|---|
| C++20 compiler (g++ / clang++ via MSYS2) | Required |
| CMake + Ninja | Required |
| Conan 2.x | C++ dependency management |
| Redis | Default `redis://127.0.0.1:6379` |
| Ollama *(optional)* | Default `http://127.0.0.1:11434` |
| Python 3.10+ | Prototype layer and tooling |
| `Python314/` directory | Must contain `Python.h`, `python314.lib`, `python314.dll` |

### Build

```powershell
# 1. Install C++ dependencies via Conan
conan install . --build=missing

# 2. Build all binaries
compile.bat

# Optional: build without edge image/speech (RDK X5 BPU / remote endpoints)
# $env:PHOENIX_DISABLE_EDGE_IMAGE = "1"
# $env:PHOENIX_DISABLE_EDGE_SPEECH = "1"
# compile.bat

# Artifacts produced:
#   phoenix_main.exe    — main gateway + AI runtime
#   bug_shooter.exe     — memory monitor subprocess
#   phoenix_sql_cli.exe — SQL CLI
```

### Download a model

```powershell
# Place .gguf models in GGUF_models/ (auto-detected if only one present)
ollama pull llama3.1:8b
# or download directly and place the .gguf file in GGUF_models/
```

### Launch

```powershell
# GUI one-click launcher (recommended)
build_start_079_oneclick_exe.bat   # build launcher exe (once)
start_079_oneclick.bat              # launch GUI

# Or directly (llama.cpp backend)
phoenix_main.exe --transformer-mode=llamacpp --llamacpp-model=GGUF_models/your_model.gguf

# Ollama backend
phoenix_main.exe --transformer-mode=ollama --ollama-model=llama3.1:8b
```

## BPU JEPA Concept-Head Training

The ResNet18-based BPU JEPA encoder (`runtime_store/models/bpu_jepa/resnet18_224`)
stores the 1x1 concept head in a separate CPU-side ONNX file.  To train it on the
frozen ImageNet-pretrained encoder with a VICReg-style loss:

```powershell
python tools/train_bpu_jepa_head.py --model-dir runtime_store/models/bpu_jepa/resnet18_224
```

This overwrites `model_encoder_head.onnx` and updates `model.manifest.json`.
Use `--variance-target` to make the concept values larger (default `2.0`).

## Deployment Matrix Generator

The 649-endpoint deployment space can be generated interactively:

```powershell
python tools/generate_model_deployment_matrix.py
# source the generated env before compiling when edge devices should be disabled
compile_env_model_deployment.bat
compile.bat
```

---

## Architecture

```
User Request
    │
    ▼
GatewayServer (Drogon, :5080)
    │  JWT / local-token auth
    │
    ├──► MemeGraph (GNN)
    │        │  graph query → top-8 memes → keywords
    │        │  Jaccard alignment with input tokens
    │        ▼
    │    GNN context + graph embeddings
    │
    ├──► Context System
    │        │  semantic window, attention sink, context hints
    │        ▼
    │    Semantic context string
    │
    ├──► [Optional] MemeBarrier scan (background thread)
    │        │  TextCNN + RNN/LSTM scoring → isolate malicious nodes
    │
    ├──► SparkArray (ensemble voting across controller pool)
    │        │  PersonaForestAverager + shard routing
    │        ▼
    │    Combined GNN + Context → Transformer backend
    │
    ├──► Transformer Backend (one of):
    │        ├── Ollama          → http://127.0.0.1:11434
    │        ├── llama.cpp       → http://127.0.0.1:8080
    │        ├── BitNet          → http://127.0.0.1:8090
    │        └── Native built-in → in-process
    │
    ├──► Response
    │
    └──► Post-dialog learning (async):
             RL → ADV → GNN-GA  (triggered by turn counters)
```

### Inference backends

| Mode flag | Description | Default port |
|---|---|---|
| `ollama` | Ollama server | 11434 |
| `ollama-fine-tuning` | Ollama with fine-tuning adapter | 11434 |
| `llamacpp` | llama-server (GGUF) | 8080 |
| `bitnet` | BitNet server (GGUF) | 8090 |
| `native` | Built-in Transformer (in-process) | — |
| `off` | Disable external inference | — |

### Service ports

| Service | Default port | Env override |
|---|---|---|
| API Gateway | 5080 | `CONTROLLER_PORT` |
| Frontend / Study proxy | 5081 | `AI_STUDY_PORT` |
| Redis | 6379 | `REDIS_URL` |

---

## CLI Parameters

All parameters use `--key=value`. Each also has an environment variable fallback listed below.

### Network

| Parameter | Env | Default | Description |
|---|---|---|---|
| `--port` | `CONTROLLER_PORT` | `5080` | Gateway listening port |
| `--study-port` | `AI_STUDY_PORT` | `5081` | Frontend proxy port |
| `--gateway-host` | `AI_GATEWAY_HOST` | `127.0.0.1` | Listening address |
| `--http-log` | `AI_HTTP_LOG` | `false` | Log all HTTP requests |

### Concurrency

| Parameter | Env | Default | Description |
|---|---|---|---|
| `--ai-count` | `AI_COUNT` | `7` | Total AI controller instances |
| `--group-count` | `AI_GROUP_COUNT` | `3` | Number of controller groups |
| `--group-size` | `AI_GROUP_SIZE` | `= ai-count` | Controllers per group |
| `--spark-num-ai` | `AI_SPARK_NUM_AI` | `= group-size` | SparkArray voting width |
| `--spark-budget` | `AI_SPARK_BUDGET` | `default` | SparkArray compute budget hint |
| `--single-proc` | `AI_SINGLE_PROC` | `false` | Run all controllers in one process |
| `--group-proc` | `AI_GROUP_PROC` | `false` | Run each group as a subprocess |
| `--group-proc-timeout-ms` | `AI_GROUP_PROC_TIMEOUT_MS` | `12000` | Group subprocess timeout (ms) |
| `--infer-mp` | `AI_INFER_MP` | `false` | Multiprocess inference pool |
| `--infer-workers` | `AI_INFER_WORKERS` | `cpu_count-1` | Worker thread count |

### Inference

| Parameter | Env | Default | Description |
|---|---|---|---|
| `--transformer-mode` | `AI_TRANSFORMER_MODE` | `ollama` | `ollama` / `ollama-fine-tuning` / `llamacpp` / `bitnet` / `native` / `off` |
| `--inference-enabled` | `AI_INFERENCE_ENABLED` | `true` | Enable external inference |
| `--ollama-model` | `AI_OLLAMA_MODEL` | auto | Ollama model name |
| `--ollama-base-url` | `AI_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `--ollama-timeout-ms` | `AI_OLLAMA_TIMEOUT_MS` | `120000` | Ollama request timeout |
| `--ollama-fine-tuning` | `AI_OLLAMA_FINE_TUNING` | `true` | Enable Ollama fine-tuning adapter |
| `--llamacpp-model` | `AI_LLAMACPP_MODEL` | auto | Path to `.gguf` file |
| `--llamacpp-base-url` | `AI_LLAMACPP_BASE_URL` | `http://127.0.0.1:8080` | llama-server URL |
| `--llamacpp-timeout-ms` | `AI_LLAMACPP_TIMEOUT_MS` | `120000` | llama-server request timeout |
| `--llamacpp-lora-files` | `AI_LLAMACPP_LORA_FILES` | — | Comma-separated LoRA adapter paths |
| `--llamacpp-lora-init-without-apply` | `AI_LLAMACPP_LORA_INIT_WITHOUT_APPLY` | `false` | Init LoRA without applying |
| `--llamacpp-fine-tuning` | `AI_LLAMACPP_FINE_TUNING` | `false` | Enable llama.cpp fine-tuning adapter |
| `--bitnet-model` | `AI_BITNET_MODEL` | auto | Path to BitNet `.gguf` file |
| `--bitnet-base-url` | `AI_BITNET_BASE_URL` | `http://127.0.0.1:8090` | BitNet server URL |
| `--bitnet-timeout-ms` | `AI_BITNET_TIMEOUT_MS` | `120000` | BitNet request timeout |
| `--gguf-models-dir` | `AI_GGUF_MODELS_DIR` | `GGUF_models/` | Directory scanned for `.gguf` files |
| `--external-style-sim` | `AI_EXTERNAL_STYLE_SIM` | `false` | Enable style similarity for external backends |
| `--transformer-bootstrap` | `AI_TRANSFORMER_BOOTSTRAP` | `false` | Bootstrap native Transformer from corpus on start |
| `--transformer-bootstrap-docs` | `AI_TRANSFORMER_BOOTSTRAP_DOCS` | `256` | Number of docs for bootstrap |

### Data / Storage

| Parameter | Env | Default | Description |
|---|---|---|---|
| `--base-dir` | `AI_BASE_DIR` | `runtime_store/` | Runtime data root |
| `--redis-url` | `REDIS_URL` | `redis://127.0.0.1:6379` | Redis connection string |
| `--redis-timeout-ms` | `AI_REDIS_CONNECT_TIMEOUT_MS` | `1500` | Redis connect timeout |
| `--channel` | `AI_REDIS_CHANNEL` | `AI-model-workspace` | Redis pub/sub channel |
| `--redis-cache-db` | `AI_REDIS_CACHE_DB` | `1` | Redis DB index for cache |
| `--redis-cache-prefix` | `AI_REDIS_CACHE_PREFIX` | `AI079` | Redis cache key prefix |
| `--db-path` | `AI_DB_PATH` | `runtime_store/ai_store.sqlite` | SQLite database path |
| `--lmdb-dir` | `LMDB_DIR` | `lmdb/` | LMDB root directory |
| `--lmdb-map-mb` | `AI_LMDB_MAP_MB` | `4096` | LMDB map size in MB |
| `--snapshot-dir` | `AI_SNAPSHOT_DIR` | `snapshots/` | Snapshot output directory |
| `--export-dir` | `AI_EXPORT_DIR` | `runtime_store/` | Export / log output directory |
| `--kvm-cache-max` | `AI_KVM_CACHE_MAX` | `50000` | KVM in-memory cache entries |

### Corpus

| Parameter | Env | Default | Description |
|---|---|---|---|
| `--robots-dir` | `AI_ROBOTS_DIR` | `robots/` | Corpus text files directory |
| `--robots-autoload` | `AI_ROBOTS_AUTOLOAD` | `true` | Auto-ingest corpus on boot |
| `--robots-limit` | `AI_ROBOTS_LIMIT` | `200` | Max corpus docs to warmup |
| `--robots-warmup-shuffle` | `AI_ROBOTS_WARMUP_SHUFFLE` | `false` | Shuffle corpus on warmup |
| `--robots-chunk-min` | `AI_ROBOTS_CHUNK_MIN` | `3` | Min words per corpus chunk |
| `--robots-chunk-max` | `AI_ROBOTS_CHUNK_MAX` | `20` | Max words per corpus chunk |
| `--lemma-csv` | `AI_LEMMA_CSV` | `lemma.csv` | Lemmatization dictionary |
| `--lemma-autoload` | `AI_LEMMA_AUTOLOAD` | `false` | Auto-load lemma CSV on boot |
| `--lemma-max-mb` | `AI_LEMMA_MAX_MB` | `64` | Max lemma CSV size in MB |
| `--lemma-force` | `AI_LEMMA_FORCE` | `false` | Force reload even if cached |
| `--tests-autoload` | `AI_TESTS_AUTOLOAD` | `true` | Auto-load test samples |
| `--search-endpoint` | `AI_SEARCH_ENDPOINT` | — | Default web search endpoint URL |

### Learning

| Parameter | Env | Default | Description |
|---|---|---|---|
| `--disable-memebarrier` | `AI_DISABLE_MEMEBARRIER` | `false` | Disable MemeBarrier background scan |
| `--disable-rl` | `AI_DISABLE_RL` | `false` | Disable reinforcement learner |
| `--disable-adv` | `AI_DISABLE_ADV` | `false` | Disable adversarial learner |
| `--disable-learning` | `AI_DISABLE_LEARNING` | `false` | Disable all online learning |
| `--disable-gnn-module` | `AI_DISABLE_GNN_MODULE` | `false` | Disable GNN module entirely |
| `--disable-context-module` | `AI_DISABLE_CONTEXT_MODULE` | `false` | Disable context system |
| `--learning-warmup` | `AI_LEARNING_WARMUP` | `false` | Run one RL+ADV cycle on boot |
| `--sync-standby` | `AI_SYNC_STANDBY_ON_BOOT` | `false` | Sync standby controllers on boot |
| `--tuned-config` | `PHOENIX_TUNED_CONFIG` | `config/phoenix_tuned.json` | Runtime tuned parameters JSON |

### Misc

| Parameter | Env | Default | Description |
|---|---|---|---|
| `--log-mode` | `AI_LOG_MODE` | `release` | Log verbosity (`release` / `debug`) |

---

## API Reference

All routes require `Authorization: Bearer <token>` except `/api/system/status`.

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Main chat — GNN context + Transformer reply + addons + MemeBarrier |
| `POST` | `/api/graph/chat` | GNN-only chat (no addon, no context hint) |
| `POST` | `/api/transformer/chat` | Direct Transformer chat with GNN embeddings |
| `POST` | `/api/array/chat` | SparkArray ensemble — all controllers vote; optionally includes Transformer NLP candidate |

### Transformer

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/transformer/pretrain` | Pre-train native Transformer from corpus samples |
| `POST` | `/api/transformer/joint_train` | Joint GNN + Transformer training step |
| `POST` | `/api/transformer/ga_optimize` | GA-optimize Transformer parameters |
| `POST` | `/api/transformer/verify` | Verify a response with the native Transformer |
| `POST` | `/api/transformer/feedback` | Submit user feedback for a dialog turn |
| `POST` | `/api/transformer/feedback/train` | Trigger training from accumulated feedback |
| `GET` | `/api/transformer/feedback/status` | Feedback training status |
| `GET/POST` | `/api/transformer/params` | Get or set Transformer hyperparameters |
| `POST` | `/api/transformer/checkpoint/save` | Save Transformer checkpoint |
| `POST` | `/api/transformer/checkpoint/load` | Load Transformer checkpoint |
| `POST` | `/api/transformer/upgrade` | Autonomy-proposed Transformer upgrade |
| `POST` | `/api/transformer/modernize` | Autonomy-proposed Transformer modernization |

### Fine-tuning / Style

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/external_style/train_step` | One fine-tuning step on external backend |
| `POST` | `/api/fine_tuning/corpus/add` | Add document to fine-tuning corpus |
| `POST` | `/api/fine_tuning/run` | Run fine-tuning on accumulated corpus |

### Online Learning

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/learn/reinforce` | Trigger RL learning cycle (`cycles` param) |
| `GET` | `/api/learn/reinforce/latest` | Latest RL result |
| `POST` | `/api/learn/gnn_ga` | Trigger GNN GA evolution |
| `GET` | `/api/learn/gnn_ga/latest` | Latest GNN GA result |
| `POST` | `/api/learn/adversarial` | Run adversarial attack-and-defend |
| `GET` | `/api/learn/adversarial/latest` | Latest ADV result |
| `POST` | `/api/learn/thresholds` | Adjust RL/ADV/GNN trigger thresholds |
| `POST` | `/api/learn/dialog/reset` | Reset dialog turn counters |

### MemeBarrier

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/memebarrier/start` | Start barrier scan (optional `maliciousThreshold`) |
| `POST` | `/api/memebarrier/stop` | Stop barrier scan |
| `GET` | `/api/memebarrier/stats` | Scan statistics (scans, isolated, scores) |

### Corpus & Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/corpus/ingest` | Ingest a single document into all controllers |
| `POST` | `/api/corpus/forget` | Remove memes matching criteria |
| `POST` | `/api/corpus/online` | Web research lookup (query or token list) |
| `POST` | `/api/corpus/crawl` | Recursive URL crawl (optional auto-ingest) |
| `POST` | `/api/robots/retrain` | Batch retrain from `robots/` corpus (offset/limit/shuffle) |
| `GET` | `/api/search/config` | Get search configuration |
| `PUT` | `/api/search/config` | Update search configuration |
| `POST` | `/api/search/endpoints/add` | Add a search endpoint URL |
| `POST` | `/api/search/endpoints/remove` | Remove a search endpoint URL |
| `POST` | `/api/export/graph` | Export active meme graph to JSON |
| `POST` | `/api/export/graph/group` | Export graph for a specific controller group |

### Addons / Plugins

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/addons` | List loaded addons |
| `POST` | `/api/addons/add` | Load addon by path or type/name |
| `POST` | `/api/addons/remove` | Unload addon by name |

### Model & Cluster

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/model/lifecycle` | Model lifecycle status |
| `POST` | `/api/model/compress` | Compress model |
| `POST` | `/api/model/explain` | Explain model decision |
| `POST` | `/api/model/deploy` | Deploy model |
| `POST` | `/api/model/update` | Update model |
| `GET/POST` | `/api/model/params` | Get or set runtime model parameters |
| `POST` | `/api/model/params/reset` | Reset model parameters to defaults |
| `GET` | `/api/cluster/status` | Cluster status |
| `POST` | `/api/cluster/nodes` | Cluster node list |
| `POST` | `/api/cluster/route` | Route query to cluster node |
| `POST` | `/api/cluster/feedback` | Send feedback to cluster |

### Autonomy

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/spider/autonomy/status` | Spider autonomy status |
| `POST` | `/api/spider/autonomy/adapt` | Trigger spider adaptation step |
| `GET` | `/api/optimizer/autonomy/status` | Optimizer autonomy status |
| `POST` | `/api/optimizer/autonomy/iterate` | Run optimizer iteration (auto-applies patch) |
| `POST` | `/api/perf/profile` | Apply performance profile patch |
| `POST` | `/api/gnn/upgrade` | Autonomy-proposed GNN upgrade |

### System & Monitoring

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/system/status` | Uptime, load, memory, controller list *(no auth)* |
| `GET` | `/api/system/config` | Active configuration summary |
| `GET` | `/api/system/ai/{name}` | Per-controller metrics |
| `GET` | `/api/groups` | Group list with controller membership |
| `GET` | `/api/groups/{gid}/metrics` | Per-group controller metrics |
| `GET` | `/api/shards` | Shard manager metrics |
| `GET` | `/api/monitoring/stats` | Per-route latency/success counters |
| `POST` | `/api/monitoring/reset` | Reset route counters |
| `GET` | `/api/monitoring/training` | Training job monitor |
| `POST` | `/api/monitoring/training/reset` | Reset training monitor |
| `GET` | `/api/study/status` | Study engine queue status |
| `POST` | `/api/study/enqueue` | Enqueue a document for background study |
| `GET` | `/api/runtime/features` | Runtime feature flags |
| `PATCH` | `/api/runtime/features` | Patch runtime feature flags |
| `GET` | `/api/provider/capabilities` | Backend provider capability matrix |
| `GET` | `/api/array/layers` | SparkArray layer configuration + history |
| `POST` | `/api/array/layers` | Update SparkArray layers |
| `GET` | `/api/array/history` | Last 20 SparkArray dispatch results |

### Dataset

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/dataset/catalog` | Dataset catalog |
| `GET` | `/api/data/governance` | Data governance report |
| `POST` | `/api/dataset/register` | Register external dataset |
| `POST` | `/api/dataset/activate` | Activate a registered dataset |
| `POST` | `/api/data/collect` | Collect data from a dataset |
| `GET/POST` | `/api/data/cleaning/profile` | Get or update data cleaning profile |

---

## Python Prototype Layer

Python is used for prototyping and as the maintainable Cython source. It is **not** the recommended production runtime.

```powershell
python -m pip install -r requirements.txt
python main.py [--port=5080] [--study-port=5081]
```

Frontend proxy at 5081 routes `/api/*` → 5080 and `/auth/*` → 5080/api/auth/\*.

---

## Directory Structure

```
phoenix/
├── main.cpp                      # Gateway entry point; all routing assembled here
├── main_hub_parts/               # Auto-split segments of main.cpp (116 files)
├── frontend_server.cpp           # Frontend service (port 5081) + reverse proxy
├── transformer_main.cpp/.py      # Transformer core (C++ production / Python prototype)
├── runtime_tuned_config.hpp      # Runtime JSON config loader (phoenix::tuned::value)
├── edge_platform.cpp/.hpp        # Edge platform / NPU control abstraction
├── mcu_posix_compat.cpp/.hpp     # Minimal POSIX layer for MCU targets
├── mcu_virtual_memory.hpp        # Virtual memory for external SDRAM/SD
├── emotion_system.cpp/.hpp       # Emotion processing layer
├── modern_context_system.cpp/    # Semantic context management
├── world_model.hpp               # World model and scene representation
├── addons/                       # Built-in addon modules (math, search, shell)
├── auth/                         # Authentication storage and user schema
├── config/                       # Runtime config (phoenix.json, phoenix_tuned.json)
├── doc/                          # Design documents, math proofs, contracts
├── GGUF_models/                  # GGUF model files (gitignored)
├── lmdb/                         # LMDB runtime data (gitignored)
├── module_overrides/             # Factory-registered component overrides
├── runtime_store/                # Runtime state, SQLite, PID files (gitignored)
├── snapshots/                    # Periodic snapshots (gitignored)
├── static/                       # Static web assets
├── tools/                        # Dev tools: benchmark, tuner, investor tests
├── uploads/                      # Upload cache (gitignored)
├── conanfile.txt                 # Conan 2.x dependency manifest
└── compile.bat                   # One-shot build script
```

---

## Conan Dependencies

Declared in `conanfile.txt`:

| Package | Version |
|---|---|
| drogon | 1.9.5 |
| nlohmann_json | 3.11.2 |
| jwt-cpp | 0.7.0 |
| eigen | 3.4.0 |
| redis-plus-plus | 1.3.7 |
| sqlite3 | 3.45.1 |
| lmdb | 0.9.29 |
| opencv | 4.5.5 |
| gtest | 1.14.0 |

---

## Documentation Index

| Document | Content |
|---|---|
| `doc/v3_contract.md` | Interface and contract specification |
| `doc/testing_strategy_v3.md` | Testing strategy |
| `doc/math/transformer_math_proof.md` | GNN + Transformer mathematical proof |
| `doc/math/fine_tuning_math_proof.md` | Style fine-tuning proof |
| `doc/math/fineT_with_GNN.md` | GNN keyword fine-tuning advantage |
| `doc/sparkarray_transformer_ensemble_and_gguf.md` | SparkArray joint weighting |
| `doc/brain_dual_track_and_conscious_compute_20260405.md` | Dual-track brain architecture |
| `doc/indexOfOutside.md` | External submodule file index |
| `doc/algorithm/algorithm.md` | Algorithm overview |
| `module_overrides/README.md` | Module mounting examples |

---

## License

GNU Lesser General Public License v3.0 — see [LICENSE](LICENSE).

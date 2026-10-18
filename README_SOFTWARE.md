# 079 Project v6.0Alixander - Software Components

## Overview

079 Project is an AI system combining Graph Neural Networks (GNN) and Transformer architectures. This document describes the software components only. For hardware-related information, see the `catastrophe/` directory.

## Architecture

### Core Design Philosophy

- **GNN + Transformer Fusion**: Graph neural networks provide structural context and relationship activation, while Transformers handle sequence reasoning and response generation
- **Multi-Backend Support**: Flexible inference backend architecture supporting Ollama, llama.cpp, BitNet, and native implementations
- **Component-Based Configuration**: Modular design allowing selective enablement of GNN, frontend-memory, brain context, dialog learning, and collective routing
- **Python Prototype + C++ Production**: Python for prototyping and algorithm design, Cython-generated C++ for performance-critical production paths

## Technology Stack

### Core Technologies
- **C++20**: Main service implementation using Drogon framework
- **Python 3.14**: Prototyping, algorithm design, and C++ bridging
- **Redis**: Caching and synchronization mechanism
- **Conan 2.x**: C++ dependency management
- **CMake + Ninja**: Build system
- **Drogon**: High-performance C++ web framework

### Inference Backends
- **Ollama**: Default inference backend (port 11434)
- **llama.cpp**: GGUF model inference (port 8082)
- **BitNet**: 1-bit model inference (port 8090)
- **native**: Built-in Transformer implementation

## Core Components

### Main Entry Points

| File | Function |
|------|----------|
| `main.cpp` | Main gateway entry, routing registration, controller assembly, runtime initialization |
| `frontend_server.cpp` / `frontend_server.hpp` | Frontend service (port 5081) and proxy implementation |
| `transformer_main.cpp` / `transformer_main.py` | Transformer main implementation and Python bridge |
| `transformer.hpp` | Transformer core interface definition |

### Core Services

| File | Function |
|------|----------|
| `model_lifecycle.cpp` / `model_lifecycle.hpp` | Model lifecycle management, service state, and switching logic |
| `autonomy_stack.cpp` / `autonomy_stack.hpp` | Autonomy stack, automated workflows, and aggregate metrics |
| `external_runtime.cpp` / `external_runtime.hpp` | External backend integration and adaptation bridge |
| `DATABASE_079.cpp` / `DATABASE_079.hpp` | Database and persistence layer |
| `addon.cpp` / `addon.hpp` | Plugin loading entry and extension registration |
| `gguf_tensor_parser.cpp` / `gguf_tensor_parser.hpp` | GGUF tensor and model format parsing |
| `speak_io.cpp` / `speak_io.hpp` | Speech I/O and related interface bridging |

### Advanced Components

| File | Function |
|------|----------|
| `edge_platform.cpp` / `edge_platform.hpp` | Edge platform adaptation (GPIO, NPU control) |
| `mcu_posix_compat.cpp` / `mcu_posix_compat.hpp` | Minimal POSIX compatibility layer for MCU targets |
| `mcu_virtual_memory.hpp` | Virtual memory subsystem for external SDRAM/PSRAM/SD card paging |
| `emotion_system.cpp` / `emotion_system.hpp` | Emotion processing and adjustment system |
| `modern_context_system.cpp` / `modern_context_system.hpp` | Modern context management system |
| `world_model.hpp` | World model and scene representation |
| `physics_world.hpp` / `physics_world_runtime.cpp` | Physics simulation and world interaction |

### Compatibility Layers

| File | Function |
|------|----------|
| `v51_runtime.cpp` / `v51_runtime.hpp` | v5.1 runtime compatibility layer |
| `transformer_ollama_fine-tuning.cpp` / `transformer_ollama_fine-tuning.py` | Ollama-style fine-tuning and conversion logic |

## Plugin System (addons/)

Built-in plugin modules:

| Plugin | Function |
|--------|----------|
| math_addon | Mathematical expression calculation |
| search_addon | Search functionality |
| computer_shell_addon | Computer shell operations |

## Service Ports

| Service | Port |
|---------|------|
| API Gateway | 5080 |
| Frontend Service | 5081 |
| Ollama | 11434 |
| llama.cpp | 8082 |
| BitNet | 8090 |
| Redis | 6379 |

## Build Process

```powershell
# Download the Models before all
# Set OLLAMA_MODELS to point to the repository's GGUF_MODELS directory
setx OLLAMA_MODELS="%~dp0GGUF_MODELS"

# Download model (example with llama3.1:8b)
ollama pull llama3.1:8b
# Compile the project
compile.bat

# Generated artifacts
- phoenix_main.exe (main program)
- bug_shooter.exe (monitoring process)
- phoenix_sql_cli.exe (SQL CLI)
```

## Runtime Dependencies

1. **Redis**: `redis://127.0.0.1:6379`
2. **Ollama** (optional): `http://127.0.0.1:11434`
3. **Python314 directory**: Must contain `Python.h`, `python314.lib`, `python314.dll`

## API Interface Overview

### Core APIs
- `/api/chat`: Chat interface
- `/api/transformer/chat`: Transformer chat
- `/api/array/chat`: GNN + Transformer joint voting

### Model Lifecycle
- `/api/model/lifecycle`
- `/api/model/compress`
- `/api/model/deploy`

### Cluster and Autonomy
- `/api/cluster/status`
- `/api/spider/autonomy/status`

## Core Algorithm Architecture

### GNN + Transformer Fusion Paradigm

- **GNN**: Graph structure retrieval, relationship activation, semantic cue aggregation
- **Transformer**: Sequence reasoning, response generation, verifiable output
- **Fusion Method**: Graph context text (graphContext) + graph embeddings (graphEmbeddings)

### Inference Pipeline

1. Input text enters graph side (MemeGraph + message passing) to obtain activated subgraph
2. Graph side outputs structured context text and graph embeddings
3. Transformer receives graph-enhanced information and generates response
4. Output reply + verify/latency/graph/addon observable fields

## Component Configuration (v6.0)

The main program supports splitting GNN, frontend-memory, brain context, dialog learning, and collective routing into optional components, enabling/disabling them through JSON, XML, and command line combinations.

### Configuration Priority

1. Code defaults
2. `--component-config=<path>` specified component configuration file
3. `--components=<spec>` specified command line override

### Built-in Examples

- `test/prof/component_presets/full_functional.json`: Functional brain preset
- `test/prof/component_presets/structural_research.xml`: Structural brain preset

### Common Usage

```powershell
phoenix_main.exe --component-config=test/prof/component_presets/structural_research.xml --components=brain=functional,frontend-memory=off
```

## Command Line Parameters

All parameters support `--key=value` format. Default values are used when not specified.

### Core Network and Paths

- `--gateway-host`: Gateway listening address, default `127.0.0.1`
- `--port`: Gateway port, default `5080`
- `--study-port`: Frontend/study service port, default `5081`
- `--frontend-enabled`: Whether to start 5081 frontend subprocess, default `true`
- `--base-dir`: Runtime data root directory, default `runtime_store`
- `--db-path`: SQLite database file path, default `<base-dir>/ai_store.sqlite`
- `--snapshot-dir`: Snapshot directory, default `snapshots`
- `--export-dir`: Export/log directory, default `runtime_store`

### Redis and Cache

- `--redis-url`: Redis address, default `redis://127.0.0.1:6379`
- `--redis-timeout-ms`: Redis connection timeout (ms), default `1500`
- `--channel`: Redis sync channel name, default `AI-model-workspace`
- `--redis-cache-db`: Redis cache DB index, default `1`
- `--redis-cache-prefix`: Redis cache prefix, default `AI079`
- `--kvm-cache-max`: KVM cache max entries, default `50000`

### LMDB and Data Directories

- `--lmdb-dir`: LMDB root directory, default `lmdb`
- `--lmdb-map-mb`: LMDB map size (MB), default `512`
- `--robots-dir`: Corpus directory, default `robots`
- `--search-endpoint`: External search endpoint (optional), default empty
- `--lemma-csv`: Lemmatization dictionary file, default `lemma.csv`
- `--lemma-autoload`: Auto-load lemma on startup, default `false`
- `--lemma-max-mb`: Lemma file size limit (MB), default `64`
- `--lemma-force`: Force refresh lemma, default `false`

### Corpus Warmup

- `--robots-limit`: Startup warmup document count limit, default `200`
- `--robots-autoload`: Auto-load corpus, default `true`
- `--robots-warmup-shuffle`: Randomize during warmup, default `false`
- `--robots-chunk-min`: Corpus chunk min word count, default `2`
- `--robots-chunk-max`: Corpus chunk max word count, default `5`
- `--tests-autoload`: Auto-import `tests/*.txt`, default `true`

### Concurrency and Grouping

- `--ai-count`: AI instance count, default `7`
- `--group-count`: Group count, default `3`
- `--group-size`: Group size, default `7`
- `--spark-num-ai`: Spark array AI count, default follows `group-size`
- `--spark-budget`: Spark budget strategy, default `default`
- `--single-proc`: Single process mode, default `false`
- `--group-proc`: Group multi-process mode, default `false`
- `--group-proc-timeout-ms`: Group process timeout (ms), default `12000`
- `--infer-mp`: Inference multi-process switch, default `false`
- `--infer-workers`: Inference worker count, default `CPU-1`

### Learning and Governance Switches

- `--learning-warmup`: Warmup learning flow after startup, default `false`
- `--sync-standby`: Sync standby/validation on startup, default `false`
- `--disable-learning`: Disable learning main switch, default `false`
- `--disable-rl`: Disable RL, default `false`
- `--disable-adv`: Disable adversarial learning, default `false`
- `--disable-memebarrier`: Disable MemeBarrier, default `false`
- `--disable-gnn-module`: Disable GNN context path, default `false`
- `--disable-context-module`: Disable context module, default `false`

### Inference Backends

- `--using-ollama`: Ollama switch for legacy script compatibility, default `true`
- `--inference-enabled`: Enable inference main chain, default `true`
- `--transformer-mode`: Inference mode (`ollama` / `ollama-fine-tuning` / `llamacpp` / `bitnet` / `off`)
- `--ollama-model`: Ollama model name
- `--ollama-base-url`: Ollama service address, default `http://127.0.0.1:11434`
- `--ollama-timeout-ms`: Gateway wait timeout for Ollama (ms), default `120000`
- `--llamacpp-model`: llama.cpp backend model path (`.gguf`)
- `--llamacpp-base-url`: llama.cpp service address, default `http://127.0.0.1:8082`
- `--llamacpp-timeout-ms`: llama.cpp request timeout (ms), default `120000`
- `--llamacpp-launch-args`: llama.cpp local service auto-launch argument template
- `--bitnet-model`: BitNet backend model path (`.gguf`)
- `--bitnet-base-url`: BitNet adapter service address, default `http://127.0.0.1:8090`
- `--bitnet-timeout-ms`: BitNet request timeout (ms), default `120000`
- `--bitnet-launch-args`: BitNet local service auto-launch argument template
- `--gguf-models-dir`: GGUF model directory, default `GGUF_models/`
- `--external-auto-launch`: Auto-launch local backend when llamacpp/bitnet port not ready, default `true`
- `--external-ready-timeout-ms`: Max wait time for local backend readiness, default `30000`
- `--external-health-poll-ms`: Local backend health poll interval, default `500`
- `--external-style-sim`: Enable frontend-input-driven engine style adaptation training step distribution, default `true`

### Security and Plugin Switches

- `--builtin-addons=math,search`: Select built-in addons to auto-load on startup, pass `none` to disable all
- `--addon-libraries=a.dll;b.dll`: Load additional addon dynamic libraries in order on startup
- `--malicious-threshold`: Control MemeBarrier behavior
- `--memebarrier-disable-rate`: MemeBarrier disable rate
- `--memebarrier-use-cnn`: Use CNN in MemeBarrier
- `--memebarrier-use-recommender`: Use recommender in MemeBarrier
- `--memebarrier-use-torch`: Use Torch in MemeBarrier
- `--mechanical-mind-enabled`: Enable "mechanical mind processing"
- `--mechanical-mind-threshold`: Mechanical mind text-level shielding threshold
- `--mechanical-mind-token-threshold`: Mechanical mind token-level shielding threshold

### Bug Shooter and Pre-crash Control

- `--bug-shooter`: Enable bug shooter, default `true`
- `--bug-shooter-exe`: Bug shooter executable path, default `bug_shooter.exe`
- `--bug-shooter-poll-ms`: Sampling period, default `1500`
- `--bug-shooter-soft-limit-mb`: Soft threshold, default `3072`
- `--bug-shooter-hard-limit-mb`: Hard threshold, default `4096`

### Inference Cache Hot/Cold Exchange

- `--inference-hot-limit`: Hot layer max key count, default `512`
- `--inference-promote-hits`: Minimum access count to promote to hot layer, default `2`
- `--inference-hot-ttl-seconds`: Hot layer TTL, default `3600`
- `--inference-rebalance-every`: Execute hot/cold rebalance every N accesses, default `64`

### Logging

- `--log-mode`: Log mode, default `release`
- `--http-log`: Print each gateway request in console (method/path/status code), default `false`
- `--frontend-http-log`: Frontend 5081 proxy log switch (independent of `--http-log`)

### Process and Concurrency

- `--pid-file`: Main process PID file path, default `runtime_store/phoenix_main.pid`
- Environment variable `AI_PID_FILE`: Override PID file path when `--pid-file` not specified
- Environment variable `AI_HTTP_THREADS`: Gateway HTTP worker thread count, default auto (min 2)
- Environment variable `FRONTEND_HTTP_THREADS`: Frontend HTTP worker thread count, default auto (min 2)

## Testing Strategy

### Priority Levels

1. Syntax and build testing (highest)
2. Functional regression testing
3. Performance baseline testing
4. Code organization and static checking

### Backend Regression Entry

- Compile: `compile.bat`
- Interface regression: `test-tools/api_regression.ps1`
- Covered interfaces:
  - `/api/chat`
  - `/api/transformer/chat`
  - `/api/transformer/verify`
  - `/api/tests/list`
  - `/api/model/lifecycle`
  - `/api/model/compress`
  - `/api/model/explain`
  - `/api/model/deploy`
  - `/api/model/update`

### Failure Classification

- **P0 Blocking**: Compilation failure, main interface unavailable
- **P1 Blocking**: Key function returns broken structure
- **P2 Deferrable**: Performance regression but function correct

## Python Prototype Layer

Python in this repository is primarily used for prototyping, algorithm organization, and as a maintainable source before Cython generation. It is not equivalent to the final production runtime chain.

### Python Environment

- Python 3.10+ (recommend 3.11)
- Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Python Startup

```powershell
python main.py
```

Note: This path is mainly for prototype verification, auxiliary development, or alignment with the C++ main chain. It does not represent the v6.0 official recommended release mode.

- API Gateway: `http://127.0.0.1:5080` (`--port`)
- Frontend Page: `http://127.0.0.1:5081` (`--study-port`)

## Documentation Index

| Document | Content |
|----------|---------|
| `doc/indexOfOutside.md` | External submodule core file index |
| `doc/math/transformer_math_proof.md` | Transformer + GNN mathematical proof |
| `doc/math/fine_tuning_math_proof.md` | Style fine-tuning mathematical proof |
| `doc/math/fineT_with_GNN.md` | GNN keyword fine-tuning advantage proof |
| `doc/sparkarray_transformer_ensemble_and_gguf.md` | SparkArray joint weighting and GGUF startup validation |
| `doc/brain_dual_track_and_conscious_compute_20260405.md` | Dual-track brain architecture and conscious compute |
| `doc/testing_strategy_v3.md` | Testing strategy and validation scope |
| `doc/v3_contract.md` | Interface and contract specification |
| `doc/algorithm/algorithm.md` | Algorithm overview |
| `module_mount.hpp` | Module mounting interface |
| `module_overrides/README.md` | External module mounting examples |

## License

This project is licensed under the GNU Lesser General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Directory Structure (Software Only)

```
v6.0Alixander/
├── main.cpp                          # Main gateway entry
├── frontend_server.cpp               # Frontend service
├── addons/                           # Built-in plugins
├── auth/                             # Authentication data and user storage
├── doc/                              # Design documents and policy documents
├── image/                            # Image resource directory
├── lmdb/                             # LMDB runtime data
├── main_hub_parts/                   # Main file split fragments
├── module_overrides/                 # Module override and mounting extensions
├── robots/                           # Corpus, robot seeds, and loading sources
├── runtime_store/                    # Runtime state, PID, and temporary binding info
├── static/                           # Static resource directory
├── test/                             # Testing entry
├── test-tools/                       # PowerShell automation testing tools
├── tests/                            # Test data files and sample inputs
├── tools/                            # Auxiliary scripts and development tools
└── uploads/                          # Upload cache directory
```

## Notes

- v3.0's core constraint is to maintain the "GNN-enhanced Transformer" combination concept: graph structure context/graph embeddings enhance Transformer's adaptability on long-chain corpus.
- Training set uses hybrid mode: repository small samples + external large samples (repository only retains external data index and validation information).
- Detailed contract specifications can be found in `doc/v3_contract.md`.

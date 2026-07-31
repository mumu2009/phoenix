# v7.0 Model Deployment Topology

Phoenix v7.0 ("Arthur") can place the three heavy models (LLM, vision, speech)
on the same host or on separate edge devices.  The placement is decided at
process startup through command-line arguments, environment variables, or a JSON
configuration file.

## Concepts

- **Local**: the model is loaded in the Phoenix process or on the same machine
  (e.g. a local Ollama / llama.cpp server, or a local RDK X5 BPU).
- **Remote**: the model is served by another host and called over HTTP.
- **Auto**: lets the model factory choose the backend.  If a remote URL is
  configured the factory uses that remote endpoint; otherwise it falls back to
  the local backend, which is fail-closed and will use deterministic fallbacks
  if no local accelerator is available.

Each of the three model types has its own record:

- `llm`  - text generation (Ollama, llama.cpp server, BitNet server).
- `vision` - image world model (JPEA / I-JEPA style encoder).
- `speech` - audio / 1D world model.

## Configuration Sources (precedence high to low)

1. Command-line arguments: `--llm-placement`, `--llm-remote-url`, ...
2. Environment variables: `AI_LLM_PLACEMENT`, `AI_LLM_REMOTE_URL`, ...
3. A JSON file passed with `--model-deployment-config`.

`loadConfig` first applies the existing legacy `transformer-mode`,
`ollama-base-url`, `ollama-model` and related flags, then overwrites those
fields with any remote LLM record defined in the deployment config.  This
means the deployment config takes precedence when it is present.

## Command-line Arguments

### Common pattern

```
--<model>-placement local|remote|auto
--<model>-remote-url <http://host:port/path>
--<model>-remote-method <calling method>
--<model>-remote-model <model name>
--<model>-remote-token <bearer token>
--<model>-remote-timeout-ms <milliseconds>
```

`<model>` is one of `llm`, `vision`, `speech`.

### LLM

- `--llm-placement remote`
- `--llm-remote-url http://192.168.1.10:11434`
- `--llm-remote-method ollama`       (also: `llamacpp`, `bitnet`)
- `--llm-remote-model llama3.1:8b`
- `--llm-remote-timeout-ms 120000`

### Vision

- `--vision-placement remote`
- `--vision-remote-url http://192.168.1.11:5000/infer`
- `--vision-remote-method http-json`
- `--vision-remote-timeout-ms 30000`

### Speech

- `--speech-placement remote`
- `--speech-remote-url http://192.168.1.12:5001/infer`
- `--speech-remote-method http-json`
- `--speech-remote-timeout-ms 30000`

## Environment Variables

Same names as the CLI arguments, uppercased and underscored:

```
AI_LLM_PLACEMENT=remote
AI_LLM_REMOTE_URL=http://192.168.1.10:11434
AI_LLM_REMOTE_METHOD=ollama
AI_LLM_REMOTE_MODEL=llama3.1:8b

AI_VISION_PLACEMENT=remote
AI_VISION_REMOTE_URL=http://192.168.1.11:5000/infer

AI_SPEECH_PLACEMENT=remote
AI_SPEECH_REMOTE_URL=http://192.168.1.12:5001/infer

AI_MODEL_DEPLOYMENT_CONFIG=config/model_deployment.json
```

## JSON Configuration File

A file can be passed with `--model-deployment-config config/model_deployment.json`
or `AI_MODEL_DEPLOYMENT_CONFIG`.

```json
{
  "llm": {
    "placement": "remote",
    "remote": {
      "url": "http://192.168.1.10:11434",
      "method": "ollama",
      "modelName": "llama3.1:8b",
      "timeoutMs": 120000
    }
  },
  "vision": {
    "placement": "remote",
    "remote": {
      "url": "http://192.168.1.11:5000/infer",
      "method": "http-json",
      "timeoutMs": 30000
    }
  },
  "speech": {
    "placement": "remote",
    "remote": {
      "url": "http://192.168.1.12:5001/infer",
      "method": "http-json",
      "timeoutMs": 30000
    }
  }
}
```

## Typical Scenarios

### 1. All local (default)

No extra arguments.  Phoenix uses `transformer-mode` as before and local vision /
speech fallbacks or local BPU.

### 2. Host LLM, edge vision + speech

```bash
phoenix_main.exe \
  --transformer-mode ollama \
  --llm-placement local \
  --vision-placement remote --vision-remote-url http://192.168.1.11:5000/infer \
  --speech-placement remote --speech-remote-url http://192.168.1.12:5001/infer
```

### 3. Edge LLM, edge vision, edge speech

```bash
phoenix_main.exe \
  --llm-placement remote --llm-remote-url http://192.168.1.10:11434 --llm-remote-method ollama --llm-remote-model llama3.1:8b \
  --vision-placement remote --vision-remote-url http://192.168.1.11:5000/infer \
  --speech-placement remote --speech-remote-url http://192.168.1.12:5001/infer
```

### 4. Mixed: three devices, one model per device

```bash
phoenix_main.exe \
  --model-deployment-config config/model_deployment.json
```

with the JSON above.

### 5. Auto placement with edge fallback

```bash
python tools/generate_model_deployment_config.py \
  --llm auto --llm-url http://192.168.1.10:11434 --llm-method ollama --llm-model llama3.1:8b \
  --vision auto --vision-url http://192.168.1.11:5000/infer \
  --speech auto --speech-url http://192.168.1.12:5001/infer \
  -o config/model_deployment.json

phoenix_main.exe --model-deployment-config config/model_deployment.json
```

With `auto`, Phoenix will use the configured remote URL when it is present.  If the
remote is unreachable the factories fall back to the local backends (which are
fail-closed and use deterministic fallbacks if no local accelerator is present).

## Remote HTTP/JSON Protocol for Vision and Speech

When `method` is `http-json`, Phoenix `POST`s a JSON body to the configured URL.

### Vision request

```json
{
  "modality": "image",
  "mimeType": "image/png",
  "width": 224,
  "height": 224,
  "payloadBase64": "...",
  "conceptDim": 128
}
```

### Vision response

```json
{
  "ok": true,
  "embedding": [0.12, -0.34, ...]
}
```

### Speech request

```json
{
  "modality": "audio",
  "mimeType": "audio/wav",
  "sampleRate": 16000,
  "payloadBase64": "...",
  "conceptDim": 128
}
```

### Speech response

```json
{
  "ok": true,
  "embedding": [0.05, 0.78, ...]
}
```

### Decode request

A decode request is sent when Phoenix needs to convert a concept vector back to
raw bytes (e.g. image or audio synthesis).  It contains the floating-point
concept vector produced by the encode step and an optional `lengthHint` for
audio:

```json
{
  "modality": "image",
  "decode": true,
  "mimeType": "image/png",
  "conceptVector": [0.12, -0.34, ...]
}
```

```json
{
  "modality": "audio",
  "decode": true,
  "mimeType": "audio/pcm",
  "lengthHint": 16000,
  "conceptVector": [0.05, 0.78, ...]
}
```

### Decode response

```json
{
  "ok": true,
  "modality": "image",
  "mimeType": "image/png",
  "payloadBase64": "..."
}
```

On failure the remote service should return `{"ok":false,"error":"..."}`.

## Helper Tools

- `tools/generate_model_deployment_config.py` — writes a JSON deployment config
  from command-line arguments.
- `tools/model_deployment_edge_example.py` — a minimal Python HTTP/JSON edge
  inference server that returns deterministic embeddings and decode payloads for
  vision/speech.  Replace the stub embedding/synthesis with a real model on the
  edge device.

```bash
# Generate a three-device config
python tools/generate_model_deployment_config.py \
  --llm remote --llm-url http://192.168.1.10:11434 --llm-method ollama --llm-model llama3.1:8b \
  --vision remote --vision-url http://192.168.1.11:5000/infer \
  --speech remote --speech-url http://192.168.1.12:5001/infer \
  -o config/model_deployment.json

# Start the example edge server for vision/speech on the edge devices
python tools/model_deployment_edge_example.py --port 5000
python tools/model_deployment_edge_example.py --port 5001
```

## Compile-Time Edge Device Selection

Phoenix also supports conditional compilation for the edge image and speech
backends.  This is useful when building a binary that should never use the RDK X5
BPU or remote vision/speech endpoints (for example, a pure text/robotics build
on a machine without the Horizon SDK).

Set these environment variables before running `compile.bat` or `compile_gtest.bat`:

```powershell
# Disable both edge image (BPU + remote) and edge speech (remote)
$env:PHOENIX_DISABLE_EDGE_IMAGE = "1"
$env:PHOENIX_DISABLE_EDGE_SPEECH = "1"
compile.bat
```

| Macro | Default | When set to `0` at compile time |
|---|---|---|
| `PHOENIX_EDGE_IMAGE_ENABLED` | `1` | `rdk_x5_bpu.cpp` does not include or link `dnn/hb_dnn.h`; `createJpeaV2ImageWorldModel()` returns a deterministic fallback. |
| `PHOENIX_EDGE_SPEECH_ENABLED` | `1` | `createJpeaV2SpeechWorldModel()` ignores remote placement and always uses the local fallback. |

The defaults keep the existing v7.0 behavior (BPU/local and remote endpoints are
enabled and selected by runtime deployment configuration).

## Implementation Notes

- `model_deployment.{hpp,cpp}` defines the topology, parses configuration and
  provides `RemoteModelClient` for synchronous HTTP calls.
- `jpea_v2_image_world_model.cpp` and `jpea_v2_speech_world_model.cpp` query
  `ModelDeploymentConfig::instance()` in their factories and create a remote
  implementation when requested.
- `loadConfig` first applies the legacy `transformer-mode` and Ollama /
  llama.cpp / BitNet flags, then overwrites the corresponding `Config` fields
  (`ollamaBaseUrl`, `llamaCppBaseUrl`, `bitnetBaseUrl`, model names and
  timeouts) with any remote LLM record from the deployment config.  The
  existing `chatWithOllama` / `chatWithLlamaCpp` / `chatWithBitNet` functions
  therefore receive the final, merged configuration.

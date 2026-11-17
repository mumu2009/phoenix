# v10.1 IO Model Inventory (speech / image IO line)

Status: verified 2026-11-17 against the **live** tree `E:\079\v6.0Alixander\phoenix`
(the running gateway/llama-server processes execute from this tree; the twin
`E:\079\v6.0Alixander\runtime_store` + `\data` at the workspace root are stale
August copies and were not used).
Line references are `file:line` inside `phoenix/`.

## 0. Executive summary (the honest version)

| # | Finding | Consequence |
|---|---------|-------------|
| 1 | The **live image IO path** is the additive-JEPA pair `runtime_store/models/additive_jepa/vision_encoder/best.onnx` (128-d concept) + `.../vision_decoder/best.onnx`. It is reached through `VideoLocalOnnxModel` because `config/phoenix.json` pins `model_deployment.vision.localBackend="cpu"`. | This ONNX encoder (not the I-JEPA ViT checkpoints, not the BPU `.bin` files) is what actually runs on this host. |
| 2 | The **live audio IO path** is `runtime_store/models/additive_jepa/speech_encoder/best.onnx` (16 kHz, 16 000 samples → 128-d) + `.../speech_decoder/best.onnx` (128 → 15 872 samples), through `AudioLocalOnnxModel`. | Same: ONNX + one-shot python runner, not the BPU engine. |
| 3 | The **`encoder_head.json` (512→128) artifact is dead code**. It is exported by `.remote_ref/train_video.py` / `.remote_ref/export_multimodal.py` together with `model_encoder_head.onnx`, but **no file in the current source tree reads it** (verified by scanning all `.cpp/.hpp/.inc/.py/.ps1/.sh/.bat` under `phoenix/` excluding `build/`, `outsides/`, `runtime_store/`). It belongs to the `resnet18_224` pipeline. | Training a "head" therefore has to be *new* work on the live 128-d concept space; the old artifact is only useful as a format precedent. |
| 4 | The `resnet18_224` pipeline is **unreachable in the current source**: `resolveOnnxModelPath()` only looks for `best.onnx` / `model.onnx`, but that directory ships `model_encoder.onnx` / `model_decoder.onnx` (`video_model.cpp:89-105`). The files are otherwise intact. | `resnet18_224` is an archive/reference pipeline; do not claim it is live. |
| 5 | The local **video HBDNN/BPU encode is a stub** (`VideoHbdnnModel::encode()` returns `"BPU image encoder preprocessing is not implemented in this build"`, `video_model.cpp:732-743`). The audio HBDNN path **is** implemented via `rdk_x5_bpu::execute()` (`audio_model.cpp:473-527`). | On RDK X5 the audio path can use BPU; the image BPU path would still need preprocessing code. On this host `hbDNN` is unavailable, so both fall back to ONNX/CPU. |
| 6 | The **LLaVA / Qwen2-Audio HTTP encoders** (`multimodal_world_model.{hpp,cpp}`, port 8085) are defined but **never instantiated** by any source file outside their own translation unit, and no process listens on 8085. | Dead code + dead 7B artifacts under `runtime_store/models/multimodal_enc_src/`, `multimodal_dec/`. |
| 7 | `large_vision_encoder/` holds ~20.9 GiB of `.npy` feature caches and ~2.4 GiB of `.pt` tensors from earlier training rounds. | Dead training leftovers, not IO runtime models. |

## 1. Image (video) IO models

### 1.1 Artifact table

| Artifact | Path under `runtime_store/models/` | Format | Input → Output (fp32) | Bytes | Loaded by current source? |
|---|---|---|---|---|---|
| Additive-JEPA vision encoder | `additive_jepa/vision_encoder/best.onnx` | ONNX opset 11 | `pixel_values [1,3,224,224]` → `concept [1,128,1,1]` | 54 612 965 | **YES** – `VideoLocalOnnxModel` (`video_model.cpp:457-503`) via `resolveOnnxModelPath("encoder", "vision_encoder")` |
| Additive-JEPA vision decoder | `additive_jepa/vision_decoder/best.onnx` | ONNX | `concept [1,128,1,1]` → `reconstruction [1,3,224,224]` | 10 444 866 | **YES for `decode()`** – bridge decode path (`autonomy_stack.cpp:4349`); `VideoLocalOnnxModel::decode` at `video_model.cpp:530-549` |
| Additive-JEPA vision encoder (BPU) | `additive_jepa/vision_encoder/best.bin` | Horizon `.bin` (`hb_mapper`, bayes-e) | same 3×224×224 → 128 | 14 358 479 | Only on RDK X5 and only if `detectLocalBackend()` returns Bpu; encode stub prevents actual use (`video_model.cpp:807-832`, `:732-743`) |
| Additive-JEPA vision decoder (BPU) | `additive_jepa/vision_decoder/best.bin` | Horizon `.bin` | 128 → 3×224×224 | 6 939 747 | Same as above |
| ResNet18 encoder | `ijepa/resnet18_224/model_encoder.onnx` | ONNX opset 11 | `pixel_values [1,3,224,224]` → `pooled [1,512,1,1]` | 32 146 521 | **NO** – not matched by `resolveOnnxModelPath` names `{best.onnx, model.onnx}` |
| ResNet18 projection head | `ijepa/resnet18_224/model_encoder_head.onnx` | ONNX (1× Conv2d 1×1) | `pooled [1,512,1,1]` → `concept [1,128,1,1]` | 262 951 | **NO** – no reader in source |
| ResNet18 head weights | `ijepa/resnet18_224/encoder_head.json` | JSON `{inDim:512, outDim:128, W:65536 floats, b:128}` | linear 512→128 | 1 456 019 | **NO** – exported by `extract_head.py` for a BPU-era loader that no longer exists | 
| ResNet18 decoder | `ijepa/resnet18_224/model_decoder.onnx` | ONNX | `concept [1,128,1,1]` → `reconstruction [1,3,224,224]` | 13 381 131 | **NO** |
| ResNet18 decoder (trained variant) | `ijepa/resnet18_224/model_decoder_trained.onnx` | ONNX | same | 13 380 761 | **NO** |
| ResNet18 BPU encoder/decoder | `ijepa/resnet18_224/model_encoder.bin`, `model_decoder.bin` | Horizon `.bin` | 224 → 512 / 128 → 224 | 8 325 447 / 3 606 699 | Only as `jepa.image.horizonModel` override on a BPU box; encode stub again makes this non-functional |
| ResNet18 head export script | `ijepa/resnet18_224/extract_head.py` | Python | reads `model_encoder_head.onnx` → writes `encoder_head.json` | 638 | tool only |

### 1.2 Variant registry and factory selection (code truth)

- Official variant list: `video_model.hpp:143-157`. Note that the first two entries are
  **not** Meta I-JEPA checkpoints: `resnet18_224` (timm ResNet18, 224 px) and
  `video-encoder` (additive-JEPA, `best.onnx` in `additive_jepa/vision_encoder`).
  The Meta ViT-H/G entries (`ijepa_vith14_1k`, `ijepa_vith16_448`, `ijepa_vith14_22k`,
  `ijepa_vitg16_22k`) have **no local weights** in this tree.
- Live config: `config/phoenix.json:773-780` → `jepa.image.variant="video-encoder"`,
  `conceptDim=128`, `backend="auto"`, and `horizonModel/horizonDecoderModel` pointing at
  `runtime_store/models/ijepa/resnet18_224/model_{encoder,decoder}.bin` (a **mismatch** with
  the `video-encoder` variant that would matter on a BPU box; harmless on the host because
  the BPU class never runs).
- Live deployment: `config/phoenix.json:998-1017` → `model_deployment.vision.placement="local"`,
  `localBackend="cpu"`, same for `speech`. Because `localBackend` is not `"auto"`,
  `chooseLocalBackend()` (`video_model.cpp:834-841`) returns Cpu directly and
  `createVideoModel()` (`video_model.cpp:845-899`) constructs `VideoLocalOnnxModel`.
- Weight/ONNX resolution: `video_model.cpp:40-45` (`imageWeightsDir`), `:62-87`
  (`resolveBpuModelPath`: `additive_jepa/<dir>` → `ijepa/<dir>` → `bpu_jepa/<dir>`),
  `:89-105` (`resolveOnnxModelPath`: **only** `best.onnx` or `model.onnx`).
- Factory and bridge wiring:
  - `frontend_server.cpp:4554-4558` creates the image model from `jepa.image.*` (so the
    frontend's image endpoint uses `video-encoder`, 128-d).
  - `external_mixed_modal_io.cpp:105-134` caches `createVideoEncoder("video-encoder", dim)`;
    the variant is hard-wired (`imageVariantFromMetadata()` ignores metadata).
  - `MixedModalConceptBridge::encode()` image branch: `external_mixed_modal_io.cpp:956-974`;
    the bridge's default concept dimension is **64** (`conceptDimension()`,
    `external_mixed_modal_io.cpp:91-93`) and `main_hub_parts/116_section_tail.inc:5095`
    calls `encode(pkt, 64, "")`. In that case `VideoLocalOnnxModel` projects
    128 → 64 with the **deterministic hash projection** `projectToDimension(v, 64, 0x57494D47U)`
    (`video_model.cpp:497-499`; implementation `semantic_unit.cpp:326-340`, sparse path
    `semantic_unit.cpp:290-324`). That hash projection is the only "projection head" the
    live image path has today.
- Model lifecycle: the image model is created lazily by the bridge; `status()` reports
  `backend`, `modelPath`, `conceptDim`, `samples`, `error`.

### 1.3 What the image IO path actually computes

```
bytes -> cv::imdecode -> RGB float 1/255 -> ImageNet mean/std (video_model.cpp:590-650, mean/std :632-638)
      -> python tools/local_onnx_runner.py  (local_onnx.cpp:34-58, 69-160)
         -> additive_jepa/vision_encoder/best.onnx -> 128 floats
      -> if targetDim != 128: projectToDimension(v, targetDim, 0x57494D47)
      -> normalizeVector                                      (semantic_unit.cpp:244-259)
```
The one-shot python runner is a per-call process spawn
(`phoenix::io::runLocalOnnx`, `local_onnx.cpp:69-160`); the interpreter is
`PHOENIX_PYTHON` or `Python314/python(.exe)` (`local_onnx.cpp:34-58`). On this box
`Python314/python.exe` has `numpy 2.5.4` + `onnxruntime 1.31.1` (no Pillow/torch).

## 2. Audio (speech) IO models

### 2.1 Artifact table

| Artifact | Path under `runtime_store/models/` | Format | Input → Output (fp32) | Bytes | Loaded by current source? |
|---|---|---|---|---|---|
| Additive-JEPA speech encoder | `additive_jepa/speech_encoder/best.onnx` | ONNX | `waveform [batch,1,1,16000]` → `concept [batch,128,1,1]` | 4 936 265 | **YES** – `AudioLocalOnnxModel` (`audio_model.cpp:1101-1139`) via `variant "audio-encoder"` → `speech_encoder` (`audio_model.cpp:41-48`) |
| Additive-JEPA speech decoder | `additive_jepa/speech_decoder/best.onnx` | ONNX | `concept [batch,128,1,1]` → `reconstruction [batch,1,1,15872]` | 15 012 860 | **YES for `decode()`** (`audio_model.cpp:1181+`) |
| Additive-JEPA speech encoder (BPU) | `additive_jepa/speech_encoder/best.bin` | Horizon `.bin` | 16 000 → 128 | 1 378 652 | RDK X5 only (`AudioHbdnnModel`, `audio_model.cpp:134-527`) |
| Additive-JEPA speech decoder (BPU) | `additive_jepa/speech_decoder/best.bin` | Horizon `.bin` | 128 → 15 872 | 8 034 486 | RDK X5 only |
| Legacy speech JEPA encoder | `ijepa/speech_16k/model_encoder.onnx` | ONNX | `waveform [1,1,1,16000]` → `concept [1,128]` | 823 873 | Variant `audio-16k` → `speech_16k`; `resolveOnnxModelPath` would need `best.onnx`/`model.onnx`, so **NO** in current source (same naming trap as ResNet18) |
| Legacy speech JEPA decoder | `ijepa/speech_16k/model_decoder.onnx` | ONNX | `concept [1,128,1,1]` → `reconstruction [1,1,1,15872]` | 8 883 222 | **NO** (name not matched) |
| Legacy speech JEPA BPU pair | `ijepa/speech_16k/model_encoder.bin`, `model_decoder.bin` | Horizon `.bin` | 16 000 → 128 / 128 → 15 872 | 527 195 / 8 651 378 | RDK X5 only, via `jepa.speech.horizonModel` (`config/phoenix.json:785-790`); `AudioHbdnnModel` is implemented and would run it there |
| Manifest | `ijepa/speech_16k/model.manifest.json` | JSON | documents `waveform [1,1,1,16000]`, `concept [1,128]`, rate 16 kHz, 15 872 decoder samples | 1 825 | read only for metadata, by tools |

There is **no separate audio projection head artifact** in any variant: the audio
encoder output is used directly as the 128-d concept (then hash-projected if
`targetDim != 128`, `audio_model.cpp:1133-1135`).

### 2.2 Constants, registry and factory

- `kFixedInputSamples=16000`, `kEncoderOutputDim=128`, `kDecoderOutputSamples=15872`
  (`audio_model.cpp:37-39`). 1-second, 16 kHz, mono; C++ pads/truncates to 16 000
  (`audio_model.cpp:452-471`), decoder output is zero-padded to 16 000 by callers.
- Variants: `audio_model.hpp:139-146` (`audio-encoder`, `audio-16k`, `audio-22k`,
  `audio-44k`, `audio-48k`; only the first two have any local files).
- Sampling/window metadata (`windowSamples=400`, `strideSamples=160`, i.e. 25 ms / 10 ms)
  is declared in `audio_model.hpp:26-28` and used only by `encodeContext` masking.
- Factory: `createAudioModel()` (`audio_model.cpp:1460-1524`); live deployment
  `localBackend="cpu"` → `AudioLocalOnnxModel`; ONNX name resolution
  `audio_model.cpp:91-107`; BPU root resolution `audio_model.cpp:65-89`.
- BPU runner: `AudioHbdnnModel::runEncoderBpu` (`audio_model.cpp:473-527`) calls
  `rdk_x5_bpu::execute()` with `bpuInputFloatsPath`; `rdk_x5_bpu.cpp` implements a
  single-input/single-file hbDNN runner (`rdk_x5_bpu.cpp:207-214` availability,
  `:375-523` execute) and is also reachable from the `edge_platform` addon
  (`edge_platform.cpp:2691-2692, 2946-2947`).

### 2.3 What the audio IO path actually computes

```
WAV bytes -> preprocessAudio (resample 16k mono, audio_model.cpp:413-445)
          -> pad/truncate 16 000 samples
          -> tools/local_onnx_runner.py -> additive_jepa/speech_encoder/best.onnx
          -> 128 floats -> projectToDimension if targetDim != 128
          -> normalizeVector
```

## 3. Multimodal (LLaVA / Qwen2-Audio) path

| Item | Path / symbol | Status |
|---|---|---|
| Image encoder client (LLaVA-1.5-7B, 4096-d unit queries) | `MultimodalImageWorldModel`, `multimodal_world_model.hpp:76-99`, impl `.cpp:309-430` | **Dead / not wired**: no construction site outside `multimodal_world_model.cpp`; base URL `http://127.0.0.1:8085` (`multimodal_world_model.hpp:62-67`), no listener on 8085 |
| Audio encoder client (Qwen2-Audio-7B, 4096-d) | `MultimodalAudioWorldModel`, `multimodal_world_model.hpp:104-125`, impl `.cpp:432-560` | **Dead / not wired** (same) |
| LLaVA weights | `multimodal_enc_src/llava-1.5-7b-hf/` (config/tokenizer + `model-00001-of-00003.safetensors`) | Partial 7B download, unused |
| Qwen2-Audio weights | `multimodal_enc_src/Qwen2-Audio-7B-Instruct/` (`model-00001-of-00005.safetensors`) | Partial 7B download, unused |
| Decoders | `multimodal_dec/image_decoder.onnx` (51.8 MB), `llava_unit_tokenizer.onnx` (33.6 MB) + `.pt` siblings | Unused by source; service-only artifacts |
| Service log | `runtime_store/logs/multimodal_encdec_server.log` (151 B, last written 2026-08-11) | stale |

The bridge (`MixedModalConceptBridge`) does **not** call these classes; its image/audio
branches call the additive-JEPA `VideoModel`/`AudioModel` interfaces instead
(`external_mixed_modal_io.cpp:956-985`). The doc comment in `external_mixed_modal_io.cpp:915-918`
still says "external LLaVA-based image world model" – stale comment, code is additive-JEPA.

## 4. Other model artifacts in the tree (not IO runtime)

| Path | Contents | Verdict |
|---|---|---|
| `runtime_store/models/large_vision_encoder/` | `llava_features.npy`, `llava_images.npy`, `vjepa2_1_vitb_features.pt`, `qwen_audio_forward.json`, … (~20.9 GiB `.npy`, ~2.4 GiB `.pt`) | Dead training/experiment leftovers |
| `runtime_store/models/intent_carrier/`, `meme_pair/` | report JSON + `resident_check.log` only | Other subsystems (intent/meme), no weights |
| `runtime_store/models/granite-3.1-1b-a400m-instruct/`, `ext` under `GGUF_models/` | LLM weights | Text runtime, out of scope |
| `runtime_store/models/*/calibration_encoder/**` (under `ijepa/<variant>/`) | 100 raw NCHW calibration `.bin` per encoder/decoder | hb_mapper calibration inputs; unused on host |
| `runtime_store/models/bpu_jepa/resnet18_224/` | A second copy of the ResNet18 ONNX/BIN + `model.manifest.json` (top-level `E:\079\v6.0Alixander\runtime_store` only) | Stale August twin; `resolveBpuModelPath` would search `runtime_store/models/bpu_jepa/...` **inside the live tree**, where it does not exist |

## 5. Local corpora available for IO training (read-only)

| Corpus | Path | Size / shape | Labels | Notes |
|---|---|---|---|---|
| Tiny-ImageNet-200 | `runtime_store/datasets/tiny-imagenet-200/` | 100 000 train / 10 000 val / 10 000 test, 64×64 JPEG, 200 classes | yes, class = parent dir; `val/val_annotations.txt` present | `SOURCE.json` documents HF-mirror provenance; non-commercial |
| MUSAN 16 kHz | `runtime_store/datasets/musan_extracted/musan/` | 430 `speech/`, 672 `music/`, 934 `noise/` 16 kHz mono WAVs; tarball `musan_16k.tar.gz` 6.87 GB | coarse 3-class by directory | `musan_16k/manifest.json` (1 665 entries) mirrors the tarball; `musan_test/` empty |
| `runtime_store/datasets/hf_cache`, `hf_home`, `torch_home`, `wikitext2`, `specialist_bank`, `study_expert`, `self_mind` | misc caches / text / other subsystems | n/a | not IO training corpora |
| `phoenix/data/`, `phoenix/uploads/tmp/` | empty / transient upload cache | n/a | not corpora |
| `E:\079\v6.0Alixander\data\{audio,images}` (workspace root) | 5 WAV + `captions.json`, 5 PNG | toy samples | **outside the live tree**, stale August copy, not used |
| `runtime_store/large_vision_encoder/data/{beans,librispeech_dummy}` | tiny dataset leftovers | n/a | not used |

## 6. How to re-verify this inventory

```powershell
cd E:\079\v6.0Alixander\phoenix
# 1) ONNX I/O shapes + timings of every IO artifact (numpy+onnxruntime only)
Python314\python.exe -c @"
import onnxruntime as ort
for p in ['runtime_store/models/additive_jepa/vision_encoder/best.onnx',
          'runtime_store/models/additive_jepa/vision_decoder/best.onnx',
          'runtime_store/models/additive_jepa/speech_encoder/best.onnx',
          'runtime_store/models/additive_jepa/speech_decoder/best.onnx',
          'runtime_store/models/ijepa/resnet18_224/model_encoder.onnx',
          'runtime_store/models/ijepa/resnet18_224/model_decoder.onnx']:
    s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    print(p, [(i.name, i.shape) for i in s.get_inputs()],
              [(o.name, o.shape) for o in s.get_outputs()])
"@
# 2) who reads encoder_head.json? (expect: no C++ hits under phoenix root)
Get-ChildItem -Recurse -Include *.cpp,*.hpp,*.inc -File |
  Where-Object FullName -notmatch '\\build\\|\\outsides\\|\\runtime_store\\' |
  Select-String -Pattern 'encoder_head'
# 3) live process evidence
Get-Process phoenix_main,llama-server | Select-Object Id,Path
# 4) runtime gates
Select-String config\phoenix.json -Pattern 'jepa|localBackend|placement'
```
Verified 2026-11-17 ~22:00 +08:00: 3 llama-server, 9 `phoenix_main` processes all
executing from `E:\079\v6.0Alixander\phoenix`; `runtime_store/logs/runtime_debug.log`
is being written live in that tree.

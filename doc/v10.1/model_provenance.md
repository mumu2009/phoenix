# Model Provenance — what our video/audio models actually are

Status: verified **2026-11-17 ~23:00–00:00 +08:00** against the live tree
`E:\079\v6.0Alixander\phoenix` (`git HEAD = 4f4bbd7`, no git operations performed).
Every claim below has an artifact + a verifiable probe; the probes are listed at the
end so a future session can re-run them.

> **User correction this document exists to enforce:** our shipped video/audio models
> are **LLaVA-based / ResNet-modified in-house models**, **not** the Meta I-JEPA
> series. The directory names `additive_jepa/` and `ijepa/` are **legacy labels kept
> for path compatibility**. "I-JEPA" must not be used as a description of what our
> own checkpoints are.

---

## 1. Correct vocabulary (stop saying X → say Y)

| Current (wrong/misleading) | Use instead | Why |
|---|---|---|
| "I-JEPA vision model" / "JEPA speech model" | **in-house vision concept autoencoder** / **in-house speech concept autoencoder** | The shipped `additive_jepa/{vision,speech}_*/best.onnx` are additive-residual concept autoencoders trained in-house; no Meta I-JEPA weights are involved. |
| "`additive_jepa/` is our JEPA model" | "`additive_jepa/` is a **legacy directory name**; the runtime image/speech concept autoencoders live there" | Directory name predates the architectural rename; C++ resolver + config hardcode it. |
| "`ijepa/resnet18_224`" (as a model family) | **ResNet18-modified in-house autoencoder (512-d encoder + 512→128 head + trained decoder)** | It is a timm ResNet18, not I-JEPA. |
| "LLaVA encoder" / "Qwen2-Audio encoder" (as live) | **LLaVA-feature in-house unit tokenizer + image decoder** (trained, never wired); "Qwen2-Audio path (abandoned)" | The only complete LLaVA artifacts are the two ONNX files in `multimodal_dec/`; the raw 7B models are incomplete downloads. |
| "JEPA ViT-H / ViT-g weights" | "historical variant ids with **no local weights**" | `video_model.hpp` registers `ijepa_vith14_1k/vith16_448/vith14_22k/vitg16_22k`, but no such directories exist in this tree. |
| `encoder_head.json` "read by C++" | "dead BPU-era export, no reader" | 0 C++/config hits (see `dead_assets_audit.md` C3). |

Use these names in future docs/commit messages:

- **Image live path**: "in-house vision concept autoencoder (128-d), legacy dir
  `additive_jepa/vision_encoder`"
- **Speech live path**: "in-house speech concept autoencoder (1-D, 16 kHz, 128-d),
  legacy dir `additive_jepa/speech_encoder`"
- **ResNet18 archive**: "ResNet18-modified in-house autoencoder (512-d → 128-d)"
- **LLaVA artefacts**: "LLaVA-1.5-7B feature cache" and "LLaVA-feature in-house
  unit tokenizer / image decoder"
- **Meta I-JEPA**: only ever as "historical/registry-only variant ids (no weights)".

---

## 2. What is actually on disk (verified I/O, not names)

### 2.1 Tensor/graph interfaces (onnxruntime 1.31.1, Python314)

| Artifact | Producer | Input | Output | What it is |
|---|---|---|---|---|
| `additive_jepa/vision_encoder/best.onnx` (54,612,965 B, sha256 `18E31191…`) | pytorch | `pixel_values [1,3,224,224]` | `concept [1,128,1,1]` | live image concept encoder |
| `additive_jepa/vision_decoder/best.onnx` (10,444,866 B) | pytorch | `concept [1,128,1,1]` | `reconstruction [1,3,224,224]` | live image decoder |
| `additive_jepa/speech_encoder/best.onnx` (4,936,265 B) | pytorch | `waveform [batch,1,1,16000]` | `concept [batch,128,1,1]` | live 16 kHz speech encoder |
| `additive_jepa/speech_decoder/best.onnx` (15,012,860 B) | pytorch | `concept [batch,128,1,1]` | `reconstruction [batch,1,1,15872]` | live speech decoder (15 872 samples, C++ pads to 16 000) |
| `ijepa/resnet18_224/model_encoder.onnx` (32,146,521 B) | pytorch | `pixel_values [1,3,224,224]` | `pooled [1,512,1,1]` | ResNet18-modified encoder |
| `ijepa/resnet18_224/model_encoder_head.onnx` (262,951 B) | pytorch | `pooled [1,512,1,1]` | `concept [1,128,1,1]` | 512→128 projection (1×1 conv) |
| `ijepa/resnet18_224/model_decoder_trained.onnx` (13,380,761 B) | pytorch | `concept [1,128,1,1]` | `reconstruction [1,3,224,224]` | ResNet18 pipeline decoder |
| `ijepa/speech_16k/model_encoder.onnx` (823,873 B) | pytorch | `waveform [1,1,1,16000]` | `concept [1,128]` | legacy 1-D speech encoder |
| `ijepa/speech_16k/model_decoder.onnx` (8,883,222 B) | pytorch | `concept [1,128,1,1]` | `reconstruction [1,1,1,15872]` | legacy 1-D speech decoder |
| `multimodal_dec/llava_unit_tokenizer.onnx` (33,625,788 B) | pytorch | `image_features [batch,576,4096]` | `unit_query [batch,4096]` | **LLaVA-feature** tokenizer (in-house) |
| `multimodal_dec/image_decoder.onnx` (51,833,682 B) | pytorch | `unit_query [batch,4096]` | `output [batch,3,224,224]` | **LLaVA-feature** image decoder (in-house) |
| `runtime_store/models/large_vision_encoder/llava_features.npy` | numpy | — | `(512, 576, 4096)` float16 | LLaVA vision-tower features, 512 images |
| `…/llava_features_more.npy` | numpy | — | `(4000, 576, 4096)` float16 | same, 4 000 images |
| `…/llava_images.npy` | numpy | — | `(512, 3, 224, 224)` uint8 | paired images |
| `…/llava_images_more.npy` | numpy | — | `(4000, 3, 224, 224)` uint8 | paired images |
| `…/llava_unit_feature_cache.pt` | torch | — | dict: `features_fp16`, `images_uint8_224`, `paths` | self-contained cache of the 512-image set |
| `…/image_decoder.pt` | torch | — | `state_dict` (`fc.*`, `blocks.*`) | trained LLaVA-image decoder weights |
| `…/llava_forward_sample.pt` | torch | — | dict: `model=llava-hf/llava-1.5-7b-hf`, `components=vision_tower+multi_modal_projector`, `image_features`, `unit_query`, `source_image`, path | one-image forward sample; provenance proof |
| `…/vjepa2_1_vitb_features.pt` | torch | — | single `features` tensor + `input_shape` | **V-JEPA2 ViT-B** probe output (7 MB), not a model |

### 2.2 Manifests

- `runtime_store/models/additive_jepa/vision_encoder/model.manifest.json`:
  `name="additive_jpea_vision_encoder"` (sic, typo in the artifact itself),
  `modality=image`, `concept_dim=128`, `input_name=pixel_values`,
  `input_shape=[1,3,224,224]`, `output_name=concept`, `n_blocks=1`,
  `source_checkpoint=null`, `resolution=224`.
- `…/speech_encoder/model.manifest.json`: `name="audio-16k-encoder"`,
  `chunk_size=16000`, `decoder_output_samples=15872`, `input_name=waveform`.
- `…/speech_decoder/model.manifest.json`: `name="audio-16k-decoder"`,
  `concept_dim=128`.
- `runtime_store/models/ijepa/resnet18_224/model.manifest.json`:
  `modelName="bpu_jepa_resnet18_224"` (legacy name),
  `modelType="bpu_jepa_autoencoder"`, `source={"encoder":"resnet18","pretrained":true}`,
  `onnx.encoder.path="model_encoder.onnx"`, `onnx.decoder.path="model_decoder_trained.onnx"`,
  notes: *"Encoder is a pretrained timm ResNet18 with a BPU-friendly Flatten+Linear
  head. Decoder has been trained as the inverse of the encoder on 100 calibration
  (concept, image) pairs."*
- `runtime_store/models/ijepa/speech_16k/model.manifest.json`:
  `source.encoder="jepa_1d_conv2d"`, `pretrained=false`, notes: *"1D JEPA-v2 speech
  autoencoder with 1D-as-2D Conv2d/ConvTranspose2d for BPU compatibility."* — this is
  a **legacy naming artefact**; the model is a custom 1-D conv autoencoder, not Meta
  I-JEPA.

### 2.3 The `.pt` metadata that pins the LLaVA cache provenance

Zip-pickle string scan of `runtime_store/models/large_vision_encoder/*.pt` (no
unpickling required) surfaced:

```
llava_forward_sample : model='llava-hf/llava-1.5-7b-hf',
                       components='vision_tower+multi_modal_projector',
                       image_features, unit_query, source_image,
                       source path: runtime_store/datasets/tiny-imagenet-200/train/...
llava_unit_feature_cache : components='frozen vision_tower + frozen multi_modal_projector',
                       model_is='LLaVA', feature_shape, features_fp16,
                       images_uint8_224, paths
image_decoder         : state_dict: fc.weight, fc.bias, blocks.{0,1,...}.weight/bias,
                       blocks.*.running_mean/var
```

This is the hard evidence that the 22.85 GiB are **LLaVA-1.5-7B feature caches over
Tiny-ImageNet-200** (not downloaded ViT-g/ViT-H weights). The 576 tokens × 4096 d
matches LLaVA-1.5-7B's vision tower + projector output exactly (input 224×224 →
14×14 patches + CLS = 197? no — 576 = 24×24 patches from the 336-px LLaVA preprocessing
grid flattened; the artefact uses 576 tokens, which is what the tokenizer's
`image_features [batch,576,4096]` input contract requires).

---

## 3. Provenance chains (creator scripts)

### 3.1 Live image/speech concept autoencoders (`additive_jepa/`, legacy name)

Chain:

1. **Architecture/training**: `tools/additive_jepa.py` (model classes: additive residual
   blocks on a ResNet18 base, `additive_jepa_<model_name>` manifests, concept 128-d;
   `ResNet18Base` used by the evolver at `tools/bpu_evolve_additive.py:918`).
2. **Training driver**: `tools/run_all_additive_training.py` +
   `run_all_additive_training.bat` (and the BPU evolution loop
   `tools/bpu_evolve_additive.py`; speech variants in
   `tools/bpu_evolve_speech_decoder{,_medium}.py`).
3. **Export**: `tools/export_runtime_models.py` — *"exports the full encoder/decoder as
   best.onnx for C++ local ONNX"*; `tools/export_additive_jepa.py`; manifests written
   by the same exporter (`name="additive_jpea_..."`).
4. **BPU compile**: `tools/compile_bpu.sh` / `compile_bpu_speech_small.sh` →
   `best.bin` per dir.
5. **Runtime**: `video_model.cpp` / `audio_model.cpp` — `resolveOnnxModelPath`
   (`best.onnx`/`model.onnx`) and `resolveBpuModelPath` (`best.bin`), selected by
   `model_deployment.{vision,speech}.localBackend` = `cpu` in
   `config/phoenix.json:1005–1022`.

Conclusion: in-house concept autoencoders; **not** LLaVA-feature models (they don't
consume 4096-d features) and **not** Meta I-JEPA.

### 3.2 ResNet18-modified in-house autoencoder (`ijepa/resnet18_224/`)

Chain: `.remote_ref/train_video.py` (writes `model_encoder.onnx`,
`model_encoder_head.onnx`, `encoder_head.json`; lines 258/284) →
`.remote_ref/export_multimodal.py` / `.remote_ref/export_runtime_models.py`
(also `tools/export_multimodal.py` / `tools/export_runtime_models.py`) →
`tools/compile_bpu.sh` (hb_mapper → `model_encoder.bin`, `model_decoder.bin`) →
`tools/x5_bpu_real_verify.py` (X5 verification). Runtime is unreachable by name
today (`resolveOnnxModelPath` wants `best.onnx`/`model.onnx`; dir has
`model_encoder.onnx`/`model_decoder.onnx`). See `dead_assets_audit.md` C1–C9.

### 3.3 LLaVA-feature in-house unit tokenizer + image decoder (`multimodal_dec/` + `large_vision_encoder/`)

Chain (2026-10-16 → 10-18):

1. `tools/prepare_multimodal_encoders.py` — loads the LLaVA-1.5 vision tower and
   Qwen2-Audio encoder and exports adapters to `runtime_store/models/multimodal_enc/`
   (**that directory was never produced** — the pipeline was abandoned mid-way).
2. `tools/train_unit_query_align.py` — "cache both sides to disk (so the heavy servers
   can be stopped)", align `unit_query`; writes `aligner_*.pt`/`.onnx` under
   `runtime_store/models/multimodal_align/` (also never produced).
3. `tools/train_multimodal_decoders.py` — trains `{modality}_decoder.pt`
   (`:511`) and `caption_head.pt` (`:523`).
4. `tools/multimodal_encdec_server.py` — a :8085 service that would serve
   `llava_vision_projector.onnx`, `qwen2_audio_connector.onnx`,
   `multimodal_dec/image_decoder.onnx`. **Never started**: no listener, and its
   encoder inputs (`multimodal_enc/*`) are absent.
5. The C++ client for that service, `multimodal_world_model.{hpp,cpp}`
   (`MultimodalImageWorldModel` / `MultimodalAudioWorldModel`, base URL
   `http://127.0.0.1:8085`), is **never instantiated** anywhere.

Conclusion: this is the **LLaVA-based in-house integration** the user refers to. It
produced two complete, loadable ONNX artefacts (`llava_unit_tokenizer.onnx`,
`image_decoder.onnx`) plus 22.85 GiB of feature caches, but it was **never wired into
the runtime** and 8.31 GiB of its raw 7B downloads are **incomplete**
(`multimodal_enc_src/`: LLaVA 1/3 shards, Qwen2-Audio 1/5 shards).

### 3.4 Legacy 1-D speech pair (`ijepa/speech_16k/`)

Chain: `.remote_ref/train_audio.py` / `tools/train_audio.py` →
`tools/export_audio_onnx.py` / `tools/export_bpu_speech_{small,medium}.py` →
`tools/compile_bpu_speech_small.sh` → `model_encoder.bin`/`model_decoder.bin`.
Only the `.bin` pair is still config-referenced (`config/phoenix.json:788–789`) for
the X5 BPU path; the ONNX files are unreachable by the runtime resolver. The manifest
is honest: it is a custom 1-D conv autoencoder, **not** Meta I-JEPA.
`tools/inspect_speech_onnx.py` still points at it (tool-only).

---

## 4. What is genuinely "jepa" in this tree — and what is not

| Item | Genuine Meta I-JEPA / V-JEPA? | Evidence |
|---|---|---|
| `additive_jepa/*` | **No** — legacy directory name for in-house concept autoencoders | manifests (`additive_jpea_*`), training tools, 128-d concept I/O |
| `ijepa/resnet18_224/*` | **No** — ResNet18-modified in-house autoencoder | manifest `source.encoder="resnet18"`, `pretrained=true`, ONNX `pooled [1,512,1,1]` |
| `ijepa/speech_16k/*` | **No** — custom 1-D conv autoencoder | manifest `source.encoder="jepa_1d_conv2d"`, notes say "1D JEPA-v2" only as a historical label |
| `ijepa_vith14_1k`, `ijepa_vith16_448`, `ijepa_vith14_22k`, `ijepa_vitg16_22k` variant ids (`video_model.hpp:143–157`) | historical ids only — **no local weights** | directory listing of `runtime_store/models/ijepa/` contains only `resnet18_224/` + `speech_16k/`; `ijepa/README.md` claims four ~11 GB HuggingFace dirs but they do not exist here |
| `vjepa2_1_vitb_features.pt` (7 MB) | V-JEPA2 **ViT-B feature probe output** only | zip pickle: single `features` tensor + `input_shape`; no model weights |
| `multimodal_enc_src/llava-1.5-7b-hf`, `Qwen2-Audio-7B-Instruct` | **No** — LLaVA / Qwen, both **incomplete downloads** | index files declare 3 and 5 shards; only 1 shard each present |
| `multimodal_dec/*` | **No** — LLaVA-feature in-house tokenizer/decoder | I/O: `image_features [batch,576,4096]` → `unit_query [batch,4096]` → image |

No Meta I-JEPA `.safetensors`/weights and no ViT-H/ViT-g checkpoint exists in the live
tree. The only "ViT-g/ViT-H-like" names are the registry strings and the stale README.

---

## 5. Config keys that still say "jepa" (do not rename now)

`config/phoenix.json:773–790` (same in `config/phoenix.example.json`):

```json
"jepa": {
  "image": {
    "variant": "video-encoder",
    "conceptDim": 128,
    "backend": "auto",
    "horizonModel": "runtime_store/models/ijepa/resnet18_224/model_encoder.bin",
    "horizonDecoderModel": "runtime_store/models/ijepa/resnet18_224/model_decoder.bin"
  },
  "speech": {
    "horizonModel": "runtime_store/models/ijepa/speech_16k/model_encoder.bin",
    "horizonDecoderModel": "runtime_store/models/ijepa/speech_16k/model_decoder.bin"
  }
}
```

These keys are read as `jepa.image.*` / `jepa.speech.*` by `video_model.cpp` /
`audio_model.cpp` (`resolveConfig<>("jepa.*")`). They are **legacy API names**, not a
claim about the model family. Renaming them is a code+config migration and is out of
scope for this read-only audit; when it happens, migrate C++ + both JSON configs +
docs in one change, and keep `additive_jepa/` path compatibility until the resolver
is updated too. Note the latent mismatch: `variant="video-encoder"` (128-d) while
`horizonModel` points at the 512-d ResNet18 `.bin` — harmless on this host
(`localBackend=cpu`, and the BPU image encode path is a stub) but should be fixed if
the X5 image path is ever revived.

---

## 6. Corrections this audit makes to existing docs

1. `AGENTS.md` → "Video / Audio BPU Pipeline" says *"`video_model.cpp` loads
   `encoder_head.json` from the model directory"*. **False today**: 0 readers
   (`dead_assets_audit.md` §1.2/C3). The current BPU-era head file is dead; the
   ONNX head (`model_encoder_head.onnx`) is the complete artifact.
2. `ijepa/README.md` describes four Meta I-JEPA weight dirs totalling ~11 GB that are
   **not present** in this tree; it also recommends `tools/download_ijepa_models.py`,
   which would download models **we do not use**. Treat that README as historical;
   the live variants are the in-house ones in `doc/v10.1/io_model_inventory.md`.
3. Any doc saying "our JEPA vision model" / "JEPA speech model" must be rewritten to
   the vocabulary in §1 (this includes older `doc/v7.0/model_deployment.md:404`
   "frozen BPU encoder (`model_encoder.onnx`) using a VICReg-style loss" — accurate as
   a loss description, but the model is the ResNet18 in-house autoencoder).
4. `doc/remote_training_package.md:125–128,243` describes `encoder_head.json` as
   "read by C++ at runtime" — stale; see (1).

---

## 7. Re-verification recipe (read-only)

```powershell
cd E:\079\v6.0Alixander\phoenix
# 1) ONNX I/O + producer for every IO artifact
Python314\python.exe -c @"
import onnxruntime as ort
for p in ['runtime_store/models/additive_jepa/vision_encoder/best.onnx',
          'runtime_store/models/additive_jepa/speech_decoder/best.onnx',
          'runtime_store/models/ijepa/resnet18_224/model_encoder.onnx',
          'runtime_store/models/multimodal_dec/llava_unit_tokenizer.onnx',
          'runtime_store/models/multimodal_dec/image_decoder.onnx']:
    s=ort.InferenceSession(p,providers=['CPUExecutionProvider'])
    print(p, s.get_modelmeta().producer_name,
          [(i.name,list(i.shape)) for i in s.get_inputs()],
          [(o.name,list(o.shape)) for o in s.get_outputs()])
"@
# 2) .npy shapes without loading
Python314\python.exe -c @"
import numpy as np, glob
for f in glob.glob('runtime_store/models/large_vision_encoder/*.npy'):
    a=np.load(f,mmap_mode='r'); print(f,a.shape,a.dtype)
"@
# 3) .pt provenance strings (zip scan, no torch)
Python314\python.exe -c @"
import zipfile,re,glob
for f in glob.glob('runtime_store/models/large_vision_encoder/*.pt'):
    z=zipfile.ZipFile(f); d=z.read([n for n in z.namelist() if n.endswith('data.pkl')][0])
    print(f,[s.decode() for s in re.findall(rb'[ -~]{5,60}',d)][:12])
"@
# 4) confirm no C++ reader for encoder_head
Get-ChildItem -Recurse -Include *.cpp,*.hpp,*.inc -File |
  Where-Object FullName -notmatch '\\build\\|\\outsides\\|\\runtime_store\\' |
  Select-String -Pattern 'encoder_head'   # expect: no output
```

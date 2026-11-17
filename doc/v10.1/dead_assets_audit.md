# Dead Assets & Dead Code Audit — live tree

Status: evidence-backed audit, **2026-11-17 ~23:00–00:30 +08:00** (scans < 10 min each).
Scope: `E:\079\v6.0Alixander\phoenix` (the running tree: gateway pid 23920 on :5097,
7 llama-server.exe, all executing from this tree). The workspace-root copies
(`E:\079\v6.0Alixander\runtime_store`, `\data`, `\build`, `\config`, `\doc`,
`\phoenix_milestone_judge`, `_phoenix_weight_hold`) are stale August snapshots and
were **not** used. `git HEAD = 4f4bbd7` (read only — no git branch/commit/push was
run anywhere).

**Nothing in this document has been deleted or moved.** Every row is a proposal.
Deletion only happens after Lead approval and only for the rows the Lead names.

Verdict legend:

| Verdict | Meaning |
|---|---|
| **DELETE-SAFE** | No reader in code or config; the functionality it would have provided is either absent by design, superseded by a live artifact, or regenerable. Safe to delete after Lead approval. |
| **REUSE** | Loadable/usable data or weights with a concrete downstream use; do not delete — see §5. |
| **KEEP** | Live runtime dependency, provenance evidence, or cheap archive needed to rebuild a kept artifact. |
| **UNCLEAR-needs-user** | Ownership/intent unknown; needs a human decision before any action. |

---

## 1. How the "not referenced" claims were proved (searches run)

A plain `rg`/grep over `phoenix/` silently skips `tools/` and `runtime_store/`
because both are in `.gitignore` (`.gitignore:4` and `:52`) — so the scan below was
run with explicit directory enumeration, not gitignore-aware ripgrep.

1. **Code/config reference scan.** 1,709 files under `phoenix/` with extensions
   `.cpp .hpp .inc .py .bat .sh .ps1 .json`, **excluding**
   `outsides/`, `build/`, `.git/`, `runtime_store/` (data — scanned separately),
   `Python314/`, `.venv/`, `node_modules/`, `__pycache__/`, `papers/`, `dist/`,
   `catastrophe/`, `robots/`, `tests/`, `.idea/`, `.vscode/`, `uploads/`,
   `checkpoints/`, `gguf/`, `GGUF_models/`, `snapshots/`, `lmdb/`, `auth/`,
   `brain_maps/`, `structured_exports/`, `tmp/`.
   Patterns (SimpleMatch): `large_vision_encoder`, `llava_features`, `llava_images`,
   `llava_unit_feature_cache`, `vjepa2_1_vitb`, `image_decoder`, `encoder_head`,
   `model_encoder_head`, `multimodal_enc_src`, `multimodal_dec`, `qwen_audio`,
   `Qwen2-Audio`, `multimodal_world_model`, `MultimodalImageWorldModel`,
   `MultimodalAudioWorldModel`.
   Result: 0 hits for the `large_vision_encoder` cache names and 0 hits for
   `multimodal_enc_src`; hits for `image_decoder` / `multimodal_dec` only inside
   `tools/multimodal_encdec_server.py` + `tools/run_multimodal_encdec_pipeline.py`
   (a service that is not running); `encoder_head` only in export scripts that
   *write* it and `tools/compile_bpu.sh` that *copies* it.
2. **Second scan including `tools/`, `.remote_ref/`, `scripts/`, `config/`.**
   1,275 files (`*.py *.sh *.bat *.ps1 *.json` + root `*.cpp/*.hpp/*.inc` +
   `main_hub_parts/` + `addons/`) for `model_encoder.onnx`, `model_decoder.onnx`,
   `additive_jepa`, `speech_16k`, `audio_decoder.onnx`, `multimodal_enc`,
   `best.onnx`, `best.bin`. This is how the export/compile tools were identified.
3. **Loader-pattern scan (C++).** `resolveOnnxModelPath`, `resolveBpuModelPath`,
   `best.onnx`, `model.onnx`, `localBackend`, `placement` across `*.cpp` (35 hits);
   functions read in full: `video_model.cpp:40–105` (`imageWeightsDir`,
   `additiveDirFor`, `resolveBpuModelPath`, `resolveOnnxModelPath`),
   `video_model.cpp:455–550`, `audio_model.cpp:65–157`, `audio_model.cpp:1100–1200`,
   `audio_model.cpp:1420–1524`, `model_deployment.cpp:140–270`.
4. **Artifact inspection.** `Get-FileHash -Algorithm SHA256` for every candidate
   file (table), `.npy` headers parsed with Python314/numpy 2.5.4
   (`np` header only, no full load), ONNX graph I/O + producer read with
   onnxruntime 1.31.1 in Python314 (no torch), `.pt` containers inspected as zip
   archives (torch pickle string scan) without unpickling.
5. **Runtime state.** `Win32_Process` + `Get-NetTCPConnection` snapshot;
   `runtime_store/autonomy_state.json` parsed as JSON
   (`MissionState` enum read from `mission_lifecycle.hpp:42`); gateway
   `runtime_store/long_run/gateway.log` tail; `/api/mission/status` queried
   directly returned **401 (auth)**, so the authoritative-count read was
   taken from `tools/stability_run.ps1:409–426` + the Lead's stability run log.
6. **Size accounting.** Recursive `Measure-Object Length` per directory; exact byte
   totals are in the table and in `build/tmp/dead_assets.json`.

---

## 2. Master table

`size` is the file size or recursive directory size on 2026-11-17.
`sha256 (sample)` is the full-file SHA-256 of the named representative file
(for a directory row the sample file is shown).

### 2.1 `large_vision_encoder/` — 22.85 GiB of LLaVA feature caches (REUSE, not ViT-g/H)

**Key provenance finding: these are NOT downloaded ViT-g / ViT-H checkpoints.
They are LLaVA-1.5-7B feature caches extracted over the local Tiny-ImageNet-200
corpus, plus the in-house tokenizer/decoder training artifacts built on them.**
Evidence: `.npy` shapes (§2.1 table), `.pt` pickle strings
(`llava-hf/llava-1.5-7b-hf`, `components=vision_tower+multi_modal_projector`,
`feature_shape`, `unit_query`, source paths
`runtime_store/datasets/tiny-imagenet-200/train/...`), and the companion ONNX pair
in `multimodal_dec/` whose I/O is `image_features [batch,576,4096] →
unit_query [batch,4096] → output [batch,3,224,224]`.

| ID | Path (under `phoenix/`) | Size | sha256 (sample) | Created by | Referenced by code/config? | Verdict |
|---|---|---|---|---|---|---|
| A1 | `runtime_store/models/large_vision_encoder/` (dir) | 24,531,414,006 B (22.85 GiB) | see A1a | No in-tree script; timestamps 2026-10-16 21:29 → 10-17 13:35, same session as the LLaVA/Qwen multimodal pipeline (B1); `.pt` metadata names `llava-hf/llava-1.5-7b-hf` frozen `vision_tower+multi_modal_projector` and Tiny-ImageNet-200 paths | **0 hits** in 1,709 code/config files; also 0 hits in `tools/`, `.remote_ref/`, `scripts/` | **REUSE** (see §5 R1) |
| A1a | `…/llava_features.npy` | 2,415,919,232 B | `572DDE5F…FC03CFB8` | LLaVA vision-tower extraction, 512 images | none | **REUSE** — `(512, 576, 4096)` fp16; train/val set for the unit tokenizer |
| A1b | `…/llava_images.npy` | 77,070,464 B | `0440CAF2…C1BDDD2A` | same run | none | **REUSE** — `(512,3,224,224)` uint8; pixel ground truth paired with A1a |
| A1c | `…/llava_features_more.npy` | 18,874,368,128 B | `446E1909…A3C05C27` | same run, 4,000-image scale-up | none | **REUSE** — `(4000,576,4096)` fp16; the large training pool. If disk pressure wins, this is the one row to drop (19.4 GB incl. A1d), but it is **not** cheaply regenerable: the LLaVA HF download on disk is incomplete (see B1). |
| A1d | `…/llava_images_more.npy` | 602,112,128 B | `53568D2E…7EA77EC0D` | same run | none | **REUSE** — `(4000,3,224,224)` uint8; paired with A1c |
| A1e | `…/llava_unit_feature_cache.pt` | 2,493,052,589 B | `71DC9042…7199CF8E` | same run | none | **REUSE** — self-contained cache (`features_fp16`, `images_uint8_224`, `paths`); overlaps A1a/A1b. Keep this **or** the npy pair, not both. |
| A1f | `…/image_decoder.pt` | 51,841,575 B | `68E3B1F5…4994698C5` | in-house decoder training (`tools/train_multimodal_decoders.py:511` writes `{modality}_decoder.pt`; `tools/train_unit_query_align.py` trains the query aligner) | none (C++); service script only | **REUSE** — trained weights behind `multimodal_dec/image_decoder.onnx`; keep as re-export source |
| A1g | `…/llava_forward_sample.pt` | 9,455,821 B | `26EE23DB…E018017DF6` | one-image forward probe (holds `unit_query` + `source_image` + `path`) | none | **KEEP** — 9 MB provenance specimen proving the LLaVA pipeline was real; useful when writing model docs/tests |
| A1h | `…/vjepa2_1_vitb_features.pt` | 7,079,698 B | `F65338B4…7C0D9F84` | V-JEPA2 ViT-B feature probe (single `features` tensor + `input_shape`) | none | **DELETE-SAFE** — probe output only; 0 readers; it is the only genuinely-"jepa" tensor here and it is a 7 MB sample, not ViT-g/H weights |
| A1i | `…/llava_more_paths.json` | 499,114 B | — | path list for the 4,000-image set | none | **KEEP** — label/path mapping for A1c; without it the 4,000 features have no image correspondence |
| A1j | 7 status/progress JSONs: `llava_cache_more_progress.json` 190 B, `llava_decoder_result.json` 1,549 B, `llava_gpu_train.json` 3,232 B, `llava_vision_multimodal.json` 3,385 B, `qwen_audio_forward.json` 1,069 B, `result.json` 3,860 B, `vjepa2_1_vitb_forward.json` 1,972 B | 15,257 B total | — | one-off run status | none | **DELETE-SAFE** — run logs/status; no functionality (contents are already summarised in this audit) |

### 2.2 LLaVA / Qwen2-Audio artifacts (the dead 7B path)

| ID | Path | Size | sha256 (sample) | Created by | Referenced by code/config? | Verdict |
|---|---|---|---|---|---|---|
| B1 | `runtime_store/models/multimodal_enc_src/llava-1.5-7b-hf/` | part of 8,923,801,660 B dir | — | HF download started 2026-10-16 21:28 for `tools/prepare_multimodal_encoders.py` / `tools/multimodal_encdec_server.py` | only the service scripts, and they read `runtime_store/models/multimodal_enc/` — **which does not exist** | **DELETE-SAFE** — **incomplete**: `model.safetensors.index.json` declares 3 shards; only `model-00001-of-00003.safetensors` is present (4,761.63 MiB). Cannot load the model. |
| B2 | `runtime_store/models/multimodal_enc_src/Qwen2-Audio-7B-Instruct/` | part of the same 8.31 GiB | — | same session | same | **DELETE-SAFE** — index declares 5 shards; only shard 1 (3,730.15 MiB) is present plus a 2.99 MiB `.incomplete` download fragment. Cannot load. |
| B3 | `runtime_store/models/multimodal_dec/llava_unit_tokenizer.onnx` | 33,625,788 B | `3A280BBC…3BBF72C` | `tools/train_unit_query_align.py` / `train_multimodal_decoders.py` (Oct 16–18) | only `tools/multimodal_encdec_server.py` (service never started) — no C++ reader | **REUSE** — complete, loadable `[batch,576,4096] → [batch,4096]` (see §5 R1) |
| B4 | `runtime_store/models/multimodal_dec/image_decoder.onnx` | 51,833,682 B | `DD0952CA…2540A635` | same | same | **REUSE** — complete, loadable `[batch,4096] → [batch,3,224,224]` |
| B5 | `…/llava_unit_tokenizer.pt` + `…/image_decoder.pt` | 33,626,639 + 51,839,813 B | — | training checkpoints | service only | **KEEP** — source weights the ONNX files were exported from (needed to retrain/re-export) |
| B6 | `…/image_decoder_prev512.pt` + `…/llava_unit_tokenizer_prev512.pt` | 51,840,645 + 33,626,703 B | — | earlier 512-d-bottleneck variant | none | **DELETE-SAFE** — superseded by the 4096-d ONNX pair; a 512-d intermediate that no artifact now references |
| B7 | `…/continue_gpu_train.json` | 34,966 B | — | training resume state | none | **DELETE-SAFE** — one-off training status |
| B8 | `multimodal_world_model.{hpp,cpp}` (`MultimodalImageWorldModel`, `MultimodalAudioWorldModel`, base URL `http://127.0.0.1:8085`) — dead **code** | ~23 KB source | n/a | v7 era | **never instantiated**: all 9 grep hits are inside `multimodal_world_model.cpp` itself; `Get-NetTCPConnection` shows **no listener on 8085** | **DELETE-SAFE (code)** — functionality = LLaVA-4096-d/Qwen-audio HTTP client; absent because nothing constructs it and the :8085 server that would answer it was never completed (B1/B2 incomplete, `multimodal_enc/` never produced). Keep until Lead decides code deletion; it is *not* a data dependency. |

### 2.3 `ijepa/resnet18_224/` — ResNet18-modified in-house autoencoder (132.85 MB)

| ID | Path (under `runtime_store/models/ijepa/resnet18_224/`) | Size | sha256 (sample) | Created by | Referenced by code/config? | Verdict |
|---|---|---|---|---|---|---|
| C1 | `model_encoder.onnx` | 32,146,521 B | `BB96F89F…243C088F` | `.remote_ref/train_video.py:258` + `tools/export_multimodal.py` (timm ResNet18 + Flatten+Linear head) | tool-only: `tools/io_scaleup.py:20–22` uses it as an encoder-swap rung | **REUSE** (see §5 R2) |
| C2 | `model_encoder_head.onnx` | 262,951 B | `A6148DF5…5AB7B6CA` | same export | tool-only | **REUSE** — 512→128 projection companion to C1 |
| C3 | `encoder_head.json` | 1,456,019 B | `871AF9F6…D79DFD394F` | `.remote_ref/train_video.py:284`, `.remote_ref/export_multimodal.py:184`, `tools/export_*.py`; `tools/compile_bpu.sh:182` copies it | **0 readers** — no C++/config hit (the AGENTS.md claim that `video_model.cpp` loads it is stale; see §2.6 F6) | **DELETE-SAFE** — functionality = BPU-era 512→128 head loader; absent because the loader was removed and today's 128-d path uses `model_encoder_head.onnx` (C2) or the deterministic hash projection `projectToDimension` (`video_model.cpp:497–499`, `semantic_unit.cpp:326–340`). The export scripts can regenerate the file; `tools/io_train_head.py` documents the same raw format. |
| C4 | `model_decoder.onnx` | 13,381,131 B | `9127D58A…55049E6968A` | export (initial/unrefined decoder) | none | **DELETE-SAFE** — superseded by C5 (manifest's declared decoder is `model_decoder_trained.onnx`); no resolver match (`resolveOnnxModelPath` wants `best.onnx`/`model.onnx`, `video_model.cpp:89–105`) |
| C5 | `model_decoder_trained.onnx` | 13,380,761 B | `B2E85141…BFC30C2BF` | decoder trained on 100 calibration pairs (manifest `notes`) | none | **REUSE** — complete 128→224 decoder; companion for R2 |
| C6 | `model_encoder.bin`, `model_decoder.bin` | 8,325,447 + 3,606,699 B | — | `hb_mapper` BPU compile | **config-referenced**: `config/phoenix.json:778–779` (`jepa.image.horizonModel/DecoderModel`) and `config/phoenix.example.json:672–673` | **KEEP (archive)** — cannot run today because `VideoHbdnnModel::encode()` is a stub ("BPU image encoder preprocessing is not implemented in this build"), but the keys are live config; deleting them requires the config cleanup in §4 Batch 2. |
| C7 | `calibration_encoder/` + `calibration_decoder/` (100 + 100 raw NCHW `.bin`) | ~57 MB + ~0 | — | hb_mapper calibration inputs (2026-08-01) | only hb_mapper recompiles | **DELETE-SAFE** if C6 is kept as-is (the compiled `.bin` do not need calibration at runtime; only recompiling does). Marked KEEP-OPTIONAL if Lead wants to retune/recompile. |
| C8 | `model_encoder_hb_mapper_config.yaml`, `model_decoder_hb_mapper_config.yaml`, `hb_mapper_checker.log`, `model_decoder_hb_mapper_checker.log` | ~30 KB | — | BPU compile run | none | **DELETE-SAFE** — build logs/config of a compile that already produced C6 |
| C9 | `extract_head.py` | 638 B | — | one-off export helper | none | **DELETE-SAFE** — writes C3; the same logic is in `.remote_ref/train_video.py` |

### 2.4 `ijepa/speech_16k/` — legacy 1D speech autoencoder (25.34 MB)

| ID | Path (under `runtime_store/models/ijepa/speech_16k/`) | Size | sha256 (sample) | Created by | Referenced by code/config? | Verdict |
|---|---|---|---|---|---|---|
| D1 | `model_encoder.onnx` | 823,873 B | `472D7147…40275C5E` | `.remote_ref/train_audio.py` / `tools/export_audio_onnx.py` / `tools/export_bpu_speech_*.py` | tool-only (`tools/inspect_speech_onnx.py`); **runtime resolver misses it** — variant `audio-16k` maps to `speech_16k` but `resolveOnnxModelPath` only looks for `best.onnx`/`model.onnx` (`audio_model.cpp:91–107`) | **DELETE-SAFE** — functionality = legacy 1D JEPA-v2 speech encoder (16 kHz→128); provided today by `additive_jepa/speech_encoder/best.onnx` (live) and by the BPU pair D3 |
| D2 | `model_decoder.onnx` | 8,883,222 B | `0E01EA7E…BAA21330` | same | same | **DELETE-SAFE** — same reasoning; live decoder is `additive_jepa/speech_decoder/best.onnx` |
| D3 | `model_encoder.bin`, `model_decoder.bin` | 527,195 + 8,651,378 B | — | hb_mapper | **config-referenced**: `config/phoenix.json:788–789`; `AudioHbdnnModel` is implemented (`audio_model.cpp:134–527`, `runEncoderBpu` calls `rdk_x5_bpu::execute`) | **KEEP** — the only working X5 BPU audio pair |
| D4 | `calibration_encoder/` + `calibration_decoder/` | ~6.1 MB | — | hb_mapper inputs | recompile only | **KEEP (optional)** — cheap; needed to rebuild D3 |

### 2.5 `additive_jepa/` — live runtime pair + two unreachable intermediate exports (135.67 MB total)

| ID | Path (under `runtime_store/models/additive_jepa/`) | Size | sha256 (sample) | Created by | Referenced by code/config? | Verdict |
|---|---|---|---|---|---|---|
| E1 | `vision_encoder/best.onnx` | 54,612,965 B | `18E31191…06750F25` | additive-residual training (`tools/additive_jepa.py`, `tools/bpu_evolve_additive.py`, `tools/run_all_additive_training.py`) then `tools/export_runtime_models.py` (`best.onnx` = fused export) | **YES** — `VideoLocalOnnxModel` via `resolveOnnxModelPath("encoder","vision_encoder")`, `video_model.cpp:461–503`; `config/phoenix.json:1005–1012` pins `localBackend=cpu` | **KEEP** |
| E2 | `vision_decoder/best.onnx` | 10,444,866 B | `720BED0C…55697339D2108` | same | **YES** — decode path `video_model.cpp:530–549` | **KEEP** |
| E3 | `speech_encoder/best.onnx` | 4,936,265 B | `ECAADC84…381695D4` | same (audio) | **YES** — `AudioLocalOnnxModel`, `audio_model.cpp:1101–1139` | **KEEP** |
| E4 | `speech_decoder/best.onnx` | 15,012,860 B | `A5E68435…FDE0FC636` | same | **YES** — `audio_model.cpp:1181+` | **KEEP** |
| E5 | `speech_encoder/model_encoder.onnx` | 4,936,265 B | `29F2AF01…F55AF6B07D` | intermediate export from `tools/export_audio_onnx.py` | **unreachable**: `resolveOnnxModelPath` only matches `best.onnx`/`model.onnx`; `additiveDirFor("speech_encoder","encoder")` → `speech_encoder`, then `best.onnx` wins first | **DELETE-SAFE** — same I/O as E3 (not byte-identical: different SHA-256, i.e. a different export of the same architecture). E3 is the fused/live file; nothing loads E5. |
| E6 | `speech_decoder/model_decoder.onnx` | 15,012,860 B | `9B2FC9BC…04A9D5F95B0` | same | same | **DELETE-SAFE** — decoder twin of E5; E4 is live |
| E7 | `vision_encoder/best.bin`, `vision_decoder/best.bin`, `speech_encoder/best.bin`, `speech_decoder/best.bin` | 14,358,479 + 6,939,747 + 1,378,652 + 8,034,486 B | — | hb_mapper compile | **YES** — `resolveBpuModelPath` (`video_model.cpp:62–87`, `audio_model.cpp:65–89`) on RDK X5; audio path functional, image encode is a stub | **KEEP** (RDK fallback) |
| E8 | 4 × `model.manifest.json` | ~1.6 KB each | — | export | metadata only | **KEEP** — provenance + `io_train_head.py` reads them |
| E9 | Directory name `additive_jepa/` itself ("jepa" in the path) | n/a | n/a | legacy naming | hardcoded in C++ resolver + config | **KEEP (do not rename now)** — rename is a code+config migration, not a deletion. Correct vocabulary in `model_provenance.md` §3. |

### 2.6 Dead code / dead process / config debt

| ID | Item | Evidence | Verdict |
|---|---|---|---|
| F1 | `encoder_head.json` loader | 0 hits for `encoder_head` in `.cpp/.hpp/.inc` (scan §1.2). AGENTS.md's "video_model.cpp loads encoder_head.json" is **stale** (contradicted by the scan) | covered by C3 — **DELETE-SAFE (data)**; docs must be corrected |
| F2 | `resnet18_224` pipeline reachability | `imageWeightsDir("resnet18_224")=="resnet18_224"` (`video_model.cpp:43`); `resolveOnnxModelPath` tries `additive_jepa/resnet18_224/<best.onnx|model.onnx>` then `ijepa/resnet18_224/<best.onnx|model.onnx>` — the dir has `model_encoder.onnx` / `model_decoder.onnx`, so nothing matches (`video_model.cpp:89–105`; same trap `audio_model.cpp:91–107` and `speech_16k`) | **DELETE-SAFE (archive)** for unreachable files (C3/C4/D1/D2); **REUSE** the loadable C1/C2/C5 (R2) |
| F3 | `MultimodalImageWorldModel` / `MultimodalAudioWorldModel` | 9 hits each, all inside `multimodal_world_model.cpp`; no construction site; no :8085 listener; server artifacts never produced | **DELETE-SAFE (code)** after Lead decides; not a data dependency |
| F4 | 7 zombie `phoenix_main.exe` processes | pids 46520 (2026-11-16 13:58), 47604 (11-16 20:23), 48252 (11-16 20:05), 44932 (11-17 00:55), 37916 (11-17 08:12), 19460 (11-17 08:14), 38228 (11-17 08:15). All: **empty CommandLine**, **RSS = 0.0 MB**, not listening (only pid **23920** owns 127.0.0.1:5097; its child 13032 is `--frontend-only=1`), no children. The live gateway + frontend must NOT be touched. | **DELETE-SAFE (process kill)** — safe to terminate after Lead approval; not a file deletion. Do **not** kill 23920/13032 or any llama-server. |
| F5 | Config debt: `jepa.image.variant="video-encoder"` but `jepa.image.horizonModel` = `ijepa/resnet18_224/model_encoder.bin` (`config/phoenix.json:773–780`) | variant's encoder is 128-d; the ResNet18 BPU model emits 512-d. Harmless today (`localBackend=cpu`, image BPU encode stub) but a latent mismatch if BPU image path is revived | **KEEP + FIX later** (config edit, not deletion) |
| F6 | Docs claiming "jepa" models / `encoder_head.json` reader | `AGENTS.md` (Video/Audio BPU Pipeline section) says the C++ loads `encoder_head.json`; `ijepa/README.md` says four Meta I-JEPA weight dirs (~11 GB) exist — they do **not** in this tree | **KEEP + CORRECT** via `model_provenance.md` §3 |

### 2.7 Runtime mission state (Lead's added item) — diagnosis row

| ID | Item | Size | Evidence | Verdict |
|---|---|---|---|---|
| G1 | `runtime_store/autonomy_state.json` + `runtime_store/mission_workspace/` + in-memory `missions_` | state 908,440 B (mtime 23:16:46); workspace 8,726,767 B / **38,001 dirs** | see §3 | **ACTIVE-CHURN-BY-DESIGN** (not legacy junk, not a mission-state bug) — legacy tail prunable per §3 rule. No `ACTIVE-BUG` escalation for the failure count. |

### 2.8 Other large/stale artifacts found while scanning (not in the original candidate list)

| ID | Path / item | Size | Referenced? | Verdict |
|---|---|---|---|---|
| H1 | `runtime_store/logs/runtime_debug.log` | 35,011.10 MiB (growing; tree total `runtime_store/logs` = 36,721,361,677 B in 39 files) | written by `loggerCXX.cpp:91–93` (Debug mode, `std::ios::app`, **no rotation/size cap**) | **DELETE-SAFE (truncate/rotate at a restart)** — see §4 Batch 2; keep `fatal_runtime.log` (0.6 MB crash channel) |
| H2 | 37 other historical logs in `runtime_store/logs/` (llama_8082.err.log 1.1 MB, runtime_release.log 7.2 MB, etc.) | ~1.9 MB | old sessions (≤ 2026-10-18) | **DELETE-SAFE** except `fatal_runtime.log` |
| H3 | `runtime_store/autonomy_state*.json` backups (106 files) | 41 MiB total | written by tests/self-mod checkpoints | **DELETE-SAFE** except the live `autonomy_state.json` + the 2 newest `autonomy_state.pre-*`/`after-*` (prune rule) |
| H4 | `phoenix_main.exe.dupfix`, `.dupfix2`, `.dupfix3` | 3 × ~57 MB (Jul 9) | one-off duplicate-handle debug binaries | **DELETE-SAFE** |
| H5 | Stale test binaries: `gtest_new_runner.exe` 101.7 MB, `gtest_mixed_modal_patched.exe` 61.9 MB, `phoenix_main_edge_disabled.exe` 44.1 MB, `gtest_emotion_speak.exe` 25.4 MB, `world_model_prompt_tests.exe` 3.5 MB, `world_model_scene_brain_tests.exe` 4.1 MB | ~241 MB | superseded by `gtest_runner.exe` (252.7 MB, 2026-11-16) and `phoenix_main.exe` | **DELETE-SAFE** (all older than 2026-08-01; nothing references them) |
| H6 | `runtime_store/large_vision_encoder/` (data/{beans,librispeech_dummy}) | 39.49 MiB | 0 code refs; separate from `runtime_store/models/large_vision_encoder/` | **DELETE-SAFE** — demo dataset leftovers; real corpora live in `runtime_store/datasets/` |
| H7 | `build/` | 23.46 GiB (`build/tmp` 14.84 GiB incl. bundled gdb; conan caches 6.38 GiB; rdk-sysroot 1.96 GiB; obj_gtest 0.45 GiB) | active build/compile caches | **KEEP** (needed to build; cleanup is a separate change after the build freeze) |
| H8 | `runtime_store/tmp/` | 2,870.41 MiB (858 files: one-off probe scripts/results) | some stability/probe scripts; ownership unclear | **UNCLEAR-needs-user** |
| H9 | `runtime_store/models/granite-3.1-1b-a400m-instruct/` | 2,673,990,159 B | **live**: `tools/intent_carrier_lora.py`, `intent_carrier_train.py`, `intent_meaning_train.py`, `meme_pair_train.py`, `intent_signal_trace.py` | **KEEP** |
| H10 | `runtime_store/models/io_train/` (5.5 MB) + `io_train2/` (29.8 MB) | active 2026-11-17 head-training runs (`tools/io_train_head.py`, `io_scaleup.py`) | **KEEP** (other teammates' active work) |
| H11 | Workspace-root twins (`E:\079\v6.0Alixander\runtime_store`, `\data`, `\build`, `\config`, `\doc`, `\phoenix_milestone_judge`, `_phoenix_weight_hold`, `\v6.0Alixander-from-remote`) | out of live tree; stale August snapshots | none (per user instruction) | **OUT OF SCOPE** — do not delete from this audit; Lead may schedule a separate stale-twin cleanup |

---

## 3. Mission-state diagnosis (Lead's extra item)

**Question:** gateway reports `final missions: running=6 completed=2 failed=9750 total=9758`.
Is that (a) legacy junk in `runtime_store/autonomy_state.json`, or (b) an active bug
marking missions Failed at a high rate?

**Verdict: ACTIVE-CHURN-BY-DESIGN.** Neither legacy junk nor a mission-state bug:
an **external self-modification/candidate harness** assigns one mission per code-edit
candidate and retires it with `goalAchieved=false`. The count is a by-product of a
`~10k`-iteration self-mod loop, and it is still running (new missions at 23:27).

Evidence:

1. **Persisted state is small and bounded by design.**
   `runtime_store/autonomy_state.json` (908,440 B, `schemaTag: v10.1`) contains
   `missions` = **66** entries: **2 Running** (state=1) + **64 Failed** (state=3;
   `MissionState { Idle=0, Running=1, Completed=2, Failed=3 }`,
   `mission_lifecycle.hpp:42`). Start times run 2026-11-16 22:00:00 →
   2026-11-17 23:15:59. All 66 goals are of the form
   `SELF-MOD: … [iter N-M]` (e.g. `SELF-MOD: make the plannedEditCount lround
   comment exact [iter 10041-40]`), each ran 20–77 s
   (`completionTimeMs` 19948–77234), pressure 0.0 after finish.
   The file holds only 64 finished missions because `exportState()` deliberately
   caps the snapshot at `kKeepFinished = 64` (`autonomy_stack.cpp:2267–2291`).
   So the file is the *tail* of a much longer history, not the whole thing.

2. **The authoritative in-memory count is 9,758.**
   The Lead's stability run (`tools/stability_run.ps1`) computes those numbers from
   `GET /api/mission/status` → `result.missions[]` → `stats.mission.state`, counting
   state==1/2/3 (`stability_run.ps1:409–426`). The live gateway's in-memory
   `missions_` map therefore holds ~9,758 entries: 6 Running, 2 Completed, **9,750
   Failed**. A direct `GET /api/mission/status` from this audit returned 401
   (gateway auth), so the script's read-out + the Lead's log line are the
   authoritative source.

3. **The only code path that can set Failed is an explicit
   `reportMissionOutcome(goalAchieved=false)`.**
   `reportMissionOutcome()` reads `p.value("goalAchieved", false)`; false →
   `msn->markFailed()` (`autonomy_stack.cpp:3195–3215`). The in-tree C++ callers
   pass **true** (`main_hub_parts/111_class_gatewayserver.inc:919–921`, `:3458–3459`);
   the public route forwards the *client's* payload
   (`main_hub_parts/116_section_tail.inc:5058`). There is **no loop in C++** that
   reports failures — so the failure rate is driven by whoever calls the HTTP route,
   not by the autonomous loop or by child boxes.

4. **The caller is visible in the live gateway log.**
   `runtime_store/long_run/gateway.log` (13.3 MB, mtime 23:48) shows the
   assign→deliberate→report cycle for candidate-scoped mission ids:
   `[mission-genome] scope=tr-10148-14`, `[mission-deliberate] …`,
   then repeated `POST /api/mission/report -> 200` and
   `POST /api/mission/assign -> 200` (4× each in one tail window). The goal wording
   matches the candidate list in `build/tmp/growth_loop.ps1:23–28`, and the
   `[iter N-M]` suffix is generated by that external harness — it appears **nowhere**
   in the C++/Python sources (0 hits for `"tr-"`, `[iter` in source).

5. **The churn is happening right now, not just history.**
   `runtime_store/mission_workspace/` has **38,001** mission-id directories
   (8.7 MB total): ids `tr-998-14 … tr-10065-29`; newest `tr-10065-29` was created
   **23:27:20**, i.e. after the Lead's dry run. The ids map 1:1 to the 38k mission
   entries accumulated since the self-mod tree started (iteration counter is at
   ~10,065, ~2–4 missions per iteration).

**Consequence / what to do (no code bug to escalate):**

- If the 9,750 failures are unwanted noise, **stop the self-mod candidate harness**
  (the process that POSTs `/api/mission/assign` + `/report`), not the gateway.
  The gateway behaves exactly as specified.
- **Prune rule for the legacy tail** (only after the harness is stopped and Lead
  approves): in `runtime_store/mission_workspace/`, keep
  (a) every directory whose id appears in `autonomy_state.json.missions[].id`
  (the 2 running + 64 recent), (b) every directory modified in the last 24 h,
  (c) `runtime_store/selfmod/tree.json`; delete the rest (`tr-998-14` … ~`tr-10000-*`).
  Do **not** edit `autonomy_state.json` while the gateway is running (it rewrites it
  on every persist); the in-memory `missions_` history cannot be trimmed without a
  restart.
- **Two separate real observations** (not the failure count, but worth Lead action):
  1. `missions_` is never pruned **in memory** — only the exported snapshot is
     bounded (`autonomy_stack.cpp:2267–2291`). 9,758 retained mission objects ≈
     >100 MB (0.9 MB state file ≈ 66 missions ⇒ ~14 KB each). A `pruneFinishedLocked()`
     keeping the last N finished missions would fix it; `importState` already does a
     trim on load (`autonomy_stack.cpp:2373`).
  2. `runtime_store/logs/runtime_debug.log` is **35 GB** and unbounded
     (`loggerCXX.cpp:91–93`, append-only, no rotation) — the largest single dead-data
     item in the tree, see H1.

---

## 4. Proposed deletion set (exact rows, for Lead approval)

**Nothing below is executed yet.** "Reclaim" is the file bytes only.

### Batch 1 — no preconditions (pure dead files; ~9.58 GB / 8.92 GiB)

| # | Row | Path | Reclaim |
|---|---|---|---|
| 1 | A1h | `runtime_store/models/large_vision_encoder/vjepa2_1_vitb_features.pt` | 7,079,698 B |
| 2 | A1j | the 7 status JSONs under `…/large_vision_encoder/` (list in A1j; **not** `llava_more_paths.json`) | 15,257 B |
| 3 | B1+B2 | `runtime_store/models/multimodal_enc_src/` (incomplete LLaVA-1.5-7B + Qwen2-Audio-7B downloads) | 8,923,801,660 B |
| 4 | B6+B7 | `runtime_store/models/multimodal_dec/{image_decoder_prev512.pt,llava_unit_tokenizer_prev512.pt,continue_gpu_train.json}` | 85,502,314 B |
| 5 | C3 | `runtime_store/models/ijepa/resnet18_224/encoder_head.json` | 1,456,019 B |
| 6 | C4 | `runtime_store/models/ijepa/resnet18_224/model_decoder.onnx` (superseded by C5) | 13,381,131 B |
| 7 | C7 | `…/resnet18_224/calibration_encoder/` + `calibration_decoder/` (if C6 kept as archive) | ~57 MB |
| 8 | C8+C9 | `…/resnet18_224/{model_*_hb_mapper_config.yaml,hb_mapper_checker.log,model_decoder_hb_mapper_checker.log,extract_head.py}` | ~32 KB |
| 9 | D1+D2 | `runtime_store/models/ijepa/speech_16k/{model_encoder.onnx,model_decoder.onnx}` | 9,707,095 B |
| 10 | E5+E6 | `runtime_store/models/additive_jepa/speech_encoder/model_encoder.onnx`, `…/speech_decoder/model_decoder.onnx` (unreachable intermediate exports) | 19,949,125 B |
| 11 | H4 | `phoenix_main.exe.dupfix`, `.dupfix2`, `.dupfix3` | ~171 MB |
| 12 | H5 | `gtest_new_runner.exe`, `gtest_mixed_modal_patched.exe`, `phoenix_main_edge_disabled.exe`, `gtest_emotion_speak.exe`, `world_model_prompt_tests.exe`, `world_model_scene_brain_tests.exe` | ~241 MB |
| 13 | H6 | `runtime_store/large_vision_encoder/` (workspace copy of demo data, 39.49 MiB) | 41,410,000 B |

### Batch 2 — needs a precondition (quiesce/restart/config edit; ~35.2 GiB)

| # | Row | Precondition | Reclaim |
|---|---|---|---|
| 14 | H1 | rotate/truncate `runtime_store/logs/runtime_debug.log` — the gateway holds it open; rename-at-restart (or stop Debug logging per `loggerCXX.cpp:91–93`, then delete/truncate). Keep a ≤100 MB tail. | ~35 GiB |
| 15 | H2 | delete 37 stale log files in `runtime_store/logs/` **except** `fatal_runtime.log` | ~1.9 MB |
| 16 | H3 | keep live `autonomy_state.json` + 2 newest snapshots; delete the other 104 backups | ~40 MiB |
| 17 | G1 prune | stop the self-mod harness first, then prune `mission_workspace/` per §3 rule | 8.7 MB (dirs/entry count, not bytes) |
| 18 | F4 | `Stop-Process` the 7 zombie pids 46520/48252/47604/44932/37916/19460/38228 (RSS 0, empty cmdline) — **never** 23920/13032/llama-server | 0 B (process hygiene) |
| 19 | C6 (optional) | if the Lead decides the ResNet18 BPU archive is dead: delete `model_encoder.bin` + `model_decoder.bin` **and** remove/redirect `config/phoenix.json:778–779` (+ example config 672–673) in the same change | 11,932,146 B |

### Batch 3 — do **not** delete (REUSE / KEEP)

A1a–A1g, A1i, B3–B5, C1, C2, C5, C6 (default), D3, D4, E1–E4, E7–E9, H7, H9, H10,
plus `runtime_store/datasets/`, `GGUF_models/`, `config/`, all live `.exe`/`.dll`,
`phoenix_main.map`, `fatal_runtime.log`, `runtime_store/selfmod/tree.json`.

---

## 5. What is worth REUSING (plans; no implementation)

### R1 (top pick) — LLaVA-1.5-7B feature caches + in-house unit tokenizer/decoder (A1a–A1f, B3–B5, ~28 GB)

What it is: a workable **visual-semantic pipeline** trained in-house on top of the
LLaVA-1.5-7B vision tower: `image [3,224,224] → vision_tower+projector → 576×4096
features → llava_unit_tokenizer.onnx → unit_query [4096] → image_decoder.onnx →
image [3,224,224]`. The caches contain 4,512 paired (features, image, path) samples
from Tiny-ImageNet-200 — the data that trained the tokenizer/decoder.

Concrete uses, cheapest first:

1. **Offline evaluation now** (no training): run `multimodal_dec/image_decoder.onnx`
   and `llava_unit_tokenizer.onnx` on held-out Tiny-ImageNet images using the
   `vision-cu111` venv (torch/onnxruntime already installed) and compute
   reconstruction/retrieval numbers. This quantifies whether the LLaVA semantic
   channel is better than the live 128-d additive encoder — the result decides R1.2.
2. **Distill into the live concept space**: train a `4096 → 128` linear/MLP projector
   with the frozen LLaVA `unit_query` as teacher and the live `vision_encoder/best.onnx`
   concept as student input (or use features directly), writing checkpoints to
   `runtime_store/models/io_train2/<ts>/` exactly like `tools/io_train_head.py` /
   `tools/io_scaleup.py` already do. If it improves the held-out probe, ship it as a
   *head* artifact (128→128) — the runtime already supports "concept = normalize(W @
   encoder(input) + b)" per `io_train_head.py:17–20`, and the legacy
   `encoder_head.json` format is the precedent for the file layout (C3 is safe to
   delete precisely because this new head would be trained instead).
3. **Serve the LLaVA unit decoder** in the image-decode direction (e.g. a
   "concept → image" preview at 4096-d) if a use is found; it needs no LLaVA
   download, only the 85 MB ONNX pair and the caches.

What must change: nothing in product C++ to reuse the data (training scripts are
out-of-process); a runtime wiring would need a new loader for a `best.onnx` in a new
`runtime_store/models/vision_llava/` dir (the resolver already finds `best.onnx`) and
a `jepa.image.variant` config value. Do **not** edit product sources in this audit.

Why keep the big file: `llava_features_more.npy` (18 GB) is the only large training
pool, and it is **not regenerable from disks as they are** — the LLaVA HF download is
incomplete (B1), so re-extracting features would need a fresh ~13 GB model download
plus GPU time. If the Lead wants the 18 GB back regardless, delete A1c+A1d only
(19.4 GB) and keep A1e (self-contained 512-sample cache, 2.4 GB).

### R2 (second pick) — ResNet18-modified encoder kit (C1, C2, C5; ~46 MB)

What it is: a complete, **loadable today** ResNet18-modified image autoencoder:
`model_encoder.onnx` (`[1,3,224,224] → pooled [1,512,1,1]`) +
`model_encoder_head.onnx` (`512→128`) + `model_decoder_trained.onnx`
(`128 → [1,3,224,224]`, trained on 100 calibration pairs per manifest).

Concrete uses:

1. **Encoder-swap experiment** — `tools/io_scaleup.py:20–22` already names this as
   rung (d): "in-house ResNet18-modified 512-d encoder + linear 512→128 head. If
   (d) >> (a..c) the ENCODER is the bottleneck." Running it costs nothing and answers
   whether the live 128-d additive encoder is the IO-line bottleneck.
2. **Deployable alternative vision pair without code changes** — copy
   `model_encoder.onnx → runtime_store/models/vision_resnet18/best.onnx` and
   `model_decoder_trained.onnx → …/vision_resnet18_decoder/best.onnx`, then point
   `jepa.image.variant="vision_resnet18"` at it: `resolveOnnxModelPath` matches
   `best.onnx` and `imageWeightsDir` passes the variant through, so the existing
   `VideoLocalOnnxModel` would load it (512→128 self-projection needs a small head;
   keep C2 for that or train the new head per R1.2).
3. The `.bin` twins (C6) let the same pipeline run on the X5 **if** the image BPU
   encode stub (`video_model.cpp` "BPU image encoder preprocessing is not
   implemented in this build") is finished — that is a code task, not this audit.

What must change: only new files/dirs + config (for option 2); no product source
edit is needed for the experiment. Keep the three ONNX files until the experiment
result is recorded in `doc/v10.1/io_training.md`.

---

## 6. Things that look dead but are NOT (guard rails)

- `additive_jepa/{vision,speech}_{encoder,decoder}/best.onnx` — the live IO path
  (config pins `localBackend=cpu`, `config/phoenix.json:1005–1022`).
- `best.bin` ×4 — RDK X5 BPU fallback via `resolveBpuModelPath` (audio BPU path is
  implemented; do not delete with the ONNX story).
- `ijepa/speech_16k/*.bin` + calibration — config-referenced BPU audio pair.
- `runtime_store/models/granite-3.1-1b-a400m-instruct/` — used by the intent/meme
  training tools (H9).
- `runtime_store/models/io_train{,2}/` — another teammate's active work (2026-11-17).
- `runtime_store/autonomy_state.json` itself — live agent state; only backups are
  prunable.
- `phoenix_main.exe` / `phoenix_main.next.exe` / `.prev.exe` / `.timing.exe` — A/B
  and timing artifacts of the current stability work; keep until that work closes.
- `runtime_store/logs/fatal_runtime.log` — the crash-forensics channel used by
  `tools/stability_run.ps1`; keep even when truncating H1.

---

## 7. Residual risk / open questions (UNCLEAR)

1. **A1c/A1d (19.4 GB)**: keep for R1 retraining, or delete for disk? Both defensible;
   needs Lead/user decision because regeneration requires re-downloading LLaVA-1.5-7B.
2. **C6 ResNet18 `.bin` (11.9 MB)**: archive for RDK or delete with config cleanup?
3. **B8 multimodal_world_model code**: dead but not data; deletion is a source edit
   outside this audit's write scope.
4. **H8 `runtime_store/tmp/` (2.85 GB)**: ownership unclear (one-off probe scripts
   from earlier sessions); needs Lead decision, not guessed.
5. **`runtime_debug.log` truncation** cannot be done safely while pid 23920 holds the
   file open on Windows; must be done at a gateway restart or by adding rotation to
   `loggerCXX.cpp` (a source change — not this audit).

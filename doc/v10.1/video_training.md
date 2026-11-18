# Video / image IO training — LLaVA-cache distillation + video-path characterisation

Status: run completed **2026-11-18 19:20–19:52 +08:00** on the live tree
`E:\079\v6.0Alixander\phoenix`. CPU-only (GPU left to the live stack); gateway
pid 17376 (:5097) and the 3 llama-servers were **not touched** (health 200 before
and after); no compile.bat, no git, live model dirs read-only.

Run dir: `runtime_store/models/io_train2/video_20261118_192034/`
Tool: `tools/video_train_head.py` (new)

> **Round-2 update (2026-11-18 ~20:00–21:30).** The "runtime BLOCKED" status below is
> superseded. The LLaVA-1.5-7B `vision_tower` + `multi_modal_projector` were found
> **on disk** (stale tree) and a validated 576×4096 signal-matrix tool now exists;
> the distilled head pipeline is proven end-to-end (teacher reproduction cosine
> = 1.000000, head concept cosine = 1.000000). **§9** has the signal matrix, L1/L2
> read-outs, CPU cost, pretrained-ResNet base results, HMDB51 temporal heads,
> deploy contract and GPU budget.

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Does 4096→128 distillation of the LLaVA teacher onto the live concept space work? | **Yes, metric gate PASSES**: +6.83 pp 1-NN top-1, +6.85 pp top-5, +6.95 pp ridge probe over the identity live concept (3 seeds, 2000 held-out each). |
| Is the head deployable today? | **Superseded by §9**: metrics pass, and the 576×4096 teacher extractor now exists on disk (`tools/llava_signal_matrix.py`, validated cos 1.000000). The head pipeline is proven end-to-end; **runtime wiring is pending** — the exact C++ seam is documented in §9.6. The original blocker was the missing teacher extractor. |
| Wider MLP? | **Fails worse than rung-c**: MLP 4096→512→128 has *lower* MSE to the live concept (0.00290 vs 0.00321) yet **collapses** retrieval (top-1 2.9%, probe 2.3%). Fidelity-to-teacher alone is the wrong gate — exactly the rung-c lesson. |
| How good is the teacher geometry? | 4096-d teacher features: **41.2 / 62.9 / 71.7** (top-1/top-5/probe). The 128-d linear head keeps only 32.2 / 55.7 / 42.2 → **the 128-d bottleneck, not the head, is the limit** (loses 9.0 pp top-1, 7.2 pp top-5, 29.5 pp probe). |
| Are `multimodal_dec/{llava_unit_tokenizer,image_decoder}.onnx` usable? | **No.** Tokenizer is near-mean/collapsed (pairwise cos 0.94 across different images, 0.15 mean cos to its own input's mean-pool). Decoder reconstruction is **4.2× worse than a constant mean-image predictor** (MSE 0.320 vs 0.077). The loadable artifacts remain the feature caches + the trained `.pt` weights. |
| Video? | **No local video corpus** (0 mp4/avi/mkv/mov/webm; 0 frame/clip dirs; 0 gif/bmp/png under `runtime_store/datasets`, `data/`, `uploads/`). The live runtime treats **Video == Image**: one payload → one 128-d concept, no temporal model. Minimum dataset spec + required interface are in §5; nothing was trained on invented data. |

---

## 1. Distillation protocol (exactly as run)

**Universe (file-disjoint):** 4,488 images with Tiny-ImageNet-200 class labels
(`n0XXXXXXX` parent dir) built from the LLaVA caches:

- `llava_more_paths.json` 4,000 paths ↔ `llava_features_more.npy (4000,576,4096)` fp16 ↔
  `llava_images_more.npy (4000,3,224,224)` uint8;
- `llava_unit_feature_cache.pt` 512 paths ↔ `llava_features.npy (512,576,4096)` /
  `llava_images.npy` — **488 file-disjoint added, 24 path-overlaps dropped**.
- All caches opened with `np.load(..., mmap_mode="r")`; the 18 GiB file is never
  bulk-loaded (only per-image 576×4096 rows are streamed).

**Teacher input X (4096-d):** mean over the 576 LLaVA tokens, float32, then L2-normalised.
**Distillation target T (128-d):** the LIVE encoder
`runtime_store/models/additive_jepa/vision_encoder/best.onnx` run on the SAME paired
224 px images (RGB/255, ImageNet mean/std, NCHW `[1,3,224,224]`), L2-normalised
(sha256 `18e31191…`). This is the "targets = live encoder features on the same paired
images" requirement.

**Splits / seeds:** 3 seeds = **3 independent random file-disjoint splits**
(seed 20261118/19/20); each split: 2,488 train + 2,000 held-out (held-out exactly as
required, never in train). No image appears in both sides of a split.

**Heads:**
- `lin_ridge4096`: closed-form ridge regression X(4096) → T(128), λ ∈ {1e-4,1e-3,1e-2,1e-1}
  chosen on a 10% internal-val slice of each train split; weights stored `W[128,4096]+b`.
- `mlp512`: MLP 4096 → 512 ReLU → 128 (2.16 M params), MSE to T, Adam lr=1e-3,
  wd=1e-4, batch=256, 40 epochs, torch CPU (`torch.set_num_threads(8)`), output L2-normalised.

**Read-outs (on held-out, identical definitions to `tools/io_train_head.py`):**
cosine **1-NN top-1**, cosine **1-NN top-5** (retrieval into the split's train set),
closed-form **ridge probe** accuracy (λ=1e-2), plus **MSE / cosine to the live concept**
(distillation fidelity).

**Gate (rung-c lesson):** a rung is gate-qualified only if top-1 **and** top-5 **and**
probe all improve over the identity baseline (normalised live 128-d concept) across
3-seed means.

---

## 2. Three-seed results (mean ± sd; per-seed in brackets)

| Rung | 1-NN top-1 | 1-NN top-5 | ridge probe | MSE → live | cos → live |
|---|---|---|---|---|---|
| identity (live 128-d) | **0.2538 ± 0.0037** [0.2590, 0.2505, 0.2520] | **0.4888 ± 0.0057** [0.4945, 0.4810, 0.4910] | **0.3523 ± 0.0042** [0.3545, 0.3560, 0.3465] | 0 | 1.0 |
| teacher 4096-d (no head) | 0.4123 ± 0.0013 [0.4135, 0.4105, 0.4130] | 0.6290 ± 0.0095 [0.6355, 0.6155, 0.6360] | 0.7165 ± 0.0086 [0.7285, 0.7090, 0.7120] | — | — |
| **lin_ridge4096 (distilled)** | **0.3222 ± 0.0123** [0.3365, 0.3065, 0.3235] | **0.5573 ± 0.0099** [0.5660, 0.5435, 0.5625] | **0.4218 ± 0.0020** [0.4235, 0.4190, 0.4230] | 0.003212 ± 0.0000133 | 0.7944 ± 0.0009 |
| mlp512 | 0.0288 ± 0.0029 [0.0285, 0.0255, 0.0325] | 0.1028 ± 0.0061 [0.0985, 0.0985, 0.1115] | 0.0227 ± 0.0022 [0.0200, 0.0225, 0.0255] | 0.002899 ± 0.0000044 | 0.8144 ± 0.0003 |

**Gate (relative to identity):**

| Rung | Δ top-1 | Δ top-5 | Δ probe | all three improve? |
|---|---|---|---|---|
| **lin_ridge4096** | **+6.83 pp** | **+6.85 pp** | **+6.95 pp** | **PASS** |
| mlp512 | −22.50 pp | −38.60 pp | −32.97 pp | FAIL |

The MLP is the important negative result: it is *better* than the linear head on both
distillation-fidelity read-outs (MSE 0.00290 vs 0.00321, cos 0.814 vs 0.794) and still
destroys the global concept geometry. **Never gate an IO head on MSE-to-teacher alone.**

---

## 3. Deploy verdict

> **Superseded by §9 (round 2):** the teacher extractor was found on disk and validated;
> the head pipeline is proven. This section records the round-1 state.

**Metric gate: PASS. Runtime deployment: BLOCKED.**

The distilled linear head satisfies the stated gate, but it consumes a 4096-d
mean-pooled LLaVA feature that the runtime **cannot produce**:

1. The raw LLaVA-1.5-7B checkpoint is not on disk anymore — Batch 1 deleted the
   incomplete `multimodal_enc_src/` download (only 1 of 3 shards). The 22.85 GiB
   caches are the only surviving representation of the teacher, and they are offline
   data, not a callable extractor.
2. The two complete ONNX files that were meant to supply a 4096-d interface both fail
   the usability criteria (§4): the tokenizer maps different images to nearly the same
   query, so it cannot reproduce teacher geometry.
3. `video_model.*` has no 4096-d input path at all (its live encoder is the 128-d
   in-house model, see §5).

So the head is a **research artifact / deployment-ready contract**, not a shipped
model. Minimum unblock (one of):
- **A. Re-acquire the teacher:** download LLaVA-1.5-7B (~13 GB, 3 shards) and export
  `vision_tower+multi_modal_projector` as ONNX `[1,3,336,336] → [1,576,4096]` (~0.3 B
  params); then this head is directly loadable.
- **B. Re-train a real tokenizer:** the current `llava_unit_tokenizer.onnx` is
  collapsed; training it properly against cached unit-query targets still requires a
  genuine 576×4096 tower at *inference* time, so this alone is not a serving path.
- **C. Stay offline:** use the head only over the caches (evaluation/probing) — which
  is what this run does.

**Deployable-today fallback:** the 128→128 linear head from
`runtime_store/models/io_train2/20261117_2330/image/` (prior work, +5.75 pp top-1 /
+6.3 pp top-5 on its own split). It needs no teacher and is the only IO head that can
actually be wired into the current runtime.

---

## 4. Loader contract (`head_lin_ridge4096`)

Files (run dir `runtime_store/models/io_train2/video_20261118_192034/image_distill/`):

| File | Size | Purpose |
|---|---|---|
| `head_lin_ridge4096.f32` | 2,097,664 B (W 128×4096 f32-le row-major + b 128 f32-le) | runtime weights |
| `head_lin_ridge4096.contract.json` | 7 KB | `phoenix.io_head.v1` loader contract |
| `head_lin_ridge4096.npz` | 2.05 MB | numpy checkpoint (W[128,4096], b[128]) |
| `head_mlp512.npz` | 8.45 MB | MLP checkpoint (documented as FAILING the gate) |

Contract fields (abridged; sha256 `86579bf0…`):

```json
{
  "format": "phoenix.io_head.v1",
  "modality": "image", "type": "linear",
  "inDim": 4096, "outDim": 128,
  "normalize": true, "inputNormalize": true, "activation": "none",
  "tensors": [{"name": "W", "shape": [128, 4096], "order": "rowmajor"},
              {"name": "b", "shape": [128], "order": "rowmajor"}],
  "weightBytes": 2097664, "weightDtype": "float32-le",
  "applyAfter": "mean-pooled LLaVA-1.5-7B vision-tower+projector features over 576 tokens (4096-d), L2-normalised, before normalizeVector",
  "inputShape": [4096], "outputShape": [128],
  "requires": {"featureExtractor": "LLaVA-1.5-7B vision_tower+multi_modal_projector producing [576,4096]",
               "shippedLocally": false},
  "deployable": false, "metricsGatePassed": true,
  "deploymentStatus": "METRICS PASS / RUNTIME BLOCKED: no local 576x4096 teacher extractor"
}
```

**Apply order (exact):**
`frames/images → teacher [576,4096] → mean over tokens (4096) → L2-normalise → W@x + b
(W 128×4096, b 128) → L2-normalise → concept vector (128) → existing normalizeVector path`.
Verified end-to-end from the contract bytes: applying to all 4,488 cached teacher
vectors reproduces the in-sample concept with cos 0.899 (mean; held-out 0.794).

**Proposed config keys** (same convention as the prior 128→128 head; no C++ reader
exists yet for either key — the loader must be written before this is live):
`jepa.image.headCheckpoint = <contract path>`, `jepa.image.headRequire = false`.

---

## 5. Video path — what it consumes, and why nothing was trained

### 5.1 Code truth (read from the live tree)

- **Live runtime: Video is encoded as ONE image.** `MixedModalConceptBridge::encode`
  routes `Image` and `Video` through the same branch: one `packet.payload` →
  `videoEncoder(dim).encode(payload, w, h, mime)` → one 128-d concept
  (`external_mixed_modal_io.cpp:956–974`). `MixedModalPacket` carries a single
  `payload` byte vector — no frame list (`external_mixed_modal_io.hpp`, struct).
- **`VideoModel` has no temporal API.** `video_model.hpp:47–136` defines
  `encode(image)`, `encodeContext(mask)`, `encodeTarget(blocks)`,
  `predictTarget`, `adapt`, `decode`; the masks/blocks are **spatial JEPA patch
  masks**, not time. In the live `VideoLocalOnnxModel`, `encodeContext` and
  `encodeTarget` simply call `encode()` and ignore the mask
  (`video_model.cpp:505–519`), `predictTarget` is identity (`:521–523`), `adapt` is
  unsupported (`:525–528`). Factory: `createVideoModel` (`frontend_server.cpp:4554–4558`),
  bridge cache `external_mixed_modal_io.cpp:105–115` (variant hard-wired
  `"video-encoder"`).
- **The only sequence-capable contract is dead.** `multimodal_world_model.hpp:17–47`
  describes per-patch `unitQueries` + mean unit query (LLaVA 4096-d, 576 patches) and
  per-40 ms audio queries — but those classes are never instantiated, the :8085
  service does not run, and even that contract is per-*patch*, not per-frame time.

**Therefore a video head would consume:** either (a) **nothing new** — N per-frame
128-d concepts aggregated by mean/last/attention *outside* the model (no artifact,
deployable today; this is only a fusion policy), or (b) **a new temporal head**
`R^(N×128) → R^128` plus a new interface (frames list in the packet) and a new loader
— neither exists in the current source.

### 5.2 Data truth (numbers)

| Search (2026-11-18) | Result |
|---|---|
| `runtime_store/datasets/**`, `data/**`, `uploads/**` for `*.mp4,*.avi,*.mkv,*.mov,*.webm` | **0 files** |
| directories named `frame*/video*/clip*/sequence*` | **0** |
| `*.gif`, `*.bmp`, `*.png` under those roots | **0** |
| only local image corpus | Tiny-ImageNet-200 (100k JPEG 64×64 train + val) — static images, no clips |

No honest video training is possible locally, so the tool refuses and emits
`video_dataset_requirement.json` instead of inventing data.

### 5.3 Minimum dataset spec (to train the smallest honest temporal head)

| Requirement | Value |
|---|---|
| Clips | ≥2,000 (held-out ≥400 clips, **clip-disjoint**) |
| Frames per clip | 8 (range 4–16) at 224×224 RGB |
| Labels | clip-level class/action, 10–50 classes |
| Storage | ~240 MB JPEG (2,000×8×~15 KB) + manifest |
| Manifest | JSONL: `{"clipId","frames":[paths sorted by time],"label"}` (schema in `video_dataset_requirement.json`) |
| Candidate corpora (do not download here) | UCF101 (13k clips/101 classes, ~7 GB), Kinetics-400 subset, SSv2 subset, or local camera recordings |
| Training shape | per-frame 128-d live concepts → mean-pool + optional linear `(N×128)→128`; same gate as image (top-1/top-5/probe must all improve) |
| Code work needed | extend `MixedModalPacket` with a frame list, add a temporal head interface + contract, and a `jepa.video.*` config key |

---

## 6. Comparison with the previous 128→128 head work (don't mix the numbers)

The prior `io_train/20261117_2220` run (different image universe and split):
identity baseline **0.321 / 0.563 / 0.4985**; best rung linear 128→128
**0.3742 / 0.6258 / 0.5040** (+5.75 pp top-1, +6.3 pp top-5). This distillation run
uses the LLaVA-cache universe (4,488 images, 2000 held-out, 3 random splits), so its
identity baseline is 0.2538/0.4888/0.3523 — **not directly comparable to 0.321/0.563**.
On its own split the distilled 4096→128 head gains +6.83/+6.85/+6.95 pp, which is of
the same magnitude as the prior 128→128 head's +5.75/+6.3, and it adds a probe gain
the prior head barely kept. Both heads pass the rung-c gate on their own protocols;
only the 128→128 head is deployable without new infrastructure.

---

## 7. Artifacts & reproduction

Run dir `runtime_store/models/io_train2/video_20261118_192034/`:

```
manifest.json                      # protocol, per-seed rows, gate, deployment verdict
eval_teacher_models.json           # tokenizer/decoder near-mean probe (§0/§4 numbers)
video_dataset_requirement.json     # video spec (§5.3)
features/distill_features.npz      # 138.7 MB: teacher[4488,4096]f32 live[4488,128]f32 y paths + 16 samples
features/features_meta.json        # encoder sha, preprocessing, counts
image_distill/head_lin_ridge4096.{f32,contract.json,npz,json}
image_distill/head_mlp512.{npz,json}
```

Reproduce (CPU only, ~7 min cold / ~3 min with cached features):

```powershell
cd E:\079\v6.0Alixander\phoenix
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\video_train_head.py `
  --mode image-distill --seeds 20261118,20261119,20261120 --heldout 2000 --threads 8
# reuse the computed features (no 18 GB re-read):
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\video_train_head.py `
  --mode image-distill --reuse-cache runtime_store/models/io_train2/video_20261118_192034
# teacher ONNX probe / video spec:
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\video_train_head.py `
  --mode eval-teacher-models --reuse-cache runtime_store/models/io_train2/video_20261118_192034
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\video_train_head.py --mode video
```

**Files added this task:** `tools/video_train_head.py`, this doc, and the run dir above.
No product source, config, live model dir, gateway or llama-server was modified; no
compile/git was run.

---

## 8. Limitations / what would move the needle

1. **128-d bottleneck is structural for a linear head**: teacher (4096) → 128 linear
   loses 9.0 pp top-1 / 29.5 pp probe; MLP loses everything. A nonlinear head needs a
   much larger training set (2488 images here) or a teacher-student schedule; the
   caches only contain 4,512 unique images.
2. **Teacher features are frozen caches**; no new images can be encoded without the
   tower. Any larger distillation needs the re-download in §3-A.
3. **Pooling choice is untested**: mean over 576 tokens is one option; attention/cls
   pooling or the (collapsed) unit tokenizer were not usable. A properly trained
   tokenizer could sharpen the teacher signal.
4. **3 seeds = 3 data splits, not 3 weight inits** (linear head is closed-form, so
   that is the meaningful variance); the sd is therefore split variance.
5. **The video problem is data + interface, not modelling**: with the spec in §5.3 the
   smallest honest head is a mean-pool + linear temporal projection; the measurement
   gate is identical to this one.

---

## 9. Round 2 (2026-11-18 late): on-disk LLaVA teacher, 576×4096 signal matrix, pretrained ResNet base, HMDB51 temporal heads, GPU budget

Status of this round: CPU-only; gateway pid 17376 (:5097) and the 3 llama-servers were
never stopped (health 200 throughout); no compile.bat, no git; live model dirs were
read-only (the stale-tree weights were read in place). All numbers below are measured
on this box (Xeon E5-2680 v4, 2×14 cores @2.4 GHz, 128 GB RAM; GTX 1660 SUPER with
**5.78 / 6.14 GiB VRAM held by the live stack** → effectively no GPU).

### 9.1 Teacher weights provenance (on disk, no 7B LLM needed)

The crucial discovery: the **vision_tower + multi_modal_projector are already on disk**
in the stale August tree. The full 7B checkpoint is not needed (and was not used).

| Artifact | Path | Bytes | sha256 |
|---|---|---|---|
| CLIP ViT-L/14-336 vision tower (fp32, 391 tensors) | `E:\079\v6.0Alixander\runtime_store\models\multimodal_enc\llava-1.5-7b-hf_full\vision_tower\model.safetensors` | 1,214,077,616 | `4a0eefa6f9ab14c0062f65bf6ce4d4db90b844d7fd9b0bd9b1eae6b419797eeb` |
| multi_modal_projector (linear_1 1024→4096, linear_2 4096→4096, GELU, fp32) | `…\llava-1.5-7b-hf_full\multi_modal_projector.pt` | 83,921,513 | `d41e99a5957e375e5024408ef19d6e6c32b2c355bcbba5ba42f4e50dab62443c` |
| CLIP 336 preprocessor config | `…\llava-1.5-7b-hf_full\preprocessor_config.json` | 505 B | — |
| vision config (CLIPVisionModel, hidden 1024, 24 layers, patch 14, image 336) | `…\llava-1.5-7b-hf_full\vision_tower\config.json` | ~0.5 KB | — |

Model family/licence: LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`, licence `llama2`); the
vision tower is OpenAI CLIP ViT-L/14-336 (`quick_gelu`), the projector is the LLaVA
2-layer GELU MLP. Pipeline selected by `vision_feature_layer=-2` +
`vision_feature_select_strategy="default"` (drop CLS → 576 tokens).

Note: a full `hf-mirror.com` download of `model-00001-of-00003.safetensors`
(4,992,930,200 B, sha256 not needed) had already completed for a re-acquisition when
the Lead identified the on-disk weights; it is **not needed** and can be deleted
(`runtime_store/models/llava_1_5_7b/`, 4.65 GiB) once the team confirms.

### 9.2 Tool: `tools/llava_signal_matrix.py`

Loads the tower + projector (torch CPU, streaming safetensors; projector from `.pt`),
applies CLIP 336 preprocessing, and emits:

* **matrix** `[576, 4096]` (fp16 when stored) — the full token-level semantic signal;
* **mean-token** `[4096]` — the compatibility view the distillation head consumes.

Modes: batch/single extraction (`--paths-json/--image-dir`, `--save-matrices`),
`--validate` (mandatory cache check), `--l1` (single-frame read-outs), `--l2-extract`
/ `--l2-metrics` (temporal probes over HMDB51), `--cost` (CPU projection), and
`--head <contract>` (apply the distilled head, emit 128-d concepts). Deterministic
(eval mode, no grad); reports peak RSS.

Model in memory: tower 1.214 GB + projector 84 MB; **measured peak RSS 3.67–4.16 GB**
(batch 8, 16 threads). Matrix storage: 576×4096 fp16 = **4.5 MiB/frame** (vs 512 B for a
128-d concept); a T=8 clip matrix is 36 MiB.

### 9.3 Validation against the recorded caches (mandatory) — PASS

Ground truth: `runtime_store/models/large_vision_encoder/llava_features_more.npy`
(4000,576,4096) fp16 / `llava_more_paths.json`. `--validate --n 20`:

| metric | mean | p5 | min |
|---|---|---|---|
| **matrix cosine** (per-token cosine, averaged) | **1.000000** | 1.000000 | 1.000000 |
| **mean-token cosine** (4096 compat view) | **1.000000** | 0.99999994 | 0.99999988 |
| relative Frobenius error | 2.08e-04 | — | — |

**PASS (target ≥0.99).** `p5/min` are also 1.0/0.9999999. The 2e-4 Frobenius error is
the expected fp16-cache vs fp32-extraction quantization; the geometry is identical
(RSA Spearman matrix-vs-cached = 0.9999999985 on 64 images). Reproduced with the same
weights from two independent sources (stale-tree tower vs downloaded HF shard),
which also rules out a revision mismatch.

### 9.4 L1 — single-frame signal matrix vs the 128-d view (non-classification)

64 cached images (`--l1 --n 64`):

| read-out | value |
|---|---|
| matrix cosine to cached | 1.000000 |
| mean-token cosine to cached | 1.000000 |
| relative Frobenius error | 2.08e-04 |
| mean token L2 energy | 36.59 |
| effective rank of mean-token matrix (64 samples, 4096-d) | **23.95** |
| anisotropy (‖mean‖² / mean‖x‖²) | 0.907 |
| RSA (Spearman) matrix-pool vs cached-pool | **1.0000** |
| RSA (Spearman) matrix-pool (4096-d) vs **live 128-d concept** | **−0.0438** |

Interpretation: the 128-d live concept geometry is **essentially uncorrelated** with
the teacher's semantic geometry (RSA −0.04). The 4096-d matrix carries a semantic
structure the 128-d view does not have — consistent with §2 (the 128-d linear head
loses 9 pp top-1 / 29.5 pp probe vs the teacher) and with the user's "encoders lack
semantics" concern. The matrix's effective rank is ~24 for 64 samples (not collapsed,
but strongly anisotropic).

### 9.5 L2 — temporal matrix on HMDB51 (rollout + ARI/NMI probes)

120 clips (44 classes), 4 frames each; per-frame matrix computed transiently as
`[4,576,4096]`; stored per-frame mean-token 4096-d and the live 128-d concept.
Clip-disjoint 80/20 train/test for rollout; K-means probes for separability.

| probe | matrix 4096-d | live 128-d |
|---|---|---|
| class separability (KMeans k=classes): **NMI / ARI** | **0.7370 / 0.1053** | 0.7287 / 0.0968 |
| clip-identity from frames (k=100): **NMI / ARI** | 0.9511 / 0.5689 | **0.9561 / 0.6100** |
| next-frame rollout cosine (t→t+1) | **0.9918** | 0.9820 |
| rollout R² | 0.7334 | **0.8950** |
| identity baseline cosine (predict current = next) | 1.0000 | 1.0000 |
| rollout improvement over identity | −0.0082 | −0.0180 |

Interpretation (honest): on 4-frame HMDB51 clips the frames are so temporally
redundant that *identity* is the best one-step predictor; neither representation
beats it. The matrix slightly improves class-level NMI/ARI (+0.008 NMI) and rollout
cosine, but loses on clip-identity separability and rollout R² vs the cheap 128-d view,
at **67× the per-frame compute**. Verdict for the matrix path: **not justified for
inline temporal use on this hardware**; keep it as an offline/teacher signal and as
the honest upper bound for the 128-d view.

### 9.6 Deployment: the 4096→128 head is now pipeline-proven (runtime wiring pending)

`head_pipeline_proof.json` (32 held-out-of-training images): teacher means from the
on-disk extractor vs cached teacher → cosine **1.000000**; head applied to both →
**concept cosine 1.000000 (min 0.99999988)**. So the round-1 head
(`lin_ridge4096`, +6.83 pp top-1 / +6.85 pp top-5 / +6.95 pp probe) is now usable
end-to-end. Contract updated: `deployable=true`, `runtimeWired=false`,
`deploymentStatus="PIPELINE PROVEN / RUNTIME WIRING PENDING"`.

**Deployment command (one approved step later):**

```powershell
python tools/llava_signal_matrix.py --image-dir <dir> --out teacher.npz --batch 8 --threads 16 `
  --head runtime_store/models/io_train2/video_20261118_192034/image_distill/head_lin_ridge4096.contract.json
# -> teacher.npz (matrix/mean) + teacher.concept128.npz (128-d concepts + paths)
```

**Exact C++ seam** (production side; no product edits made):
`external_mixed_modal_io.cpp:956–974` — the Image/Video branch currently does
`videoEncoder(dim).encode(packet.payload, width, height, mimeType)` and expects a
128-d concept. Wiring the teacher path means either
1. a new `VideoModel` variant whose `encode()` shells out to
   `tools/llava_signal_matrix.py` (mirroring `tools/local_onnx_runner.py` used by
   `VideoLocalOnnxModel`), returning `concept128`; or
2. a first-class C++ loader for the LLaVA teacher (not recommended: 2 s/frame CPU,
   no GPU headroom).
Also required: an `image_resnet`-style config switch, because the factory picks the
variant from `jepa.image.variant` (`frontend_server.cpp:4554–4558`) while
`external_mixed_modal_io.cpp:105–134` **hardcodes** `"video-encoder"` for the bridge
cache — that hardcode must become config-driven for any variant swap to take effect.
CPU cost (below) means this must stay an **async/subprocess** path, not inline per bridge call.

### 9.7 Task B — pretrained ResNet50 base + light adapter (no encoder from scratch)

Frozen ImageNet base `timm/resnet50.a1_in1k` (apache-2.0, sha256
`773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb`), features 2048-d;
light adapter `Linear(2048→128)` trained with CE (+ a training-only linear classifier);
the **existing** 128→128 head and a live+adapter fusion were also tested. Data =
the recorded 8000-image live-concept cache (6000/2000 per split, 200 classes), 3 seeds.

| rung | 1-NN top-1 | 1-NN top-5 | ridge probe |
|---|---|---|---|
| baseline live 128-d | 0.3105 ± 0.0033 | 0.5482 ± 0.0047 | 0.4465 ± 0.0071 |
| **frozen R50 + linear adapter** | **0.5637 ± 0.0155** | **0.7437 ± 0.0150** | **0.6435 ± 0.0104** |
| adapter + existing 128→128 head | 0.5082 ± 0.0120 | 0.7013 ± 0.0059 | 0.6365 ± 0.0098 |
| fusion (live+adapter) + existing head | 0.4865 ± 0.0141 | 0.6877 ± 0.0043 | 0.6173 ± 0.0126 |

**Gate: PASS on all three (+25.3 pp top-1, +19.5 pp top-5, +19.7 pp probe) → verdict
DEPLOY** (this is the "starting from a base" win the user asked for: nearly doubles the
from-scratch 128-d top-1 on the same split). The existing 128→128 head *hurts* when
applied after the adapter (dimension/geometry mismatch) — ship the adapter alone.

Exported ONNX (`image_resnet_adapter/best.onnx`, 94,986,883 B): input
`pixel_values [batch,3,224,224]` (ImageNet mean/std), output `concept [batch,128,1,1]`,
L2-normalised; ONNX-vs-torch cosine **1.000000**; `model.manifest.json` declares the
I/O so `VideoLocalOnnxModel::loadManifest` can consume it. This is **resolver
compatible with no C++ change** once placed as
`runtime_store/models/additive_jepa/vision_resnet50_a1_in1k_adapter/best.onnx` (+ a
decoder or a decoder-dir split): set `jepa.image.variant` to that name for the
frontend image path; the bridge still needs the hardcoded-variant fix (§9.6/2).
Contract: `…/bases_20261118_201910_resnet50/image_resnet_adapter/contract.json`.

### 9.8 HMDB51 dataset + temporal head (previous task, completed)

Dataset provenance (verified live tree): **HMDB51**, 6,766 clips / 51 classes, from
`hf-mirror.com/datasets/innat/HMDB51` (`hmdb51_org.zip`, 2,123,212,891 B, sha256
`4F988A3DE7166CE7D9037FC74C4B41C71BBE4E21B2EE5993D5F7071020CE4FCE`, download 196.7 s,
extract 57 s, archive deleted after extraction; 2.01 GB kept as class/clip dirs).
Licence: HMDB51 research-use (Kuehne et al. 2011, Brown University); the mirror has no
licence file → verify before any redistribution. Note: the stale tree has **no**
video data; the 6,766 clips are in the live tree (acquired this round).
Protocol: 50 classes × 60 clips = 3,000 clips, 8 frames/clip, clip-disjoint 70/30 ×
3 seeds, 128-d live per-frame concepts; heads: mean-pool (baseline), attention-pool,
1-layer GRU; read-outs clip-level 1-NN top-1/top-5 + ridge probe (+ per-frame context).

| rung | top-1 | top-5 | probe |
|---|---|---|---|
| mean-pool baseline | 0.4381 | 0.6159 | 0.3770 |
| attention-pool | 0.4396 | 0.6148 | 0.3793 |
| GRU | 0.2581 | 0.4559 | 0.2804 |
| per-frame context | 0.4344 | 0.5513 | — |

**Verdict: NOT-DEPLOY** — attention-pool gains +0.15 pp top-1 / +0.22 pp probe but
loses 0.11 pp top-5 (gate requires all three), GRU collapses; mean pooling is as good
as anything learned on this 3,000-clip/8-frame budget. This is also the honest
statement that a temporal head has no runtime interface (see §5.1). Checkpoints:
`runtime_store/models/io_train2/video_20261118_195541/temporal/head_{attention,gru}.{npz,contract.json}`
(`phoenix.io_temporal_head.v1`, `deployable=false`, `requires` frame-list packet +
`VideoModel.encodeSequence`).

### 9.9 Task C — what a full encoder fine-tune would cost (GPU-hours)

Assumptions: Tiny-ImageNet-200 (100k train images @224, 200 classes, 90 epochs) as the
"small" base-adaptation target; the user's 10M-sample remote target as the "full"
case; a free mid-range GPU (GTX 1660 SUPER ~6 GB, or equivalent). The live box's GPU is
**not usable** (5.78/6.14 GiB held by the live stack).

| job | GPU-hours (est.) |
|---|---|
| ResNet18 fine-tune, 100k imgs × 90 epochs | **3–6 h** |
| ResNet50 fine-tune, same | **10–20 h** |
| ViT-B/16 fine-tune, same | **40–80 h** |
| ViT-L/14-336 fine-tune, same | **150–300 h** |
| ResNet50 from scratch, 10M samples × 90 epochs | **10,000–25,000 h on this class of GPU** (~1,000–2,500 h on an A100) |
| LLaVA-7B LoRA (vision-language, 100k pairs, 1 epoch) | **50–150 h on a 24 GB GPU**; full fine-tune 300–800 h |
| CPU-only equivalent | 50–100× the above → not viable |
| this round's frozen-base adapters (what we actually ran) | **0 GPU-h** (CPU, minutes) |

Conclusion: "start from a base" is 3–6 GPU-hours (ResNet18) or 10–20 GPU-hours
(ResNet50) for a real encoder adaptation — affordable on rented hardware; starting
from scratch (the current in-house encoders' problem) is a different order of magnitude
and was never going to be solved on this CPU box.

### 9.10 Files added in round 2

* `tools/llava_signal_matrix.py` — 576×4096 matrix tool (extract/validate/L1/L2/cost/head-apply)
* `tools/resnet_head_adapter.py` — frozen ResNet base + light adapter trainer/exporter
* `runtime_store/models/io_train2/bases_20261118_204528_llava_signal/` — evidence bundle
  (validation/L1/L2/cost/proof + updated head contract + `head_lin_ridge4096.f32`)
* `runtime_store/models/io_train2/bases_20261118_201910_resnet50/` — ResNet50 adapter
  run (features, adapter, ONNX, contract, manifest)
* `runtime_store/models/io_train2/video_20261118_195541/` — HMDB51 features + temporal heads
* `runtime_store/models/llava_1_5_7b/` — HF shard 1 download (4.65 GiB; not needed,
  candidate for deletion) and `runtime_store/models/resnet_bases/…` (44.6 + 97.7 MB)

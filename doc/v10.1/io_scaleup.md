# v10.1 IO scale-up: image head ladder + audio real objective

Status: executed 2026-11-17 ~22:55–23:03 (+08:00) on the live host, **no gateway/llama-server
restart, no `compile.bat`, no product source edits, no git, local data only**.
Builds on `doc/v10.1/io_model_inventory.md` and `doc/v10.1/io_training.md`.

> **Naming correction (user directive, 2026-11-17).** The shipped models are
> **llava / ResNet-modified in-house models, NOT the Meta I-JEPA series**. The
> `additive_jepa/` and `ijepa/` directory names are **legacy labels**.
> Identity of the artifacts used by this round:
>
> | Artifact | What it is |
> |---|---|
> | `runtime_store/models/additive_jepa/vision_encoder/best.onnx` (54.6 MB) | in-house 128-d vision concept encoder, `pixel_values [1,3,224,224] → concept [1,128]` |
> | `runtime_store/models/additive_jepa/speech_encoder/best.onnx` (4.9 MB) | in-house 1D speech/audio concept encoder, 16 kHz / 16 000 samples → 128-d |
> | `runtime_store/models/ijepa/resnet18_224/model_encoder.onnx` (32.1 MB) | in-house **ResNet18-modified** 512-d encoder, `pixel_values [1,3,224,224] → pooled [1,512]` (legacy dir name) |
> | `runtime_store/models/ijepa/resnet18_224/encoder_head.json` | the 512→128 linear head **of that ResNet18-modified encoder** |

- Tool: `tools/io_scaleup.py` (reuses the verified split/feature/metric helpers from
  `tools/io_train_head.py`; new tool only, no source changes)
- Run/checkpoints: `runtime_store/models/io_train2/20261117_2330/` (`20261117_2330` is a run label, not wall-clock; image manifest 23:00:12, audio manifest 23:02:43 +08:00)
- Log: `runtime_store/models/io_train2/20261117_2330_train.log`

## 1. Protocol (mandatory, applied identically to every rung)

- **Data**: tiny-imagenet-200 **train** split only, local. File-disjoint split exactly as
  the accepted `io_train` run (`seed=20261117`): 6 000 train images (30/class), 600 of them
  used as internal early-stop split, **2 000 held-out images (10/class, 200 classes)**.
  Held-out paths/labels were asserted equal to the cached feature set before reuse.
- **Features**: frozen encoder outputs (no encoder gradients anywhere).
- **Read-outs on held-out**: cosine **1-NN top-1**, **5-NN top-5**, and a closed-form
  **ridge linear probe** (λ=1e-2, fitted on train embeddings) — identical code for baseline
  and every rung/seed.
- **Seeds**: 3 training seeds (`20261117, 20261118, 20261119`) over the *same* held-out
  split → mean ± sd reported; **no single-seed claims**. Seeds vary head init and batch
  order only; the data split is fixed.
- **Baseline**: `normalize(encoder concept)` — exactly what `VideoLocalOnnxModel::encode()`
  returns today at `targetDim == conceptDim == 128` (live `jepa.image.conceptDim=128`).
- **Costs**: feature extraction is one-off and reused from the accepted run (2 000 held-out
  were encoded in the accepted run); head training per rung/seed is seconds (CPU, numpy).
  Training ran on CPU: `runtime_store/venvs/vision-cu111` has **onnxruntime 1.19.2 with only
  `AzureExecutionProvider` + `CPUExecutionProvider`** (verified) and no CUDA ORT provider
  DLL anywhere under `runtime_store/venvs`, so **no GPU/VRAM was used**.

## 2. Image ladder — results (held-out, 3 seeds mean ± sd)

| rung | head | params | train time / seed | 1-NN top-1 | 5-NN top-5 | ridge probe acc | probe MSE |
|---|---|---|---|---|---|---|---|
| **baseline** (no head) | — | 0 | — | **0.3210** | **0.5630** | **0.4985** | 0.004648 |
| a `a_linear128` | linear 128→128 (+bias, identity init) | **16 512** | 2.8 s | **0.3742 ± 0.0025** (+5.32 pp) | **0.6258 ± 0.0012** (+6.28 pp) | **0.5040 ± 0.0009** (+0.55 pp) | 0.004671 |
| b `b_mlp512` | linear 128→512 → ReLU → linear 512→128 | 131 712 | 6.6 s | 0.3918 ± 0.0038 (+7.08 pp) | 0.5470 ± 0.0062 (−1.60 pp) | 0.4057 ± 0.0059 (−9.28 pp) | 0.004638 |
| c `c_mlp1024_dropout` | linear 128→1024 → ReLU → dropout 0.2 → linear 1024→128 | 263 296 | 12.2 s | **0.4115 ± 0.0080** (+9.05 pp) | 0.5520 ± 0.0078 (−1.10 pp) | 0.4278 ± 0.0158 (−7.07 pp) | 0.004567 |

Held-out = 2 000 images (10/class, 200 classes), file-disjoint and identical for every
rung/seed; means ± sample sd over 3 training seeds. Per-seed numbers are in
`image/manifest.json` under `rungs[*].seeds`. All training was CPU-only (numpy); the
`vision-cu111` venv's onnxruntime exposes only `CPUExecutionProvider`, verified.

## 3. Chosen rung and justification

**Deployed/loaded rung: `a_linear128` (16 512 params).** The tool's automatic pick is
`c_mlp1024_dropout` because it maximises mean 1-NN top-1, but that rung is **rejected for
deployment**: going from linear to 2-layer MLP buys top-1 (+~2-4 pp over linear) while
**collapsing 5-NN top-5 by 7-8 pp and the ridge probe by 8-10 pp vs the linear rung** (and
both are below the no-head baseline). The runtime consumes the concept as a *global
vector* for cosine similarity, prototype fusion and cross-modal matching, so 5-NN density
and linear separability matter as much as top-1; the MLP re-shapes only the nearest-
neighbour identity. Justification of the rule: **choose the cheapest rung that improves
every read-out**, which is exactly the linear control.

Practical corollary (the real scale-up answer): **head capacity is not the bottleneck.**
A 16-263 k-param head saturates at ~+5-9 pp top-1 and cannot improve (and quickly harms)
the other two read-outs. The bottleneck is the frozen 128-d encoder and/or the data —
see §4. If a future consumer needs only 1-NN retrieval, rung (c) is available as an
optional checkpoint (`image/head_c_mlp1024_dropout.npz`), not as the default.

The deployment checkpoint therefore is
`image/head_a_linear128_deploy.{contract.json,f32,npz,json}` (best seed 20261119,
held-out 1-NN **0.3770**, 5-NN **0.6265**, probe **0.5050**), sha256 of the `.f32`
weights `3216baa2200cadfaef4087d62a68e842ddac4b358d9d9afde7f00934289afb2e`
(66 048 B). `deployment_choice.json` records the choice and the rejected alternatives.

## 4. Encoder-bottleneck check: rung (d) ResNet18-modified 512-d encoder

Rung (d) asks: if we simply **swap the encoder artifact** for the bigger in-house
ResNet18-modified 512-d encoder (`ijepa/resnet18_224/model_encoder.onnx`, legacy dir name),
does the concept get better? Same split, same protocol, encoder frozen: a linear
**512→128** head (65 664 params, 2.9 s/seed, 3 seeds) is trained on top of its 512-d
`pooled` output.

| encoder | head | params | 1-NN top-1 | 5-NN top-5 | ridge probe acc | probe MSE |
|---|---|---|---|---|---|---|
| in-house 128-d vision encoder (**live**) | linear 128→128 (rung a) | 16 512 | **0.3742 ± 0.0025** | **0.6258 ± 0.0012** | **0.5040 ± 0.0009** | 0.004671 |
| ResNet18-modified 512-d, raw (no head) | — | 0 | 0.0220 | 0.0765 | 0.0625 | 0.004978 |
| ResNet18-modified 512-d | linear 512→128 | 65 664 | 0.0275 ± 0.0013 | 0.0925 ± 0.0013 | 0.0612 ± 0.0010 | 0.004947 |

**Result: this artifact is NOT a viable encoder upgrade.** Its raw 512-d pooled features
retrieve at 2.2 % top-1 (near the 0.5 % chance of 200 classes), and a trained linear probe
reaches only 6.1 % — ~8× *worse* than the live 128-d encoder even after the same
classification objective. This is consistent with the earlier reconstruction probe: the
ResNet pair's decoder output is close to a mean image (`mse_vs_raw` 0.15-0.65) while the
live 128-d decoder reconstructs at 0.03-0.08. So the shipped `ijepa/resnet18_224/*`
checkpoint looks undertrained/non-semantic, and "swap to the bigger artifact" is not the
lever. A serious encoder scale-up means **training a new bigger in-house model** (§4.1).

### 4.1 How to swap to a bigger in-house llava/ResNet variant (budget)

**Hardware/data reality on this host.** GPU is a GTX 1660 SUPER (6 GB); the available
CUDA stack is `torch 1.10.2+cu111` in `runtime_store/venvs/vision-cu111`, but ONNX Runtime
is CPU-only (no CUDA provider DLL in any venv — verified), so all training/eval here is
CPU. Local image data is only tiny-imagenet-200 (100 k images at 64×64, upsampled to 224)
plus 5 toy PNGs; the LLaVA weights under `multimodal_enc_src/` are **partial** downloads
(shard 1 of 3 / 1 of 5, no index-complete checkpoint) and the in-house 128-d encoder ships
only as ONNX (its training source/weights are not in the tree). The existing upstream
pipeline that *can* train and export a ResNet-style encoder + projection head is
`.remote_ref/train_video.py` + `.remote_ref/export_runtime_models.py` (not part of the
live build, remote-GPU oriented).

| upgrade path | params | local data needed | train VRAM (AMP) | est. wall on 1660S 6 GB | feasible in 90 min? |
|---|---|---|---|---|---|
| head only (rungs a-c) | 16 k – 263 k | 6 k images | 0 | 3–12 s | **done** |
| ResNet18-modified fine-tune | 11.7 M | 100 k+ real 224 px images | ~2.5–3.5 GB (b128) | 11–33 h (20–30 ep) | no (overnight) |
| ResNet34/50 fine-tune | 21.8 / 25.6 M | 200 k+ real 224 px images | ~3.5–5 GB (b64) | 1–2 days | no |
| ViT-L/14-sized vision tower, LoRA | ~300 M | 1 M image-text pairs | ~8–14 GB (224 px, b32 + ckpt) | days; needs 24 GB GPU | no |
| in-house LLaVA-7B, QLoRA 4-bit | 7 B | 1 M pairs | ~10–12 GB | not on 6 GB; rent 24 GB | no |
| in-house LLaVA-7B, full FT | 7 B | 1 M+ pairs | ~80–120 GB | no | no |

**Recommended encoder plan if the user commits to bigger IO models** (not executed —
needs product-side training code, which is out of this member's edit scope):

1. Keep the linear head (rung a) as the shipped concept projection — it is cheap,
   reproducible and improves every read-out (contract in §7).
2. Collect ≥200 k **real 224 px** images from the running system (camera frames, uploads,
   screenshots) — upsampled tiny-imagenet is only a proxy and caps quality.
3. Extend `.remote_ref/train_video.py` from ResNet18 to ResNet34/50 (or a ViT-B/16 export),
   train 20–30 epochs with AMP on a 24 GB GPU (or overnight on the 1660S for ResNet18),
   keep the same 512→128-style projection head, export ONNX with
   `.remote_ref/export_runtime_models.py`.
4. Re-run `tools/io_scaleup.py --image-rungs a_linear128` with the new encoder artifact and
   require the same 3-seed, file-disjoint evidence before shipping.
5. Latency budget: the bridge currently spawns a python ONNX runner **per encode**; measured
   ~24-30 ms for the 128-d encoder and ~24 ms for the 512-d ResNet on CPU. A ResNet50-class
   encoder is ~60-100 ms and a ViT-L-class encoder ~0.3-0.5 s per image on CPU (estimates),
   which would dominate bridge latency → bigger encoders need ORT-CUDA or BPU
   quantisation (`.bin`) before deployment, not just a file swap.

## 5. Audio — real objective

**Label search first (no fabrication).** Local audio corpora searched:

- `runtime_store/datasets/musan_extracted/musan/speech/librivox/ANNOTATIONS` → **real
  per-file gender + language annotations** (172 files, 17 languages, 86 f / 86 m).
- music subdirectories → **real source/genre labels** (`fma`, `fma-western-art`,
  `hd-classical`, `jamendo`, `rfm`; each with an `ANNOTATIONS` file).
- noise → 2 subtypes only (`free-sound` 843, `sound-bible` 87).
- **No transcripts / ASR corpora exist locally** (searched `runtime_store/datasets/**`,
  `large_vision_encoder/data/**` for `*.trans.txt`, transcript/ASR files: zero hits;
  `librispeech_dummy/` contains only an empty HF cache marker). So a genuine audio-text
  contrastive objective is **not possible with local data** (minimum dataset in §5.3).

Two real objectives were run instead (labels derived from the corpus annotations):

### 5.1 Objective A — language ID (MUSAN librivox annotations)

Balanced 4 languages (english/german/spanish/french), 11 files/language, 12 × 1 s windows
per file, **file-disjoint 65/35** split → 336 train / **192 held-out windows**.
Chance = 25 %.

| rung | params | 1-NN top-1 | 5-NN top-5 | ridge probe acc | probe MSE |
|---|---|---|---|---|---|
| baseline (no head) | 0 | **0.2812** | **0.7292** | 0.2656 | 0.4003 |
| a linear 128→128 | 16 512 | 0.2535 ± 0.0108 | 0.6910 ± 0.0246 | 0.2483 ± 0.0030 | 0.3221 ± 0.0156 |
| b MLP 128→512→128 | 131 712 | 0.2361 ± 0.0060 | 0.5694 ± 0.0246 | 0.2622 ± 0.0305 | 0.3366 ± 0.0321 |

### 5.2 Objective B — music source/genre (MUSAN music sub-corpora)

5 real genres (`fma`, `fma-western-art`, `hd-classical`, `jamendo`, `rfm`), 75 files/genre
balanced, 12 × 1 s windows per file, **file-disjoint 70/30** split → 3 180 train /
**1 320 held-out windows**. Chance = 20 %.

| rung | params | 1-NN top-1 | 5-NN top-5 | ridge probe acc | probe MSE |
|---|---|---|---|---|---|
| baseline (no head) | 0 | 0.3455 | **0.7591** | **0.4379** | **0.14365** |
| a linear 128→128 | 16 512 | **0.3780 ± 0.0052** (+3.25 pp) | 0.7497 ± 0.0048 (−0.94 pp) | 0.4293 ± 0.0027 (−0.86 pp) | 0.14470 ± 0.00069 |
| b MLP 128→512→128 | 131 712 | 0.3742 ± 0.0050 | 0.7253 ± 0.0173 | 0.4311 ± 0.0145 | 0.15202 ± 0.00397 |

### 5.1 Audio verdict

**Audio verdict: keep the honest negative for language ID; music-genre is a real objective
but the head is still not the lever.**

- **Language ID (A) — NEGATIVE.** The frozen 128-d speech concept carries almost no
  linearly accessible language information at 1 s granularity: baseline 1-NN top-1 is
  28.1 % vs 25 % chance, and both heads are *below* baseline. Additional head capacity
  makes it worse (MLP top-5 collapses 72.9 % → 56.9 %). No head should be deployed for LID.
- **Music genre (B) — mixed, top-1 only.** The linear head raises 1-NN top-1 by +3.25 pp
  (0.3455 → 0.3780, 3-seed sd 0.0052) but 5-NN top-5 and the ridge probe are slightly
  *below* baseline, and the MLP is not better. This is the same pattern as the image
  ladder: a linear re-projection sharpens nearest-neighbour identity but does not add
  linearly separable information.
- **Root cause**: the audio encoder is an in-house 1D *concept/autoencoder* model trained
  for reconstruction, not a contrastive/ASR embedding. Heads trained on frozen features
  cannot invent language phonotactics. The real audio scale-up lever is the encoder
  objective (transcripts/contrastive alignment) and more data, not a bigger head.
- The music-genre linear head is checkpointed as **experimental only**
  (`audio_music/head_a_linear128.*`); it is **not** recommended for the runtime contract
  (its probe/top-5 regress).

### 5.2 Minimum dataset to collect for a *useful* audio head

Measured against what we have (not what would be nice):

| task | what we had | why it is not enough | minimum to collect |
|---|---|---|---|
| language ID | 11 files/language (≈15-30 min/language), 1 s windows | 1 s is too short; per-window LID is near chance; speaker/file correlation inflates | ≥ 20-50 h multilingual speech, **utterance-level ≥3 s windows**, ≥20 speakers/language, ≥200 utterances/language; e.g. Common Voice per-language subsets + held-out speaker split |
| speaker ID | MUSAN librivox has no reader ids (only gender/language per file) | labels absent | VoxCeleb1-style corpus (≥1 000 speakers, ≥100 utterances each, ≈350 h) or record ≥100 local speakers |
| music genre | 75 files/class × 5 genres (12 k windows) | only 5 sources, 1 s windows, track-correlated | ≥1 000 tracks/genre across ≥8 genres, **track-level** mean-pooled embeddings, artist-disjoint split |
| audio-text contrastive | **no transcripts locally at all** | cannot train `contrastiveAdapt()` | LibriSpeech train-clean-100 (100 h, 16 kHz, ~6 GB) + per-utterance text + a text concept encoder checkpoint |

For a *useful* improvement over the frozen audio encoder, plan on ≥50 h of labelled audio
(or 100 h transcribed speech) and a contrastive/ASR-style pretraining step; a head on top
of the current reconstruction encoder is not the lever.

### 5.3 Transcript/ASR objective — minimum dataset

To train the interface's intended audio-text alignment (`contrastiveAdapt()`,
`audio_model.hpp:109-113`) locally we would need at minimum:

- **≥ 20 h** of transcribed speech (≈ 100 h preferred), ideally 16 kHz mono, with
  per-utterance text (e.g., LibriSpeech `train-clean-100` = 100 h, already 16 kHz),
- a text concept encoder checkpoint that the audio branch can align to (the bridge's
  `transformerTextEncoderConcept()` path), or pair the audio encoder with a frozen
  text encoder;
- disk: LibriSpeech-100h ≈ 6 GB FLAC + transcripts; embeddings for 100 h at 1 s windows
  ≈ 3.6 M concepts × 128 × 4 B ≈ 1.8 GB (cacheable).
- This could not be fabricated from MUSAN: MUSAN has **no transcripts**.

## 6. Checkpoints, manifests and disk/VRAM budget

```
runtime_store/models/io_train2/20261117_2330/          # 3-seed run root
├── deployment_choice.json           # recommended rung + rejected alternatives + config keys
├── image/
│   ├── manifest.json                # 3-seed ladder (params, train time, per-seed + mean±sd)
│   ├── head_a_linear128_deploy.contract.json   # loader contract (§7) for the shipped head
│   ├── head_a_linear128_deploy.f32  # 66 048 B weights (W[128,128] then b[128], fp32-le)
│   ├── head_a_linear128_deploy.npz/.json       # self-describing checkpoint (same weights)
│   ├── head_c_mlp1024_dropout.npz/.json        # optional 1-NN-only rung (NOT deployed)
│   └── features_image.npz.{train,held}.npz
├── image_resnet512/                 # rung (d) encoder-swap experiment
│   ├── manifest.json, head_d_resnet512_linear128.npz/.json, features_resnet512.*
├── audio_lid/                       # real objective A (negative)
│   ├── manifest.json, head_a_linear128.npz/.json, features_lid.*
└── audio_music/                     # real objective B (experimental)
    ├── manifest.json, head_a_linear128.npz/.json, head_a_linear128.contract.json/.f32,
    └── features_music.*
```
Log: `runtime_store/models/io_train2/20261117_2330_train.log` (6.4 KB, full 3-seed stdout).
Run cost: 28.46 MiB / 27 files total (including the contract `.f32`/`.json` files);
feature caches dominate; the shipped image head is 66 KB of fp32 weights.

| artifact | sha256 |
|---|---|
| `image/head_a_linear128_deploy.f32` (66 048 B) | `3216baa2200cadfaef4087d62a68e842ddac4b358d9d9afde7f00934289afb2e` |
| `image/head_a_linear128_deploy.npz` | `39a659566fa6efae3ce7c9d4297b4889bb3fa797afed60285aaa205f264231ac` |
| `image/head_c_mlp1024_dropout.npz` (optional) | `ffefd6098435d58e117f701f07c198a7460552676d68d4856780ae2e47ddeed4` |
| `image_resnet512/head_d_resnet512_linear128.npz` | `dd370aaea17be63d89c3f8550ff9c617be363c5156796fd556d5cec9eb320302` |
| `audio_lid/head_a_linear128.npz` | `ec60a303028c1d8bb79b2ee0e1bc1783f99e4ee3f8604f1fff0f2806e4840902` |
| `audio_music/head_a_linear128.npz` (experimental) | `4212a178e256d023360a4da24ec59fe0c865c047ab8be84fc3abb874f810528e` |
| `audio_music/head_a_linear128.f32` (experimental) | `d21de08b4573334401f93c63368349dba240eb9d5050ae8e03a92a4368167c15` |

Every `manifest.json` records: tool/UTC timestamp, encoder path+bytes+sha256+identity+ONNX
I/O, dataset root/split counts, per-rung params + per-seed train time + per-seed metrics +
`agg.{mean,sd,per_seed}`, chosen rung, and the checkpoint sha256. Note: the image
`manifest.json`'s `chosen_rung` field is the tool's **top-1-only auto pick
(`c_mlp1024_dropout`)**; the **deployment** choice (rung a) is recorded separately in
`deployment_choice.json` per §3, so read that file for the shipped head.

Reproduce the whole run (~6-8 min wall on CPU; feature reuse avoids re-encoding):

```powershell
cd E:\079\v6.0Alixander\phoenix
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\io_scaleup.py `
  --epochs 30 --seeds 20261117,20261118,20261119 `
  --reuse-image-features runtime_store/models/io_train/20261117_2220/image `
  --run-name <new-timestamp>
```

## 7. LOADER CONTRACT (for the agent that writes the C++ side)

The C++ side does **not** exist yet (no product-source edits by this member). This is the
complete contract for the agent that writes it.

### 7.1 What the shipped head is

`image/head_a_linear128_deploy.*` — a **linear** map `R^128 → R^128`, applied to the *raw*
concept output of the in-house 128-d vision encoder, then L2-normalised by the existing
`normalizeVector()`. It is an accuracy layer only; it never touches the BPU `.bin` models
and does not change encoder weights.

### 7.2 Files

- `head_a_linear128_deploy.contract.json` — small metadata + sha256 + tensor specs
- `head_a_linear128_deploy.f32` — 66 048 bytes, weights in tensor order:
  `W` (128×128 floats, **row-major, out-major**: `W[o][i]`), then `b` (128 floats),
  little-endian float32.

Contract JSON (verbatim shape):

```json
{
  "format": "phoenix.io_head.v1",
  "modality": "image",
  "type": "linear",
  "inDim": 128,
  "outDim": 128,
  "normalize": true,
  "activation": "none",
  "weightFile": "head_a_linear128_deploy.f32",
  "weightDtype": "float32-le",
  "weightOrder": "rowmajor",
  "tensors": [{"name": "W", "shape": [128, 128], "order": "rowmajor"},
              {"name": "b", "shape": [128], "order": "rowmajor"}],
  "weightBytes": 66048,
  "weightSha256": "3216baa2...",
  "applyAfter": "raw encoder concept vector (before normalizeVector)",
  "inputShape": [128],
  "outputShape": [128],
  "referenceEval": {"heldout_top1": 0.377, "heldout_top5": 0.6265,
                    "heldout_probe_acc": 0.505, "three_seed_top1_mean": 0.3742,
                    "three_seed_top1_sd": 0.0025}
}
```

`type:"mlp"` is defined for future rungs: tensors `W1,b1,W2,b2` in order, ReLU between
layers, dropout inactive at inference; `normalize:true` means L2-normalise the final
vector. The current MLP rung is deliberately **not** shipped.

### 7.3 Config keys to add (default = baseline, no behaviour change)

| key | type | default | meaning |
|---|---|---|---|
| `jepa.image.headCheckpoint` | string | `""` | path to the contract JSON (relative to phoenix root). Empty ⇒ no head, current behaviour |
| `jepa.speech.headCheckpoint` | string | `""` | same for the speech/audio concept path |
| `jepa.image.headRequire` | bool | `false` | if true, a configured-but-unloadable head is a hard error; if false, log + fall back to baseline (recommended) |
| `jepa.speech.headRequire` | bool | `false` | same |
| `jepa.image.headSha256` / `jepa.speech.headSha256` | string | `""` | optional pin of the `.f32` sha256; mismatch ⇒ fall back |

Env overrides follow the existing `resolveConfig<T>` pattern in `phoenix_config.hpp`
(`JEPA_IMAGE_HEAD_CHECKPOINT`, `JEPA_SPEECH_HEAD_CHECKPOINT`).

Suggested enabled values:

```json
"jepa": { "image": { "headCheckpoint": "runtime_store/models/io_train2/20261117_2330/image/head_a_linear128_deploy.contract.json",
                     "headRequire": false }, ... }
```
Leave `jepa.speech.headCheckpoint` empty: no audio head is recommended (§5.1).

### 7.4 Call sites and tensor shapes

| site | where | expected input | expected output |
|---|---|---|---|
| `VideoLocalOnnxModel::encode` | `video_model.cpp:492-502`, after `runLocalOnnx()` returns `values` | `std::vector<float>` size 128 (raw encoder `concept`, pre-normalise) | size 128; then existing `normalizeVector(values)` |
| `VideoLocalOnnxModel::encodeContext/Target` | same path via `encode()` | 128 | 128 |
| `VideoHbdnnModel::encode` (when the BPU preprocessing stub is implemented) | `video_model.cpp:732-743` | 128 | 128 |
| `AudioLocalOnnxModel::encode` | `audio_model.cpp:1128-1138` | 128 (raw speech `concept`) | 128 |
| `AudioHbdnnModel::runEncoderBpu` | `audio_model.cpp:520-526` | 128 | 128 |

Order of operations must become: **encoder → head (128→128) → if `targetDim != 128`
`projectToDimension(values,targetDim,seed)` → `normalizeVector`**. Do not apply the head
after the hash projection, and do not apply it to already-normalised vectors.

### 7.5 Reference loader/apply code (drop-in, minimal)

```cpp
struct IoHead {
  std::vector<float> W, b;
  int inDim = 0, outDim = 0;
  bool loaded = false;
  std::string sha;
};

static bool loadIoHead(const std::filesystem::path& contractPath, IoHead& out) {
  std::ifstream in(contractPath);
  if (!in) return false;
  nlohmann::json j; try { in >> j; } catch (...) { return false; }
  if (j.value("format", std::string()) != "phoenix.io_head.v1") return false;
  if (j.value("type", std::string()) != "linear") return false;  // MLP: extend tensors
  out.inDim = j.value("inDim", 0);
  out.outDim = j.value("outDim", 0);
  if (out.inDim <= 0 || out.outDim <= 0) return false;
  const auto wf = contractPath.parent_path() / j.value("weightFile", std::string());
  std::ifstream w(wf, std::ios::binary);
  if (!w) return false;
  std::vector<float> buf(static_cast<size_t>(out.inDim) * out.outDim + out.outDim);
  w.read(reinterpret_cast<char*>(buf.data()),
         static_cast<std::streamsize>(buf.size() * sizeof(float)));
  if (!w) return false;
  out.W.assign(buf.begin(), buf.begin() + out.inDim * out.outDim);
  out.b.assign(buf.begin() + out.inDim * out.outDim, buf.end());
  out.loaded = true;
  return true;
}

inline void applyIoHead(const IoHead& h, std::vector<float>& v) {
  if (!h.loaded || static_cast<int>(v.size()) != h.inDim) return;
  std::vector<float> out(static_cast<size_t>(h.outDim));
  for (int r = 0; r < h.outDim; ++r) {
    float s = h.b[static_cast<size_t>(r)];
    const float* row = &h.W[static_cast<size_t>(r) * h.inDim];
    for (int c = 0; c < h.inDim; ++c) s += row[c] * v[static_cast<size_t>(c)];
    out[static_cast<size_t>(r)] = s;
  }
  v.swap(out);
}
```

Validation/error policy: verify `weightBytes` and (if configured) `weightSha256` before
use; on any mismatch, log `[io-head] <reason>; using baseline concept` and keep the
baseline (fail-open — the head is an accuracy optimization, never a safety gate). Expose
`headLoaded`, `headType`, `headSha256`, `headCheckpoint` in the existing
`VideoModel::status()` / `AudioModel::status()` JSON so `/api` diagnostics can confirm it.

### 7.6 Build/enable path

Rebuild through the sidecar flow only — `build\tmp\compile_next.bat` →
`phoenix_main.next.exe` (never `compile.bat` while the gateway runs; the Lead decides when
to promote the sidecar). Then enable via config keys and restart the gateway in a scheduled
window. The change is config + one loader helper; no new source file is strictly required
(the helper can live in `video_model.cpp`/`audio_model.cpp` or a small new header).

## 8. Honest limitations

1. **Head capacity saturates.** Linear → 512-hidden MLP → 1024-hidden+droput only raises
   1-NN top-1 (0.374 → 0.412) while 5-NN top-5 and the ridge probe fall **below the
   no-head baseline**. The MLP re-shapes nearest-neighbour identity, it does not make the
   concept more globally useful. Rung (a) is the only rung that improves every read-out.
2. **One dataset, one split.** 2 000 held-out images from tiny-imagenet train (200 classes,
   64×64 upsampled to 224); val/test splits untouched. Camera/UI images are a different
   distribution; no augmentation, no real 224 px data. 3 seeds vary only init/batch order
   on a **fixed split** — the ±sd understates split-to-split uncertainty.
3. **Rung (d) is a negative result about one artifact, not about encoders in general.**
   The shipped ResNet18-modified 512-d checkpoint is far weaker than the live 128-d
   encoder (top-1 2.2 % raw). No training code/weights for the in-house encoders exist in
   the tree (ONNX exports only), so a genuine encoder upgrade requires product-side
   training code and new data (§4.1) — out of this member's edit scope.
4. **Audio has no transcript corpus locally.** The LID objective is genuinely negative and
   the music-genre head is top-1-only; neither is recommended. MUSAN windows from one file
   are acoustically correlated even though the file split is disjoint, so the effective
   sample size is smaller than the window counts. Balanced subsets were used (11 files/
   language, 75 files/genre); most MUSAN files are not used.
5. **Nothing is wired into the runtime.** Deliverable is the checkpoint + manifest +
   loader contract; the live gateway still returns the baseline concept. No end-to-end
   quality or latency A/B was possible without a rebuild/restart (Lead-scheduled only).
6. **CPU training only.** `vision-cu111` onnxruntime exposes `AzureExecutionProvider` +
   `CPUExecutionProvider` only (verified); no CUDA ORT provider DLL exists in any venv.
   All VRAM/time figures for encoder upgrades in §4.1 are estimates (±2×), not measured.
7. **Fixed hyperparameters.** No per-rung tuning: linear lr 0.02, MLP lr 0.005, wd 1e-4,
   batch 256, 30 epochs, Adam+cosine, early stop on 10 % of train. A tuned MLP could shift
   the numbers, but the top-1-versus-global-geometry tradeoff is consistent across
   3 seeds and appears in both modalities.
8. **Auto-pick vs deployment pick.** `tools/io_scaleup.py` picks the rung by mean 1-NN
   top-1 only (`c_mlp1024_dropout`), while the deployment decision applies the
   "improve every read-out" guardrail (`a_linear128`). `deployment_choice.json` records the
   latter; do not read the manifest's `chosen_rung` as the deployment recommendation.
9. **Feature-cache provenance.** Image features were reused from the accepted
   `io_train/20261117_2220` run after asserting the cached paths/labels equal the
   deterministic split; the encoder sha256 is recorded in the manifests. Re-running
   `io_train_head.py` first is only needed if that cache is deleted.
10. **Shared host.** Runs executed while the gateway + 3 llama-servers were live; wall
    times are noisy but the metrics are deterministic given seeds.

### 8.1 Re-verify the deployed head without retraining (seconds)

```powershell
cd E:\079\v6.0Alixander\phoenix
runtime_store\venvs\vision-cu111\Scripts\python.exe -c @"
import json, numpy as np, os, sys; sys.path.insert(0,'tools')
import io_train_head as T
d='runtime_store/models/io_train2/20261117_2330/image'
z=np.load(d+'/head_a_linear128_deploy.npz'); W,b=z['W'],z['b']
tr=np.load(d+'/features_image.npz.train.npz'); te=np.load(d+'/features_image.npz.held.npz')
m=T.evaluate(T.l2_normalize(tr['x']@W.T+b),tr['y'],T.l2_normalize(te['x']@W.T+b),te['y'])
print({'top1':round(m['knn_top1'],4),'top5':round(m['knn_top5'],4),'probe':round(m['linear_probe_acc'],4)})
assert abs(m['knn_top1']-0.377)<1e-9
"@
```
Expected: `{'top1': 0.377, 'top5': 0.6265, 'probe': 0.505}` — this was executed and matches
`deployment_choice.json`.

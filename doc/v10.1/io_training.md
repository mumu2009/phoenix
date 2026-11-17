# v10.1 IO training loop (speech / image concept head)

Status: implemented and executed 2026-11-17 22:12–22:17 (+08:00) on the live host,
**without touching the live gateway, the 3 llama-servers, or `compile.bat`**.
Inventory that this document builds on: `doc/v10.1/io_model_inventory.md`.

> **Naming correction (user directive, 2026-11-17).** The shipped models are
> **llava / ResNet-modified in-house models, NOT the Meta I-JEPA series**. The
> `additive_jepa/` and `ijepa/` directory names are **legacy path labels**. In this
> document "encoder" means the in-house 128-d vision concept encoder /
> in-house 1D speech concept encoder (both under the legacy `additive_jepa/` prefix),
> and "resnet" means the in-house ResNet18-modified 512-d encoder under the legacy
> `ijepa/resnet18_224/` prefix. See inventory §0 for the per-checkpoint identity table.

- Trainer: `tools/io_train_head.py`
- Run/checkpoints: `runtime_store/models/io_train/20261117_2220/`
- Verification log: `runtime_store/models/io_train/20261117_2220_train.log`

## 1. Why a separate tracking line

The IO path already has two frozen, working ONNX autoencoders (image and speech,
see inventory). What it does **not** have is a place where a small learnable
projection can be trained and versioned independently of the agent runtime:

- the runtime only *reads* model files at factory time and keeps no training state,
- the shipped `ijepa/resnet18_224/encoder_head.json` (512→128, head of the in-house ResNet18-modified encoder) is an artifact of a
  pipeline the current source cannot load (inventory §0 finding 3/4),
- `x5_online_learner.cpp` proves the right *shape* of training (small SGD MLP over
  frozen features) but only runs on the X5 / from a C++ binary.

`tools/io_train_head.py` is that missing piece: a standalone python trainer that

1. reads the existing ONNX artifacts as **frozen feature extractors** (no downloads,
   no base-model re-training),
2. reads local corpora under `runtime_store/datasets/`,
3. trains one linear projection head per modality with a supervised linear-probe
   objective, and
4. writes versioned checkpoints + manifests under `runtime_store/models/io_train/`,
   leaving the live model directories strictly read-only.

## 2. Exact training command (executed)

```powershell
cd E:\079\v6.0Alixander\phoenix
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\io_train_head.py `
  --modality both `
  --train-per-class 30 --heldout-per-class 10 `
  --audio-files-per-class 300 --audio-windows 2 `
  --epochs 25 --run-name 20261117_2220
```

Wall clock: **4 min 37 s** total (image feature extraction 205 s + 53 s held-out;
head training seconds; audio features 1.8 s; audio head training seconds).
Run started 22:12:41, finished 22:17:18 (log mtimes). CPU-only ONNX Runtime
(8 intra-op threads, `CPUExecutionProvider`); the GTX 1660 was left to the running
llama-servers. The gateway and all 3 llama-servers were **not** stopped or restarted.

Smaller / other invocations:

```powershell
# image only, quick
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\io_train_head.py `
  --modality image --train-per-class 10 --heldout-per-class 5 --epochs 10

# audio only
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\io_train_head.py `
  --modality audio --audio-files-per-class 300 --epochs 25
```

Python environment requirements:

| package | required for | `runtime_store/venvs/vision-cu111` | `Python314` |
|---|---|---|---|
| numpy | everything | 1.23.5 | 2.5.4 |
| onnxruntime | everything | 1.19.2 | 1.31.1 |
| Pillow | `--modality image` only | 11.3.0 | **missing** |

The trainer reuses per-run feature caches (`features_image.npz.*.npz`,
`features_audio.*.npz`) so a re-run with the same `--run-name` skips re-encoding.
Determinism: fixed `--seed 20261117`, deterministic file selection, Adam with
cosine LR; ONNX inference itself is run-to-run stable here (CPU ExecutionProvider).

## 3. What data it actually used (local only, no downloads)

| modality | corpus | selection | train / internal-val / held-out | labels |
|---|---|---|---|---|
| image | `runtime_store/datasets/tiny-imagenet-200/train/` (100 000 imgs, 200 classes, 64×64 JPEG) | 30 + 10 images per class, files sorted then shuffled with seed; held-out is file-disjoint from train | **5 400 / 600 / 2 000** | 200 class ids (parent directory) |
| audio | `runtime_store/datasets/musan_extracted/musan/{speech,music,noise}/` (430/672/934 16 kHz WAVs) | 300 files per class, 2×1 s windows per file, 30 % of files held out (file-disjoint) | **1 106 / 123 / 522** windows | 3 coarse classes from directory layout |

Not used (and documented as unusable/stale): `phoenix/data/` (empty),
`phoenix/uploads/tmp/` (transient upload cache), workspace-root
`E:\079\v6.0Alixander\data/{audio,images}` (5 toy files, outside the live tree,
August copy). The 6.87 GB `musan_16k.tar.gz` tarball was **not** re-extracted;
the already-extracted tree was used. Tiny-ImageNet images are upsampled 64→224 to
match the encoder input.

Image preprocessing replicates `video_model.cpp:590-650`: decode → RGB → resize 224
bilinear → `/255` → ImageNet mean/std `(0.485,0.456,0.406)/(0.229,0.224,0.225)`.
Audio preprocessing replicates `audio_model.cpp:413-471`: 16 kHz mono, pad/truncate to
16 000 samples.

## 4. Head and baseline definition

```
baseline concept = normalize( encoder(x) )                      # what runtime returns today
trained  concept = normalize( W @ encoder(x) + b ),  W: 128x128  # trainer output
```
The head is a single linear layer (+bias), initialised to identity, followed by the
same L2 normalisation the runtime applies (`semantic_unit.cpp:244-259`). The classifier
used during training is a linear probe on the *normalised* concept; it is **discarded**
after training, only `W,b` are checkpointed. Loss = cross-entropy over the local labels,
Adam (0.9/0.999), cosine LR decay, weight decay 1e-4 on `W,b`, batch 256, 25 epochs,
early stopping on the 10 % internal train split (held-out never seen during training).

Baseline = exactly what `video_model.cpp VideoLocalOnnxModel::encode()` and
`audio_model.cpp AudioLocalOnnxModel::encode()` return today when
`targetDim == conceptDim == 128` (which is the live config `jepa.image.conceptDim=128`,
`config/phoenix.json:776`).

Read-out protocol (identical for baseline and trained, so the delta measures the head):
cosine k-NN (k=1, k=5) and a closed-form ridge linear probe (λ=1e-2) fitted on the
train features and evaluated on the held-out features.

## 5. Baseline vs trained — held-out results

### 5.1 Image (2 000 held-out images, 10/class, 200 classes)

| metric | baseline | trained | delta |
|---|---|---|---|
| cosine 1-NN top-1 | **0.3210** | **0.3785** | **+0.0575** (+5.75 pp) |
| cosine 5-NN top-5 | **0.5630** | **0.6230** | **+0.0600** (+6.00 pp) |
| ridge-probe accuracy | 0.4985 | 0.5035 | +0.0050 (+0.50 pp) |
| ridge-probe MSE | 0.004648 | 0.004671 | +0.000023 (slightly worse) |

Training curve: train CE 5.012 → 1.332; internal-val CE 4.550 → 2.289 with
internal-val accuracy 15.8 % → 45.8 % (200-way, 5 400 train samples); best epoch 24 of 25.
**Conclusion: a clear retrieval win (+5.75 pp top-1, +6.0 pp top-5), probe accuracy
marginally up; the head is a genuine but modest improvement and is safe to iterate on.**

### 5.2 Audio (522 held-out windows, 3 classes)

| metric | baseline | trained | delta |
|---|---|---|---|
| cosine 1-NN top-1 | 0.6379 | 0.6130 | **−0.0249** |
| cosine 5-NN top-5 | 0.9349 | 0.8908 | **−0.0441** |
| ridge-probe accuracy | 0.6916 | 0.6877 | −0.0038 |
| ridge-probe MSE | 0.16259 | 0.16563 | +0.00303 |

**Honest negative result: do not ship the audio head.** The 3-class MUSAN proxy task is
too easy and already almost linearly separable in the frozen 128-d in-house speech-concept space;
the extra 128×128 layer (1106 train windows) overfits (train CE 1.049 → 0.596 while
internal-val CE improves only 1.028 → 0.825) and loses a little held-out retrieval.
The audio *loop* itself is functional and reproducible; the audio *objective* needs a
harder task (transcripts / text-concept contrastive alignment, speaker-id, more windows)
before a head is worth checkpointing.

## 6. Checkpoints and manifest

```
runtime_store/models/io_train/20261117_2220/
├── 20261117_2220_train.log        (full training log, 16 KB)
├── image/
│   ├── manifest.json              (6.6 KB: artifact sha256, data counts, hyperparams, loss curve, metrics)
│   ├── head.npz                   (W[128,128], b[128], inDim, outDim, normalize=1)
│   ├── head.json                  (inDim/outDim/W/b, same keys as encoder_head.json)
│   └── features_image.npz.train.npz / .held.npz   (reproducible feature caches)
└── audio/
    ├── manifest.json              (7.0 KB)
    ├── head.npz
    ├── head.json
    └── features_audio.train.npz / .held.npz
```

| modality | head.npz sha256 | dims | classes |
|---|---|---|---|
| image | `5ce66735c4b065eaa647abf4b25fafceff3eb80230857eeb4d37438af1e3415a` | 128 → 128 | 200 |
| audio | `1d536eae87a75f1f17ede390345b476f77804f93f41b6f7a7bcd2d0b2cfca444` | 128 → 128 | 3 |

`manifest.json` contains: tool/command/cwd/UTC timestamp, python/numpy/onnxruntime/
Pillow/platform versions, encoder path + size + **sha256** + manifest sha, dataset root
and exact split counts, hyperparameters (incl. selected early-stop epoch), the full
per-epoch loss curve, and `heldout_metrics.{baseline,trained,delta}`. The trainer never
writes outside `runtime_store/models/io_train/`.

`head.json` deliberately keeps the legacy `encoder_head.json` keys
(`inDim`, `outDim`, `W` row-major, `b`) so a future loader in `video_model.cpp` /
`audio_model.cpp` can reuse the old parsing shape.

## 7. How the head is (and is not) consumed by `video_model` / `audio_model`

**Today: it is not consumed.** The runtime has no loader for these checkpoints
(verified: `encoder_head` is not referenced by any current source file, see inventory
§0). The live path therefore remains the baseline, byte-for-byte unchanged.

The intended insertion is one small, local change per modality — same place both
factories already normalise the concept:

```cpp
// video_model.cpp, VideoLocalOnnxModel::encode() - after runLocalOnnx() -> `values`,
// before normalizeVector(values) (video_model.cpp:492-502)
if (headDim_ == (int)values.size()) {                 // 128 == 128
  std::vector<float> projected(headDim_);
  for (int o = 0; o < headDim_; ++o) {
    float s = headBias_[o];
    for (int i = 0; i < headDim_; ++i) s += headW_[o * headDim_ + i] * values[i];
    projected[o] = s;
  }
  values = std::move(projected);
}
return phoenix::multimodal::normalizeVector(values);
```

The audio equivalent is `audio_model.cpp:1128-1139` (`AudioLocalOnnxModel::encode`),
inserting the same linear map before `projectToDimension`/`normalizeVector`.
When `targetDim != conceptDim` (bridge default `targetDim=64`,
`main_hub_parts/116_section_tail.inc:5095`), the 128→128 head should be applied first
and the existing hash projection `projectToDimension(..., 64)` kept afterwards —
the checkpoint is not a 64-d replacement yet.

Wiring would require (not done, out of scope for this round): a ~30-line C++ loader
(nlohmann JSON is already a dependency, so `head.json` can be parsed directly), a new
config key such as `jepa.image.headCheckpoint` / `jepa.speech.headCheckpoint`, and a
rebuild via the sidecar flow (`build\tmp\compile_next.bat` → `phoenix_main.next.exe`);
`compile.bat` must not be used while the live gateway runs (Lead owns that decision).

## 8. Honest limitation list

1. **Not wired into the runtime.** The trained heads are checkpoints only; the live
   gateway still returns the baseline concept. No config key, no loader, no hot reload.
2. **Proxy objective, not the runtime task.** Training optimises 200-way
   Tiny-ImageNet classification (image) / 3-way MUSAN classification (audio). Retrieval
   and probe deltas are evidence of concept-space quality, not proof that downstream
   agent behaviour improves.
3. **Domain shift (image).** Tiny-ImageNet is 64×64, single-object, web-crawled;
   upsampled to 224. Camera frames / UI images may behave differently. No augmentation
   and no calibration set was used.
4. **Baseline dimension mismatch.** The comparison is at 128-d (encoder concept dim,
   live `jepa.image.conceptDim=128`), not the 64-d hash projection the bridge uses by
   default. A 128→64 variant is future work.
5. **Audio head is a negative result** (§5.2) and must not be deployed. The coarse
   speech/music/noise labels are a proxy; no transcripts were available locally, so the
   interface's intended `contrastiveAdapt()` audio-text objective was not trained.
6. **Frozen base models.** No encoder/decoder fine-tuning, no BPU (`.bin`) update:
   Horizon `.bin` models are immutable without `hb_mapper` off-board, and the host has
   no hbDNN. The head is a CPU-side layer, consistent with the live
   `model_deployment.{vision,speech}.localBackend="cpu"`.
7. **Single seed, no error bars.** Deterministic split/seed, one run; no confidence
   intervals, no cross-validation. Internal early-stop split is 10 % of train; the
   held-out split was never used for tuning.
8. **Small head capacity.** Linear only (16 385 params). A 2-layer MLP or a head trained
   jointly with a linear probe on more data may do better; deliberately kept small to
   stay in the `encoder_head.json` / `x5_online_learner` spirit.
9. **Timing is host-load dependent**, not a benchmark: feature extraction ran while the
   gateway + 3 llama-servers were live (205 s / 6 000 images image, 1.8 s / 1 229 audio).
10. **Dead pipelines untouched.** `resnet18_224`, `encoder_head.json`,
    `model_encoder_head.onnx`, the LLaVA/Qwen service and its 7B weights were left as-is
    (documented dead in the inventory); this round does not attempt to resurrect them.

## 9. Reproduce and verify

```powershell
cd E:\079\v6.0Alixander\phoenix

# 1) reproduce the run (4-5 min; overwrite-safe: use a new --run-name)
runtime_store\venvs\vision-cu111\Scripts\python.exe tools\io_train_head.py `
  --modality both --train-per-class 30 --heldout-per-class 10 `
  --audio-files-per-class 300 --audio-windows 2 --epochs 25 --run-name reproduce

# 2) re-verify the checkpoint against the manifest from the saved features (seconds)
runtime_store\venvs\vision-cu111\Scripts\python.exe -c @"
import json, numpy as np, os, sys
root = os.getcwd(); sys.path.insert(0, os.path.join(root, 'tools'))
import io_train_head as T
for mod in ('image','audio'):
    d = os.path.join(root, 'runtime_store/models/io_train/20261117_2220', mod)
    m = json.load(open(os.path.join(d,'manifest.json')))
    h = np.load(os.path.join(d,'head.npz')); W,b = h['W'], h['b']
    tr = np.load(os.path.join(d,'features_%s.npz.train.npz'%mod))
    te = np.load(os.path.join(d,'features_%s.npz.held.npz'%mod))
    Xtr,ytr,Xte,yte = tr['x'],tr['y'],te['x'],te['y']
    base = T.evaluate(T.l2_normalize(Xtr), ytr, T.l2_normalize(Xte), yte)
    trnd = T.evaluate(T.l2_normalize(Xtr@W.T+b), ytr, T.l2_normalize(Xte@W.T+b), yte)
    print(mod, 'baseline', base); print(mod, 'trained ', trnd)
    assert m['heldout_metrics']['trained']['knn_top1'] == trnd['knn_top1']
"@
```
This re-verification was executed against the shipped run: image re-derived
`baseline.knn_top1=0.3210 / trained.knn_top1=0.3785`, audio
`0.6379 / 0.6130`, matching `manifest.json` exactly.

## 10. Suggested next steps (not done)

1. Add the JSON head loader + `jepa.*.headCheckpoint` config keys and rebuild through
   the sidecar binary; gate with `targetDim == 128` so the 64-d hash path is unchanged.
2. Train a 128→64 head to replace `projectToDimension(..., 0x57494D47)` at the bridge's
   default dimension; compare end-to-end bridge retrieval.
3. Give the audio line a real objective: use the available text side
   (`transformerTextEncoderConcept`, `external_mixed_modal_io.cpp:1035`) to build
   audio-text contrastive pairs from local transcripts, or speaker-id from MUSAN
   `librivox`/`us-gov` subdirectories; keep the 3-class run as a regression floor.
4. Add image augmentation + a camera-frame calibration set when such frames are
   captured (do not fabricate a source); revisit the 64×64→224 shift.
5. Once a head is wired, add an end-to-end A/B on the same held-out images comparing
   `/api` image-bridge concepts before/after (needs a gateway restart, so schedule it
   with the Lead, not in this round).

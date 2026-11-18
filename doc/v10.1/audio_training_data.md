# v10.1 Real-data audio training (Fluent Speech Commands)

Status: executed 2026-11-18 ~19:11–19:55 (+08:00) on the live host.
Constraints honoured: gateway pid 17376 (:5097) and the 3 llama-servers were **not**
stopped; `compile.bat` not run; no git branch/commit/push; live model dirs read-only;
training ran on **CPU** (GPU was at 85–99 % with 5.9/6.1 GB VRAM used by the live stack,
and the available onnxruntime is CPU-only anyway).

- Dataset tool: `tools/audio_train_head.py`
- Run/checkpoints: `runtime_store/models/io_train2/audio_<ts>/`
- Dataset: `runtime_store/datasets/audio/fluent_speech_commands/`

> **Naming correction (user directive, 2026-11-17).** The frozen encoder is an
> **in-house llava/ResNet-era model**, NOT Meta I-JEPA. `additive_jepa/` is a **legacy
> directory name**. The checkpoint used here is
> `runtime_store/models/additive_jepa/speech_encoder/best.onnx`:
> `waveform [batch,1,1,16000] → concept [batch,128]`, 16 kHz mono, 1 s fixed input
> (model manifest in the same directory; sha256 recorded in the run manifest).

## 1. Data acquisition — network reachability first (10 s timeouts)

| host | result | notes |
|---|---|---|
| `huggingface.co` | **UNREACHABLE** (curl exit 28, timeout after 10 s) | blocked from this network |
| `hf-mirror.com` | reachable (HTTP 200 in 1.5 s) | HF mirror works, but `google/speech_commands` there is only a loader script (no audio files); LFS repos are parquet wrappers, and the API rate-limits aggressively |
| `modelscope.cn` (魔塔社区) | reachable (HTTP 302 in 0.3 s) | **chosen source**; LFS throughput measured **~3.7 MB/s** (2 MB range test: 3.67 MB/s; 4 MB test: 2.9–3.7 MB/s) |
| `openslr.org` | reachable (HTTP 200 in 2.1 s) | ~64 KB/s from here — too slow for this budget; holds LibriSpeech (337 MB dev-clean) as fallback |
| `download.tensorflow.org` | HTTP reachable, **HTTPS broken** (TLS cert hostname mismatch)` | 2.43 GB SpeechCommands tar; measured throughput only ~18–65 KB/s (would take >12 h) → rejected |
| `storage.googleapis.com` | HTTPS reachable | same 2.43 GB object; also slow |

Decision: **Fluent Speech Commands (FSC)** from ModelScope
`OmniData/Fluent_Speech_Commands` — a real spoken-language-understanding corpus with
per-utterance **intent + action/object/location** labels and **speaker ids**
(97 speakers, 30 043 utterances by design; the 1.54 GB raw archive is the wav bundle).
Rejected along the way: Google SpeechCommands v0.02 (2.43 GB, throttled to ~20 KB/s),
`DynamicSuperb/SpeechCommandRecognition_GoogleSpeechCommandsV1` (test parquet only,
5.7 MB, no train split), `modelscope/aishell1_subset` (labels csv only, audio hosted
elsewhere), LibriSpeech dev-clean (OpenSLR 64 KB/s; no word-level labels).

### 1.1 Provenance (recorded in `provenance.json` next to the data)

| field | value |
|---|---|
| dataset | Fluent Speech Commands (FSC) |
| publisher / paper | University of Montreal, Lugosch et al. 2019 (`arXiv:1904.03670`) |
| source platform | ModelScope (魔塔社区), repo `OmniData/Fluent_Speech_Commands` |
| source file | `raw/jf8398hf30f0381738rucj3828chfdnchs.tar.gz` (repo path) |
| download URL | `https://modelscope.cn/api/v1/datasets/OmniData/Fluent_Speech_Commands/repo?Revision=master&FilePath=raw/jf8398hf30f0381738rucj3828chfdnchs.tar.gz` |
| archive size | **1 544 093 324 bytes** (1.54 GB) |
| archive sha256 | **`4376699f7daf134a9fa57a1d880ffcaaf94a3e2551ba0b40ad894d7abb71aacb`** |
| download | started 2026-11-18 19:18:29 +08:00, **344.8 s**, ~4.5 MB/s |
| licence | **CC BY-NC 4.0** (ModelScope dataset card; non-commercial research use) |
| language / rate | English, 16 kHz mono, 16-bit PCM |
| labels | per-utterance `action`, `object`, `location`, intent(=tuple), `transcription`, `speakerId` |
| extraction | 100.3 s → 30 049 files / **30 043 wavs** / 2.07 GB |
| archive handling | **deleted after verified extraction** (duplicate removal, 2026-11-18) |
| stored layout | `runtime_store/datasets/audio/fluent_speech_commands/{provenance.json, extracted/fluent_speech_commands_dataset/{data/*.csv, wavs/speakers/<speakerId>/*.wav}}` |

Rejected candidates and why (all tested before choosing): Google SpeechCommands v0.02
(2.43 GB official tar reachable only over throttled HTTP/HTTPS path: measured ~18–65 KB/s
→ 12 h+; HF mirror hosts only a loader script), `DynamicSuperb/SpeechCommandRecognition_
GoogleSpeechCommandsV1` (test parquet only, 5.7 MB, no train split),
`modelscope/aishell1_subset` (CSV labels only, audio not in repo), LibriSpeech dev-clean
(OpenSLR reachable but ~64 KB/s; no word-level labels).

## 2. Dataset layout, labels and splits

Real data layout after extraction (validated by script before training):

- **30 043 utterances**, 97 speakers, 31 intent classes (`action|object|location` tuple:
  e.g. `activate|lights|kitchen`, `increase|volume|none`), average ~1–4 s each.
- Official FSC **speaker-independent** splits from `data/*.csv`:
  **train 23 132 (77 speakers), valid 3 118 (10 speakers), test 3 793 (10 speakers)**;
  train↔test speaker overlap = **0**, train↔valid overlap = **0** (verified).
- Held-out = official test split (3 793 ≥ 1 000 utterances, speaker-disjoint). Internal
  early stop = official valid split. No utterance/window is shared between splits.
- Label distribution is imbalanced (largest intent in train: `increase|volume|none`,
  2 296 utterances); per-class train cap = 200 → 6 200 training utterances.
- Secondary real-labelled goals derived from the same corpus: `action` (5 values),
  `object` (10 values), `location` (6 values) per-utterance fields, and `speakerId`.
  These were used to test whether the frozen encoder carries *any* label information
  (see §4) instead of only one arbitrary objective.

`provenance.json` in the dataset root records all of the above, including the archive
sha256, download timing, licence and reachability test results. The archive itself is
deleted; only the extracted tree is kept.

## 3. Objective and fixed-1 s encoder handling

The only trainable part is a small head on top of the **frozen** in-house 1 s speech
concept encoder; base weights are never modified. FSC utterances are 1–4 s long, while
the encoder input is fixed at 16 000 samples, so the feature for one utterance is the
**mean-pool of N evenly spaced 1 s windows** through the frozen encoder
(`--crops 3`; N=1 degrades to a centre crop). This is an explicit, documented
compromise — the label refers to the whole utterance, but each window sees only part
of it.

Objective: **intent classification** (labels from `intent_class`; fallback composite
`action|object|location`), trained with cross-entropy on the frozen features.
Head ladder (cheap → expensive, same as the image work):
`a_linear128` (128→128), `b_mlp512` (128→512→128 ReLU), `c_mlp1024_dropout`
(128→1024→128 ReLU + dropout 0.2).

Protocol: 3 training seeds (`20261118,20261119,20261120`) on a fixed split →
**mean ± sd**; read-outs are cosine **1-NN top-1**, **5-NN top-5** and a closed-form
**ridge linear probe** (λ=1e-2) fitted on train features and evaluated on held-out.
Held-out = official FSC **test** split (speaker-independent by design; speaker overlap
asserted zero). Internal early stop = official **valid** split. Windows are never
split across train/held-out (utterance-level split).

**Deployability guardrail** (the image lesson): a rung is deployable only if the 3-seed
mean improves top-1 **and** does not regress top-5 or probe by more than 0.1 pp.

## 4. Results (held-out, 3 seeds mean ± sd)

All metrics are on the official **speaker-independent test split (3 793 utterances,
10 unseen speakers)**; mean ± sample sd over 3 training seeds; 20 epochs, batch 256,
linear lr 0.02 / MLP lr 0.005, 3-crop mean-pooled frozen concepts.
The frozen-encoder baseline (identity, no head) is shown for reference; a head is only
useful if it beats it.

### 4.1 Intent (31-way; chance 3.23 %, majority-class 12.39 %)

| read-out | baseline | a linear 128→128 (16 512 p) | b MLP 128→512→128 (131 712 p) | c MLP 128→1024→128 (263 296 p) |
|---|---|---|---|---|
| 1-NN top-1 | 0.0287 | **0.0339 ± 0.0031** | 0.0294 ± 0.0024 | 0.0297 ± 0.0012 |
| 5-NN top-5 | 0.1434 | **0.1559 ± 0.0030** | 0.1425 ± 0.0054 | 0.1502 ± 0.0060 |
| ridge probe acc | 0.0345 | **0.0378 ± 0.0007** | 0.0345 ± 0.0012 | 0.0377 ± 0.0023 |

The linear head passes the *relative* guardrail (all three tiny gains, +0.5 pp top-1) but
**top-1 is at chance**: 3.39 % vs 3.23 % chance and 12.4 % majority class ⇒ fails the
"beat trivial" rule (`≥ max(2× chance, majority+5 pp) = 17.4 %`). **NOT DEPLOY.**

### 4.2 Action (6-way; chance 16.67 %, majority-class 24.76 %)

| read-out | baseline | a linear | b MLP512 | c MLP1024 |
|---|---|---|---|---|
| 1-NN top-1 | **0.1751** | 0.1680 ± 0.0019 | 0.1701 ± 0.0099 | 0.1719 ± 0.0046 |
| 5-NN top-5 | **0.6230** | 0.5928 ± 0.0185 | 0.5941 ± 0.0183 | 0.5893 ± 0.0094 |
| ridge probe acc | **0.1766** | 0.1709 ± 0.0019 | 0.1823 ± 0.0078 | 0.1723 ± 0.0055 |

Baseline is already ≈ chance; every head is at/below it on top-1 and top-5.
**NOT DEPLOY.**

### 4.3 Object (14-way; chance 7.14 %, majority-class 30.32 %)

| read-out | baseline | a linear | b MLP512 | c MLP1024 |
|---|---|---|---|---|
| 1-NN top-1 | **0.0902** | 0.0905 ± 0.0020 | 0.0890 ± 0.0026 | 0.0856 ± 0.0023 |
| 5-NN top-5 | **0.3775** | 0.3567 ± 0.0095 | 0.3594 ± 0.0092 | 0.3487 ± 0.0048 |
| ridge probe acc | **0.1226** | 0.1196 ± 0.0008 | 0.1089 ± 0.0035 | 0.1126 ± 0.0044 |

Top-1 is flat/chance, top-5 and probe regress ⇒ **NOT DEPLOY.**

**Diagnosis.** The frozen 1 s concept encoder is a *category/timbre* encoder, not a
semantic/linguistic one: on the earlier MUSAN 3-class task it separates
speech/music/noise well (63–69 %), but on FSC it cannot separate intents, actions or
objects (all at chance). Per-dimension feature std is healthy (0.08–0.14) and crops of
the same utterance agree (cosine 0.73), so this is **not an extraction bug** — the
information simply is not linearly (or 5-NN) accessible in the 128-d concept. Heads
trained on frozen features cannot create it.

### 4.4 Denoise / reconstruction in concept space (objective b)

Setup: clean FSC 1 s windows (3-crop mean-pooled) vs the same windows mixed with local
**MUSAN noise** at a given SNR; head `H: 128→128` trained with
`loss = 1 − cos(H(z_noisy), z_clean)`; evaluated by paired retrieval (retrieve the clean
counterpart among 3 793 held-out utterances) and a ridge probe `H(z_noisy) → z_clean`.

Held-out = 3 793 FSC test utterances (10 unseen speakers). 3 seeds mean ± sd.

**10 dB SNR (mild noise):**

| read-out | baseline (identity) | a linear 128→128 | b MLP 128→512→128 |
|---|---|---|---|
| paired retrieval top-1 | **0.9966** | 0.9945 ± 0.0000 | 0.9823 ± 0.0006 |
| paired retrieval top-5 | **0.9992** | 0.9977 ± 0.0002 | 0.9911 ± 0.0006 |
| concept cosine | 0.9931 | **0.9953 ± 0.0000** | 0.9923 ± 0.0001 |
| ridge probe MSE | 0.00007 | 0.00007 | 0.00010 |
| ridge probe cosine | 0.9954 | 0.9954 | 0.9933 |

**0 dB SNR (noise as loud as speech):**

| read-out | baseline (identity) | a linear 128→128 | b MLP 128→512→128 |
|---|---|---|---|
| paired retrieval top-1 | **0.6612** | 0.5715 ± 0.0052 | 0.3780 ± 0.0101 |
| paired retrieval top-5 | **0.7701** | 0.6831 ± 0.0048 | 0.5132 ± 0.0128 |
| concept cosine | 0.9227 | **0.9741 ± 0.0001** | 0.9726 ± 0.0001 |
| ridge probe MSE | 0.00040 | 0.00039 ± 0.00000 | 0.00041 |
| ridge probe cosine | 0.9743 | **0.9745 ± 0.0000** | 0.9730 |

Reading: at 10 dB the frozen encoder is already essentially noise-invariant
(cosine 0.993, paired top-1 99.66 %) — there is nothing to gain. At 0 dB the linear head
*does* denoise in embedding space (cosine 0.923 → 0.974), **but it destroys instance
identity**: paired retrieval drops 66.1 % → 57.1 % and top-5 77.0 % → 68.3 %. A concept
that is clean on average but no longer identifies the utterance is **NOT DEPLOYABLE** for
the IO path (fusion/retrieval needs identity).

## 5. Verdict

# VERDICT: **NOT DEPLOY** — no audio projection head on the current frozen encoder.

The numbers that decide it (all on 3 793 speaker-disjoint held-out utterances, 3 seeds):

| objective | baseline | best trained head | decision rule | result |
|---|---|---|---|---|
| intent (31-way) | top-1 2.87 % (chance 3.23 %, majority 12.39 %) | 3.39 ± 0.31 % | must beat `max(2×chance, majority+5pp)=17.4 %` | **fail** — at chance |
| action (6-way) | top-1 17.51 %, top-5 62.30 % | 17.19 %, 58.93 % | top-1 must improve, top-5/probe must not regress | **fail** (all regress) |
| object (14-way) | top-1 9.02 %, top-5 37.75 %, probe 12.26 % | 9.05 %, 35.67 %, 11.96 % | same | **fail** (top-5/probe regress) |
| denoise 10 dB | paired top-1 99.66 % | 99.45 % | must beat identity baseline | **fail** (nothing to gain) |
| denoise 0 dB | paired top-1 66.12 %, cos 0.923 | 57.15 %, cos 0.974 | retrieval must not regress | **fail** (identity destroyed) |

**Root cause.** The frozen encoder is a 1 s *category/autoencoder* concept model (it
separates speech/music/noise well — the earlier MUSAN baseline was 63–69 % — but carries
no linearly/5-NN-accessible intent or lexical information, and at mild noise it is already
invariant). Heads on frozen features cannot add semantic content; the denoise head that
buys embedding cosine pays for it with instance identity. Therefore the lever is the
**encoder objective/architecture**, not a bigger head:

1. keep the current behaviour (baseline concept) — no regression;
2. train a new audio encoder with a contrastive/ASR objective. Local path: FSC already
   ships **transcripts** (`data/train_data.csv: transcription`), so an audio-text
   contrastive alignment is possible without any further download once a text-concept
   encoder exists (or train one on the same 23 k transcripts);
3. the image-side lesson holds: only ship after a 3-seed, speaker-disjoint run beats the
   identity baseline on top-1, top-5 **and** probe.

## 6. Checkpoints + loader contract (C++ side)

### 6.1 Checkpoints (negative evidence, archived)

All runs under `runtime_store/models/io_train2/` (each `manifest.json` records encoder
sha256, data provenance, counts, 3-seed per-seed + mean±sd metrics, guardrail flags):

| run dir | task | verdict | dir size | manifest sha256 |
|---|---|---|---|---|
| `audio_fsc_intent_20261118/` | 31-way intent | NOT DEPLOY (practical rule) | 34.2 MB | `bc9b4438…f27adce` |
| `audio_fsc_action_20261118/` | 6-way action | NOT DEPLOY | 24.9 MB | — |
| `audio_fsc_object_20261118/` | 14-way object | NOT DEPLOY | 27.9 MB | — |
| `audio_fsc_denoise_20261118/` | denoise 10 dB | NOT DEPLOY | 39.1 MB | — |
| `audio_fsc_denoise0dB_20261118/` | denoise 0 dB | NOT DEPLOY | 39.1 MB | `649b402b…4d77c5e` |
| `audio_fsc_20261118/` | 31-way intent, **superseded** run under the old relative-only guardrail (manifest says DEPLOY) | superseded — do not use; kept only for audit | 34.2 MB | — |

Head weights for every rung are saved as `head_<rung>.npz`/`.json` (66 KB for linear,
~1 MB for MLP) inside each run dir; the class-level manifest is the source of truth.
The superseded `audio_fsc_20261118` is why the final rule now requires beating the
trivial baseline (see §5).

### 6.2 Loader contract for the C++ side (spec; no artifact shipped)

No head is deployed, so **no loader contract is enabled**. The format to implement is the
already-agreed `phoenix.io_head.v1` (identical to the image side), so a future passing
head can be dropped in without a second design round:

| item | value |
|---|---|
| config key | `jepa.speech.headCheckpoint` (string path to the contract JSON, relative to phoenix root) |
| enable/require | `jepa.speech.headRequire` (bool, default `false`); optional `jepa.speech.headSha256` pin |
| weights | `head_deploy.f32`: float32 little-endian, **W `[outDim][inDim]` row-major** then **b `[outDim]`** |
| tensor shape (this work) | inDim **128**, outDim **128** (66 048 B total) |
| semantics | `out[o] = b[o] + Σ_i W[o][i] * x[i]`, then existing `normalizeVector(out)` |
| input | the **raw** 128-d speech encoder `concept` (pre-normalisation) |
| apply order | encoder → head → (if `targetDim != 128`) `projectToDimension` → `normalizeVector` |
| call sites | `audio_model.cpp:1128-1138` (`AudioLocalOnnxModel::encode`) and `audio_model.cpp:514-526` (`AudioHbdnnModel::runEncoderBpu`) |
| contract JSON | `{format:"phoenix.io_head.v1", modality:"speech", type:"linear", inDim:128, outDim:128, normalize:true, weightFile:"head_deploy.f32", weightDtype:"float32-le", weightOrder:"rowmajor", tensors:[{name:"W",shape:[128,128]},{name:"b",shape:[128]}], weightBytes:66048, weightSha256, applyAfter:"raw encoder concept", inputShape:[128], outputShape:[128], referenceEval:{…}}` |
| fail behaviour | empty/unreadable/sha-mismatch ⇒ log `[io-head] <reason>` and keep the baseline concept (fail-open; accuracy feature, never a gate); expose `headLoaded/headType/headSha256` in `AudioModel::status()` |

For loader integration testing only (NOT deployable), a **reference contract** was
generated from the 0 dB denoise linear head:
`runtime_store/models/io_train2/audio_fsc_denoise0dB_20261118/head_deploy.contract.json`
(`head_deploy.f32`, 66 048 B, sha256
`ac5f22c41d90afa7e9214b59a9af4535a6228c25db787ad6ecfedcd70a8e94ee`, marked
`referenceEval.note = "REFERENCE ONLY - verdict NOT DEPLOY"`). Use it to exercise the
loader; do not enable it in `config/phoenix.json`.

Rebuild path when a passing head exists: sidecar
`build\tmp\compile_next.bat` → `phoenix_main.next.exe` (never `compile.bat` while the
gateway runs), then set the config key in a Lead-scheduled restart window.

## 7. Honest limitations

1. **The frozen encoder is the bottleneck, not the head.** All five real objectives say
   the 1 s in-house concept encoder carries no linearly/5-NN accessible intent or lexical
   information, and is already noise-invariant at 10 dB. This round deliberately does not
   touch encoder weights (read-only model dirs, no product-source edits).
2. **Fixed 1 s input vs 1–4 s utterances.** The runtime encoder accepts exactly 16 000
   samples; FSC utterances are longer, so each utterance is mean-pooled over 3 evenly
   spaced 1 s windows. A centre-crop control gave the same chance-level result, so pooling
   is not the cause of the negative (but it does mean part of each utterance is unused).
3. **Label set is 31 intents derived from `action|object|location`** (the CSV has no
   single `intent_class` column in this mirror). Action=6, object=14 values were tested as
   coarser real labels; all are at/below trivial baselines.
4. **Denoise head trade-off.** At 0 dB the head improves concept cosine (0.923→0.974) but
   lowers paired retrieval (0.661→0.571). A retrieval-preserving denoise objective
   (e.g. metric-learning + denoise, or contrastive with noisy views) would be needed
   before denoise could be deployed.
5. **Licence.** FSC is **CC BY-NC 4.0** (non-commercial). The archive was deleted; only
   local extracted audio, feature caches (derived, 128-d), checkpoints and manifests are
   kept. Do not redistribute the wavs.
6. **Test distribution.** Speaker-disjoint (10 unseen speakers, 3 793 utterances) — this is
   the correct protocol for SLU, but no cross-corpus/device generalisation was measured.
7. **Fixed hyperparameters.** 20 epochs, batch 256, lr 0.02 (linear) / 0.005 (MLP),
   weight decay 1e-4; no tuning per rung. 3 seeds vary init/batch order on a fixed split;
   the ±sd therefore captures training noise, not split variance.
8. **Feature extraction** is the frozen ONNX encoder on CPU (GPU was busy with the live
   stack: 85–99 %, 5.9/6.1 GB VRAM). 3-crop extraction ≈ 39 k forward passes/run,
   deterministic; caches are in each run dir.
9. **No audio-text contrastive run.** FSC ships transcripts, so this is possible locally
   once a text-concept encoder exists; it was not attempted in this round (time-box), and
   it is the recommended next step instead of more head variants.
10. **Live-system constraints honoured:** gateway pid 17376 and 3 llama-servers untouched,
    no `compile.bat`, no git writes, model dirs read-only.
11. **Archive deletion irreversibility.** The 1.54 GB tar was deleted only after the
    extraction was verified (30 043 wavs, CSVs, spot-checked paths); the exact re-download
    command and sha256 are recorded in `provenance.json` and §1.1.

## 8. Reproduction

### 8.1 Re-download the corpus (only needed if the dataset dir is lost)

```powershell
cd E:\079\v6.0Alixander\phoenix
$d = "runtime_store\datasets\audio\fluent_speech_commands"; New-Item -ItemType Directory -Force $d
curl.exe -L --retry 3 -o "$d\fsc_raw.tar.gz" "https://modelscope.cn/api/v1/datasets/OmniData/Fluent_Speech_Commands/repo?Revision=master&FilePath=raw/jf8398hf30f0381738rucj3828chfdnchs.tar.gz"
(Get-FileHash "$d\fsc_raw.tar.gz" -Algorithm SHA256).Hash.ToLower()   # must be 4376699f7daf134a9fa57a1d880ffcaaf94a3e2551ba0b40ad894d7abb71aacb
New-Item -ItemType Directory -Force "$d\extracted"
tar.exe -xzf "$d\fsc_raw.tar.gz" -C "$d\extracted"
Remove-Item "$d\fsc_raw.tar.gz" -Force     # delete duplicate after extraction
```

### 8.2 Re-run the objectives (CPU, ~2–4 min each)

```powershell
$py = "runtime_store\venvs\vision-cu111\Scripts\python.exe"
$ds = "runtime_store\datasets\audio\fluent_speech_commands\extracted"

# real classification objectives
& $py tools\audio_train_head.py --dataset-dir $ds --train-per-class 200 --epochs 20 --crops 3 --label-field intent --run-name fsc_intent_20261118
& $py tools\audio_train_head.py --dataset-dir $ds --train-per-class 200 --epochs 20 --crops 3 --label-field action --run-name fsc_action_20261118
& $py tools\audio_train_head.py --dataset-dir $ds --train-per-class 200 --epochs 20 --crops 3 --label-field object --run-name fsc_object_20261118

# denoise / reconstruction objective (local MUSAN noise, 10 dB and 0 dB)
& $py tools\audio_train_head.py --objective denoise --dataset-dir $ds --train-per-class 200 --epochs 20 --crops 3 --snr-db 10 --run-name fsc_denoise_20261118
& $py tools\audio_train_head.py --objective denoise --dataset-dir $ds --train-per-class 200 --epochs 20 --crops 3 --snr-db 0  --run-name fsc_denoise0dB_20261118
```

### 8.3 Verify the verdict from the manifests (seconds, no retraining)

```powershell
runtime_store\venvs\vision-cu111\Scripts\python.exe -c @"
import json, os
root = 'runtime_store/models/io_train2'
for d in ('audio_fsc_intent_20261118','audio_fsc_action_20261118',
          'audio_fsc_object_20261118','audio_fsc_denoise_20261118',
          'audio_fsc_denoise0dB_20261118'):
    m = json.load(open(os.path.join(root, d, 'manifest.json')))
    print(d, '|', m['verdict'], '| chosen =', m['chosen_rung'],
          '| held-out =', (m.get('dataset') or m.get('data'))['heldout_samples'])
"@
```
All five manifests must print `NOT DEPLOY` / `chosen = None` — that is the round's
decision. The superseded `audio_fsc_20261118` (old relative-only rule) is the only one
that prints `DEPLOY` and must not be used.

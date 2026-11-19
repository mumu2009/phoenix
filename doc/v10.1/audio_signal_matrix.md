# Audio Signal Matrix — audit, fixes, understanding measurements

Status: verified/fixed/measured **2026-11-19 18:14–18:45 +08:00** on the live tree
`E:\079\v6.0Alixander\phoenix`. The previous owner of `tools/audio_signal_matrix.py`
died mid-task (its run log shows it completed `qwen/clean` + `qwen/noise10` and then
stopped, losing all inference). This round audits that script, fixes its bugs, and runs
the U1–U4 understanding measurements on the local FSC corpus. CPU-only (GPU left to the
live stack); gateway pid 17376 (:5097) and the 3 llama-servers were not touched
(health 200 before/after); no compile.bat, no git, live model dirs read-only.

Vocabulary follows `doc/v10.1/io_signal_matrix_design.md`: the representation is a
**signal matrix**; the audio contract is **one 4096-d unit query per 40 ms frame →
`[F,4096]`**; 128 is only the projected interface view; **classification accuracy is not
the headline metric** (U1–U4 replace it).

Run dir: `runtime_store/models/audio_signal_matrix/fsc_both_20261119/`

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Does the Qwen frontend really produce a 4096-d 40 ms frame space? | **Yes, verified.** ONNX I/O `input_features[batch,128,3000] → audio_features[batch,750,4096]`; single wavs give `[F,4096] float32` at **24.7–25.1 fps** (= 40 ms/frame). |
| Does the live speech encoder produce it? | **No — it cannot.** It is a 1 s pooled 128-d concept encoder; this run measures it honestly as `[F,128]` (sliding 1 s window every 40 ms, 25 fps). A 4096-d view is only written when explicitly requested as a labelled random projection (`--live-projection random4096`, off by default). **The live speech encoder cannot produce a 4096-d frame space.** |
| U1 next-frame predictability | **Qwen matrix: YES** — 3-step rollout MSE 4.86 vs identity 13.40 / mean 8.75 (ratio 0.363, R² 0.637). Live 128-d view: weaker but present (ratio 0.594, R² 0.406). |
| U2 environment separability | Speaker identity: **partial** for Qwen (NMI 0.654 / ARI 0.198) and slightly higher for live (NMI 0.714 / ARI 0.310, n=30 caveat). Synthetic noise-condition separability: **no** for both (ARI 0.068 / 0.033). |
| U3 time-arrow | **Qwen: strong PASS** (balanced accuracy **0.967 ± 0.067**). Live: **marginal PASS** (0.583 ± 0.197, threshold 0.55). |
| U4 meme-space alignment | **Not established** (proxy only): top-1 cosine 0.085 (Qwen) / 0.069 (live) against 12 hash-based meme nodes; the learned W_t/W_g alignment and the memetic α matrix are not available offline (§5). |
| Minimum artifact for a real 4096-d text-audio teacher | **None to download** — the Qwen2-Audio connector + Whisper feature extractor are already on disk (stale tree, 2.57 GB `.data`) and verified here. A 40 ms × 4096-d encoder for the *live* path would be new work; the 1 s/128-d encoder cannot be upgraded by projection. |

---

## 1. What the two frontends actually produce (verified, not claimed)

Provenance (hashes recomputed this round; they match the dead author's
`frontends_provenance/artifacts.json` byte-for-byte):

| Artifact | Bytes | sha256 |
|---|---|---|
| `…\runtime_store\models\multimodal_enc\qwen2_audio_connector.onnx` | 2,781,377 | `6cb8871b50c37bd504ccddc692dd2800e3d4827729941e605ea105a503ba4437` |
| `…\qwen2_audio_connector.onnx.data` (Qwen2-Audio acoustic encoder + connector) | 2,568,945,664 | `90afbf803f3bbc760b9aec744913f7d45a6a8e37b505c3d3cffee2f539f8200f` |
| `…\qwen2_audio_feature_extractor\preprocessor_config.json` (WhisperFeatureExtractor, 128 mels, 30 s/3000 frames) | 329 | `7f1028d0bf0bcd528d4ffe02b6286ef0d0722e650237e0852774118cb18aa14d` |
| `runtime_store\models\additive_jepa\speech_encoder\best.onnx` (live) | 4,936,265 | `ecaadc84d50e811b4713dad6ffad151d386a4801b7ebfb393a7e74e1381695d4` |

### 1.1 `--frontend qwen` — TRUE 40 ms × 4096 frame space

ONNX session (CPU, 16 threads, load 2.8 s):
`input_features [batch,128,3000]` → `audio_features [batch,750,4096]`; 750 frames / 30 s
= 40 ms/frame. Single-wav results:

| wav | duration | matrix | dtype | frames/s |
|---|---|---|---|---|
| FSC #1 | 3.158 s | `[79, 4096]` | float32 | 25.02 |
| FSC #2 | 2.043 s | `[51, 4096]` | float32 | 24.96 |
| FSC #3 | 2.508 s | `[63, 4096]` | float32 | 25.12 |
| FSC extra A | — | `[39, 4096]` | float32 | 24.70 |
| FSC extra B | — | `[81, 4096]` | float32 | 24.92 |

**CPU cost: ~0.11 s/frame** (30 clean utterances: 189 s / 1,713 frames; 2.29 s mean
utterance → ~6.3 s compute, i.e. **~2.75× real-time**); ~6.3 s per utterance. This is
the only frontend that satisfies `multimodal_world_model.hpp:35-47`.

### 1.2 `--frontend live` — HONEST `[F,128]`, never a fake 4096

The live encoder is a **1 s pooled 128-d concept** model (`[1,1,1,16000] → [128]`). The
script slides a 1 s window every 40 ms and emits the **true `[F,128]`** frame matrix,
`true_frame_space=False`:

| wav | matrix | dtype | frames/s |
|---|---|---|---|
| FSC #1 | `[79, 128]` | float32 | 25.02 |
| FSC #3 | `[63, 128]` | float32 | 25.12 |
| FSC extra C | `[33, 128]` | float32 | 25.38 |

**CPU cost: ~1.2 ms/frame** (30 utterances × 3 conditions in ~2 s each) — ~0.03 s per
1 s of audio. `--live-projection random4096` is the only way to get a 4096-wide live
matrix; it is explicitly flagged `projection_is_qwen_space=False` and is off by default.

---

## 2. Script audit — bugs found and fixed (`tools/audio_signal_matrix.py`)

The CLI is unchanged (same flags/modes); only additive/optional behaviour was added.

| # | Bug | Evidence | Fix |
|---|---|---|---|
| 1 | **CWD-dependent paths**: `LIVE_ENC`, `FSC`, `MUSAN_NOISE`, `MEME_SERIAL`, default `--out` were relative and resolved against the process CWD; the script only worked if launched from the repo root. | Launch from `E:\079\v6.0Alixander` → `NoSuchFile: runtime_store/models/additive_jepa/speech_encoder/best.onnx`. | Added `ROOT = dirname(dirname(__file__))` + `_abs()`; all relative paths/docs resolve to the repo root. |
| 2 | **Live frontend silently faked 4096-d frames** via a random 128→4096 projection (`matrix()`), violating "do not fake frames" and the contract's honesty requirement. | Code returned `[F,4096]` for live with no per-call flag. | `LiveFrontend.matrix()` now returns the TRUE `[F,128]`; `matrix_4096()` exists only for the explicit, labelled `--live-projection random4096`; `describe()` reports `frame_dim=128, true_frame_space=False`. |
| 3 | **Single mode wrote no metadata** — no shape/dtype/space/fps/flag, so a consumer could not tell a true 4096 matrix from a projected one. | `np.savez_compressed` of `M_*` only. | Single output now includes `meta` JSON: wav, duration, and per-frontend `{true_frame_space, frame_dim, n_frames, frames_per_second, dtype, projected_4096}`, plus qwen contract assertion `[F,4096]`. |
| 4 | **Default single output written next to the wav**, polluting `runtime_store/datasets/...` (30k-file corpus dir). | `--out` default = `os.path.splitext(wav)[0] + ".matrix.npz"`. | Default output is `build/tmp/audio_signal_matrix/<name>.matrix.npz`. |
| 5 | **A crash mid-run lost all Qwen inference** (this is exactly how the previous author lost the run: log stops after `qwen/noise10`, nothing cached). | `np.save` of the cache only happened after *all* conditions + reversed pass. | Atomic incremental `_save_cache()` after every (frontend, condition) and after the reversed pass (`os.replace`); partial caches are detected and recomputed. |

Also verified: `--mode single --frontend both` works (qwen `[F,4096]` + live `[F,128]` +
metadata); reversed-waveform path (U3) is honest (no reversed cached frames); the
`np.load(...).item()` cache round-trip works; `_kmeans/_ari/_nmi` are dependency-free
(the ORT env has no sklearn).

**Remaining known limitation (not a bug):** U2 speaker clustering uses n=30 utterances
with k=10 clusters, so ARI is unstable and not comparable to a full-split study; U4 is a
proxy by construction (see §5).

---

## 3. U1–U4 understanding measurements (FSC test subset, 30 utterances / 10 unseen speakers)

Protocol: FSC official **test** split (10 unseen speakers), 3 utterances/speaker = 30
utterances (mean 2.29 s); conditions **clean / MUSAN 10 dB / MUSAN 0 dB**; time-reversal
on 16 utterances; held-out for U1 is **speaker-disjoint**; no classification accuracy is
reported as a headline. Run total 747.8 s (Qwen inference dominates).

| metric | Qwen `[F,4096]` true space | Live `[F,128]` pooled view |
|---|---|---|
| **U1** rollout MSE (3-step, PCA-64) | **4.861** | 0.00878 |
| U1 identity baseline MSE | 13.395 | 0.01478 |
| U1 mean baseline MSE | 8.746 | 0.01281 |
| U1 ratio vs identity (lower better) | **0.363** | 0.594 |
| U1 ratio vs mean | **0.556** | 0.685 |
| U1 R² vs identity | **0.637** | 0.406 |
| **U2** speaker clustering ARI / NMI (n=30, k=10) | 0.198 / 0.654 | **0.310 / 0.714** |
| **U2** noise-condition ARI / NMI (n=90, k=3) | 0.068 / 0.165 | 0.033 / 0.049 |
| **U3** time-arrow balanced accuracy (16+16, 5-fold) | **0.967 ± 0.067 (PASS)** | 0.583 ± 0.197 (marginal PASS) |
| **U4** meme proxy top-1 cosine / mean-all cosine | 0.085 / −0.003 | 0.069 / −0.010 |

Interpretation (measured, not assumed):

- **U1 — next-frame predictability: supported by the Qwen frame matrix.** The rollout
  beats both the identity and mean baselines on speaker-disjoint held-out utterances
  (ratio 0.363; R² 0.637). On the live view the signal is much weaker (ratio 0.594),
  because its frames are 1 s windows over a pooled 128-d concept — consecutive frames
  are highly redundant.
- **U2 — environment separability: partial (speaker) / absent (synthetic noise).** The
  Qwen matrix separates speakers at NMI 0.654 (ARI 0.198, small-n caveat); the live
  pooled concept is *slightly better* at speaker NMI (0.714) — expected, because a 1 s
  pooled concept is speaker/session-biased. Neither representation separates the
  synthetic clean/10 dB/0 dB conditions (ARI 0.068 / 0.033): noise mixing barely moves
  the pooled statistics versus speaker variance. This is a negative result for the
  "noise condition" probe as constructed; real room/channel conditions would be needed.
- **U3 — time-arrow: strongly supported by the Qwen frame matrix** (0.967, chance 0.5;
  five folds). The live view is only marginally above threshold (0.583 ± 0.197) because
  its 1 s pooling window blurs local time direction.
- **U4 — meme-space alignment: not established.** The proxy top-1 cosine (0.085 Qwen /
  0.069 live, 12 hash-based meme nodes) is near zero. This is *not* evidence that the
  matrix is unrelated to the meme layer: the real alignment is
  `cos(W_t·u(x), W_g·z_g)` with learned projections (`doc/math/transformer_math_proof.md:37-40`)
  and an α matrix from `memetic_existence` that are not available offline. The proxy is
  exactly what it says: a hash-space cosine.

---

## 4. Verdict — which understanding signals the CURRENT audio frontend supports

**Qwen2-Audio connector (the true `[F,4096]` @40 ms frontend, already on disk):**
- **U1 next-frame predictability — YES** (ratio 0.363, R² 0.637 vs identity).
- **U3 time-arrow — YES** (0.967 ± 0.067).
- **U2 speaker/environment separability — PARTIAL** (NMI 0.654; small-n, speaker-biased).
- **U2 noise-condition separability — NO** (ARI 0.068).
- **U4 meme alignment — NOT ESTABLISHED** (proxy only; seam §5).

**Live speech encoder (`additive_jepa/speech_encoder/best.onnx`):**
- **Cannot produce a 4096-d frame space — explicit.** It is a 1 s pooled 128-d concept
  encoder (`audio_model.cpp:1100-1139`). This run measures it honestly as `[F,128]`.
- U1 weak signal (ratio 0.594), U3 marginal (0.583), U2 speaker separability slightly
  better than Qwen at NMI level (0.714) but that is a property of pooled speaker-biased
  concepts, not of a frame space; U2 noise and U4 absent.

**Minimum artifact needed for a real 4096-d text-audio teacher:**
1. **Nothing to download.** `qwen2_audio_connector.onnx` + `.data` (2.57 GB) + the
   Whisper feature extractor are on disk (stale tree) and were verified this round:
   `[batch,128,3000] → [batch,750,4096]`, 25 fps. This is the minimum artifact for the
   true frame-space path. (Only the Qwen **LLM** would be needed for audio→text
   generation — not for the matrix.)
2. For the **live** path to get a genuine 40 ms × 4096-d space, a new encoder is
   required (a 1 s pooled 128-d model cannot be projected into that space; the random
   projection is not a substitute). That is new training work, not a missing download.
3. For **U4** to become a real measurement (not a proxy), the C++-side α matrix
   (`memetic_existence.hpp:88-114`) and the learned `W_t/W_g` projections
   (`doc/math/transformer_math_proof.md:37-40`) must be exposed to Python, or the
   alignment must run inside the C++ process.

---

## 5. U4 meme-space alignment — the exact seam (documented, no direct accessor)

There is **no Python accessor** for the memetic α/tensor space. The exact seams:

| Seam | Location | What it gives |
|---|---|---|
| α matrix (`memeId → word → mass`) | `memetic_existence.hpp:88-114` (`mappingWordAlpha(tf,df)=tf/df`, `MappingWordHit{memeId,alpha}`, `MappingMassFrom`) | the real meme-weight matrix U4 should align to |
| Node embeddings in C++ | `main_hub_parts/045_class_graphtensorbridge.inc:17-43` (`computeNodeEmbeddings(ids,512)`) → `textToMiniEmbedding(joined,512)` (`main_hub_parts/014_section_before_rng32.inc:18`) | hash-based 512-d node vectors (the scheme `tools/audio_signal_matrix.py` reproduces) |
| Serialized meme graph | `tools/phoenix_gnn_meme_serial.ps1:14-17` writes `runtime_store/gnn_meme_serial.json` (from `gnn_meme_screen.json`; journal `.jsonl`) | 12 meme groups (`id`, `words`, `seedCarrier`, `rounds`, `presentRounds`) = the offline proxy input |
| Learned alignment | `doc/math/transformer_math_proof.md:37-40` (`cos(W_t·u(x), W_g·z_g)`) | **unavailable offline** — this is why U4 here is a proxy |

The script's U4 is therefore: pooled matrix statistics → hash-projected to 512-d →
cosine against `text_to_mini_embedding` of each serialized meme group. It reproduces the
C++ `textToMiniEmbedding` scheme but **not** the trained alignment.

---

## 6. Production seam if the audio matrix path is wired (design, not implemented)

- Producer: today `AudioLocalOnnxModel::encode` (`audio_model.cpp:1100-1139`) returns one
  128-d concept after a hard dimension check + projection (`:1129-1135`). The matrix path
  needs a new producer (Qwen connector via a python bridge, like
  `tools/local_onnx_runner.py`) returning `[F,4096]` and a pooling/128 view.
- Contract: `multimodal_world_model.hpp:35-47` (one 4096-d unit query per 40 ms frame;
  mean query = summary); consumer side already exists (`main_hub_parts/116_section_tail.inc:508-521`,
  `inference_unit_pipeline.hpp:114-137`), per `doc/v10.1/io_signal_matrix_design.md`.
- The 128 hard constraint and the projection in `audio_model.cpp:1129-1135` must become
  "matrix → pool/project", not "reject if dim ≠ 128".
- Cost decision: Qwen matrix at **0.11 s/frame** is ~2.75× real-time on this CPU; it must
  be a batched/async subprocess (or GPU) path, never inline per bridge call. The live
  128-d view is ~1.2 ms/frame and can stay inline.

---

## 7. Reproduce

```powershell
cd E:\079\v6.0Alixander\phoenix
$py = "runtime_store\venvs\vision-cu111\Scripts\python.exe"
# single (honest metadata; qwen true [F,4096], live true [F,128]):
& $py tools\audio_signal_matrix.py --mode single --frontend both --wav <fsc.wav> --out build\tmp\x.npz
# full U1-U4 (≈12.5 min CPU):
& $py tools\audio_signal_matrix.py --mode metrics --frontend both --n-per-speaker 3 --rev-n 16 `
    --threads 16 --out-dir runtime_store/models/audio_signal_matrix --run-name fsc_both_20261119
```

Artifacts in `runtime_store/models/audio_signal_matrix/fsc_both_20261119/`:
`manifest.json` (frontends, honesty flags, U1–U4, limitations), `fsc_subset_stats.npz`,
`matrices.pkl.npy` (88 MB, incremental cache: 90 matrices × clean/10 dB/0 dB + reversed).

## 8. Limitations (honest)

1. FSC subset is small (30 utterances / 10 speakers) — U1/U2/U3 are indicative; k=10 on
   n=30 makes speaker ARI unstable (the previous run used n=20 and died before metrics).
2. U2 noise conditions are synthetic MUSAN mixes, not real rooms/channels.
3. U4 is a hash-space proxy; the real α matrix / learned alignment is C++-only (§5).
4. Qwen frontend is 2.75× real-time on CPU; not an inline path without batching/GPU.
5. The live frontend's higher speaker NMI is a property of 1 s pooled concepts and does
   not turn it into a 4096-d frame space.

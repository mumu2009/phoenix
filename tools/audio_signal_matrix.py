#!/usr/bin/env python3
"""audio_signal_matrix.py - produce the audio SIGNAL MATRIX and measure understanding.

Per `doc/v10.1/io_signal_matrix_design.md`, the system representation is a signal matrix,
not labels. The audio contract in `multimodal_world_model.hpp:35-47` is:
    one 4096-d unit query per 40 ms frame  ->  matrix [F, 4096]
128 is only the projected interface view; classification accuracy is not the headline.

FRONT-ENDS (checked locally, 2026-11-18):
  qwen  = E:\\079\\v6.0Alixander\\runtime_store\\models\\multimodal_enc\\qwen2_audio_connector.onnx
          + qwen2_audio_feature_extractor/ (WhisperFeatureExtractor, 128 mels).
          Verified ONNX I/O: input_features [batch,128,3000] -> audio_features [batch,750,4096]
          => 750 frames / 30 s = 40 ms/frame, TRUE 4096-d frame space (the .data is 2.57 GB
          and contains the Qwen2-Audio acoustic encoder itself, not just a linear connector).
  live  = runtime_store/models/additive_jepa/speech_encoder/best.onnx (16 kHz, 1 s -> 128-d).
          This CANNOT produce a 4096-d 40 ms frame space: it is a 1 s pooled concept encoder.
          Fallback = sliding 1 s window every 40 ms -> [F,128], then a deterministic random
          projection 128->4096 to satisfy the storage contract. Flagged true_frame_space=False;
          the 4096 view is an arbitrary projection, NOT the Qwen space.

UNDERSTANDING METRICS (no classification accuracy headline):
  U1 next-frame predictability: 1-step (and 3-step) rollout MSE vs identity + mean baselines,
     learned in a PCA-64 space; held-out utterances are speaker-disjoint.
  U2 environment separability: KMeans on pooled frame statistics; ARI/NMI against speaker
     identity and against synthetic noise conditions (clean / 10 dB / 0 dB MUSAN).
  U3 time-arrow: forward vs time-reversed audio, logistic-regression balanced accuracy (5-fold),
     want > 0.55 (0.5 = chance).
  U4 meme-space alignment (PROXY): cosine between projected matrix statistics and hashed
     meme-node vectors reproduced from runtime_store/gnn_meme_serial.json via the
     textToMiniEmbedding hash scheme (main_hub_parts/014_...inc:18-45). The learned
     alignment projections W_t/W_g (doc/math/transformer_math_proof.md:37-40) are not
     available offline, so this is a proxy and is reported as such.

Usage (from the phoenix root):
  # single wav -> matrix .npz
  runtime_store\\venvs\\vision-cu111\\Scripts\\python.exe tools\\audio_signal_matrix.py ^
      --mode single --frontend qwen --wav some.wav --out matrix.npz
  # full understanding measurement on FSC
  ... --mode metrics --frontend both --n-per-speaker 2 --rev-n 16 ^
      --out-dir runtime_store/models/audio_signal_matrix/<ts>
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import random
import sys
import time
import wave

import numpy as np

SAMPLES = 16000
HOP_MS = 40

STALE = r"E:\079\v6.0Alixander\runtime_store\models\multimodal_enc"
QWEN_ONNX = os.path.join(STALE, "qwen2_audio_connector.onnx")
QWEN_CFG = os.path.join(STALE, "qwen2_audio_feature_extractor")
LIVE_ENC = "runtime_store/models/additive_jepa/speech_encoder/best.onnx"
FSC = "runtime_store/datasets/audio/fluent_speech_commands/extracted"
MUSAN_NOISE = "runtime_store/datasets/musan_extracted/musan/noise"
MEME_SERIAL = "runtime_store/gnn_meme_serial.json"


def log(msg):
    print("[matrix] %s" % msg, flush=True)


def sha256_file(path, step=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(step), b""):
            h.update(b)
    return h.hexdigest()


def read_wav(path):
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise SystemExit("need 16 kHz mono 16-bit wav: %s" % path)
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


# --------------------------------------------------------------------------
# qwen front-end: Whisper mel -> Qwen2-Audio encoder+connector -> [F,4096] @ 40 ms
# --------------------------------------------------------------------------
class QwenFrontend:
    name = "qwen2_audio"
    true_frame_space = True
    frame_dim = 4096
    hop_ms = 40

    def __init__(self, threads=32):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        self.fe = WhisperFeatureExtractor.from_pretrained(QWEN_CFG)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        t0 = time.time()
        self.sess = ort.InferenceSession(QWEN_ONNX, so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        log("qwen front-end loaded in %.1fs (in %s %s -> out %s %s)" %
            (time.time() - t0, self.in_name, self.sess.get_inputs()[0].shape,
             self.out_name, self.sess.get_outputs()[0].shape))

    def matrix(self, wav):
        feat = self.fe(wav, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
        out = self.sess.run([self.out_name], {self.in_name: feat})[0]      # [1,750,4096]
        frames = np.asarray(out[0], np.float32)
        n = max(1, int(round(len(wav) / 16000.0 * 1000.0 / self.hop_ms)))
        return frames[:min(n, frames.shape[0])]

    def describe(self):
        return {"frontend": self.name, "true_frame_space": True,
                "hop_ms": self.hop_ms, "frame_dim": self.frame_dim,
                "onnx": QWEN_ONNX, "onnx_bytes": os.path.getsize(QWEN_ONNX),
                "data_bytes": os.path.getsize(QWEN_ONNX + ".data"),
                "feature_extractor": QWEN_CFG,
                "why_true": "ONNX I/O is input_features[1,128,3000] -> audio_features[1,750,4096]; "
                            "750 frames / 30 s = 40 ms, the exact multimodal_world_model.hpp "
                            "audio contract; the 2.57 GB external data carries the Qwen2-Audio "
                            "acoustic encoder, not just a linear connector."}


# --------------------------------------------------------------------------
# live front-end fallback: 1 s pooled 128-d concept, sliding every 40 ms
# --------------------------------------------------------------------------
class LiveFrontend:
    name = "live_speech_encoder_sliding_window"
    true_frame_space = False
    frame_dim = 4096          # storage contract shape; actual encoder dim = 128
    actual_dim = 128
    hop_ms = 40

    def __init__(self, threads=32, seed=20261118):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(LIVE_ENC, so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        rng = np.random.RandomState(seed)
        self.P = (rng.randn(self.actual_dim, self.frame_dim) /
                  np.sqrt(self.actual_dim)).astype(np.float32)

    def raw_frames(self, wav):
        hop = int(16000 * self.hop_ms / 1000)
        starts = list(range(0, max(1, len(wav) - 1), hop))
        out = []
        for s in starts:
            chunk = wav[s:s + 16000]
            if chunk.size < 16000:
                chunk = np.pad(chunk, (0, 16000 - chunk.size))
            x = chunk.reshape(1, 1, 1, 16000)
            out.append(np.asarray(self.sess.run([self.out_name], {self.in_name: x})[0],
                                  np.float32).reshape(-1))
        return np.stack(out)  # [F,128]

    def matrix(self, wav):
        raw = self.raw_frames(wav)
        proj = raw @ self.P
        n = np.linalg.norm(proj, axis=1, keepdims=True)
        return (proj / np.maximum(n, 1e-8)).astype(np.float32)

    def describe(self):
        return {"frontend": self.name, "true_frame_space": False,
                "hop_ms": self.hop_ms, "frame_dim": self.frame_dim,
                "actual_encoder_dim": self.actual_dim,
                "onnx": LIVE_ENC, "onnx_bytes": os.path.getsize(LIVE_ENC),
                "why_not_true": "the live encoder is a 1 s pooled 128-d concept model; it has no "
                                "40 ms / 4096-d frame output. The matrix is built by sliding a 1 s "
                                "window every 40 ms and projecting 128->4096 with a deterministic "
                                "random matrix; the 4096 view is arbitrary, NOT the Qwen space."}


# --------------------------------------------------------------------------
# U1-U4 metrics
# --------------------------------------------------------------------------
def pooled_stats(M):
    """Per-utterance summary: mean and std over frames (+ mean |delta|)."""
    d = np.abs(np.diff(M, axis=0)).mean(0) if M.shape[0] > 1 else np.zeros(M.shape[1], np.float32)
    return np.concatenate([M.mean(0), M.std(0), d]).astype(np.float32)


def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    # top-k right singular vectors via covariance eigendecomposition (d moderate)
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(-w)[:k]
    return mu, V[:, idx], np.maximum(w[idx], 0)


def u1_next_frame(train_mats, held_mats, k=64, steps=3):
    tr_pairs = [(M[:-1], M[1:]) for M in train_mats if M.shape[0] > 1]
    X = np.concatenate([p[0] for p in tr_pairs], 0)
    Y = np.concatenate([p[1] for p in tr_pairs], 0)
    mu, V, w = pca_fit(X, k)
    Xp = (X - mu) @ V
    Yp = (Y - mu) @ V
    A = np.linalg.solve(Xp.T @ Xp + 1e-2 * np.eye(k, dtype=np.float32), Xp.T @ Yp)
    out = {"pca_dim": k, "train_frames": int(X.shape[0]), "steps": steps}
    # one-step on held-out
    Hx, Hy = [], []
    for M in held_mats:
        if M.shape[0] < steps + 1:
            continue
        Hx.append((M[:-steps] - mu) @ V)
        Hy.append((M[steps:] - mu) @ V)
    if not Hx:
        return {"error": "no held-out frames"}
    Hx, Hy = np.concatenate(Hx), np.concatenate(Hy)
    # rollout `steps` times from each start frame
    pred = Hx.copy()
    for _ in range(steps):
        pred = pred @ A
    mse_pred = float(np.mean((pred - Hy) ** 2))
    mse_identity = float(np.mean(((Hx - Hy)) ** 2))
    mean_frame = Yp.mean(0, keepdims=True)
    mse_mean = float(np.mean((mean_frame - Hy) ** 2))
    out.update({"rollout_mse": mse_pred, "identity_mse": mse_identity,
                "mean_mse": mse_mean,
                "ratio_vs_identity": mse_pred / max(1e-12, mse_identity),
                "ratio_vs_mean": mse_pred / max(1e-12, mse_mean),
                "r2_vs_identity": 1.0 - mse_pred / max(1e-12, mse_identity)})
    return out


def _kmeans(X, k, iters=60, n_init=5, seed=0):
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        idx = [rng.randint(len(X))]
        d2 = ((X - X[idx[0]]) ** 2).sum(1)
        for _c in range(1, k):
            p = d2 / (d2.sum() + 1e-12)
            j = rng.choice(len(X), p=p)
            idx.append(j)
            d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(1))
        C = X[np.array(idx)].copy()
        lab = np.zeros(len(X), np.int64)
        for _it in range(iters):
            D = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
            lab = D.argmin(1)
            newC = np.stack([X[lab == c].mean(0) if np.any(lab == c) else C[c] for c in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy(), C.copy())
    return best[1], best[2]


def _ari(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    u = {v: i for i, v in enumerate(set(labels_true.tolist()))}
    v = {v: i for i, v in enumerate(set(labels_pred.tolist()))}
    C = np.zeros((len(u), len(v)), np.int64)
    for a, b in zip(labels_true, labels_pred):
        C[u[a], v[b]] += 1
    def comb(n):
        return n * (n - 1) / 2.0
    sum_comb = sum(comb(x) for x in C.ravel())
    sum_rows = sum(comb(x) for x in C.sum(1))
    sum_cols = sum(comb(x) for x in C.sum(0))
    total = comb(len(labels_true))
    expected = sum_rows * sum_cols / total if total else 0.0
    max_index = 0.5 * (sum_rows + sum_cols)
    return float((sum_comb - expected) / (max_index - expected)) if max_index != expected else 1.0


def _nmi(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    u = {v: i for i, v in enumerate(set(labels_true.tolist()))}
    v = {v: i for i, v in enumerate(set(labels_pred.tolist()))}
    C = np.zeros((len(u), len(v)), np.float64)
    for a, b in zip(labels_true, labels_pred):
        C[u[a], v[b]] += 1
    P = C / n
    pi = P.sum(1)
    pj = P.sum(0)
    mask = P > 0
    nz = P[mask]
    mi = float((nz * np.log(nz / (pi[:, None] * pj[None, :])[mask])).sum())
    hu = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    hv = float(-(pj[pj > 0] * np.log(pj[pj > 0])).sum())
    return float(2.0 * mi / (hu + hv)) if hu + hv > 0 else 0.0


def u2_env_separability(feats, labels, k):
    X = feats.astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pred, _C = _kmeans(X, min(k, len(X)), seed=0)
    return {"ari": _ari(labels, pred), "nmi": _nmi(labels, pred),
            "n": int(len(labels)), "k": int(min(k, len(X))),
            "note": "dependency-free numpy KMeans/ARI/NMI (the ORT env has no sklearn)"}


def _ridge_direction(Xtr, ytr, Xte):
    """Ridge regression to +/-1 targets, decision at 0 (dependency-free)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    b = (ytr * 2 - 1).astype(np.float64)
    w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ b)
    return ((Xte - mu) / sd) @ w[:-1] + w[-1]


def u3_time_arrow(fwd_stats, rev_stats, folds=5, seed=0):
    X = np.concatenate([fwd_stats, rev_stats], 0).astype(np.float64)
    y = np.concatenate([np.ones(len(fwd_stats)), np.zeros(len(rev_stats))]).astype(np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    f_idx = idx[y[idx] == 1]
    r_idx = idx[y[idx] == 0]
    fold_of = np.zeros(len(y), np.int64)
    fold_of[f_idx] = np.arange(len(f_idx)) % folds
    fold_of[r_idx] = np.arange(len(r_idx)) % folds
    bals = []
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        scores = _ridge_direction(X[tr], y[tr], X[te])
        pred = (scores > 0).astype(np.int64)
        tpr = float((pred[y[te] == 1] == 1).mean()) if np.any(y[te] == 1) else 0.0
        tnr = float((pred[y[te] == 0] == 0).mean()) if np.any(y[te] == 0) else 0.0
        bals.append(0.5 * (tpr + tnr))
    bals = np.array(bals)
    return {"bal_acc_mean": float(bals.mean()), "bal_acc_sd": float(bals.std()),
            "n_forward": int(len(fwd_stats)), "n_reversed": int(len(rev_stats)),
            "chance": 0.5, "pass_0_55": bool(bals.mean() > 0.55),
            "classifier": "ridge direction (+/-) fitted per fold, dependency-free"}


# --------------------------------------------------------------------------
# meme-space proxy (U4)
# --------------------------------------------------------------------------
def fnv1a32(s):
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def hash_str_simple(s, seed=1315423911):
    h = seed
    for b in s.encode("utf-8"):
        h = (h ^ ((h << 5) & 0xFFFFFFFF) + b + (h >> 2)) & 0xFFFFFFFF
    return h


STOP = {"a", "an", "and", "are", "as", "at", "be", "both", "but", "by", "for", "from",
        "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "of",
        "on", "or", "she", "that", "the", "their", "them", "they", "this", "to", "was",
        "were", "with", "you", "not", "no", "so", "if", "then", "than", "which", "who"}


def text_to_mini_embedding(text, dim=512):
    words = [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]
    words = [w for w in words if w not in STOP]
    vec = np.zeros(dim, np.float32)
    for w in words:
        vec[hash_str_simple(w) % dim] += 1.0
    for a, b in zip(words, words[1:]):
        vec[hash_str_simple(a + "_" + b) % dim] += 0.5
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def load_meme_vectors(path, dim=512):
    if not os.path.isfile(path):
        return None, []
    d = json.load(open(path, encoding="utf-8"))
    groups = d.get("groups") or []
    vecs, names = [], []
    for g in groups:
        words = g.get("words") or []
        if not words:
            continue
        vecs.append(text_to_mini_embedding(" ".join(words), dim))
        names.append(g.get("id", ""))
    if not vecs:
        return None, []
    return np.stack(vecs).astype(np.float32), names


def u4_meme_alignment(feats, meme_vecs, seed=7):
    """Proxy: project matrix stats (D) -> meme dim, then cosine to meme-node vectors."""
    if meme_vecs is None:
        return {"available": False,
                "reason": "runtime_store/gnn_meme_serial.json missing or no meme groups"}
    dim = meme_vecs.shape[1]
    rng = np.random.RandomState(seed)
    P = rng.randn(feats.shape[1], dim) / np.sqrt(feats.shape[1])
    A = feats @ P
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-8)
    sims = A @ meme_vecs.T
    top1 = sims.argmax(1)
    return {"available": True, "n_memes": int(meme_vecs.shape[0]),
            "mean_top1_cosine": float(sims.max(1).mean()),
            "mean_all_cosine": float(sims.mean()),
            "unique_top1_memes": int(len(set(top1.tolist()))),
            "proxy_note": "no learned W_t/W_g alignment projections are available offline; "
                          "meme vectors are hash-based node embeddings reproduced from "
                          "gnn_meme_serial.json, so this is a proxy, not the trained alignment."}


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
def fsc_split(split):
    root = os.path.join(FSC, "fluent_speech_commands_dataset")
    path = os.path.join(root, "data", "%s_data.csv" % split)
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            low = {k.lower().strip(): v for k, v in r.items() if k}
            rel = (low.get("path") or "").replace("\\", "/")
            rows.append({"wav": os.path.join(root, rel), "speaker": low.get("speakerid", ""),
                         "text": low.get("transcription", "")})
    return rows


def load_noise():
    out = []
    for dp, _dn, fns in os.walk(MUSAN_NOISE):
        out.extend(os.path.join(dp, f) for f in fns if f.lower().endswith(".wav"))
    return sorted(out)


_NOISE = {}


def mix_noise(wav, noise_files, snr_db, rng):
    if not noise_files:
        return wav
    for _t in range(8):
        p = noise_files[rng.randrange(len(noise_files))]
        if p not in _NOISE:
            _NOISE[p] = read_wav(p)
        a = _NOISE[p]
        if a is None or a.size < wav.size:
            continue
        off = rng.randrange(0, a.size - wav.size + 1)
        n = a[off:off + wav.size]
        cr = np.sqrt(np.mean(wav ** 2) + 1e-12)
        nr = np.sqrt(np.mean(n ** 2) + 1e-12)
        out = wav + n * (cr / nr) / (10.0 ** (snr_db / 20.0))
        m = np.max(np.abs(out))
        if m > 1:
            out = out / m
        return out.astype(np.float32)
    return wav


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["single", "metrics"], default="metrics")
    ap.add_argument("--frontend", choices=["qwen", "live", "both"], default="both")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="runtime_store/models/audio_signal_matrix")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--n-per-speaker", type=int, default=2)
    ap.add_argument("--rev-n", type=int, default=16)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args(argv)
    root = os.getcwd()
    ts = args.run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "single":
        if not args.wav:
            raise SystemExit("--mode single needs --wav")
        wav = read_wav(args.wav)
        outs = {}
        if args.frontend in ("qwen", "both"):
            q = QwenFrontend(args.threads)
            outs["qwen"] = q.matrix(wav)
        if args.frontend in ("live", "both"):
            lv = LiveFrontend(args.threads)
            outs["live"] = lv.matrix(wav)
        out_path = args.out or (os.path.splitext(args.wav)[0] + ".matrix.npz")
        np.savez_compressed(out_path, **{("M_" + k): v for k, v in outs.items()})
        log("wrote %s (%s)" % (out_path, {k: v.shape for k, v in outs.items()}))
        return 0

    run_dir = os.path.join(root, args.out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    log("run dir: %s" % run_dir)
    rng = random.Random(20261118)

    cache_path = os.path.join(run_dir, "matrices.pkl.npy")
    t0_metrics = time.time()
    q = QwenFrontend(args.threads) if args.frontend in ("qwen", "both") else None
    lv = LiveFrontend(args.threads) if args.frontend in ("live", "both") else None

    if os.path.isfile(cache_path):
        z = np.load(cache_path, allow_pickle=True).item()
        data, rev, chosen = z["data"], z["rev"], z["chosen"]
        log("loaded cached matrices from %s" % cache_path)
    else:
        test = fsc_split("test")
        by_spk = {}
        for r in test:
            by_spk.setdefault(r["speaker"], []).append(r)
        speakers = sorted(by_spk)
        chosen = []
        for spk in speakers:
            rs = sorted(by_spk[spk], key=lambda r: r["wav"])
            rng.shuffle(rs)
            chosen.extend(rs[:args.n_per_speaker])
        log("FSC test subset: %d utterances over %d speakers" % (len(chosen), len(speakers)))

        conditions = [("clean", None), ("noise10", 10.0), ("noise0", 0.0)]
        noise_files = load_noise()
        data = {}
        t0 = time.time()
        for fe_name, fe in (("qwen", q), ("live", lv)):
            if fe is None:
                continue
            data[fe_name] = {}
            for cond, snr in conditions:
                mats, spks, stats = [], [], []
                for r in chosen:
                    wav = read_wav(r["wav"])
                    if snr is not None:
                        wav = mix_noise(wav, noise_files, snr, rng)
                    M = fe.matrix(wav)
                    mats.append(M)
                    spks.append(r["speaker"])
                    stats.append(pooled_stats(M))
                data[fe_name][cond] = {"mats": mats, "speakers": spks,
                                       "stats": np.stack(stats)}
                log("%s/%s: %d matrices, shapes %s..%s (%.0fs total)" %
                    (fe_name, cond, len(mats), mats[0].shape, mats[-1].shape,
                     time.time() - t0))

        rev_chosen = chosen[:args.rev_n]
        rev = {}
        for fe_name, fe in (("qwen", q), ("live", lv)):
            if fe is None:
                continue
            stats = []
            for r in rev_chosen:
                wav = read_wav(r["wav"])[::-1].copy()
                stats.append(pooled_stats(fe.matrix(wav)))
            rev[fe_name] = np.stack(stats)
            log("%s: reversed matrices for %d utterances" % (fe_name, len(rev_chosen)))
        np.save(cache_path, {"data": data, "rev": rev, "chosen": chosen}, allow_pickle=True)
        log("cached matrices -> %s" % cache_path)
    speakers = sorted({r["speaker"] for r in chosen})
    conditions = [("clean", None), ("noise10", 10.0), ("noise0", 0.0)]
    # -------- metrics --------
    results = {}
    meme_vecs, meme_names = load_meme_vectors(os.path.join(root, MEME_SERIAL))
    for fe_name, fe in (("qwen", q), ("live", lv)):
        if fe is None:
            continue
        clean = data[fe_name]["clean"]
        stats = clean["stats"]
        spk = clean["speakers"]
        n_spk = len(set(spk))
        res = {"frontend": fe.describe(),
               "true_frame_space": fe.true_frame_space}
        # U1: speaker-disjoint split
        held_speakers = set(sorted(set(spk))[-max(1, n_spk // 4):])
        tr_mats = [M for M, s in zip(clean["mats"], spk) if s not in held_speakers]
        he_mats = [M for M, s in zip(clean["mats"], spk) if s in held_speakers]
        res["U1_next_frame"] = u1_next_frame(tr_mats, he_mats) if len(he_mats) else {"error": "no held-out"}
        # U2 speaker separability (clean)
        res["U2_speaker_clustering"] = u2_env_separability(stats, spk, k=n_spk)
        # U2 noise-condition separability (cluster across 3 conditions)
        all_stats, cond_labels = [], []
        for ci, (cond, _snr) in enumerate(conditions):
            all_stats.append(data[fe_name][cond]["stats"])
            cond_labels += [ci] * len(data[fe_name][cond]["speakers"])
        res["U2_noise_condition_clustering"] = u2_env_separability(
            np.concatenate(all_stats, 0), cond_labels, k=len(conditions))
        # U3 time arrow
        res["U3_time_arrow"] = u3_time_arrow(stats[:len(rev[fe_name])], rev[fe_name])
        # U4 meme proxy
        res["U4_meme_alignment_proxy"] = u4_meme_alignment(stats, meme_vecs)
        results[fe_name] = res
        log("%s metrics: U1 %s | U2 spk ARI %.3f NMI %.3f | U2 noise ARI %.3f | U3 %.3f | U4 %s" %
            (fe_name,
             {k: round(v, 5) for k, v in res["U1_next_frame"].items()
              if k in ("ratio_vs_identity", "ratio_vs_mean", "r2_vs_identity")},
             res["U2_speaker_clustering"]["ari"], res["U2_speaker_clustering"]["nmi"],
             res["U2_noise_condition_clustering"]["ari"],
             res["U3_time_arrow"]["bal_acc_mean"],
             {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in res["U4_meme_alignment_proxy"].items()
              if k in ("mean_top1_cosine", "mean_all_cosine", "available")}))

    # -------- manifest --------
    manifest = {
        "tool": "tools/audio_signal_matrix.py",
        "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "design_doc": "doc/v10.1/io_signal_matrix_design.md",
        "contract": "audio = 40 ms/frame x 4096 (multimodal_world_model.hpp:35-47); "
                    "128 is only the projected interface view",
        "frontends": {"qwen": q.describe() if q else None, "live": lv.describe() if lv else None},
        "fsc_subset": {"split": "test (official, 10 unseen speakers)",
                       "n_per_speaker": args.n_per_speaker,
                       "utterances": len(chosen), "speakers": speakers,
                       "conditions": [c for c, _ in conditions]},
        "metrics": results,
        "runtime_seconds_total": round(time.time() - t0_metrics, 1),
        "limitations": [
            "Qwen front-end verified as a TRUE 40 ms x 4096 frame space (I/O and 750/30s rate); "
            "the live speech encoder CANNOT produce it (1 s pooled 128-d; fallback 4096 is an "
            "arbitrary projection).",
            "FSC subset is small (n_per_speaker x 10 speakers) because Qwen2-Audio encoder "
            "inference is ~9 s/utterance on CPU (95M+635M params); metrics are indicative.",
            "U4 is a PROXY: learned alignment projections W_t/W_g are unavailable offline; "
            "meme vectors are hash-based node embeddings reproduced from gnn_meme_serial.json "
            "(12 groups), and audio stats are hash-projected to 512-d.",
            "U2 noise-condition separability uses synthetic MUSAN noise conditions, not real "
            "rooms; speaker labels are the only real environment proxy in FSC.",
            "Time-reversal (U3) is run through the encoder honestly (reversed waveform), not by "
            "reversing cached frames.",
        ],
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    # save pooled stats + a couple of matrices for reproducibility
    np.savez_compressed(os.path.join(run_dir, "fsc_subset_stats.npz"),
                        **{("%s_%s_stats" % (k, c)): v["stats"]
                           for k, cd in data.items() for c, v in cd.items()})
    log("manifest: %s" % os.path.join(run_dir, "manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

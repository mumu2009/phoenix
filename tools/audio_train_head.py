#!/usr/bin/env python3
"""audio_train_head.py - real-data audio head training on the frozen Phoenix speech encoder.

Data: Fluent Speech Commands (FSC), downloaded locally from ModelScope
(OmniData/Fluent_Speech_Commands, CC BY-NC 4.0, 1.54 GB archive -> 30 043 wavs, 97
speakers, per-utterance action/object/location/intent + transcript + speakerId).
See doc/v10.1/audio_training_data.md for provenance, sha256 and licence.

Objectives (--objective):
  classification (default): classify a real per-utterance field
      (--label-field intent|action|object|location) with a small head on the frozen
      encoder; held-out = official FSC test split (speaker-independent, 10 unseen speakers).
  denoise: concept-space denoising -- clean FSC windows vs the same windows mixed with
      local MUSAN noise at --snr-db; head H: 128->128 trained with
      loss = 1 - cos(H(z_noisy), z_clean); evaluated by paired retrieval top-1/top-5
      (retrieve the clean counterpart among held-out utterances) and a ridge probe
      H(z_noisy) -> z_clean.  Also supports --dataset-format speech_commands (word labels).

Frozen encoder: runtime_store/models/additive_jepa/speech_encoder/best.onnx
waveform [batch,1,1,16000] -> concept [batch,128] (16 kHz, 1 s).  The `additive_jepa`
directory name is LEGACY: this is an in-house llava/ResNet-era model, NOT Meta I-JEPA.
FSC utterances are 1-4 s, so --crops N mean-pools N evenly spaced 1 s windows per
utterance (the label refers to the whole utterance; documented compromise).

Protocol: 3 seeds on a fixed split -> mean ± sd; read-outs always include top-1, top-5
and a closed-form ridge probe (for denoise the probe maps noisy->clean concept).
An MLP that buys top-1 but collapses top-5/probe is a NO-GO (the image lesson).
A head is DEPLOYABLE only if it beats the identity baseline on top-1 AND does not
regress top-5/probe, AND clears the trivial baseline (classification: >= max(2x chance,
majority+5pp)) or preserves instance retrieval (denoise: paired top-1/top-5).

Outputs: runtime_store/models/io_train2/audio_<timestamp>/{manifest.json, head_*.npz/json,
features_*.npz}, plus a C++ loader contract (.f32 + contract.json) when a rung passes.
Current real-data verdict: NOT DEPLOY (see doc/v10.1/audio_training_data.md section 5).

Usage (from the phoenix root):
  runtime_store\\venvs\\vision-cu111\\Scripts\\python.exe tools\\audio_train_head.py ^
      --dataset-dir runtime_store/datasets/audio/fluent_speech_commands --crops 3 ^
      --train-per-class 200 --epochs 20 --label-field intent --run-name fsc_intent
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_train_head as T  # noqa: E402
import io_scaleup as S  # noqa: E402

ENCODER = "runtime_store/models/additive_jepa/speech_encoder/best.onnx"
ENCODER_MANIFEST = "runtime_store/models/additive_jepa/speech_encoder/model.manifest.json"
SAMPLES = 16000
BASE_SEED = 20261118
GUARD_EPS = 0.001


# --------------------------------------------------------------------------
# dataset discovery
# --------------------------------------------------------------------------
def find_dataset_root(dataset_dir):
    for candidate in (dataset_dir, os.path.join(dataset_dir, "speech_commands_v0.02"),
                      os.path.join(dataset_dir, "speech_commands")):
        if os.path.isfile(os.path.join(candidate, "testing_list.txt")):
            return candidate
    for dp, _dn, fns in os.walk(dataset_dir):
        if "testing_list.txt" in fns:
            return dp
    raise SystemExit("testing_list.txt not found under %s" % dataset_dir)


def read_file_list(root, name):
    path = os.path.join(root, name)
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rel = line.strip().replace("\\", "/")
            if rel:
                out.append(rel)
    return out


def list_classes(root):
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
    return classes


def wav_files_by_class(root, classes):
    out = {}
    for ci, c in enumerate(classes):
        d = os.path.join(root, c)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        out[c] = [c + "/" + f for f in files]
    return out


# --------------------------------------------------------------------------
# Fluent Speech Commands (ModelScope OmniData/Fluent_Speech_Commands, CC BY-NC 4.0)
# --------------------------------------------------------------------------
def find_fsc_root(dataset_dir):
    for candidate in (dataset_dir, os.path.join(dataset_dir, "fluent_speech_commands_dataset")):
        if os.path.isdir(candidate):
            for dp, _dn, fns in os.walk(candidate):
                csvs = [f for f in fns if f.lower().endswith(".csv") and
                        any(k in f.lower() for k in ("train", "valid", "test"))]
                if len(csvs) >= 2:
                    return candidate
    for dp, _dn, fns in os.walk(dataset_dir):
        csvs = [f for f in fns if f.lower().endswith(".csv")]
        if csvs:
            return dp
    raise SystemExit("FSC csv splits not found under %s" % dataset_dir)


def _fsc_resolve_wav(root, csv_dir, rel):
    rel = rel.replace("\\", "/")
    candidates = [os.path.join(root, rel), os.path.join(root, "wavs", rel),
                  os.path.join(csv_dir, rel), os.path.join(csv_dir, "..", rel)]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isfile(c):
            return c
    return None


def read_fsc_csvs(root):
    """Returns list of dicts {path, speaker, label_str, action, object, location}."""
    csvs = {}
    for dp, _dn, fns in os.walk(root):
        for f in fns:
            low = f.lower()
            if not low.endswith(".csv"):
                continue
            if "train" in low:
                csvs.setdefault("train", os.path.join(dp, f))
            elif "valid" in low or "dev" in low:
                csvs.setdefault("valid", os.path.join(dp, f))
            elif "test" in low:
                csvs.setdefault("test", os.path.join(dp, f))
    if "train" not in csvs or "test" not in csvs:
        raise SystemExit("FSC: need at least train+test csv, found %s" % list(csvs))
    import csv as _csv
    rows, seen_cols = {}, None
    for split, path in csvs.items():
        csv_dir = os.path.dirname(path)
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = _csv.DictReader(f)
            cols = [c.lower().strip() for c in (reader.fieldnames or [])]
            if seen_cols is None:
                seen_cols = cols
            out = []
            for r in reader:
                low = {k.lower().strip(): v for k, v in r.items() if k}
                rel = low.get("path") or low.get("file") or low.get("audio")
                if not rel:
                    continue
                wav = _fsc_resolve_wav(root, csv_dir, rel)
                if wav is None:
                    continue
                action = (low.get("action") or "").strip()
                obj = (low.get("object") or "").strip()
                loc = (low.get("location") or "").strip()
                intent = (low.get("intent_class") or low.get("intent") or "").strip()
                label_str = intent if intent else "|".join([action, obj, loc])
                out.append({"path": wav,
                            "speaker": (low.get("speakerid") or low.get("speaker_id") or
                                        low.get("speaker") or "").strip(),
                            "label_str": label_str, "action": action,
                            "object": obj, "location": loc})
            rows[split] = out
    return rows, seen_cols


def build_fsc_splits(dataset_dir, train_per_class, seed, label_field="intent"):
    root = find_fsc_root(dataset_dir)
    rows, cols = read_fsc_csvs(root)

    def key_of(r):
        if label_field == "intent":
            return r["label_str"]
        if label_field == "speaker":
            return r["speaker"]
        return r.get(label_field, "") or "none"

    labels = sorted({key_of(r) for r in rows["train"]})
    label_idx = {l: i for i, l in enumerate(labels)}
    rng = random.Random(seed)
    by_class = {}
    for r in rows["train"]:
        by_class.setdefault(key_of(r), []).append(r)
    train_items = []
    for lab in labels:
        rs = sorted(by_class.get(lab, []), key=lambda x: x["path"])
        rng.shuffle(rs)
        take = rs if train_per_class <= 0 else rs[:train_per_class]
        train_items.extend((r["path"], label_idx[lab]) for r in take)
    def to_items(split):
        out = []
        for r in rows.get(split, []):
            k = key_of(r)
            if k in label_idx:
                out.append((r["path"], label_idx[k]))
        return out
    val_items = to_items("valid")
    held_items = to_items("test")
    tr_speakers = {r["speaker"] for r in rows["train"]}
    te_speakers = {r["speaker"] for r in rows["test"]}
    overlap = tr_speakers & te_speakers
    info = {"root": root, "csv_columns": cols, "splits": {k: len(v) for k, v in rows.items()},
            "split": "official FSC speaker-independent train/valid/test CSV splits",
            "labels": labels, "label_field": label_field,
            "label_source": ("intent_class (fallback action|object|location)" if label_field == "intent"
                             else "FSC per-utterance '%s' field" % label_field),
            "train_speakers": len(tr_speakers), "test_speakers": len(te_speakers),
            "speaker_overlap": sorted(overlap),
            "speaker_disjoint": len(overlap) == 0}
    return labels, train_items, val_items, held_items, info


def build_splits(root, classes, train_per_class, val_per_class, seed):
    by_class = wav_files_by_class(root, classes)
    test_rel = set(read_file_list(root, "testing_list.txt"))
    val_rel = set(read_file_list(root, "validation_list.txt"))
    rng = random.Random(seed)
    train_items, val_items, held_items = [], [], []
    for ci, c in enumerate(classes):
        files = [p for p in by_class[c] if p not in test_rel and p not in val_rel]
        held = [p for p in by_class[c] if p in test_rel]
        val_pool = [p for p in by_class[c] if p in val_rel]
        rng.shuffle(files)
        rng.shuffle(held)
        rng.shuffle(val_pool)
        train_items.extend((os.path.join(root, p), ci) for p in files[:train_per_class])
        val_items.extend((os.path.join(root, p), ci) for p in val_pool[:val_per_class])
        held_items.extend((os.path.join(root, p), ci) for p in held)
    return train_items, val_items, held_items


# --------------------------------------------------------------------------
# audio features
# --------------------------------------------------------------------------
def read_wav_16k(path):
    """16 kHz mono 16-bit -> exactly 16000 samples; centre-crop long clips, zero-pad short."""
    try:
        with wave.open(path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                return None
            n = wf.getnframes()
            if n > SAMPLES:
                start = (n - SAMPLES) // 2      # centre crop (encoder is fixed at 1 s)
                wf.setpos(start)
                raw = wf.readframes(SAMPLES)
            else:
                raw = wf.readframes(n)
    except Exception:
        return None
    if len(raw) < 2:
        return None
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if a.size < SAMPLES:
        a = np.pad(a, (0, SAMPLES - a.size))
    else:
        a = a[:SAMPLES]
    return a


def read_wav_full_16k(path, max_seconds=120):
    """Read up to max_seconds of a 16 kHz mono 16-bit wav as float32 (no crop)."""
    try:
        with wave.open(path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                return None
            n = min(wf.getnframes(), 16000 * max_seconds)
            raw = wf.readframes(n)
    except Exception:
        return None
    if len(raw) < 2:
        return None
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def crop_windows(a, n_crops):
    """Return n_crops fixed 16000-sample windows evenly spaced over the utterance."""
    n = int(a.size)
    if n <= SAMPLES:
        return [np.pad(a, (0, SAMPLES - n)) if n < SAMPLES else a[:SAMPLES]]
    if n_crops <= 1:
        starts = [(n - SAMPLES) // 2]
    else:
        starts = np.linspace(0, n - SAMPLES, n_crops).astype(np.int64).tolist()
    return [a[s:s + SAMPLES] for s in starts]


def expand_crops(items, n_crops):
    """FSC utterances are longer than the encoder's fixed 1 s: mean-pool over n_crops windows."""
    exp, idx = [], []
    for i, (path, label) in enumerate(items):
        a = read_wav_full_16k(path)
        if a is None:
            continue
        for w in crop_windows(a, n_crops):
            exp.append((w, label))
            idx.append(i)
    return exp, idx


def reduce_crops(x_exp, y_exp, exp_idx, n_items):
    """Mean-pool crop features back to one feature vector per original utterance."""
    dim = x_exp.shape[1]
    out = np.zeros((n_items, dim), np.float32)
    cnt = np.zeros(n_items, np.float32)
    labels = np.full(n_items, -1, np.int64)
    for row, oi in enumerate(exp_idx):
        out[oi] += x_exp[row]
        cnt[oi] += 1.0
        labels[oi] = y_exp[row]
    cnt[cnt == 0] = 1.0
    out /= cnt[:, None]
    return out, labels


def _item_path(item, root):
    p = item[0]
    if isinstance(p, str):
        return os.path.relpath(p, root).replace(os.sep, "/")
    return "crop"


def extract_features(root, run_dir, sess, in_name, out_name, items, tag, batch=128, cache=True):
    cache_path = os.path.join(run_dir, "features_%s.npz" % tag)
    if cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        T.log("reusing feature cache %s (%d samples)" % (cache_path, z["x"].shape[0]))
        return z["x"], z["y"], list(z["paths"])
    xs, ys, paths = [], [], []
    t0 = time.time()
    buf = np.zeros((batch, 1, 1, SAMPLES), np.float32)
    n_in_buf, kept = 0, 0
    for i, (path, label) in enumerate(items):
        if isinstance(path, np.ndarray):
            wav = path
        else:
            wav = read_wav_16k(path)
        if wav is None:
            continue
        buf[n_in_buf, 0, 0, :] = wav
        n_in_buf += 1
        kept += 1
        if n_in_buf == batch:
            y = sess.run([out_name], {in_name: buf[:n_in_buf]})[0]
            xs.append(np.asarray(y, np.float32).reshape(n_in_buf, -1))
            ys.extend([label] * n_in_buf)
            paths.extend([_item_path(items[j], root)
                          for j in range(i - n_in_buf + 1, i + 1)])
            n_in_buf = 0
        if (i + 1) % 5000 == 0:
            T.log("  %s features %d/%d (%.1fs)" % (tag, i + 1, len(items), time.time() - t0))
    if n_in_buf:
        y = sess.run([out_name], {in_name: buf[:n_in_buf]})[0]
        xs.append(np.asarray(y, np.float32).reshape(n_in_buf, -1))
        ys.extend([label] * n_in_buf)
        paths.extend([_item_path(items[j], root)
                      for j in range(len(items) - n_in_buf, len(items))])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, np.int64)
    paths = np.asarray(paths)
    T.log("%s features: %d x %d in %.1fs" % (tag, x.shape[0], x.shape[1], time.time() - t0))
    if cache:
        np.savez_compressed(cache_path, x=x, y=y, paths=paths)
    return x, y, list(paths)


# --------------------------------------------------------------------------
# loader contract
# --------------------------------------------------------------------------
def write_contract(out_dir, kind, head, in_dim, out_dim, meta):
    if kind == "linear":
        arrays = [("W", head.W), ("b", head.b)]
        tensors = [{"name": "W", "shape": list(head.W.shape), "order": "rowmajor"},
                   {"name": "b", "shape": list(head.b.shape), "order": "rowmajor"}]
        typ, activation = "linear", "none"
    else:
        arrays = [("W1", head.W1), ("b1", head.b1), ("W2", head.W2), ("b2", head.b2)]
        tensors = [{"name": n, "shape": list(np.asarray(a).shape), "order": "rowmajor"}
                   for n, a in arrays]
        typ, activation = "mlp", "relu"
    f32 = os.path.join(out_dir, "head_deploy.f32")
    with open(f32, "wb") as f:
        for _n, a in arrays:
            f.write(np.asarray(a, np.float32).reshape(-1).tobytes())
    doc = {
        "format": "phoenix.io_head.v1", "modality": "speech", "type": typ,
        "inDim": int(in_dim), "outDim": int(out_dim), "normalize": True,
        "activation": activation, "weightFile": os.path.basename(f32),
        "weightDtype": "float32-le", "weightOrder": "rowmajor",
        "tensors": tensors, "weightBytes": os.path.getsize(f32),
        "weightSha256": T.sha256_file(f32),
        "applyAfter": "raw encoder concept vector (before normalizeVector)",
        "inputShape": [int(in_dim)], "outputShape": [int(out_dim)],
        "referenceEval": meta,
        "configKey": "jepa.speech.headCheckpoint",
    }
    cpath = os.path.join(out_dir, "head_deploy.contract.json")
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return {"contract_json": cpath, "weight_file": f32,
            "weight_sha256": doc["weightSha256"], "weight_bytes": doc["weightBytes"],
            "type": typ, "dims": {"inDim": int(in_dim), "outDim": int(out_dim)}}


# ==========================================================================
# OBJECTIVE (b): denoise / reconstruction in concept space
#   frozen encoder: z_clean = enc(clean 1s), z_noisy = enc(clean + MUSAN noise)
#   head H: 128 -> 128 trained so cos(H(z_noisy), z_clean) -> 1
#   eval: paired retrieval top-1/top-5 (retrieve the clean counterpart among
#   held-out utterances) + ridge probe mapping H(z_noisy) -> z_clean (MSE/cosine)
# ==========================================================================
_NOISE_CACHE = {}


def load_noise_files(noise_dir):
    files = []
    for dp, _dn, fns in os.walk(noise_dir):
        files.extend(os.path.join(dp, f) for f in fns if f.lower().endswith(".wav"))
    files.sort()
    if not files:
        raise SystemExit("no MUSAN noise wavs under %s" % noise_dir)
    return files


def _noise_cache(path):
    if path not in _NOISE_CACHE:
        _NOISE_CACHE[path] = read_wav_full_16k(path, max_seconds=120)
    return _NOISE_CACHE[path]


def add_noise(clean, rng, noise_files, snr_db=10.0):
    """Mix a random MUSAN noise segment at the given SNR (RMS-normalised)."""
    n = clean.size
    for _try in range(8):
        path = noise_files[rng.randrange(len(noise_files))]
        a = _noise_cache(path)
        if a is None or a.size < n:
            continue
        off = rng.randrange(0, a.size - n + 1)
        noise = a[off:off + n].copy()
        cr = np.sqrt(np.mean(clean * clean) + 1e-12)
        nr = np.sqrt(np.mean(noise * noise) + 1e-12)
        noise = noise * (cr / nr) / (10.0 ** (snr_db / 20.0))
        out = clean + noise
        m = np.max(np.abs(out))
        if m > 1.0:
            out = out / m
        return out.astype(np.float32)
    return clean


class ProjHead:
    def __init__(self, in_dim, out_dim, seed, hidden=0, dropout=0.0):
        rng = np.random.RandomState(seed)
        self.in_dim, self.out_dim, self.hidden, self.dropout = in_dim, out_dim, hidden, dropout
        self.W = np.eye(out_dim, in_dim, dtype=np.float32)
        self.b = np.zeros(out_dim, np.float32)
        if hidden:
            self.W1 = (rng.randn(hidden, in_dim) * np.sqrt(2.0 / in_dim)).astype(np.float32)
            self.b1 = np.zeros(hidden, np.float32)
            self.W2 = (rng.randn(out_dim, hidden) * np.sqrt(2.0 / hidden)).astype(np.float32)
            self.b2 = np.zeros(out_dim, np.float32)
        self.rng = np.random.RandomState(seed + 3)

    def embed(self, x):
        if not self.hidden:
            return T.l2_normalize(x @ self.W.T + self.b)
        h = np.maximum(x @ self.W1.T + self.b1, 0.0)
        return T.l2_normalize(h @ self.W2.T + self.b2)

    def params(self):
        return [self.W, self.b] if not self.hidden else [self.W1, self.b1, self.W2, self.b2]

    def n_params(self):
        return int(sum(p.size for p in self.params()))

    def step(self, xn, xc, wd):
        c = T.l2_normalize(xc)
        if not self.hidden:
            u = xn @ self.W.T + self.b
            n = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8)
            e = u / n
            dots = (e * c).sum(axis=1, keepdims=True)
            loss = float((1.0 - dots).mean())
            g_e = -c / xn.shape[0]
            g_u = (g_e - (g_e * e).sum(axis=1, keepdims=True) * e) / n
            gW = g_u.T @ xn + wd * self.W
            gb = g_u.sum(axis=0) + wd * self.b
            return loss, [gW, gb]
        z1 = xn @ self.W1.T + self.b1
        a1 = np.maximum(z1, 0.0)
        if self.dropout > 0:
            keep = (self.rng.rand(*a1.shape) >= self.dropout).astype(np.float32)
            a1 = a1 * keep / (1.0 - self.dropout)
        u = a1 @ self.W2.T + self.b2
        n = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8)
        e = u / n
        loss = float((1.0 - (e * c).sum(axis=1)).mean())
        g_e = -c / xn.shape[0]
        g_u = (g_e - (g_e * e).sum(axis=1, keepdims=True) * e) / n
        gW2 = g_u.T @ a1 + wd * self.W2
        gb2 = g_u.sum(axis=0) + wd * self.b2
        g_a1 = g_u @ self.W2
        if self.dropout > 0:
            g_a1 = g_a1 * keep / (1.0 - self.dropout)
        g_z1 = g_a1 * (z1 > 0)
        gW1 = g_z1.T @ xn + wd * self.W1
        gb1 = g_z1.sum(axis=0) + wd * self.b1
        return loss, [gW1, gb1, gW2, gb2]


def train_proj(xn, xc, vn, vc, in_dim, hidden, epochs, lr, wd, batch, seed):
    head = ProjHead(in_dim, in_dim, seed, hidden=hidden, dropout=0.2 if hidden else 0.0)
    ps = head.params()
    m = [np.zeros_like(p) for p in ps]
    v = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.RandomState(seed)
    n = xn.shape[0]
    steps = max(1, (n + batch - 1) // batch)
    step, curve, best = 0, [], (-1.0, 0, None)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        tot, seen = 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            ce, grads = head.step(xn[idx], xc[idx], wd)
            step += 1
            lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * step / max(1, epochs * steps)))
            for i, g in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                ps[i][...] -= lr_t * (m[i] / (1 - b1 ** step)) / (np.sqrt(v[i] / (1 - b2 ** step)) + eps)
            tot += ce * len(idx)
            seen += len(idx)
        vc_cos = float((head.embed(vn) * T.l2_normalize(vc)).sum(axis=1).mean())
        curve.append({"epoch": epoch, "train_loss": tot / max(1, seen), "val_cosine": vc_cos})
        if vc_cos > best[0]:
            best = (vc_cos, epoch, [p.copy() for p in head.params()])
    if best[2] is not None:
        for i, p in enumerate(head.params()):
            p[...] = best[2][i]
    return head, curve, best[1], time.time() - t0


def paired_metrics(pred, clean):
    P, C = T.l2_normalize(pred), T.l2_normalize(clean)
    sims = P @ C.T
    order = np.argsort(-sims, axis=1)
    n = len(P)
    top1 = float((order[:, 0] == np.arange(n)).mean())
    top5 = float(np.mean([i in order[i, :5] for i in range(n)]))
    cos = float((P * C).sum(axis=1).mean())
    return {"paired_top1": top1, "paired_top5": top5, "concept_cosine": cos}


def ridge_to_clean(pred, clean, lam=1e-2):
    """Closed-form ridge probe: held-out pred -> clean concept (MSE / cosine)."""
    X = T.l2_normalize(pred[0].astype(np.float32))
    Y = T.l2_normalize(clean[0].astype(np.float32))
    d = X.shape[1]
    W = np.linalg.solve(X.T @ X + lam * np.eye(d, dtype=np.float32), X.T @ Y)
    Xte = T.l2_normalize(pred[1].astype(np.float32))
    Yte = T.l2_normalize(clean[1].astype(np.float32))
    P = Xte @ W
    mse = float(((P - Yte) ** 2).mean())
    cos = float((T.l2_normalize(P) * Yte).sum(axis=1).mean())
    return mse, cos


def run_denoise(args, root, train_items, val_items, held_items, ds_info):
    noise_files = load_noise_files(os.path.join(root, args.noise_dir))
    T.log("denoise objective: %d MUSAN noise files, SNR=%g dB" % (len(noise_files), args.snr_db))
    ts = args.run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, args.out, "audio_%s" % ts)
    os.makedirs(run_dir, exist_ok=True)
    T.log("run dir: %s" % run_dir)
    enc_path = os.path.join(root, ENCODER)
    sess = T.make_session(enc_path, args.threads)
    in_name, out_name = T.encoder_io(sess)
    rng = random.Random(BASE_SEED + 5)

    def build(split_items, tag):
        exp, idx = expand_crops(split_items, args.crops)
        noisy = [(add_noise(w, rng, noise_files, args.snr_db), lbl) for (w, lbl) in exp]
        xc, yc, _ = extract_features(root, run_dir, sess, in_name, out_name, exp,
                                     tag + "_crops", args.feature_batch)
        xn, yn, _ = extract_features(root, run_dir, sess, in_name, out_name, noisy,
                                     tag + "_noisy_crops", args.feature_batch)
        xc, yc = reduce_crops(xc, yc, idx, len(split_items))
        xn, yn = reduce_crops(xn, yn, idx, len(split_items))
        keep = (yc >= 0) & (yn >= 0)
        return xn[keep], xc[keep]

    xn_tr, xc_tr = build(train_items, "train")
    xn_va, xc_va = build(val_items, "val")
    xn_te, xc_te = build(held_items, "held")
    T.log("denoise features: train %s held %s" % (xn_tr.shape, xn_te.shape))
    base = paired_metrics(xn_te, xc_te)
    base_probe = ridge_to_clean((xn_tr, xn_te), (xc_tr, xc_te))
    T.log("baseline identity: %s probe mse=%.4f cos=%.4f" %
          (json.dumps({k: round(v, 4) for k, v in base.items()}, sort_keys=True),
           base_probe[0], base_probe[1]))

    rungs = []
    for name in ("a_linear128", "b_mlp512"):
        rows = []
        for seed in args.seeds:
            hidden = 0 if name == "a_linear128" else 512
            head, curve, best_ep, train_s = train_proj(
                xn_tr, xc_tr, xn_va, xc_va, xn_tr.shape[1], hidden,
                args.epochs, 0.01 if hidden else 0.02, 1e-4, args.batch, seed)
            pred = head.embed(xn_te)
            m = paired_metrics(pred, xc_te)
            rmse, rcos = ridge_to_clean((head.embed(xn_tr), pred), (xc_tr, xc_te))
            rows.append({"seed": seed, "params": head.n_params(),
                         "train_time_s": round(train_s, 2), "best_epoch": int(best_ep),
                         **m, "ridge_mse": rmse, "ridge_cosine": rcos})
            T.log("  [%s seed=%d] top1=%.4f top5=%.4f cos=%.4f ridge_mse=%.4f ridge_cos=%.4f" %
                  (name, seed, m["paired_top1"], m["paired_top5"], m["concept_cosine"], rmse, rcos))
        rungs.append({"rung": name, "params": rows[0]["params"],
                      "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
                      "seeds": rows,
                      "agg": S.agg(rows, ["paired_top1", "paired_top5", "concept_cosine",
                                          "ridge_mse", "ridge_cosine"])})

    deployable = []
    for r in rungs:
        a = r["agg"]
        ok = (a["paired_top1"]["mean"] > base["paired_top1"]
              and a["paired_top5"]["mean"] >= base["paired_top5"] - GUARD_EPS
              and a["ridge_mse"]["mean"] <= base_probe[0] * 1.02)
        r["deployable"] = bool(ok)
        if ok:
            deployable.append(r)
    chosen = max(deployable, key=lambda r: r["agg"]["paired_top1"]["mean"]) if deployable else None
    verdict = "DEPLOY" if chosen else "NOT DEPLOY"

    for r in rungs:
        best = max(r["seeds"], key=lambda x: x["paired_top1"])
        if r["rung"] == "a_linear128":
            head, _c, _e, _t = train_proj(xn_tr, xc_tr, xn_va, xc_va, xn_tr.shape[1], 0,
                                          args.epochs, 0.02, 1e-4, args.batch, best["seed"])
            r["checkpoint"] = S.save_head(run_dir, root, r["rung"], head, "linear",
                                          xn_tr.shape[1], xn_tr.shape[1])
            kind = "linear"
        else:
            head, _c, _e, _t = train_proj(xn_tr, xc_tr, xn_va, xc_va, xn_tr.shape[1], 512,
                                          args.epochs, 0.01, 1e-4, args.batch, best["seed"])
            r["checkpoint"] = S.save_head(run_dir, root, r["rung"], head, "mlp",
                                          xn_tr.shape[1], xn_tr.shape[1])
            kind = "mlp"
        if chosen is not None and r["rung"] == chosen["rung"]:
            meta = {"paired_top1_mean": r["agg"]["paired_top1"]["mean"],
                    "paired_top1_sd": r["agg"]["paired_top1"]["sd"],
                    "paired_top5_mean": r["agg"]["paired_top5"]["mean"],
                    "ridge_mse_mean": r["agg"]["ridge_mse"]["mean"],
                    "baseline": {"paired_top1": base["paired_top1"],
                                 "paired_top5": base["paired_top5"],
                                 "ridge_mse": base_probe[0]}}
            r["contract"] = write_contract(run_dir, kind, head, xn_tr.shape[1],
                                           xn_tr.shape[1], meta)
            T.log("contract: %s" % r["contract"]["contract_json"])

    provenance = {}
    for p in (os.path.join(ds_info["root"], "provenance.json"),
              os.path.join(os.path.dirname(ds_info["root"]), "provenance.json")):
        if os.path.isfile(p):
            try:
                provenance = json.load(open(p, encoding="utf-8"))
                break
            except Exception:
                pass
    manifest = {
        "tool": "tools/audio_train_head.py", "objective": "denoise/reconstruction in concept space",
        "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "audio", "verdict": verdict,
        "chosen_rung": chosen["rung"] if chosen else None,
        "guardrail": "deployable iff paired top-1 > baseline AND paired top-5 >= baseline-0.001 "
                     "AND ridge MSE <= baseline*1.02",
        "encoder": {"path": ENCODER, "sha256": T.sha256_file(enc_path),
                    "identity": "in-house 1D speech concept encoder (legacy additive_jepa dir)"},
        "data": {"clean": "Fluent Speech Commands 16 kHz",
                 "noise": "MUSAN noise corpus (local): free-sound + sound-bible",
                 "snr_db": args.snr_db, "crops": args.crops,
                 "train_samples": int(xn_tr.shape[0]), "heldout_samples": int(xn_te.shape[0]),
                 "split": "official FSC speaker-independent test split held-out",
                 "provenance": provenance},
        "baseline_identity": {"paired_top1": base["paired_top1"],
                              "paired_top5": base["paired_top5"],
                              "concept_cosine": base["concept_cosine"],
                              "ridge_mse": base_probe[0], "ridge_cosine": base_probe[1]},
        "hyperparameters": {"epochs": args.epochs, "batch": args.batch, "seeds": args.seeds,
                            "optimizer": "adam + cosine decay",
                            "loss": "1 - cos(H(z_noisy), z_clean)"},
        "rungs": rungs,
    }
    mpath = os.path.join(run_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    T.log("manifest: %s" % mpath)
    T.log("VERDICT: %s (chosen=%s)" % (verdict, manifest["chosen_rung"]))
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="runtime_store/datasets/audio/fluent_speech_commands")
    ap.add_argument("--dataset-format", choices=["fsc", "speech_commands"], default="fsc",
                    help="fsc = Fluent Speech Commands (intent labels); "
                         "speech_commands = Google SpeechCommands v0.02 (word labels)")
    ap.add_argument("--task", default=None,
                    help="free-form task label (default derived from --dataset-format)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="runtime_store/models/io_train2")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--train-per-class", type=int, default=200,
                    help="cap per class in train (0 = all)");
    ap.add_argument("--val-per-class", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--feature-batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", default="20261118,20261119,20261120")
    ap.add_argument("--rungs", default="a_linear128,b_mlp512,c_mlp1024_dropout")
    ap.add_argument("--objective", choices=["classification", "denoise"], default="classification")
    ap.add_argument("--snr-db", type=float, default=10.0,
                    help="denoise objective: MUSAN noise SNR in dB")
    ap.add_argument("--noise-dir", default="runtime_store/datasets/musan_extracted/musan/noise")
    ap.add_argument("--crops", type=int, default=3,
                    help="fsc only: fixed 1 s windows per utterance, mean-pooled (encoder is 1 s)")
    ap.add_argument("--label-field", default="intent",
                    choices=["intent", "action", "object", "location", "speaker"],
                    help="fsc only: which per-utterance label to classify")
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.rungs = [s for s in args.rungs.split(",") if s.strip()]
    root = os.path.abspath(args.root)
    os.chdir(root)
    ds_dir = os.path.join(root, args.dataset_dir)
    if args.dataset_format == "fsc":
        classes, train_items, val_items, held_items, ds_info = build_fsc_splits(
            ds_dir, args.train_per_class, BASE_SEED, args.label_field)
        ds_info["format"] = "fluent_speech_commands"
        ds_info["root_abs"] = ds_info["root"]
        task = args.task or ("Fluent Speech Commands %s classification" % args.label_field)
    else:
        ds_root = find_dataset_root(ds_dir)
        classes = list_classes(ds_root)
        train_items, val_items, held_items = build_splits(
            ds_root, classes, args.train_per_class, args.val_per_class, BASE_SEED)
        ds_info = {"root": ds_root, "format": "speech_commands", "class_names": classes,
                   "split": "official testing_list.txt held-out (file-level); validation_list.txt internal"}
        task = args.task or "Google Speech Commands v0.02 35-word keyword classification"
    ds_root = ds_info["root_abs"] if "root_abs" in ds_info else ds_info["root"]
    T.log("dataset (%s) root: %s" % (args.dataset_format, ds_root))
    T.log("splits: train=%d val=%d heldout=%d, classes=%d" %
          (len(train_items), len(val_items), len(held_items), len(classes)))
    if "speaker_disjoint" in ds_info:
        T.log("speaker-disjoint split: %s (train speakers=%d, test speakers=%d, overlap=%d)" %
              (ds_info["speaker_disjoint"], ds_info["train_speakers"],
               ds_info["test_speakers"], len(ds_info["speaker_overlap"])))
    if len(held_items) < 1000:
        raise SystemExit("held-out utterances < 1000 (%d); refusing to claim a real evaluation"
                         % len(held_items))
    if args.objective == "denoise":
        if args.dataset_format != "fsc":
            raise SystemExit("--objective denoise currently requires --dataset-format fsc")
        return run_denoise(args, root, train_items, val_items, held_items, ds_info)

    ts = args.run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, args.out, "audio_%s" % ts)
    os.makedirs(run_dir, exist_ok=True)
    T.log("run dir: %s" % run_dir)

    enc_path = os.path.join(root, ENCODER)
    sess = T.make_session(enc_path, args.threads)
    in_name, out_name = T.encoder_io(sess)

    x_tr, y_tr, _ = extract_features(root, run_dir, sess, in_name, out_name,
                                     train_items, "train", args.feature_batch)
    x_va, y_va, _ = extract_features(root, run_dir, sess, in_name, out_name,
                                     val_items, "val", args.feature_batch)
    x_te, y_te, te_paths = extract_features(root, run_dir, sess, in_name, out_name,
                                            held_items, "held", args.feature_batch)
    if args.dataset_format == "fsc" and args.crops > 1:
        # mean-pool frozen 1 s encoder over --crops windows per utterance
        T.log("FSC multi-crop: %d windows per utterance, mean-pooled" % args.crops)
        exp_tr, idx_tr = expand_crops(train_items, args.crops)
        xe_tr, ye_tr, _ = extract_features(root, run_dir, sess, in_name, out_name,
                                           exp_tr, "train_crops", args.feature_batch)
        x_tr, y_tr = reduce_crops(xe_tr, ye_tr, idx_tr, len(train_items))
        exp_va, idx_va = expand_crops(val_items, args.crops)
        xe_va, ye_va, _ = extract_features(root, run_dir, sess, in_name, out_name,
                                           exp_va, "val_crops", args.feature_batch)
        x_va, y_va = reduce_crops(xe_va, ye_va, idx_va, len(val_items))
        exp_te, idx_te = expand_crops(held_items, args.crops)
        xe_te, ye_te, _ = extract_features(root, run_dir, sess, in_name, out_name,
                                           exp_te, "held_crops", args.feature_batch)
        x_te, y_te = reduce_crops(xe_te, ye_te, idx_te, len(held_items))
        keep_tr = y_tr >= 0
        keep_va = y_va >= 0
        keep_te = y_te >= 0
        x_tr, y_tr = x_tr[keep_tr], y_tr[keep_tr]
        x_va, y_va = x_va[keep_va], y_va[keep_va]
        x_te, y_te = x_te[keep_te], y_te[keep_te]
        te_paths = [p for (p, _l), ok in zip(held_items, keep_te) if ok]
    in_dim = x_tr.shape[1]
    n_classes = int(max(y_tr.max(), y_te.max())) + 1
    T.log("features: train %s val %s held %s dim=%d classes=%d" %
          (x_tr.shape, x_va.shape, x_te.shape, in_dim, n_classes))

    base = S.eval_embedding(T.l2_normalize(x_tr), y_tr, T.l2_normalize(x_te), y_te)
    T.log("baseline (identity-normalised frozen encoder): %s" %
          json.dumps({k: round(v, 4) for k, v in base.items()}, sort_keys=True))

    rungs = []
    for name in args.rungs:
        rows = []
        for seed in args.seeds:
            if name == "a_linear128":
                head, curve, best_ep, train_s = S.train_linear(
                    x_tr, y_tr, x_va, y_va, in_dim, in_dim, n_classes,
                    args.epochs, 0.02, 1e-4, args.batch, seed)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = int(head.W.size + head.b.size)
            else:
                hidden, drop = (512, 0.0) if name == "b_mlp512" else (1024, 0.2)
                head, curve, best_ep, train_s = S.train_mlp(
                    x_tr, y_tr, x_va, y_va, in_dim, hidden, in_dim, n_classes,
                    args.epochs, 0.005, 1e-4, args.batch, seed, drop)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = head.n_params()
            m = S.eval_embedding(z_tr, y_tr, z_te, y_te)
            rows.append({"seed": seed, "params": params, "train_time_s": round(train_s, 2),
                         "best_epoch": int(best_ep), "final_val_ce": round(curve[-1]["val_ce"], 4),
                         "final_val_acc": round(curve[-1]["val_acc"], 4), **m})
            T.log("  [%s seed=%d] params=%d train=%.1fs top1=%.4f top5=%.4f probe=%.4f" %
                  (name, seed, params, train_s, m["knn_top1"], m["knn_top5"],
                   m["linear_probe_acc"]))
        rungs.append({"rung": name, "params": rows[0]["params"],
                      "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
                      "seeds": rows,
                      "agg": S.agg(rows, ["knn_top1", "knn_top5", "linear_probe_acc",
                                          "linear_probe_mse"])})

    # deployability guardrail: improve top-1, do not regress top-5/probe beyond 0.1 pp,
    # and beat 2x chance (relative wins that hover at chance are NOT deployable).
    chance = 1.0 / max(1, n_classes)
    majority = float(np.bincount(y_te, minlength=n_classes).max()) / max(1, len(y_te))
    deployable = []
    for r in rungs:
        a = r["agg"]
        rel = (a["knn_top1"]["mean"] > base["knn_top1"]
               and a["knn_top5"]["mean"] >= base["knn_top5"] - GUARD_EPS
               and a["linear_probe_acc"]["mean"] >= base["linear_probe_acc"] - GUARD_EPS)
        practical = a["knn_top1"]["mean"] >= max(2.0 * chance, majority + 0.05)
        r["formal_guardrail_pass"] = bool(rel)
        r["practical_above_trivial"] = bool(practical)
        r["deployable"] = bool(rel and practical)
        if r["deployable"]:
            deployable.append(r)
        T.log("  [guardrail %s] relative=%s practical(>=max(2x chance, majority+5pp)=%.3f)=%s"
              % (r["rung"], rel, max(chance, majority), practical))
    chosen = max(deployable, key=lambda r: r["agg"]["knn_top1"]["mean"]) if deployable else None
    verdict = "DEPLOY" if chosen else "NOT DEPLOY"
    T.log("chance=%.4f majority=%.4f -> %s" % (chance, majority, verdict))

    # checkpoints for every rung (best seed), contract for the chosen one
    for r in rungs:
        spec = r
        best = max(spec["seeds"], key=lambda x: x["knn_top1"])
        seed = best["seed"]
        if r["rung"] == "a_linear128":
            head, _c, _e, _t = S.train_linear(x_tr, y_tr, x_va, y_va, in_dim, in_dim, n_classes,
                                              args.epochs, 0.02, 1e-4, args.batch, seed)
            kind = "linear"
        else:
            hidden, drop = (512, 0.0) if r["rung"] == "b_mlp512" else (1024, 0.2)
            head, _c, _e, _t = S.train_mlp(x_tr, y_tr, x_va, y_va, in_dim, hidden, in_dim,
                                           n_classes, args.epochs, 0.005, 1e-4, args.batch,
                                           seed, drop)
            kind = "mlp"
        r["checkpoint"] = S.save_head(run_dir, root, r["rung"], head, kind, in_dim, in_dim)
        r["best_seed"] = seed
        if chosen is not None and r["rung"] == chosen["rung"]:
            meta = {"heldout_top1_mean": r["agg"]["knn_top1"]["mean"],
                    "heldout_top1_sd": r["agg"]["knn_top1"]["sd"],
                    "heldout_top5_mean": r["agg"]["knn_top5"]["mean"],
                    "heldout_probe_acc_mean": r["agg"]["linear_probe_acc"]["mean"],
                    "baseline": base}
            r["contract"] = write_contract(run_dir, kind, head, in_dim, in_dim, meta)
            T.log("contract: %s" % r["contract"]["contract_json"])

    provenance = {}
    for prov_path in (os.path.join(ds_root, "provenance.json"),
                      os.path.join(ds_dir, "provenance.json"),
                      os.path.join(os.path.dirname(ds_root), "provenance.json")):
        if os.path.isfile(prov_path):
            try:
                provenance = json.load(open(prov_path, encoding="utf-8"))
                break
            except Exception:
                provenance = {}
    manifest = {
        "tool": "tools/audio_train_head.py",
        "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "audio", "task": task,
        "verdict": verdict, "chosen_rung": chosen["rung"] if chosen else None,
        "chance": float(chance), "majority_baseline": float(majority),
        "guardrail": "deployable iff 3-seed mean top-1 > baseline AND top-5 >= baseline-0.001 "
                     "AND probe >= baseline-0.001 AND top-1 >= max(2x chance, majority+5pp)",
        "encoder": {
            "path": ENCODER, "bytes": os.path.getsize(enc_path),
            "sha256": T.sha256_file(enc_path),
            "identity": "in-house 1D speech concept encoder exported to ONNX (16 kHz, 16000 "
                        "samples -> 128-d); the 'additive_jepa' directory name is LEGACY, "
                        "NOT Meta I-JEPA",
            "manifest_sha256": T.sha256_file(os.path.join(root, ENCODER_MANIFEST)),
            "io": [{"name": "waveform", "shape": ["batch", 1, 1, SAMPLES]},
                   {"name": "concept", "shape": ["batch", 128, 1, 1]}],
        },
        "dataset": {
            "format": ds_info.get("format"),
            "name": ("Fluent Speech Commands (ModelScope OmniData mirror; CC BY-NC 4.0)"
                     if args.dataset_format == "fsc" else "Google Speech Commands v0.02"),
            "root": os.path.relpath(ds_root, root),
            "classes": len(classes), "class_names": list(classes),
            "label_field": ds_info.get("label_field"),
            "train_samples": int(x_tr.shape[0]), "val_samples": int(x_va.shape[0]),
            "heldout_samples": int(x_te.shape[0]),
            "heldout_files": len(set(te_paths)),
            "split": ds_info.get("split", "see dataset info"),
            "speaker_disjoint": ds_info.get("speaker_disjoint"),
            "speaker_overlap": ds_info.get("speaker_overlap"),
            "train_speakers": ds_info.get("train_speakers"),
            "test_speakers": ds_info.get("test_speakers"),
            "raw_split_counts": ds_info.get("splits"),
            "provenance": provenance,
            "label_caveat": ("official FSC speaker-independent train/valid/test split; the "
                             "encoder is fixed at 1 s so longer utterances are CENTRE-CROPPED "
                             "to 16000 samples (label may refer to audio beyond the crop)"
                             if args.dataset_format == "fsc" else
                             "official file-level split; SpeechCommands has known speaker "
                             "overlap, so this is file-disjoint, not speaker-disjoint"),
        },
        "hyperparameters": {"epochs": args.epochs, "batch": args.batch, "seeds": args.seeds,
                            "optimizer": "adam(0.9,0.999) + cosine lr decay",
                            "linear_lr": 0.02, "mlp_lr": 0.005, "weight_decay": 1e-4,
                            "early_stop": "official validation split"},
        "baseline_identity": base,
        "rungs": rungs,
    }
    mpath = os.path.join(run_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    T.log("manifest: %s" % mpath)
    T.log("VERDICT: %s (chosen=%s)" % (verdict, manifest["chosen_rung"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

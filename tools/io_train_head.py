#!/usr/bin/env python3
"""io_train_head.py - standalone training loop for the Phoenix speech/image IO
concept projection head.

This tool is deliberately decoupled from the Phoenix agent runtime:

  * it reads the *existing* ONNX artifacts under runtime_store/models/ as frozen
    feature extractors (no downloads, no re-training of the base encoder),
  * it uses only local corpora under runtime_store/datasets/,
  * it trains a small linear projection head (the same shape as the legacy
    `encoder_head.json` artifact: outDim x inDim + bias) with a supervised
    linear-probe objective (N-way classification of local labels),
  * it writes a self-contained checkpoint + manifest under
    runtime_store/models/io_train/<timestamp>/ and never touches the live model
    directories.

The live runtime path for both modalities is:
    concept = normalize(encoder(input))          # video_model/audio_model today
This tool learns an extra linear head so the future runtime path becomes:
    concept = normalize(W @ encoder(input) + b)  # same dims, same normalisation

Baseline vs trained comparison is done on a file-disjoint held-out split with
two identical read-outs (cosine 1-NN retrieval and a closed-form ridge linear
probe), so the delta measures the projection head, not the classifier.

Usage (run from the phoenix root, e.g. E:\\079\\v6.0Alixander\\phoenix):

  runtime_store\\venvs\\vision-cu111\\Scripts\\python.exe tools\\io_train_head.py \
      --modality both

  # image only, smaller run
  runtime_store\\venvs\\vision-cu111\\Scripts\\python.exe tools\\io_train_head.py \
      --modality image --train-per-class 20 --heldout-per-class 10 --epochs 15

Required python packages: numpy, onnxruntime, Pillow (image modality only).
The `vision-cu111` venv has all three; Python314 has numpy+onnxruntime but no
Pillow.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import sys
import time

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    sys.exit("[io-train] numpy is required: %s" % exc)

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    sys.exit("[io-train] onnxruntime is required: %s" % exc)

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - audio-only runs work without Pillow
    Image = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

IMAGE_ENCODER = "runtime_store/models/additive_jepa/vision_encoder/best.onnx"
IMAGE_ENCODER_MANIFEST = "runtime_store/models/additive_jepa/vision_encoder/model.manifest.json"
AUDIO_ENCODER = "runtime_store/models/additive_jepa/speech_encoder/best.onnx"
AUDIO_ENCODER_MANIFEST = "runtime_store/models/additive_jepa/speech_encoder/model.manifest.json"
IMAGE_DATASET = "runtime_store/datasets/tiny-imagenet-200/train"
AUDIO_DATASET = "runtime_store/datasets/musan_extracted/musan"
AUDIO_CLASSES = ("speech", "music", "noise")
AUDIO_SAMPLES = 16000


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def log(msg):
    print("[io-train] %s" % msg, flush=True)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_entry(path, root):
    st = os.stat(path)
    return {
        "path": os.path.relpath(path, root).replace(os.sep, "/"),
        "bytes": st.st_size,
        "sha256": sha256_file(path),
    }


def l2_normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.time()
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    log("loaded %s (%.1fs threads=%d)" % (os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path),
                                          time.time() - t0, threads))
    return sess


def encoder_io(sess):
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    log("encoder io: in=%s %s  out=%s %s" % (inp.name, list(inp.shape), out.name, list(out.shape)))
    return inp.name, out.name


# --------------------------------------------------------------------------
# data: image (tiny-imagenet-200)
# --------------------------------------------------------------------------
def list_image_classes(root):
    train_dir = os.path.join(root, IMAGE_DATASET)
    classes = sorted(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
    if not classes:
        raise SystemExit("no tiny-imagenet classes under %s" % train_dir)
    return classes


def select_image_split(root, train_per_class, heldout_per_class, seed):
    classes = list_image_classes(root)
    rng = random.Random(seed)
    train, held = [], []
    for ci, cls in enumerate(classes):
        d = os.path.join(root, IMAGE_DATASET, cls, "images")
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png")))
        rng.shuffle(files)
        need = train_per_class + heldout_per_class
        if len(files) < need:
            raise SystemExit("class %s has only %d images (need %d)" % (cls, len(files), need))
        train.extend((os.path.join(d, f), ci) for f in files[:train_per_class])
        held.extend((os.path.join(d, f), ci) for f in files[train_per_class:need])
    return classes, train, held


def load_image_224(path, res=224):
    if Image is None:
        raise SystemExit("Pillow is required for --modality image (use the vision-cu111 venv python)")
    im = Image.open(path).convert("RGB")
    im = im.resize((res, res), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    return ((a - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]


def extract_image_features(root, sess, items, threads, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        z = np.load(cache_path)
        log("reusing image feature cache %s (%d samples)" % (cache_path, z["x"].shape[0]))
        return z["x"], z["y"], z["paths"]
    in_name, out_name = encoder_io(sess)
    xs, ys, paths = [], [], []
    t0 = time.time()
    for i, (path, label) in enumerate(items):
        x = load_image_224(path)
        y = sess.run([out_name], {in_name: x})[0]
        xs.append(np.asarray(y, np.float32).reshape(-1))
        ys.append(label)
        paths.append(os.path.relpath(path, root).replace(os.sep, "/"))
        if (i + 1) % 500 == 0:
            log("  image features %d/%d (%.1fs)" % (i + 1, len(items), time.time() - t0))
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, np.int64)
    paths = np.asarray(paths)
    log("image features: %d x %d in %.1fs" % (x.shape[0], x.shape[1], time.time() - t0))
    if cache_path:
        np.savez_compressed(cache_path, x=x, y=y, paths=paths)
        log("wrote image feature cache %s" % cache_path)
    return x, y, paths


# --------------------------------------------------------------------------
# data: audio (MUSAN)
# --------------------------------------------------------------------------
def list_audio_files(root):
    base = os.path.join(root, AUDIO_DATASET)
    out = {}
    for cls in AUDIO_CLASSES:
        files = []
        for dirpath, _dirnames, filenames in os.walk(os.path.join(base, cls)):
            for f in filenames:
                if f.lower().endswith(".wav"):
                    files.append(os.path.join(dirpath, f))
        files.sort()
        if not files:
            raise SystemExit("no MUSAN wav files under %s" % os.path.join(base, cls))
        out[cls] = files
    return out


def read_wav_windows(path, samples, n_windows=2):
    import wave
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return []
        total = min(wf.getnframes(), 16000 * 120)
        if total < samples:
            return []
        offsets = [0]
        if n_windows > 1:
            offsets.append(max(0, total // 2))
        windows = []
        for off in offsets[:n_windows]:
            wf.setpos(off)
            raw = wf.readframes(samples)
            if len(raw) < samples * 2:
                continue
            a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            windows.append(a)
        return windows


def select_audio_split(root, file_limit_per_class, heldout_fraction, n_windows, seed):
    rng = random.Random(seed)
    files = list_audio_files(root)
    n = min(len(v) for v in files.values())
    n = min(n, file_limit_per_class) if file_limit_per_class else n
    train, held = [], []
    for ci, cls in enumerate(AUDIO_CLASSES):
        chosen = rng.sample(files[cls], n)
        n_hold = max(1, int(round(n * heldout_fraction)))
        hold_files = set(chosen[:n_hold])
        for path in chosen:
            dest = held if path in hold_files else train
            for w in read_wav_windows(path, AUDIO_SAMPLES, n_windows):
                dest.append((w, ci))
    return AUDIO_CLASSES, train, held


def extract_audio_features(root, sess, items, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        z = np.load(cache_path)
        log("reusing audio feature cache %s (%d samples)" % (cache_path, z["x"].shape[0]))
        return z["x"], z["y"]
    in_name, out_name = encoder_io(sess)
    xs, ys = [], []
    t0 = time.time()
    for i, (wav, label) in enumerate(items):
        x = wav.reshape(1, 1, 1, AUDIO_SAMPLES)
        y = sess.run([out_name], {in_name: x})[0]
        xs.append(np.asarray(y, np.float32).reshape(-1))
        ys.append(label)
        if (i + 1) % 500 == 0:
            log("  audio features %d/%d (%.1fs)" % (i + 1, len(items), time.time() - t0))
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, np.int64)
    log("audio features: %d x %d in %.1fs" % (x.shape[0], x.shape[1], time.time() - t0))
    if cache_path:
        np.savez_compressed(cache_path, x=x, y=y)
        log("wrote audio feature cache %s" % cache_path)
    return x, y


# --------------------------------------------------------------------------
# head + objective
# --------------------------------------------------------------------------
class LinearHead:
    """concept = l2_normalize(x @ W.T + b); trained with a linear-probe CE loss."""

    def __init__(self, dim, classes, seed, init="identity"):
        rng = np.random.RandomState(seed)
        if init == "identity":
            self.W = np.eye(dim, dtype=np.float32)
        else:
            self.W = (rng.randn(dim, dim) / np.sqrt(dim)).astype(np.float32)
        self.b = np.zeros(dim, np.float32)
        self.C = (rng.randn(classes, dim) / np.sqrt(dim)).astype(np.float32)  # classifier
        self.cb = np.zeros(classes, np.float32)

    def embed_raw(self, x):
        return x @ self.W.T + self.b

    def embed(self, x):
        return l2_normalize(self.embed_raw(x))

    def logits(self, x):
        return self.embed(x) @ self.C.T + self.cb

    def params(self):
        return [self.W, self.b, self.C, self.cb]

    def zero_grads(self):
        return [np.zeros_like(p) for p in self.params()]

    def forward_backward(self, x, y, wd):
        """One batch: returns (ce_loss, grads) for an unnormalised-embedding CE."""
        u = self.embed_raw(x)                      # [B, D]
        n = np.linalg.norm(u, axis=1, keepdims=True)
        n = np.maximum(n, 1e-8)
        e = u / n                                  # unit concept
        logits = e @ self.C.T + self.cb            # [B, K]
        p = softmax(logits)
        b = x.shape[0]
        ce = -np.log(np.maximum(p[np.arange(b), y], 1e-12)).mean()
        g_logits = p.copy()
        g_logits[np.arange(b), y] -= 1.0
        g_logits /= b
        gC = g_logits.T @ e
        gcb = g_logits.sum(axis=0)
        g_e = g_logits @ self.C                    # [B, D]
        # d(e)/du for row-wise l2 normalisation
        dot = (g_e * e).sum(axis=1, keepdims=True)
        g_u = (g_e - dot * e) / n
        gW = g_u.T @ x
        gb = g_u.sum(axis=0)
        # weight decay on the projection head only (classifier is discarded)
        gW = gW + wd * self.W
        gb = gb + wd * self.b
        return ce, [gW, gb, gC, gcb]


def fit_head(x_tr, y_tr, x_va, y_va, dim, classes, epochs, lr, wd, batch, seed):
    head = LinearHead(dim, classes, seed)
    m = [np.zeros_like(p) for p in head.params()]
    v = [np.zeros_like(p) for p in head.params()]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    rng = np.random.RandomState(seed)
    curve = []
    best = (float("inf"), 0, None)
    n = x_tr.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total, seen = 0.0, 0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            ce, grads = head.forward_backward(x_tr[idx], y_tr[idx], wd)
            step += 1
            lr_t = lr * (0.5 * (1.0 + np.cos(np.pi * step / max(1, epochs * ((n + batch - 1) // batch)))))
            for i, g in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                mh = m[i] / (1 - b1 ** step)
                vh = v[i] / (1 - b2 ** step)
                head.params()[i][...] -= lr_t * mh / (np.sqrt(vh) + eps)
            total += ce * len(idx)
            seen += len(idx)
        tr_ce = total / max(1, seen)
        va_logits = head.logits(x_va)
        va_ce = float(-np.log(np.maximum(softmax(va_logits)[np.arange(len(y_va)), y_va], 1e-12)).mean())
        va_acc = float((va_logits.argmax(axis=1) == y_va).mean())
        curve.append({"epoch": epoch, "train_ce": tr_ce, "val_ce": va_ce, "val_acc": va_acc})
        if va_ce < best[0]:
            best = (va_ce, epoch, [p.copy() for p in head.params()])
        log("  epoch %02d  train_ce=%.4f  val_ce=%.4f  val_acc=%.3f" % (epoch, tr_ce, va_ce, va_acc))
    if best[2] is not None:
        for i, p in enumerate(head.params()):
            p[...] = best[2][i]
    return head, curve, best[1]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def knn_metrics(x_tr, y_tr, x_te, y_te, k=5):
    xt = l2_normalize(x_te.astype(np.float32))
    xtr = l2_normalize(x_tr.astype(np.float32))
    sims = xt @ xtr.T
    top = np.argsort(-sims, axis=1)[:, :k]
    top1 = float((y_tr[top[:, 0]] == y_te).mean())
    topk = float(np.mean([y_te[i] in y_tr[top[i]] for i in range(len(y_te))]))
    return {"knn_top1": top1, "knn_top%d" % k: topk}


def ridge_probe(x_tr, y_tr, x_te, y_te, lam=1e-2):
    classes = int(y_tr.max()) + 1
    X = l2_normalize(x_tr.astype(np.float32))
    Xt = l2_normalize(x_te.astype(np.float32))
    Y = np.zeros((len(y_tr), classes), np.float32)
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float32)
    Wp = np.linalg.solve(A, X.T @ Y)
    scores = Xt @ Wp
    pred = scores.argmax(axis=1)
    acc = float((pred == y_te).mean())
    mse = float(((scores - np.eye(classes, dtype=np.float32)[y_te]) ** 2).mean())
    return {"linear_probe_acc": acc, "linear_probe_mse": mse}


def evaluate(trainz, trainy, heldz, heldy):
    out = {}
    out.update(knn_metrics(trainz, trainy, heldz, heldy))
    out.update(ridge_probe(trainz, trainy, heldz, heldy))
    return out


# --------------------------------------------------------------------------
# checkpoint writing
# --------------------------------------------------------------------------
def write_checkpoint(run_dir, root, modality, head, dim, classes, manifest):
    os.makedirs(run_dir, exist_ok=True)
    npz_path = os.path.join(run_dir, "head.npz")
    np.savez(npz_path, W=head.W, b=head.b, inDim=np.int64(dim), outDim=np.int64(dim),
             normalize=np.int64(1))
    json_path = os.path.join(run_dir, "head.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "format": "linear_projection_head_v1",
            "inDim": int(dim),
            "outDim": int(dim),
            "normalize": True,
            "W": [float(v) for v in head.W.reshape(-1)],
            "b": [float(v) for v in head.b.reshape(-1)],
        }, f)
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = dict(manifest)
    manifest["checkpoint"] = {
        "head_npz": os.path.relpath(npz_path, root).replace(os.sep, "/"),
        "head_json": os.path.relpath(json_path, root).replace(os.sep, "/"),
        "head_npz_sha256": sha256_file(npz_path),
        "head_json_sha256": sha256_file(json_path),
        "dims": {"inDim": int(dim), "outDim": int(dim), "classes": int(classes)},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log("checkpoint: %s" % npz_path)
    log("manifest:   %s" % manifest_path)
    return manifest_path


# --------------------------------------------------------------------------
# per-modality drivers
# --------------------------------------------------------------------------
def run_image(args, root, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    classes, train_items, held_items = select_image_split(
        root, args.train_per_class, args.heldout_per_class, args.seed)
    log("image split: %d train / %d held-out over %d classes (tiny-imagenet-200 train)" % (
        len(train_items), len(held_items), len(classes)))
    enc_path = os.path.join(root, IMAGE_ENCODER)
    sess = make_session(enc_path, args.threads)
    cache = os.path.join(run_dir, "features_image.npz")
    xtr, ytr, _ = extract_image_features(root, sess, train_items, args.threads, cache + ".train.npz")
    xte, yte, _ = extract_image_features(root, sess, held_items, args.threads, cache + ".held.npz")

    # deterministic 10% of train for early stopping (held-out is never touched)
    rng = np.random.RandomState(args.seed + 1)
    idx = rng.permutation(len(ytr))
    n_va = max(1, int(round(0.1 * len(ytr))))
    va_idx, tr_idx = idx[:n_va], idx[n_va:]
    dim = xtr.shape[1]
    log("training image head %dx%d, %d classes, %d train, %d internal-val" % (
        dim, dim, len(classes), len(tr_idx), len(va_idx)))
    head, curve, best_epoch = fit_head(
        xtr[tr_idx], ytr[tr_idx], xtr[va_idx], ytr[va_idx],
        dim, len(classes), args.epochs, args.lr, args.wd, args.batch, args.seed)

    base_train, base_held = l2_normalize(xtr), l2_normalize(xte)
    trained_train, trained_held = head.embed(xtr), head.embed(xte)
    base = evaluate(base_train, ytr, base_held, yte)
    trained = evaluate(trained_train, ytr, trained_held, yte)
    delta = {k: trained[k] - base[k] for k in base}
    log("image baseline : %s" % json.dumps(base, sort_keys=True))
    log("image trained  : %s" % json.dumps(trained, sort_keys=True))
    log("image delta    : %s" % json.dumps(delta, sort_keys=True))

    manifest = {
        "tool": "tools/io_train_head.py",
        "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "image",
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "pillow": getattr(__import__("PIL"), "__version__", "?"),
            "platform": platform.platform(),
        },
        "baseline": {
            "description": "normalize(frozen encoder output) - exactly what "
                           "video_model.cpp VideoLocalOnnxModel::encode returns today "
                           "when targetDim == conceptDim (128)",
            "encoder": file_entry(enc_path, root),
            "encoder_manifest": file_entry(os.path.join(root, IMAGE_ENCODER_MANIFEST), root),
        },
        "trained": {
            "head": "linear %dx%d + bias, L2-normalised output, init=identity" % (dim, dim),
            "consumption": "concept = normalize(W @ encoder(x) + b)",
        },
        "dataset": {
            "name": "tiny-imagenet-200 train split (local)",
            "root": IMAGE_DATASET,
            "classes": len(classes),
            "train_samples": int(len(tr_idx)),
            "internal_val_samples": int(len(va_idx)),
            "heldout_samples": int(len(yte)),
            "train_per_class": args.train_per_class,
            "heldout_per_class": args.heldout_per_class,
            "split": "file-disjoint, deterministic seed=%d" % args.seed,
        },
        "hyperparameters": {
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.wd,
            "batch": args.batch, "optimizer": "adam(0.9,0.999), cosine lr decay",
            "early_stopping": "best internal-val CE (epoch %d)" % best_epoch,
            "seed": args.seed,
        },
        "loss_curve": curve,
        "heldout_metrics": {"baseline": base, "trained": trained, "delta": delta},
    }
    return write_checkpoint(run_dir, root, "image", head, dim, len(classes), manifest)


def run_audio(args, root, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    classes, train_items, held_items = select_audio_split(
        root, args.audio_files_per_class, args.heldout_fraction, args.audio_windows, args.seed)
    log("audio split: %d train / %d held-out windows over %d classes (MUSAN 16k local)" % (
        len(train_items), len(held_items), len(classes)))
    enc_path = os.path.join(root, AUDIO_ENCODER)
    sess = make_session(enc_path, args.threads)
    cache = os.path.join(run_dir, "features_audio")
    xtr, ytr = extract_audio_features(root, sess, train_items, cache + ".train.npz")
    xte, yte = extract_audio_features(root, sess, held_items, cache + ".held.npz")

    rng = np.random.RandomState(args.seed + 1)
    idx = rng.permutation(len(ytr))
    n_va = max(1, int(round(0.1 * len(ytr))))
    va_idx, tr_idx = idx[:n_va], idx[n_va:]
    dim = xtr.shape[1]
    log("training audio head %dx%d, %d classes, %d train, %d internal-val" % (
        dim, dim, len(classes), len(tr_idx), len(va_idx)))
    head, curve, best_epoch = fit_head(
        xtr[tr_idx], ytr[tr_idx], xtr[va_idx], ytr[va_idx],
        dim, len(classes), args.epochs, args.lr, args.wd, args.batch, args.seed)

    base_train, base_held = l2_normalize(xtr), l2_normalize(xte)
    trained_train, trained_held = head.embed(xtr), head.embed(xte)
    base = evaluate(base_train, ytr, base_held, yte)
    trained = evaluate(trained_train, ytr, trained_held, yte)
    delta = {k: trained[k] - base[k] for k in base}
    log("audio baseline : %s" % json.dumps(base, sort_keys=True))
    log("audio trained  : %s" % json.dumps(trained, sort_keys=True))
    log("audio delta    : %s" % json.dumps(delta, sort_keys=True))

    manifest = {
        "tool": "tools/io_train_head.py",
        "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "audio",
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": os.getcwd(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "platform": platform.platform(),
        },
        "baseline": {
            "description": "normalize(frozen 1D speech-JEPA encoder output) - exactly what "
                           "audio_model.cpp AudioLocalOnnxModel::encode returns today "
                           "when targetDim == conceptDim (128)",
            "encoder": file_entry(enc_path, root),
            "encoder_manifest": file_entry(os.path.join(root, AUDIO_ENCODER_MANIFEST), root),
        },
        "trained": {
            "head": "linear %dx%d + bias, L2-normalised output, init=identity" % (dim, dim),
            "consumption": "concept = normalize(W @ encoder(waveform) + b)",
        },
        "dataset": {
            "name": "MUSAN 16 kHz (local), 3-balanced-class proxy task",
            "root": AUDIO_DATASET,
            "classes": list(classes),
            "train_samples": int(len(tr_idx)),
            "internal_val_samples": int(len(va_idx)),
            "heldout_samples": int(len(yte)),
            "files_per_class": args.audio_files_per_class,
            "windows_per_file": args.audio_windows,
            "heldout_fraction": args.heldout_fraction,
            "split": "file-disjoint, deterministic seed=%d" % args.seed,
            "label_caveat": "speech/music/noise is a coarse proxy label derived from "
                            "the MUSAN directory layout; it is not an ASR transcript task",
        },
        "hyperparameters": {
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.wd,
            "batch": args.batch, "optimizer": "adam(0.9,0.999), cosine lr decay",
            "early_stopping": "best internal-val CE (epoch %d)" % best_epoch,
            "seed": args.seed,
        },
        "loss_curve": curve,
        "heldout_metrics": {"baseline": base, "trained": trained, "delta": delta},
    }
    return write_checkpoint(run_dir, root, "audio", head, dim, len(classes), manifest)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["image", "audio", "both"], default="both")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="phoenix root (default: parent of tools/)")
    ap.add_argument("--out", default="runtime_store/models/io_train",
                    help="output root, relative to --root")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--train-per-class", type=int, default=30,
                    help="image: training images per tiny-imagenet class")
    ap.add_argument("--heldout-per-class", type=int, default=10,
                    help="image: held-out images per class")
    ap.add_argument("--audio-files-per-class", type=int, default=300,
                    help="audio: MUSAN files per class used (0 = all)")
    ap.add_argument("--audio-windows", type=int, default=2,
                    help="audio: 1-second windows taken per file")
    ap.add_argument("--heldout-fraction", type=float, default=0.3,
                    help="audio: fraction of files held out (file-disjoint)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    os.chdir(root)
    ts = args.run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(root, args.out, ts)
    os.makedirs(run_root, exist_ok=True)
    log("phoenix root: %s" % root)
    log("run dir:      %s" % run_root)

    written = []
    if args.modality in ("image", "both"):
        written.append(run_image(args, root, os.path.join(run_root, "image")))
    if args.modality in ("audio", "both"):
        written.append(run_audio(args, root, os.path.join(run_root, "audio")))
    log("done, manifests: %s" % ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())

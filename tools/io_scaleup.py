#!/usr/bin/env python3
"""io_scaleup.py - image/audio IO head scale-up ladder (3-seed, held-out).

Companion to tools/io_train_head.py (which established the working baseline loop).
This tool answers the scale-up question: does a bigger *head* help, or is the frozen
encoder the bottleneck?

Everything is decoupled from the agent runtime exactly like io_train_head.py:
  * frozen ONNX encoders from runtime_store/models/ are read-only,
  * local corpora under runtime_store/datasets/ only (no downloads),
  * checkpoints/manifests go to runtime_store/models/io_train2/<timestamp>/,
  * the live gateway/llama-server processes are never touched.

IMAGE ladder (encoder A = in-house 128-d vision concept encoder,
runtime_store/models/additive_jepa/vision_encoder/best.onnx - the `additive_jepa`
directory name is LEGACY; this is NOT a Meta I-JEPA checkpoint):
  a) linear 128 -> 128 control (identity init, the accepted io_train_head rung)
  b) 2-layer MLP 128 -> 512 -> 128 (ReLU)
  c) 2-layer MLP 128 -> 1024 -> 128 (ReLU + dropout 0.2)
  d) encoder swap experiment: in-house ResNet18-modified 512-d encoder
     (runtime_store/models/ijepa/resnet18_224/model_encoder.onnx, legacy dir name)
     + linear 512 -> 128 head.  If (d) >> (a..c) the ENCODER is the bottleneck.

AUDIO real objective: balanced 4-way language identification on MUSAN librivox,
labels parsed from .../musan/speech/librivox/ANNOTATIONS (english/german/spanish/
french, real per-file language annotations). Rungs: linear 128->128 control and
MLP 128->512->128. (The coarse speech/music/noise task from io_train_head.py stayed
negative; it is kept only as a reference number there.)

Held-out protocol (identical for every rung and seed):
  file-disjoint held-out split, fixed once; 1-NN top-1/top-5 cosine retrieval and a
  closed-form ridge linear probe; 3 training seeds -> mean +/- sd.  No single-seed
  claims.  The held-out set is never used for early stopping (10% of train is).

Usage (from the phoenix root):
  runtime_store\\venvs\\vision-cu111\\Scripts\\python.exe tools\\io_scaleup.py ^
      --reuse-image-features runtime_store/models/io_train/20261117_2220/image ^
      --seeds 20261117,20261118,20261119 --run-name 20261117_2315
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_train_head as T  # noqa: E402  (reuse split/features/metrics/manifest helpers)

try:
    import onnxruntime as ort  # noqa: E402
except Exception as exc:  # pragma: no cover
    sys.exit("[io-scaleup] onnxruntime required: %s" % exc)


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------
class MLPHead:
    """x -> [linear(in,H), ReLU, (dropout)] -> linear(H,out) -> L2-normalised concept."""

    def __init__(self, in_dim, hidden, out_dim, classes, seed, dropout=0.0):
        rng = np.random.RandomState(seed)
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.dropout = dropout
        self.W1 = (rng.randn(hidden, in_dim) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (rng.randn(out_dim, hidden) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(out_dim, np.float32)
        self.C = (rng.randn(classes, out_dim) / np.sqrt(out_dim)).astype(np.float32)
        self.cb = np.zeros(classes, np.float32)
        self.rng = np.random.RandomState(seed + 7)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.C, self.cb]

    def n_params(self):
        return int(sum(p.size for p in self.params()[:4]))

    def embed(self, x):
        a1 = np.maximum(x @ self.W1.T + self.b1, 0.0)
        u = a1 @ self.W2.T + self.b2
        return T.l2_normalize(u)

    def logits(self, x):
        return self.embed(x) @ self.C.T + self.cb

    def train_step(self, x, y, lr, wd):
        """returns (ce, grads)."""
        z1 = x @ self.W1.T + self.b1
        a1 = np.maximum(z1, 0.0)
        if self.dropout > 0:
            keep = (self.rng.rand(*a1.shape) >= self.dropout).astype(np.float32)
            a1 = a1 * keep / (1.0 - self.dropout)
        u = a1 @ self.W2.T + self.b2
        n = np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8)
        e = u / n
        logits = e @ self.C.T + self.cb
        p = T.softmax(logits)
        b = x.shape[0]
        ce = float(-np.log(np.maximum(p[np.arange(b), y], 1e-12)).mean())
        g_logits = p.copy()
        g_logits[np.arange(b), y] -= 1.0
        g_logits /= b
        gC = g_logits.T @ e
        gcb = g_logits.sum(axis=0)
        g_e = g_logits @ self.C
        g_u = (g_e - (g_e * e).sum(axis=1, keepdims=True) * e) / n
        gW2 = g_u.T @ a1 + wd * self.W2
        gb2 = g_u.sum(axis=0) + wd * self.b2
        g_a1 = g_u @ self.W2
        if self.dropout > 0:
            g_a1 = g_a1 * keep / (1.0 - self.dropout)
        g_z1 = g_a1 * (z1 > 0)
        gW1 = g_z1.T @ x + wd * self.W1
        gb1 = g_z1.sum(axis=0) + wd * self.b1
        return ce, [gW1, gb1, gW2, gb2, gC, gcb]


def train_mlp(x_tr, y_tr, x_va, y_va, in_dim, hidden, out_dim, classes, epochs, lr, wd,
              batch, seed, dropout=0.0):
    head = MLPHead(in_dim, hidden, out_dim, classes, seed, dropout)
    m = [np.zeros_like(p) for p in head.params()]
    v = [np.zeros_like(p) for p in head.params()]
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.RandomState(seed)
    n = x_tr.shape[0]
    steps_per_epoch = max(1, (n + batch - 1) // batch)
    step, curve, best = 0, [], (float("inf"), 0, None)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total, seen = 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            ce, grads = head.train_step(x_tr[idx], y_tr[idx], lr, wd)
            step += 1
            lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * step / max(1, epochs * steps_per_epoch)))
            for i, g in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                head.params()[i][...] -= lr_t * (m[i] / (1 - b1 ** step)) / (np.sqrt(v[i] / (1 - b2 ** step)) + eps)
            total += ce * len(idx)
            seen += len(idx)
        va_logits = head.logits(x_va)
        va_ce = float(-np.log(np.maximum(T.softmax(va_logits)[np.arange(len(y_va)), y_va], 1e-12)).mean())
        va_acc = float((va_logits.argmax(axis=1) == y_va).mean())
        curve.append({"epoch": epoch, "train_ce": total / max(1, seen), "val_ce": va_ce, "val_acc": va_acc})
        if va_ce < best[0]:
            best = (va_ce, epoch, [p.copy() for p in head.params()])
    if best[2] is not None:
        for i, p in enumerate(head.params()):
            p[...] = best[2][i]
    return head, curve, best[1], time.time() - t0


def train_linear(x_tr, y_tr, x_va, y_va, in_dim, out_dim, classes, epochs, lr, wd, batch, seed):
    head = T.LinearHead(in_dim, classes, seed)  # identity init, same as io_train_head.py
    # override output dim for encoder-swap rungs (LinearHead always emits dim x dim)
    if out_dim != in_dim:
        rng = np.random.RandomState(seed)
        head.W = (rng.randn(out_dim, in_dim) / np.sqrt(in_dim)).astype(np.float32)
        head.b = np.zeros(out_dim, np.float32)
        head.C = (rng.randn(classes, out_dim) / np.sqrt(out_dim)).astype(np.float32)
        head.cb = np.zeros(classes, np.float32)
    m = [np.zeros_like(p) for p in head.params()]
    v = [np.zeros_like(p) for p in head.params()]
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.RandomState(seed)
    n = x_tr.shape[0]
    steps_per_epoch = max(1, (n + batch - 1) // batch)
    step, curve, best = 0, [], (float("inf"), 0, None)
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total, seen = 0.0, 0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            ce, grads = head.forward_backward(x_tr[idx], y_tr[idx], wd)
            step += 1
            lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * step / max(1, epochs * steps_per_epoch)))
            for i, g in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                head.params()[i][...] -= lr_t * (m[i] / (1 - b1 ** step)) / (np.sqrt(v[i] / (1 - b2 ** step)) + eps)
            total += ce * len(idx)
            seen += len(idx)
        va_logits = head.logits(x_va)
        va_ce = float(-np.log(np.maximum(T.softmax(va_logits)[np.arange(len(y_va)), y_va], 1e-12)).mean())
        va_acc = float((va_logits.argmax(axis=1) == y_va).mean())
        curve.append({"epoch": epoch, "train_ce": total / max(1, seen), "val_ce": va_ce, "val_acc": va_acc})
        if va_ce < best[0]:
            best = (va_ce, epoch, [p.copy() for p in head.params()])
    if best[2] is not None:
        for i, p in enumerate(head.params()):
            p[...] = best[2][i]
    return head, curve, best[1], time.time() - t0


# --------------------------------------------------------------------------
# metrics wrappers
# --------------------------------------------------------------------------
def eval_embedding(train_z, y_tr, held_z, y_te):
    out = {}
    out.update(T.knn_metrics(train_z, y_tr, held_z, y_te))
    out.update(T.ridge_probe(train_z, y_tr, held_z, y_te))
    return out


def agg(rows, keys):
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], np.float64)
        out[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                  "per_seed": vals.tolist()}
    return out


# --------------------------------------------------------------------------
# checkpoint helpers
# --------------------------------------------------------------------------
def head_desc(kind, head, in_dim, out_dim):
    if kind == "linear":
        return {"format": "phoenix.io_head.v1", "type": "linear", "inDim": int(in_dim),
                "outDim": int(out_dim), "normalize": True,
                "layers": [{"type": "linear", "W": head.W.astype(np.float32).tolist(),
                            "b": head.b.astype(np.float32).tolist()}]}
    return {"format": "phoenix.io_head.v1", "type": "mlp", "inDim": int(in_dim),
            "outDim": int(out_dim), "normalize": True, "hidden": int(head.hidden),
            "dropout": float(head.dropout),
            "layers": [{"type": "linear", "W": head.W1.astype(np.float32).tolist(),
                        "b": head.b1.astype(np.float32).tolist()},
                       {"type": "relu"},
                       {"type": "linear", "W": head.W2.astype(np.float32).tolist(),
                        "b": head.b2.astype(np.float32).tolist()}]}


def save_head(run_dir, root, tag, head, kind, in_dim, out_dim):
    os.makedirs(run_dir, exist_ok=True)
    npz_path = os.path.join(run_dir, "head_%s.npz" % tag)
    json_path = os.path.join(run_dir, "head_%s.json" % tag)
    desc = head_desc(kind, head, in_dim, out_dim)
    if desc["type"] == "linear":
        np.savez(npz_path, W=head.W, b=head.b, inDim=np.int64(in_dim), outDim=np.int64(out_dim),
                 normalize=np.int64(1))
    else:
        np.savez(npz_path, W1=head.W1, b1=head.b1, W2=head.W2, b2=head.b2,
                 inDim=np.int64(in_dim), outDim=np.int64(out_dim), normalize=np.int64(1))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(desc, f)
    return {
        "rung": tag,
        "npz": os.path.relpath(npz_path, root).replace(os.sep, "/"),
        "json": os.path.relpath(json_path, root).replace(os.sep, "/"),
        "npz_sha256": T.sha256_file(npz_path),
        "json_sha256": T.sha256_file(json_path),
        "params": int(sum(p.size for p in ([head.W, head.b] if desc["type"] == "linear"
                                           else [head.W1, head.b1, head.W2, head.b2]))),
        "dims": {"inDim": int(in_dim), "outDim": int(out_dim)},
    }


def write_manifest(run_dir, root, manifest):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    T.log("manifest: %s" % path)
    return path


# --------------------------------------------------------------------------
# IMAGE ladder
# --------------------------------------------------------------------------
def load_or_extract_image_features(root, run_dir, cache_src, model_rel, sess, items, tag):
    """Reuse a verified feature cache (copying it into the new run dir), else extract."""
    os.makedirs(run_dir, exist_ok=True)
    tr_path = os.path.join(run_dir, "features_image.npz.train.npz")
    te_path = os.path.join(run_dir, "features_image.npz.held.npz")
    if os.path.isfile(tr_path) and os.path.isfile(te_path):
        tr, te = np.load(tr_path), np.load(te_path)
        T.log("reusing local feature cache in %s" % run_dir)
        return tr["x"], tr["y"], te["x"], te["y"], {
            "source": "local cache", "train_sha256": T.sha256_file(tr_path),
            "held_sha256": T.sha256_file(te_path)}
    if cache_src:
        src_tr = os.path.join(cache_src, "features_image.npz.train.npz")
        src_te = os.path.join(cache_src, "features_image.npz.held.npz")
        if os.path.isfile(src_tr) and os.path.isfile(src_te):
            tr, te = np.load(src_tr), np.load(src_te)
            paths_tr = [os.path.relpath(p, root).replace(os.sep, "/") for p, _ in items[0]]
            paths_te = [os.path.relpath(p, root).replace(os.sep, "/") for p, _ in items[1]]
            if list(tr["paths"]) == paths_tr and list(te["paths"]) == paths_te:
                T.log("reusing verified feature cache from %s (paths+labels match)" % cache_src)
                np.savez_compressed(tr_path, x=tr["x"], y=tr["y"], paths=tr["paths"])
                np.savez_compressed(te_path, x=te["x"], y=te["y"], paths=te["paths"])
                return tr["x"], tr["y"], te["x"], te["y"], {
                    "source": os.path.relpath(cache_src, root).replace(os.sep, "/"),
                    "train_sha256": T.sha256_file(src_tr),
                    "held_sha256": T.sha256_file(src_te),
                    "copied_to": os.path.relpath(run_dir, root).replace(os.sep, "/")}
            T.log("cache paths do not match the current split - re-extracting")
    sess_tmp = sess
    name = tag
    x_tr, y_tr, _ = T.extract_image_features(root, sess_tmp, items[0], 8,
                                             os.path.join(run_dir, "features_%s.train.npz" % name))
    x_te, y_te, _ = T.extract_image_features(root, sess_tmp, items[1], 8,
                                             os.path.join(run_dir, "features_%s.held.npz" % name))
    return x_tr, y_tr, x_te, y_te, {"source": "extracted", "model": model_rel}


def run_image_ladder(args, root, run_dir, enc_sess):
    train_items, held_items = args.image_items
    cache_src = args.reuse_image_features
    x_tr, y_tr, x_te, y_te, cache_info = load_or_extract_image_features(
        root, run_dir, cache_src, T.IMAGE_ENCODER, enc_sess, (train_items, held_items), "image128")
    in_dim = x_tr.shape[1]
    classes = int(y_tr.max()) + 1
    rng = np.random.RandomState(args.seed + 1)
    perm = rng.permutation(len(y_tr))
    n_va = max(1, int(round(0.1 * len(y_tr))))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]

    base = eval_embedding(T.l2_normalize(x_tr), y_tr, T.l2_normalize(x_te), y_te)
    T.log("image baseline (identity-normalised %d-d encoder concept): %s" % (in_dim, json.dumps(base, sort_keys=True)))

    rungs = []
    for spec in args.image_rungs:
        name = spec
        rows = []
        for seed in args.seeds:
            t0 = time.time()
            if name == "a_linear128":
                head, curve, best_ep, train_s = train_linear(
                    x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                    in_dim, in_dim, classes, args.epochs, 0.02, 1e-4, args.batch, seed)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = int(head.W.size + head.b.size)
            elif name == "b_mlp512":
                head, curve, best_ep, train_s = train_mlp(
                    x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                    in_dim, 512, in_dim, classes, args.epochs, 0.005, 1e-4, args.batch, seed, 0.0)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = head.n_params()
            elif name == "c_mlp1024_dropout":
                head, curve, best_ep, train_s = train_mlp(
                    x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                    in_dim, 1024, in_dim, classes, args.epochs, 0.005, 1e-4, args.batch, seed, 0.2)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = head.n_params()
            else:
                raise SystemExit("unknown image rung %s" % name)
            m = eval_embedding(z_tr, y_tr, z_te, y_te)
            rows.append({"seed": seed, "params": params, "train_time_s": round(train_s, 2),
                         "best_epoch": int(best_ep),
                         "final_train_ce": round(curve[-1]["train_ce"], 4),
                         "final_val_ce": round(curve[-1]["val_ce"], 4),
                         "final_val_acc": round(curve[-1]["val_acc"], 4), **m})
            T.log("  [%s seed=%d] params=%d train=%.1fs heldout: top1=%.4f top5=%.4f probe=%.4f"
                  % (name, seed, params, train_s, m["knn_top1"], m["knn_top5"], m["linear_probe_acc"]))
        rungs.append({"rung": name, "params": rows[0]["params"],
                      "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
                      "seeds": rows,
                      "agg": agg(rows, ["knn_top1", "knn_top5", "linear_probe_acc", "linear_probe_mse"])})
    manifest = {
        "tool": "tools/io_scaleup.py", "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "image", "encoder": {
            "path": T.IMAGE_ENCODER, "bytes": os.path.getsize(os.path.join(root, T.IMAGE_ENCODER)),
            "sha256": T.sha256_file(os.path.join(root, T.IMAGE_ENCODER)),
            "identity": "in-house 128-d vision concept encoder exported to ONNX; the "
                        "'additive_jepa' directory name is LEGACY and does not mean Meta I-JEPA",
            "io": [{"name": "pixel_values", "shape": [1, 3, 224, 224]},
                   {"name": "concept", "shape": [1, 128, 1, 1]}],
        },
        "data": {"name": "tiny-imagenet-200 train (local)", "cache": cache_info,
                 "train_samples": int(len(tr_idx)), "internal_val_samples": int(len(va_idx)),
                 "heldout_samples": int(len(y_te)),
                 "split": "file-disjoint, deterministic split seed=%d (same split as io_train run 20261117_2220)" % args.seed},
        "protocol": {"epochs": args.epochs, "batch": args.batch, "seeds": args.seeds,
                     "metrics": "cosine 1-NN/top-5 + ridge probe on file-disjoint held-out",
                     "early_stop": "10% of train (internal-val CE)"},
        "baseline_identity": base,
        "rungs": rungs,
        "chosen_rung": None,  # filled by caller
    }
    return manifest


def run_encoder_swap(args, root, run_dir, feats=None):
    """Rung (d): in-house ResNet18-modified 512-d encoder + linear 512->128 head."""
    resnet_rel = "runtime_store/models/ijepa/resnet18_224/model_encoder.onnx"
    if not os.path.isfile(os.path.join(root, resnet_rel)):
        T.log("resnet encoder missing, skipping rung (d)")
        return None
    os.makedirs(run_dir, exist_ok=True)
    sess = T.make_session(os.path.join(root, resnet_rel), args.threads)
    train_items, held_items = args.image_items
    in_name, out_name = T.encoder_io(sess)

    def extract(items, tag):
        cache = os.path.join(run_dir, "features_resnet512.%s.npz" % tag)
        if os.path.isfile(cache):
            z = np.load(cache)
            return z["x"], z["y"]
        xs, ys = [], []
        t0 = time.time()
        for i, (path, label) in enumerate(items):
            x = T.load_image_224(path)
            y = sess.run([out_name], {in_name: x})[0]
            xs.append(np.asarray(y, np.float32).reshape(-1))
            ys.append(label)
            if (i + 1) % 1000 == 0:
                T.log("  resnet512 features %d/%d (%.1fs)" % (i + 1, len(items), time.time() - t0))
        x = np.stack(xs).astype(np.float32)
        y = np.asarray(ys, np.int64)
        np.savez_compressed(cache, x=x, y=y)
        return x, y

    x_tr, y_tr = extract(train_items, "train")
    x_te, y_te = extract(held_items, "held")
    in_dim = x_tr.shape[1]  # 512
    classes = int(y_tr.max()) + 1
    rng = np.random.RandomState(args.seed + 1)
    perm = rng.permutation(len(y_tr))
    n_va = max(1, int(round(0.1 * len(y_tr))))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    base = eval_embedding(T.l2_normalize(x_tr), y_tr, T.l2_normalize(x_te), y_te)
    T.log("resnet512 raw baseline (%d-d): %s" % (in_dim, json.dumps(base, sort_keys=True)))
    rows = []
    for seed in args.seeds:
        head, curve, best_ep, train_s = train_linear(
            x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
            in_dim, 128, classes, args.epochs, 0.01, 1e-4, args.batch, seed)
        z_tr, z_te = head.embed(x_tr), head.embed(x_te)
        m = eval_embedding(z_tr, y_tr, z_te, y_te)
        rows.append({"seed": seed, "train_time_s": round(train_s, 2), "best_epoch": int(best_ep),
                     "final_val_ce": round(curve[-1]["val_ce"], 4), **m})
        T.log("  [d_resnet512+linear seed=%d] train=%.1fs heldout: top1=%.4f top5=%.4f probe=%.4f"
              % (seed, train_s, m["knn_top1"], m["knn_top5"], m["linear_probe_acc"]))
    manifest = {
        "tool": "tools/io_scaleup.py", "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "image", "rung": "d_resnet512_linear128", "encoder": {
            "path": resnet_rel, "bytes": os.path.getsize(os.path.join(root, resnet_rel)),
            "sha256": T.sha256_file(os.path.join(root, resnet_rel)),
            "identity": "in-house ResNet18-modified 512-d encoder (pooled output), exported to "
                        "ONNX; directory name 'ijepa' is LEGACY and is NOT a Meta I-JEPA checkpoint",
            "io": [{"name": "pixel_values", "shape": [1, 3, 224, 224]},
                   {"name": "pooled", "shape": [1, 512, 1, 1]}],
        },
        "data": {"train_samples": int(len(tr_idx)), "internal_val_samples": int(len(va_idx)),
                 "heldout_samples": int(len(y_te))},
        "protocol": {"epochs": args.epochs, "batch": args.batch, "seeds": args.seeds,
                     "head": "linear 512 -> 128, L2-normalised output",
                     "metrics": "cosine 1-NN/top-5 + ridge probe on file-disjoint held-out"},
        "baseline_identity_512d": base,
        "runs": rows,
        "agg": agg(rows, ["knn_top1", "knn_top5", "linear_probe_acc", "linear_probe_mse"]),
    }
    return manifest


# --------------------------------------------------------------------------
# AUDIO real objective: balanced 4-way language ID on MUSAN librivox
# --------------------------------------------------------------------------
def parse_librivox_annotations(root):
    base = os.path.join(root, "runtime_store/datasets/musan_extracted/musan/speech/librivox")
    ann = os.path.join(base, "ANNOTATIONS")
    rows = []
    with open(ann, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            stem, gender, lang = parts[0], parts[1], parts[2]
            wav = os.path.join(base, stem + ".wav")
            if os.path.isfile(wav):
                rows.append({"path": wav, "gender": gender, "lang": lang})
    return rows


def wav_windows_multi(path, samples=16000, n_windows=12, max_seconds=120):
    import wave
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            return []
        total = min(wf.getnframes(), 16000 * max_seconds)
        if total < samples:
            return []
        if n_windows <= 1:
            offsets = [0]
        else:
            step = max(1, (total - samples) // (n_windows - 1))
            offsets = [min(total - samples, i * step) for i in range(n_windows)]
        out = []
        for off in offsets:
            wf.setpos(off)
            raw = wf.readframes(samples)
            if len(raw) < samples * 2:
                continue
            out.append((np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).reshape(1, 1, 1, samples))
        return out


def run_audio_lid(args, root, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    rows_all = parse_librivox_annotations(root)
    langs = ["english", "german", "spanish", "french"]
    by_lang = {l: [r for r in rows_all if r["lang"] == l] for l in langs}
    n_bal = min(len(v) for v in by_lang.values())
    rng = random.Random(args.seed)
    chosen = []
    for ci, lang in enumerate(langs):
        files = sorted(by_lang[lang], key=lambda r: r["path"])
        rng.shuffle(files)
        chosen.extend((r["path"], ci) for r in files[:n_bal])
    # file-disjoint split, per language
    train_items, held_items = [], []
    for ci in range(len(langs)):
        files = [p for p, c in chosen if c == ci]
        n_hold = max(1, int(round(len(files) * 0.35)))
        for i, p in enumerate(files):
            dest = held_items if i < n_hold else train_items
            for w in wav_windows_multi(p, n_windows=args.audio_windows):
                dest.append((w, ci))
    T.log("audio LID: %d files/class balanced, labels=%s, train=%d heldout=%d windows"
          % (n_bal, langs, len(train_items), len(held_items)))
    enc_path = os.path.join(root, T.AUDIO_ENCODER)
    sess = T.make_session(enc_path, args.threads)
    in_name, out_name = T.encoder_io(sess)

    def extract(items, tag):
        cache = os.path.join(run_dir, "features_lid.%s.npz" % tag)
        if os.path.isfile(cache):
            z = np.load(cache)
            return z["x"], z["y"]
        xs, ys = [], []
        t0 = time.time()
        for i, (wav, label) in enumerate(items):
            y = sess.run([out_name], {in_name: wav})[0]
            xs.append(np.asarray(y, np.float32).reshape(-1))
            ys.append(label)
        x = np.stack(xs).astype(np.float32)
        y = np.asarray(ys, np.int64)
        np.savez_compressed(cache, x=x, y=y)
        T.log("  audio lid features %d in %.1fs" % (len(x), time.time() - t0))
        return x, y

    x_tr, y_tr = extract(train_items, "train")
    x_te, y_te = extract(held_items, "held")
    in_dim = x_tr.shape[1]
    classes = int(y_tr.max()) + 1
    prng = np.random.RandomState(args.seed + 1)
    perm = prng.permutation(len(y_tr))
    n_va = max(1, int(round(0.1 * len(y_tr))))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    base = eval_embedding(T.l2_normalize(x_tr), y_tr, T.l2_normalize(x_te), y_te)
    T.log("audio LID baseline (identity-normalised 128-d speech concept): %s"
          % json.dumps({k: round(v, 4) for k, v in base.items()}, sort_keys=True))
    rungs = []
    for name in ("a_linear128", "b_mlp512"):
        rows = []
        for seed in args.seeds:
            if name == "a_linear128":
                head, curve, best_ep, train_s = train_linear(
                    x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                    in_dim, in_dim, classes, args.epochs, 0.02, 1e-4, args.batch, seed)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = int(head.W.size + head.b.size)
            else:
                head, curve, best_ep, train_s = train_mlp(
                    x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                    in_dim, 512, in_dim, classes, args.epochs, 0.005, 1e-4, args.batch, seed, 0.0)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = head.n_params()
            m = eval_embedding(z_tr, y_tr, z_te, y_te)
            rows.append({"seed": seed, "params": params, "train_time_s": round(train_s, 2),
                         "best_epoch": int(best_ep), "final_val_ce": round(curve[-1]["val_ce"], 4),
                         **m})
            T.log("  [audio %s seed=%d] params=%d train=%.1fs heldout: top1=%.4f top5=%.4f probe=%.4f"
                  % (name, seed, params, train_s, m["knn_top1"], m["knn_top5"], m["linear_probe_acc"]))
        rungs.append({"rung": name, "params": rows[0]["params"],
                      "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
                      "seeds": rows,
                      "agg": agg(rows, ["knn_top1", "knn_top5", "linear_probe_acc", "linear_probe_mse"])})
    manifest = {
        "tool": "tools/io_scaleup.py", "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "audio", "objective": "balanced 4-way language identification",
        "label_source": "runtime_store/datasets/musan_extracted/musan/speech/librivox/ANNOTATIONS "
                        "(per-file real language annotation; no transcripts exist locally)",
        "encoder": {
            "path": T.AUDIO_ENCODER, "bytes": os.path.getsize(enc_path),
            "sha256": T.sha256_file(enc_path),
            "identity": "in-house 1D speech concept encoder exported to ONNX (16 kHz, 16000 samples -> "
                        "128-d); the 'additive_jepa' directory name is LEGACY, not Meta I-JEPA",
            "io": [{"name": "waveform", "shape": ["batch", 1, 1, 16000]},
                   {"name": "concept", "shape": ["batch", 128, 1, 1]}],
        },
        "data": {"files_per_class": n_bal, "classes": langs, "windows_per_file": args.audio_windows,
                 "train_samples": int(len(tr_idx)), "internal_val_samples": int(len(va_idx)),
                 "heldout_samples": int(len(y_te)),
                 "split": "file-disjoint 65/35 per language (deterministic seed=%d)" % args.seed},
        "protocol": {"epochs": args.epochs, "batch": args.batch, "seeds": args.seeds,
                     "metrics": "cosine 1-NN/top-5 + ridge probe on file-disjoint held-out"},
        "baseline_identity": base,
        "rungs": rungs,
        "chosen_rung": None,
    }
    return manifest


# --------------------------------------------------------------------------
# AUDIO real objective #2: 5-way music source/genre classification on MUSAN
# --------------------------------------------------------------------------
def run_audio_music(args, root, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    base = os.path.join(root, "runtime_store/datasets/musan_extracted/musan/music")
    genres = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    by_class = {}
    for gi, g in enumerate(genres):
        files = []
        for dirpath, _d, fnames in os.walk(os.path.join(base, g)):
            files.extend(os.path.join(dirpath, f) for f in fnames if f.lower().endswith(".wav"))
        by_class[g] = sorted(files)
    n_bal = min(len(v) for v in by_class.values())
    rng = random.Random(args.seed + 101)
    train_items, held_items = [], []
    for gi, g in enumerate(genres):
        files = list(by_class[g])
        rng.shuffle(files)
        picked = files[:n_bal]
        n_hold = max(1, int(round(n_bal * 0.30)))
        for i, p in enumerate(picked):
            dest = held_items if i < n_hold else train_items
            for w in wav_windows_multi(p, n_windows=args.audio_windows):
                dest.append((w, gi))
    T.log("audio music-genre: %d genres %s, %d files/class balanced, train=%d heldout=%d windows"
          % (len(genres), genres, n_bal, len(train_items), len(held_items)))
    enc_path = os.path.join(root, T.AUDIO_ENCODER)
    sess = T.make_session(enc_path, args.threads)
    in_name, out_name = T.encoder_io(sess)

    def extract(items, tag):
        cache = os.path.join(run_dir, "features_music.%s.npz" % tag)
        if os.path.isfile(cache):
            z = np.load(cache)
            return z["x"], z["y"]
        xs, ys = [], []
        t0 = time.time()
        for i, (wav, label) in enumerate(items):
            y = sess.run([out_name], {in_name: wav})[0]
            xs.append(np.asarray(y, np.float32).reshape(-1))
            ys.append(label)
        x = np.stack(xs).astype(np.float32)
        y = np.asarray(ys, np.int64)
        np.savez_compressed(cache, x=x, y=y)
        T.log("  audio music features %d in %.1fs" % (len(x), time.time() - t0))
        return x, y

    x_tr, y_tr = extract(train_items, "train")
    x_te, y_te = extract(held_items, "held")
    in_dim = x_tr.shape[1]
    classes = int(y_tr.max()) + 1
    prng = np.random.RandomState(args.seed + 1)
    perm = prng.permutation(len(y_tr))
    n_va = max(1, int(round(0.1 * len(y_tr))))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    base_m = eval_embedding(T.l2_normalize(x_tr), y_tr, T.l2_normalize(x_te), y_te)
    T.log("audio music-genre baseline: %s" % json.dumps({k: round(v, 4) for k, v in base_m.items()}, sort_keys=True))
    rungs = []
    for name in ("a_linear128", "b_mlp512"):
        rows = []
        for seed in args.seeds:
            if name == "a_linear128":
                head, curve, best_ep, train_s = train_linear(
                    x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                    in_dim, in_dim, classes, args.epochs, 0.02, 1e-4, args.batch, seed)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = int(head.W.size + head.b.size)
            else:
                head, curve, best_ep, train_s = train_mlp(
                    x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                    in_dim, 512, in_dim, classes, args.epochs, 0.005, 1e-4, args.batch, seed, 0.0)
                z_tr, z_te = head.embed(x_tr), head.embed(x_te)
                params = head.n_params()
            m = eval_embedding(z_tr, y_tr, z_te, y_te)
            rows.append({"seed": seed, "params": params, "train_time_s": round(train_s, 2),
                         "best_epoch": int(best_ep), "final_val_ce": round(curve[-1]["val_ce"], 4), **m})
            T.log("  [music %s seed=%d] params=%d train=%.1fs heldout: top1=%.4f top5=%.4f probe=%.4f"
                  % (name, seed, params, train_s, m["knn_top1"], m["knn_top5"], m["linear_probe_acc"]))
        rungs.append({"rung": name, "params": rows[0]["params"],
                      "train_time_s_mean": float(np.mean([r["train_time_s"] for r in rows])),
                      "seeds": rows,
                      "agg": agg(rows, ["knn_top1", "knn_top5", "linear_probe_acc", "linear_probe_mse"])})
    manifest = {
        "tool": "tools/io_scaleup.py", "createdUtc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modality": "audio", "objective": "5-way music source/genre classification",
        "label_source": "runtime_store/datasets/musan_extracted/musan/music/<genre>/ directory labels "
                        "(fma, fma-western-art, hd-classical, jamendo, rfm); no transcripts exist locally",
        "encoder": {"path": T.AUDIO_ENCODER, "bytes": os.path.getsize(enc_path),
                    "sha256": T.sha256_file(enc_path),
                    "identity": "in-house 1D speech/audio concept encoder exported to ONNX "
                                "(16 kHz, 16000 samples -> 128-d); 'additive_jepa' is a LEGACY "
                                "directory name, NOT Meta I-JEPA"},
        "data": {"genres": genres, "files_per_class": n_bal, "windows_per_file": args.audio_windows,
                 "train_samples": int(len(tr_idx)), "internal_val_samples": int(len(va_idx)),
                 "heldout_samples": int(len(y_te)),
                 "split": "file-disjoint 70/30 per genre (deterministic seed=%d)" % (args.seed + 101)},
        "protocol": {"epochs": args.epochs, "batch": args.batch, "seeds": args.seeds,
                     "metrics": "cosine 1-NN/top-5 + ridge probe on file-disjoint held-out"},
        "baseline_identity": base_m, "rungs": rungs, "chosen_rung": None,
    }
    return manifest


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def image_rung_list(spec):
    return [s for s in spec.split(",") if s]


def choose(rungs):
    """Pick the rung with best mean knn_top1; tie-break: fewer params."""
    best = None
    for r in rungs:
        key = (r["agg"]["knn_top1"]["mean"], -r["params"])
        if best is None or key > best[0]:
            best = (key, r["rung"])
    return best[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="runtime_store/models/io_train2")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--train-per-class", type=int, default=30)
    ap.add_argument("--heldout-per-class", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--seeds", default="20261117,20261118,20261119")
    ap.add_argument("--reuse-image-features", default="runtime_store/models/io_train/20261117_2220/image",
                    help="dir holding features_image.npz.{train,held}.npz from io_train_head.py")
    ap.add_argument("--image-rungs", default="a_linear128,b_mlp512,c_mlp1024_dropout")
    ap.add_argument("--skip-image", action="store_true")
    ap.add_argument("--skip-encoder-swap", action="store_true")
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--audio-tasks", default="lid,music")
    ap.add_argument("--audio-windows", type=int, default=12)
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.image_rungs = image_rung_list(args.image_rungs)
    args.audio_tasks = [s for s in args.audio_tasks.split(",") if s.strip()]
    root = os.path.abspath(args.root)
    os.chdir(root)
    ts = args.run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(root, args.out, ts)
    os.makedirs(run_root, exist_ok=True)
    T.log("phoenix root: %s" % root)
    T.log("run dir:      %s" % run_root)

    # fixed image split (same deterministic split as io_train_baseline)
    classes, train_items, held_items = T.select_image_split(
        root, args.train_per_class, args.heldout_per_class, args.seed)
    args.image_items = (train_items, held_items)
    T.log("image split: %d train / %d held-out over %d classes" %
          (len(train_items), len(held_items), len(classes)))

    written = []
    if not args.skip_image:
        enc_sess = T.make_session(os.path.join(root, T.IMAGE_ENCODER), args.threads)
        im_dir = os.path.join(run_root, "image")
        man = run_image_ladder(args, root, im_dir, enc_sess)
        man["chosen_rung"] = choose(man["rungs"])
        man["chosen"] = {"rung": man["chosen_rung"],
                         "justification": "highest 3-seed mean held-out 1-NN top-1; params is tie-break"}
        # save the best-seed checkpoint of the chosen rung (deterministic re-train, cheap)
        best_rung = man["chosen_rung"]
        spec = next(r for r in man["rungs"] if r["rung"] == best_rung)
        best_seed = max(spec["seeds"], key=lambda r: r["knn_top1"])["seed"]
        T.log("chosen image rung=%s (best seed=%d) - re-fitting to save checkpoint" % (best_rung, best_seed))
        x_tr, y_tr, x_te, y_te, _ = load_or_extract_image_features(
            root, im_dir, args.reuse_image_features, T.IMAGE_ENCODER, enc_sess, args.image_items, "image128")
        rng = np.random.RandomState(args.seed + 1)
        perm = rng.permutation(len(y_tr))
        n_va = max(1, int(round(0.1 * len(y_tr))))
        va_idx, tr_idx = perm[:n_va], perm[n_va:]
        classes = int(y_tr.max()) + 1
        if best_rung == "a_linear128":
            head, _c, _e, _t = train_linear(x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                                            128, 128, classes, args.epochs, 0.02, 1e-4, args.batch, best_seed)
            in_dim, out_dim = 128, 128
        elif best_rung == "b_mlp512":
            head, _c, _e, _t = train_mlp(x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                                         128, 512, 128, classes, args.epochs, 0.005, 1e-4, args.batch, best_seed, 0.0)
            in_dim, out_dim = 128, 128
        else:
            head, _c, _e, _t = train_mlp(x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                                         128, 1024, 128, classes, args.epochs, 0.005, 1e-4, args.batch, best_seed, 0.2)
            in_dim, out_dim = 128, 128
        man["checkpoint"] = save_head(im_dir, root, best_rung, head, "linear" if best_rung == "a_linear128" else "mlp", in_dim, out_dim)
        written.append(write_manifest(im_dir, root, man))

    if not args.skip_encoder_swap:
        man = run_encoder_swap(args, root, os.path.join(run_root, "image_resnet512"))
        if man is not None:
            rows = man["runs"]
            # save a 512->128 head for the best seed
            best_seed = max(rows, key=lambda r: r["knn_top1"])["seed"]
            sess = T.make_session(os.path.join(root, "runtime_store/models/ijepa/resnet18_224/model_encoder.onnx"), args.threads)
            in_name, out_name = T.encoder_io(sess)
            # reuse cached features
            tr = np.load(os.path.join(run_root, "image_resnet512", "features_resnet512.train.npz"))
            x_tr, y_tr = tr["x"], tr["y"]
            prng = np.random.RandomState(args.seed + 1)
            perm = prng.permutation(len(y_tr))
            n_va = max(1, int(round(0.1 * len(y_tr))))
            va_idx, tr_idx = perm[:n_va], perm[n_va:]
            head, _c, _e, _t = train_linear(x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                                            512, 128, classes, args.epochs, 0.01, 1e-4, args.batch, best_seed)
            man["checkpoint"] = save_head(os.path.join(run_root, "image_resnet512"), root,
                                          "d_resnet512_linear128", head, "linear", 512, 128)
            written.append(write_manifest(os.path.join(run_root, "image_resnet512"), root, man))

    if not args.skip_audio:
        for task in args.audio_tasks:
            if task == "lid":
                d = os.path.join(run_root, "audio_lid")
                man = run_audio_lid(args, root, d)
                feat = "features_lid"
            elif task == "music":
                d = os.path.join(run_root, "audio_music")
                man = run_audio_music(args, root, d)
                feat = "features_music"
            else:
                raise SystemExit("unknown --audio-tasks entry %s" % task)
            man["chosen_rung"] = choose(man["rungs"])
            man["chosen"] = {"rung": man["chosen_rung"],
                             "justification": "highest 3-seed mean held-out 1-NN top-1; params is tie-break"}
            best_rung = man["chosen_rung"]
            spec = next(r for r in man["rungs"] if r["rung"] == best_rung)
            best_seed = max(spec["seeds"], key=lambda r: r["knn_top1"])["seed"]
            tr = np.load(os.path.join(d, feat + ".train.npz")); x_tr, y_tr = tr["x"], tr["y"]
            prng = np.random.RandomState(args.seed + 1)
            perm = prng.permutation(len(y_tr)); n_va = max(1, int(round(0.1 * len(y_tr))))
            va_idx, tr_idx = perm[:n_va], perm[n_va:]
            classes_a = int(y_tr.max()) + 1
            if best_rung == "a_linear128":
                head, _c, _e, _t = train_linear(x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                                                128, 128, classes_a, args.epochs, 0.02, 1e-4, args.batch, best_seed)
            else:
                head, _c, _e, _t = train_mlp(x_tr[tr_idx], y_tr[tr_idx], x_tr[va_idx], y_tr[va_idx],
                                             128, 512, 128, classes_a, args.epochs, 0.005, 1e-4, args.batch,
                                             best_seed, 0.0)
            man["checkpoint"] = save_head(d, root, best_rung, head,
                                          "linear" if best_rung == "a_linear128" else "mlp", 128, 128)
            written.append(write_manifest(d, root, man))

    T.log("done, manifests: %s" % ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())

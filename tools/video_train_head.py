#!/usr/bin/env python3
"""video_train_head.py - LLaVA-cache -> live 128-d concept distillation + video-path head.

Companion to tools/io_train_head.py and tools/io_scaleup.py. This tool answers the
question the v10.1 dead-asset audit raised: the 22.85 GiB LLaVA-1.5-7B feature
caches under runtime_store/models/large_vision_encoder/ are *reusable*, so can they
produce a DEPLOYABLE video/image concept head?

Protocol (image distillation, --mode image-distill):
  * data: LLaVA caches only, memmapped (never loads 22 GB):
      llava_features_more.npy  (4000, 576, 4096) fp16
      llava_features.npy        (512, 576, 4096) fp16   (24 paths overlap the 4000 set)
      llava_images_more.npy     (4000, 3, 224, 224) uint8
      llava_images.npy          ( 512, 3, 224, 224) uint8
      llava_more_paths.json     (4000 image paths -> class name n0XXXXXXX)
    universe = 4000 + 488 file-disjoint 512-set images = 4488 (the 24 overlaps are dropped)
  * teacher input X: mean over the 576 LLaVA tokens -> 4096-d, L2-normalised
  * distillation targets T: the LIVE in-house vision encoder
      runtime_store/models/additive_jepa/vision_encoder/best.onnx  (224 -> 128, CPU)
    run on the SAME paired 224px images, L2-normalised.
  * heads: (a) closed-form ridge 4096->128 (linear), (b) MLP 4096->512->128 (torch CPU)
  * 3 seeds = 3 independent file-disjoint splits; held-out = 2000 images each
  * read-outs on held-out: cosine 1-NN top-1/top-5, closed-form ridge probe accuracy,
    and MSE/cosine of the head output to the live 128-d concept.
  * gate (rung-c lesson): a head is DEPLOYABLE only when top-1 AND top-5 AND probe all
    improve over the identity baseline (normalized live concept).

Videos (--mode video):
  There is currently NO local video corpus in the tree (checked: no mp4/avi/mkv/mov/
  webm/frame folders under runtime_store/datasets, data/, uploads/). The video mode
  therefore validates a frames-manifest and refuses to invent data; with
  --frames-manifest plus --allow-train it trains the smallest honest head (per-frame
  128-d live concepts -> mean pool + optional linear temporal projection, clip labels).

Other modes:
  --mode eval-teacher-models : probe multimodal_dec/{llava_unit_tokenizer,image_decoder}.onnx
                               with a near-mean-output criterion (writes JSON).

Everything is read-only on live model dirs; checkpoints go to
runtime_store/models/io_train2/video_<ts>/. CPU only (GPU is left to the live stack).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time

import numpy as np

import onnxruntime as ort

try:
    import torch  # only for reading the 512-set .pt path list and the MLP rung
except Exception:  # pragma: no cover
    torch = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MV = os.path.join(ROOT, "runtime_store", "models")
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

LIVE_ENCODER = "runtime_store/models/additive_jepa/vision_encoder/best.onnx"
TOKENIZER = "runtime_store/models/multimodal_dec/llava_unit_tokenizer.onnx"
DECODER = "runtime_store/models/multimodal_dec/image_decoder.onnx"


def log(msg):
    print("[video-train] %s" % msg, flush=True)


def now_iso():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def l2_normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def make_session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def class_from_path(path):
    for part in path.replace("\\", "/").split("/"):
        if len(part) == 9 and part[0] == "n" and part[1:].isdigit():
            return part
    return None


# ---------------------------------------------------------------------------
# data prep (memmap, never bulk-load the 22 GB)
# ---------------------------------------------------------------------------
def collect_items():
    """Return (paths, feature_keys, image_keys) for the LLaVA cache universe."""
    lv = os.path.join(MV, "large_vision_encoder")
    paths_more = json.load(open(os.path.join(lv, "llava_more_paths.json"), encoding="utf-8"))
    paths_more = [os.path.normpath(p) for p in paths_more]
    seen = set(paths_more)
    paths = list(paths_more)
    feat_keys = [("more", i) for i in range(len(paths_more))]
    img_keys = [("more", i) for i in range(len(paths_more))]

    pt = os.path.join(lv, "llava_unit_feature_cache.pt")
    paths_512 = []
    if os.path.exists(pt) and torch is not None:
        try:
            d = torch.load(pt, map_location="cpu")
            paths_512 = [os.path.normpath(str(p)) for p in d.get("paths", [])]
            keep = []
            for i, p in enumerate(paths_512):
                if p in seen:
                    continue
                seen.add(p)
                keep.append(i)
            for i in keep:
                paths.append(paths_512[i])
                feat_keys.append(("full", i))
                img_keys.append(("full", i))
            log("512-set: %d paths, %d file-disjoint added (24 overlap dropped)"
                % (len(paths_512), len(keep)))
        except Exception as exc:  # pragma: no cover
            log("WARN: could not read %s (%s); continuing with 4000-set only" % (pt, exc))
    else:
        log("WARN: torch unavailable or .pt missing; using 4000-set only")
    return paths, feat_keys, img_keys


def prepare_features(args):
    """Compute live 128-d targets + 4096-d mean-pooled teacher features; cache them."""
    lv = os.path.join(MV, "large_vision_encoder")
    paths, feat_keys, img_keys = collect_items()
    cls = [class_from_path(p) for p in paths]
    if any(c is None for c in cls):
        raise SystemExit("could not parse n0XXXXXXX class from every image path")
    classes = sorted(set(cls))
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[c] for c in cls], dtype=np.int64)

    f_more = np.load(os.path.join(lv, "llava_features_more.npy"), mmap_mode="r")
    f_full = np.load(os.path.join(lv, "llava_features.npy"), mmap_mode="r")
    i_more = np.load(os.path.join(lv, "llava_images_more.npy"), mmap_mode="r")
    i_full = np.load(os.path.join(lv, "llava_images.npy"), mmap_mode="r")

    enc_path = os.path.join(ROOT, LIVE_ENCODER)
    sess = make_session(enc_path, args.threads)
    t0 = time.time()
    n = len(paths)
    live = np.zeros((n, 128), dtype=np.float32)
    teacher = np.zeros((n, 4096), dtype=np.float32)
    sample_idx = np.linspace(0, n - 1, min(16, n)).astype(int) if n >= 16 else np.arange(n)
    sample_tokens = np.zeros((len(sample_idx), 576, 4096), dtype=np.float16)
    sample_imgs = np.zeros((len(sample_idx), 3, 224, 224), dtype=np.uint8)
    for k in range(n):
        fk, ii = feat_keys[k]
        ik, ij = img_keys[k]
        feat = f_more[ii] if fk == "more" else f_full[ii]
        img = i_more[ij] if ik == "more" else i_full[ij]
        teacher[k] = np.asarray(feat, dtype=np.float32).mean(axis=0)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - IMG_MEAN) / IMG_STD
        out = sess.run(["concept"], {"pixel_values": x[None, ...]})[0]
        live[k] = np.asarray(out, dtype=np.float32).reshape(-1)
        if k in sample_idx:
            j = int(np.where(sample_idx == k)[0][0])
            sample_tokens[j] = np.asarray(feat, dtype=np.float16)
            sample_imgs[j] = np.asarray(img, dtype=np.uint8)
        if (k + 1) % 1000 == 0:
            log("  features %d/%d (%.1fs)" % (k + 1, n, time.time() - t0))
    log("features done: %d images in %.1fs (%.1f ms/img)"
        % (n, time.time() - t0, (time.time() - t0) / max(1, n) * 1000))

    cd = os.path.join(args.out_dir, "features")
    os.makedirs(cd, exist_ok=True)
    np.savez_compressed(
        os.path.join(cd, "distill_features.npz"),
        teacher=teacher.astype(np.float32),
        live=live.astype(np.float32),
        y=y,
        paths=np.array(paths),
        sample_tokens=sample_tokens,
        sample_images=sample_imgs,
        sample_index=sample_idx,
    )
    meta = {
        "createdAt": now_iso(),
        "universe": {"count": int(n), "classes": int(len(classes)),
                      "more4000": 4000, "full512_disjoint_added": int(n - 4000),
                      "note": "512/4000 overlap by path is dropped; labels = n0XXXXXXX parent dir"},
        "teacher": {"source": ["runtime_store/models/large_vision_encoder/llava_features_more.npy",
                               "runtime_store/models/large_vision_encoder/llava_features.npy"],
                    "pooling": "mean over 576 tokens -> 4096, saved float32",
                    "shape": [int(n), 4096]},
        "target": {"encoder": LIVE_ENCODER, "sha256": sha256_file(enc_path),
                   "preprocess": "uint8 RGB /255, ImageNet mean/std, NCHW [1,3,224,224]",
                   "io": [{"name": "pixel_values", "shape": [1, 3, 224, 224]},
                          {"name": "concept", "shape": [1, 128, 1, 1]}],
                   "shape": [int(n), 128]},
        "cache": os.path.join(cd, "distill_features.npz"),
    }
    write_json(os.path.join(cd, "features_meta.json"), meta)
    log("cache: %s" % os.path.join(cd, "distill_features.npz"))
    return meta


def load_cache(out_dir):
    z = np.load(os.path.join(out_dir, "features", "distill_features.npz"), allow_pickle=False)
    return z


# ---------------------------------------------------------------------------
# evaluation (identical definitions to tools/io_train_head.py)
# ---------------------------------------------------------------------------
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
    return {"linear_probe_acc": float((pred == y_te).mean()),
            "linear_probe_mse": float(((scores - np.eye(classes, dtype=np.float32)[y_te]) ** 2).mean())}


def evaluate(trainz, trainy, heldz, heldy):
    out = {}
    out.update(knn_metrics(trainz, trainy, heldz, heldy))
    out.update(ridge_probe(trainz, trainy, heldz, heldy))
    return out


def agg(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=0)), "per_seed": v.tolist()}
    return out


# ---------------------------------------------------------------------------
# distillation rungs
# ---------------------------------------------------------------------------
def fit_ridge(X, Y, lam):
    """Closed-form ridge  X (n,d) -> Y (n,c). Returns W (d+1,c) with bias row."""
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
    A = Xb.T @ Xb
    A[np.arange(A.shape[0]), np.arange(A.shape[0])] += lam
    W = np.linalg.solve(A, Xb.T @ Y)
    return W.astype(np.float32)


def apply_ridge(W, X):
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=np.float32)], axis=1)
    return Xb @ W


class TorchMLP:
    def __init__(self, in_dim, hidden, out_dim, seed):
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim))
        self.in_dim = in_dim

    def fit(self, X, Y, epochs, lr, wd, batch, seed, threads):
        torch.set_num_threads(int(threads))
        rng = np.random.RandomState(seed)
        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
        loss_fn = torch.nn.MSELoss()
        xt = torch.from_numpy(X)
        yt = torch.from_numpy(Y)
        n = len(X)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    def embed(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.from_numpy(X)).numpy()

    def state_dict_np(self):
        return {k: v.detach().numpy() for k, v in self.net.state_dict().items()}


# ---------------------------------------------------------------------------
# image distillation
# ---------------------------------------------------------------------------
def run_image_distill(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.reuse_cache:
        cache = np.load(os.path.join(args.reuse_cache, "features", "distill_features.npz"), allow_pickle=False)
        log("reusing feature cache %s" % args.reuse_cache)
    else:
        prepare_features(args)
        cache = load_cache(args.out_dir)
    X = np.asarray(cache["teacher"], dtype=np.float32)
    T = np.asarray(cache["live"], dtype=np.float32)
    y = np.asarray(cache["y"], dtype=np.int64)
    paths = np.asarray(cache["paths"])
    classes = int(y.max()) + 1
    log("data: X=%s T=%s classes=%d" % (X.shape, T.shape, classes))

    Xn = l2_normalize(X)
    Tn = l2_normalize(T)
    heldout = min(args.heldout, len(X) // 2)
    seeds = args.seeds
    rungs = ["identity_live", "teacher4096", "lin_ridge4096", "mlp512"]
    rows = {r: [] for r in rungs}

    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        order = rng.permutation(len(X))
        held = order[:heldout]
        tr = order[heldout:]
        Xtr, Xte = Xn[tr], Xn[held]
        Ttr, Tte = Tn[tr], Tn[held]
        ytr, yte = y[tr], y[held]
        # internal val = last 10% of train (for lambda choice / early stop)
        nv = max(50, len(tr) // 10)
        va, tr2 = tr[:nv], tr[nv:]
        Xva, Tva = Xn[va], Tn[va]
        base = {"rung": "identity_live", "seed": int(seed), **evaluate(Tte, yte, Tte, yte)}
        # identity: retrieval uses heldout as its own train set is meaningless -> use train split
        base = {"rung": "identity_live", "seed": int(seed),
                **evaluate(Ttr, ytr, Tte, yte), "mse_to_live": 0.0, "cos_to_live": 1.0}
        rows["identity_live"].append(base)
        if args.with_teacher_baseline:
            teacher_m = evaluate(Xtr, ytr, Xte, yte)
            rows["teacher4096"].append({"rung": "teacher4096", "seed": int(seed),
                                        **teacher_m, "mse_to_live": None, "cos_to_live": None})
        # linear ridge 4096->128 target = normalized live concept
        Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1), dtype=np.float32)], axis=1)
        Xvab = np.concatenate([Xva, np.ones((len(Xva), 1), dtype=np.float32)], axis=1)
        best = None
        A0 = Xb.T @ Xb
        B0 = Xb.T @ Ttr
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            A = A0.copy()
            A[np.arange(A.shape[0]), np.arange(A.shape[0])] += lam
            W = np.linalg.solve(A, B0)
            vloss = float(((Xvab @ W - Tva) ** 2).mean())
            if best is None or vloss < best[1]:
                best = (lam, vloss, W.astype(np.float32))
        lam, vloss, W = best
        Ztr = l2_normalize(Xb @ W)
        Zte = l2_normalize(np.concatenate([Xte, np.ones((len(Xte), 1), dtype=np.float32)], axis=1) @ W)
        m = evaluate(Ztr, ytr, Zte, yte)
        m.update({"mse_to_live": float(((Zte - Tte) ** 2).mean()),
                  "cos_to_live": float(np.mean(np.sum(Zte * Tte, axis=1)))})
        rows["lin_ridge4096"].append({"rung": "lin_ridge4096", "seed": int(seed),
                                      "lambda": lam, "lambda_internal_val_mse": vloss, **m})
        # MLP 4096->512->128
        if torch is not None:
            mlp = TorchMLP(4096, 512, 128, int(seed))
            mlp.fit(Xtr, Ttr, epochs=args.epochs, lr=args.lr, wd=args.wd,
                    batch=args.batch, seed=int(seed), threads=args.threads)
            Ptr = l2_normalize(mlp.embed(Xtr))
            Pte = l2_normalize(mlp.embed(Xte))
            m = evaluate(Ptr, ytr, Pte, yte)
            m.update({"mse_to_live": float(((Pte - Tte) ** 2).mean()),
                      "cos_to_live": float(np.mean(np.sum(Pte * Tte, axis=1)))})
            rows["mlp512"].append({"rung": "mlp512", "seed": int(seed), **m})
            npz = os.path.join(args.out_dir, "image_distill", "head_mlp512.npz")
            os.makedirs(os.path.dirname(npz), exist_ok=True)
            np.savez(npz, **mlp.state_dict_np())
            write_json(os.path.join(args.out_dir, "image_distill", "head_mlp512.json"),
                       {"rung": "mlp512", "inDim": 4096, "outDim": 128, "hidden": 512,
                        "params": int(sum(v.size for v in mlp.state_dict_np().values())),
                        "seed": int(seed), **m})
        # keep last-seed linear head for checkpoint export (all seeds reported)
        # NOTE: fitted W is (in+1, out); the runtime contract uses W[outDim][inDim]
        # row-major (same convention as runtime_store/.../image/head_a_linear128*),
        # so export the transpose and feed L2-normalised teacher features.
        W_out = np.ascontiguousarray(W[:-1].T, dtype=np.float32)   # (128, 4096)
        b_out = np.ascontiguousarray(W[-1], dtype=np.float32)      # (128,)
        np.savez(os.path.join(args.out_dir, "image_distill", "head_lin_ridge4096.npz"),
                 W=W_out, b=b_out, inDim=np.int64(4096), outDim=np.int64(128),
                 normalize=np.int64(1), inputNormalize=np.int64(1),
                 weightOrder=np.array("outDim x inDim, row-major"))
        write_json(os.path.join(args.out_dir, "image_distill", "head_lin_ridge4096.json"),
                   {"rung": "lin_ridge4096", "inDim": 4096, "outDim": 128,
                    "lambda": lam, "seed": int(seed), **m})
        log("seed %s: identity top1=%.4f top5=%.4f probe=%.4f | lin top1=%.4f top5=%.4f probe=%.4f mse=%.5f"
            % (seed, base["knn_top1"], base["knn_top5"], base["linear_probe_acc"],
               rows["lin_ridge4096"][-1]["knn_top1"], rows["lin_ridge4096"][-1]["knn_top5"],
               rows["lin_ridge4096"][-1]["linear_probe_acc"], rows["lin_ridge4096"][-1]["mse_to_live"]))

    keys = ["knn_top1", "knn_top5", "linear_probe_acc"]
    aggs = {}
    for r in rungs:
        if not rows[r]:
            continue
        ks = list(keys)
        for extra in ("mse_to_live", "cos_to_live"):
            if all(isinstance(row.get(extra), (int, float)) for row in rows[r]):
                ks.append(extra)
        aggs[r] = agg(rows[r], ks)
    gate = {}
    for r in ("lin_ridge4096", "mlp512"):
        if r not in aggs:
            continue
        g = {k: (aggs[r][k]["mean"] > aggs["identity_live"][k]["mean"]) for k in keys}
        g["all_improve"] = bool(all(g.values()))
        g["delta"] = {k: aggs[r][k]["mean"] - aggs["identity_live"][k]["mean"] for k in keys}
        gate[r] = g
    metrics_gate_passed = bool(gate.get("lin_ridge4096", {}).get("all_improve", False))
    # Deployment needs BOTH the metric gate AND a 576x4096 teacher feature extractor at
    # serving time. No such extractor is shipped (the raw LLaVA download is gone and the
    # two multimodal_dec ONNX models are untrained), so the runtime deployable flag is False
    # even when the metric gate passes.
    extractor_available = False
    deployable = bool(metrics_gate_passed and extractor_available)

    # ----- checkpoint export: linear head, deployable contract -----
    image_dir = os.path.join(args.out_dir, "image_distill")
    npz_path = os.path.join(image_dir, "head_lin_ridge4096.npz")
    d = np.load(npz_path)
    W = np.asarray(d["W"], dtype=np.float32)      # (128, 4096) row-major
    b = np.asarray(d["b"], dtype=np.float32)      # (128,)
    f32_path = os.path.join(image_dir, "head_lin_ridge4096.f32")
    with open(f32_path, "wb") as f:
        f.write(W.tobytes(order="C"))
        f.write(b.tobytes(order="C"))
    primary = "lin_ridge4096"
    contract = {
        "format": "phoenix.io_head.v1",
        "modality": "image",
        "type": "linear",
        "inDim": 4096,
        "outDim": 128,
        "normalize": True,
        "activation": "none",
        "weightFile": "head_lin_ridge4096.f32",
        "weightDtype": "float32-le",
        "weightOrder": "rowmajor",
        "tensors": [{"name": "W", "shape": [128, 4096], "order": "rowmajor"},
                    {"name": "b", "shape": [128], "order": "rowmajor"}],
        "weightBytes": int(W.size * 4 + b.size * 4),
        "weightSha256": sha256_file(f32_path),
        "applyAfter": "mean-pooled LLaVA-1.5-7B vision-tower+projector features over 576 tokens (4096-d), L2-normalised, before normalizeVector",
        "inputNormalize": True,
        "inputShape": [4096],
        "outputShape": [128],
        "requires": {
            "featureExtractor": "LLaVA-1.5-7B vision_tower+multi_modal_projector producing [576,4096]",
            "shippedLocally": False,
            "note": "The raw LLaVA checkpoint is NOT on disk (Batch-1 deleted the incomplete download); "
                    "the 4096-d input can only be produced from the caches, so this head is research-only "
                    "until a real 576x4096 image feature extractor is available.",
        },
        "deployable": deployable,
        "metricsGatePassed": metrics_gate_passed,
        "deploymentStatus": ("METRICS PASS / RUNTIME BLOCKED: no local 576x4096 teacher extractor"
                             if metrics_gate_passed else "METRICS FAIL: read-outs do not all beat identity"),
        "metrics": {r: {k: aggs[r][k] for k in aggs[r]} for r in aggs},
        "gate": gate,
        "referenceEval": {
            "identity_baseline": {k: aggs["identity_live"][k] for k in keys},
            "distilled": {k: aggs[primary][k] for k in keys} if primary in aggs else {},
        },
    }
    write_json(os.path.join(image_dir, "head_lin_ridge4096.contract.json"), contract)

    manifest = {
        "tool": "tools/video_train_head.py",
        "mode": "image-distill",
        "createdAt": now_iso(),
        "data": {
            "universe": int(len(X)), "classes": classes,
            "train_per_seed": int(len(X) - heldout), "heldout_per_seed": int(heldout),
            "seeds": [int(s) for s in seeds],
            "splits": "3 independent random file-disjoint splits (seed decides split); no image in both",
            "labels": "Tiny-ImageNet-200 class from path n0XXXXXXX",
            "teacher": "LLaVA-1.5-7B vision-tower+projector caches (mean-pool 576 tokens -> 4096)",
            "target": "live additive_jepa/vision_encoder/best.onnx concept (128-d)",
        },
        "protocol": {"epochs_mlp": args.epochs, "lr_mlp": args.lr, "wd_mlp": args.wd,
                     "batch_mlp": args.batch, "ridge_lambdas": [1e-4, 1e-3, 1e-2, 1e-1],
                     "readouts": ["cosine 1-NN top-1/top-5", "closed-form ridge probe",
                                  "MSE/cos to live concept"]},
        "rungs": {r: {"rows": rows[r], "agg": aggs[r]} for r in aggs},
        "three_seed_table": {
            r: {k: aggs[r][k] for k in aggs[r]} for r in aggs
        },
        "gate": gate,
        "chosen": {"rung": primary, "metricsGatePassed": metrics_gate_passed,
                   "deployable": deployable},
        "checkpoints": {
            "linear_npz": os.path.join(image_dir, "head_lin_ridge4096.npz").replace(ROOT + os.sep, ""),
            "linear_f32": os.path.join(image_dir, "head_lin_ridge4096.f32").replace(ROOT + os.sep, ""),
            "contract": os.path.join(image_dir, "head_lin_ridge4096.contract.json").replace(ROOT + os.sep, ""),
            "mlp_npz": os.path.join(image_dir, "head_mlp512.npz").replace(ROOT + os.sep, ""),
        },
        "deployment": {
            "metricsGatePassed": metrics_gate_passed,
            "deployable": deployable,
            "blocked_by": (None if deployable else
                           "no local 576x4096 LLaVA image feature extractor; the two multimodal_dec ONNX "
                           "models are untrained (see eval_teacher_models.json), and the raw LLaVA download was deleted"),
            "loader_contract": os.path.join(image_dir, "head_lin_ridge4096.contract.json").replace(ROOT + os.sep, ""),
            "config_keys": {"jepa.image.headCheckpoint": "<contract path>",
                            "jepa.image.headRequire": False},
            "fallback": "deployable-today alternative remains the 128->128 linear head from "
                        "runtime_store/models/io_train2/20261117_2330/image/",
        },
    }
    write_json(os.path.join(args.out_dir, "manifest.json"), manifest)
    log("manifest: %s" % os.path.join(args.out_dir, "manifest.json"))
    return {"aggs": aggs, "gate": gate, "manifest": manifest, "cache": cache}


# ---------------------------------------------------------------------------
# teacher ONNX probe (near-mean criterion)
# ---------------------------------------------------------------------------
def mean_metrics(a):
    """a (n,d) -> collapse descriptors."""
    m = a.mean(axis=0)
    rel = np.linalg.norm(a - m, axis=1) / max(1e-9, np.linalg.norm(m))
    nrm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    pw = [float(nrm[i] @ nrm[j]) for i in range(len(a)) for j in range(i + 1, len(a))]
    return {"rel_dev_from_mean_mean": float(rel.mean()),
            "rel_dev_from_mean_min": float(rel.min()),
            "pairwise_cos_mean": float(np.mean(pw)) if pw else None,
            "pairwise_cos_min": float(np.min(pw)) if pw else None,
            "pairwise_cos_max": float(np.max(pw)) if pw else None}


def run_eval_teacher(args):
    out = os.path.join(args.out_dir, "eval_teacher_models.json")
    if args.reuse_cache:
        z = np.load(os.path.join(args.reuse_cache, "features", "distill_features.npz"), allow_pickle=False)
        tokens = np.asarray(z["sample_tokens"], dtype=np.float32)
        imgs = np.asarray(z["sample_images"], dtype=np.float32) / 255.0
    else:
        raise SystemExit("eval-teacher-models needs --reuse-cache <run_dir> with features/")
    so = make_session(os.path.join(ROOT, TOKENIZER), args.threads)
    dec = make_session(os.path.join(ROOT, DECODER), args.threads)
    n = len(tokens)
    uq = np.stack([np.asarray(so.run(["unit_query"], {"image_features": tokens[i][None, ...]})[0],
                              dtype=np.float32).reshape(-1) for i in range(n)])
    pool = tokens.mean(axis=1)
    cos_in = [float(np.dot(uq[i], pool[i]) / max(1e-9, np.linalg.norm(uq[i]) * np.linalg.norm(pool[i])))
              for i in range(n)]
    tok = {"input": "[n,576,4096] float32", "output": "[n,4096] float32", "n": n,
           "unit_query_norms": [float(np.linalg.norm(u)) for u in uq],
           "cos_to_meanpool_of_input_mean": float(np.mean(cos_in)),
           "cos_to_meanpool_of_input_per_sample": cos_in,
           **mean_metrics(uq)}
    tok["near_mean_output"] = bool(tok["pairwise_cos_mean"] is not None
                                   and tok["pairwise_cos_mean"] > 0.9
                                   and tok["rel_dev_from_mean_mean"] < 0.5)
    outs = np.stack([np.asarray(dec.run(["output"], {"unit_query": uq[i][None, ...]})[0],
                                dtype=np.float32).reshape(3, 224, 224) for i in range(n)])
    ds = imgs[:n]
    imgmean = ds.mean(axis=0)
    outmean = outs.mean(axis=0)
    mse = lambda a, b: float(((a - b) ** 2).mean())
    dec_m = {
        "input": "[n,4096] float32", "output": "[n,3,224,224] float32", "n": n,
        "out_min": float(outs.min()), "out_max": float(outs.max()),
        "out_mean": float(outs.mean()), "out_std": float(outs.std()),
        "dataset_mean": float(imgmean.mean()), "dataset_std": float(ds.std()),
        "pairwise_cos_mean": float(np.mean([float(np.dot(outs[i].ravel(), outs[j].ravel())
                                        / max(1e-9, np.linalg.norm(outs[i]) * np.linalg.norm(outs[j])))
                                        for i in range(n) for j in range(i + 1, n)])),
        "mse_to_source_mean": float(np.mean([mse(outs[i], ds[i]) for i in range(n)])),
        "mse_to_dataset_mean_image": float(np.mean([mse(outs[i], imgmean) for i in range(n)])),
        "constant_mean_image_baseline_mse": float(np.mean([mse(ds[i], imgmean) for i in range(n)])),
    }
    dec_m["usable_decoder"] = bool(dec_m["mse_to_source_mean"] < dec_m["constant_mean_image_baseline_mse"])
    res = {"createdAt": now_iso(),
           "models": {
               "llava_unit_tokenizer": {"path": TOKENIZER, "sha256": sha256_file(os.path.join(ROOT, TOKENIZER)),
                                        "criterion": "near-mean output if pairwise cos > 0.9 and rel-dev < 0.5",
                                        "metrics": tok, "usable_as_teacher_pooling": not tok["near_mean_output"]},
               "image_decoder": {"path": DECODER, "sha256": sha256_file(os.path.join(ROOT, DECODER)),
                                 "criterion": "usable if reconstruction MSE < constant-mean-image baseline",
                                 "metrics": dec_m, "usable_as_decoder": dec_m["usable_decoder"]},
           },
           "verdict": ("both multimodal_dec ONNX models are UNUSABLE as teacher/decoder; the caches + the "
                       "trained .pt weights are the loadable artifacts" if (tok["near_mean_output"] or not dec_m["usable_decoder"])
                       else "at least one model passed its usability criterion")}
    write_json(out, res)
    log("wrote %s" % out)
    return res


# ---------------------------------------------------------------------------
# video mode (honest: no local frames -> refuse, print spec)
# ---------------------------------------------------------------------------
VIDEO_SPEC = {
    "manifest_format": "JSONL, one clip per line",
    "required_fields": {"clipId": "unique string", "frames": "[>=4 local image paths, sorted by time]",
                        "label": "string (class/action)"},
    "minimum_usable_dataset": {
        "clips": 2000, "frames_per_clip": 8, "resolution": [224, 224],
        "classes": [10, 50], "bytes_estimate": "2000*8*~15KB(jpeg) = ~240 MB + labels",
        "splits": "file-disjoint clip-level train/heldout, heldout >= 400 clips",
        "public_candidates": ["UCF101 (13k clips, 101 classes, ~7 GB)", "Kinetics-400 subset",
                              "SSv2 subset", "local camera recordings (best: no download)"],
    },
    "current_local_status": "NO video files found under runtime_store/datasets, data/, uploads/ "
                            "(searched mp4/avi/mkv/mov/webm and frame/clip folders on 2026-11-18)",
    "what_the_runtime_consumes": "external_mixed_modal_io.cpp routes Video == Image: one payload/frame -> "
                                 "videoEncoder(dim).encode() -> one 128-d concept. There is no temporal/"
                                 "sequence API in VideoModel (encodeContext/predictTarget are SPATIAL "
                                 "patch masks, not time). A video head therefore needs a new temporal "
                                 "aggregation layer over per-frame concepts (e.g. N x 128 -> 128).",
}


def run_video(args):
    if not args.frames_manifest:
        write_json(os.path.join(args.out_dir, "video_dataset_requirement.json"),
                   {"createdAt": now_iso(), **VIDEO_SPEC})
        log("video: no local frames corpus -> refusing to train; spec written")
        print(json.dumps(VIDEO_SPEC, indent=2))
        return {"trained": False, "reason": "no local video frames", "spec": VIDEO_SPEC}
    entries = [json.loads(l) for l in open(args.frames_manifest, encoding="utf-8") if l.strip()]
    ok = [e for e in entries if e.get("clipId") and len(e.get("frames", [])) >= 4 and e.get("label")]
    log("video: manifest %d clips (%d usable)" % (len(entries), len(ok)))
    if not ok:
        raise SystemExit("frames manifest has no usable clips (need clipId, >=4 frames, label)")
    if not args.allow_train:
        log("video: usable clips found; pass --allow-train to train the temporal head")
        return {"trained": False, "usable_clips": len(ok)}
    enc = make_session(os.path.join(ROOT, LIVE_ENCODER), args.threads)
    clips = []
    for e in ok:
        zs = []
        for fp in e["frames"][:args.video_frames]:
            from PIL import Image
            im = Image.open(fp).convert("RGB").resize((224, 224))
            x = np.asarray(im, dtype=np.float32) / 255.0
            x = (np.transpose(x, (2, 0, 1)) - IMG_MEAN) / IMG_STD
            zs.append(np.asarray(enc.run(["concept"], {"pixel_values": x[None, ...]})[0],
                                 dtype=np.float32).reshape(-1))
        clips.append(zs)
    labels = sorted({e["label"] for e in ok})
    log("video: %d clips x %d frames, %d classes -> temporal head training is a separate approval "
        "(smallest honest head = frame-mean + linear %d->128 on clip labels); stopping here"
        % (len(clips), len(clips[0]), len(labels), 128))
    return {"trained": False, "usable_clips": len(clips), "note": "frames present; training gate not opened"}


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="image-distill",
                    choices=["image-distill", "eval-teacher-models", "video", "prepare"])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--reuse-cache", default="", help="run dir with features/distill_features.npz")
    ap.add_argument("--seeds", default="20261118,20261119,20261120")
    ap.add_argument("--heldout", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--with-teacher-baseline", action="store_true", default=True)
    ap.add_argument("--no-teacher-baseline", dest="with_teacher_baseline", action="store_false")
    ap.add_argument("--frames-manifest", default="")
    ap.add_argument("--video-frames", type=int, default=8)
    ap.add_argument("--allow-train", action="store_true")
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if not args.out_dir:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(MV, "io_train2", "video_" + ts)
    args.out_dir = os.path.abspath(args.out_dir)
    log("out-dir: %s" % args.out_dir)
    if args.mode == "image-distill":
        res = run_image_distill(args)
        print(json.dumps({"gate": res["gate"],
                          "agg": {k: {m: res["aggs"][k][m]["mean"] for m in res["aggs"][k]}
                                  for k in res["aggs"]}}, indent=2))
    elif args.mode == "prepare":
        meta = prepare_features(args)
        print(json.dumps(meta, indent=2))
    elif args.mode == "eval-teacher-models":
        run_eval_teacher(args)
    else:
        run_video(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

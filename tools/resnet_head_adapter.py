#!/usr/bin/env python3
"""resnet_head_adapter.py - frozen pretrained ResNet + light adapter head for the image IO line.

Task B of the "get the BASES back" round: instead of training encoders from scratch,
start from an ImageNet-pretrained ResNet (timm a1_in1k, fetched from hf-mirror) and fit
only a LIGHT adapter (D -> 128) plus the existing 128-d head, then compare with the live
in-house 128-d concept encoder on the same 3-seed held-out protocol.

Data: the recorded live-encoder feature cache from the previous head work
  runtime_store/models/io_train2/20261117_2330/image/features_image.npz.{train,held}.npz
  (x = 6000+2000 live 128-d concepts, y = 200 Tiny-ImageNet classes, paths = local JPEGs)
The pretrained backbone is frozen; only the adapter (+ optional existing head) is fitted.

Modes:
  --backbone resnet50|resnet18   (weights under runtime_store/models/resnet_bases/<tag>/)
  --extract-only                 only build/cache backbone features
  --seeds 20261118,...           3 random file-disjoint splits of the 8000 images
  --export-onnx                  when a variant wins the gate, export image -> [1,128,1,1]

Outputs (under --out-dir):
  features/resnet_features.npz          frozen backbone pooled features
  image_resnet_adapter/adapter_*.npz/.json
  image_resnet_adapter/best.onnx + model.manifest.json + contract + install instructions
  manifest.json                         protocol, provenance, 3-seed table, gate, verdict
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

try:
    import torchvision
except Exception as exc:  # pragma: no cover
    raise SystemExit("torchvision required: %s" % exc)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = "runtime_store/models/io_train2/20261117_2330/image"
BASE_DIR = "runtime_store/models/resnet_bases"
HEAD_NPZ = "runtime_store/models/io_train2/20261117_2330/image/head_a_linear128_deploy.npz"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def log(msg):
    print("[resnet-adapter] %s" % msg, flush=True)


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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2)


def l2(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


# ---------------- metrics (same defs as io_train_head.py / video_train_head.py) -------
def knn_metrics(x_tr, y_tr, x_te, y_te, k=5):
    xt = l2(x_te.astype(np.float32))
    xtr = l2(x_tr.astype(np.float32))
    sims = xt @ xtr.T
    top = np.argsort(-sims, axis=1)[:, :k]
    return {"knn_top1": float((y_tr[top[:, 0]] == y_te).mean()),
            "knn_top%d" % k: float(np.mean([y_te[i] in y_tr[top[i]] for i in range(len(y_te))]))}


def ridge_probe(x_tr, y_tr, x_te, y_te, lam=1e-2):
    classes = int(y_tr.max()) + 1
    X = l2(x_tr.astype(np.float32))
    Xt = l2(x_te.astype(np.float32))
    Y = np.zeros((len(y_tr), classes), np.float32)
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, X.T @ Y)
    scores = Xt @ W
    pred = scores.argmax(axis=1)
    return {"linear_probe_acc": float((pred == y_te).mean()),
            "linear_probe_mse": float(((scores - np.eye(classes, dtype=np.float32)[y_te]) ** 2).mean())}


def evaluate(tr, ytr, te, yte):
    out = knn_metrics(tr, ytr, te, yte)
    out.update(ridge_probe(tr, ytr, te, yte))
    return out


def agg(rows, keys):
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "sd": float(np.std([r[k] for r in rows])),
                "per_seed": [r[k] for r in rows]} for k in keys}


# ---------------- backbone -------------------------------------------------------------
def load_backbone(name, device="cpu"):
    tag = {"resnet50": "resnet50_a1_in1k", "resnet18": "resnet18_a1_in1k"}[name]
    wdir = os.path.join(ROOT, BASE_DIR, tag)
    st = os.path.join(wdir, "model.safetensors")
    if not os.path.exists(st):
        raise SystemExit("missing %s (download timm/%s.a1_in1k)" % (st, name))
    from safetensors import safe_open
    sd = {}
    with safe_open(st, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    model = getattr(torchvision.models, name)(pretrained=False, num_classes=1000)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 4 or len(unexpected) > 4:
        raise SystemExit("state_dict mismatch missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    model.fc = nn.Identity()                 # keep pooled 2048/512-d features, not logits
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    feat_dim = 2048 if name == "resnet50" else 512
    cfg = json.load(open(os.path.join(wdir, "config.json"), encoding="utf-8"))
    meta = {"backbone": name, "tag": tag, "weights": st, "weights_sha256": sha256_file(st),
            "weights_bytes": os.path.getsize(st), "feat_dim": feat_dim,
            "source": "hf-mirror.com timm/%s.a1_in1k" % name,
            "license": cfg.get("license", "apache-2.0"),
            "input_size": cfg.get("pretrained_cfg", {}).get("input_size"),
            "mean": cfg.get("pretrained_cfg", {}).get("mean"),
            "std": cfg.get("pretrained_cfg", {}).get("std"),
            "crop_pct": cfg.get("pretrained_cfg", {}).get("crop_pct")}
    return model, feat_dim, meta


def _load_image_224(path, crop_pct=0.95):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = 224.0 / crop_pct / min(w, h)
    im = im.resize((max(224, int(round(w * scale))), max(224, int(round(h * scale)))), Image.BICUBIC)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))


def extract_features(model, paths, batch=32, crop_pct=0.95):
    feats = None
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            chunk = paths[s:s + batch]
            xs = np.stack([_load_image_224(p, crop_pct) for p in chunk])
            out = model(torch.from_numpy(xs))
            if feats is None:
                feats = np.zeros((len(paths), out.shape[1]), dtype=np.float32)
            feats[s:s + len(chunk)] = out.numpy()
            if (s // batch + 1) % 10 == 0 or s + batch >= len(paths):
                done = min(s + batch, len(paths))
                el = time.time() - t0
                log("  features %d/%d (%.1fs, %.3fs/img)" % (done, len(paths), el, el / done))
    return feats


# ---------------- adapter training ------------------------------------------------------
def fit_adapter_ce(Ftr, ytr, in_dim, classes, seed, epochs=120, lr=1e-2, wd=1e-4, batch=256, threads=8):
    torch.manual_seed(int(seed))
    torch.set_num_threads(int(threads))
    adapter = nn.Linear(in_dim, 128)
    classifier = nn.Linear(128, classes)          # training-only; the concept is adapter's 128-d output
    opt = torch.optim.Adam(list(adapter.parameters()) + list(classifier.parameters()),
                           lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(Ftr.astype(np.float32))
    yt = torch.from_numpy(ytr.astype(np.int64))
    rng = np.random.RandomState(int(seed))
    adapter.train(); classifier.train()
    for _ in range(epochs):
        perm = rng.permutation(len(xt))
        for s in range(0, len(xt), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            logits = classifier(adapter(xt[idx]))
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
    adapter.eval()
    return adapter


def adapt_features(model, F):
    with torch.no_grad():
        return model(torch.from_numpy(F.astype(np.float32))).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet18"])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--seeds", default="20261118,20261119,20261120")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--export-onnx", action="store_true")
    ap.add_argument("--reuse-features", default="")
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    if not args.out_dir:
        args.out_dir = os.path.join(ROOT, "runtime_store/models/io_train2",
                                    "video_%s_resnet_%s" % (time.strftime("%Y%m%d_%H%M%S"), args.backbone))
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    log("out-dir %s" % args.out_dir)

    # ----- data: recorded live features + paths -----
    zp = np.load(os.path.join(ROOT, DATA, "features_image.npz.train.npz"), allow_pickle=False)
    zh = np.load(os.path.join(ROOT, DATA, "features_image.npz.held.npz"), allow_pickle=False)
    live = np.concatenate([np.asarray(zp["x"], np.float32), np.asarray(zh["x"], np.float32)])
    y = np.concatenate([np.asarray(zp["y"], np.int64), np.asarray(zh["y"], np.int64)])
    paths = [str(p) for p in np.concatenate([np.asarray(zp["paths"]), np.asarray(zh["paths"])])]
    paths = [p if os.path.isabs(p) else os.path.join(ROOT, p) for p in paths]
    log("data: %d images x %d live concepts, classes=%d" % (len(live), live.shape[1], int(y.max()) + 1))

    # existing 128->128 head (prior work), used as the "existing head" rung
    head = np.load(os.path.join(ROOT, HEAD_NPZ))
    Wh, bh = np.asarray(head["W"], np.float32), np.asarray(head["b"], np.float32)

    # ----- frozen backbone features -----
    model, feat_dim, meta = load_backbone(args.backbone)
    fdir = os.path.join(args.out_dir, "features")
    os.makedirs(fdir, exist_ok=True)
    fpath = os.path.join(fdir, "resnet_features.npz")
    if args.reuse_features and os.path.exists(args.reuse_features):
        z = np.load(args.reuse_features)
        R, paths_cached = np.asarray(z["features"], np.float32), list(z["paths"])
        assert len(R) == len(live), "feature cache length mismatch"
        log("reusing features %s" % args.reuse_features)
    else:
        log("extracting frozen %s features for %d images" % (args.backbone, len(paths)))
        R = extract_features(model, paths, crop_pct=meta.get("crop_pct") or 0.95)
        np.savez_compressed(fpath, features=R, paths=np.array(paths), y=y)
        log("features cached %s (%.1f MB)" % (fpath, os.path.getsize(fpath) / 1e6))
    log("backbone features %s (%d-d)" % (R.shape, feat_dim))
    if args.extract_only:
        write_json(os.path.join(args.out_dir, "extract_meta.json"),
                   {"backbone": meta, "features": fpath, "n": len(R)})
        return 0

    # ----- 3-seed protocol -----
    classes = int(y.max()) + 1
    n = len(y); n_held = 2000
    variants = ["baseline_live", "adapter_ce", "adapter_ce_existing_head", "fusion_existing_head"]
    rows = {v: [] for v in variants}
    for seed in args.seeds:
        rng = np.random.RandomState(int(seed))
        order = rng.permutation(n)
        held, tr = order[:n_held], order[n_held:]
        base = evaluate(live[tr], y[tr], live[held], y[held])
        rows["baseline_live"].append({"seed": seed, **base})
        adapter = fit_adapter_ce(R[tr], y[tr], feat_dim, classes, seed, threads=args.threads)
        A_tr = l2(adapt_features(adapter, R[tr]))
        A_te = l2(adapt_features(adapter, R[held]))
        m = evaluate(A_tr, y[tr], A_te, y[held])
        rows["adapter_ce"].append({"seed": seed, **m})
        # existing head applied to adapter output
        E_tr = l2(A_tr @ Wh.T + bh)
        E_te = l2(A_te @ Wh.T + bh)
        m = evaluate(E_tr, y[tr], E_te, y[held])
        rows["adapter_ce_existing_head"].append({"seed": seed, **m})
        # fusion: live concept + adapter concept, then existing head
        F_tr = l2(l2(live[tr]) + A_tr)
        F_te = l2(l2(live[held]) + A_te)
        F_tr = l2(F_tr @ Wh.T + bh)
        F_te = l2(F_te @ Wh.T + bh)
        m = evaluate(F_tr, y[tr], F_te, y[held])
        rows["fusion_existing_head"].append({"seed": seed, **m})
        log("seed %d: base top1=%.4f top5=%.4f probe=%.4f | adapter top1=%.4f top5=%.4f probe=%.4f | "
            "adapter+head top1=%.4f | fusion top1=%.4f"
            % (seed, base["knn_top1"], base["knn_top5"], base["linear_probe_acc"],
               rows["adapter_ce"][-1]["knn_top1"], rows["adapter_ce"][-1]["knn_top5"],
               rows["adapter_ce"][-1]["linear_probe_acc"],
               rows["adapter_ce_existing_head"][-1]["knn_top1"],
               rows["fusion_existing_head"][-1]["knn_top1"]))
    keys = ["knn_top1", "knn_top5", "linear_probe_acc"]
    aggs = {v: agg(rows[v], keys) for v in variants}
    gate = {}
    for v in variants[1:]:
        g = {k: aggs[v][k]["mean"] > aggs["baseline_live"][k]["mean"] for k in keys}
        g["all_improve"] = bool(all(g.values()))
        g["delta"] = {k: aggs[v][k]["mean"] - aggs["baseline_live"][k]["mean"] for k in keys}
        gate[v] = g
    best = max(variants[1:], key=lambda v: aggs[v]["knn_top1"]["mean"])
    win = bool(gate[best]["all_improve"])

    # ----- checkpoints -----
    adir = os.path.join(args.out_dir, "image_resnet_adapter")
    os.makedirs(adir, exist_ok=True)
    # retrain best-seed adapter for export (last seed, deterministic)
    seed = args.seeds[-1]
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(n)
    held, tr = order[:n_held], order[n_held:]
    adapter = fit_adapter_ce(R[tr], y[tr], feat_dim, classes, seed, threads=args.threads)
    W = adapter.weight.detach().numpy().astype(np.float32)
    b = adapter.bias.detach().numpy().astype(np.float32)
    np.savez(os.path.join(adir, "adapter.npz"), W=W, b=b, inDim=np.int64(feat_dim),
             outDim=np.int64(128), normalize=np.int64(1), backbone=args.backbone)
    write_json(os.path.join(adir, "adapter.json"),
               {"type": "linear_ce_adapter", "inDim": feat_dim, "outDim": 128,
                "backbone": meta, "seed": seed, "classes": classes,
                "metrics": {v: aggs[v] for v in variants}, "gate": gate, "win": win})

    verdict = "DEPLOY" if win else "NOT-DEPLOY"
    contract = {
        "format": "phoenix.io_encoder.v1",
        "modality": "image",
        "type": "frozen_pretrained_resnet_adapter",
        "backbone": {"name": meta["backbone"], "tag": meta["tag"],
                     "weights_sha256": meta["weights_sha256"],
                     "source": meta["source"], "license": meta["license"]},
        "input": {"name": "pixel_values", "shape": [1, 3, 224, 224],
                  "layout": "NCHW", "colorspace": "RGB",
                  "preprocess": "resize shortest side to 224/0.95 (bicubic), center-crop 224, /255, ImageNet mean/std"},
        "output": {"name": "concept", "shape": [1, 128, 1, 1], "dtype": "float32"},
        "adapter": {"type": "linear", "inDim": feat_dim, "outDim": 128,
                    "weightFile": "adapter.npz"},
        "integration": {
            "resolver_compatible": True,
            "install": "copy this dir (or just best.onnx + model.manifest.json + adapter) to "
                       "runtime_store/models/additive_jepa/vision_%s_adapter/ and set "
                       "jepa.image.variant=\"vision_%s_adapter\" (decoder can stay additive_jepa/vision_decoder)" %
                       (meta["tag"], meta["tag"]),
            "no_cpp_change": True,
            "why": "resolveOnnxModelPath already loads <variant>/best.onnx and the manifest declares the I/O; "
                   "the existing image decode path expects a 128-d concept, which this encoder emits.",
        },
        "metrics": {v: aggs[v] for v in variants},
        "gate": gate,
        "verdict": verdict,
    }
    write_json(os.path.join(adir, "contract.json"), contract)

    # ----- ONNX export (backbone + adapter fused) -----
    onnx_path = None
    if args.export_onnx:
        class EncoderWrap(nn.Module):
            def __init__(self, backbone, W, b):
                super().__init__()
                self.backbone = backbone
                self.adapter = nn.Linear(W.shape[1], W.shape[0])
                with torch.no_grad():
                    self.adapter.weight.copy_(torch.from_numpy(W))
                    self.adapter.bias.copy_(torch.from_numpy(b))

            def forward(self, x):
                f = self.backbone(x)
                z = self.adapter(f)
                return torch.nn.functional.normalize(z, dim=1).reshape(-1, 128, 1, 1)

        m = EncoderWrap(model, W, b).eval()
        dummy = torch.zeros(1, 3, 224, 224)
        onnx_path = os.path.join(adir, "best.onnx")
        torch.onnx.export(m, dummy, onnx_path, opset_version=11,
                          input_names=["pixel_values"], output_names=["concept"],
                          dynamic_axes={"pixel_values": {0: "batch"}, "concept": {0: "batch"}})
        # manifest expected by VideoLocalOnnxModel::loadManifest
        manifest = {
            "name": "vision_%s_adapter" % meta["tag"],
            "model_name": "vision_%s_adapter" % meta["tag"],
            "modality": "image",
            "concept_dim": 128,
            "input_name": "pixel_values",
            "input_shape": [1, 3, 224, 224],
            "output_name": "concept",
            "output_shape": [1, 128, 1, 1],
            "n_blocks": 0,
            "source_checkpoint": "timm/%s.a1_in1k (frozen) + linear CE adapter" % meta["backbone"],
            "resolution": 224,
        }
        write_json(os.path.join(adir, "model.manifest.json"), manifest)
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ref = sess.run(["concept"], {"pixel_values": np.zeros((1, 3, 224, 224), np.float32)})[0]
        log("onnx export OK %s bytes=%d out=%s" % (onnx_path, os.path.getsize(onnx_path), ref.shape))
    manifest = {
        "tool": "tools/resnet_head_adapter.py",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "B - pretrained ResNet base + light adapter head (no encoder from scratch)",
        "data": {"cache": DATA, "images": n, "classes": classes,
                 "train_per_seed": n - n_held, "heldout_per_seed": n_held,
                 "splits": "3 random file-disjoint splits (seed decides split)",
                 "note": "live 128-d concepts are the recorded baseline for the same images"},
        "backbone": meta,
        "protocol": {"adapter": "Linear(D->128) trained with cross-entropy on clip/image labels",
                     "head": "existing 128->128 head from io_train2/20261117_2330/image",
                     "readouts": ["cosine 1-NN top-1/top-5", "closed-form ridge probe"]},
        "three_seed_table": {v: aggs[v] for v in variants},
        "gate": gate,
        "best_variant": best,
        "verdict": verdict,
        "checkpoints": {"adapter": os.path.join(adir, "adapter.npz"),
                        "contract": os.path.join(adir, "contract.json"),
                        "onnx": onnx_path},
        "gpu_note": "frozen-backbone CPU run; a full encoder fine-tune is a GPU job (see doc/v10.1/video_training.md)",
    }
    write_json(os.path.join(args.out_dir, "manifest.json"), manifest)
    print(json.dumps({"verdict": verdict, "best": best,
                      "baseline": {k: aggs["baseline_live"][k]["mean"] for k in keys},
                      "best_metrics": {k: aggs[best][k]["mean"] for k in keys}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())



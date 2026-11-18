#!/usr/bin/env python3
"""llava_signal_matrix.py - LLaVA-1.5-7B vision teacher as a 576x4096 SIGNAL MATRIX.

Uses the ON-DISK weights (no 7B LLM needed):
  <model-dir>/vision_tower/model.safetensors   (CLIP ViT-L/14-336, 391 tensors, fp32)
  <model-dir>/multi_modal_projector.pt         (linear_1 1024->4096, linear_2 4096->4096, GELU)
  <model-dir>/preprocessor_config.json         (CLIP 336 preprocessing)
Default model-dir is the stale-but-valid tree:
  E:\\079\\v6.0Alixander\\runtime_store\\models\\multimodal_enc\\llava-1.5-7b-hf_full

For one image:
  image (RGB) -> resize shortest 336 / center-crop 336 / CLIP normalize
    -> CLIP ViT-L/14-336, hidden_states[-2] (vision_feature_layer)
    -> drop CLS (select "default") -> [576, 1024]
    -> projector (linear_1 -> GELU -> linear_2) -> MATRIX [576, 4096]
  mean over the 576 tokens -> compatibility view [4096]

Modes
-----
  (default)              batch/single extraction; --out npz (means + optional matrices)
  --validate             mandatory check vs runtime_store/models/large_vision_encoder caches
  --l1                   L1 single-frame read-outs (matrix vs 128-d view, non-classification)
  --l2-extract           L2: T x 576 x 4096 per HMDB51 clip shard (mean-token stored per frame)
  --l2-metrics           L2: rollout/predictability + ARI/NMI environment separability
  --cost                 CPU cost per frame and projected per mission-tick cost

Everything is CPU-only, deterministic (eval mode, no grad); live model dirs are read-only.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import glob
import json
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
from safetensors import safe_open

try:
    from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor
except Exception as exc:  # pragma: no cover
    raise SystemExit("transformers required: %s" % exc)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALE_DEFAULT = r"E:\079\v6.0Alixander\runtime_store\models\multimodal_enc\llava-1.5-7b-hf_full"
LIVE_CACHE_DIR = "runtime_store/models/large_vision_encoder"
LIVE_ENCODER = "runtime_store/models/additive_jepa/vision_encoder/best.onnx"


def log(msg):
    print("[signal-matrix] %s" % msg, flush=True)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def peak_rss_mb():
    """Peak working set of this process (Windows psapi)."""
    class PPMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        c = PPMC(); c.cb = ctypes.sizeof(PPMC)
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PPMC), wt.DWORD]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return float(c.PeakWorkingSetSize) / (1024 * 1024)
        return -2.0
    except Exception:
        return -1.0


def sha256_file(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# model loading (tower + projector)
# ---------------------------------------------------------------------------
def load_model(model_dir):
    vt = os.path.join(model_dir, "vision_tower")
    st = os.path.join(vt, "model.safetensors")
    if not os.path.exists(st):
        st = os.path.join(model_dir, "model.safetensors")
    cfg_path = os.path.join(vt, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else \
        json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))["vision_config"]
    tower_cfg = CLIPVisionConfig(
        hidden_size=cfg["hidden_size"], image_size=cfg["image_size"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_hidden_layers=cfg["num_hidden_layers"],
        patch_size=cfg["patch_size"], projection_dim=cfg.get("projection_dim", 768),
        hidden_act=cfg.get("hidden_act", "quick_gelu"),
    )
    tower = CLIPVisionModel(tower_cfg)
    sd = {}
    with safe_open(st, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("vision_model."):
                sd[k] = f.get_tensor(k)
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit("vision tower missing keys: %s" % missing[:5])
    tower.eval()
    # projector
    ppt = os.path.join(model_dir, "multi_modal_projector.pt")
    proj = torch.load(ppt, map_location="cpu")
    if not isinstance(proj, dict):
        proj = dict(proj)
    need = ["linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias"]
    if not all(k in proj for k in need):
        raise SystemExit("projector keys missing: %s" % [k for k in need if k not in proj])
    l1 = nn.Linear(proj["linear_1.weight"].shape[1], proj["linear_1.weight"].shape[0])
    l2 = nn.Linear(proj["linear_2.weight"].shape[1], proj["linear_2.weight"].shape[0])
    with torch.no_grad():
        l1.weight.copy_(proj["linear_1.weight"].float()); l1.bias.copy_(proj["linear_1.bias"].float())
        l2.weight.copy_(proj["linear_2.weight"].float()); l2.bias.copy_(proj["linear_2.bias"].float())
    l1.eval(); l2.eval()
    act = nn.GELU()
    pp = os.path.join(model_dir, "preprocessor_config.json")
    if os.path.exists(pp):
        processor = CLIPImageProcessor.from_pretrained(model_dir)
    else:
        processor = CLIPImageProcessor(size={"shortest_edge": 336}, crop_size={"height": 336, "width": 336},
                                       image_mean=[0.48145466, 0.4578275, 0.40821073],
                                       image_std=[0.26862954, 0.26130258, 0.27577711],
                                       resample=3, do_center_crop=True, do_resize=True,
                                       do_rescale=True, do_normalize=True)
    meta = {
        "createdAt": now(), "modelDir": model_dir,
        "visionTower": {"path": st, "bytes": os.path.getsize(st), "sha256": sha256_file(st),
                        "config": {k: cfg.get(k) for k in
                                   ("hidden_size", "image_size", "num_hidden_layers", "patch_size",
                                    "num_attention_heads", "intermediate_size", "hidden_act")}},
        "projector": {"path": ppt, "bytes": os.path.getsize(ppt), "sha256": sha256_file(ppt),
                      "linear_1": [l1.in_features, l1.out_features],
                      "linear_2": [l2.in_features, l2.out_features], "act": "gelu",
                      "dtype": "float32"},
        "preprocess": {"size": 336, "crop": 336, "resample": 3,
                       "mean": [0.48145466, 0.4578275, 0.40821073],
                       "std": [0.26862954, 0.26130258, 0.27577711]},
        "visionFeatureLayer": -2, "selectStrategy": "default (drop CLS)",
    }
    return tower, l1, l2, act, processor, meta


# ---------------------------------------------------------------------------
# matrix extraction
# ---------------------------------------------------------------------------
def preprocess(processor, images):
    return processor(images=images, return_tensors="pt")["pixel_values"].float()


def matrix_batch(tower, l1, l2, act, processor, images, layer=-2):
    px = preprocess(processor, images)
    with torch.no_grad():
        out = tower(pixel_values=px, output_hidden_states=True)
        hs = out.hidden_states[layer][:, 1:, :]          # drop CLS -> [B,576,1024]
        mat = l2(act(l1(hs)))                            # [B,576,4096]
    return mat.numpy()


def load_paths(args):
    if args.paths_json:
        data = json.load(open(args.paths_json, encoding="utf-8"))
        return [str(p) for p in data]
    if args.image_dir:
        out = []
        for e in ("*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG"):
            out += glob.glob(os.path.join(args.image_dir, e))
        return sorted(out)
    if args.image:
        return [args.image]
    raise SystemExit("no images: --image/--image-dir/--paths-json")


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def run_extract(args):
    tower, l1, l2, act, processor, meta = load_model(args.model_dir)
    torch.set_num_threads(int(args.threads))
    paths = load_paths(args)
    log("extract %d images; model loaded, peakRSS=%.0f MB" % (len(paths), peak_rss_mb()))
    matrices = [] if args.save_matrices else None
    means = np.zeros((len(paths), 4096), np.float32)
    timings = []
    for s in range(0, len(paths), args.batch):
        chunk = paths[s:s + args.batch]
        from PIL import Image
        ims = [Image.open(_abs(p)).convert("RGB") for p in chunk]
        t = time.time()
        mat = matrix_batch(tower, l1, l2, act, processor, ims)
        timings.append(time.time() - t)
        means[s:s + len(chunk)] = mat.mean(axis=1)
        if matrices is not None:
            matrices.append(mat.astype(np.float16))
        if (s // args.batch + 1) % 5 == 0 or s + args.batch >= len(paths):
            done = min(s + len(chunk), len(paths))
            log("  %d/%d  mean %.2fs/frame" % (done, len(paths), np.mean(timings) / args.batch))
    per_frame = float(np.mean(timings) / args.batch)
    out = {"mean": means, "paths": np.array(paths),
           "meta": json.dumps({**meta, "perFrameSeconds": per_frame, "threads": args.threads})}
    if matrices is not None:
        out["matrix"] = np.concatenate(matrices, axis=0)
    out_path = args.out or os.path.join(ROOT, "build/tmp/llava_signal_matrix.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(out_path, **out)
    side = {"createdAt": now(), "out": out_path, "n": len(paths),
            "matrixShape": [len(paths), 576, 4096] if matrices is not None else None,
            "meanShape": [len(paths), 4096],
            "perFrameSeconds": per_frame, "threads": args.threads,
            "peakRssMB": peak_rss_mb(), "meta": meta}
    json.dump(side, open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8"), indent=2)
    log("wrote %s (%.1f MB); perFrame=%.3fs peakRSS=%.0f MB"
        % (out_path, os.path.getsize(out_path) / 1e6, per_frame, peak_rss_mb()))
    return side


# ---------------------------------------------------------------------------
# validation against recorded caches
# ---------------------------------------------------------------------------
def run_validate(args):
    tower, l1, l2, act, processor, meta = load_model(args.model_dir)
    torch.set_num_threads(int(args.threads))
    cache = np.load(os.path.join(ROOT, LIVE_CACHE_DIR, "llava_features_more.npy"), mmap_mode="r")
    cache_paths = [str(p) for p in json.load(open(os.path.join(ROOT, LIVE_CACHE_DIR, "llava_more_paths.json"))) ]
    idx = {os.path.normcase(os.path.normpath(p)): i for i, p in enumerate(cache_paths)}
    picks = []
    for i in range(len(cache_paths)):
        if i % max(1, len(cache_paths) // args.n) == 0:
            picks.append((i, cache_paths[i]))
        if len(picks) >= args.n:
            break
    from PIL import Image
    rows = []
    for k, (ci, p) in enumerate(picks):
        im = Image.open(_abs(p)).convert("RGB")
        t = time.time()
        mat = matrix_batch(tower, l1, l2, act, processor, [im])[0]
        dt = time.time() - t
        ref = np.asarray(cache[ci], np.float32)
        fn = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
        rn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-8)
        tok = float(np.mean(np.sum(fn * rn, axis=1)))
        pool_m = mat.mean(axis=0); pool_r = ref.mean(axis=0)
        pool = float(np.dot(pool_m, pool_r) / max(1e-9, np.linalg.norm(pool_m) * np.linalg.norm(pool_r)))
        rel = float(np.linalg.norm(mat - ref) / max(1e-9, np.linalg.norm(ref)))
        rows.append({"path": p, "cacheRow": ci, "matrix_cos": tok, "pooled_cos": pool,
                     "rel_fro_err": rel, "seconds": dt})
        log("  %2d %-55s matrix_cos=%.6f pooled_cos=%.6f rel=%.2e %.2fs"
            % (k + 1, os.path.basename(p), tok, pool, rel, dt))
    tok = np.array([r["matrix_cos"] for r in rows]); pool = np.array([r["pooled_cos"] for r in rows])
    summary = {"n": len(rows), "matrix_cos_mean": float(tok.mean()),
               "matrix_cos_p5": float(np.percentile(tok, 5)), "matrix_cos_min": float(tok.min()),
               "pooled_cos_mean": float(pool.mean()), "pooled_cos_p5": float(np.percentile(pool, 5)),
               "pooled_cos_min": float(pool.min()),
               "target": args.min_cos, "pass": bool(pool.mean() >= args.min_cos),
               "perFrameSeconds": float(np.mean([r["seconds"] for r in rows])),
               "peakRssMB": peak_rss_mb(), "rows": rows, "meta": meta}
    log("VALIDATION n=%d matrix_cos mean=%.6f p5=%.6f min=%.6f | pooled mean=%.6f p5=%.6f min=%.6f -> %s"
        % (summary["n"], summary["matrix_cos_mean"], summary["matrix_cos_p5"], summary["matrix_cos_min"],
           summary["pooled_cos_mean"], summary["pooled_cos_p5"], summary["pooled_cos_min"],
           "PASS" if summary["pass"] else "FAIL"))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        json.dump(summary, open(args.report, "w", encoding="utf-8"), indent=2)
        log("report %s" % args.report)
    return 0 if summary["pass"] else 2


# ---------------------------------------------------------------------------
# L1 read-outs (non-classification)
# ---------------------------------------------------------------------------
def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _pairwise_cos_dist(x):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    sim = x @ x.T
    return 1.0 - sim


def run_l1(args):
    tower, l1, l2, act, processor, meta = load_model(args.model_dir)
    torch.set_num_threads(int(args.threads))
    cache = np.load(os.path.join(ROOT, LIVE_CACHE_DIR, "llava_features_more.npy"), mmap_mode="r")
    cache_paths = [str(p) for p in json.load(open(os.path.join(ROOT, LIVE_CACHE_DIR, "llava_more_paths.json")))]
    from PIL import Image
    n = min(args.n, 256)
    picks = list(range(0, len(cache_paths), max(1, len(cache_paths) // n)))[:n]
    # live 128-d view for the same images
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = int(args.threads); so.log_severity_level = 3
    enc = ort.InferenceSession(os.path.join(ROOT, LIVE_ENCODER), so, providers=["CPUExecutionProvider"])
    mean_in = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std_in = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    mats = []; means = []; live = []
    t0 = time.time()
    for ci in picks:
        p = cache_paths[ci]
        im = Image.open(_abs(p)).convert("RGB")
        mats.append(matrix_batch(tower, l1, l2, act, processor, [im])[0])
        x = np.asarray(im.resize((224, 224)), np.float32) / 255.0
        x = ((np.transpose(x, (2, 0, 1)) - mean_in) / std_in)[None, ...]
        live.append(np.asarray(enc.run(["concept"], {"pixel_values": x})[0], np.float32).reshape(-1))
        means.append(mats[-1].mean(axis=0))
    mats = np.stack(mats); means = np.stack(means); live = np.stack(live)
    refs = np.stack([np.asarray(cache[ci], np.float32) for ci in picks])
    ref_means = refs.mean(axis=1)
    # fidelity
    cos_matrix = float(np.mean([np.mean(np.sum(
        (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)) *
        (r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)), axis=1))
        for m, r in zip(mats, refs)]))
    cos_pool = float(np.mean(np.sum((means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-8)) *
                                    (ref_means / np.maximum(np.linalg.norm(ref_means, axis=1, keepdims=True), 1e-8)), axis=1)))
    rel_err = float(np.mean(np.linalg.norm(mats - refs, axis=(1, 2)) / np.linalg.norm(refs, axis=(1, 2))))
    # geometry / non-classification read-outs
    mat_flat = mats.reshape(len(mats), -1)
    D_matrix = _pairwise_cos_dist(means)          # 4096 pooled signal geometry
    D_cache = _pairwise_cos_dist(ref_means)
    D_live = _pairwise_cos_dist(live)             # 128-d live view geometry
    rsa_matrix_cache = _spearman(D_matrix[np.triu_indices(len(means), 1)], D_cache[np.triu_indices(len(means), 1)])
    rsa_matrix_live = _spearman(D_matrix[np.triu_indices(len(means), 1)], D_live[np.triu_indices(len(means), 1)])
    sv = np.linalg.svd(means, compute_uv=False)
    eff_rank = float(np.exp(-np.sum((sv / sv.sum()) * np.log(np.maximum(sv / sv.sum(), 1e-12)))))
    token_energy = float(np.mean(np.linalg.norm(mats, axis=2)))
    anisotropy = float((np.linalg.norm(means.mean(axis=0)) ** 2) /
                       np.mean(np.linalg.norm(means, axis=1) ** 2))
    out = {
        "createdAt": now(), "n": len(picks), "shapes": {"matrix": list(mats.shape), "meanToken": list(means.shape), "live128": list(live.shape)},
        "fidelity": {"matrix_cos_mean": cos_matrix, "pooled_cos_mean": cos_pool, "rel_fro_err_mean": rel_err,
                     "token_count": 576, "dim": 4096},
        "signal_geometry": {"token_energy_mean": token_energy,
                            "mean_token_effective_rank": eff_rank,
                            "anisotropy": anisotropy,
                            "rsa_matrix_vs_cached_spearman": rsa_matrix_cache,
                            "rsa_matrix_vs_live128_spearman": rsa_matrix_live},
        "runtime": {"seconds": time.time() - t0, "perFrameSeconds": float((time.time() - t0) / len(picks)),
                    "peakRssMB": peak_rss_mb()},
        "meta": meta,
    }
    log("L1: matrix_cos=%.6f pooled_cos=%.6f rel_err=%.2e | eff_rank(pool4096)=%.1f | "
        "RSA(matrix,cached)=%.4f RSA(matrix,live128)=%.4f | %.2fs/frame"
        % (cos_matrix, cos_pool, rel_err, eff_rank, rsa_matrix_cache, rsa_matrix_live,
           out["runtime"]["perFrameSeconds"]))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        json.dump(out, open(args.report, "w", encoding="utf-8"), indent=2)
        log("L1 report %s" % args.report)
    return 0


# ---------------------------------------------------------------------------
# L2: temporal matrix extraction over HMDB51 shards
# ---------------------------------------------------------------------------
def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_clip_frames(ff, path, n_frames=4, res=336):
    """Decode n_frames evenly spaced RGB frames at res x res (336 for the CLIP teacher)."""
    import re, subprocess
    def probe(p):
        r = subprocess.run([ff, "-hide_banner", "-i", p], capture_output=True, text=True,
                           errors="replace", timeout=30)
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr or "")
        f = re.search(r"(\d+(?:\.\d+)?) fps", r.stderr or "")
        dur = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else None
        return dur, (float(f.group(1)) if f else None)
    dur, fps = probe(path)
    est = max(1.0, (dur or 2.0) * (fps or 25.0))
    step = max(1, int(round(est / n_frames)))
    vf = "select='not(mod(n\\,%d))',scale=%d:%d" % (step, res, res)
    buf = subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", path, "-vf", vf,
                          "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True, timeout=180).stdout or b""
    size = res * res * 3
    got = len(buf) // size
    frames = np.frombuffer(buf[:got * size], np.uint8).reshape(got, res, res, 3)
    if got == 0:
        raise RuntimeError("no frames from %s" % path)
    if got < n_frames:
        frames = np.concatenate([frames, np.repeat(frames[-1:], n_frames - got, axis=0)], 0)
    return frames[:n_frames]


def run_l2_extract(args):
    tower, l1, l2, act, processor, meta = load_model(args.model_dir)
    torch.set_num_threads(int(args.threads))
    ff = _ffmpeg_exe()
    clips_root = _abs(args.clips_dir)
    class_dirs = [d for d in sorted(glob.glob(os.path.join(clips_root, "*"))) if os.path.isdir(d)]
    clips = []
    for d in class_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.avi"))):
            clips.append((os.path.basename(d), f))
    if args.clips > 0:
        rng = np.random.RandomState(20261118)
        idx = np.sort(rng.permutation(len(clips))[:args.clips])
        clips = [clips[i] for i in idx]
    shard = clips[args.shard::args.num_shards]
    log("L2-extract: %d clips total, shard %d/%d -> %d clips x %d frames"
        % (len(clips), args.shard, args.num_shards, len(shard), args.frames))
    from PIL import Image
    means = np.zeros((len(shard), args.frames, 4096), np.float32)
    live = np.zeros((len(shard), args.frames, 128), np.float32)
    labels = []; paths = []
    mean_in = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    std_in = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = int(args.threads); so.log_severity_level = 3
    enc = ort.InferenceSession(os.path.join(ROOT, LIVE_ENCODER), so, providers=["CPUExecutionProvider"])
    t0 = time.time()
    for k, (cls, path) in enumerate(shard):
        frames = decode_clip_frames(ff, path, args.frames)
        ims = [Image.fromarray(f) for f in frames]
        mat = matrix_batch(tower, l1, l2, act, processor, ims)
        means[k] = mat.mean(axis=1)
        for j, im in enumerate(ims):
            x = np.asarray(im.resize((224, 224)), np.float32) / 255.0
            x = ((np.transpose(x, (2, 0, 1)) - mean_in) / std_in)[None, ...]
            live[k, j] = np.asarray(enc.run(["concept"], {"pixel_values": x})[0], np.float32).reshape(-1)
        labels.append(cls); paths.append(path)
        if (k + 1) % 20 == 0:
            el = time.time() - t0
            log("  %d/%d clips (%.1fs, %.2fs/frame)" % (k + 1, len(shard), el, el / ((k + 1) * args.frames)))
    out = os.path.join(args.out_dir, "l2_shard_%d.npz" % args.shard)
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(out, frame_means=means, live128=live,
                        labels=np.array(labels), paths=np.array(paths))
    sec = time.time() - t0
    side = {"createdAt": now(), "shard": args.shard, "numShards": args.num_shards,
            "clips": len(shard), "frames": args.frames,
            "shapes": {"frame_means": list(means.shape), "live128": list(live.shape)},
            "seconds": sec, "perFrameSeconds": sec / max(1, len(shard) * args.frames),
            "peakRssMB": peak_rss_mb(), "out": out, "meta": meta}
    json.dump(side, open(out + ".json", "w", encoding="utf-8"), indent=2)
    log("L2 shard done: %s (%.2fs/frame)" % (out, side["perFrameSeconds"]))
    return 0


# ---------------------------------------------------------------------------
# L2 metrics: rollout/predictability + ARI/NMI environment separability
# ---------------------------------------------------------------------------
def _kmeans(x, k, seed=0, iters=25):
    rng = np.random.RandomState(seed)
    n = len(x)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    centers = x[rng.choice(n, size=min(k, n), replace=False)]
    labels = np.zeros(n, np.int64)
    for _ in range(iters):
        sims = x @ centers.T
        new = np.argmax(sims, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(len(centers)):
            m = labels == c
            if m.any():
                centers[c] = x[m].mean(axis=0)
    return labels


def _ari(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ka, kb = a.max() + 1, b.max() + 1
    n = np.zeros((ka, kb), np.int64)
    for i in range(len(a)):
        n[a[i], b[i]] += 1
    def comb2(x):
        return x * (x - 1) / 2.0
    sum_comb = comb2(n).sum()
    ai = comb2(n.sum(axis=1)).sum(); bj = comb2(n.sum(axis=0)).sum()
    total = comb2(len(a))
    exp = ai * bj / total if total > 0 else 0.0
    mx = (ai + bj) / 2.0
    return float((sum_comb - exp) / (mx - exp)) if mx != exp else 1.0


def _nmi(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ka, kb = a.max() + 1, b.max() + 1
    n = np.zeros((ka, kb), np.float64)
    for i in range(len(a)):
        n[a[i], b[i]] += 1
    n = n / n.sum()
    pa = n.sum(axis=1); pb = n.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(n * np.log(n / (pa[:, None] * pb[None, :])))
        ha = -np.nansum(pa * np.log(pa)); hb = -np.nansum(pb * np.log(pb))
    return float(2 * mi / (ha + hb)) if (ha + hb) > 0 else 0.0


def run_l2_metrics(args):
    import glob as _g
    parts = sorted(_g.glob(os.path.join(args.l2_dir, "l2_shard_*.npz")))
    if not parts:
        raise SystemExit("no l2 shard npz under %s" % args.l2_dir)
    fm = np.concatenate([np.load(p)["frame_means"] for p in parts], 0)
    lv = np.concatenate([np.load(p)["live128"] for p in parts], 0)
    labels = np.concatenate([np.load(p)["labels"] for p in parts], 0)
    paths = np.concatenate([np.load(p)["paths"] for p in parts], 0)
    C, T = fm.shape[0], fm.shape[1]
    log("l2 metrics: %d clips x %d frames" % (C, T))
    l2n = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    clip_m = l2n(fm.mean(axis=1))       # [C,4096]
    clip_l = l2n(lv.mean(axis=1))       # [C,128]
    # ---- ARI/NMI environment separability (class = environment) ----
    cls_names = sorted(set(labels.tolist()))
    cls_idx = {c: i for i, c in enumerate(cls_names)}
    y = np.array([cls_idx[c] for c in labels])
    km = _kmeans(clip_m, len(cls_names), seed=0)
    kl = _kmeans(clip_l, len(cls_names), seed=0)
    sep = {"classes": len(cls_names), "clips": C,
           "matrix4096": {"ARI": _ari(y, km), "NMI": _nmi(y, km)},
           "live128": {"ARI": _ari(y, kl), "NMI": _nmi(y, kl)}}
    # ---- clip-identity (temporal environment) separability from frames ----
    fr_m = l2n(fm[:, ::max(1, T // 2)].reshape(-1, 4096))
    fr_l = l2n(lv[:, ::max(1, T // 2)].reshape(-1, 128))
    fid = np.repeat(np.arange(C), fr_m.shape[0] // C)
    km2 = _kmeans(fr_m, min(C, 100), seed=0)
    kl2 = _kmeans(fr_l, min(C, 100), seed=0)
    sep["clip_identity_from_frames"] = {
        "matrix4096": {"ARI": _ari(fid, km2), "NMI": _nmi(fid, km2)},
        "live128": {"ARI": _ari(fid, kl2), "NMI": _nmi(fid, kl2)}}
    # ---- rollout / predictability: frame t -> frame t+1 (ridge), clip-disjoint ----
    rng = np.random.RandomState(20261118)
    order = rng.permutation(C)
    ntr = int(0.8 * C)
    tr, te = order[:ntr], order[ntr:]

    def rollout(x, name):
        x = l2n(x)
        Xtr = x[tr, :-1].reshape(-1, x.shape[-1]); Ytr = x[tr, 1:].reshape(-1, x.shape[-1])
        Xte = x[te, :-1].reshape(-1, x.shape[-1]); Yte = x[te, 1:].reshape(-1, x.shape[-1])
        lam = 1e-2
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
        W = np.linalg.solve(A, Xtr.T @ Ytr)
        pred = Xte @ W
        cos = float(np.mean(np.sum(l2n(pred) * l2n(Yte), axis=1)))
        r2 = float(1 - ((pred - Yte) ** 2).sum() / max(1e-9, ((Yte - Ytr.mean(0)) ** 2).sum()))
        ident = float(np.mean(np.sum(l2n(Xte) * l2n(Yte), axis=1)))
        return {"featureDim": int(x.shape[-1]), "top1_step_rollout_cos": cos,
                "r2": r2, "identity_cos": ident, "improvement_over_identity": cos - ident, "ridge_lambda": lam}

    roll = {"matrix4096": rollout(np.asarray(fm, np.float32), "matrix"),
            "live128": rollout(np.asarray(lv, np.float32), "live")}
    out = {"createdAt": now(), "shards": [os.path.basename(p) for p in parts],
           "clips": C, "frames": T, "separability": sep, "rollout": roll,
           "cost": {"perFrameSeconds": args.per_frame_seconds,
                    "perClipFrames": T,
                    "perClipSeconds": args.per_frame_seconds * T,
                    "note": "matrix path; live128 is negligible (0.03 s/frame)"}}
    out_path = os.path.join(args.l2_dir, "l2_metrics.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    log("L2 metrics -> %s" % out_path)
    print(json.dumps(out, indent=2)[:4000])
    return 0


# ---------------------------------------------------------------------------
# CPU cost projection
# ---------------------------------------------------------------------------
def run_cost(args):
    tower, l1, l2, act, processor, meta = load_model(args.model_dir)
    torch.set_num_threads(int(args.threads))
    from PIL import Image
    cache_paths = [str(p) for p in json.load(open(os.path.join(ROOT, LIVE_CACHE_DIR, "llava_more_paths.json")))][:16]
    ims = [Image.open(_abs(p)).convert("RGB") for p in cache_paths[:8]]
    # warmup
    matrix_batch(tower, l1, l2, act, processor, ims[:2])
    t = time.time()
    matrix_batch(tower, l1, l2, act, processor, ims)
    dt = time.time() - t
    per_frame = dt / 8
    tick_1 = per_frame
    tick_8 = per_frame * 8
    out = {"createdAt": now(), "threads": args.threads,
           "modelLoadBytes": {"visionTower": meta["visionTower"]["bytes"],
                              "projector": meta["projector"]["bytes"]},
           "peakRssMB": peak_rss_mb(),
           "perFrameSeconds": per_frame,
           "perClip8FramesSeconds": tick_8,
           "projections": {
               "missionTick_1frame": tick_1,
               "missionTick_4frames": per_frame * 4,
               "missionTick_8frames": tick_8,
               "framesPerMinute": 60.0 / per_frame,
               "clips8FramesPerMinute": 60.0 / tick_8,
           },
           "interpretation": "2-3 s/frame on this CPU (Xeon E5-2680 v4, no usable GPU: 187/6144 MiB "
                             "free VRAM). Inline single-frame encoding would add ~2-3 s per bridge "
                             "call; the existing python subprocess bridge is the right seam, but a "
                             "GPU or a smaller/quantized teacher is required for real-time use."}
    log(json.dumps(out, indent=2))
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        json.dump(out, open(args.report, "w", encoding="utf-8"), indent=2)
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=STALE_DEFAULT)
    ap.add_argument("--image", default="")
    ap.add_argument("--image-dir", default="")
    ap.add_argument("--paths-json", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--save-matrices", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--l1", action="store_true")
    ap.add_argument("--l2-extract", action="store_true")
    ap.add_argument("--l2-metrics", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-cos", type=float, default=0.99)
    ap.add_argument("--report", default="")
    # L2 args
    ap.add_argument("--clips-dir", default="runtime_store/datasets/video/hmdb51")
    ap.add_argument("--clips", type=int, default=120)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--l2-dir", default="")
    ap.add_argument("--per-frame-seconds", type=float, default=2.0)
    args = ap.parse_args(argv)
    if args.cost:
        run_cost(args); return 0
    if args.validate:
        return run_validate(args)
    if args.l1:
        return run_l1(args)
    if args.l2_metrics:
        return run_l2_metrics(args)
    if args.l2_extract:
        if not args.out_dir:
            raise SystemExit("--l2-extract needs --out-dir")
        return run_l2_extract(args)
    run_extract(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

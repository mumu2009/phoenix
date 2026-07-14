from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None

HF_AVAILABLE = torch is not None and AutoModelForCausalLM is not None and AutoTokenizer is not None


try:
    from transformer import GnnTransformerModel as _BaseModel  # type: ignore
except Exception:
    class _BaseModel:
        def __init__(self, params: Any) -> None:
            self._base_params = params


try:
    from transformer import HashTokenizer as _BaseTokenizer  # type: ignore
except Exception:
    class _BaseTokenizer:
        def __init__(self, vocab_size: int) -> None:
            self.vocab_size = int(vocab_size)


try:
    from transformer import TransformerParams as _ImportedTransformerParams  # type: ignore
except Exception:
    _ImportedTransformerParams = None


try:
    from transformer import TrainSample as _ImportedTrainSample  # type: ignore
except Exception:
    _ImportedTrainSample = None


if _ImportedTrainSample is None:
    @dataclass
    class TrainSample:
        input: str
        target: str
        graph: str = ""
else:
    TrainSample = _ImportedTrainSample


@dataclass
class _FallbackTransformerParams:
    vocabSize: int = 8192
    dModel: int = 256
    nHeads: int = 8
    nLayers: int = 4
    dFF: int = 512
    maxLen: int = 256
    maxTokens: int = 128
    temperature: float = 1.0
    dynamicSampling: bool = False
    temperatureMin: float = 0.7
    temperatureMax: float = 1.2
    topK: int = 20
    topP: float = 0.95
    topPMin: float = 0.85
    topPMax: float = 0.98
    dynamicWindow: int = 48
    dynamicRepetitionBoost: float = 0.5
    repetitionPenalty: float = 1.08
    repetitionSegmented: bool = True
    repetitionWindowRecent: int = 24
    repetitionWindowMid: int = 96
    repetitionPenaltyRecent: float = 1.18
    repetitionPenaltyMid: float = 1.12
    repetitionPenaltyOld: float = 1.05
    noRepeatNgram: int = 3
    minNewTokens: int = 8
    useLrSchedule: bool = True
    lrWarmupSteps: int = 32
    lrCosineSteps: int = 400
    lrMinRatio: float = 0.1
    gradAccumSteps: int = 1
    lossNoiseAlpha: float = 0.0
    attnChunkSize: int = 0
    useAlibi: bool = False
    alibiSlope: float = 0.02
    bidirectionalWeight: float = 0.0
    selfFlowWeight: float = 0.0
    enableMoE: bool = False
    moeExperts: int = 4
    moeTopK: int = 2
    moeAuxWeight: float = 0.01
    enableMLA: bool = False
    mlaRank: int = 8
    mlaScale: float = 0.2
    enableCoefAttention: bool = False
    coefAttentionScale: float = 0.1
    enableHierEmbedding: bool = False
    hierStride: int = 8
    windowClipTokens: int = 0
    draftTokens: int = 1
    enableMultiTokenObjective: bool = False
    multiTokenHeads: int = 1
    multiTokenLossWeight: float = 0.2
    enableCoTScaffold: bool = False
    cotSteps: int = 1
    enableProgramSynthesis: bool = False
    programSynthesisBias: float = 0.0
    enableDynamicReasoning: bool = False
    simpleTokenThreshold: int = 16
    enableRLHF: bool = False
    rlhfLr: float = 1e-4
    rlhfBatchSize: int = 8
    rlhfReplaySize: int = 256
    rlhfMargin: float = 0.2
    enableIRL: bool = False
    irlLr: float = 1e-4
    enableAddonModuleLearning: bool = False
    addonRewardScale: float = 1.0
    tokenizerMode: str = "bpe"
    graphWeight: float = 0.8
    cacheTtlMs: int = 180000
    cacheTokenEntries: int = 256
    weightDecay: float = 0.0
    adamBeta1: float = 0.9
    adamBeta2: float = 0.999
    adamEps: float = 1e-8
    lr: float = 2e-4
    gradClip: float = 1.0


TransformerParams = _ImportedTransformerParams or _FallbackTransformerParams


class Matrix:
    def __init__(self, rows: int, cols: int, data: Optional[List[float]] = None) -> None:
        self.rows = int(rows)
        self.cols = int(cols)
        if data is None:
            self.data = [0.0] * (self.rows * self.cols)
        else:
            self.data = list(data)
            need = self.rows * self.cols
            if len(self.data) < need:
                self.data.extend([0.0] * (need - len(self.data)))
            elif len(self.data) > need:
                self.data = self.data[:need]

    def index(self, r: int, c: int) -> int:
        return r * self.cols + c

    def get(self, r: int, c: int) -> float:
        return self.data[self.index(r, c)]

    def set(self, r: int, c: int, v: float) -> None:
        self.data[self.index(r, c)] = float(v)

    def to_json(self) -> Dict[str, Any]:
        return {"rows": self.rows, "cols": self.cols, "data": self.data}

    @classmethod
    def from_json(cls, value: Dict[str, Any]) -> "Matrix":
        return cls(int(value.get("rows", 0)), int(value.get("cols", 0)), list(value.get("data", [])))


class TransformerReservedArena:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._inited = False
        self._data = bytearray()

    def init(self, bytes_size: int) -> None:
        with self._lock:
            if self._inited:
                return
            size = int(max(8 * 1024 * 1024, min(1024 * 1024 * 1024, int(bytes_size))))
            self._data = bytearray(size)
            for i in range(0, len(self._data), 4096):
                self._data[i] = 0
            self._inited = True


class MatrixHotspotCache:
    @dataclass
    class Config:
        redisUrl: str = "redis://127.0.0.1:6379"
        redisDb: int = 1
        prefix: str = "AI079:attn"
        ttlSeconds: int = 900
        hotThreshold: int = 3
        maxElements: int = 200000

    @dataclass
    class _Entry:
        data: List[List[float]]
        lastMs: int
        hits: int

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cfg = MatrixHotspotCache.Config()
        self._local: Dict[str, MatrixHotspotCache._Entry] = {}

    def configure(self, cfg: "MatrixHotspotCache.Config") -> None:
        with self._lock:
            self._cfg = cfg

    def enabled(self) -> bool:
        return True

    def get(self, key: str, rows: int, cols: int, out: Optional[List[List[float]]] = None) -> bool:
        now = _now_ms()
        with self._lock:
            item = self._local.get(key)
            if item is None:
                return False
            if now - item.lastMs > self._cfg.ttlSeconds * 1000:
                return False
            if rows > 0 and len(item.data) != rows:
                return False
            if cols > 0 and item.data and len(item.data[0]) != cols:
                return False
            item.hits += 1
            if out is not None:
                out.clear()
                out.extend([row[:] for row in item.data])
            return True

    def record(self, key: str, rows: int, cols: int, costMs: int, data: List[List[float]]) -> None:
        if rows <= 0 or cols <= 0:
            return
        if rows * cols > self._cfg.maxElements:
            return
        now = _now_ms()
        with self._lock:
            old = self._local.get(key)
            hits = 1 if old is None else old.hits + 1
            self._local[key] = MatrixHotspotCache._Entry([row[:] for row in data], now, hits)
            if len(self._local) > 512:
                cutoff = now - self._cfg.ttlSeconds * 1000
                stale = [k for k, v in self._local.items() if v.lastMs < cutoff]
                for k in stale:
                    self._local.pop(k, None)
        _ = costMs


_arena_singleton = TransformerReservedArena()
_attention_cache = MatrixHotspotCache()


def transformerArena() -> TransformerReservedArena:
    return _arena_singleton


def attentionCache() -> MatrixHotspotCache:
    return _attention_cache


def ensureTransformerArena(bytes_size: int = 64 * 1024 * 1024) -> TransformerReservedArena:
    _arena_singleton.init(int(bytes_size))
    return _arena_singleton


def setAttentionCacheConfig(cfg: MatrixHotspotCache.Config) -> None:
    _attention_cache.configure(cfg)


def tfLogger(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return {"args": list(args), "kwargs": dict(kwargs), "ts": _now_ms()}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _rand_uniform(rng: random.Random, lo: float = -0.08, hi: float = 0.08) -> float:
    return float(rng.uniform(lo, hi))


def randUniform(rng: random.Random, lo: float = -0.08, hi: float = 0.08) -> float:
    return _rand_uniform(rng, lo, hi)


def _clip_grad(g: float, clip: float) -> float:
    if clip <= 0.0:
        return g
    if g > clip:
        return clip
    if g < -clip:
        return -clip
    return g


def clipGrad(g: float, clip: float) -> float:
    return _clip_grad(float(g), float(clip))


def _softmax_inplace(vec: List[float]) -> None:
    if not vec:
        return
    mx = max(vec)
    s = 0.0
    for i, v in enumerate(vec):
        e = math.exp(v - mx)
        vec[i] = e
        s += e
    if s <= 0.0:
        return
    inv = 1.0 / s
    for i in range(len(vec)):
        vec[i] *= inv


def softmaxInplace(vec: List[float]) -> None:
    _softmax_inplace(vec)


def softmax(vec: Sequence[float]) -> List[float]:
    out = list(vec)
    _softmax_inplace(out)
    return out


def _apply_topk_logit_mask(logits: List[float], top_k: int) -> None:
    if top_k <= 0 or top_k >= len(logits):
        return
    idx = list(range(len(logits)))
    idx.sort(key=lambda i: logits[i], reverse=True)
    keep = set(idx[:top_k])
    for i in range(len(logits)):
        if i not in keep:
            logits[i] = -1e9


def applyTopKLogitMask(logits: List[float], top_k: int) -> None:
    _apply_topk_logit_mask(logits, int(top_k))


def _gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def geluFn(x: float) -> float:
    return _gelu(float(x))


def geluGrad(x: float) -> float:
    x = float(x)
    c = math.sqrt(2.0 / math.pi)
    y = c * (x + 0.044715 * x * x * x)
    t = math.tanh(y)
    sech2 = 1.0 - t * t
    dy = c * (1.0 + 3.0 * 0.044715 * x * x)
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * dy


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoidf(x: float) -> float:
    return _sigmoid(float(x))


def _params_to_dict(params: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(params):
        return dataclasses.asdict(params)
    if isinstance(params, dict):
        return dict(params)
    out: Dict[str, Any] = {}
    for name in dir(params):
        if name.startswith("_"):
            continue
        value = getattr(params, name)
        if callable(value):
            continue
        out[name] = value
    return out


def _build_params_from(base: Any, patch: Dict[str, Any]) -> Any:
    if dataclasses.is_dataclass(base):
        return dataclasses.replace(base, **patch)
    cls = base.__class__
    try:
        obj = cls()
        for k, v in _params_to_dict(base).items():
            setattr(obj, k, v)
        for k, v in patch.items():
            setattr(obj, k, v)
        return obj
    except Exception:
        merged = _params_to_dict(base)
        merged.update(patch)
        return _FallbackTransformerParams(**{k: v for k, v in merged.items() if k in _FallbackTransformerParams.__annotations__})


def _get(params: Any, name: str, default: Any) -> Any:
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def _set(params: Any, name: str, value: Any) -> None:
    if isinstance(params, dict):
        params[name] = value
    else:
        setattr(params, name, value)


def _fnv_hash_u64(values: Iterable[int]) -> int:
    h = 1469598103934665603
    prime = 1099511628211
    for v in values:
        h ^= int(v) & 0xFFFFFFFFFFFFFFFF
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def hashKey(values: Iterable[int]) -> int:
    return _fnv_hash_u64(values)


def hashMatrixSample(mat: Sequence[Sequence[float]], sampleRows: int = 8, sampleCols: int = 4) -> int:
    rows = min(int(sampleRows), len(mat))
    vals: List[int] = []
    for r in range(rows):
        row = mat[r]
        cols = min(int(sampleCols), len(row))
        acc = len(row)
        for c in range(cols):
            acc += int(abs(float(row[c])) * 1000.0)
        vals.append(acc)
    return _fnv_hash_u64(vals)


def l2norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    return 1.0 if (not sa and not sb) else (len(sa & sb) / max(1, len(sa | sb)))


def nowMs() -> int:
    return _now_ms()


def isUtf8ContinuationByte(value: int) -> bool:
    v = int(value) & 0xFF
    return (v & 0xC0) == 0x80


def sanitizeUtf8Lossy(data: bytes | str) -> str:
    if isinstance(data, str):
        return data.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return bytes(data).decode("utf-8", errors="ignore")


def reverseTokensPreserveSpecial(tokens: Sequence[int], special: Optional[Iterable[int]] = None) -> List[int]:
    s = set(int(x) for x in (special or [0, 1, 2, 3]))
    normal = [int(t) for t in tokens if int(t) not in s]
    normal.reverse()
    out: List[int] = []
    j = 0
    for t in tokens:
        ti = int(t)
        if ti in s:
            out.append(ti)
        else:
            out.append(normal[j])
            j += 1
    return out


def normalizeGraphText(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").replace("\t", " ").split())


def isProgramSynthesisPrompt(text: str) -> bool:
    t = str(text).lower()
    keys = ("code", "function", "class", "algorithm", "python", "cpp", "c++")
    return any(k in t for k in keys)


class Linear:
    def __init__(self, inp: int = 0, out: int = 0) -> None:
        self.w = Matrix(out, inp)
        self.b = [0.0] * out
        rng = random.Random(0x1234567)
        for i in range(len(self.w.data)):
            self.w.data[i] = _rand_uniform(rng)
        self.mW = [0.0] * len(self.w.data)
        self.vW = [0.0] * len(self.w.data)
        self.gW = [0.0] * len(self.w.data)
        self.mB = [0.0] * len(self.b)
        self.vB = [0.0] * len(self.b)
        self.gB = [0.0] * len(self.b)
        self.step = 0

    def forward(self, x: Sequence[float]) -> List[float]:
        out = [0.0] * self.w.rows
        for r in range(self.w.rows):
            acc = self.b[r] if r < len(self.b) else 0.0
            base = r * self.w.cols
            for c in range(self.w.cols):
                xv = x[c] if c < len(x) else 0.0
                acc += self.w.data[base + c] * xv
            out[r] = acc
        return out

    def backward_accumulate(self, x: Sequence[float], grad_out: Sequence[float], p: Any) -> List[float]:
        grad_x = [0.0] * self.w.cols
        grad_clip = float(_get(p, "gradClip", 1.0))
        for r in range(self.w.rows):
            g = grad_out[r] if r < len(grad_out) else 0.0
            g = _clip_grad(g, grad_clip)
            if r < len(self.gB):
                self.gB[r] += g
            base = r * self.w.cols
            for c in range(self.w.cols):
                xv = x[c] if c < len(x) else 0.0
                gw = _clip_grad(g * xv, grad_clip)
                self.gW[base + c] += gw
                grad_x[c] += self.w.data[base + c] * g
        return grad_x

    def apply_accum(self, p: Any) -> None:
        accum = max(1, int(_get(p, "gradAccumSteps", 1)))
        beta1 = float(_get(p, "adamBeta1", 0.9))
        beta2 = float(_get(p, "adamBeta2", 0.999))
        eps = float(_get(p, "adamEps", 1e-8))
        lr = float(_get(p, "lr", 2e-4))
        wd = float(_get(p, "weightDecay", 0.0))
        self.step += 1
        t = max(1, self.step)
        for i in range(len(self.w.data)):
            g = self.gW[i] / accum
            if wd > 0.0:
                g += wd * self.w.data[i]
            self.mW[i] = beta1 * self.mW[i] + (1.0 - beta1) * g
            self.vW[i] = beta2 * self.vW[i] + (1.0 - beta2) * g * g
            m_hat = self.mW[i] / (1.0 - beta1 ** t)
            v_hat = self.vW[i] / (1.0 - beta2 ** t)
            self.w.data[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            self.gW[i] = 0.0
        for i in range(len(self.b)):
            g = self.gB[i] / accum
            if wd > 0.0:
                g += wd * self.b[i]
            self.mB[i] = beta1 * self.mB[i] + (1.0 - beta1) * g
            self.vB[i] = beta2 * self.vB[i] + (1.0 - beta2) * g * g
            m_hat = self.mB[i] / (1.0 - beta1 ** t)
            v_hat = self.vB[i] / (1.0 - beta2 ** t)
            self.b[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            self.gB[i] = 0.0

    def adamUpdate(self, p: Any) -> None:
        self.apply_accum(p)

    def linearApplyAccum(self, p: Any) -> None:
        self.apply_accum(p)

    def to_json(self) -> Dict[str, Any]:
        return {
            "w": self.w.to_json(),
            "b": self.b,
            "mW": self.mW,
            "vW": self.vW,
            "gW": self.gW,
            "mB": self.mB,
            "vB": self.vB,
            "gB": self.gB,
            "step": self.step,
        }

    @classmethod
    def from_json(cls, value: Dict[str, Any]) -> "Linear":
        w = Matrix.from_json(value.get("w", {}))
        out = cls(w.cols, w.rows)
        out.w = w
        out.b = list(value.get("b", out.b))
        out.mW = list(value.get("mW", out.mW))
        out.vW = list(value.get("vW", out.vW))
        out.gW = list(value.get("gW", out.gW))
        out.mB = list(value.get("mB", out.mB))
        out.vB = list(value.get("vB", out.vB))
        out.gB = list(value.get("gB", out.gB))
        out.step = int(value.get("step", 0))
        return out


class LayerNorm:
    def __init__(self, d: int = 0) -> None:
        self.gamma = [1.0] * d
        self.beta = [0.0] * d
        self.mGamma = [0.0] * d
        self.vGamma = [0.0] * d
        self.gGamma = [0.0] * d
        self.mBeta = [0.0] * d
        self.vBeta = [0.0] * d
        self.gBeta = [0.0] * d
        self.step = 0
        self.eps = 1e-5

    def forward(self, x: Sequence[float]) -> List[float]:
        n = max(1, len(x))
        mean = sum(x) / n
        var = sum((v - mean) ** 2 for v in x) / n
        inv = 1.0 / math.sqrt(var + self.eps)
        out = [0.0] * len(x)
        for i, xv in enumerate(x):
            g = self.gamma[i] if i < len(self.gamma) else 1.0
            b = self.beta[i] if i < len(self.beta) else 0.0
            out[i] = ((xv - mean) * inv) * g + b
        return out

    def forward_token(self, x: Sequence[float]) -> Tuple[List[float], float, float]:
        n = max(1, len(x))
        mean = sum(x) / n
        var = sum((v - mean) ** 2 for v in x) / n
        inv = 1.0 / math.sqrt(var + self.eps)
        out = [0.0] * len(x)
        for i, xv in enumerate(x):
            g = self.gamma[i] if i < len(self.gamma) else 1.0
            b = self.beta[i] if i < len(self.beta) else 0.0
            out[i] = ((xv - mean) * inv) * g + b
        return out, mean, inv

    def layerNormForwardToken(self, x: Sequence[float]) -> Tuple[List[float], float, float]:
        return self.forward_token(x)

    def layerNormForwardBatch(self, batch: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self.forward(x) for x in batch]

    def layerNormApplyAccum(self, p: Any) -> None:
        accum = max(1, int(_get(p, "gradAccumSteps", 1)))
        beta1 = float(_get(p, "adamBeta1", 0.9))
        beta2 = float(_get(p, "adamBeta2", 0.999))
        eps = float(_get(p, "adamEps", 1e-8))
        lr = float(_get(p, "lr", 2e-4))
        wd = float(_get(p, "weightDecay", 0.0))
        self.step += 1
        t = max(1, self.step)
        for i in range(len(self.gamma)):
            g = (self.gGamma[i] / accum) if i < len(self.gGamma) else 0.0
            if wd > 0.0:
                g += wd * self.gamma[i]
            self.mGamma[i] = beta1 * self.mGamma[i] + (1.0 - beta1) * g
            self.vGamma[i] = beta2 * self.vGamma[i] + (1.0 - beta2) * g * g
            m_hat = self.mGamma[i] / (1.0 - beta1 ** t)
            v_hat = self.vGamma[i] / (1.0 - beta2 ** t)
            self.gamma[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            self.gGamma[i] = 0.0
        for i in range(len(self.beta)):
            g = (self.gBeta[i] / accum) if i < len(self.gBeta) else 0.0
            if wd > 0.0:
                g += wd * self.beta[i]
            self.mBeta[i] = beta1 * self.mBeta[i] + (1.0 - beta1) * g
            self.vBeta[i] = beta2 * self.vBeta[i] + (1.0 - beta2) * g * g
            m_hat = self.mBeta[i] / (1.0 - beta1 ** t)
            v_hat = self.vBeta[i] / (1.0 - beta2 ** t)
            self.beta[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)
            self.gBeta[i] = 0.0

    def to_json(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "mGamma": self.mGamma,
            "vGamma": self.vGamma,
            "gGamma": self.gGamma,
            "mBeta": self.mBeta,
            "vBeta": self.vBeta,
            "gBeta": self.gBeta,
            "step": self.step,
            "eps": self.eps,
        }

    @classmethod
    def from_json(cls, value: Dict[str, Any]) -> "LayerNorm":
        d = len(list(value.get("gamma", [])))
        out = cls(d)
        out.gamma = list(value.get("gamma", out.gamma))
        out.beta = list(value.get("beta", out.beta))
        out.mGamma = list(value.get("mGamma", out.mGamma))
        out.vGamma = list(value.get("vGamma", out.vGamma))
        out.gGamma = list(value.get("gGamma", out.gGamma))
        out.mBeta = list(value.get("mBeta", out.mBeta))
        out.vBeta = list(value.get("vBeta", out.vBeta))
        out.gBeta = list(value.get("gBeta", out.gBeta))
        out.step = int(value.get("step", 0))
        out.eps = float(value.get("eps", 1e-5))
        return out


@dataclass
class KVCache:
    k: List[List[float]] = field(default_factory=list)
    v: List[List[float]] = field(default_factory=list)


class MultiHeadAttention:
    def __init__(self, d: int, h: int) -> None:
        self.nHeads = max(1, int(h))
        self.dModel = int(d)
        self.dHead = max(1, self.dModel // self.nHeads)
        self.wq = Linear(self.dModel, self.dModel)
        self.wk = Linear(self.dModel, self.dModel)
        self.wv = Linear(self.dModel, self.dModel)
        self.wo = Linear(self.dModel, self.dModel)
        self.useAlibi = False
        self.alibiSlope = 0.02
        self.useCoefAttention = False
        self.coefAttentionScale = 0.1
        self.useMLA = False
        self.mlaRank = 8
        self.mlaScale = 0.2

    def _alibi_bias(self, head: int, t: int, s: int, causal: bool) -> float:
        if self.alibiSlope <= 0.0:
            return 0.0
        scale = self.alibiSlope * (1.0 + head / max(1, self.nHeads - 1))
        dist = (t - s) if causal else abs(t - s)
        if dist < 0:
            dist = 0
        return -scale * dist

    def alibiBias(self, head: int, t: int, s: int, causal: bool) -> float:
        return self._alibi_bias(head, t, s, causal)

    def forward(
        self,
        x: List[List[float]],
        causal: bool,
        memory: Optional[List[List[float]]] = None,
        chunkSize: int = 0,
        topKMask: int = 0,
        useCache: bool = True,
    ) -> List[List[float]]:
        ksrc = memory if memory is not None else x
        if not x or not ksrc:
            return [[0.0] * self.dModel for _ in x]
        t_len = len(x)
        s_len = len(ksrc)
        cache_key = ""
        if useCache:
            hx = _fnv_hash_u64([len(row) + int(sum(abs(v) * 1000 for v in row[:4])) for row in x[:8]])
            hk = _fnv_hash_u64([len(row) + int(sum(abs(v) * 1000 for v in row[:4])) for row in ksrc[:8]])
            cache_key = f"attn|d={self.dModel}|h={self.nHeads}|T={t_len}|S={s_len}|c={1 if causal else 0}|x={hx}|k={hk}|topk={topKMask}|chunk={chunkSize}"
            hit: List[List[float]] = []
            if _attention_cache.get(cache_key, t_len, self.dModel, hit):
                return hit
        q_all = [self.wq.forward(v) for v in x]
        k_all = [self.wk.forward(v) for v in ksrc]
        v_all = [self.wv.forward(v) for v in ksrc]
        out = [[0.0] * self.dModel for _ in range(t_len)]
        inv = 1.0 / math.sqrt(max(1, self.dHead))
        for t in range(t_len):
            merged = [0.0] * self.dModel
            s_start, s_end = 0, s_len
            if chunkSize > 0:
                if causal and memory is None:
                    s_end = min(s_len, t + 1)
                    s_start = max(0, s_end - chunkSize)
                else:
                    s_start = max(0, s_len - chunkSize)
            for h in range(self.nHeads):
                base = h * self.dHead
                scores = [0.0] * (s_end - s_start)
                for si, s in enumerate(range(s_start, s_end)):
                    if causal and memory is None and s > t:
                        scores[si] = -1e9
                        continue
                    acc = 0.0
                    mla_rank = max(1, min(self.dHead, self.mlaRank))
                    for i in range(self.dHead):
                        w = 1.0 + (self.mlaScale if self.useMLA and i < mla_rank else 0.0)
                        acc += w * q_all[t][base + i] * k_all[s][base + i]
                    score = acc * inv
                    if self.useCoefAttention:
                        score *= 1.0 + self.coefAttentionScale
                    if self.useAlibi and memory is None:
                        score += self._alibi_bias(h, t, s, causal)
                    scores[si] = score
                if topKMask > 0:
                    _apply_topk_logit_mask(scores, topKMask)
                _softmax_inplace(scores)
                head = [0.0] * self.dHead
                for si, s in enumerate(range(s_start, s_end)):
                    p = scores[si]
                    kv = v_all[s]
                    for i in range(self.dHead):
                        head[i] += p * kv[base + i]
                for i in range(self.dHead):
                    merged[base + i] = head[i]
            out[t] = self.wo.forward(merged)
        if useCache:
            _attention_cache.record(cache_key, t_len, self.dModel, 0, out)
        return out

    def forwardStep(self, x: List[float], cache: KVCache, appendKV: bool, chunkSize: int = 0, topKMask: int = 0) -> List[float]:
        if appendKV:
            cache.k.append(self.wk.forward(x))
            cache.v.append(self.wv.forward(x))
        q = self.wq.forward(x)
        s_len = len(cache.k)
        if s_len == 0:
            return self.wo.forward([0.0] * self.dModel)
        s_start = max(0, s_len - chunkSize) if chunkSize > 0 and s_len > chunkSize else 0
        merged = [0.0] * self.dModel
        inv = 1.0 / math.sqrt(max(1, self.dHead))
        for h in range(self.nHeads):
            base = h * self.dHead
            scores = [0.0] * (s_len - s_start)
            for si, s in enumerate(range(s_start, s_len)):
                acc = 0.0
                mla_rank = max(1, min(self.dHead, self.mlaRank))
                for i in range(self.dHead):
                    w = 1.0 + (self.mlaScale if self.useMLA and i < mla_rank else 0.0)
                    acc += w * q[base + i] * cache.k[s][base + i]
                scores[si] = acc * inv
            if topKMask > 0:
                _apply_topk_logit_mask(scores, topKMask)
            _softmax_inplace(scores)
            head = [0.0] * self.dHead
            for si, s in enumerate(range(s_start, s_len)):
                p = scores[si]
                for i in range(self.dHead):
                    head[i] += p * cache.v[s][base + i]
            for i in range(self.dHead):
                merged[base + i] = head[i]
        return self.wo.forward(merged)


class FeedForward:
    def __init__(self, dModel: int, dFF: int) -> None:
        self.w1 = Linear(dModel, dFF)
        self.w2 = Linear(dFF, dModel)
        self.moeGate = Linear(dModel, 1)
        self.moeEnabled = False
        self.moeTopK = 2
        self.moeAuxWeight = 0.01
        self.moeW1: List[Linear] = []
        self.moeW2: List[Linear] = []
        self.expertUsageEma: List[float] = []

    def configure_moe(self, enabled: bool, experts: int, top_k: int, aux_weight: float) -> None:
        self.moeEnabled = bool(enabled)
        self.moeTopK = max(1, int(top_k))
        self.moeAuxWeight = max(0.0, float(aux_weight))
        if not self.moeEnabled:
            return
        experts = max(1, int(experts))
        if len(self.moeW1) != experts or len(self.moeW2) != experts:
            d_model = self.w1.w.cols
            d_ff = self.w1.w.rows
            self.moeW1 = [Linear(d_model, d_ff) for _ in range(experts)]
            self.moeW2 = [Linear(d_ff, d_model) for _ in range(experts)]
            self.moeGate = Linear(d_model, experts)
            self.expertUsageEma = [1.0 / experts] * experts

    def configureFeedForwardMoE(self, enabled: bool, experts: int, top_k: int, aux_weight: float) -> None:
        self.configure_moe(enabled, experts, top_k, aux_weight)

    def forward(self, x: List[float]) -> List[float]:
        if self.moeEnabled and self.moeW1 and self.moeW2:
            experts = min(len(self.moeW1), len(self.moeW2))
            top_k = max(1, min(experts, self.moeTopK))
            logits = self.moeGate.forward(x)
            if len(logits) != experts:
                logits = [1.0 / experts] * experts
            probs = logits[:]
            _softmax_inplace(probs)
            ids = sorted(range(experts), key=lambda i: probs[i], reverse=True)[:top_k]
            wsum = sum(probs[i] for i in ids) or 1.0
            out = [0.0] * len(x)
            for i in ids:
                w = probs[i] / wsum
                h = self.moeW1[i].forward(x)
                for j in range(len(h)):
                    h[j] = _gelu(h[j])
                y = self.moeW2[i].forward(h)
                for d in range(min(len(out), len(y))):
                    out[d] += w * y[d]
            return out
        h = self.w1.forward(x)
        for i in range(len(h)):
            h[i] = _gelu(h[i])
        return self.w2.forward(h)

    def ffnForward(self, x: List[float]) -> List[float]:
        return self.forward(x)


class EncoderLayer:
    def __init__(self, dModel: int, nHeads: int, dFF: int) -> None:
        self.selfAttn = MultiHeadAttention(dModel, nHeads)
        self.ffn = FeedForward(dModel, dFF)
        self.ln1 = LayerNorm(dModel)
        self.ln2 = LayerNorm(dModel)

    def forward(self, x: List[List[float]], chunkSize: int = 0, topKMask: int = 0) -> List[List[float]]:
        att = self.selfAttn.forward(x, False, None, chunkSize=chunkSize, topKMask=topKMask)
        y = [self.ln1.forward([a + b for a, b in zip(xi, ai)]) for xi, ai in zip(x, att)]
        out: List[List[float]] = []
        for yi in y:
            ff = self.ffn.forward(yi)
            out.append(self.ln2.forward([a + b for a, b in zip(yi, ff)]))
        return out


class DecoderLayer:
    def __init__(self, dModel: int, nHeads: int, dFF: int) -> None:
        self.selfAttn = MultiHeadAttention(dModel, nHeads)
        self.crossAttn = MultiHeadAttention(dModel, nHeads)
        self.ffn = FeedForward(dModel, dFF)
        self.ln1 = LayerNorm(dModel)
        self.ln2 = LayerNorm(dModel)
        self.ln3 = LayerNorm(dModel)

    def forward(self, x: List[List[float]], memory: List[List[float]], chunkSize: int = 0, topKMask: int = 0) -> List[List[float]]:
        att = self.selfAttn.forward(x, True, None, chunkSize=chunkSize, topKMask=topKMask)
        y = [self.ln1.forward([a + b for a, b in zip(xi, ai)]) for xi, ai in zip(x, att)]
        cross = self.crossAttn.forward(y, False, memory, chunkSize=chunkSize, topKMask=topKMask)
        z = [self.ln2.forward([a + b for a, b in zip(yi, ci)]) for yi, ci in zip(y, cross)]
        out: List[List[float]] = []
        for zi in z:
            ff = self.ffn.forward(zi)
            out.append(self.ln3.forward([a + b for a, b in zip(zi, ff)]))
        return out


class Tokenizer(_BaseTokenizer):
    def __init__(self, vocab_size: int, mode: str = "bpe") -> None:
        super().__init__(int(vocab_size))
        self.vocab_size = int(vocab_size)
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.pad_id = 0
        self._lock = threading.RLock()
        self._mode = "bpe"
        self._corpus: List[str] = []
        self._corpus_chars = 0
        self._corpus_max_chars = 1_000_000
        self._corpus_min_chars = 256
        self._bpe_ready = False
        self._bpe_id_to_token: List[str] = []
        self._bpe_token_to_id: Dict[str, int] = {}
        self._bpe_merges: List[Dict[str, int]] = []
        self._bpe_max_merges = max(1024, self.vocab_size - 4)
        self.setMode(mode)

    def mode(self) -> str:
        return self._mode

    def mode_(self) -> str:
        return self._mode

    def setMode(self, mode: str) -> None:
        m = str(mode or "bpe").strip().lower()
        if m in {"byte", "bpe", "hash"}:
            self._mode = "bpe" if m == "hash" else m
        else:
            self._mode = "bpe"
        if self._mode != "bpe":
            return
        with self._lock:
            self._bpe_ready = False
            self._bpe_id_to_token.clear()
            self._bpe_token_to_id.clear()
            self._bpe_merges.clear()

    def normalizeText(self, text: str) -> str:
        out: List[str] = []
        last_space = False
        for c in str(text):
            oc = ord(c)
            if oc < 0x20 or oc == 0x7F:
                if not last_space:
                    out.append(" ")
                    last_space = True
                continue
            ch = c.lower()
            if ch.isspace():
                if not last_space:
                    out.append(" ")
                    last_space = True
                continue
            word = ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "'"
            if not word:
                if not last_space:
                    out.append(" ")
                    last_space = True
                continue
            out.append(ch)
            last_space = False
        while out and out[-1] == " ":
            out.pop()
        return "".join(out)

    def observe(self, text: str) -> None:
        if self._mode != "bpe":
            return
        norm = self.normalizeText(text)
        if len(norm) > 4096:
            norm = norm[:4096]
        if not norm:
            return
        with self._lock:
            while self._corpus and self._corpus_chars + len(norm) > self._corpus_max_chars:
                dropped = self._corpus.pop(0)
                self._corpus_chars -= len(dropped)
            self._corpus.append(norm)
            self._corpus_chars += len(norm)
            self._bpe_ready = False

    def ensureBpeReady(self) -> None:
        if self._mode != "bpe":
            return
        with self._lock:
            if self._bpe_ready:
                return
            self.trainBpe()
            self._bpe_ready = True

    def trainBpe(self) -> None:
        with self._lock:
            self._bpe_id_to_token.clear()
            self._bpe_token_to_id.clear()
            self._bpe_merges.clear()
            if self._corpus_chars < self._corpus_min_chars:
                return
            freq: Dict[str, int] = {}
            for line in self._corpus:
                for w in line.split():
                    if len(w) == 1 and w not in {"a", "i"}:
                        continue
                    freq[w] = freq.get(w, 0) + 1
            if not freq:
                return
            items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
            self._bpe_token_to_id["unk"] = 0
            self._bpe_id_to_token.append("unk")
            cap = max(1, min(self.vocab_size - 4, self._bpe_max_merges))
            for w, _ in items:
                if len(self._bpe_id_to_token) >= cap:
                    break
                if w in self._bpe_token_to_id:
                    continue
                tid = len(self._bpe_id_to_token)
                self._bpe_token_to_id[w] = tid
                self._bpe_id_to_token.append(w)

    def _encode_bpe(self, text: str) -> List[int]:
        norm = self.normalizeText(text)
        return [self._bpe_token_to_id.get(w, 0) for w in norm.split()] if norm else []

    def encodeBpe(self, text: str) -> List[int]:
        self.ensureBpeReady()
        with self._lock:
            return self._encode_bpe(text)

    def encode(self, text: str, max_len: int, add_bos_eos: bool = True) -> List[int]:
        max_len = max(1, int(max_len))
        ids: List[int] = [self.bos_id] if add_bos_eos else []
        if self._mode == "byte":
            for c in str(text).encode("utf-8", errors="ignore"):
                tid = 4 + int(c)
                if tid >= self.vocab_size:
                    tid = 4 + (tid % max(1, self.vocab_size - 4))
                ids.append(tid)
                if len(ids) >= max_len:
                    break
            if add_bos_eos and len(ids) < max_len:
                ids.append(self.eos_id)
            return ids[:max_len]
        if self._mode == "bpe":
            self.observe(text)
            self.ensureBpeReady()
            with self._lock:
                for token in self._encode_bpe(text):
                    tid = 4 + token
                    if tid >= self.vocab_size:
                        tid = self.unk_id
                    ids.append(tid)
                    if len(ids) >= max_len:
                        break
            if add_bos_eos and len(ids) < max_len:
                ids.append(self.eos_id)
            return ids[:max_len]
        cur: List[str] = []
        for ch in str(text):
            if ch.isspace():
                if cur:
                    ids.append(4 + (hash("".join(cur)) % max(1, self.vocab_size - 4)))
                    cur.clear()
                    if len(ids) >= max_len:
                        break
            else:
                cur.append(ch)
        if cur and len(ids) < max_len:
            ids.append(4 + (hash("".join(cur)) % max(1, self.vocab_size - 4)))
        if add_bos_eos and len(ids) < max_len:
            ids.append(self.eos_id)
        return ids[:max_len]

    def decode(self, ids: Sequence[int]) -> str:
        parts: List[str] = []
        if self._mode == "bpe":
            self.ensureBpeReady()
        with self._lock:
            for tid in ids:
                if tid <= 3:
                    continue
                if self._mode == "byte":
                    b = tid - 4
                    if 0 <= b < 256:
                        parts.append(bytes([b]).decode("latin1"))
                    continue
                if self._mode == "bpe":
                    t = tid - 4
                    parts.append(self._bpe_id_to_token[t] if 0 <= t < len(self._bpe_id_to_token) else "unk")
                else:
                    parts.append(f"tok{tid}")
        raw = " ".join(parts)
        clean: List[str] = []
        last_space = True
        for c in raw:
            keep = c if c.isalnum() or c in {"'", "-"} else " "
            if keep == " ":
                if not last_space:
                    clean.append(" ")
                last_space = True
            else:
                clean.append(keep)
                last_space = False
        while clean and clean[-1] == " ":
            clean.pop()
        return "".join(clean)

    def activeVocabLimit(self) -> int:
        if self._mode == "byte":
            return max(8, min(self.vocab_size, 260))
        if self._mode == "bpe":
            self.ensureBpeReady()
            with self._lock:
                return min(self.vocab_size, max(8, 4 + len(self._bpe_id_to_token)))
        return max(8, self.vocab_size)

    def toJson(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "vocabSize": self.vocab_size,
                "mode": self._mode,
                "bpeReady": self._bpe_ready,
                "corpus": list(self._corpus),
                "corpusChars": self._corpus_chars,
                "corpusMaxChars": self._corpus_max_chars,
                "corpusMinChars": self._corpus_min_chars,
                "bpeIdToTokenHex": [t.encode("utf-8", errors="ignore").hex() for t in self._bpe_id_to_token],
                "bpeMerges": list(self._bpe_merges),
                "bpeMaxMerges": self._bpe_max_merges,
            }

    def fromJson(self, state: Dict[str, Any]) -> bool:
        try:
            if not isinstance(state, dict):
                return False
            with self._lock:
                self.vocab_size = max(16, int(state.get("vocabSize", self.vocab_size)))
                self._mode = str(state.get("mode", self._mode))
                self._bpe_ready = bool(state.get("bpeReady", False))
                self._corpus = list(state.get("corpus", []))
                self._corpus_chars = int(state.get("corpusChars", 0))
                self._corpus_max_chars = int(state.get("corpusMaxChars", self._corpus_max_chars))
                self._corpus_min_chars = int(state.get("corpusMinChars", self._corpus_min_chars))
                self._bpe_max_merges = int(state.get("bpeMaxMerges", self._bpe_max_merges))
                self._bpe_id_to_token.clear()
                for hx in state.get("bpeIdToTokenHex", []):
                    try:
                        self._bpe_id_to_token.append(bytes.fromhex(str(hx)).decode("utf-8", errors="ignore"))
                    except Exception:
                        self._bpe_id_to_token.append("")
                if not self._bpe_id_to_token and isinstance(state.get("bpeIdToToken"), list):
                    self._bpe_id_to_token = list(state.get("bpeIdToToken", []))
                self._bpe_token_to_id = {t: i for i, t in enumerate(self._bpe_id_to_token)}
                merges = state.get("bpeMerges", [])
                self._bpe_merges = [dict(m) for m in merges] if isinstance(merges, list) else []
                if self._mode != "bpe":
                    self._bpe_ready = False
                    self._bpe_id_to_token.clear()
                    self._bpe_token_to_id.clear()
                    self._bpe_merges.clear()
            return True
        except Exception:
            return False


class TransformerModel(_BaseModel):
    def __init__(self, params: Any) -> None:
        super().__init__(params)
        self.params = params
        self._train_step = 0
        self._loss_ema = 0.0
        self._tokenizer = Tokenizer(int(_get(params, "vocabSize", 8192)), str(_get(params, "tokenizerMode", "bpe")))
        d_model = int(_get(params, "dModel", 256))
        vocab = int(_get(params, "vocabSize", 8192))
        max_len = int(_get(params, "maxLen", 256))
        n_layers = int(_get(params, "nLayers", 4))
        n_heads = int(_get(params, "nHeads", 8))
        d_ff = int(_get(params, "dFF", 512))
        self.tokEmbed = Matrix(vocab, d_model)
        self.posEmbed = Matrix(max_len, d_model)
        self.outProj = Linear(d_model, vocab)
        self.fuseAttn = MultiHeadAttention(d_model, n_heads)
        self.fuseGate = Linear(d_model * 2, d_model)
        self.encLayers = [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.decLayers = [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        rng = random.Random(0x42)
        for i in range(len(self.tokEmbed.data)):
            self.tokEmbed.data[i] = _rand_uniform(rng)
        for i in range(len(self.posEmbed.data)):
            self.posEmbed.data[i] = _rand_uniform(rng)
        self._sync_component_params()

    def _sync_component_params(self) -> None:
        def apply(attn: MultiHeadAttention) -> None:
            attn.useAlibi = bool(_get(self.params, "useAlibi", False))
            attn.alibiSlope = float(_get(self.params, "alibiSlope", 0.02))
            attn.useCoefAttention = bool(_get(self.params, "enableCoefAttention", False))
            attn.coefAttentionScale = float(_get(self.params, "coefAttentionScale", 0.1))
            attn.useMLA = bool(_get(self.params, "enableMLA", False))
            attn.mlaRank = int(_get(self.params, "mlaRank", 8))
            attn.mlaScale = float(_get(self.params, "mlaScale", 0.2))

        apply(self.fuseAttn)
        for layer in self.encLayers:
            apply(layer.selfAttn)
            layer.ffn.configure_moe(bool(_get(self.params, "enableMoE", False)), int(_get(self.params, "moeExperts", 4)), int(_get(self.params, "moeTopK", 2)), float(_get(self.params, "moeAuxWeight", 0.01)))
        for layer in self.decLayers:
            apply(layer.selfAttn)
            apply(layer.crossAttn)
            layer.ffn.configure_moe(bool(_get(self.params, "enableMoE", False)), int(_get(self.params, "moeExperts", 4)), int(_get(self.params, "moeTopK", 2)), float(_get(self.params, "moeAuxWeight", 0.01)))

    def updateParams(self, p: Any) -> None:
        self.params = _build_params_from(self.params, _params_to_dict(p))
        self._tokenizer.setMode(str(_get(self.params, "tokenizerMode", self._tokenizer.mode())))
        self._sync_component_params()

    def tokenizer_(self) -> Tokenizer:
        return self._tokenizer

    def applyParams(self, p: Any) -> None:
        self.updateParams(p)

    def _apply_rope(self, vec: List[float], pos: int) -> None:
        if len(vec) < 2 or pos <= 0:
            return
        base = 10000.0
        d_model = float(len(vec))
        for i in range(0, len(vec) - 1, 2):
            freq = base ** (-(i / d_model))
            ang = pos * freq
            c = math.cos(ang)
            s = math.sin(ang)
            x0, x1 = vec[i], vec[i + 1]
            vec[i] = x0 * c - x1 * s
            vec[i + 1] = x0 * s + x1 * c

    def applyRoPEToEmbedding(self, vec: List[float], pos: int) -> None:
        self._apply_rope(vec, pos)

    def embedTokens(self, tokens: Sequence[int]) -> List[List[float]]:
        d_model = int(_get(self.params, "dModel", 256))
        stride = max(1, int(_get(self.params, "hierStride", 8)))
        hier = bool(_get(self.params, "enableHierEmbedding", False))
        vocab = int(_get(self.params, "vocabSize", 8192))
        out: List[List[float]] = []
        for t, raw_id in enumerate(tokens):
            token_id = max(0, min(vocab - 1, int(raw_id)))
            seg_start = (t // stride) * stride
            seg_id = max(0, min(vocab - 1, int(tokens[seg_start] if tokens else 0)))
            vec = [0.0] * d_model
            for d in range(d_model):
                base = self.tokEmbed.get(token_id, d)
                h = 0.12 * self.tokEmbed.get(seg_id, d) if hier else 0.0
                pos = self.posEmbed.get(min(t, self.posEmbed.rows - 1), d)
                vec[d] = base + h + 0.06 * pos
            self._apply_rope(vec, t)
            out.append(vec)
        return out

    def encode(self, tokens: Sequence[int]) -> List[List[float]]:
        x = self.embedTokens(tokens)
        chunk = int(_get(self.params, "attnChunkSize", 0))
        topk = int(_get(self.params, "topK", 0))
        for layer in self.encLayers:
            x = layer.forward(x, chunkSize=chunk, topKMask=topk)
        return x

    def decode(self, tokens: Sequence[int], memory: Optional[List[List[float]]] = None) -> List[List[float]]:
        x = self.embedTokens(tokens)
        mem = memory if memory is not None else self.encode(tokens)
        chunk = int(_get(self.params, "attnChunkSize", 0))
        topk = int(_get(self.params, "topK", 0))
        for layer in self.decLayers:
            x = layer.forward(x, mem, chunkSize=chunk, topKMask=topk)
        return x

    def logitsAt(self, hidden: Sequence[float]) -> List[float]:
        return self.outProj.forward(list(hidden))

    def _avg_norm(self, mat: List[List[float]]) -> float:
        if not mat:
            return 0.0
        return sum(math.sqrt(sum(v * v for v in row)) for row in mat) / max(1, len(mat))

    def avgNorm(self, mat: List[List[float]]) -> float:
        return self._avg_norm(mat)

    def fuseMemory(self, textMem: List[List[float]], graphMem: List[List[float]], graphWeight: float) -> List[List[float]]:
        if not textMem or not graphMem or graphWeight <= 0.0:
            return [row[:] for row in textMem]
        text_norm = self._avg_norm(textMem)
        graph_norm = self._avg_norm(graphMem)
        ratio = graph_norm / max(1e-6, text_norm)
        dyn_weight = graphWeight * max(0.5, min(2.0, ratio))
        ctx = self.fuseAttn.forward(textMem, False, graphMem, chunkSize=int(_get(self.params, "attnChunkSize", 0)))
        out = [row[:] for row in textMem]
        for i in range(min(len(out), len(ctx))):
            gate = self.fuseGate.forward(textMem[i] + ctx[i])
            for d in range(min(len(out[i]), len(ctx[i]), len(gate))):
                out[i][d] += dyn_weight * _sigmoid(gate[d]) * ctx[i][d]
            norm = math.sqrt(sum(v * v for v in out[i]))
            if norm > 0.0:
                out[i] = [v / norm for v in out[i]]
        return out

    def _apply_self_flow(self, mem: List[List[float]], weight: float) -> List[List[float]]:
        if not mem or weight <= 0.0:
            return mem
        out = [row[:] for row in mem]
        for i in range(1, len(out)):
            for d in range(min(len(out[i]), len(out[i - 1]))):
                out[i][d] += weight * out[i - 1][d]
        return out

    def applySelfFlow(self, mem: List[List[float]], weight: float) -> List[List[float]]:
        return self._apply_self_flow(mem, weight)

    def applyTopKLogitMask(self, logits: List[float], top_k: int) -> None:
        _apply_topk_logit_mask(logits, int(top_k))

    def scheduledLr(self) -> float:
        base_lr = float(_get(self.params, "lr", 2e-4))
        if not bool(_get(self.params, "useLrSchedule", True)):
            return base_lr
        step = max(1, int(self._train_step))
        warmup = max(1, int(_get(self.params, "lrWarmupSteps", 32)))
        cosine_steps = max(warmup + 1, int(_get(self.params, "lrCosineSteps", 400)))
        min_ratio = max(0.0, min(1.0, float(_get(self.params, "lrMinRatio", 0.1))))
        if step <= warmup:
            return base_lr * (step / warmup)
        if step >= cosine_steps:
            return base_lr * min_ratio
        progress = (step - warmup) / max(1, (cosine_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)

    def sampleLoss(self, inputTokens: Sequence[int], targetTokens: Sequence[int], graphTokens: Optional[Sequence[int]] = None) -> float:
        mem = self.encode(inputTokens)
        if graphTokens:
            gmem = self.encode(graphTokens)
            mem = self.fuseMemory(mem, gmem, float(_get(self.params, "graphWeight", 0.8)))
        dec_in = [1]
        losses: List[float] = []
        for tgt in targetTokens:
            hidden_seq = self.decode(dec_in, mem)
            if not hidden_seq:
                break
            logits = self.logitsAt(hidden_seq[-1])
            if not logits:
                continue
            target = max(0, min(len(logits) - 1, int(tgt)))
            probs = logits[:]
            _softmax_inplace(probs)
            losses.append(-math.log(max(1e-9, probs[target])))
            dec_in.append(target)
            if len(dec_in) >= int(_get(self.params, "maxLen", 256)):
                break
        return float(sum(losses) / max(1, len(losses)))

    def computeContrastiveLoss(self, a: Sequence[int], b: Sequence[int], c: Sequence[int], margin: float = 0.2) -> float:
        la = self.sampleLoss(a, b)
        lb = self.sampleLoss(a, c)
        return max(0.0, float(margin) + la - lb)

    def jointTrain(self, samples: Sequence[TrainSample], epochs: int = 1, lr: Optional[float] = None) -> Dict[str, Any]:
        losses: List[float] = []
        tok = self._tokenizer
        for _ in range(max(1, int(epochs))):
            for s in samples:
                inp = tok.encode(str(getattr(s, "input", "")), int(_get(self.params, "maxLen", 256)), True)
                tgt = tok.encode(str(getattr(s, "target", "")), int(_get(self.params, "maxLen", 256)), True)
                graph = tok.encode(str(getattr(s, "graph", "")), int(_get(self.params, "maxLen", 256)), True)
                use_lr = float(lr) if lr is not None else self.scheduledLr()
                losses.append(self.trainOnSample(inp, tgt, graph, use_lr))
        return {"ok": True, "meanLoss": (sum(losses) / max(1, len(losses))), "samples": len(samples), "epochs": max(1, int(epochs))}

    def pretrain(self, samples: Sequence[TrainSample], epochs: int = 1) -> Dict[str, Any]:
        return self.jointTrain(samples, epochs=epochs, lr=self.scheduledLr())

    def rlhfStats(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "enabled": bool(_get(self.params, "enableRLHF", False)),
            "trainStep": self._train_step,
            "lossEma": self._loss_ema,
            "rlhfLr": float(_get(self.params, "rlhfLr", 1e-4)),
            "replaySize": int(_get(self.params, "rlhfReplaySize", 256)),
        }

    def _apply_no_repeat_ngram(self, logits: List[float], out: Sequence[int], n: int) -> None:
        if len(out) < n - 1:
            return
        prefix = list(out[-(n - 1):])
        banned: set[int] = set()
        for i in range(0, len(out) - n + 1):
            if list(out[i:i + n - 1]) == prefix:
                banned.add(int(out[i + n - 1]))
        for tid in banned:
            if 0 <= tid < len(logits):
                logits[tid] = -1e9

    def _apply_repetition_penalty(self, logits: List[float], out: Sequence[int]) -> None:
        segmented = bool(_get(self.params, "repetitionSegmented", True))
        if segmented:
            recent_w = max(1, int(_get(self.params, "repetitionWindowRecent", 24)))
            mid_w = max(recent_w + 1, int(_get(self.params, "repetitionWindowMid", 96)))
            pen_recent = max(1.0, float(_get(self.params, "repetitionPenaltyRecent", 1.18)))
            pen_mid = max(1.0, float(_get(self.params, "repetitionPenaltyMid", 1.12)))
            pen_old = max(1.0, float(_get(self.params, "repetitionPenaltyOld", 1.05)))
            seen: set[int] = set()
            for i in range(len(out) - 1, -1, -1):
                tid = int(out[i])
                if tid in seen:
                    continue
                seen.add(tid)
                dist = len(out) - 1 - i
                penalty = pen_recent if dist <= recent_w else (pen_mid if dist <= mid_w else pen_old)
                if 0 <= tid < len(logits):
                    logits[tid] /= penalty
        else:
            penalty = max(1.0, float(_get(self.params, "repetitionPenalty", 1.0)))
            if penalty > 1.0:
                for tid in out:
                    if 0 <= tid < len(logits):
                        logits[int(tid)] /= penalty

    def generate(
        self,
        inputTokens: Sequence[int],
        graphTokens: Optional[Sequence[int]] = None,
        graphEmbeddings: Optional[List[List[float]]] = None,
        maxTokens: int = 128,
        graphWeight: Optional[float] = None,
        vocabLimit: Optional[int] = None,
        temperature: Optional[float] = None,
        topK: Optional[int] = None,
        topP: Optional[float] = None,
    ) -> List[int]:
        max_tokens = max(1, int(maxTokens))
        clip = int(_get(self.params, "windowClipTokens", 0))
        if clip > 0:
            max_tokens = min(max_tokens, clip)
        sampled_vocab = int(_get(self.params, "vocabSize", 8192))
        if vocabLimit is not None and int(vocabLimit) >= 8:
            sampled_vocab = max(8, min(sampled_vocab, int(vocabLimit)))
        mem = self.encode(inputTokens)
        g_weight = float(_get(self.params, "graphWeight", 0.8) if graphWeight is None else graphWeight)
        gmem: List[List[float]] = []
        if graphEmbeddings:
            d_model = int(_get(self.params, "dModel", 256))
            for vec in graphEmbeddings:
                v = [0.0] * d_model
                for i in range(min(d_model, len(vec))):
                    v[i] = vec[i]
                norm = math.sqrt(sum(x * x for x in v))
                if norm > 0.0:
                    v = [x / norm for x in v]
                gmem.append(v)
        elif graphTokens:
            gmem = self.encode(graphTokens)
        if gmem:
            mem = self.fuseMemory(mem, gmem, g_weight)
            mem.extend([[x * g_weight for x in row] for row in gmem])
        mem = self._apply_self_flow(mem, float(_get(self.params, "selfFlowWeight", 0.0)))

        active_layers = len(self.decLayers)
        if bool(_get(self.params, "enableDynamicReasoning", False)):
            complexity = len(inputTokens) + len(graphTokens or [])
            if complexity <= max(2, int(_get(self.params, "simpleTokenThreshold", 16))) + 4:
                active_layers = max(1, active_layers // 2)

        self_caches = [KVCache() for _ in range(active_layers)]
        cross_caches = [KVCache() for _ in range(active_layers)]
        for li in range(active_layers):
            layer = self.decLayers[li]
            cross_caches[li].k = [layer.crossAttn.wk.forward(m) for m in mem]
            cross_caches[li].v = [layer.crossAttn.wv.forward(m) for m in mem]

        out = [1]
        temp = float(_get(self.params, "temperature", 1.0) if temperature is None else temperature)
        top_k = int(_get(self.params, "topK", 20) if topK is None else topK)
        top_p = float(_get(self.params, "topP", 0.95) if topP is None else topP)
        draft = max(1, int(_get(self.params, "draftTokens", 1)))

        for _ in range(max_tokens):
            for _ in range(draft):
                pos = len(out) - 1
                last_id = max(0, min(int(_get(self.params, "vocabSize", 8192)) - 1, out[-1]))
                x = [self.tokEmbed.get(last_id, d) for d in range(int(_get(self.params, "dModel", 256)))]
                self._apply_rope(x, pos)
                for li in range(active_layers):
                    layer = self.decLayers[li]
                    self_out = layer.selfAttn.forwardStep(x, self_caches[li], True, int(_get(self.params, "attnChunkSize", 0)), max(0, top_k))
                    y, _, _ = layer.ln1.forward_token([a + b for a, b in zip(x, self_out)])
                    cross_out = layer.crossAttn.forwardStep(y, cross_caches[li], False, int(_get(self.params, "attnChunkSize", 0)), max(0, top_k))
                    z, _, _ = layer.ln2.forward_token([a + b for a, b in zip(y, cross_out)])
                    ff = layer.ffn.forward(z)
                    x, _, _ = layer.ln3.forward_token([a + b for a, b in zip(z, ff)])

                logits = self.logitsAt(x)
                for i in range(sampled_vocab, len(logits)):
                    logits[i] = -1e9
                if len(logits) > 0:
                    logits[0] = -1e9
                if len(logits) > 1:
                    logits[1] = -1e9
                if len(logits) > 3:
                    logits[3] = -1e9
                self._apply_repetition_penalty(logits, out)
                ngram = int(_get(self.params, "noRepeatNgram", 3))
                if ngram >= 2:
                    self._apply_no_repeat_ngram(logits, out, ngram)
                real_temp = max(0.1, temp)
                for i in range(len(logits)):
                    logits[i] /= real_temp
                if top_k > 0:
                    _apply_topk_logit_mask(logits, top_k)
                deterministic = (real_temp <= 0.20) and (top_p <= 0.0) and (top_k <= 1)
                if deterministic:
                    next_id = max(range(len(logits)), key=lambda i: logits[i]) if logits else 2
                else:
                    probs = logits[:]
                    _softmax_inplace(probs)
                    if 0.0 < top_p < 1.0:
                        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                        keep = [0.0] * len(probs)
                        cum = 0.0
                        for idx in order:
                            cum += probs[idx]
                            keep[idx] = probs[idx]
                            if cum >= top_p:
                                break
                        probs = keep
                        _softmax_inplace(probs)
                    next_id = random.choices(range(len(probs)), weights=probs, k=1)[0] if sum(probs) > 0 else (max(range(len(logits)), key=lambda i: logits[i]) if logits else 2)
                out.append(next_id)
                if next_id == 2 and len(out) - 1 >= max(0, int(_get(self.params, "minNewTokens", 0))):
                    return out
                if len(out) - 1 >= max_tokens:
                    return out
        return out

    def trainOnSample(self, inputTokens: Sequence[int], targetTokens: Sequence[int], graphTokens: Optional[Sequence[int]] = None, lr: float = 0.0) -> float:
        if lr > 0.0:
            _set(self.params, "lr", float(lr))
        self._train_step += 1
        if not targetTokens:
            return 0.0
        mem = self.encode(inputTokens)
        if graphTokens:
            gmem = self.encode(graphTokens)
            mem = self.fuseMemory(mem, gmem, float(_get(self.params, "graphWeight", 0.8)))
        dec_in = [1]
        losses: List[float] = []
        hiddens: List[List[float]] = []
        logits_seq: List[List[float]] = []
        for tgt in targetTokens:
            hidden_seq = self.decode(dec_in, mem)
            if not hidden_seq:
                break
            h = hidden_seq[-1]
            hiddens.append(h)
            logits = self.logitsAt(h)
            logits_seq.append(logits)
            dec_in.append(int(tgt))
            if len(dec_in) >= int(_get(self.params, "maxLen", 256)):
                break
        for i, logits in enumerate(logits_seq):
            target = max(0, min(len(logits) - 1, int(targetTokens[i]) if i < len(targetTokens) else 2)) if logits else 0
            probs = logits[:]
            _softmax_inplace(probs)
            p = max(1e-9, probs[target] if probs else 1e-9)
            losses.append(-math.log(p))
            grad = probs
            if grad:
                grad[target] -= 1.0
                self.outProj.backward_accumulate(hiddens[i], grad, self.params)
        self.outProj.apply_accum(self.params)
        loss = sum(losses) / max(1, len(losses))
        alpha = float(_get(self.params, "lossNoiseAlpha", 0.0))
        if alpha > 0.0:
            self._loss_ema = loss if self._loss_ema == 0.0 else alpha * loss + (1.0 - alpha) * self._loss_ema
            loss = self._loss_ema
        return float(loss)

    def toJson(self) -> Dict[str, Any]:
        return _params_to_dict(self.params)

    def _attn_to_json(self, attn: MultiHeadAttention) -> Dict[str, Any]:
        return {
            "nHeads": attn.nHeads,
            "dModel": attn.dModel,
            "dHead": attn.dHead,
            "useAlibi": attn.useAlibi,
            "alibiSlope": attn.alibiSlope,
            "useCoefAttention": attn.useCoefAttention,
            "coefAttentionScale": attn.coefAttentionScale,
            "useMLA": attn.useMLA,
            "mlaRank": attn.mlaRank,
            "mlaScale": attn.mlaScale,
            "wq": attn.wq.to_json(),
            "wk": attn.wk.to_json(),
            "wv": attn.wv.to_json(),
            "wo": attn.wo.to_json(),
        }

    def _ffn_to_json(self, ffn: FeedForward) -> Dict[str, Any]:
        return {
            "w1": ffn.w1.to_json(),
            "w2": ffn.w2.to_json(),
            "moeEnabled": ffn.moeEnabled,
            "moeTopK": ffn.moeTopK,
            "moeAuxWeight": ffn.moeAuxWeight,
            "moeGate": ffn.moeGate.to_json(),
            "expertUsageEma": ffn.expertUsageEma,
            "moeW1": [x.to_json() for x in ffn.moeW1],
            "moeW2": [x.to_json() for x in ffn.moeW2],
        }

    def stateDict(self) -> Dict[str, Any]:
        return {
            "params": self.toJson(),
            "tokenizer": self._tokenizer.toJson(),
            "tokEmbed": self.tokEmbed.to_json(),
            "posEmbed": self.posEmbed.to_json(),
            "outProj": self.outProj.to_json(),
            "fuseAttn": self._attn_to_json(self.fuseAttn),
            "fuseGate": self.fuseGate.to_json(),
            "encLayers": [{"selfAttn": self._attn_to_json(l.selfAttn), "ffn": self._ffn_to_json(l.ffn), "ln1": l.ln1.to_json(), "ln2": l.ln2.to_json()} for l in self.encLayers],
            "decLayers": [{"selfAttn": self._attn_to_json(l.selfAttn), "crossAttn": self._attn_to_json(l.crossAttn), "ffn": self._ffn_to_json(l.ffn), "ln1": l.ln1.to_json(), "ln2": l.ln2.to_json(), "ln3": l.ln3.to_json()} for l in self.decLayers],
            "trainStep": self._train_step,
            "lossEma": self._loss_ema,
        }

    def _load_attn_json(self, attn: MultiHeadAttention, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            return
        attn.nHeads = int(value.get("nHeads", attn.nHeads))
        attn.dModel = int(value.get("dModel", attn.dModel))
        attn.dHead = int(value.get("dHead", max(1, attn.dModel // max(1, attn.nHeads))))
        attn.useAlibi = bool(value.get("useAlibi", attn.useAlibi))
        attn.alibiSlope = float(value.get("alibiSlope", attn.alibiSlope))
        attn.useCoefAttention = bool(value.get("useCoefAttention", attn.useCoefAttention))
        attn.coefAttentionScale = float(value.get("coefAttentionScale", attn.coefAttentionScale))
        attn.useMLA = bool(value.get("useMLA", attn.useMLA))
        attn.mlaRank = int(value.get("mlaRank", attn.mlaRank))
        attn.mlaScale = float(value.get("mlaScale", attn.mlaScale))
        if isinstance(value.get("wq"), dict):
            attn.wq = Linear.from_json(value["wq"])
        if isinstance(value.get("wk"), dict):
            attn.wk = Linear.from_json(value["wk"])
        if isinstance(value.get("wv"), dict):
            attn.wv = Linear.from_json(value["wv"])
        if isinstance(value.get("wo"), dict):
            attn.wo = Linear.from_json(value["wo"])

    def _load_ffn_json(self, ffn: FeedForward, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            return
        if isinstance(value.get("w1"), dict):
            ffn.w1 = Linear.from_json(value["w1"])
        if isinstance(value.get("w2"), dict):
            ffn.w2 = Linear.from_json(value["w2"])
        ffn.moeEnabled = bool(value.get("moeEnabled", ffn.moeEnabled))
        ffn.moeTopK = int(value.get("moeTopK", ffn.moeTopK))
        ffn.moeAuxWeight = float(value.get("moeAuxWeight", ffn.moeAuxWeight))
        if isinstance(value.get("moeGate"), dict):
            ffn.moeGate = Linear.from_json(value["moeGate"])
        ffn.expertUsageEma = list(value.get("expertUsageEma", ffn.expertUsageEma))
        if isinstance(value.get("moeW1"), list):
            ffn.moeW1 = [Linear.from_json(v) for v in value["moeW1"] if isinstance(v, dict)]
        if isinstance(value.get("moeW2"), list):
            ffn.moeW2 = [Linear.from_json(v) for v in value["moeW2"] if isinstance(v, dict)]

    def loadStateDict(self, state: Dict[str, Any], errorOut: Optional[List[str]] = None) -> bool:
        try:
            if not isinstance(state, dict):
                raise ValueError("model state must be object")
            if isinstance(state.get("params"), dict):
                self.params = _build_params_from(self.params, state["params"])
            self._sync_component_params()
            if isinstance(state.get("tokenizer"), dict):
                self._tokenizer.fromJson(state["tokenizer"])
            self.tokEmbed = Matrix.from_json(state.get("tokEmbed", self.tokEmbed.to_json()))
            self.posEmbed = Matrix.from_json(state.get("posEmbed", self.posEmbed.to_json()))
            self.outProj = Linear.from_json(state.get("outProj", self.outProj.to_json()))
            self.fuseGate = Linear.from_json(state.get("fuseGate", self.fuseGate.to_json()))
            self._load_attn_json(self.fuseAttn, state.get("fuseAttn", {}))
            enc = state.get("encLayers", [])
            dec = state.get("decLayers", [])
            if len(enc) == len(self.encLayers):
                for i, item in enumerate(enc):
                    self._load_attn_json(self.encLayers[i].selfAttn, item.get("selfAttn", {}))
                    self._load_ffn_json(self.encLayers[i].ffn, item.get("ffn", {}))
                    if isinstance(item.get("ln1"), dict):
                        self.encLayers[i].ln1 = LayerNorm.from_json(item["ln1"])
                    if isinstance(item.get("ln2"), dict):
                        self.encLayers[i].ln2 = LayerNorm.from_json(item["ln2"])
            if len(dec) == len(self.decLayers):
                for i, item in enumerate(dec):
                    self._load_attn_json(self.decLayers[i].selfAttn, item.get("selfAttn", {}))
                    self._load_attn_json(self.decLayers[i].crossAttn, item.get("crossAttn", {}))
                    self._load_ffn_json(self.decLayers[i].ffn, item.get("ffn", {}))
                    if isinstance(item.get("ln1"), dict):
                        self.decLayers[i].ln1 = LayerNorm.from_json(item["ln1"])
                    if isinstance(item.get("ln2"), dict):
                        self.decLayers[i].ln2 = LayerNorm.from_json(item["ln2"])
                    if isinstance(item.get("ln3"), dict):
                        self.decLayers[i].ln3 = LayerNorm.from_json(item["ln3"])
            self._train_step = int(state.get("trainStep", self._train_step))
            self._loss_ema = float(state.get("lossEma", self._loss_ema))
            return True
        except Exception as ex:
            if errorOut is not None:
                errorOut.clear()
                errorOut.append(str(ex))
            return False

    def saveCheckpoint(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.stateDict(), f, ensure_ascii=False)
        return {"ok": True, "path": str(p)}

    def loadCheckpoint(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {"ok": False, "error": f"checkpoint not found: {p}"}
        with p.open("r", encoding="utf-8") as f:
            state = json.load(f)
        err: List[str] = []
        ok = self.loadStateDict(state, err)
        return {"ok": ok, "path": str(p), "error": err[0] if (not ok and err) else ""}


class TransformerMainEngine:
    def __init__(self, params: Optional[Any] = None) -> None:
        self.params = params if params is not None else TransformerParams()
        _arena_singleton.init(64 * 1024 * 1024)
        self.model = TransformerModel(self.params)
        self.started_at_ms = _now_ms()

    def chat(self, text: str, graphContext: str = "", maxTokens: int = 128) -> Dict[str, Any]:
        tok = self.model._tokenizer
        inp = tok.encode(str(text), int(_get(self.params, "maxLen", 256)), True)
        graph = tok.encode(str(graphContext), int(_get(self.params, "maxLen", 256)), True)
        out = self.model.generate(inp, graph, maxTokens=max(8, int(maxTokens)), graphWeight=float(_get(self.params, "graphWeight", 0.8)), vocabLimit=tok.activeVocabLimit())
        return {"reply": tok.decode(out), "tokens": out}

    def verify(self, text: str, graphContext: str, reply: str) -> Dict[str, Any]:
        tok = self.model._tokenizer
        t = set(tok.encode(text, int(_get(self.params, "maxLen", 256)), False))
        g = set(tok.encode(graphContext, int(_get(self.params, "maxLen", 256)), False))
        r = set(tok.encode(reply, int(_get(self.params, "maxLen", 256)), False))
        def j(a: set[int], b: set[int]) -> float:
            return 1.0 if (not a and not b) else (len(a & b) / max(1, len(a | b)))
        jt = j(t, r)
        jg = j(g, r)
        score = 0.6 * jt + 0.4 * jg
        return {"ok": True, "score": score, "jText": jt, "jGraph": jg}

    def updateParams(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        self.params = _build_params_from(self.params, dict(patch))
        self.model.updateParams(self.params)
        return self.paramsJson()

    def paramsJson(self) -> Dict[str, Any]:
        return _params_to_dict(self.params)

    def train(self, samples: Sequence[Any], epochs: int = 1, lr: Optional[float] = None) -> Dict[str, Any]:
        losses: List[float] = []
        tok = self.model._tokenizer
        for _ in range(max(1, int(epochs))):
            for s in samples:
                inp = tok.encode(str(getattr(s, "input", "")), int(_get(self.params, "maxLen", 256)), True)
                tgt = tok.encode(str(getattr(s, "target", "")), int(_get(self.params, "maxLen", 256)), True)
                graph = tok.encode(str(getattr(s, "graph", "")), int(_get(self.params, "maxLen", 256)), True)
                losses.append(self.model.trainOnSample(inp, tgt, graph, 0.0 if lr is None else float(lr)))
        return {"ok": True, "meanLoss": (sum(losses) / max(1, len(losses))), "samples": len(samples), "epochs": max(1, int(epochs))}

    def optimizeGA(self, samples: Sequence[Any], generations: int = 4, population: int = 6) -> Dict[str, Any]:
        base = self.paramsJson()
        best = dict(base)
        best_score = -1e9
        for _ in range(max(1, int(generations))):
            for _ in range(max(1, int(population))):
                cand = dict(base)
                cand["temperature"] = max(0.1, min(1.8, float(base.get("temperature", 1.0)) + random.uniform(-0.25, 0.25)))
                cand["topP"] = max(0.2, min(1.0, float(base.get("topP", 0.95)) + random.uniform(-0.15, 0.15)))
                cand["topK"] = max(1, min(64, int(base.get("topK", 20)) + random.randint(-4, 4)))
                self.updateParams(cand)
                score = -float(self.train(samples, epochs=1, lr=float(base.get("lr", 2e-4))).get("meanLoss", 1e9))
                if score > best_score:
                    best_score = score
                    best = dict(cand)
        self.updateParams(best)
        return {"ok": True, "bestScore": best_score, "params": best}

    def addFeedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not hasattr(self, "_feedback"):
            self._feedback: List[Dict[str, Any]] = []
        self._feedback.append({
            "text": str(payload.get("text", "")),
            "graph": str(payload.get("graphContext", payload.get("graph", ""))),
            "reply": str(payload.get("reply", "")),
            "score": float(payload.get("score", 0.0)),
        })
        keep = int(_get(self.params, "rlhfReplaySize", 256))
        if len(self._feedback) > keep:
            self._feedback = self._feedback[-keep:]
        return {"ok": True, "count": len(self._feedback)}

    def trainFromFeedback(self, epochs: int = 1, lr: Optional[float] = None) -> Dict[str, Any]:
        feedback = getattr(self, "_feedback", [])
        samples: List[TrainSample] = []
        for item in feedback:
            if float(item.get("score", 0.0)) >= 0.0:
                samples.append(TrainSample(input=item.get("text", ""), target=item.get("reply", ""), graph=item.get("graph", "")))
        if not samples:
            return {"ok": True, "meanLoss": 0.0, "samples": 0}
        return self.train(samples, epochs=max(1, int(epochs)), lr=lr)

    def saveCheckpoint(self, path: str) -> Dict[str, Any]:
        return self.model.saveCheckpoint(path)

    def loadCheckpoint(self, path: str) -> Dict[str, Any]:
        out = self.model.loadCheckpoint(path)
        if out.get("ok"):
            self.params = self.model.params
        return out

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "uptimeMs": _now_ms() - self.started_at_ms,
            "params": self.paramsJson(),
            "tokenizer": self.model._tokenizer.toJson(),
            "trainStep": self.model._train_step,
            "lossEma": self.model._loss_ema,
        }


_engine_lock = threading.RLock()
_engine_singleton: Optional[TransformerMainEngine] = None
_worker_engine_singleton: Optional[TransformerMainEngine] = None
_worker_params_patch: Optional[Dict[str, Any]] = None


def get_engine() -> TransformerMainEngine:
    global _engine_singleton
    with _engine_lock:
        if _engine_singleton is None:
            _engine_singleton = TransformerMainEngine()
        return _engine_singleton


def reset_engine(params: Optional[Any] = None) -> TransformerMainEngine:
    global _engine_singleton
    with _engine_lock:
        _engine_singleton = TransformerMainEngine(params)
        return _engine_singleton


def _preset_patch(name: str) -> Dict[str, Any]:
    key = str(name or "default").strip().lower()
    if key in {"tiny", "tiny-fast", "fast", "tiny_fast"}:
        return {
            "vocabSize": 2048,
            "dModel": 64,
            "nHeads": 4,
            "nLayers": 2,
            "dFF": 128,
            "maxLen": 96,
            "maxTokens": 64,
            "topK": 8,
            "topP": 0.9,
            "temperature": 0.8,
            "dynamicSampling": False,
            "attnChunkSize": 24,
            "tokenizerMode": "byte",
            "enableDynamicReasoning": True,
            "simpleTokenThreshold": 24,
            "noRepeatNgram": 2,
            "minNewTokens": 4,
            "draftTokens": 1,
        }
    return {}


def _make_engine_from_patch(params_patch: Optional[Dict[str, Any]]) -> TransformerMainEngine:
    if params_patch:
        params = _build_params_from(TransformerParams(), dict(params_patch))
        return TransformerMainEngine(params)
    return TransformerMainEngine()


def _worker_init(params_patch: Optional[Dict[str, Any]]) -> None:
    global _worker_params_patch
    _worker_params_patch = dict(params_patch or {})


def _worker_get_engine() -> TransformerMainEngine:
    global _worker_engine_singleton
    if _worker_engine_singleton is None:
        _worker_engine_singleton = _make_engine_from_patch(_worker_params_patch)
    return _worker_engine_singleton


def _worker_chat_call(text: str, graph_context: str, max_tokens: int) -> Dict[str, Any]:
    engine = _worker_get_engine()
    return engine.chat(text, graph_context, maxTokens=max_tokens)


class HfChatBackend:
    def __init__(
        self,
        model_name: str,
        default_max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        device_preference: str = "auto",
        use_compile: bool = False,
    ) -> None:
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace backend requires torch + transformers")

        self.model_name = str(model_name)
        self.default_max_new_tokens = max(8, int(default_max_new_tokens))
        self.temperature = float(max(0.0, temperature))
        self.top_p = float(max(0.0, min(1.0, top_p)))
        self.top_k = int(max(0, top_k))
        self.device_preference = str(device_preference or "auto").lower()
        self.use_compile = bool(use_compile)
        self._lock = threading.RLock()

        self.device = "cpu"
        if torch is not None and torch.cuda.is_available() and self.device_preference in {"auto", "cuda", "gpu"}:
            self.device = "cuda"

        if torch is not None and hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        model_dtype = torch.float16 if (torch is not None and self.device == "cuda") else (torch.float32 if torch is not None else None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=model_dtype)
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()

        if self.use_compile and torch is not None and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
            except Exception:
                pass

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "hf",
            "model": self.model_name,
            "device": self.device,
            "temperature": self.temperature,
            "topP": self.top_p,
            "topK": self.top_k,
            "compile": self.use_compile,
        }

    def chat(self, text: str, graph_context: str = "", max_tokens: Optional[int] = None) -> Dict[str, Any]:
        prompt = str(text or "")
        graph = str(graph_context or "")
        if graph.strip():
            prompt = f"Context:\n{graph}\n\nUser:\n{prompt}\n\nAssistant:\n"

        max_new_tokens = self.default_max_new_tokens if max_tokens is None else max(8, int(max_tokens))
        do_sample = self.temperature > 0.0
        temperature = self.temperature if self.temperature > 0.0 else 1.0

        with self._lock:
            encoded = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                encoded = {k: v.to("cuda") for k, v in encoded.items()}

            with torch.inference_mode():
                out = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

        input_len = int(encoded["input_ids"].shape[-1]) if "input_ids" in encoded else 0
        out_ids = out[0].tolist()
        gen_ids = out_ids[input_len:] if input_len < len(out_ids) else []
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return {
            "reply": reply,
            "tokens": gen_ids,
            "provider": "hf",
            "model": self.model_name,
            "device": self.device,
        }


__all__ = [
    "TransformerReservedArena",
    "MatrixHotspotCache",
    "Linear",
    "LayerNorm",
    "MultiHeadAttention",
    "FeedForward",
    "EncoderLayer",
    "DecoderLayer",
    "Tokenizer",
    "TransformerModel",
    "TransformerMainEngine",
    "get_engine",
    "reset_engine",
    "TransformerParams",
    "TrainSample",
]


def _build_experiment_app(
    default_max_tokens: int = 128,
    preset: str = "default",
    backend: str = "auto",
    worker_processes: int = 1,
    chat_timeout_ms: int = 120000,
    hf_model: str = "distilgpt2",
    hf_temperature: float = 0.7,
    hf_top_p: float = 0.9,
    hf_top_k: int = 40,
    hf_device: str = "auto",
    hf_use_compile: bool = False,
):
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)
    params_patch = _preset_patch(preset)
    engine = _make_engine_from_patch(params_patch)
    resolved_backend = "custom"
    hf_backend: Optional[HfChatBackend] = None
    backend_key = str(backend or "auto").strip().lower()
    if backend_key in {"hf", "auto"} and HF_AVAILABLE:
        try:
            hf_backend = HfChatBackend(
                model_name=hf_model,
                default_max_new_tokens=max(8, int(default_max_tokens)),
                temperature=float(hf_temperature),
                top_p=float(hf_top_p),
                top_k=int(hf_top_k),
                device_preference=hf_device,
                use_compile=hf_use_compile,
            )
            resolved_backend = "hf"
        except Exception:
            hf_backend = None
            resolved_backend = "custom"
    infer_lock = threading.RLock()
    default_max_tokens = max(8, int(default_max_tokens))
    worker_processes = max(1, min(16, int(worker_processes)))
    chat_timeout_ms = max(1000, min(600000, int(chat_timeout_ms)))
    if resolved_backend != "custom":
        worker_processes = 1
    process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
    if worker_processes > 1 and resolved_backend == "custom":
        process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_processes,
            initializer=_worker_init,
            initargs=(params_patch,),
        )

    def run_chat_once(text: str, graph_context: str, max_tokens: int) -> Dict[str, Any]:
        if resolved_backend == "hf" and hf_backend is not None:
            return hf_backend.chat(text, graph_context, max_tokens)
        if process_pool is None:
            with infer_lock:
                return engine.chat(text, graph_context, maxTokens=max_tokens)
        future = process_pool.submit(_worker_chat_call, text, graph_context, max_tokens)
        return future.result(timeout=chat_timeout_ms / 1000.0)

    index_html = """<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Transformer Flask 实验</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 980px; margin: 20px auto; padding: 0 12px; }
        textarea, input { width: 100%; box-sizing: border-box; margin-top: 8px; }
        textarea { min-height: 110px; }
        button { margin-top: 12px; padding: 8px 14px; }
        pre { background: #f5f5f5; padding: 12px; overflow: auto; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .muted { color: #666; font-size: 13px; }
    </style>
</head>
<body>
    <h2>Transformer 单机实验（Flask）</h2>
    <div class="muted">用于隔离测试 Python 版推理时延，不经过现有 C++ 网关链路。</div>

    <label>text</label>
    <textarea id="text">你好，请简短介绍你自己。</textarea>

    <div class="row">
        <div>
            <label>graphContext</label>
            <textarea id="graphContext"></textarea>
        </div>
        <div>
            <label>maxTokens</label>
            <input id="maxTokens" type="number" min="8" max="512" value="" />
            <div class="muted" id="defaults"></div>
        </div>
    </div>

    <button id="send">发送 /chat</button>
    <button id="bench">执行 /bench(5轮)</button>

    <h3>结果</h3>
    <pre id="out">等待请求...</pre>

    <script>
        const out = document.getElementById('out');
        const defaults = document.getElementById('defaults');
        defaults.textContent = '默认 maxTokens: ' + String(__DEFAULT_MAX_TOKENS__);

        async function postJson(path, payload){
            const t0 = performance.now();
            const res = await fetch(path, {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify(payload)
            });
            const txt = await res.text();
            let json;
            try { json = JSON.parse(txt); } catch { json = {raw: txt}; }
            return { status: res.status, elapsedMs: Math.round(performance.now()-t0), body: json };
        }

        document.getElementById('send').onclick = async () => {
            const text = document.getElementById('text').value;
            const graphContext = document.getElementById('graphContext').value;
            const maxTokensRaw = document.getElementById('maxTokens').value;
            const payload = { text, graphContext };
            if (maxTokensRaw) payload.maxTokens = Number(maxTokensRaw);
            out.textContent = '请求中...';
            try {
                const result = await postJson('/chat', payload);
                out.textContent = JSON.stringify(result, null, 2);
            } catch (e) {
                out.textContent = String(e && e.stack ? e.stack : e);
            }
        };

        document.getElementById('bench').onclick = async () => {
            const text = document.getElementById('text').value;
            const graphContext = document.getElementById('graphContext').value;
            out.textContent = '压测中...';
            try {
                const result = await postJson('/bench', { text, graphContext, rounds: 5, warmup: 1 });
                out.textContent = JSON.stringify(result, null, 2);
            } catch (e) {
                out.textContent = String(e && e.stack ? e.stack : e);
            }
        };
    </script>
</body>
</html>
"""
    index_html = index_html.replace("__DEFAULT_MAX_TOKENS__", str(default_max_tokens))

    @app.get("/")
    def index():
        return Response(index_html, content_type="text/html; charset=utf-8")

    @app.get("/health")
    def health():
        backend_status = {"backend": resolved_backend}
        if hf_backend is not None:
            backend_status.update(hf_backend.status())
        return jsonify(
            {
                "ok": True,
                "ts": _now_ms(),
                "status": engine.status(),
                "preset": preset,
                "backend": resolved_backend,
                "backendStatus": backend_status,
                "workerProcesses": worker_processes,
                "chatTimeoutMs": chat_timeout_ms,
            }
        )

    @app.post("/chat")
    def chat_route():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", ""))
        graph_context = str(payload.get("graphContext", ""))
        max_tokens = int(payload.get("maxTokens", default_max_tokens) or default_max_tokens)
        max_tokens = max(8, min(512, max_tokens))
        if not text.strip():
            return jsonify({"ok": False, "error": "text required"}), 400

        t0 = time.perf_counter()
        try:
            result = run_chat_once(text, graph_context, max_tokens)
        except concurrent.futures.TimeoutError:
            return jsonify({"ok": False, "error": "chat-timeout", "timeoutMs": chat_timeout_ms}), 504

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        reply = str(result.get("reply", ""))
        tokens = result.get("tokens", [])
        return jsonify(
            {
                "ok": True,
                "elapsedMs": elapsed_ms,
                "reply": reply,
                "replyLength": len(reply),
                "tokenCount": len(tokens) if isinstance(tokens, list) else 0,
                "preset": preset,
                "backend": resolved_backend,
                "workerProcesses": worker_processes,
                "result": result,
            }
        )

    @app.post("/bench")
    def bench_route():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text", "你好"))
        graph_context = str(payload.get("graphContext", ""))
        rounds = max(1, min(30, int(payload.get("rounds", 5) or 5)))
        warmup = max(0, min(10, int(payload.get("warmup", 1) or 1)))
        max_tokens = max(8, min(512, int(payload.get("maxTokens", default_max_tokens) or default_max_tokens)))

        warmups: List[int] = []
        samples: List[int] = []

        try:
            for _ in range(warmup):
                t0 = time.perf_counter()
                _ = run_chat_once(text, graph_context, max_tokens)
                warmups.append(int((time.perf_counter() - t0) * 1000))

            for _ in range(rounds):
                t0 = time.perf_counter()
                _ = run_chat_once(text, graph_context, max_tokens)
                samples.append(int((time.perf_counter() - t0) * 1000))
        except concurrent.futures.TimeoutError:
            return jsonify({"ok": False, "error": "bench-timeout", "timeoutMs": chat_timeout_ms}), 504

        avg_ms = int(sum(samples) / max(1, len(samples)))
        sorted_samples = sorted(samples)
        p50 = sorted_samples[len(sorted_samples) // 2]
        p95 = sorted_samples[min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.95))]

        return jsonify(
            {
                "ok": True,
                "textLength": len(text),
                "rounds": rounds,
                "warmup": warmup,
                "maxTokens": max_tokens,
                "preset": preset,
                "backend": resolved_backend,
                "workerProcesses": worker_processes,
                "warmupMs": warmups,
                "samplesMs": samples,
                "avgMs": avg_ms,
                "p50Ms": p50,
                "p95Ms": p95,
                "minMs": min(samples),
                "maxMs": max(samples),
            }
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Transformer Flask experiment")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5099)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--preset", default="default", choices=["default", "tiny-fast"])
    parser.add_argument("--backend", default="auto", choices=["auto", "custom", "hf"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chat-timeout-ms", type=int, default=120000)
    parser.add_argument("--hf-model", default="distilgpt2")
    parser.add_argument("--hf-temperature", type=float, default=0.7)
    parser.add_argument("--hf-top-p", type=float, default=0.9)
    parser.add_argument("--hf-top-k", type=int, default=40)
    parser.add_argument("--hf-device", default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    parser.add_argument("--hf-use-compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        import flask  # noqa: F401
    except Exception:
        raise SystemExit("Flask is required for this experiment. Install it with: pip install flask")

    app = _build_experiment_app(
        default_max_tokens=max(8, args.max_tokens),
        preset=args.preset,
        backend=args.backend,
        worker_processes=max(1, int(args.workers)),
        chat_timeout_ms=max(1000, int(args.chat_timeout_ms)),
        hf_model=str(args.hf_model),
        hf_temperature=float(args.hf_temperature),
        hf_top_p=float(args.hf_top_p),
        hf_top_k=int(args.hf_top_k),
        hf_device=str(args.hf_device),
        hf_use_compile=bool(args.hf_use_compile),
    )
    app.run(host=args.host, port=max(1, int(args.port)), debug=bool(args.debug), threaded=True)


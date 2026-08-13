#!/usr/bin/env python3
"""(1+lambda) additive residual BPU evolution controller for video/audio models.

Runs on the compile host (Kali / Linux with hb_mapper or Docker) and pushes
compiled .bin models to the RDK X5 for BPU evaluation.  Each round:

  1. Load current ``best.pt``.
  2. Sample a fresh batch from the local data pool.
  3. Generate ``--lambda`` candidate models (parent + one new residual block).
  4. Export each candidate to ONNX and calibrate.
  5. Compile candidates in parallel with ``tools/compile_bpu.sh``.
  6. Push the compiled .bin files and the batch to the X5 and run
     ``tools/x5_bpu_evaluate.py``.
  7. Select the lowest-MSE candidate as the new ``best.pt/best.onnx``.

Usage example:
    python tools/bpu_evolve_additive.py \
        --model-name speech_decoder \
        --data-pool /media/sf_phoenix/additive/speech_decoder/pool \
        --work-dir /media/sf_phoenix/additive/work \
        --x5-host 192.168.0.107 --x5-user sunrise --x5-pass sunrise \
        --max-rounds 10 --lambda 4 --batch-size 1000 --parallel 2

To generate a synthetic smoke-test pool:
    python tools/bpu_evolve_additive.py \
        --model-name speech_decoder \
        --work-dir /media/sf_phoenix/additive/work \
        --prepare-synthetic-pool 1000 \
        --data-pool /media/sf_phoenix/additive/speech_decoder/pool
"""

import argparse
import atexit
import concurrent.futures
import copy
import json
import os
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from vboxsf_safe import safe_copy, safe_json_dump, safe_np_save, safe_tofile

# Allow importing additive_jepa.py from the tools directory.
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from additive_jepa import (
    AdditiveResidualModel,
    CONCEPT,
    SPEECH_CHUNK,
    SPEECH_TARGET,
    VISION_RES,
    MODEL_INPUT_SHAPES,
    MODEL_OUTPUT_SHAPES,
    MODEL_INPUT_NAMES,
    get_input_shape,
    get_output_shape,
    build_block,
    export_to_onnx,
    _torch_load_safe,
)

try:
    import paramiko
except Exception:
    paramiko = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import torchvision.transforms as T
except Exception:
    T = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.lower().split("x"))


def shape_to_str(shape: Tuple[int, ...]) -> str:
    return "x".join(str(x) for x in shape)


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_json_dump(path, state, indent=2)


def load_state(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data pool
# ---------------------------------------------------------------------------

class DataPool:
    """Load or generate a local pool of (input, target) float32 samples."""

    def __init__(self, pool_dir: Path, model_name: str, concept: int = CONCEPT):
        self.pool_dir = Path(pool_dir)
        self.model_name = model_name
        self.concept = concept
        self.input_shape = get_input_shape(model_name, concept)
        self.output_shape = get_output_shape(model_name, concept)

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        inputs_path = self.pool_dir / "inputs.npy"
        targets_path = self.pool_dir / "targets.npy"
        if not inputs_path.is_file() or not targets_path.is_file():
            raise FileNotFoundError(
                f"Data pool {self.pool_dir} missing inputs.npy / targets.npy"
            )
        inputs = np.load(inputs_path).astype(np.float32)
        targets = np.load(targets_path).astype(np.float32)
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets length mismatch in data pool")
        if inputs.shape[1:] != self.input_shape or targets.shape[1:] != self.output_shape:
            raise ValueError(
                f"Data pool shape mismatch: found {inputs.shape[1:]}/{targets.shape[1:]} "
                f"but expected {self.input_shape}/{self.output_shape}. "
                f"Regenerate the pool with --concept {self.concept}."
            )
        return inputs, targets

    def sample(self, batch_size: int, round_idx: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        inputs, targets = self.load()
        rng = np.random.default_rng(seed + round_idx)
        n = len(inputs)
        if batch_size <= n:
            idx = rng.choice(n, size=batch_size, replace=False)
        else:
            idx = rng.choice(n, size=batch_size, replace=True)
        return inputs[idx].copy(), targets[idx].copy()

    def write_bin_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        out_dir: Path,
        round_idx: int,
    ) -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        inp_path = out_dir / f"round_{round_idx:04d}_inputs.bin"
        tgt_path = out_dir / f"round_{round_idx:04d}_targets.bin"
        safe_tofile(inputs, inp_path)
        safe_tofile(targets, tgt_path)
        return inp_path, tgt_path

    def prepare_synthetic(self, n: int, seed: int = 0):
        """Generate a random pool for smoke testing (no teacher required)."""
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        # Use float32 directly so large image pools do not allocate float64.
        inputs = rng.standard_normal(size=(n, *self.input_shape), dtype=np.float32) * 0.5
        targets = rng.standard_normal(size=(n, *self.output_shape), dtype=np.float32) * 0.5
        safe_np_save(self.pool_dir / "inputs.npy", inputs)
        safe_np_save(self.pool_dir / "targets.npy", targets)
        print(f"[pool] synthetic {n} samples -> {self.pool_dir}")

    def prepare_from_dataset(
        self,
        dataset_dir: Path,
        teacher_path: Optional[Path],
        max_samples: int,
        seed: int = 0,
    ):
        """Generate a real pool from a dataset and an optional teacher model."""
        if not teacher_path:
            raise ValueError("--teacher-pt is required for real data pool preparation")
        teacher = load_teacher(teacher_path, self.concept)
        teacher.eval().cpu()

        self.pool_dir.mkdir(parents=True, exist_ok=True)
        inputs, targets = [], []

        if "speech" in self.model_name:
            for wav in _iter_speech_dataset(dataset_dir, max_samples, seed):
                x = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
                with torch.no_grad():
                    z = infer_teacher(teacher, x, self.model_name)
                if self.model_name == "speech_encoder":
                    inputs.append(x.numpy())
                    targets.append(z.numpy())
                else:  # speech_decoder
                    inputs.append(z.numpy())
                    t = x[..., :SPEECH_TARGET]
                    targets.append(t.numpy())

        elif "vision" in self.model_name:
            for raw, norm in _iter_vision_dataset(dataset_dir, max_samples, seed):
                raw = raw.unsqueeze(0)
                norm = norm.unsqueeze(0)
                with torch.no_grad():
                    z = infer_teacher(teacher, norm, self.model_name)
                if self.model_name == "vision_encoder":
                    inputs.append(norm.numpy())
                    targets.append(z.numpy())
                else:  # vision_decoder
                    inputs.append(z.numpy())
                    targets.append(raw.numpy())

        else:
            raise ValueError(f"unknown model_name {self.model_name}")

        if not inputs:
            raise RuntimeError(f"no samples found in {dataset_dir}")

        safe_np_save(self.pool_dir / "inputs.npy", np.stack(inputs).astype(np.float32))
        safe_np_save(self.pool_dir / "targets.npy", np.stack(targets).astype(np.float32))
        print(f"[pool] generated {len(inputs)} samples -> {self.pool_dir}")


def load_teacher(path: Path, concept: int) -> torch.nn.Module:
    """Load a teacher checkpoint. Supports AdditiveResidualModel or standard JEPA autoencoders."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")

    ckpt = _torch_load_safe(path)

    # 1) AdditiveResidualModel checkpoint
    if isinstance(ckpt, dict) and ckpt.get("model_name") in MODEL_INPUT_SHAPES:
        return AdditiveResidualModel.from_checkpoint(path)

    # 2) JEPA autoencoder state dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        if "enc_channels" in ckpt or "dec_channels" in ckpt:
            from train_audio import AudioAutoencoder
            enc_channels = ckpt.get("enc_channels", [32, 64, 128, 256])
            dec_channels = ckpt.get("dec_channels", [256, 128, 64, 32, 1])
            blocks = ckpt.get("blocks", 0)
            model = AudioAutoencoder(
                enc_channels, dec_channels, concept=concept, blocks=blocks
            )
            model.load_state_dict(state)
            return model
        else:
            from train_video import VideoAutoencoder
            resnet = ckpt.get("resnet", "resnet18")
            dec_channels = ckpt.get("dec_channels")
            dec_depth = ckpt.get("dec_depth", 0)
            model = VideoAutoencoder(
                resnet_name=resnet,
                concept=concept,
                pretrained=False,
                unfreeze=False,
                dec_width=1.0,
                dec_depth=dec_depth,
                dec_channels=dec_channels,
            )
            model.load_state_dict(state)
            return model

    raise ValueError(f"unrecognized teacher checkpoint format: {path}")


def infer_teacher(model: torch.nn.Module, x: torch.Tensor, target_model_name: str) -> torch.Tensor:
    """Get a concept-like output from the teacher, reshaped for the pool."""
    if isinstance(model, AdditiveResidualModel):
        y = model(x)
    elif hasattr(model, "encoder"):
        y = model.encoder(x)
    else:
        y = model(x)
    # Ensure 4-D NCHW concept layout [1, concept, 1, 1]
    if y.dim() == 2:
        y = y.view(y.size(0), -1, 1, 1)
    return y


def _iter_speech_dataset(data_dir: Path, max_samples: int, seed: int):
    if sf is None:
        raise ImportError("soundfile is required for speech data pool")
    files = sorted(Path(data_dir).glob("*.wav"))
    if not files:
        files = sorted(Path(data_dir).rglob("*.wav"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    count = 0
    for f in files:
        if count >= max_samples:
            break
        x, sr = sf.read(str(f), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        for start in range(0, max(1, len(x) - SPEECH_CHUNK + 1), SPEECH_CHUNK):
            chunk = x[start : start + SPEECH_CHUNK]
            if len(chunk) < SPEECH_CHUNK:
                chunk = np.pad(chunk, (0, SPEECH_CHUNK - len(chunk)))
            peak = np.max(np.abs(chunk))
            if peak > 1e-8:
                chunk = chunk / peak
            yield chunk
            count += 1
            if count >= max_samples:
                break


def _iter_vision_dataset(data_dir: Path, max_samples: int, seed: int):
    if Image is None or T is None:
        raise ImportError("PIL and torchvision are required for vision data pool")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for ext in exts:
        files.extend(Path(data_dir).rglob(f"*{ext}"))
    if not files:
        raise FileNotFoundError(f"no images found in {data_dir}")
    files = sorted(files)
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    resize = T.Resize((VISION_RES, VISION_RES))

    for f in files[:max_samples]:
        img = Image.open(str(f)).convert("RGB")
        raw = to_tensor(resize(img))
        norm = normalize(raw)
        yield raw, norm


# ---------------------------------------------------------------------------
# X5 remote helpers
# ---------------------------------------------------------------------------

class X5Remote:
    """Paramiko wrapper to copy files and run commands on the X5, with reconnect and
    SFTP size verification so a single transfer glitch does not corrupt a .bin."""

    def __init__(self, host: str, user: str, password: str, port: int = 22, timeout: int = 30):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.timeout = timeout
        self.client = None
        self.sftp = None

    def _connect(self):
        if paramiko is None:
            raise RuntimeError("paramiko is not installed; install it to use X5 SSH")
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(self.host, port=self.port, username=self.user, password=self.password, timeout=self.timeout)
        c.get_transport().set_keepalive(30)
        self.client = c
        self.sftp = c.open_sftp()

    def _ensure(self):
        if self.client is None:
            self._connect()
            return
        try:
            transport = self.client.get_transport()
            if transport is None or not transport.is_active():
                raise OSError("transport inactive")
        except Exception:
            self.close()
            self._connect()

    def pkill(self, pattern: str):
        """Best-effort kill of any remote process whose command line matches
        ``pattern``.  Used to clean up orphaned processes left behind by a
        dropped SSH channel before (re)launching the same logical job, so
        retries never pile up duplicate long-running processes on the X5
        (this previously caused an OOM that killed Xorg / crashed the box).

        Waits for the remote ``pkill`` to actually finish (reads its exit
        status) instead of firing-and-forgetting; otherwise the channel can
        be torn down before the kill is actually delivered, letting the
        "orphan" race right back in.
        """
        try:
            self._ensure()
            stdin, stdout, stderr = self.client.exec_command(
                f"pkill -9 -f {shlex.quote(pattern)}", timeout=15
            )
            stdout.channel.recv_exit_status()
        except Exception as exc:
            print(f"[x5 ssh] pkill({pattern!r}) failed (non-fatal): {exc}", file=sys.stderr)

    def reap_stale_evaluators(self, max_age_s: int = 650) -> None:
        """Kill any single-candidate ``x5_bpu_evaluate.py --bin ...`` child
        process older than ``max_age_s``.

        Each training round evaluates candidates under a *fresh* round
        directory, so a ``kill_pattern`` scoped to the current round can
        never clean up a process that got stuck (e.g. the BPU runtime
        hanging in an uninterruptible D-state on a bad candidate) in a
        *previous* round -- that process is simply abandoned and leaks
        forever, one per affected round, until the X5 runs out of memory.
        This age-based reaper is a round-independent safety net.

        Only the per-candidate ``--bin <file>`` child processes are
        targeted (not the ``--bin-dir`` parent, which legitimately runs for
        ``n_candidates * 600s`` and must not be killed early).  Each child
        is already given a hard 600s budget by its own parent's
        ``subprocess.run(..., timeout=600)``; anything still alive past
        that is by definition stuck, regardless of which round spawned it.
        """
        cmd = (
            "ps -eo pid,etimes,cmd | "
            "awk -v age=" + str(max_age_s) + " "
            "'$2 > age && /x5_bpu_evaluate\\.py/ && / --bin / && !/--bin-dir/ {print $1}' | "
            "xargs -r kill -9"
        )
        try:
            self._ensure()
            stdin, stdout, stderr = self.client.exec_command(cmd, timeout=20)
            stdout.channel.recv_exit_status()
        except Exception as exc:
            print(f"[x5 ssh] reap_stale_evaluators failed (non-fatal): {exc}", file=sys.stderr)
            return
        # Report anything that survived kill -9 (almost certainly stuck in an
        # uninterruptible kernel/driver wait -- e.g. a BPU runtime hang).
        # Those need a physical/power reboot of the X5 to clear; we can only
        # warn loudly so a human notices instead of the box silently OOMing.
        try:
            stdin, stdout, stderr = self.client.exec_command(
                "ps -eo pid,stat,etimes,cmd | "
                "awk -v age=" + str(max_age_s) + " "
                "'$2 ~ /D/ && $3 > age && /x5_bpu_evaluate\\.py/ && / --bin / && !/--bin-dir/'",
                timeout=15,
            )
            leftover = stdout.read().decode("utf-8", errors="replace").strip()
            if leftover:
                print(
                    "[x5 WARNING] uninterruptible (D-state) evaluator process(es) "
                    "survived kill -9 -- the X5 BPU driver is likely wedged and "
                    "needs a physical reboot:\n" + leftover,
                    file=sys.stderr,
                )
        except Exception:
            pass

    def raise_if_memory_critical(self, min_available_mb: int = 300) -> None:
        """Abort the current round rather than push a nearly-OOM X5 further.

        If a BPU runtime hang has left unkillable (D-state) processes
        holding memory (see ``reap_stale_evaluators``), kill -9 cannot free
        it and every subsequent round just adds more pressure until the
        kernel OOM-killer starts shooting essential system processes
        (observed: it killed Xorg and froze the device with a black
        HDMI/dbus-launch error).  Refusing to schedule new work while
        memory is critically low turns that silent slow-motion crash into a
        clear, actionable error instead.
        """
        try:
            self._ensure()
            stdin, stdout, stderr = self.client.exec_command(
                "awk '/MemAvailable/ {print $2}' /proc/meminfo", timeout=10
            )
            out = stdout.read().decode("utf-8", errors="replace").strip()
            stdout.channel.recv_exit_status()
            available_mb = int(out) // 1024
        except Exception as exc:
            print(f"[x5 ssh] memory check failed (non-fatal): {exc}", file=sys.stderr)
            return
        if available_mb < min_available_mb:
            raise RuntimeError(
                f"X5 available memory critically low ({available_mb} MiB < "
                f"{min_available_mb} MiB) -- likely a wedged BPU runtime "
                "holding unkillable D-state processes. Refusing to schedule "
                "more evaluation work; the X5 needs a physical reboot."
            )

    def exec(self, cmd: str, timeout: int = 300, kill_pattern: Optional[str] = None):
        """Run ``cmd`` on the X5, retrying on transient SSH failures.

        If ``kill_pattern`` is given, any previous process matching it is
        killed before every attempt (including the first).  This guarantees
        that a retry -- triggered by a dropped channel, a client-side read
        timeout, etc. -- can never result in two copies of the same
        long-running remote job (e.g. the per-round BPU evaluator) running
        at once.  Without this, transient network hiccups silently
        accumulate orphaned evaluator processes on the X5 until it runs out
        of memory (observed: OOM killer eventually killing Xorg/dbus and
        freezing the device).
        """
        last_exc = None
        for attempt in range(3):
            if kill_pattern:
                self.pkill(kill_pattern)
            try:
                self._ensure()
                stdin, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
                out = stdout.read().decode("utf-8", errors="replace")
                err = stderr.read().decode("utf-8", errors="replace")
                return stdout.channel.recv_exit_status(), out, err
            except (paramiko.SSHException, OSError, EOFError) as exc:
                last_exc = exc
                print(f"[x5 ssh] exec failed ({exc}), reconnect attempt {attempt + 1}/3", file=sys.stderr)
                self.close()
                time.sleep(2 ** attempt)
        # Make sure we don't leave a dangling process behind for the caller
        # to accidentally race with a subsequent call.
        if kill_pattern:
            self.pkill(kill_pattern)
        raise last_exc

    def put(self, local: Path, remote: Path):
        local = Path(local)
        remote = Path(remote)
        expected = local.stat().st_size
        self.mkdir(remote.parent)
        for attempt in range(3):
            try:
                self._ensure()
                self.sftp.put(str(local), str(remote))
                actual = self.sftp.stat(str(remote)).st_size
                if actual != expected:
                    raise RuntimeError(f"size mismatch after put: {actual} != {expected}")
                return
            except Exception as exc:
                print(f"[x5 sftp] put {local.name} failed ({exc}), retry {attempt + 1}/3", file=sys.stderr)
                self.close()
                time.sleep(2 ** attempt)
        raise RuntimeError(f"failed to put {local} -> {remote}")

    def get(self, remote: Path, local: Path):
        remote = Path(remote)
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                self._ensure()
                self.sftp.get(str(remote), str(local))
                return
            except Exception as exc:
                print(f"[x5 sftp] get {remote.name} failed ({exc}), retry {attempt + 1}/3", file=sys.stderr)
                self.close()
                time.sleep(2 ** attempt)
        raise RuntimeError(f"failed to get {remote} -> {local}")

    def connect(self):
        """Public entry point used by main(); ensures the SSH/SFTP channel exists."""
        self._ensure()

    def mkdir(self, remote: Path):
        self.exec(f"mkdir -p {remote}")

    def close(self):
        if self.sftp:
            try:
                self.sftp.close()
            except Exception:
                pass
            self.sftp = None
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_candidate(
    cand_dir: Path,
    model_name: str,
    concept: int,
    compile_script: Path,
    compile_backend: str,
    per_channel: bool,
    calib_type: str,
    run_hb_mapper: Optional[str],
    march: Optional[str] = None,
    timeout: int = 900,
    skip_compile: bool = False,
) -> Optional[Path]:
    onnx_path = cand_dir / "model.onnx"
    calib_dir = cand_dir / "calibration"
    out_dir = cand_dir / "bpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_compile:
        # Local ORT debugging: create a placeholder .bin; the real model.onnx
        # next to it will be used for evaluation.
        bin_path = out_dir / f"{model_name}.bin"
        bin_path.touch()
        return bin_path

    input_name = MODEL_INPUT_NAMES[model_name]
    input_shape = shape_to_str(get_input_shape(model_name, concept))

    # Use python3 for .py compile scripts, bash for .sh.
    if compile_script.suffix == ".py":
        runner = ["python3", str(compile_script)]
    else:
        runner = ["bash", str(compile_script)]

    cmd = runner + [
        "--backend", compile_backend,
        "--model-name", model_name,
        "--onnx", str(onnx_path),
        "--calib-dir", str(calib_dir),
        "--input-name", input_name,
        "--input-shape", input_shape,
        "--out-dir", str(out_dir),
        "--per-channel", str(per_channel),
        "--calib-type", calib_type,
    ]
    if march:
        cmd += ["--march", march]

    env = os.environ.copy()
    if run_hb_mapper:
        env["RUN_HB_MAPPER"] = run_hb_mapper

    log_path = out_dir / "compile.log"
    try:
        with open(log_path, "w") as logf:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                timeout=timeout,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
    except Exception as e:
        print(f"[compile] failed for {cand_dir}: {e}")
        return None

    bin_path = out_dir / f"{model_name}.bin"
    if bin_path.is_file():
        return bin_path
    onnx_path = out_dir / f"{model_name}.onnx"
    return onnx_path if onnx_path.is_file() else None


# ---------------------------------------------------------------------------
# Batch compilation via persistent Docker worker
# ---------------------------------------------------------------------------

_PERSISTENT_SCRIPT = Path(__file__).resolve().parent / "compile_bpu_docker_persistent.sh"
_HOST_WORK = "/home/kali/phoenix"
_CONTAINER_WORK = "/workspace"
_WORKER_VERIFIED = False  # Cache: once verified running, skip rechecking


def _ensure_persistent_worker() -> bool:
    """Start the persistent Docker compile worker if not running.
    Returns True if worker is available."""
    global _WORKER_VERIFIED
    if _WORKER_VERIFIED:
        return True
    if not _PERSISTENT_SCRIPT.is_file():
        return False
    try:
        status = subprocess.run(
            ["bash", str(_PERSISTENT_SCRIPT), "status"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        if status == "running":
            _WORKER_VERIFIED = True
            return True
        rc = subprocess.run(
            ["bash", str(_PERSISTENT_SCRIPT), "start"],
            timeout=180,
        ).returncode
        if rc == 0:
            _WORKER_VERIFIED = True
        return rc == 0
    except Exception as e:
        print(f"[compile] persistent worker startup failed: {e}")
        return False


def _path_to_container(p: str) -> str:
    """Rewrite a host path under HOST_WORK to the container mount."""
    if p.startswith(_HOST_WORK):
        return _CONTAINER_WORK + p[len(_HOST_WORK):]
    return p


def compile_candidates_batch(
    candidate_dirs: List[Path],
    model_name: str,
    concept: int,
    per_channel: bool,
    calib_type: str,
    march: Optional[str] = None,
    parallel: int = 2,
    skip_compile: bool = False,
) -> List[Optional[Path]]:
    """Compile all candidates in a single batch request to the persistent Docker worker.

    Returns a list of Path (or None) in the same order as candidate_dirs.
    Falls back to individual compile_candidate() calls if the worker is unavailable.
    """
    n = len(candidate_dirs)
    results: List[Optional[Path]] = [None] * n

    if skip_compile:
        for i, cand_dir in enumerate(candidate_dirs):
            out_dir = cand_dir / "bpu"
            out_dir.mkdir(parents=True, exist_ok=True)
            bin_path = out_dir / f"{model_name}.bin"
            bin_path.touch()
            results[i] = bin_path
        return results

    input_name = MODEL_INPUT_NAMES[model_name]
    input_shape = shape_to_str(get_input_shape(model_name, concept))

    # Build job list
    jobs = []
    for i, cand_dir in enumerate(candidate_dirs):
        onnx_path = cand_dir / "model.onnx"
        calib_dir_path = cand_dir / "calibration"
        out_dir = cand_dir / "bpu"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "mapper_work").mkdir(parents=True, exist_ok=True)

        jobs.append({
            "job_id": f"candidate_{i:04d}",
            "model_name": model_name,
            "onnx_path": _path_to_container(str(onnx_path)),
            "calib_dir": _path_to_container(str(calib_dir_path)),
            "input_name": input_name,
            "input_shape": input_shape,
            "out_dir": _path_to_container(str(out_dir)),
            "per_channel": per_channel,
            "calib_type": calib_type,
            "march": march or "bayes-e",
        })

    request = json.dumps({
        "action": "compile",
        "jobs": jobs,
        "parallel": parallel,
    })

    # Send to persistent worker
    try:
        proc = subprocess.run(
            ["bash", str(_PERSISTENT_SCRIPT), "compile"],
            input=request,
            capture_output=True,
            text=True,
            timeout=960,
        )
        if proc.returncode != 0:
            print(f"[compile] batch compile failed: {proc.stderr}")
            return results

        resp = json.loads(proc.stdout.strip())
        if resp.get("status") != "ok":
            print(f"[compile] batch compile error: {resp.get('error')}")
            return results

        elapsed = resp.get("elapsed", 0)
        batch_results = resp.get("results", [])
        print(f"[compile] batch done: {len(batch_results)} jobs in {elapsed:.1f}s")

        # Map results back
        job_id_to_idx = {f"candidate_{i:04d}": i for i in range(n)}
        for r in batch_results:
            idx = job_id_to_idx.get(r.get("job_id"))
            if idx is None:
                continue
            if r.get("ok") and r.get("bin"):
                # The .bin path is inside the container; rewrite to host path
                bin_container = r["bin"]
                bin_host = bin_container.replace(_CONTAINER_WORK, _HOST_WORK, 1)
                results[idx] = Path(bin_host) if Path(bin_host).is_file() else None
                if results[idx] is None:
                    # Try original path as-is (might already be host path)
                    if Path(bin_container).is_file():
                        results[idx] = Path(bin_container)
            else:
                err = r.get("error", "unknown")
                print(f"[compile] {r.get('job_id')} failed: {err}")

    except subprocess.TimeoutExpired:
        print("[compile] batch compile timed out (960s)")
    except Exception as e:
        print(f"[compile] batch compile exception: {e}")

    return results


def compile_model_pt_to_onnx(
    pt_path: Path,
    out_dir: Path,
    model_name: str,
    base_path: Optional[Path],
    n_calib: int = 10,
):
    """Export an existing .pt checkpoint to an ONNX directory."""
    model = AdditiveResidualModel.from_checkpoint(pt_path, base_path=base_path)
    return export_to_onnx(
        model,
        out_dir,
        n_calib=n_calib,
        source_checkpoint=pt_path,
        save_pt=False,
    )


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="(1+lambda) additive BPU evolution")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["speech_encoder", "speech_decoder", "vision_encoder", "vision_decoder"],
    )
    parser.add_argument("--work-dir", default="runtime_store/evolve_additive")
    parser.add_argument("--data-pool", required=False)
    parser.add_argument("--concept", type=int, default=CONCEPT)

    # Control flow
    parser.add_argument(
        "--start-from-zero",
        action="store_true",
        help="Start with no residual blocks (zero output)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from evolve_state.json")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--lambda", type=int, default=4, dest="lambda_")
    parser.add_argument("--block-size", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0x1DEA)

    # Optional ResNet18 base for vision_encoder
    parser.add_argument("--base-path", default=None, help=".pt path for frozen ResNet18 base")

    # Data pool preparation
    parser.add_argument(
        "--prepare-synthetic-pool",
        type=int,
        default=None,
        help="Generate N synthetic samples into --data-pool and exit",
    )
    parser.add_argument(
        "--prepare-pool",
        action="store_true",
        help="Generate real pool from --dataset and --teacher-pt into --data-pool and exit",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--teacher-pt", default=None)
    parser.add_argument("--pool-size", type=int, default=10000)

    # Compilation
    parser.add_argument("--compile-script", default="compile_bpu_docker.sh")
    parser.add_argument("--compile-backend", default="horizon_bpu",
                        help="Target compile backend (horizon_bpu, rockchip_rknn, nvidia_tensorrt)")
    parser.add_argument("--run-hb-mapper", default=None)
    parser.add_argument("--march", default=None)

    # Edge-device evaluation
    parser.add_argument("--eval-script", default="x5_bpu_evaluate.py",
                        help="Remote evaluation script to push and run")
    parser.add_argument("--x5-host", default=None)
    parser.add_argument("--x5-user", default=None)
    parser.add_argument("--x5-pass", default=None)
    parser.add_argument("--x5-port", type=int, default=22)
    parser.add_argument("--x5-work", default="/tmp/phoenix_evolve_real")
    parser.add_argument("--eval-local", action="store_true", help="Evaluate on local CPU with onnxruntime")
    parser.add_argument(
        "--no-bpu",
        action="store_true",
        help="Skip BPU evaluation; use local ORT to compute MSE for debugging",
    )

    parser.add_argument("--keep-candidates", action="store_true")
    parser.add_argument(
        "--keep-rounds",
        action="store_true",
        help="Keep per-round eval_bins/eval_data and X5 round directories (default: prune to save space)",
    )
    return parser.parse_args()


def build_initial_model(args) -> AdditiveResidualModel:
    base = None
    if args.model_name == "vision_encoder":
        from additive_jepa import ResNet18Base
        # Always start from a pretrained ImageNet ResNet18 for the vision encoder.
        # If a custom base .pt is supplied, it is loaded on top of that.
        base = ResNet18Base(
            concept=args.concept,
            base_path=args.base_path if args.base_path else None,
            pretrained=True,
        )

    block_config = {}
    if args.model_name == "speech_decoder":
        first_map = {"small": 12, "medium": 16, "large": 24}
        block_config["first"] = first_map[args.block_size]
    elif args.model_name == "vision_decoder":
        first_map = {"small": 8, "medium": 14, "large": 20}
        block_config["first"] = first_map[args.block_size]

    return AdditiveResidualModel(
        args.model_name,
        concept=args.concept,
        base=base,
        block_config=block_config,
    )


def local_ort_evaluate(
    bin_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, dict]:
    """Local ORT fallback that evaluates the ONNX next to each compiled .bin."""
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime not installed for --no-bpu / --eval-local") from e

    results = {}
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider = ["CPUExecutionProvider"]

    # Evaluate both .bin (BPU) and .onnx (ORT/RKNN/TensorRT fallback) artifacts.
    artifacts = sorted(bin_dir.glob("*.bin")) + sorted(bin_dir.glob("*.onnx"))
    for model_path in artifacts:
        if model_path.suffix == ".bin":
            # For BPU .bin we need the corresponding ONNX for ORT.
            onnx_path = model_path.with_suffix(".onnx")
            if not onnx_path.is_file():
                candidate_onnx = model_path.parent.parent / "model.onnx"
                if candidate_onnx.is_file():
                    onnx_path = candidate_onnx
                else:
                    results[model_path.name] = {"loss": float("inf"), "ok": False, "error": "ONNX not found"}
                    continue
            model_path = onnx_path

        try:
            sess = ort.InferenceSession(str(model_path), sess_options, providers=provider)
        except Exception as e:
            results[model_path.name] = {"loss": float("inf"), "ok": False, "error": str(e)}
            continue

        input_meta = sess.get_inputs()
        no_inputs = len(input_meta) == 0
        in_name = input_meta[0].name if not no_inputs else None
        total = 0.0
        count = 0
        for inp, tgt in zip(inputs, targets):
            tgt = np.asarray(tgt, dtype=np.float32)
            if no_inputs:
                out = sess.run(None, {})[0]
            else:
                inp = np.asarray(inp, dtype=np.float32)
                out = sess.run(None, {in_name: inp})[0]
            out = out.reshape(-1)
            tgt = tgt.reshape(-1)
            min_len = min(out.size, tgt.size)
            out = out[:min_len]
            tgt = tgt[:min_len]
            total += float(np.mean((out - tgt) ** 2))
            count += 1
        loss = total / count if count else float("inf")
        results[model_path.name] = {"loss": loss, "ok": True}
        print(f"[local-ort] {model_path.name}: loss={loss:.6f}")
    return results


def x5_evaluate(
    x5: X5Remote,
    x5_work: Path,
    round_dir: Path,
    model_name: str,
    concept: int,
    input_bin: Path,
    target_bin: Path,
    eval_script: Path,
    eval_local: bool,
    no_bpu: bool,
) -> Dict[str, dict]:
    eval_bins_dir = round_dir / "eval_bins"
    if not any(eval_bins_dir.glob("*.bin")) and not any(eval_bins_dir.glob("*.onnx")):
        raise RuntimeError(f"no compiled .bin or .onnx files in {eval_bins_dir}")

    in_shape = get_input_shape(model_name, concept)
    out_shape = get_output_shape(model_name, concept)
    input_shape_str = shape_to_str(in_shape)
    target_shape_str = shape_to_str(out_shape)

    if no_bpu or eval_local:
        # For local ORT we need the original inputs/targets arrays, not the bin files.
        inputs = np.fromfile(input_bin, dtype=np.float32).reshape(-1, *in_shape)
        targets = np.fromfile(target_bin, dtype=np.float32).reshape(-1, *out_shape)
        return local_ort_evaluate(eval_bins_dir, inputs, targets)

    if x5 is None:
        raise RuntimeError("--x5-host required for BPU evaluation (or use --no-bpu / --eval-local)")

    # Round-independent safety net: kill any evaluator process from an
    # earlier round that got stuck past its own timeout.  Must run before
    # every round (not just retries of the current one), since a stuck
    # process from round N is otherwise never revisited once round N+1's
    # eval uses a brand new round directory / kill_pattern.
    x5.reap_stale_evaluators()
    x5.raise_if_memory_critical()

    x5.mkdir(x5_work)
    x5_round = x5_work / round_dir.name
    # Remove any stale files from previous (interrupted) runs so the
    # evaluator does not pick up leftover .bin files and disk does not fill.
    clean_cmd = f"rm -rf {x5_work}/round_*"
    rc_clean, out_clean, err_clean = x5.exec(clean_cmd, timeout=300)
    if rc_clean != 0:
        print(
            f"[warn] X5 pre-round cleanup failed: rc={rc_clean} err={err_clean.strip()}",
            file=sys.stderr,
        )
    x5.mkdir(x5_round)
    x5_bins = x5_round / "bins"
    x5.mkdir(x5_bins)

    # Upload compiled artifacts (.bin for BPU, .onnx for ORT/RKNN/TensorRT)
    for artifact in sorted(eval_bins_dir.glob("*.bin")) + sorted(eval_bins_dir.glob("*.onnx")):
        x5.put(artifact, x5_bins / artifact.name)
    x5.put(input_bin, x5_round / "inputs.bin")
    x5.put(target_bin, x5_round / "targets.bin")
    eval_name = eval_script.name
    x5.put(eval_script, x5_round / eval_name)
    # x5_bpu_evaluate.py imports helpers from edge_evaluate_common.py; make sure
    # the dependency is also present on the X5 in the same round directory.
    common_helper = eval_script.parent / "edge_evaluate_common.py"
    if common_helper.is_file():
        x5.put(common_helper, x5_round / common_helper.name)

    losses_remote = x5_round / "losses.json"
    cmd = (
        f"cd {x5_round} && python3 {eval_name} "
        f"--bin-dir bins "
        f"--inputs inputs.bin "
        f"--targets targets.bin "
        f"--input-shape {input_shape_str} "
        f"--target-shape {target_shape_str} "
        f"--format bin "
        f"--out {losses_remote}"
    )
    # x5_bpu_evaluate.py evaluates each candidate in its own 600s-capped
    # subprocess, sequentially.  Size the client-side channel timeout to the
    # worst case (all candidates take the full 600s) plus margin, so a
    # slow-but-healthy round never looks like a dropped connection.  Using a
    # fixed 600s here caused `exec()` to time out mid-evaluation as models
    # grew larger / candidate counts increased, which triggered a retry that
    # relaunched the *entire* --bin-dir evaluation while the original one was
    # still running on the X5 -- orphaned duplicates piled up over time and
    # eventually exhausted the X5's RAM (OOM-killer killing Xorg/dbus and
    # freezing the device).  `kill_pattern` below also guarantees any such
    # leftover process is killed before each attempt, so retries can no
    # longer create duplicates even if the timeout estimate is still wrong.
    n_candidates = len(list(eval_bins_dir.glob("*.bin"))) + len(list(eval_bins_dir.glob("*.onnx")))
    eval_timeout = max(600, 650 * max(n_candidates, 1))
    rc, out, err = x5.exec(cmd, timeout=eval_timeout, kill_pattern=str(x5_round))
    print(out)
    if rc != 0:
        print(err, file=sys.stderr)
        raise RuntimeError(f"X5 evaluation failed: rc={rc}")

    local_json = round_dir / "losses.json"
    x5.get(losses_remote, local_json)

    # Free X5 disk immediately; the next round also cleans stale rounds, but
    # doing it here prevents accumulation if the process crashes before the
    # next round starts.
    post_clean_cmd = f"rm -rf {x5_round}"
    rc_post, out_post, err_post = x5.exec(post_clean_cmd, timeout=300)
    if rc_post != 0:
        print(
            f"[warn] X5 post-round cleanup failed: rc={rc_post} err={err_post.strip()}",
            file=sys.stderr,
        )

    with open(local_json, "r", encoding="utf-8") as f:
        return json.load(f)


def prune_eval_artifacts(round_dir: Path) -> None:
    """Remove per-round compiled/evaluation artifacts that are no longer needed."""
    for sub in ("eval_bins", "eval_data"):
        p = round_dir / sub
        if p.exists():
            try:
                shutil.rmtree(p)
            except Exception as exc:
                print(f"[warn] failed to prune {p}: {exc}", file=sys.stderr)


def run_round(
    args,
    work_dir: Path,
    model: AdditiveResidualModel,
    state: dict,
    pool: DataPool,
    x5: Optional[X5Remote],
    compile_script: Path,
    eval_script: Path,
    round_idx: int,
) -> Tuple[AdditiveResidualModel, dict]:
    round_dir = work_dir / "candidates" / f"round_{round_idx:04d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    eval_bins_dir = round_dir / "eval_bins"
    eval_bins_dir.mkdir(parents=True, exist_ok=True)

    # Sample fresh batch
    inputs, targets = pool.sample(args.batch_size, round_idx, args.seed)
    eval_data_dir = work_dir / "eval_data"
    input_bin, target_bin = pool.write_bin_batch(inputs, targets, eval_data_dir, round_idx)
    print(f"[round {round_idx}] batch inputs={inputs.shape} targets={targets.shape}")

    # Baseline parent .bin/.onnx (if it exists)
    # eval_local still compiles real .bin (for deployment) but evaluates on the
    # local CPU with ONNX Runtime instead of copying to the X5 BPU.
    skip_compile = args.no_bpu
    best_bin = work_dir / "best.bin"
    best_onnx = work_dir / "best.onnx"
    if best_bin.is_file():
        safe_copy(best_bin, eval_bins_dir / "candidate_parent.bin")
        if best_onnx.is_file():
            safe_copy(best_onnx, eval_bins_dir / "candidate_parent.onnx")
    elif skip_compile and best_onnx.is_file():
        # Local ORT debug mode: use the ONNX without a real .bin.
        (eval_bins_dir / "candidate_parent.bin").touch()
        safe_copy(best_onnx, eval_bins_dir / "candidate_parent.onnx")

    # Generate lambda candidates
    candidate_dirs: List[Path] = []
    per_channel = "encoder" in args.model_name
    calib_type = "kl" if "encoder" in args.model_name else "max"

    for i in range(args.lambda_):
        cand_dir = round_dir / f"candidate_{i:04d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        cand = copy.deepcopy(model)
        seed = args.seed + round_idx * 10000 + i + 1
        cand.add_block(seed=seed)
        export_to_onnx(
            cand,
            cand_dir,
            n_calib=10,
            source_checkpoint=None,
            save_pt=True,
        )
        candidate_dirs.append(cand_dir)

    # Compile candidates
    # Try batch compilation via persistent Docker worker first (much faster:
    # eliminates ~19s/candidate Docker+hb_mapper cold-start overhead).
    # Falls back to the old per-candidate subprocess approach if unavailable.
    use_batch = (
        args.compile_backend == "horizon_bpu"
        and _PERSISTENT_SCRIPT.is_file()
        and not skip_compile
    )
    if use_batch and _ensure_persistent_worker():
        print(f"[compile] using persistent worker (batch of {len(candidate_dirs)}, parallel={args.parallel})")
        batch_results = compile_candidates_batch(
            candidate_dirs,
            args.model_name,
            args.concept,
            per_channel,
            calib_type,
            args.march,
            parallel=args.parallel,
            skip_compile=skip_compile,
        )
        for i, bin_path in enumerate(batch_results):
            if bin_path is not None:
                suffix = bin_path.suffix
                safe_copy(bin_path, eval_bins_dir / f"candidate_{i:04d}{suffix}")
                cand_onnx = candidate_dirs[i] / "model.onnx"
                if cand_onnx.is_file() and suffix != ".onnx":
                    safe_copy(cand_onnx, eval_bins_dir / f"candidate_{i:04d}.onnx")
    else:
        # Legacy: per-candidate subprocess compilation
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {
                ex.submit(
                    compile_candidate,
                    d,
                    args.model_name,
                    args.concept,
                    compile_script,
                    args.compile_backend,
                    per_channel,
                    calib_type,
                    args.run_hb_mapper,
                    args.march,
                    900,
                    skip_compile,
                ): i
                for i, d in enumerate(candidate_dirs)
            }
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                bin_path = future.result()
                if bin_path is not None:
                    suffix = bin_path.suffix
                    safe_copy(bin_path, eval_bins_dir / f"candidate_{i:04d}{suffix}")
                    cand_onnx = candidate_dirs[i] / "model.onnx"
                    if cand_onnx.is_file() and suffix != ".onnx":
                        safe_copy(cand_onnx, eval_bins_dir / f"candidate_{i:04d}.onnx")

    missing = [i for i, d in enumerate(candidate_dirs) if not any(eval_bins_dir.glob(f"candidate_{i:04d}.*"))]
    if missing:
        print(f"[warn] round {round_idx}: candidates {missing} failed to compile")

    if not any(eval_bins_dir.glob("candidate_*.bin")) and not any(eval_bins_dir.glob("candidate_*.onnx")):
        print(f"[error] no candidates compiled for round {round_idx}; skipping")
        return model, state

    # Evaluate on X5 (or local)
    try:
        results = x5_evaluate(
            x5,
            Path(args.x5_work),
            round_dir,
            args.model_name,
            args.concept,
            input_bin,
            target_bin,
            eval_script,
            args.eval_local,
            args.no_bpu,
        )
    except Exception as exc:
        print(f"[error] round {round_idx}: X5 evaluation failed: {exc}; skipping round", file=sys.stderr)
        state["round"] = round_idx
        state["history"].append({
            "round": round_idx,
            "best_loss": state.get("best_loss", float("inf")),
            "best_key": "none",
            "n_blocks": len(model.blocks),
            "new_block_added": False,
            "error": str(exc),
        })
        save_state(work_dir / "evolve_state.json", state)
        if not args.keep_candidates:
            for d in candidate_dirs:
                shutil.rmtree(d, ignore_errors=True)
        if not args.keep_rounds:
            prune_eval_artifacts(round_dir)
        return model, state

    if not results:
        print(f"[warn] round {round_idx}: no eval results; treating as no improvement")
        state["round"] = round_idx
        state["history"].append({
            "round": round_idx,
            "best_loss": state.get("best_loss", float("inf")),
            "best_key": "none",
            "n_blocks": len(model.blocks),
            "new_block_added": False,
        })
        save_state(work_dir / "evolve_state.json", state)
        if not args.keep_candidates:
            for d in candidate_dirs:
                shutil.rmtree(d, ignore_errors=True)
        if not args.keep_rounds:
            prune_eval_artifacts(round_dir)
        return model, state

    # Select best
    best_key = min(
        results,
        key=lambda k: results[k].get("loss", float("inf")) if results[k].get("ok") else float("inf"),
    )
    best_loss = results[best_key].get("loss", float("inf"))
    print(f"[round {round_idx}] best={best_key} loss={best_loss:.6f}")

    parent_key = None
    for k in results:
        if k.startswith("candidate_parent"):
            parent_key = k
            break
    parent_loss = results.get(parent_key, {}).get("loss", float("inf")) if parent_key else float("inf")
    new_block_added = False

    if best_key == parent_key:
        # Keep existing best
        print(f"[round {round_idx}] parent retained (loss {best_loss:.6f})")
        state["best_loss"] = best_loss
    elif best_loss < state.get("best_loss", float("inf")):
        idx = int(best_key.split("_")[1].split(".")[0])
        cand_dir = candidate_dirs[idx]
        # Promote candidate to best
        safe_copy(cand_dir / "model.pt", work_dir / "best.pt")
        safe_copy(cand_dir / "model.onnx", work_dir / "best.onnx")
        best_artifact = eval_bins_dir / best_key
        suffix = best_artifact.suffix
        # Copy the native compiled artifact (.bin / .rknn / .trt / .onnx).
        if suffix in (".bin", ".rknn", ".trt", ".onnx"):
            safe_copy(best_artifact, work_dir / f"best{suffix}")
        # Always keep an ONNX fallback.
        if (work_dir / "best.onnx").is_file():
            pass
        elif cand_dir.joinpath("model.onnx").is_file():
            safe_copy(cand_dir / "model.onnx", work_dir / "best.onnx")
        model = AdditiveResidualModel.from_checkpoint(work_dir / "best.pt")
        new_block_added = True
        state["best_loss"] = best_loss
        state["n_blocks"] = len(model.blocks)
        print(f"[round {round_idx}] selected candidate {idx} -> best.pt ({len(model.blocks)} blocks)")
    else:
        # No improvement; keep parent but record the evaluated loss
        print(f"[round {round_idx}] no improvement over best {state.get('best_loss')}; keeping parent")

    state["round"] = round_idx
    state["history"].append({
        "round": round_idx,
        "best_loss": state["best_loss"],
        "best_key": best_key,
        "n_blocks": state["n_blocks"],
        "new_block_added": new_block_added,
    })
    save_state(work_dir / "evolve_state.json", state)

    if not args.keep_candidates:
        # Keep the selected candidate and parent; prune the rest to save space.
        for i, d in enumerate(candidate_dirs):
            if new_block_added and d == candidate_dirs[idx]:
                continue
            try:
                shutil.rmtree(d)
            except Exception:
                pass

    if not args.keep_rounds:
        # Compiled/eval artifacts are copied to best.* or kept in losses.json.
        prune_eval_artifacts(round_dir)

    return model, state


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # signal 0: check liveness without killing
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else -- assume alive


def _acquire_work_dir_lock(work_dir: Path) -> None:
    """Ensure only one live process owns this work_dir, killing any other.

    Two independent ``bpu_evolve_additive.py`` instances for the same model
    were previously able to run concurrently (e.g. the orchestrator restarts
    a model in the background while the user also manually re-runs
    ``run_all_real_training.bat`` in the foreground -- the user has no way
    to know about or reach the background instance to stop it first).  Both
    write the same ``evolve_state.json`` / ``best.pt`` and drive the same X5
    work directory, whose per-round cleanup (``rm -rf round_*``) deletes
    whatever the *other* instance is currently uploading/evaluating.  That
    produced spurious X5 evaluation failures, which triggered SSH retries,
    which (before the ``exec()`` fix above) piled up orphaned evaluator
    processes on the X5 until it ran out of memory.

    Since progress is checkpointed to ``evolve_state.json``/``best.pt`` every
    round and reloaded via ``--resume``, it is always safe to kill whichever
    instance was here first and take over -- so we do that automatically
    instead of refusing to start.
    """
    lock_path = work_dir / ".evolve.lock"
    if lock_path.is_file():
        try:
            old_pid = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None and old_pid != os.getpid() and _is_alive(old_pid):
            print(
                f"[lock] {work_dir} is locked by running PID {old_pid}; killing it and "
                "taking over (state is checkpointed every round, so --resume picks up "
                "cleanly). This replaces the old 'refuse to start' behavior since the "
                "user has no way to reach/stop a background-started instance before "
                "manually re-running the training script.",
                file=sys.stderr,
            )
            try:
                os.kill(old_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(20):  # up to ~10s for a graceful exit
                if not _is_alive(old_pid):
                    break
                time.sleep(0.5)
            else:
                print(f"[lock] PID {old_pid} did not exit after SIGTERM, sending SIGKILL", file=sys.stderr)
                try:
                    os.kill(old_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                time.sleep(1)
    lock_path.write_text(str(os.getpid()))
    atexit.register(lambda: lock_path.unlink(missing_ok=True))


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir) / args.model_name
    work_dir.mkdir(parents=True, exist_ok=True)
    _acquire_work_dir_lock(work_dir)

    compile_script = Path(args.compile_script)
    if not compile_script.is_absolute():
        # Try relative to CWD, then fall back to the tools directory.
        cwd_candidate = Path.cwd() / compile_script
        if cwd_candidate.is_file():
            compile_script = cwd_candidate
        else:
            compile_script = _TOOLS_DIR / compile_script.name
    eval_script = Path(args.eval_script)
    if not eval_script.is_absolute():
        eval_script = _TOOLS_DIR / eval_script.name

    # Data pool preparation (one-shot)
    if args.prepare_synthetic_pool is not None:
        if not args.data_pool:
            print("--data-pool is required for --prepare-synthetic-pool", file=sys.stderr)
            return 1
        pool = DataPool(Path(args.data_pool), args.model_name, args.concept)
        pool.prepare_synthetic(args.prepare_synthetic_pool, args.seed)
        return 0

    if args.prepare_pool:
        if not args.data_pool or not args.dataset:
            print("--data-pool and --dataset required for --prepare-pool", file=sys.stderr)
            return 1
        pool = DataPool(Path(args.data_pool), args.model_name, args.concept)
        pool.prepare_from_dataset(
            Path(args.dataset),
            Path(args.teacher_pt) if args.teacher_pt else None,
            args.pool_size,
            args.seed,
        )
        return 0

    if not args.data_pool:
        print("--data-pool is required (or use --prepare-synthetic-pool / --prepare-pool)", file=sys.stderr)
        return 1
    pool = DataPool(Path(args.data_pool), args.model_name, args.concept)

    # X5 remote
    x5 = None
    if args.x5_host and not (args.no_bpu or args.eval_local):
        if paramiko is None:
            print("paramiko not installed; cannot connect to X5", file=sys.stderr)
            return 1
        x5 = X5Remote(args.x5_host, args.x5_user, args.x5_pass, port=args.x5_port)
        x5.connect()
        # Kill any evaluator process orphaned by a previous crash/restart of
        # this model's controller (e.g. after a Ctrl-C, SSH drop, or a prior
        # OOM freeze).  Scoped to this model's work dir so it never touches
        # the other models' concurrent evaluations on the same X5.
        x5.pkill(f"{Path(args.x5_work)}/round_")

    # Load or initialize model and state
    state_path = work_dir / "evolve_state.json"
    best_pt = work_dir / "best.pt"
    best_onnx = work_dir / "best.onnx"

    if args.resume and state_path.is_file() and best_pt.is_file():
        state = load_state(state_path)
        model = AdditiveResidualModel.from_checkpoint(best_pt, base_path=args.base_path)
        print(f"[resume] round={state.get('round')} n_blocks={len(model.blocks)} best_loss={state.get('best_loss')}")
    else:
        state = {
            "round": 0,
            "n_blocks": 0,
            "best_loss": float("inf"),
            "history": [],
        }
        model = build_initial_model(args)
        if args.start_from_zero:
            state["n_blocks"] = 0
        else:
            # Start with a single residual block.
            model.add_block(seed=args.seed)
            state["n_blocks"] = 1

        export_to_onnx(model, best_onnx.parent, n_calib=10, source_checkpoint=None, save_pt=True)
        # export_to_onnx writes model.pt and model.onnx in out_dir; rename to best.*.
        if (best_onnx.parent / "model.pt").is_file():
            shutil.move(best_onnx.parent / "model.pt", best_pt)
        if (best_onnx.parent / "model.onnx").is_file():
            shutil.move(best_onnx.parent / "model.onnx", best_onnx)
        save_state(state_path, state)
        print(f"[init] {args.model_name} n_blocks={len(model.blocks)} best.pt -> {best_pt}")

    # Round loop
    try:
        for round_idx in range(state["round"] + 1, args.max_rounds + 1):
            print(f"\n========== Round {round_idx}/{args.max_rounds} ==========")
            t0 = time.time()
            model, state = run_round(
                args,
                work_dir,
                model,
                state,
                pool,
                x5,
                compile_script,
                eval_script,
                round_idx,
            )
            print(f"[round {round_idx}] elapsed {time.time() - t0:.1f}s")
    finally:
        if x5:
            try:
                if not args.keep_rounds:
                    # Best artifacts are already copied back to Kali; the X5
                    # work directory only contains temporary eval files.
                    x5.exec(f"rm -rf {Path(args.x5_work)}/{args.model_name}/round_*", timeout=300)
            except Exception as exc:
                print(f"[warn] X5 final cleanup failed: {exc}", file=sys.stderr)
            x5.close()

    print(f"\n[done] best_loss={state['best_loss']:.6f} n_blocks={state['n_blocks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Black-box (1+lambda) evolution of the small speech autoencoder decoder.

This is a Windows orchestrator that drives the Kali BPU-compile host and the
RDK X5 evaluation host via paramiko.  It was written as an orchestrator because
Kali (192.168.0.100) has no route to the X5 (192.168.0.107); the Windows host
sits in the middle and uses its local temp directory as a file bridge.

A single run performs one generation with ``--lambda`` decoder candidates:

1. On Kali: load ``--n-chunks`` 16 kHz WAV chunks, run the fixed base encoder
   (ONNX Runtime CPU) to build concept/target batches, and save them.
2. On Kali: start from the exported ``init4`` decoder ONNX, mutate the
   trainable weight vector with Gaussian noise (``sigma``), export a candidate
   decoder ONNX for each child, and compile it to ``candidate_i.bin`` with the
   Horizon OpenExplorer Docker image.
3. On Windows: pull each ``.bin`` from Kali and push it to the X5.
4. On X5: run ``x5_bpu_evaluate.py`` with the pre-computed concepts and audio
   targets, producing a ``losses.json``.
5. On Windows: pick the lowest-MSE candidate, adapt ``sigma``, update the
   parent, and save the best state + manifest.

Usage (run from a Windows host inside the Phoenix repo):

    python tools\\bpu_evolve_speech_decoder.py \
        --kali-host 192.168.0.100 --kali-user kali --kali-pass kali \
        --x5-host 192.168.0.107 --x5-user sunrise --x5-pass sunrise \
        --generations 1 --lambda 4

The default is exactly one generation (``--generations 1 --lambda 4``), which is
what this first milestone uses to prove the end-to-end loop.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import numpy as np
import paramiko

# ---------------------------------------------------------------------------
# User-configurable defaults
# ---------------------------------------------------------------------------
KALI_HOST = "192.168.0.100"
KALI_USER = "kali"
KALI_PASS = "kali"
X5_HOST = "192.168.0.107"
X5_USER = "sunrise"
X5_PASS = "sunrise"

KALI_VENV = "/opt/ijepa_build/venv/bin/python3"
DOCKER_IMAGE = "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"

KALI_WORK = PurePosixPath("/media/sf_phoenix/speech_evolve/evolve")
KALI_INIT = PurePosixPath("/media/sf_phoenix/speech_evolve/init4")
KALI_DATASET = PurePosixPath("/home/kali/phoenix/datasets/musan_16k")

X5_WORK = PurePosixPath("/home/sunrise/phoenix/evolve")

WIN_TEMP = Path(os.environ.get("TEMP", r"C:\Users\%USERNAME%\AppData\Local\Temp"))
if "%USERNAME%" in str(WIN_TEMP):
    WIN_TEMP = Path(r"C:\Users\木木\AppData\Local\Temp")
WIN_BRIDGE = WIN_TEMP / "bpu_evolve_bridge"

CHUNK = 16000
CONCEPT = 32

# ---------------------------------------------------------------------------
# Compile wrapper run inside the OpenExplorer Docker container
# ---------------------------------------------------------------------------
COMPILE_WRAPPER = r"""#!/usr/bin/env bash
# Compile a single decoder ONNX to Horizon .bin.
# Designed to run inside the OpenExplorer Docker container with
# /workspace/speech and /workspace/tools mounted as in compile_bpu_speech_small.sh.

set -uo pipefail

ONNX_DIR="${1:-/workspace/speech/onnx}"
OUT_DIR="${2:-/workspace/speech/bin}"
BIN_NAME="${3:-model_decoder}"
ONNX_NAME="${4:-$BIN_NAME}"
MARCH="${MARCH:-bayes-e}"

mkdir -p "$OUT_DIR" "$OUT_DIR/mapper_work"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

fix_onnx_ir() {
    python3 - <<PY
import onnx
m = onnx.load('$ONNX_DIR/${ONNX_NAME}.onnx')
m.ir_version = 8
onnx.save(m, '$ONNX_DIR/${ONNX_NAME}.onnx')
PY
}

compile_onnx() {
    local onnx_path="$ONNX_DIR/${ONNX_NAME}.onnx"
    local calib_dir="$ONNX_DIR/calibration_decoder"
    local config_file="$OUT_DIR/${BIN_NAME}_config.yaml"
    local mapper_out="$OUT_DIR/mapper_work/${BIN_NAME}"

    rm -rf "$mapper_out"
    rm -f "$OUT_DIR/${BIN_NAME}.bin"
    rm -f "$OUT_DIR/${BIN_NAME}.onnx"
    mkdir -p "$mapper_out"

    if [[ ! -f "$onnx_path" ]]; then
        echo "ERROR: ONNX not found: $onnx_path" >&2
        exit 1
    fi
    if [[ ! -d "$calib_dir" ]]; then
        echo "ERROR: Calibration not found: $calib_dir" >&2
        exit 1
    fi

    cat > "$config_file" <<EOF
model_parameters:
  onnx_model: '$onnx_path'
  march: '$MARCH'
  output_model_file_prefix: '${BIN_NAME}_${MARCH}'
  working_dir: '$mapper_out'
  layer_out_dump: False

input_parameters:
  input_name: 'concept'
  input_shape: '1x32x1x1'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
  input_type_rt: 'featuremap'
  input_layout_rt: 'NHWC'
  norm_type: 'no_preprocess'

calibration_parameters:
  cal_data_dir: '$calib_dir'
  cal_data_type: 'float32'
  calibration_type: 'max'
  per_channel: False
EOF

    log "Compiling $BIN_NAME from $onnx_path with march=$MARCH"
    # hb_mapper may abort with a free()/invalid-pointer error after producing
    # the .bin; ignore the exit code and search for the output file.
    python3 /workspace/tools/run_hb_mapper.py makertbin \
        --config "$config_file" --model-type onnx || true

    local bin_file
    bin_file=$(find "$mapper_out" -maxdepth 2 -name '*.bin' | head -n 1)
    if [[ ! -f "$bin_file" ]]; then
        echo "ERROR: No .bin produced for $BIN_NAME in $mapper_out" >&2
        exit 1
    fi
    cp "$bin_file" "$OUT_DIR/${BIN_NAME}.bin"
    cp "$onnx_path" "$OUT_DIR/${BIN_NAME}.onnx"
    log "Wrote $OUT_DIR/${BIN_NAME}.bin"
}

fix_onnx_ir
compile_onnx
log "Done: $OUT_DIR/${BIN_NAME}.bin"
true
"""

# ---------------------------------------------------------------------------
# Kali worker script (run on the Kali host inside the venv)
# ---------------------------------------------------------------------------
KALI_WORKER = r'''#!/usr/bin/env python3
"""Kali-side worker for decoder black-box evolution."""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnxruntime as ort

CHUNK = 16000
CONCEPT = 32


def load_wav(path: str, n: int = CHUNK) -> np.ndarray:
    """Load the first n samples of a WAV file and normalize to [-1, 1]."""
    with wave.open(path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        fr = w.getframerate()
        frames = w.readframes(n)

    if sw == 2:
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(frames, dtype=np.float32)
    else:
        raise ValueError(f"unsupported sample width {sw} in {path}")

    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)))
    arr = arr[:n]

    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)

    peak = float(np.max(np.abs(arr)))
    if peak > 1e-8:
        arr = arr / peak
    return arr


def generate_eval_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Path(args.dataset)
    files = sorted([f for f in os.listdir(dataset) if f.lower().endswith(".wav")])
    rng = np.random.default_rng(args.seed)
    if args.n_chunks < len(files):
        files = rng.choice(files, size=args.n_chunks, replace=False).tolist()
    else:
        files = files[:args.n_chunks]

    sess = ort.InferenceSession(
        str(args.encoder_onnx),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    chunks = []
    for f in files:
        chunks.append(load_wav(str(dataset / f)))

    x = np.stack(chunks).reshape(-1, 1, 1, CHUNK).astype(np.float32)
    concepts = sess.run(None, {input_name: x})[0]
    np.save(out_dir / "concepts.npy", concepts)
    np.save(out_dir / "targets.npy", x)
    print(f"[data] generated {len(chunks)} chunks -> {out_dir}")
    print(f"[data] concepts shape={concepts.shape}, targets shape={x.shape}")


def get_mutable_weights(model: onnx.ModelProto) -> List[str]:
    """Return initialiser names that hold float32 weights/biases."""
    names = []
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        if arr.dtype == np.float32:
            names.append(init.name)
    # Keep a stable order that matches a small speech decoder.
    expected = [
        "fc.weight", "fc.bias",
        "deconv.0.weight", "deconv.0.bias",
        "deconv.1.weight", "deconv.1.bias",
        "deconv.3.weight", "deconv.3.bias",
        "deconv.4.weight", "deconv.4.bias",
        "deconv.6.weight", "deconv.6.bias",
    ]
    ordered = [n for n in expected if n in names]
    extra = [n for n in names if n not in expected]
    return ordered + extra


def load_state(model: onnx.ModelProto, names: List[str]) -> np.ndarray:
    vec = []
    name_to_arr: Dict[str, np.ndarray] = {}
    for name in names:
        init = next(i for i in model.graph.initializer if i.name == name)
        arr = onnx.numpy_helper.to_array(init).astype(np.float32)
        name_to_arr[name] = arr
        vec.append(arr.reshape(-1))
    return np.concatenate(vec)


def set_state(model: onnx.ModelProto, names: List[str], vec: np.ndarray) -> None:
    offset = 0
    for name in names:
        init = next(i for i in model.graph.initializer if i.name == name)
        arr = onnx.numpy_helper.to_array(init).astype(np.float32)
        size = arr.size
        new_arr = vec[offset:offset + size].reshape(arr.shape).astype(np.float32)
        offset += size
        new_init = onnx.numpy_helper.from_array(new_arr, name=name)
        init.CopyFrom(new_init)


def prepare_candidate(args: argparse.Namespace) -> None:
    model = onnx.load(str(args.base_onnx))
    names = get_mutable_weights(model)
    parent = load_state(model, names)

    rng = np.random.default_rng(args.seed)
    child = parent + rng.normal(0.0, args.sigma, size=parent.shape).astype(np.float32)

    set_state(model, names, child)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model_decoder.onnx"
    onnx.save(model, str(onnx_path))
    print(f"[candidate] wrote {onnx_path} with sigma={args.sigma:.6f}")

    calib_src = Path(args.calib_src)
    calib_dst = out_dir / "calibration_decoder"
    if calib_dst.exists():
        shutil.rmtree(calib_dst)
    shutil.copytree(calib_src, calib_dst)
    print(f"[candidate] copied calibration {calib_src} -> {calib_dst}")


def compile_candidate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(args.onnx_dir)
    name = args.name

    # The compile wrapper lives in the shared folder, mounted at
    # /workspace/speech inside Docker.
    wrapper = "/workspace/speech/compile_bpu_speech_decoder.sh"
    onnx_name = "model_decoder"

    def dockerize(p: Path) -> str:
        s = str(p)
        return s.replace("/media/sf_phoenix/speech_evolve", "/workspace/speech", 1)

    cmd = [
        "docker", "run", "--rm",
        "-v", "/media/sf_phoenix/speech_evolve:/workspace/speech",
        "-v", "/opt/ijepa_build/tools:/workspace/tools",
        os.environ.get("DOCKER_IMAGE", "openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310"),
        "bash", wrapper,
        dockerize(onnx_dir), dockerize(out_dir), name, onnx_name,
    ]
    print(f"[compile] {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    print(f"[compile] docker finished in {time.time() - t0:.1f}s exit={proc.returncode}")

    bin_path = out_dir / f"{name}.bin"
    if not bin_path.exists():
        # Fallback: search the mapper work dir.
        for p in (out_dir / "mapper_work" / name).rglob("*.bin"):
            shutil.copy(p, bin_path)
            break
    if not bin_path.exists():
        raise RuntimeError(f"No .bin produced in {out_dir}")
    print(f"[compile] produced {bin_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data")
    p_data.add_argument("--dataset", required=True)
    p_data.add_argument("--encoder-onnx", required=True)
    p_data.add_argument("--out-dir", required=True)
    p_data.add_argument("--n-chunks", type=int, default=4)
    p_data.add_argument("--seed", type=int, default=0)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--base-onnx", required=True)
    p_prep.add_argument("--out-dir", required=True)
    p_prep.add_argument("--calib-src", required=True)
    p_prep.add_argument("--sigma", type=float, default=0.01)
    p_prep.add_argument("--seed", type=int, default=0)

    p_compile = sub.add_parser("compile")
    p_compile.add_argument("--onnx-dir", required=True)
    p_compile.add_argument("--out-dir", required=True)
    p_compile.add_argument("--name", default="model_decoder")

    args = parser.parse_args()
    if args.command == "data":
        generate_eval_data(args)
    elif args.command == "prepare":
        prepare_candidate(args)
    elif args.command == "compile":
        compile_candidate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# Remote host helpers
# ---------------------------------------------------------------------------
class RemoteHost:
    def __init__(self, host: str, user: str, password: str, timeout: int = 20):
        self.host = host
        self.user = user
        self.password = password
        self.timeout = timeout
        self.client: Optional[paramiko.SSHClient] = None
        self.sftp: Optional[paramiko.SFTPClient] = None

    def connect(self) -> None:
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.host, username=self.user, password=self.password, timeout=self.timeout)
        self.sftp = self.client.open_sftp()

    def close(self) -> None:
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.client:
            self.client.close()
            self.client = None

    def exec(self, cmd: str, timeout: int = 60) -> Tuple[int, str, str]:
        if self.client is None:
            raise RuntimeError(f"not connected to {self.host}")
        stdin, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return stdout.channel.recv_exit_status(), out, err

    def upload_text(self, data: str, remote: str, mode: int = 0o755) -> None:
        if self.sftp is None:
            raise RuntimeError(f"not connected to {self.host}")
        with self.sftp.file(remote, "w") as f:
            f.write(data)
        self.sftp.chmod(remote, mode)

    def upload_file(self, local: str, remote: str, mode: int = 0o644) -> None:
        if self.sftp is None:
            raise RuntimeError(f"not connected to {self.host}")
        self.sftp.put(local, remote)
        self.sftp.chmod(remote, mode)

    def download_file(self, remote: str, local: str) -> None:
        if self.sftp is None:
            raise RuntimeError(f"not connected to {self.host}")
        self.sftp.get(remote, local)

    def mkdir(self, remote: str) -> None:
        self.exec(f"mkdir -p {remote}")


class KaliHost(RemoteHost):
    pass


class X5Host(RemoteHost):
    pass


# ---------------------------------------------------------------------------
# Windows bridge helpers
# ---------------------------------------------------------------------------
def ensure_bridge() -> Path:
    WIN_BRIDGE.mkdir(parents=True, exist_ok=True)
    return WIN_BRIDGE


def reset_bridge() -> None:
    if WIN_BRIDGE.exists():
        shutil.rmtree(WIN_BRIDGE)
    WIN_BRIDGE.mkdir(parents=True, exist_ok=True)


def local_bridge(*parts: str) -> Path:
    return WIN_BRIDGE.joinpath(*parts)


# ---------------------------------------------------------------------------
# Kali side of the pipeline
# ---------------------------------------------------------------------------
class KaliPipeline:
    def __init__(self, host: KaliHost, work: PurePosixPath, init: PurePosixPath, dataset: PurePosixPath, venv: str, docker_image: str):
        self.host = host
        self.work = work
        self.init = init
        self.dataset = dataset
        self.venv = venv
        self.docker_image = docker_image

    def setup(self) -> None:
        self.host.exec(f"mkdir -p {self.work} {self.work}/eval_data {self.work}/gen")
        self.host.upload_text(COMPILE_WRAPPER, "/media/sf_phoenix/speech_evolve/compile_bpu_speech_decoder.sh", 0o755)
        self.host.upload_text(KALI_WORKER, "/tmp/bpu_evolve_worker.py", 0o755)

    def generate_eval_data(self, n_chunks: int, seed: int) -> None:
        print("[Kali] generating eval data ...")
        cmd = (
            f"{self.venv} /tmp/bpu_evolve_worker.py data "
            f"--dataset {self.dataset} "
            f"--encoder-onnx {self.init}/model_encoder.onnx "
            f"--out-dir {self.work}/eval_data "
            f"--n-chunks {n_chunks} --seed {seed}"
        )
        rc, out, err = self.host.exec(cmd, timeout=120)
        print(out)
        if rc != 0:
            print(err, file=sys.stderr)
            raise RuntimeError(f"Kali eval-data generation failed: rc={rc}")

    def prepare_candidate(self, gen: int, idx: int, sigma: float, seed: int, parent_onnx: PurePosixPath, base_calib: PurePosixPath) -> PurePosixPath:
        cand_dir = self.work / f"gen_{gen}" / f"candidate_{idx}"
        print(f"[Kali] preparing candidate {idx} (gen {gen}) sigma={sigma:.6f}")
        cmd = (
            f"{self.venv} /tmp/bpu_evolve_worker.py prepare "
            f"--base-onnx {parent_onnx} "
            f"--out-dir {cand_dir} "
            f"--calib-src {base_calib} "
            f"--sigma {sigma} --seed {seed}"
        )
        rc, out, err = self.host.exec(cmd, timeout=120)
        print(out)
        if rc != 0:
            print(err, file=sys.stderr)
            raise RuntimeError(f"Kali candidate prepare failed: rc={rc}")
        return cand_dir

    def compile_candidate(self, gen: int, idx: int, onnx_dir: PurePosixPath, name: str = "model_decoder") -> PurePosixPath:
        bin_dir = self.work / f"gen_{gen}" / "bins"
        print(f"[Kali] compiling candidate {idx} (gen {gen}) ...")
        # Pass the Docker image to the worker via environment.
        env = f"DOCKER_IMAGE={self.docker_image}"
        cmd = (
            f"{env} {self.venv} /tmp/bpu_evolve_worker.py compile "
            f"--onnx-dir {onnx_dir} "
            f"--out-dir {bin_dir} "
            f"--name candidate_{idx}"
        )
        rc, out, err = self.host.exec(cmd, timeout=360)
        print(out)
        if rc != 0:
            print(err, file=sys.stderr)
            # Still accept if .bin exists.
        bin_path = bin_dir / f"candidate_{idx}.bin"
        if not self._exists(str(bin_path)):
            raise RuntimeError(f"Compiled .bin missing: {bin_path}")
        return bin_path

    def _exists(self, path: str) -> bool:
        rc, _, _ = self.host.exec(f"test -f {path}")
        return rc == 0


# ---------------------------------------------------------------------------
# X5 side of the pipeline
# ---------------------------------------------------------------------------
class X5Pipeline:
    def __init__(self, host: X5Host, work: PurePosixPath, local_eval_script: Path):
        self.host = host
        self.work = work
        self.local_eval_script = local_eval_script

    def prepare_gen(self, gen: int) -> PurePosixPath:
        gen_dir = self.work / f"speech_gen_{gen}"
        # Remove any stale .bin files from previous manual/debug runs.
        self.host.exec(f"rm -rf {gen_dir}/bins && mkdir -p {gen_dir}/bins")
        return gen_dir

    def upload_data(self, gen_dir: PurePosixPath, local_concepts: Path, local_targets: Path) -> None:
        self.host.upload_file(str(local_concepts), str(gen_dir / "concepts.npy"))
        self.host.upload_file(str(local_targets), str(gen_dir / "targets.npy"))
        self.host.upload_file(str(self.local_eval_script), str(gen_dir / "x5_bpu_evaluate.py"))

    def upload_bin(self, gen_dir: PurePosixPath, local_bin: Path, name: str) -> None:
        self.host.upload_file(str(local_bin), str(gen_dir / "bins" / name))

    def evaluate(self, gen_dir: PurePosixPath, out_name: str = "losses.json") -> Dict:
        print("[X5] running BPU evaluator ...")
        cmd = (
            f"cd {gen_dir} && python3 x5_bpu_evaluate.py "
            f"--bin-dir bins "
            f"--inputs concepts.npy "
            f"--targets targets.npy "
            f"--out {gen_dir / out_name} "
            f"--pattern '*.bin'"
        )
        rc, out, err = self.host.exec(cmd, timeout=180)
        print(out)
        if rc != 0:
            print(err, file=sys.stderr)
            raise RuntimeError(f"X5 evaluation failed: rc={rc}")
        # Fetch JSON
        local_json = local_bridge(f"gen_{gen_dir.name}_losses.json")
        self.host.download_file(str(gen_dir / out_name), str(local_json))
        with open(local_json, "r", encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Main (1+lambda) ES loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="(1+lambda) BPU decoder evolution")
    parser.add_argument("--kali-host", default=KALI_HOST)
    parser.add_argument("--kali-user", default=KALI_USER)
    parser.add_argument("--kali-pass", default=KALI_PASS)
    parser.add_argument("--x5-host", default=X5_HOST)
    parser.add_argument("--x5-user", default=X5_USER)
    parser.add_argument("--x5-pass", default=X5_PASS)
    parser.add_argument("--kali-work", default=str(KALI_WORK))
    parser.add_argument("--init-dir", default=str(KALI_INIT))
    parser.add_argument("--dataset", default=str(KALI_DATASET))
    parser.add_argument("--x5-work", default=str(X5_WORK))
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--lambda", type=int, default=4, dest="lambda_")
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--sigma-expand", type=float, default=1.05)
    parser.add_argument("--sigma-shrink", type=float, default=0.8)
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regenerate-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reset_bridge()

    kali = KaliHost(args.kali_host, args.kali_user, args.kali_pass)
    x5 = X5Host(args.x5_host, args.x5_user, args.x5_pass)
    kali.connect()
    x5.connect()
    print("[SSH] connected to Kali and X5")

    kali_work = PurePosixPath(args.kali_work)
    init_dir = PurePosixPath(args.init_dir)
    x5_work = PurePosixPath(args.x5_work)

    kali_pipe = KaliPipeline(kali, kali_work, init_dir, PurePosixPath(args.dataset), KALI_VENV, DOCKER_IMAGE)
    x5_pipe = X5Pipeline(x5, x5_work, Path(__file__).with_name("x5_bpu_evaluate.py"))

    kali_pipe.setup()

    # Generate or reuse eval data.
    eval_dir = kali_work / "eval_data"
    if args.regenerate_data or not kali_pipe._exists(str(eval_dir / "concepts.npy")):
        kali_pipe.generate_eval_data(args.n_chunks, args.seed)

    # Pull eval data to Windows bridge and push to X5 (X5 lives in a different
    # subnet, so Kali cannot reach it directly).
    kali.download_file(str(eval_dir / "concepts.npy"), str(local_bridge("concepts.npy")))
    kali.download_file(str(eval_dir / "targets.npy"), str(local_bridge("targets.npy")))

    # Parent: the exported init4 decoder ONNX.
    parent_onnx = init_dir / "model_decoder.onnx"
    parent_name = "parent"
    parent_loss: Optional[float] = None
    parent_vec_local = local_bridge("parent_state.npy")

    best_overall_loss = float("inf")
    best_overall_name = None
    best_overall_gen = -1

    sigma = args.sigma
    total_start = time.time()

    for gen in range(args.generations):
        print(f"\n========== generation {gen} (sigma={sigma:.6f}) ==========")
        gen_start = time.time()

        gen_dir = x5_pipe.prepare_gen(gen)
        x5_pipe.upload_data(gen_dir, local_bridge("concepts.npy"), local_bridge("targets.npy"))

        # Include the base parent .bin as a baseline so the X5 evaluator can
        # compare the parent loss to the mutants in a single run.  The parent
        # .bin is reused from the init4 compilation if it exists; otherwise we
        # compile the un-mutated decoder as candidate_0 later.
        base_bin = init_dir.parent / "speech_bin" / "model_decoder.bin"
        if kali_pipe._exists(str(base_bin)):
            kali.download_file(str(base_bin), str(local_bridge("parent.bin")))
            x5_pipe.upload_bin(gen_dir, local_bridge("parent.bin"), "parent.bin")
            parent_name = "parent.bin"
        else:
            print("[WARN] base parent .bin not found, will compile candidate_0 as parent")

        # Generate and compile lambda mutants.
        all_bin_names = [parent_name] if kali_pipe._exists(str(base_bin)) else []
        for i in range(args.lambda_):
            seed = args.seed + gen * 1000 + i + 1
            cand_dir = kali_pipe.prepare_candidate(gen, i, sigma, seed, parent_onnx, init_dir / "calibration_decoder")
            bin_path = kali_pipe.compile_candidate(gen, i, cand_dir, f"candidate_{i}")

            local_bin = local_bridge(f"gen_{gen}_candidate_{i}.bin")
            kali.download_file(str(bin_path), str(local_bin))
            x5_pipe.upload_bin(gen_dir, local_bin, f"candidate_{i}.bin")
            all_bin_names.append(f"candidate_{i}.bin")

        # Run evaluation on X5.
        losses = x5_pipe.evaluate(gen_dir, "losses.json")

        # Extract parent loss.
        parent_loss = losses.get(parent_name, {}).get("loss")
        if parent_loss is None:
            parent_loss = float("inf")
            print(f"[WARN] parent {parent_name} not in losses, using inf")

        # Best candidate (only among candidate_*.bin, not parent).
        cand_losses = {k: v for k, v in losses.items() if k.startswith("candidate_") and v.get("ok")}
        if not cand_losses:
            print("[ERROR] no candidates evaluated successfully")
            # Keep parent, shrink sigma.
            sigma *= args.sigma_shrink
            continue

        best_name = min(cand_losses, key=lambda k: cand_losses[k]["loss"])
        best_loss = cand_losses[best_name]["loss"]

        # Update parent and sigma.
        if best_loss < parent_loss:
            print(f"[gen {gen}] IMPROVED {parent_name} loss={parent_loss:.6f} -> {best_name} loss={best_loss:.6f}")
            parent_name = best_name
            parent_loss = best_loss
            sigma *= args.sigma_expand
            # Copy the best candidate ONNX to the Kali parent location.
            src_idx = int(best_name.split("_")[1].split(".")[0])
            src_onnx = kali_work / f"gen_{gen}" / f"candidate_{src_idx}" / "model_decoder.onnx"
            # Download the best .bin and .onnx to the Windows bridge for safe-keeping.
            best_bin_remote = kali_work / f"gen_{gen}" / "bins" / best_name
            kali.download_file(str(best_bin_remote), str(local_bridge(f"best_gen_{gen}.bin")))
            kali.download_file(str(src_onnx), str(local_bridge(f"best_gen_{gen}.onnx")))
            parent_onnx = src_onnx
        else:
            print(f"[gen {gen}] NO IMPROVE best={best_name} loss={best_loss:.6f} parent={parent_loss:.6f}")
            sigma *= args.sigma_shrink

        if best_loss < best_overall_loss:
            best_overall_loss = best_loss
            best_overall_name = best_name
            best_overall_gen = gen

        gen_time = time.time() - gen_start
        print(f"[gen {gen}] time={gen_time:.1f}s best={best_name} loss={best_loss:.6f} parent={parent_loss:.6f} sigma={sigma:.6f}")

    total_time = time.time() - total_start

    # Final manifest on Kali.
    manifest = {
        "generations": args.generations,
        "lambda": args.lambda_,
        "final_sigma": sigma,
        "final_parent_loss": parent_loss,
        "best_overall_loss": best_overall_loss,
        "best_overall_name": best_overall_name,
        "best_overall_gen": best_overall_gen,
        "total_seconds": total_time,
        "eval_data": str(eval_dir),
        "best_local_bridge": str(local_bridge("best_gen_*.bin").parent) if best_overall_gen >= 0 else None,
    }
    manifest_path = local_bridge("manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[DONE] total time {total_time:.1f}s")
    if best_overall_gen >= 0:
        print(f"[DONE] best overall {best_overall_name} from gen {best_overall_gen}: MSE={best_overall_loss:.6f}")
        print(f"[DONE] best .bin Windows bridge: {local_bridge(f'best_gen_{best_overall_gen}.bin')}")
        print(f"[DONE] best .onnx Windows bridge: {local_bridge(f'best_gen_{best_overall_gen}.onnx')}")
    else:
        print("[DONE] no successful candidates")

    kali.close()
    x5.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Windows batch runner for the four additive residual JPEA-v2 models.

This script runs on the Windows host, connects to the Kali compile box and the
RDK X5, and starts / monitors BPU evolutions for the four models.  It can also
run the Windows compile / gtest stage.

Examples:
    # Validation run with synthetic data (no teacher required)
    python tools\run_all_additive_training.py

    # Real data run
    python tools\run_all_additive_training.py ^
        --speech-dataset /home/kali/phoenix/datasets/musan_16k ^
        --vision-image-dir /home/kali/phoenix/datasets/images ^
        --max-rounds 1000 --lambda 4 --batch-size 1024 --max-concurrent 2

    # Skip Windows compile/tests and only start Kali evolutions
    python tools\run_all_additive_training.py --no-local-build
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

# Add project root to path so we can import tools.edge_device_manager
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.edge_device_manager import load_edge_device, _resolve_config_path

REPO_ROOT = Path(__file__).resolve().parent.parent

KALI_DEFAULTS = {
    "host": "192.168.0.100",
    "user": "kali",
    "pass": "kali",
    "repo_path": "/home/kali/phoenix",
}

X5_DEFAULTS = {
    "host": "192.168.0.107",
    "user": "sunrise",
    "pass": "sunrise",
    "work": "/home/sunrise/phoenix/evolve_additive",
}

# python3 in the Kali venv.  paramiko is not installed there, so we set
# PYTHONPATH to the system dist-packages where it lives.
KALI_VENV = "/opt/ijepa_build/venv/bin/python3"
KALI_ENV = "env PYTHONPATH=/usr/lib/python3/dist-packages"

# Files that must be present on Kali for this to work (and that we sync on every run).
KALI_FILES = [
    "tools/additive_jpea.py",
    "tools/bpu_evolve_additive.py",
    "tools/export_additive_jpea.py",
    "tools/compile_bpu_jepa_v2.sh",
    "tools/compile_bpu_docker.sh",
    "tools/compile_target_model.py",
    "tools/edge_device_manager.py",
    "tools/run_hb_mapper.py",
    "tools/hb_mapper_patch.py",
    "tools/vboxsf_safe.py",
    "tools/x5_bpu_evaluate.py",
    "tools/rk3588_npu_evaluate.py",
    "tools/jetson_trt_evaluate.py",
    "tools/edge_ort_evaluate.py",
    "tools/x5_onnx_runtime_setup.py",
    "tools/llm_concept_encoder.py",
    "tools/prepare_multimodal_pool.py",
    "tools/run_all_additive_training.py",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-run additive residual BPU training")
    parser.add_argument("--kali-host", default=KALI_DEFAULTS["host"])
    parser.add_argument("--kali-user", default=KALI_DEFAULTS["user"])
    parser.add_argument("--kali-pass", default=KALI_DEFAULTS["pass"])
    parser.add_argument("--kali-repo-path", default=KALI_DEFAULTS["repo_path"])
    parser.add_argument("--x5-host", default=None)
    parser.add_argument("--x5-user", default=None)
    parser.add_argument("--x5-pass", default=None)
    parser.add_argument("--x5-port", type=int, default=None)
    parser.add_argument("--x5-work", default=None)
    parser.add_argument("--edge-device", default=None,
                        help="Device name from config/edge_devices.json")
    parser.add_argument("--edge-device-config", default=None,
                        help="Path to edge device config (default: config/edge_devices.json)")
    parser.add_argument("--compile-backend", default=None,
                        help="Override compile backend (horizon_bpu, rockchip_rknn, nvidia_tensorrt)")
    parser.add_argument("--eval-script", default=None,
                        help="Override remote evaluation script")

    parser.add_argument(
        "--models",
        default="speech_encoder,speech_decoder,vision_encoder,vision_decoder",
        help="Comma-separated list of models to evolve",
    )
    parser.add_argument(
        "--work-dir",
        default="/home/kali/phoenix/additive_work",
        help="Kali working directory for all models",
    )

    # Data options
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument("--synthetic", action="store_true", default=False,
                        help="Use synthetic pool for all models (no teacher)")
    data_group.add_argument("--real-data", action="store_true", default=False,
                        help="Use real MUSAN + Tiny-ImageNet data with LLM concept encoder")
    parser.add_argument("--precomputed-pool", default=None,
                        help="Skip pool preparation and use this existing pool root (subdirs: speech_encoder, ...)")
    parser.add_argument("--speech-dataset", default="/home/kali/phoenix/datasets/musan_16k",
                        help="Path to 16 kHz WAV dataset for speech models")
    parser.add_argument("--vision-image-dir", default="/home/kali/datasets/tiny-imagenet-200",
                        help="Path to image dataset for vision models")
    parser.add_argument("--bge-dir", default="/home/kali/models/bge-small-en",
                        help="Directory with BGE-small-en model for LLM concept encoding")
    parser.add_argument("--vision-base-pt", default=None,
                        help="ResNet18 base .pt for vision_encoder (if None, pretrained ImageNet)")
    parser.add_argument("--teacher-pt", default=None,
                        help="Teacher .pt for legacy real-data pool generation (not needed with --real-data)")
    parser.add_argument("--pool-size", type=int, default=10000,
                        help="Samples per data pool")
    parser.add_argument("--concept", type=int, default=128,
                        help="Concept vector dimension")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Evolution hyperparameters
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Rounds per model (use a large number for real runs)")
    parser.add_argument("--lambda", type=int, default=2, dest="lambda_",
                        help="Candidate count per round")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="X5 eval batch size")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Parallel Docker compile jobs per model")
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="Max concurrent model evolutions")
    parser.add_argument("--block-size", default="small",
                        choices=["small", "medium", "large"],
                        help="Residual block size")

    # Local Windows build
    parser.add_argument("--no-local-build", action="store_true",
                        help="Skip compile_gtest.bat and compile.bat")
    parser.add_argument("--no-gtest-filter", action="store_true",
                        help="Run the full gtest suite instead of the deployment filter")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip pushing files to Kali (assume they are already there)")

    # X5 / eval
    parser.add_argument("--eval-local", action="store_true",
                        help="Evaluate on Kali CPU with ONNX Runtime; do not connect to the X5")
    parser.add_argument("--no-bpu", action="store_true",
                        help="Skip BPU compilation and evaluate ONNX with ONNX Runtime")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each bpu_evolve from evolve_state.json if present")

    # Monitoring
    parser.add_argument("--wait", action="store_true",
                        help="Wait for all evolutions to finish (default: start and report PIDs)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status updates when --wait")

    return parser.parse_args()


def ensure_paramiko():
    try:
        import paramiko
        return paramiko
    except ImportError:
        print("ERROR: paramiko is required on the Windows host.")
        print("       pip install paramiko")
        sys.exit(1)


class KaliClient:
    """Paramiko SSH client that auto-reconnects on stale / timed-out channels."""

    def __init__(self, host, user, password, timeout=30):
        self.host = host
        self.user = user
        self.password = password
        self.timeout = timeout
        self._c = None
        self._paramiko = ensure_paramiko()

    def _connect(self):
        c = self._paramiko.SSHClient()
        c.set_missing_host_key_policy(self._paramiko.AutoAddPolicy())
        c.connect(self.host, username=self.user, password=self.password, timeout=self.timeout)
        c.get_transport().set_keepalive(30)
        return c

    def _client(self):
        if self._c is not None:
            try:
                transport = self._c.get_transport()
                if transport is not None and transport.is_active():
                    return self._c
            except Exception:
                pass
            try:
                self._c.close()
            except Exception:
                pass
            self._c = None
        print(f"[ssh] reconnecting to {self.user}@{self.host} ...")
        self._c = self._connect()
        return self._c

    def exec_command(self, cmd, timeout=300):
        last_exc = None
        for attempt in range(3):
            try:
                return self._client().exec_command(cmd, timeout=timeout)
            except (self._paramiko.SSHException, OSError, EOFError) as exc:
                last_exc = exc
                print(f"[ssh] exec_command failed ({exc}), reconnect attempt {attempt + 1}/3")
                self._c = None
                time.sleep(2 ** attempt)
        raise last_exc

    def open_sftp(self):
        for attempt in range(3):
            try:
                return self._client().open_sftp()
            except (self._paramiko.SSHException, OSError, EOFError) as exc:
                print(f"[ssh] open_sftp failed ({exc}), reconnect attempt {attempt + 1}/3")
                self._c = None
                time.sleep(2 ** attempt)
        raise last_exc

    def close(self):
        if self._c is not None:
            try:
                self._c.close()
            except Exception:
                pass
            self._c = None


def ssh_connect(host, user, password, timeout=30):
    return KaliClient(host, user, password, timeout=timeout)


def run_kali(c, cmd, timeout=300):
    """Run a command on Kali and return stdout text."""
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    if err:
        out += "\n[stderr] " + err
    return out


def push_to_kali(c, files, kali_repo):
    """Push the listed files to the Kali repo path."""
    sftp = c.open_sftp()
    for rel in files:
        local = REPO_ROOT / rel
        if not local.is_file():
            print(f"[warn] missing local file: {local}")
            continue
        remote = PurePosixPath(kali_repo) / rel
        remote_dir = str(remote.parent).replace("\\", "/")
        c.exec_command(f"mkdir -p {remote_dir}")
        sftp.put(str(local), str(remote))
        print(f"[push] {rel} -> {remote}")
    sftp.close()


def local_build():
    """Run compile_gtest.bat and compile.bat on Windows."""
    for script in ["compile_gtest.bat", "compile.bat"]:
        path = REPO_ROOT / script
        if not path.is_file():
            print(f"[warn] {script} not found, skipping")
            continue
        print(f"[local] {script} ...")
        rc = subprocess.run([str(path)], cwd=REPO_ROOT, shell=True).returncode
        if rc != 0:
            print(f"[error] {script} failed with code {rc}")
            return rc
    return 0


def run_gtest_filter():
    gtest = REPO_ROOT / "gtest_runner.exe"
    if not gtest.is_file():
        print("[warn] gtest_runner.exe not found, skipping")
        return 0
    filter_arg = "ModelDeployment.*:MixedModalIOTest.*"
    print(f"[local] gtest_runner.exe --gtest_filter={filter_arg}")
    return subprocess.run([str(gtest), f"--gtest_filter={filter_arg}"], cwd=REPO_ROOT).returncode


def prepare_pools(c, args, models):
    """Prepare or select data pools for all requested models.

    Returns a dict {model: pool_dir}.
    """
    if args.precomputed_pool:
        root = args.precomputed_pool
        print(f"[pool] using precomputed pool root: {root}")
        return {m: f"{root}/{m}" for m in models}

    if args.real_data:
        root = f"{args.work_dir}/pools_real"
        print(f"[pool] preparing real multimodal pools in {root} ...")
        python = f"{KALI_ENV} {KALI_VENV}"
        n = args.pool_size
        # Vision pools are very memory-heavy; keep the cap that prepare_multimodal_pool
        # already enforces, but pass a sensible per-modality number.
        script_args = (
            f"-u tools/prepare_multimodal_pool.py "
            f"--musan-dir {args.speech_dataset} "
            f"--imagenet-dir {args.vision_image_dir} "
            f"--bge-dir {args.bge_dir} "
            f"--out-dir {root} "
            f"--concept {args.concept} "
            f"--max-samples-per-modality {n} "
            f"--seed {args.seed}"
        )
        log = "/tmp/prepare_multimodal_pool.log"
        done_marker = f"{root}/metadata.json"
        # Start in the background because the script can take several minutes.
        full = (
            f"cd {args.kali_repo_path} && "
            f"nohup {python} {script_args} > {log} 2>&1 & "
            f"echo $!"
        )
        pid = run_kali(c, full, timeout=30).strip().split()[-1]
        print(f"[pool] prepare_multimodal_pool.py PID={pid}, log={log}")
        while is_running(c, pid):
            print("[pool] still preparing ...")
            time.sleep(args.poll_interval)
        out = run_kali(c, f"tail -n 40 {log}", timeout=30)
        print(out)
        if not run_kali(c, f"test -f {done_marker} && echo ok", timeout=10).strip():
            print(f"[error] pool preparation failed; marker {done_marker} not found")
            return {}
        return {m: f"{root}/{m}" for m in models}

    # Legacy / synthetic per-model preparation.
    pools = {}
    for model in models:
        pool_dir = f"{args.work_dir}/pools/{model}"
        python = f"{KALI_ENV} {KALI_VENV}"
        cmd_parts = [
            "cd", f"{args.kali_repo_path}", "&&",
            python, "-u", "tools/bpu_evolve_additive.py",
            "--model-name", model,
            "--work-dir", f"{args.work_dir}/{model}",
            "--data-pool", pool_dir,
        ]
        # Use a small pool for quick validation by default.
        # Vision models need tiny pools because the input tensor is very large.
        if model.startswith("vision"):
            n = min(args.pool_size, 100)
        else:
            n = args.pool_size
        cmd_parts += ["--prepare-synthetic-pool", str(n)]
        print(f"[pool] {model}: preparing synthetic ...")
        out = run_kali(c, " ".join(cmd_parts), timeout=300)
        print(out)
        pools[model] = pool_dir
    return pools


def start_evolution(c, model, pool_dir, args):
    """Start one BPU evolution on Kali and return (pid, log_path)."""
    python = f"{KALI_ENV} {KALI_VENV}"
    log = f"/tmp/bpu_evo_{model}.log"
    work_dir = f"{args.work_dir}/{model}"
    # Keep each model in its own X5 work directory to avoid file collisions.
    x5_work = f"{args.x5_work}/{model}"

    # Vision eval batch is memory-heavy on X5; keep it modest unless overridden.
    if model.startswith("vision"):
        batch_size = min(args.batch_size, 50)
    else:
        batch_size = args.batch_size

    cmd_parts = [
        "cd", f"{args.kali_repo_path}", "&&",
        "nohup", python, "-u", "tools/bpu_evolve_additive.py",
        "--model-name", model,
        "--work-dir", work_dir,
        "--data-pool", pool_dir,
        "--x5-host", args.x5_host,
        "--x5-user", args.x5_user,
        "--x5-pass", args.x5_pass,
        "--x5-port", str(args.x5_port),
        "--x5-work", x5_work,
        "--compile-script", "tools/compile_target_model.py",
        "--compile-backend", args.compile_backend,
        "--eval-script", args.eval_script,
        "--max-rounds", str(args.max_rounds),
        "--lambda", str(args.lambda_),
        "--batch-size", str(batch_size),
        "--parallel", str(args.parallel),
        "--block-size", args.block_size,
        "--concept", str(args.concept),
        "--seed", str(args.seed),
    ]
    if args.eval_local:
        cmd_parts.append("--eval-local")
    if args.no_bpu:
        cmd_parts.append("--no-bpu")
    if args.resume:
        cmd_parts.append("--resume")

    if model == "vision_encoder" and args.vision_base_pt:
        cmd_parts += ["--base-path", args.vision_base_pt]

    full_cmd = " ".join(cmd_parts) + f" > {log} 2>&1 & echo $!"
    out = run_kali(c, full_cmd, timeout=30)
    pid = out.strip().split()[-1]
    return pid, log


def tail_log(c, log, n=30):
    try:
        return run_kali(c, f"tail -n {n} {log}", timeout=30)
    except Exception as exc:
        return f"[ssh] could not tail {log}: {exc}"


def resolve_edge_device(args):
    """If --edge-device is given, load its config and fill in x5_* fields."""
    if not args.edge_device:
        return
    try:
        dev = load_edge_device(args.edge_device, config_path=args.edge_device_config)
    except FileNotFoundError:
        print(f"[error] edge device config not found: {_resolve_config_path()}", file=sys.stderr)
        print("        Copy config/edge_devices.example.json to config/edge_devices.json", file=sys.stderr)
        sys.exit(1)
    if args.x5_host is None:
        args.x5_host = dev.cfg.get("host")
    if args.x5_port is None:
        args.x5_port = dev.cfg.get("port", 22)
    if args.x5_user is None:
        args.x5_user = dev.cfg.get("user")
    if args.x5_pass is None:
        args.x5_pass = dev.password
    if args.x5_work is None:
        args.x5_work = dev.cfg.get("work_dir", "/root/phoenix/evolve_real")
    if args.compile_backend is None:
        args.compile_backend = dev.cfg.get("compile_backend", "horizon_bpu")
    if args.eval_script is None:
        args.eval_script = dev.cfg.get("eval_script", "x5_bpu_evaluate.py")


def is_running(c, pid):
    try:
        out = run_kali(c, f"ps -p {pid} -o pid 2>/dev/null | tail -n +2", timeout=10)
        return bool(out.strip())
    except Exception as exc:
        print(f"[ssh] is_running({pid}) failed: {exc}; assuming process still alive")
        return True


def main() -> int:
    args = parse_args()
    resolve_edge_device(args)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not args.no_local_build:
        if local_build() != 0:
            return 1
        if not args.no_gtest_filter:
            if run_gtest_filter() != 0:
                return 1
        else:
            gtest = REPO_ROOT / "gtest_runner.exe"
            if gtest.is_file():
                rc = subprocess.run([str(gtest)], cwd=REPO_ROOT).returncode
                if rc != 0:
                    return rc

    print(f"[kali] connecting to {args.kali_user}@{args.kali_host} ...")
    c = ssh_connect(args.kali_host, args.kali_user, args.kali_pass)

    if not args.skip_push:
        push_to_kali(c, KALI_FILES, args.kali_repo_path)

    # The bpu_evolve_additive.py controller creates the X5 work dir and pushes
    # x5_bpu_evaluate.py via paramiko, so no separate scp/ssh is needed here.
    print(f"[x5] will use work dir {args.x5_work} (created by controller)")

    # Prepare all pools first (quick on CPU).
    pools = prepare_pools(c, args, models)

    # Start evolutions with limited concurrency.
    jobs = []
    running = []
    for model in models:
        # Wait if concurrency limit reached.
        while len(running) >= args.max_concurrent:
            time.sleep(args.poll_interval)
            running = [(m, p, l) for (m, p, l) in running if is_running(c, p)]
            if not args.wait:
                break

        pid, log = start_evolution(c, model, pools[model], args)
        print(f"[start] {model} PID={pid} log={log}")
        jobs.append({"model": model, "pid": pid, "log": log})
        running.append((model, pid, log))

    if not args.wait:
        print("\n[done] started the following evolutions:")
        for j in jobs:
            print(f"  {j['model']:20s} PID={j['pid']:8s} log={j['log']}")
        print("\nTo monitor a log:  ssh kali cat /tmp/bpu_evo_<model>.log")
        return 0

    # Wait for all to finish.
    finished = set()
    while len(finished) < len(jobs):
        time.sleep(args.poll_interval)
        for j in jobs:
            if j["model"] in finished:
                continue
            if not is_running(c, j["pid"]):
                print(f"\n[done] {j['model']} finished")
                print(tail_log(c, j["log"], n=20))
                finished.add(j["model"])
            else:
                print(f"[status] {j['model']} still running, last log lines:")
                print(tail_log(c, j["log"], n=10))

    print("\n[all done]")
    for j in jobs:
        print(f"  {j['model']}: {j['log']}")
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

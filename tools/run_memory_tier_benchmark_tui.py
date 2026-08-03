import argparse
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from memory_tier_benchmark_v1 import start_local_phoenix_stack

SCENARIO_ORDER = [
    "short_dialogue",
    "long_dialogue_5_15",
    "ultra_long_dialogue_15_plus",
    "cross_session",
]

RUN_RE = re.compile(r"^\[run\] provider=(?P<provider>\S+) scenario=(?P<scenario>\S+)")
PROGRESS_RE = re.compile(
    r"^\[progress\] (?P<provider>\S+) (?P<scenario>\S+) (?P<idx>\d+)/(?P<total>\d+) benchmarkRequests=(?P<requests>\d+)"
)
ACTIVITY_RE = re.compile(
    r"^\[activity\] provider=(?P<provider>\S+) scenario=(?P<scenario>\S+) sample=(?P<idx>\d+)/(?P<total>\d+) step=(?P<step>\S+) benchmarkRequests=(?P<requests>\d+)"
)


@dataclass
class LiveState:
    provider: str = ""
    provider_index: int = 0
    provider_total: int = 0
    round_index: int = 0
    rounds_total: int = 0
    scenario: str = ""
    scenario_index: int = 0
    scenario_total: int = 0
    scenario_sample_index: int = 0
    scenario_sample_total: int = 0
    run_completed_tests: int = 0
    run_total_tests: int = 0
    total_completed_tests: int = 0
    total_tests: int = 0
    scenario_benchmark_requests: int = 0
    run_completed_benchmark_requests: int = 0
    current_step: str = "-"
    last_progress_ts: float = 0.0
    last_activity_ts: float = 0.0
    last_output_ts: float = 0.0
    last_line: str = ""
    status: str = "idle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory-tier benchmark with a 1 FPS TUI monitor")
    parser.add_argument("--sample-per-scenario", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--providers", default="phoenix,llama_server")
    parser.add_argument(
        "--scenarios",
        default="short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus,cross_session",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--warmup-timeout", type=float, default=120.0)
    parser.add_argument("--warmup-retries", type=int, default=3)
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--llama-threads", type=int, default=8, help="llama-server -t (default 8)")
    parser.add_argument("--llama-parallel", type=int, default=1, help="llama-server --parallel (default 1)")
    parser.add_argument("--llama-ctx-size", type=int, default=4096, help="llama-server -c (default 4096)")
    parser.add_argument("--seed-base", type=int, default=20260605)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--stall-seconds", type=int, default=180)
    parser.add_argument("--out-prefix", default="memory_tier_benchmark_v1_no_ollama_tui")
    parser.add_argument("--phoenix-url", default="http://127.0.0.1:5080/api/chat")
    parser.add_argument("--phoenix-token", default="local-dev")
    parser.add_argument(
        "--similarity-mode",
        default="hybrid",
        choices=["bow", "sentence", "hybrid"],
        help="semantic similarity metric passed to memory_tier_benchmark_v1.py",
    )
    parser.add_argument(
        "--sentence-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformer model name passed to memory_tier_benchmark_v1.py",
    )
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def provider_seed(provider: str, seed_base: int, round_index: int) -> int:
    # Deterministic per-provider/per-round seed, no arithmetic from shell required.
    mapping = {
        "phoenix": 1,
        "llama_server": 2,
    }
    slot = mapping.get(provider)
    if slot is None:
        raise ValueError(f"unsupported provider: {provider}")
    return seed_base * 100 + slot * 10 + round_index


def enqueue_output(pipe, out_q: queue.Queue[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            out_q.put(line.rstrip("\n"))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def scenario_index(scenarios: list[str], name: str) -> int:
    if name in scenarios:
        return scenarios.index(name) + 1
    return 0


def seconds_since(ts: float) -> str:
    if not ts:
        return "N/A"
    return str(int(time.time() - ts))


def is_stalled(state: LiveState, stall_seconds: int, process_running: bool) -> bool:
    if not process_running or not state.last_activity_ts:
        return False
    return (time.time() - state.last_activity_ts) >= stall_seconds


def is_tcp_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def render(state: LiveState, out_dir: Path, log_path: Path, process_running: bool, stall_seconds: int) -> None:
    clear_screen()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    stalled = is_stalled(state, stall_seconds, process_running)
    run_cumulative_benchmark_requests = state.run_completed_benchmark_requests + state.scenario_benchmark_requests

    print("Memory Tier Benchmark TUI (1 FPS)")
    print("=" * 64)
    print(f"Time: {now}")
    print(f"Status: {state.status}")
    print(f"Out Dir: {out_dir}")
    print(f"Run Log: {log_path}")
    print("-" * 64)
    print(f"Model: {state.provider_index}/{state.provider_total} ({state.provider or '-'})")
    print(f"Round: {state.round_index}/{state.rounds_total}")
    print(
        f"Scenario: {state.scenario_index}/{state.scenario_total} ({state.scenario or '-'})"
    )
    print(
        f"Test In Scenario: {state.scenario_sample_index}/{state.scenario_sample_total}"
    )
    print(f"Completed In Current Run: {state.run_completed_tests}/{state.run_total_tests}")
    print(f"Total Completed: {state.total_completed_tests}/{state.total_tests}")
    print(f"Scenario Benchmark Requests Sent: {state.scenario_benchmark_requests}")
    print(f"Run Cumulative Benchmark Requests Sent: {run_cumulative_benchmark_requests}")
    print(f"Current Step: {state.current_step}")
    print(f"Seconds Since Last Completed Sample: {seconds_since(state.last_progress_ts)}")
    print(f"Seconds Since Last Activity: {seconds_since(state.last_activity_ts)}")
    print(f"Seconds Since Last Output: {seconds_since(state.last_output_ts)}")
    print(f"Potential Stall: {'YES' if stalled else 'NO'}")
    print("-" * 64)
    print(f"Last Line: {state.last_line[:180]}")
    if stalled:
        print(f"Warning: no request-level activity for >= {stall_seconds}s")
    elif not process_running:
        print("Process finished. Waiting to start next run...")


def interrupt_child(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def stop_processes(processes: list[subprocess.Popen[str]]) -> None:
    for proc in reversed(processes):
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    deadline = time.time() + 10.0
    for proc in reversed(processes):
        if proc.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def main() -> int:
    args = parse_args()

    providers = split_csv(args.providers)
    scenarios = split_csv(args.scenarios)

    for p in providers:
        if p not in {"phoenix", "llama_server"}:
            raise SystemExit("providers must be subset of phoenix,llama_server")
    for s in scenarios:
        if s not in SCENARIO_ORDER:
            raise SystemExit(
                "scenarios must be subset of short_dialogue,long_dialogue_5_15,ultra_long_dialogue_15_plus,cross_session"
            )

    if args.fps <= 0:
        raise SystemExit("fps must be > 0")
    if args.stall_seconds <= 0:
        raise SystemExit("stall-seconds must be > 0")

    root = Path(__file__).resolve().parents[1]
    py_exe = root / ".venv" / "Scripts" / "python.exe"
    if not py_exe.exists():
        py_exe = Path(sys.executable)

    bench_script = root / "tools" / "memory_tier_benchmark_v1.py"
    if not bench_script.exists():
        raise SystemExit(f"missing benchmark script: {bench_script}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = root / "build" / f"{args.out_prefix}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    expected_per_run = args.sample_per_scenario * len(scenarios)
    total_runs = len(providers) * args.rounds
    total_tests = expected_per_run * total_runs
    interval = 1.0 / args.fps

    state = LiveState(
        provider_total=len(providers),
        rounds_total=args.rounds,
        scenario_total=len(scenarios),
        run_total_tests=expected_per_run,
        total_tests=total_tests,
        status="starting",
    )

    active_proc: subprocess.Popen[str] | None = None
    shared_stack_processes: list[subprocess.Popen[str]] = []
    shared_stack_log_handles: list[object] = []
    shared_stack_ready = False
    failed_runs: list[tuple[str, int, int]] = []

    with log_path.open("a", encoding="utf-8", errors="replace") as run_log:
        run_log.write(f"[INFO] started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        run_log.write(f"[INFO] providers={providers} scenarios={scenarios}\n")
        run_log.write(f"[INFO] sample_per_scenario={args.sample_per_scenario} rounds={args.rounds}\n")
        run_log.flush()
        try:
            if "phoenix" in providers or "llama_server" in providers:
                phoenix_ready_before = is_tcp_port_open("127.0.0.1", 5080)
                llama_ready_before = is_tcp_port_open("127.0.0.1", 8083)
                shared_stack_processes, shared_stack_log_handles = start_local_phoenix_stack(
                    root,
                    8083,
                    5080,
                    out_dir / "stack_logs",
                    llama_threads=args.llama_threads,
                    llama_parallel=args.llama_parallel,
                    llama_ctx_size=args.llama_ctx_size,
                )
                shared_stack_ready = True
                if shared_stack_processes:
                    run_log.write("[INFO] started shared local phoenix stack for this TUI run\n")
                elif phoenix_ready_before and llama_ready_before:
                    run_log.write("[INFO] reusing existing local phoenix stack\n")
                else:
                    run_log.write("[INFO] attached to partially pre-existing local stack and launched missing processes\n")
                run_log.flush()

            for p_idx, provider in enumerate(providers, start=1):
                state.provider = provider
                state.provider_index = p_idx

                for r_idx in range(1, args.rounds + 1):
                    state.round_index = r_idx
                    state.scenario = ""
                    state.scenario_index = 0
                    state.scenario_sample_index = 0
                    state.scenario_sample_total = args.sample_per_scenario
                    state.run_completed_tests = 0
                    state.scenario_benchmark_requests = 0
                    state.run_completed_benchmark_requests = 0
                    state.current_step = "starting_round"
                    state.last_progress_ts = 0.0
                    state.last_activity_ts = 0.0
                    state.last_output_ts = 0.0
                    state.status = f"running provider={provider} round={r_idx}"

                    seed = provider_seed(provider, args.seed_base, r_idx)
                    json_out = out_dir / f"memory_tier_benchmark_v1_{provider}_round{r_idx}.json"
                    md_out = out_dir / f"memory_tier_benchmark_v1_{provider}_round{r_idx}.md"

                    cmd = [
                        str(py_exe),
                        "-u",
                        str(bench_script),
                        "--providers",
                        provider,
                        "--scenarios",
                        ",".join(scenarios),
                        "--sample-per-scenario",
                        str(args.sample_per_scenario),
                        "--seed",
                        str(seed),
                        "--context-window",
                        str(args.context_window),
                        "--timeout",
                        str(args.timeout),
                        "--warmup-timeout",
                        str(args.warmup_timeout),
                        "--warmup-retries",
                        str(args.warmup_retries),
                        "--no-use-cache",
                        "--json-output",
                        str(json_out.relative_to(root)),
                        "--md-output",
                        str(md_out.relative_to(root)),
                    ]
                    if shared_stack_ready:
                        cmd.append("--no-launch-local-stack")
                    cmd.extend(["--phoenix-url", args.phoenix_url])
                    cmd.extend(["--phoenix-token", args.phoenix_token])
                    cmd.extend(["--similarity-mode", args.similarity_mode])
                    cmd.extend(["--sentence-model", args.sentence_model])

                    run_log.write(f"[INFO] command: {' '.join(cmd)}\n")
                    run_log.flush()

                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
                    active_proc = subprocess.Popen(
                        cmd,
                        cwd=str(root),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        creationflags=creationflags,
                    )

                    out_q: queue.Queue[str] = queue.Queue()
                    reader = threading.Thread(target=enqueue_output, args=(active_proc.stdout, out_q), daemon=True)
                    reader.start()

                    while True:
                        while True:
                            try:
                                line = out_q.get_nowait()
                            except queue.Empty:
                                break

                            now_ts = time.time()
                            state.last_line = line
                            state.last_output_ts = now_ts
                            run_log.write(line + "\n")

                            run_match = RUN_RE.match(line)
                            if run_match:
                                scenario_name = run_match.group("scenario")
                                if state.scenario and scenario_name != state.scenario:
                                    state.run_completed_benchmark_requests += state.scenario_benchmark_requests
                                state.scenario = scenario_name
                                state.scenario_index = scenario_index(scenarios, scenario_name)
                                state.scenario_sample_index = 0
                                state.scenario_sample_total = args.sample_per_scenario
                                state.scenario_benchmark_requests = 0
                                state.current_step = "starting_scenario"

                            activity_match = ACTIVITY_RE.match(line)
                            if activity_match:
                                scenario_name = activity_match.group("scenario")
                                state.scenario = scenario_name
                                state.scenario_index = scenario_index(scenarios, scenario_name)
                                state.scenario_sample_index = int(activity_match.group("idx"))
                                state.scenario_sample_total = int(activity_match.group("total"))
                                state.current_step = activity_match.group("step")
                                state.scenario_benchmark_requests = int(activity_match.group("requests"))
                                state.last_activity_ts = now_ts

                            progress_match = PROGRESS_RE.match(line)
                            if progress_match:
                                scenario_name = progress_match.group("scenario")
                                sample_idx = int(progress_match.group("idx"))
                                sample_total = int(progress_match.group("total"))
                                reqs = int(progress_match.group("requests"))

                                state.scenario = scenario_name
                                state.scenario_index = scenario_index(scenarios, scenario_name)
                                state.scenario_sample_index = sample_idx
                                state.scenario_sample_total = sample_total
                                state.scenario_benchmark_requests = reqs
                                state.current_step = "sample_complete"

                                completed_before = max(0, state.scenario_index - 1) * args.sample_per_scenario
                                state.run_completed_tests = completed_before + sample_idx
                                offset_runs = ((p_idx - 1) * args.rounds + (r_idx - 1)) * expected_per_run
                                state.total_completed_tests = offset_runs + state.run_completed_tests
                                state.last_progress_ts = now_ts
                                state.last_activity_ts = now_ts

                        run_log.flush()

                        running = active_proc.poll() is None
                        render(state, out_dir, log_path, running, args.stall_seconds)

                        if not running and out_q.empty():
                            break

                        time.sleep(interval)

                    code = active_proc.wait()
                    active_proc = None
                    if code != 0:
                        state.status = f"failed provider={provider} round={r_idx} exit={code} (continuing)"
                        render(state, out_dir, log_path, False, args.stall_seconds)
                        run_log.write(f"[ERROR] provider={provider} round={r_idx} exit={code} — continuing with remaining providers/rounds\n")
                        run_log.flush()
                        failed_runs.append((provider, r_idx, code))
                        time.sleep(1)
                        continue

                    state.run_completed_tests = expected_per_run
                    offset_runs = ((p_idx - 1) * args.rounds + (r_idx - 1)) * expected_per_run
                    state.total_completed_tests = offset_runs + expected_per_run
                    state.current_step = "run_complete"
                    state.status = f"completed provider={provider} round={r_idx}"
                    render(state, out_dir, log_path, False, args.stall_seconds)
                    time.sleep(1)

            state.status = "all runs completed"
            state.total_completed_tests = total_tests
            state.current_step = "all_complete"
            if failed_runs:
                state.status = f"completed with {len(failed_runs)} failed run(s)"
                for fp, fr, fc in failed_runs:
                    run_log.write(f"[SUMMARY] FAILED provider={fp} round={fr} exit={fc}\n")
                run_log.write(f"[SUMMARY] {len(failed_runs)} run(s) failed out of {total_runs}\n")
            render(state, out_dir, log_path, False, args.stall_seconds)
            run_log.write("[OK] all runs completed\n")
            run_log.flush()
            return 1 if failed_runs else 0
        except KeyboardInterrupt:
            state.status = "interrupting_child"
            run_log.write("[WARN] keyboard interrupt received; forwarding to child\n")
            run_log.flush()
            interrupt_child(active_proc)

            deadline = time.time() + 10.0
            while active_proc is not None and active_proc.poll() is None and time.time() < deadline:
                render(state, out_dir, log_path, True, args.stall_seconds)
                time.sleep(interval)

            if active_proc is not None and active_proc.poll() is None:
                try:
                    active_proc.kill()
                except Exception:
                    pass

            state.status = "interrupted_by_user"
            render(state, out_dir, log_path, False, args.stall_seconds)
            run_log.write("[WARN] interrupted by user\n")
            run_log.flush()
            return 130
        finally:
            stop_processes(shared_stack_processes)
            for handle in shared_stack_log_handles:
                try:
                    handle.close()
                except Exception:
                    pass


if __name__ == "__main__":
    raise SystemExit(main())

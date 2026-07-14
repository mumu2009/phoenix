import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory_tier_benchmark_v1 import start_local_phoenix_stack, workspace_root, wait_tcp_port

root = workspace_root()
print("[test] starting stack...")
procs, handles = start_local_phoenix_stack(root, 8083, 5080, root / "build" / "stack_logs_test")
print("[test] stack started, waiting for port 5080...")
ok = wait_tcp_port("127.0.0.1", 5080, 120)
print(f"[test] port open: {ok}")
time.sleep(3)
print("[test] ready")

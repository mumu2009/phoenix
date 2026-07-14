"""无 GUI 启动 phoenix_main.exe（复用 start_079_launcher 的命令构建与进程启动）。

用法：
    python tools/headless_launch.py

读取 runtime_store/start_079_launcher.json，按其中配置（含 transformer_mode=llamacpp、
external_auto_launch=true）启动 phoenix_main.exe 及其外部 llamacpp 适配器。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from start_079_launcher import (  # noqa: E402
    detect_repo_root,
    load_profile,
    build_launch_command,
    launch_process,
)


def main() -> int:
    root = detect_repo_root()
    options = load_profile(root)
    command = build_launch_command(options, root)
    print("[headless] root:", root)
    print("[headless] command:", " ".join(command))
    process, _ = launch_process(root, options)
    print(f"[headless] launched phoenix_main.exe pid={process.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

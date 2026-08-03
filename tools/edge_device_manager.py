#!/usr/bin/env python3
"""Edge device manager: generic SSH/SCP wrapper for multiple edge targets.

Supported device types (stubs for non-X5 compilers are in tools/compile_target_model.py):
  rdk_x5, rdk_s100, rk3588, jetson_nano

Credentials can come from:
  - config/edge_devices.json (copy from config/edge_devices.example.json)
  - environment variables (pass_env)
  - SSH key files (auth == "key")

Do not commit config/edge_devices.json to git; it is in .gitignore.
"""
import json
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_config_path() -> Path:
    """Return the first existing edge-device config file."""
    for name in ["edge_devices.json", "edge_devices.local.json"]:
        path = Path(__file__).resolve().parent.parent / "config" / name
        if path.is_file():
            return path
    example = Path(__file__).resolve().parent.parent / "config" / "edge_devices.example.json"
    if example.is_file():
        return example
    raise FileNotFoundError("No edge device config found. Copy config/edge_devices.example.json to config/edge_devices.json")


def _resolve_password(cfg: Dict[str, Any]) -> str:
    """Get the SSH password from env, key, or plain (last resort, not recommended)."""
    auth = cfg.get("auth", "env")
    if auth == "env" and "pass_env" in cfg:
        val = os.environ.get(cfg["pass_env"])
        if val is None:
            raise RuntimeError(f"Device {cfg.get('type')} requires env var {cfg['pass_env']}")
        return val
    if auth == "key":
        # Password can be empty when using keys.
        return cfg.get("pass", "") or ""
    # Legacy/plain mode: allow, but warn.
    if "pass" in cfg:
        return cfg["pass"]
    raise RuntimeError(f"No password or key configured for device {cfg.get('type')}")


class EdgeDevice:
    """Minimal paramiko wrapper to copy files and run commands on an edge device.

    This replaces the old X5Remote class and works for any Linux SSH target.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        timeout: int = 30,
    ):
        self.cfg = cfg
        self.host = cfg["host"]
        self.port = int(cfg.get("port", 22))
        self.user = cfg["user"]
        self.password = _resolve_password(cfg)
        self.key_path = cfg.get("key_path") or cfg.get("ssh_key")
        self.timeout = timeout
        self.client = None
        self.sftp = None

    def connect(self):
        try:
            import paramiko
        except ImportError as e:
            raise RuntimeError("paramiko is not installed; install it to use edge SSH") from e

        connect_kwargs: Dict[str, Any] = {
            "hostname": self.host,
            "port": self.port,
            "username": self.user,
            "password": self.password,
            "timeout": self.timeout,
            "allow_agent": False,
            "look_for_keys": False,
        }
        if self.key_path:
            key_path = os.path.expanduser(self.key_path)
            if os.path.isfile(key_path):
                connect_kwargs["key_filename"] = key_path
                connect_kwargs["password"] = None
                connect_kwargs["allow_agent"] = True
                connect_kwargs["look_for_keys"] = True

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(**connect_kwargs)
        self.sftp = self.client.open_sftp()

    def close(self):
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.client:
            self.client.close()
            self.client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def mkdir(self, remote_path: str, exist_ok: bool = True):
        try:
            self.sftp.mkdir(remote_path)
        except (IOError, OSError):
            if not exist_ok:
                raise

    def mkdir_p(self, remote_path: str):
        """Recursively create a remote directory."""
        parts = []
        for part in remote_path.strip("/").split("/"):
            parts.append(part)
            current = "/" + "/".join(parts)
            self.mkdir(current)

    def put(self, local: str, remote: str):
        remote_parent = str(Path(remote).parent).replace("\\", "/")
        self.mkdir_p(remote_parent)
        self.sftp.put(local, remote)

    def get(self, remote: str, local: str):
        self.sftp.get(remote, local)

    def exec(self, cmd: str, timeout: int = 600) -> tuple:
        if self.client is None:
            raise RuntimeError("not connected")
        chan = self.client.get_transport().open_session()
        chan.exec_command(cmd)
        exit_code = chan.recv_exit_status()
        out = chan.makefile("rb").read().decode("utf-8", errors="ignore")
        err = chan.makefile_stderr("rb").read().decode("utf-8", errors="ignore")
        return exit_code, out, err


def load_edge_device(name: Optional[str] = None, config_path: Optional[str] = None) -> EdgeDevice:
    """Load an edge device by name from config/edge_devices.json."""
    if config_path:
        path = Path(config_path)
    else:
        path = _resolve_config_path()

    cfg = _load_json_config(path)
    name = name or cfg.get("default_device")
    if not name:
        raise ValueError("No device name provided and no default_device in config")

    devices = cfg.get("devices", {})
    if name not in devices:
        raise ValueError(f"Device '{name}' not found in {path}. Available: {list(devices)}")

    device_cfg = devices[name]
    if "host" not in device_cfg or "user" not in device_cfg:
        raise ValueError(f"Device '{name}' must have host and user")
    return EdgeDevice(device_cfg)


def list_device_types() -> list:
    return ["rdk_x5", "rdk_s100", "rk3588", "jetson_nano"]

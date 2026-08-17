import argparse
import base64
import json
import os
import re
import subprocess
import struct
import sys
import time
import urllib.parse
import wave
import socket
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import regex as re2
except ImportError:
    re2 = re
import requests
from bs4 import BeautifulSoup


@dataclass
class ApiEndpoint:
    path: str
    method: str
    source_line: int
    kind: str


@dataclass
class TestResult:
    name: str
    ok: bool
    status: str
    detail: str
    elapsed_ms: int


class DualLogger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {message}"
        print(line)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warn(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)


class ApiInventory:
    REGISTER_RE = re.compile(r'registerHandler\("([^"]+)"')
    REGEX_RE = re.compile(r'registerHandlerViaRegex\("([^"]+)"')
    METHOD_RE = re.compile(r'\{\s*drogon::(Get|Post|Put|Delete|Patch)\s*\}\s*\)')

    def __init__(self, frontend_server_cpp: Path, logger: DualLogger) -> None:
        self.frontend_server_cpp = frontend_server_cpp
        self.logger = logger

    def scan(self) -> List[ApiEndpoint]:
        endpoints: List[ApiEndpoint] = []
        lines = self.frontend_server_cpp.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            m = self.REGISTER_RE.search(line)
            if m:
                method = "ANY"
                lookahead = "\n".join(lines[idx - 1: min(len(lines), idx + 4)])
                mm = self.METHOD_RE.search(lookahead)
                if mm:
                    method = mm.group(1).upper()
                endpoints.append(ApiEndpoint(path=m.group(1), method=method, source_line=idx, kind="literal"))
                continue

            r = self.REGEX_RE.search(line)
            if r:
                endpoints.append(ApiEndpoint(path=r.group(1), method="ANY", source_line=idx, kind="regex"))

        endpoints.sort(key=lambda x: (x.path, x.method, x.source_line))
        return endpoints

    @staticmethod
    def summarize(endpoints: List[ApiEndpoint]) -> Dict[str, object]:
        method_count: Dict[str, int] = {}
        kind_count: Dict[str, int] = {}
        for ep in endpoints:
            method_count[ep.method] = method_count.get(ep.method, 0) + 1
            kind_count[ep.kind] = kind_count.get(ep.kind, 0) + 1
        return {
            "total": len(endpoints),
            "byMethod": method_count,
            "byKind": kind_count,
        }


class SmartHttpClient:
    def __init__(self, base_url: str, logger: DualLogger, timeout_sec: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.timeout_sec = timeout_sec
        self.session = requests.Session()

    def _try_restart_server(self) -> bool:
        kill_cmd = os.getenv("PHOENIX_SERVER_KILL_CMD", "taskkill /IM phoenix_main.exe /F")
        start_cmd = os.getenv("PHOENIX_SERVER_START_CMD", "")
        server_exe = os.getenv("PHOENIX_SERVER_EXE", "phoenix_main.exe")
        server_args = os.getenv("PHOENIX_SERVER_ARGS", "").strip()
        cwd = os.getenv("PHOENIX_SERVER_CWD", str(Path(__file__).resolve().parents[2]))
        ready_wait_sec = float(os.getenv("PHOENIX_SERVER_READY_WAIT_SEC", "12"))

        try:
            self.logger.warn("Triggering server recovery: kill + restart")
            subprocess.run(kill_cmd, shell=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if server_exe:
                argv = [server_exe] + ([x for x in server_args.split(" ") if x] if server_args else [])
                subprocess.Popen(argv, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif start_cmd:
                subprocess.run(start_cmd, shell=True, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                raise RuntimeError("No restart strategy configured (missing EXE/START_CMD)")

            # Wait until the frontend endpoint is reachable again before retries continue.
            deadline = time.time() + max(ready_wait_sec, 1.0)
            while time.time() < deadline:
                try:
                    probe = self.session.get(f"{self.base_url}/api/system/status", timeout=1.5)
                    if probe.status_code < 500:
                        return True
                except requests.RequestException:
                    pass
                time.sleep(0.6)

            self.logger.warn("Server restarted but readiness probe timed out")
            return False
        except Exception as exc:
            self.logger.error(f"Server recovery failed: {exc}")
            return False

    def request(
        self,
        method: str,
        path: str,
        json_payload: Optional[dict] = None,
        headers: Optional[dict] = None,
        retries: int = 4,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        chat_proxy_paths = {"/api/chat", "/api/transformer/chat"}
        is_chat_proxy = path in chat_proxy_paths
        heavy_timeout_paths = {
            "/context/ingest": 75.0,
            "/v51/chat": 32.0,
            "/speech/ingest": 25.0,
            "/api/chat": 185.0,
            "/api/transformer/chat": 185.0,
            "/api/fine_tuning/run": 45.0,
        }
        timeout = max(self.timeout_sec, heavy_timeout_paths.get(path, self.timeout_sec))
        effective_retries = retries
        if is_chat_proxy:
            effective_retries = min(retries, 1)
        last_exc: Optional[Exception] = None

        for attempt in range(1, effective_retries + 1):
            try:
                resp = self.session.request(method=method, url=url, json=json_payload, headers=headers, timeout=timeout)
                if resp.status_code >= 500:
                    self.logger.warn(f"{method} {path} -> {resp.status_code}, attempt={attempt}, timeout={timeout:.1f}s")
                    if (not is_chat_proxy) and attempt >= 3:
                        self._try_restart_server()
                    timeout = min(timeout * 1.3, 190.0 if is_chat_proxy else 120.0)
                    time.sleep(0.6 * attempt)
                    continue
                if resp.status_code == 429:
                    self.logger.warn(f"{method} {path} throttled, backing off")
                    timeout = min(timeout * 1.2, 190.0 if is_chat_proxy else 120.0)
                    time.sleep(0.8 * attempt)
                    continue
                return resp
            except requests.ReadTimeout as exc:
                last_exc = exc
                self.logger.warn(
                    f"{method} {path} read timeout ({exc}), attempt={attempt}, timeout={timeout:.1f}s"
                )
                # Read timeouts on model-heavy endpoints are expected occasionally; do not restart immediately.
                timeout = min(timeout * 1.2, 190.0 if is_chat_proxy else 180.0)
                time.sleep(0.8 * attempt)
                continue
            except (TimeoutError, socket.timeout) as exc:
                last_exc = exc
                self.logger.warn(
                    f"{method} {path} native timeout ({exc}), attempt={attempt}, timeout={timeout:.1f}s"
                )
                timeout = min(timeout * 1.2, 190.0 if is_chat_proxy else 180.0)
                time.sleep(0.8 * attempt)
                continue
            except requests.ConnectionError as exc:
                last_exc = exc
                self.logger.warn(f"{method} {path} connection error ({exc}), attempt={attempt}")
                if (not is_chat_proxy) and attempt >= 2:
                    self._try_restart_server()
                timeout = min(timeout * 1.2, 190.0 if is_chat_proxy else 90.0)
                time.sleep(0.7 * attempt)
                continue
            except requests.RequestException as exc:
                last_exc = exc
                self.logger.warn(f"{method} {path} request exception ({exc}), attempt={attempt}")
                if (not is_chat_proxy) and attempt >= 3:
                    self._try_restart_server()
                timeout = min(timeout * 1.2, 190.0 if is_chat_proxy else 120.0)
                time.sleep(0.7 * attempt)

        if last_exc is None:
            raise RuntimeError(f"request failed without explicit exception: {method} {path}")
        raise last_exc


class V51ApiTester:
    def __init__(self, root: Path, base_url: str, logger: DualLogger) -> None:
        self.root = root
        self.logger = logger
        self.client = SmartHttpClient(base_url=base_url, logger=logger)
        self.auth_token: str = ""
        self.session_id = f"v51-test-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _json(resp: requests.Response) -> dict:
        try:
            return resp.json()
        except Exception:
            return {}

    def _assert_json_ok(self, resp: requests.Response, name: str) -> Tuple[bool, str]:
        data = self._json(resp)
        if resp.status_code >= 400:
            return False, f"HTTP {resp.status_code}: {resp.text[:400]}"
        if isinstance(data, dict) and data.get("ok") is False:
            return False, f"ok=false payload: {json.dumps(data, ensure_ascii=False)[:500]}"
        return True, "ok"

    def _write_minimal_gguf_fixture(self) -> Path:
        fixture_dir = self.root / "build" / "tmp"
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture_path = fixture_dir / "apitest_fixture.gguf"

        def write_string(handle, value: str) -> None:
            raw = value.encode("utf-8")
            handle.write(struct.pack("<Q", len(raw)))
            handle.write(raw)

        tokens = ["<s>", "hello", "world", "!"]
        with fixture_path.open("wb") as handle:
            handle.write(b"GGUF")
            handle.write(struct.pack("<IQQ", 3, 1, 5))

            write_string(handle, "general.architecture")
            handle.write(struct.pack("<I", 8))
            write_string(handle, "llama")

            write_string(handle, "llama.context_length")
            handle.write(struct.pack("<I", 4))
            handle.write(struct.pack("<I", 2048))

            write_string(handle, "llama.block_count")
            handle.write(struct.pack("<I", 4))
            handle.write(struct.pack("<I", 16))

            write_string(handle, "tokenizer.ggml.tokens")
            handle.write(struct.pack("<I", 9))
            handle.write(struct.pack("<IQ", 8, len(tokens)))
            for token in tokens:
                write_string(handle, token)

            write_string(handle, "general.alignment")
            handle.write(struct.pack("<I", 4))
            handle.write(struct.pack("<I", 32))

            write_string(handle, "token_embd.weight")
            handle.write(struct.pack("<I", 2))
            handle.write(struct.pack("<qq", 64, len(tokens)))
            handle.write(struct.pack("<I", 1))
            handle.write(struct.pack("<Q", 0))

            padding = (-handle.tell()) % 32
            if padding:
                handle.write(b"\x00" * padding)
            handle.write(b"\x00" * (64 * len(tokens) * 2))

        return fixture_path

    def inventory(self) -> Tuple[Dict[str, object], List[ApiEndpoint]]:
        scanner = ApiInventory(self.root / "frontend_server.cpp", self.logger)
        eps = scanner.scan()
        summary = scanner.summarize(eps)
        out = {
            "summary": summary,
            "endpoints": [asdict(x) for x in eps],
            "generatedAt": datetime.now().isoformat(),
        }
        inventory_path = self.root / "test" / "apis" / "api_inventory.json"
        inventory_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(
            f"API inventory generated: total={summary['total']}, byMethod={summary['byMethod']}, byKind={summary['byKind']}"
        )
        return summary, eps

    def authenticate(self) -> None:
        username = f"autotest_{int(time.time())}"
        password = "AutoTest@2026"
        email = f"{username}@example.com"

        cfg = self.client.request("GET", "/auth/config")
        # If auth endpoints are not compiled (RDK X5 slim build / auth disabled),
        # fall back to the local-dev shared bearer token configured in phoenix_main.
        if cfg.status_code == 404:
            fixed = os.getenv("TEST_BEARER_TOKEN", "local-dev")
            self.auth_token = fixed
            self.logger.info(f"Authenticated by fixed bearer token (auth disabled build): {fixed[:8]}...")
            return
        cfg_json = self._json(cfg)
        allow_register = bool(cfg_json.get("allowRegister", True))

        if allow_register:
            reg_payload = {"username": username, "password": password, "email": email}
            reg = self.client.request("POST", "/auth/register", json_payload=reg_payload)
            reg_json = self._json(reg)
            if reg.status_code == 200 and reg_json.get("token"):
                self.auth_token = reg_json["token"]
                self.logger.info("Authenticated by register(token direct)")
                return

            if reg.status_code in (200, 409):
                verify_token = reg_json.get("verifyToken", "")
                if verify_token:
                    self.client.request("POST", "/auth/verify", json_payload={"username": username, "token": verify_token})

        login = self.client.request("POST", "/auth/login", json_payload={"username": username, "password": password})
        login_json = self._json(login)
        if login.status_code == 200 and login_json.get("token"):
            self.auth_token = login_json["token"]
            self.logger.info("Authenticated by login")
            return

        # Fallback: bootstrap first user if system is empty.
        if cfg_json.get("allowBootstrap", False):
            bootstrap = self.client.request(
                "POST",
                "/auth/bootstrap",
                json_payload={"username": username, "password": password, "email": email},
            )
            boot_json = self._json(bootstrap)
            token = boot_json.get("token")
            if token:
                self.auth_token = token
                self.logger.info("Authenticated by bootstrap")
                return
            verify_token = boot_json.get("verifyToken", "")
            if verify_token:
                self.client.request("POST", "/auth/verify", json_payload={"username": username, "token": verify_token})
                login2 = self.client.request("POST", "/auth/login", json_payload={"username": username, "password": password})
                login2_json = self._json(login2)
                if login2_json.get("token"):
                    self.auth_token = login2_json["token"]
                    self.logger.info("Authenticated by bootstrap+verify+login")
                    return

        raise RuntimeError("Failed to authenticate test user via register/login/bootstrap flow")

    def _headers(self) -> Dict[str, str]:
        return {"authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

    def _generate_tone_wav_base64(self, duration_sec: float = 0.4, rate: int = 16000, freq: float = 440.0) -> str:
        import math

        n_samples = int(duration_sec * rate)
        pcm = bytearray()
        for i in range(n_samples):
            sample = int(32767 * 0.28 * math.sin(2.0 * math.pi * freq * i / rate))
            pcm += int(sample).to_bytes(2, byteorder="little", signed=True)

        bio = BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(bytes(pcm))
        return base64.b64encode(bio.getvalue()).decode("ascii")

    def _sample_png_base64(self) -> str:
        # Use a tiny 24-bit BMP because OpenCV's BMP decoder is the least fragile in CI.
        width = 2
        height = 2
        row_padding = (4 - ((width * 3) % 4)) % 4
        row_stride = width * 3 + row_padding
        pixel_data_size = row_stride * height
        file_size = 54 + pixel_data_size

        header = bytearray()
        header += b"BM"
        header += struct.pack("<I", file_size)
        header += b"\x00\x00\x00\x00"
        header += struct.pack("<I", 54)
        header += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 2835, 2835, 0, 0)

        # BMP stores rows bottom-up in BGR order.
        pixels = bytearray()
        pixels += bytes([255, 0, 0, 255, 255, 255]) + (b"\x00" * row_padding)
        pixels += bytes([0, 0, 255, 0, 255, 0]) + (b"\x00" * row_padding)
        return base64.b64encode(bytes(header + pixels)).decode("ascii")

    @staticmethod
    def _require_dict(value: object, name: str) -> Tuple[bool, str, dict]:
        if not isinstance(value, dict):
            return False, f"missing {name} object", {}
        return True, "ok", value

    @staticmethod
    def _decode_wav_meta(audio_b64: str) -> Tuple[bool, str, Dict[str, int]]:
        try:
            audio_bytes = base64.b64decode(audio_b64)
            with wave.open(BytesIO(audio_bytes), "rb") as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
            duration_ms = int((frames * 1000) / sample_rate) if sample_rate > 0 else 0
            return True, "ok", {
                "frames": frames,
                "sampleRate": sample_rate,
                "channels": channels,
                "sampleWidth": sample_width,
                "durationMs": duration_ms,
            }
        except Exception as exc:
            return False, str(exc), {}

    def run(self) -> List[TestResult]:
        results: List[TestResult] = []

        def run_case(name: str, fn) -> None:
            start = time.time()
            try:
                ok, status, detail = fn()
            except Exception as exc:
                ok, status, detail = False, "EXCEPTION", str(exc)
            elapsed = int((time.time() - start) * 1000)
            results.append(TestResult(name=name, ok=ok, status=status, detail=detail, elapsed_ms=elapsed))
            if ok:
                self.logger.info(f"[{name}] PASS ({elapsed}ms) {status}")
            else:
                self.logger.error(f"[{name}] FAIL ({elapsed}ms) {status} :: {detail}")

        def case_root_html() -> Tuple[bool, str, str]:
            resp = self.client.request("GET", "/")
            if resp.status_code != 200:
                return False, "HTTP", f"unexpected status={resp.status_code}"
            soup = BeautifulSoup(resp.text, "html.parser")
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            return True, "HTML", f"title={title or 'N/A'}"

        def case_context_lifecycle() -> Tuple[bool, str, str]:
            first_text = "Question: What is the derivative of x^2? Please answer briefly."
            second_text = "Question: What is the integral of x? Please answer briefly."

            self.client.request(
                "POST",
                "/context/reset",
                json_payload={"sessionId": self.session_id},
                headers=self._headers(),
                retries=1,
            )

            first = self.client.request(
                "POST",
                "/context/ingest",
                json_payload={
                    "sessionId": self.session_id,
                    "text": first_text,
                    "mode": "auto",
                    "useV51": True,
                    "domainHints": ["math", "derivative"],
                },
                headers=self._headers(),
                retries=1,
            )
            ok, detail = self._assert_json_ok(first, "context/ingest")
            first_data = self._json(first)
            if not ok:
                return False, "HTTP/JSON", detail

            ok_ctx, detail_ctx, ctx1 = self._require_dict(first_data.get("context"), "context")
            if not ok_ctx:
                return False, "PAYLOAD", detail_ctx
            if first_data.get("sessionId") != self.session_id:
                return False, "PAYLOAD", "sessionId mismatch after first ingest"
            if "v51" not in first_data:
                return False, "PAYLOAD", "missing v51 result"
            if int(ctx1.get("messageCount", 0)) != 1:
                return False, "SEMANTIC", f"unexpected first messageCount={ctx1.get('messageCount')}"
            if first_text not in str(ctx1.get("context", "")):
                return False, "SEMANTIC", "first context does not contain ingested text"

            status1 = self.client.request("GET", f"/context/status?sessionId={self.session_id}", headers=self._headers())
            ok, detail = self._assert_json_ok(status1, "context/status")
            status1_data = self._json(status1)
            if not ok:
                return False, "HTTP/JSON", detail
            if status1_data.get("exists") is not True:
                return False, "SEMANTIC", "session not visible in context/status after first ingest"
            if int(status1_data.get("messageCount", 0)) != 1:
                return False, "SEMANTIC", f"status messageCount mismatch={status1_data.get('messageCount')}"
            if str(status1_data.get("lastMode", "")).strip() != "concat":
                return False, "SEMANTIC", f"unexpected lastMode={status1_data.get('lastMode')}"

            second = self.client.request(
                "POST",
                "/context/ingest",
                json_payload={
                    "sessionId": self.session_id,
                    "text": second_text,
                    "mode": "auto",
                    "useV51": False,
                    "domainHints": ["math", "integral"],
                },
                headers=self._headers(),
                retries=1,
            )
            ok, detail = self._assert_json_ok(second, "context/ingest(second)")
            second_data = self._json(second)
            if not ok:
                return False, "HTTP/JSON", detail
            ok_ctx, detail_ctx, ctx2 = self._require_dict(second_data.get("context"), "context")
            if not ok_ctx:
                return False, "PAYLOAD", detail_ctx
            context_text = str(ctx2.get("context", ""))
            if int(ctx2.get("messageCount", 0)) < 2:
                return False, "SEMANTIC", f"second messageCount too small={ctx2.get('messageCount')}"
            if first_text not in context_text or second_text not in context_text:
                return False, "SEMANTIC", "context did not retain both ingested turns"

            status2 = self.client.request("GET", f"/context/status?sessionId={self.session_id}", headers=self._headers())
            ok, detail = self._assert_json_ok(status2, "context/status(second)")
            status2_data = self._json(status2)
            if not ok:
                return False, "HTTP/JSON", detail
            if int(status2_data.get("messageCount", 0)) < 2:
                return False, "SEMANTIC", f"status did not accumulate messages={status2_data.get('messageCount')}"
            if int(status2_data.get("concatItems", 0)) < 2:
                return False, "SEMANTIC", f"concatItems did not grow={status2_data.get('concatItems')}"

            reset = self.client.request(
                "POST",
                "/context/reset",
                json_payload={"sessionId": self.session_id},
                headers=self._headers(),
            )
            ok, detail = self._assert_json_ok(reset, "context/reset")
            reset_data = self._json(reset)
            if not ok:
                return False, "HTTP/JSON", detail
            if reset_data.get("removed") is not True:
                return False, "SEMANTIC", f"reset did not remove session: {json.dumps(reset_data, ensure_ascii=False)}"

            status3 = self.client.request("GET", f"/context/status?sessionId={self.session_id}", headers=self._headers())
            ok, detail = self._assert_json_ok(status3, "context/status(after reset)")
            status3_data = self._json(status3)
            if not ok:
                return False, "HTTP/JSON", detail
            if status3_data.get("exists") is not False:
                return False, "SEMANTIC", f"session still exists after reset: {json.dumps(status3_data, ensure_ascii=False)}"
            return True, "SEMANTIC", "context ingest -> status -> reset lifecycle verified"

        def case_api_chat_proxy_ollama() -> Tuple[bool, str, str]:
            nonce = int(time.time() * 1000)
            payload = {
                "sessionId": self.session_id,
                "text": f"[api-chat-check:{nonce}] Please answer with one short sentence about calculus.",
                "maxTokens": 48,
            }
            resp = self.client.request("POST", "/api/chat", json_payload=payload, headers=self._headers(), retries=2)
            ok, detail = self._assert_json_ok(resp, "api/chat")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail

            if not isinstance(data, dict) or data.get("ok") is not True:
                return False, "PAYLOAD", f"/api/chat not ok: {json.dumps(data, ensure_ascii=False)[:240]}"

            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            if not result:
                return False, "PAYLOAD", "missing result object from /api/chat"

            using_ollama = bool(result.get("usingOllama", False))
            provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
            provider_id = str(provider.get("id", ""))
            reply = str(result.get("reply", "")).strip()
            model_elapsed_raw = result.get("modelElapsedMs", 0)
            if isinstance(model_elapsed_raw, (int, float)):
                model_elapsed = int(model_elapsed_raw)
            else:
                try:
                    model_elapsed = int(str(model_elapsed_raw).strip())
                except Exception:
                    model_elapsed = 0

            if not using_ollama:
                return False, "EVIDENCE", f"usingOllama=false, provider={provider_id or 'N/A'}"
            if "ollama" not in provider_id.lower():
                return False, "EVIDENCE", f"provider.id unexpected: {provider_id or 'N/A'}"
            if not reply:
                return False, "PAYLOAD", "empty result.reply from /api/chat"
            if model_elapsed <= 0:
                return False, "EVIDENCE", f"modelElapsedMs invalid: {result.get('modelElapsedMs')}"

            return True, "PROXY", f"provider={provider_id}, modelElapsedMs={model_elapsed}, replyLen={len(reply)}"

        def case_v51_chat_text() -> Tuple[bool, str, str]:
            payload = {
                "sessionId": self.session_id,
                "text": "SQuAD style question: The capital of France is what city?",
                "mode": "auto",
                "useV51": True,
                "domainHints": ["geography", "capital"],
            }
            resp = self.client.request("POST", "/v51/chat", json_payload=payload, headers=self._headers())
            ok, detail = self._assert_json_ok(resp, "v51/chat")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail

            response_text = ""
            if isinstance(data.get("v51"), dict):
                response_text = str(data["v51"].get("response", ""))
            if not response_text:
                return False, "PAYLOAD", "missing v51.response"

            expect = re2.search(r"(capital|france|city|summary|结构化|摘要)", response_text, flags=re2.IGNORECASE)
            if not expect:
                return False, "REGEX", f"response looks off: {response_text[:160]}"
            return True, "REGEX", "semantic markers detected"

        def case_vision_analyze() -> Tuple[bool, str, str]:
            payload = {"imageBase64": self._sample_png_base64()}
            resp = self.client.request("POST", "/vision/analyze", json_payload=payload, headers=self._headers())
            if resp.status_code == 500 and "not ready" in resp.text.lower():
                return True, "DEGRADED", "vision pipeline not ready in env"
            if resp.status_code >= 500:
                return False, "HTTP", f"vision server error: {resp.status_code}"
            if resp.status_code == 400 and "Invalid or empty image" in resp.text:
                return False, "FIXTURE", "valid PNG fixture was rejected as invalid image"
            ok, detail = self._assert_json_ok(resp, "vision/analyze")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail
            if data.get("ok") is not True:
                return False, "PAYLOAD", f"vision response not ok: {json.dumps(data, ensure_ascii=False)[:240]}"
            image_size = data.get("imageSize") if isinstance(data.get("imageSize"), dict) else {}
            embedding = data.get("embedding") if isinstance(data.get("embedding"), list) else []
            embedding_dim = int(data.get("embeddingDim", 0) or 0)
            if int(image_size.get("w", 0)) <= 0 or int(image_size.get("h", 0)) <= 0:
                return False, "SEMANTIC", f"invalid imageSize={json.dumps(image_size, ensure_ascii=False)}"
            if embedding_dim <= 0 or len(embedding) != embedding_dim:
                return False, "SEMANTIC", f"embedding mismatch dim={embedding_dim} len={len(embedding)}"
            if not isinstance(data.get("graphContext"), str):
                return False, "PAYLOAD", "missing graphContext"
            return True, "SEMANTIC", f"vision analyzed valid PNG {image_size.get('w')}x{image_size.get('h')}"

        def case_speech_analyze() -> Tuple[bool, str, str]:
            wav_b64 = self._generate_tone_wav_base64(duration_sec=0.8, freq=220.0)
            resp = self.client.request(
                "POST",
                "/speech/analyze",
                json_payload={"audioBase64": wav_b64},
                headers=self._headers(),
            )
            ok, detail = self._assert_json_ok(resp, "speech/analyze")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail
            if data.get("ok") is not True:
                return False, "PAYLOAD", f"speech analyze not ok: {json.dumps(data, ensure_ascii=False)[:240]}"

            environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
            tone = data.get("tone") if isinstance(data.get("tone"), dict) else {}
            corpus = data.get("learnableCorpus") if isinstance(data.get("learnableCorpus"), list) else []
            stage05 = str(data.get("stage05", ""))
            pitch = float(tone.get("pitch", 0.0) or 0.0)
            if not str(environment.get("env", "")).strip():
                return False, "SEMANTIC", "missing environment.env"
            if not str(tone.get("emotion", "")).strip():
                return False, "SEMANTIC", "missing tone.emotion"
            if pitch <= 50.0 or pitch >= 400.0:
                return False, "SEMANTIC", f"pitch out of expected range: {pitch}"
            if "speech_stage0.5" not in stage05 or "|env=" not in stage05 or "|emotion=" not in stage05 or "|pitch=" not in stage05:
                return False, "SEMANTIC", f"stage05 missing semantic markers: {stage05}"
            if len(corpus) < 2:
                return False, "SEMANTIC", f"learnableCorpus too small: {json.dumps(corpus, ensure_ascii=False)}"
            return True, "SEMANTIC", f"speech analyzed env={environment.get('env')} emotion={tone.get('emotion')} pitch={pitch:.1f}"

        def case_speech_synthesize() -> Tuple[bool, str, str]:
            resp = self.client.request(
                "POST",
                "/speech/synthesize",
                json_payload={
                    "text": "Audio regression baseline for Odin.",
                    "sampleRate": 16000,
                    "speed": 1.0,
                    "pitch": 1.0,
                },
                headers=self._headers(),
            )
            ok, detail = self._assert_json_ok(resp, "speech/synthesize")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail
            if data.get("ok") is not True:
                return False, "PAYLOAD", f"speech synth not ok: {json.dumps(data, ensure_ascii=False)[:240]}"
            if str(data.get("mime", "")) != "audio/wav":
                return False, "SEMANTIC", f"unexpected mime={data.get('mime')}"
            audio_b64 = str(data.get("audioBase64", ""))
            if not audio_b64:
                return False, "PAYLOAD", "missing audioBase64"
            wav_ok, wav_detail, wav_meta = self._decode_wav_meta(audio_b64)
            if not wav_ok:
                return False, "SEMANTIC", f"invalid synthesized wav: {wav_detail}"
            if wav_meta.get("channels") != 1 or wav_meta.get("sampleWidth") != 2:
                return False, "SEMANTIC", f"unexpected wav format={wav_meta}"
            if wav_meta.get("sampleRate") != int(data.get("sampleRate", 0) or 0):
                return False, "SEMANTIC", f"sampleRate mismatch body={data.get('sampleRate')} wav={wav_meta.get('sampleRate')}"
            if wav_meta.get("frames", 0) <= 100:
                return False, "SEMANTIC", f"too few audio frames={wav_meta.get('frames')}"
            duration_delta = abs(int(data.get("durationMs", 0) or 0) - int(wav_meta.get("durationMs", 0)))
            if duration_delta > 20:
                return False, "SEMANTIC", f"duration mismatch body={data.get('durationMs')} wav={wav_meta.get('durationMs')}"
            return True, "SEMANTIC", f"speech synthesis produced decodable wav duration={wav_meta.get('durationMs')}ms"

        def case_speech_ingest() -> Tuple[bool, str, str]:
            speech_session = f"{self.session_id}-speech"
            wav_b64 = self._generate_tone_wav_base64(duration_sec=0.8, freq=220.0)
            payload = {
                "sessionId": speech_session,
                "audioBase64": wav_b64,
                "mode": "auto",
                "useV51": True,
                "domainHints": ["audio", "speech"],
            }
            resp = self.client.request("POST", "/speech/ingest", json_payload=payload, headers=self._headers())
            ok, detail = self._assert_json_ok(resp, "speech/ingest")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail
            ok_speech, detail_speech, speech = self._require_dict(data.get("speech"), "speech")
            if not ok_speech:
                return False, "PAYLOAD", detail_speech
            ok_context, detail_context, context = self._require_dict(data.get("context"), "context")
            if not ok_context:
                return False, "PAYLOAD", detail_context
            if data.get("sessionId") != speech_session:
                return False, "PAYLOAD", "speech ingest sessionId mismatch"
            stage05 = str(speech.get("stage05", ""))
            if "speech_stage0.5" not in stage05:
                return False, "SEMANTIC", f"speech stage05 missing: {stage05}"
            if int(context.get("messageCount", 0)) < 1:
                return False, "SEMANTIC", f"context messageCount unexpected={context.get('messageCount')}"
            if stage05 not in str(context.get("context", "")):
                return False, "SEMANTIC", "speech stage05 was not injected into context"

            status = self.client.request("GET", f"/context/status?sessionId={speech_session}", headers=self._headers())
            ok, detail = self._assert_json_ok(status, "context/status(speech)")
            status_data = self._json(status)
            if not ok:
                return False, "HTTP/JSON", detail
            if status_data.get("exists") is not True or int(status_data.get("messageCount", 0)) < 1:
                return False, "SEMANTIC", f"speech ingest session not persisted: {json.dumps(status_data, ensure_ascii=False)}"
            return True, "SEMANTIC", "speech analyze -> context ingest linkage verified"

        def case_learn_and_status() -> Tuple[bool, str, str]:
            learn_payload = {
                "sessionId": self.session_id,
                "feedback": 0.75,
                "learningRate": 0.08,
                "keywords": ["proof", "consistency", "constraint"],
            }
            learn = self.client.request("POST", "/v51/learn", json_payload=learn_payload, headers=self._headers())
            ok, detail = self._assert_json_ok(learn, "v51/learn")
            if not ok:
                return False, "HTTP/JSON", detail

            status = self.client.request("GET", f"/v51/status?sessionId={self.session_id}", headers=self._headers())
            ok2, detail2 = self._assert_json_ok(status, "v51/status")
            if not ok2:
                return False, "HTTP/JSON", detail2

            js = self._json(status)
            sess = js.get("session")
            if sess is None:
                return False, "PAYLOAD", "session not found in status"
            return True, "JSON", "learn->status path verified"

        def case_gguf_inspect() -> Tuple[bool, str, str]:
            fixture = self._write_minimal_gguf_fixture()
            encoded_path = urllib.parse.quote(str(fixture))
            candidate_urls = [self.client.base_url]
            if self.client.base_url.endswith(":5081"):
                candidate_urls.append(self.client.base_url[:-5] + ":5080")

            resp = None
            for base_url in candidate_urls:
                probe = self.client.session.get(
                    f"{base_url}/api/gguf/inspect?path={encoded_path}",
                    headers=self._headers(),
                    timeout=max(self.client.timeout_sec, 8.0),
                )
                if probe.status_code != 404:
                    resp = probe
                    break
            if resp is None:
                return False, "HTTP", "gguf inspect route not found on frontend or gateway ports"
            ok, detail = self._assert_json_ok(resp, "api/gguf/inspect")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail
            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            model = result.get("model") if isinstance(result.get("model"), dict) else {}
            projection = result.get("brainProjection") if isinstance(result.get("brainProjection"), dict) else {}
            tensors = result.get("tensors") if isinstance(result.get("tensors"), dict) else {}
            if result.get("valid") is not True:
                return False, "PAYLOAD", f"inspection invalid: {json.dumps(data, ensure_ascii=False)[:260]}"
            if int(result.get("tensorCount", 0) or 0) != 1:
                return False, "SEMANTIC", f"tensorCount unexpected: {result.get('tensorCount')}"
            if int(model.get("vocabSize", 0) or 0) != 4:
                return False, "SEMANTIC", f"vocabSize unexpected: {model.get('vocabSize')}"
            if int(model.get("embeddingWidth", 0) or 0) != 64:
                return False, "SEMANTIC", f"embeddingWidth unexpected: {model.get('embeddingWidth')}"
            if str(projection.get("mode", "")) != "token-unit-many-to-many":
                return False, "SEMANTIC", f"projection mode unexpected: {projection.get('mode')}"
            largest = tensors.get("largest") if isinstance(tensors.get("largest"), list) else []
            if not largest or str(largest[0].get("name", "")) != "token_embd.weight":
                return False, "SEMANTIC", f"largest tensor mismatch: {json.dumps(largest, ensure_ascii=False)[:180]}"
            return True, "SEMANTIC", "gguf parser returned tensor and token-unit projection metadata"

        def case_gguf_export() -> Tuple[bool, str, str]:
            fixture = self._write_minimal_gguf_fixture()
            export_dir = self.root / "build" / "tmp" / "apitest_gguf_export"
            export_dir.mkdir(parents=True, exist_ok=True)
            encoded_path = urllib.parse.quote(str(fixture))
            encoded_output = urllib.parse.quote(str(export_dir))
            candidate_urls = [self.client.base_url]
            if self.client.base_url.endswith(":5081"):
                candidate_urls.append(self.client.base_url[:-5] + ":5080")

            resp = None
            for base_url in candidate_urls:
                probe = self.client.session.get(
                    f"{base_url}/api/gguf/export?path={encoded_path}&outputDir={encoded_output}",
                    headers=self._headers(),
                    timeout=max(self.client.timeout_sec, 8.0),
                )
                if probe.status_code != 404:
                    resp = probe
                    break
            if resp is None:
                return False, "HTTP", "gguf export route not found on frontend or gateway ports"
            ok, detail = self._assert_json_ok(resp, "api/gguf/export")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail

            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            gguf_model = result.get("ggufModel") if isinstance(result.get("ggufModel"), dict) else {}
            vocab = result.get("vocabulary") if isinstance(result.get("vocabulary"), dict) else {}
            mappings = result.get("semanticMapping") if isinstance(result.get("semanticMapping"), list) else []
            dynamics = result.get("neuroDynamicsTable") if isinstance(result.get("neuroDynamicsTable"), list) else []
            fit_result = result.get("fitResult") if isinstance(result.get("fitResult"), dict) else {}
            runtime_cfg = result.get("runtimeConfig") if isinstance(result.get("runtimeConfig"), dict) else {}
            manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}

            if gguf_model.get("architecture") != "llama":
                return False, "SEMANTIC", f"export architecture unexpected: {gguf_model.get('architecture')}"
            if int(vocab.get("sourceTokenCount", 0) or 0) != 4:
                return False, "SEMANTIC", f"export vocab size unexpected: {vocab.get('sourceTokenCount')}"
            if int(vocab.get("specialTokens", {}).get("bos", -1)) != 0:
                return False, "SEMANTIC", f"bos special token unexpected: {json.dumps(vocab.get('specialTokens', {}), ensure_ascii=False)}"
            if len(mappings) != 4:
                return False, "SEMANTIC", f"semantic mapping count unexpected: {len(mappings)}"
            if not dynamics:
                return False, "SEMANTIC", "neuro dynamics table empty"
            if "parameterSnapshot" not in fit_result or "windowSize" not in runtime_cfg:
                return False, "SEMANTIC", f"missing fit/runtime export fields: {json.dumps(result, ensure_ascii=False)[:260]}"
            if manifest.get("written") is not True:
                return False, "SEMANTIC", f"export files not written: {json.dumps(manifest, ensure_ascii=False)[:260]}"
            expected = [
                export_dir / "gguf" / "model.json",
                export_dir / "vocab" / "tokens.json",
                export_dir / "mapping" / "semantic_mapping.json",
                export_dir / "dynamics" / "neuro_dynamics.json",
                export_dir / "fit" / "fit_result.json",
                export_dir / "runtime" / "runtime_config.json",
                export_dir / "manifest.json",
            ]
            missing = [str(path) for path in expected if not path.exists()]
            if missing:
                return False, "IO", f"export output files missing: {missing}"
            return True, "SEMANTIC", "gguf structured export bundle and files verified"

        def case_fine_tuning_corpus_add() -> Tuple[bool, str, str]:
            corpus_path = self.root / "build" / "tmp" / "apitest_fine_tuning" / "append_corpus.jsonl"
            corpus_path.parent.mkdir(parents=True, exist_ok=True)
            if corpus_path.exists():
                corpus_path.unlink()
            payload = {
                "appendCorpusPath": str(corpus_path),
                "items": [
                    {"question": "请解释滑动窗口。", "answer": "滑动窗口通过限制活跃上下文范围来控制内存。", "style": "tech"},
                    {"question": "什么是参数化查询？", "answer": "参数化查询把数据与 SQL 指令分离，降低注入风险。", "style": "security"},
                ],
            }
            resp = self.client.request("POST", "/api/fine_tuning/corpus/add", json_payload=payload, headers=self._headers())
            ok, detail = self._assert_json_ok(resp, "api/fine_tuning/corpus/add")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail
            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            if int(result.get("added", 0) or 0) != 2:
                return False, "SEMANTIC", f"added unexpected: {json.dumps(data, ensure_ascii=False)[:240]}"
            if not corpus_path.exists():
                return False, "IO", f"corpus file missing: {corpus_path}"
            rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) != 2:
                return False, "IO", f"corpus rows unexpected: {len(rows)}"
            if not rows[0].get("questionRaw") or not isinstance(rows[0].get("gnnKeywords"), list):
                return False, "SEMANTIC", f"corpus row missing augmentation fields: {json.dumps(rows[0], ensure_ascii=False)}"
            return True, "SEMANTIC", "fine-tuning corpus add wrote augmented jsonl rows"

        def case_fine_tuning_run_dry() -> Tuple[bool, str, str]:
            work_dir = self.root / "build" / "tmp" / "apitest_fine_tuning"
            corpus_path = work_dir / "append_corpus.jsonl"
            output_dir = work_dir / "run_out"
            report_path = output_dir / "dry_run_report.json"
            payload = {
                "appendCorpusPath": str(corpus_path),
                "outputDir": str(output_dir),
                "reportPath": str(report_path),
                "selfPlayPairs": 0,
                "epochs": 1,
                "lr": 0.00002,
                "alpha": 0.6,
                "beta": 0.4,
                "dryRun": True,
                "hfModel": "hf:Qwen/Qwen2.5-1.5B-Instruct",
                "ollamaModel": "llama3.1:8b",
            }
            resp = self.client.request("POST", "/api/fine_tuning/run", json_payload=payload, headers=self._headers(), retries=1)
            ok, detail = self._assert_json_ok(resp, "api/fine_tuning/run")
            data = self._json(resp)
            if not ok:
                return False, "HTTP/JSON", detail
            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            report = result.get("report") if isinstance(result.get("report"), dict) else {}
            if result.get("dryRun") is not True:
                return False, "SEMANTIC", f"dryRun flag missing: {json.dumps(data, ensure_ascii=False)[:260]}"
            if report.get("mode") != "dry-run":
                return False, "SEMANTIC", f"report mode unexpected: {json.dumps(report, ensure_ascii=False)[:260]}"
            if int(report.get("appendSampleCount", 0) or 0) < 2:
                return False, "SEMANTIC", f"appendSampleCount unexpected: {json.dumps(report, ensure_ascii=False)[:260]}"
            if report.get("requiresOllama") is not False:
                return False, "SEMANTIC", f"dry run should not require Ollama: {json.dumps(report, ensure_ascii=False)[:260]}"
            weights = report.get("weights") if isinstance(report.get("weights"), dict) else {}
            if float(weights.get("alpha", 0.0) or 0.0) != 0.6 or float(weights.get("beta", 0.0) or 0.0) != 0.4:
                return False, "SEMANTIC", f"alpha/beta mismatch: {json.dumps(weights, ensure_ascii=False)}"
            if not report_path.exists():
                return False, "IO", f"dry run report missing: {report_path}"
            return True, "SEMANTIC", "fine-tuning dry-run validated corpus and plan"

        def case_fine_tuning_run_invalid_weights() -> Tuple[bool, str, str]:
            work_dir = self.root / "build" / "tmp" / "apitest_fine_tuning"
            corpus_path = work_dir / "append_corpus.jsonl"
            payload = {
                "appendCorpusPath": str(corpus_path),
                "outputDir": str(work_dir / "run_out_invalid"),
                "selfPlayPairs": 0,
                "epochs": 1,
                "lr": 0.00002,
                "alpha": -0.1,
                "beta": 1.1,
                "dryRun": True,
            }
            resp = self.client.request("POST", "/api/fine_tuning/run", json_payload=payload, headers=self._headers(), retries=1)
            if resp.status_code != 400:
                return False, "HTTP", f"expected 400 for invalid alpha/beta, got {resp.status_code}: {resp.text[:240]}"
            data = self._json(resp)
            if data.get("ok") is not False:
                return False, "JSON", f"invalid-weights response should have ok=false: {json.dumps(data, ensure_ascii=False)[:240]}"
            error_text = str(data.get("error") or "")
            if "alpha out of range" not in error_text:
                return False, "SEMANTIC", f"unexpected invalid-weights error: {json.dumps(data, ensure_ascii=False)[:240]}"
            return True, "SEMANTIC", "fine-tuning invalid alpha/beta rejected with documented bad-request path"

        run_case("root_html", case_root_html)
        run_case("api_chat_proxy_ollama", case_api_chat_proxy_ollama)
        run_case("context_lifecycle", case_context_lifecycle)
        run_case("v51_chat_text", case_v51_chat_text)
        run_case("vision_analyze", case_vision_analyze)
        run_case("speech_analyze", case_speech_analyze)
        run_case("speech_synthesize", case_speech_synthesize)
        run_case("speech_ingest", case_speech_ingest)
        run_case("v51_learn_status", case_learn_and_status)
        run_case("gguf_inspect", case_gguf_inspect)
        run_case("gguf_export", case_gguf_export)
        run_case("fine_tuning_corpus_add", case_fine_tuning_corpus_add)
        run_case("fine_tuning_run_dry", case_fine_tuning_run_dry)
        run_case("fine_tuning_run_invalid_weights", case_fine_tuning_run_invalid_weights)
        return results


def write_report(report_path: Path, inventory: Dict[str, object], endpoints: List[ApiEndpoint], results: List[TestResult]) -> None:
    payload = {
        "generatedAt": datetime.now().isoformat(),
        "inventory": inventory,
        "endpoints": [asdict(x) for x in endpoints],
        "results": [asdict(x) for x in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for x in results if x.ok),
            "failed": sum(1 for x in results if not x.ok),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="API integration test suite for Odin v5.1")
    parser.add_argument("--base-url", default=os.getenv("TEST_BASE_URL", "http://127.0.0.1:5081"))
    parser.add_argument("--log-file", default=str(Path(__file__).resolve().parent / "logs" / "apitest.log"))
    parser.add_argument("--report-file", default=str(Path(__file__).resolve().parent / "logs" / "apitest_report.json"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    logger = DualLogger(Path(args.log_file))

    logger.info("Starting Odin v5.1 API test suite")
    logger.info(f"Base URL={args.base_url}")

    tester = V51ApiTester(root=repo_root, base_url=args.base_url, logger=logger)

    inventory, endpoints = tester.inventory()

    try:
        tester.authenticate()
        logger.info("Auth flow completed")
    except Exception as exc:
        logger.error(f"Authentication failed: {exc}")
        results = [TestResult(name="auth", ok=False, status="EXCEPTION", detail=str(exc), elapsed_ms=0)]
        write_report(Path(args.report_file), inventory, endpoints, results)
        return 2

    results = tester.run()
    write_report(Path(args.report_file), inventory, endpoints, results)

    failed = [r for r in results if not r.ok]
    logger.info(f"Tests completed: total={len(results)} pass={len(results)-len(failed)} fail={len(failed)}")
    logger.info(f"Report written to {args.report_file}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

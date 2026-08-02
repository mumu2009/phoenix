import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from main import DualLogger, V51ApiTester


@dataclass
class BackendMatrixResult:
    mode: str
    status: str
    ready: bool
    chat_ok: bool
    provider_id: str
    detail: str
    runtime_status: str
    model: str
    elapsed_ms: int


class BackendMatrixRunner:
    def __init__(self, root: Path, base_url: str, logger: DualLogger) -> None:
        self.root = root
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.tester = V51ApiTester(root=root, base_url=base_url, logger=logger)

    def authenticate(self) -> None:
        self.tester.authenticate()

    def _headers(self) -> Dict[str, str]:
        return self.tester._headers()

    def _request(self, method: str, path: str, json_payload: Optional[dict] = None):
        return self.tester.client.request(method, path, json_payload=json_payload, headers=self._headers(), retries=2)

    def _request_json(self, method: str, path: str, json_payload: Optional[dict] = None) -> dict:
        response = self._request(method, path, json_payload=json_payload)
        try:
            return response.json() if response.content else {}
        except Exception:
            return {}

    def _patch_mode(self, mode: str) -> dict:
        return self._request_json("PATCH", "/api/runtime/features", {"transformerMode": mode})

    def _get_features(self) -> dict:
        return self._request_json("GET", "/api/runtime/features")

    def _get_monitoring(self) -> dict:
        return self._request_json("GET", "/api/monitoring/stats")

    def _fetch_runtime_docs(self) -> Tuple[dict, dict]:
        return self._get_features(), self._get_monitoring()

    @staticmethod
    def _monitor_runtime_entry(monitoring_doc: dict, mode: str) -> dict:
        if isinstance(monitoring_doc.get("result"), dict):
            monitor_runtime = monitoring_doc["result"].get("externalRuntime")
            if isinstance(monitor_runtime, dict) and isinstance(monitor_runtime.get(mode), dict):
                return monitor_runtime.get(mode) or {}
        return {}

    def _wait_for_mode_state(self, mode: str, timeout_sec: float = 45.0) -> Tuple[dict, dict]:
        deadline = time.time() + max(timeout_sec, 1.0)
        last_features: dict = {}
        last_monitoring: dict = {}
        while time.time() < deadline:
            features_doc, monitoring_doc = self._fetch_runtime_docs()
            last_features = features_doc
            last_monitoring = monitoring_doc

            features = features_doc.get("features") if isinstance(features_doc.get("features"), dict) else {}
            pipeline = features.get("pipeline") if isinstance(features.get("pipeline"), dict) else {}
            external_runtime = features.get("externalRuntime") if isinstance(features.get("externalRuntime"), dict) else {}
            runtime_entry = external_runtime.get(mode) if isinstance(external_runtime.get(mode), dict) else {}
            monitor_entry = self._monitor_runtime_entry(monitoring_doc, mode)

            if mode == "native":
                if str(pipeline.get("transformerMode", "")) == "native":
                    return features_doc, monitoring_doc
            elif mode == "ollama":
                if str(pipeline.get("transformerMode", "")) in {"ollama", "ollama-fine-tuning"}:
                    return features_doc, monitoring_doc
            else:
                runtime_ready = bool(runtime_entry.get("ready", False) or monitor_entry.get("ready", False))
                runtime_status = str(runtime_entry.get("status", "") or monitor_entry.get("status", ""))
                if runtime_ready or runtime_status == "ready":
                    return features_doc, monitoring_doc
            time.sleep(1.0)

        return last_features, last_monitoring

    def _run_chat(self, mode: str) -> Tuple[bool, str, str]:
        max_tokens = 12 if mode == "native" else 24
        if mode == "native":
            chat_path = "/v51/chat"
            payload = {
                "sessionId": f"backend-matrix-{mode}-{int(time.time() * 1000)}",
                "text": "Please answer in one short sentence about calculus.",
                "mode": "auto",
                "useV51": True,
                "domainHints": ["math", "calculus"],
            }
        else:
            chat_path = "/api/chat"
            payload = {
                "sessionId": f"backend-matrix-{mode}-{int(time.time() * 1000)}",
                "text": "Please answer in one short sentence about calculus.",
                "maxTokens": max_tokens,
            }
        last_error = ""
        timeout = max(self.tester.client.timeout_sec, 185.0)
        for attempt in range(1, 4):
            try:
                response = self.tester.client.session.request(
                    method="POST",
                    url=f"{self.base_url}{chat_path}",
                    json=payload,
                    headers=self._headers(),
                    timeout=timeout,
                )
            except Exception as exc:
                last_error = str(exc)
                if mode in {"llamacpp", "bitnet"}:
                    time.sleep(20.0 * attempt)
                else:
                    time.sleep(1.0 * attempt)
                continue

            data = {}
            try:
                data = response.json() if response.content else {}
            except Exception:
                pass

            if response.status_code >= 400:
                error = data.get("error") if isinstance(data, dict) else response.text[:240]
                last_error = f"HTTP {response.status_code}: {error}"
                if response.status_code == 503 and mode in {"llamacpp", "bitnet"}:
                    time.sleep(20.0 * attempt)
                    continue
                return False, "", last_error

            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            provider = result.get("provider") if isinstance(result.get("provider"), dict) else {}
            provider_id = str(provider.get("id", ""))
            reply = str(result.get("reply", "")).strip()
            if mode == "native" and not reply and isinstance(data.get("v51"), dict):
                reply = str(data["v51"].get("response", "")).strip()
            if mode == "native" and not provider_id and reply:
                provider_id = "core"
            if not reply:
                last_error = "empty reply"
                time.sleep(0.5 * attempt)
                continue
            return True, provider_id, f"replyLen={len(reply)}"

        return False, "", last_error or "chat failed"

    def run(self) -> List[BackendMatrixResult]:
        results: List[BackendMatrixResult] = []
        modes = ["ollama", "llamacpp", "bitnet", "native"]

        for mode in modes:
            started = time.time()
            detail = ""
            provider_id = ""
            runtime_status = ""
            model = ""
            ready = False
            chat_ok = False
            status = "FAIL"

            try:
                self._patch_mode(mode)
                features_doc, monitoring_doc = self._wait_for_mode_state(mode)
                features = features_doc.get("features") if isinstance(features_doc.get("features"), dict) else {}
                pipeline = features.get("pipeline") if isinstance(features.get("pipeline"), dict) else {}
                external_runtime = features.get("externalRuntime") if isinstance(features.get("externalRuntime"), dict) else {}
                runtime_entry = external_runtime.get(mode) if isinstance(external_runtime.get(mode), dict) else {}
                monitor_entry = self._monitor_runtime_entry(monitoring_doc, mode)

                if mode == "native":
                    ready = str(pipeline.get("transformerMode", "")) == "native"
                    model = "native-transformer"
                elif mode == "ollama":
                    ready = str(pipeline.get("transformerMode", "")) in {"ollama", "ollama-fine-tuning"}
                    model = str(pipeline.get("ollamaModel", ""))
                elif mode == "llamacpp":
                    ready = bool(runtime_entry.get("ready", False) or monitor_entry.get("ready", False))
                    runtime_status = str(runtime_entry.get("status", "") or monitor_entry.get("status", ""))
                    model = str(pipeline.get("llamacppModel", ""))
                elif mode == "bitnet":
                    ready = bool(runtime_entry.get("ready", False) or monitor_entry.get("ready", False))
                    runtime_status = str(runtime_entry.get("status", "") or monitor_entry.get("status", ""))
                    model = str(pipeline.get("bitnetModel", ""))

                chat_ok, provider_id, chat_detail = self._run_chat(mode)

                if mode == "native":
                    if chat_ok and provider_id == "core":
                        status = "PASS"
                        detail = chat_detail
                    else:
                        detail = chat_detail or "native provider mismatch"
                elif mode == "ollama":
                    if chat_ok and "ollama" in provider_id.lower():
                        status = "PASS"
                        detail = f"model={model} {chat_detail}"
                    else:
                        detail = chat_detail or f"ollama provider mismatch: {provider_id or 'N/A'}"
                else:
                    blocking_statuses = {"binary_missing", "waiting_manual_launch", "launch_failed", "health_timeout", "invalid_base_url"}
                    runtime_error = str(runtime_entry.get("error", "") or monitor_entry.get("error", ""))
                    if chat_ok and provider_id == mode:
                        status = "PASS"
                        detail = chat_detail
                    elif (not ready) or runtime_status in blocking_statuses or not model:
                        status = "BLOCKED"
                        detail = runtime_error
                        if not model:
                            detail = "missing configured .gguf model path"
                        elif not detail:
                            detail = runtime_status or chat_detail or "external backend blocked"
                    else:
                        detail = chat_detail or runtime_error or "external backend chat failed"
            except Exception as exc:
                detail = str(exc)

            elapsed = int((time.time() - started) * 1000)
            result = BackendMatrixResult(
                mode=mode,
                status=status,
                ready=ready,
                chat_ok=chat_ok,
                provider_id=provider_id,
                detail=detail,
                runtime_status=runtime_status,
                model=model,
                elapsed_ms=elapsed,
            )
            results.append(result)
            log_fn = self.logger.info if status == "PASS" else self.logger.warn if status == "BLOCKED" else self.logger.error
            log_fn(f"[{mode}] {status} ready={ready} chat_ok={chat_ok} provider={provider_id or 'N/A'} detail={detail}")

        return results


def write_report(report_path: Path, base_url: str, results: List[BackendMatrixResult]) -> None:
    summary = {
        "total": len(results),
        "pass": sum(1 for item in results if item.status == "PASS"),
        "blocked": sum(1 for item in results if item.status == "BLOCKED"),
        "fail": sum(1 for item in results if item.status == "FAIL"),
        "allUsable": all(item.status == "PASS" for item in results),
    }
    payload = {
        "generatedAt": datetime.now().isoformat(),
        "baseUrl": base_url,
        "summary": summary,
        "results": [asdict(item) for item in results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backend matrix test for Odin NLP providers")
    parser.add_argument("--base-url", default=os.getenv("TEST_BASE_URL", "http://127.0.0.1:5081"))
    parser.add_argument("--log-file", default=str(Path(__file__).resolve().parent / "logs" / "backend_matrix.log"))
    parser.add_argument("--report-file", default=str(Path(__file__).resolve().parent / "logs" / "backend_matrix_report.json"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    logger = DualLogger(Path(args.log_file))
    logger.info(f"Starting backend matrix test baseUrl={args.base_url}")

    runner = BackendMatrixRunner(root=repo_root, base_url=args.base_url, logger=logger)
    runner.authenticate()
    results = runner.run()
    write_report(Path(args.report_file), args.base_url, results)
    logger.info(
        f"Backend matrix finished pass={sum(1 for item in results if item.status == 'PASS')} blocked={sum(1 for item in results if item.status == 'BLOCKED')} fail={sum(1 for item in results if item.status == 'FAIL')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
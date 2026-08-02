import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from main import DualLogger, V51ApiTester


@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str


class MemeBarrierPhraseFeedbackSmoke:
    def __init__(self, root: Path, base_url: str, logger: DualLogger) -> None:
        self.root = root
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.tester = V51ApiTester(root=root, base_url=base_url, logger=logger)

    def authenticate(self) -> None:
        self.tester.authenticate()

    def _headers(self) -> Dict[str, str]:
        return self.tester._headers()

    def _request_json(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        response = self.tester.client.request(method, path, json_payload=payload, headers=self._headers(), retries=2)
        try:
            return response.json() if response.content else {}
        except Exception:
            return {}

    def run(self) -> Dict[str, object]:
        started = time.time()
        unique_suffix = str(int(started * 1000))
        positive_phrase = f"please explain bypass the payment safety gate {unique_suffix}"
        negative_phrase = f"alpha beta gamma delta {unique_suffix}"

        features_before = self._request_json("GET", "/api/runtime/features")
        memebarrier_before = ((features_before.get("features") or {}).get("memebarrier") or {})
        phrase_feedback_before = memebarrier_before.get("phraseFeedback") or {}
        old_step = phrase_feedback_before.get("step", 0.05)
        old_max_offset = phrase_feedback_before.get("maxOffset", 0.30)

        results = []
        try:
            patch_doc = self._request_json(
                "PATCH",
                "/api/runtime/features",
                {"memebarrierPhraseFeedbackStep": 0.07, "memebarrierPhraseFeedbackMaxOffset": 0.14},
            )
            patched_feedback = (((patch_doc.get("features") or {}).get("memebarrier") or {}).get("phraseFeedback") or {})
            ok_patch = abs(float(patched_feedback.get("step", 0.0)) - 0.07) < 1e-9 and abs(float(patched_feedback.get("maxOffset", 0.0)) - 0.14) < 1e-9
            results.append(SmokeResult("runtime_patch", ok_patch, json.dumps(patched_feedback, ensure_ascii=False)))

            positive_doc = self._request_json(
                "POST",
                "/api/memebarrier/phrase_feedback",
                {"question": positive_phrase, "feedbackType": "positive"},
            )
            ok_positive = bool(positive_doc.get("ok", False)) and positive_doc.get("persistent") is True and abs(float(positive_doc.get("persistentOffset", 0.0)) - 0.07) < 1e-9
            results.append(SmokeResult("positive_feedback", ok_positive, json.dumps(positive_doc, ensure_ascii=False)))

            negative_doc = self._request_json(
                "POST",
                "/api/memebarrier/phrase_feedback",
                {"question": negative_phrase, "feedbackType": "negative"},
            )
            ok_negative = bool(negative_doc.get("ok", False)) and negative_doc.get("persistent") is False and abs(float(negative_doc.get("transientOffset", 0.0)) + 0.07) < 1e-9
            results.append(SmokeResult("negative_feedback", ok_negative, json.dumps(negative_doc, ensure_ascii=False)))

            features_after = self._request_json("GET", "/api/runtime/features")
            feedback_after = ((((features_after.get("features") or {}).get("memebarrier") or {}).get("phraseFeedback")) or {})
            ok_summary = int(feedback_after.get("persistentPositiveCount", 0)) >= 1 and int(feedback_after.get("transientNegativeCount", 0)) >= 1
            results.append(SmokeResult("summary_counts", ok_summary, json.dumps(feedback_after, ensure_ascii=False)))
        finally:
            self._request_json(
                "PATCH",
                "/api/runtime/features",
                {
                    "memebarrierPhraseFeedbackStep": old_step,
                    "memebarrierPhraseFeedbackMaxOffset": old_max_offset,
                },
            )

        elapsed_ms = int((time.time() - started) * 1000)
        ok = all(item.ok for item in results)
        return {
            "ok": ok,
            "elapsedMs": elapsed_ms,
            "results": [asdict(item) for item in results],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for MemeBarrier phrase feedback runtime patching and submission")
    parser.add_argument("--base-url", default="http://127.0.0.1:5081", help="Phoenix frontend base URL")
    parser.add_argument("--log-file", default="test/apis/logs/memebarrier_phrase_feedback_smoke.log", help="Path to write the text log")
    parser.add_argument("--report-file", default="test/apis/logs/memebarrier_phrase_feedback_smoke_report.json", help="Path to write the JSON report")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    logger = DualLogger(Path(args.log_file))
    smoke = MemeBarrierPhraseFeedbackSmoke(root=root, base_url=args.base_url, logger=logger)

    try:
        smoke.authenticate()
        report = smoke.run()
    except Exception as exc:
        report = {"ok": False, "error": str(exc), "results": []}

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if report.get("ok"):
        logger.info("MemeBarrier phrase feedback smoke passed")
        return 0

    logger.error(f"MemeBarrier phrase feedback smoke failed: {json.dumps(report, ensure_ascii=False)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
import importlib.util
import pathlib
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).resolve().parent / "intelligence" / "main.py"
SPEC = importlib.util.spec_from_file_location("phoenix_intelligence_main", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class IntelligenceMetricsTests(unittest.TestCase):
    def test_warmup_system_uses_provided_endpoint(self) -> None:
        with mock.patch.object(MODULE, "post_json", return_value=(200, '{"ok": true, "result": {"reply": "ok"}}')) as post_mock:
            MODULE.warmup_system("http://127.0.0.1:5080/api/chat", 5.0, "")

        post_mock.assert_called_once()
        self.assertEqual(post_mock.call_args.args[0], "http://127.0.0.1:5080/api/chat")

    def test_score_text_rewards_overlap(self) -> None:
        score, detail = MODULE.score_text(
            "幂等性表示重复请求仍然得到相同结果，不会反复改变状态。",
            "幂等性表示同一个操作在重复请求下产生相同结果，不会因为多次执行而不断改变系统状态。",
        )
        self.assertGreater(score, 30.0)
        self.assertGreater(detail["sequence"], 20.0)
        self.assertGreater(detail["clauseCoverage"], 40.0)

    def test_hai_summary_uses_case_dimensions(self) -> None:
        results = [
            MODULE.EvalResult("qa-1", "qa", True, 200, 10.0, "q1", "a1", "text", 40.0, {}),
            MODULE.EvalResult("reason-1", "reasoning", True, 200, 12.0, "q2", "a2", "text", 60.0, {}),
        ]
        cases_by_id = {
            "qa-1": {"haiDimensions": ["factual_grounding"]},
            "reason-1": {"haiDimensions": ["reasoning_planning"]},
        }
        hai = MODULE.compute_hai_summary(results, cases_by_id)
        self.assertAlmostEqual(hai["overall"], 50.0)
        self.assertGreater(hai["coverage"], 30.0)
        self.assertIn("factual_grounding", hai["dimensions"])
        self.assertIn("reasoning_planning", hai["dimensions"])

    def test_hai_summary_falls_back_to_task_mapping(self) -> None:
        results = [
            MODULE.EvalResult("fmt-1", "formatting", True, 200, 9.0, "q", "a", "text", 55.0, {}),
        ]
        hai = MODULE.compute_hai_summary(results, {"fmt-1": {}})
        self.assertIn("instruction_following", hai["dimensions"])
        self.assertEqual(hai["dimensions"]["instruction_following"]["scoreAvg"], 55.0)


if __name__ == "__main__":
    unittest.main()
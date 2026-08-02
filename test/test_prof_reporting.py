import importlib.util
import pathlib
import tempfile
import types
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parent / "prof" / "main.py"
SPEC = importlib.util.spec_from_file_location("phoenix_prof_main", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProfReportingTests(unittest.TestCase):
    def test_external_line_pairs_respect_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            prompts = root / "prompts.txt"
            answers = root / "answers.txt"
            prompts.write_text("\n".join(f"prompt-{index}" for index in range(50)), encoding="utf-8")
            answers.write_text("\n".join(f"answer-{index}" for index in range(50)), encoding="utf-8")

            cases = MODULE.load_external_line_pairs(str(prompts), str(answers), 20, 5.0)

        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case.case_id for case in cases}), 20)

    def test_incomplete_report_is_marked_as_snapshot(self) -> None:
        args = types.SimpleNamespace(
            system_url="http://127.0.0.1:5080/api/chat",
            ollama_url="http://127.0.0.1:11434/api/chat",
            llamacpp_url="http://127.0.0.1:8082/v1/chat/completions",
            llamacpp_api_style="openai-chat",
            ollama_model="llama3.1:8b",
            llamacpp_model="llamacpp",
            enable_llamacpp=False,
            cases_file="test/intelligence/cases.baseline.json",
            benchmark_presets=[],
            benchmark_cache_dir="runtime_store/prof_cache",
            dataset_server_url="https://datasets-server.huggingface.co/rows",
            benchmark_fetch_retries=0,
            benchmark_cache_only=False,
            external_dataset_files=[],
            external_prompts_file="questionaire.txt",
            external_answers_file="answer.txt",
            external_limit=20,
            auto_discover_tests_datasets=False,
            tests_dataset_limit=0,
            shared_local_qa=False,
            resolved_questionnaire_files=["questionaire.txt"],
            questionnaire_limit=2,
            rounds=1,
            concurrency=1,
            timeout=60.0,
            shuffle_cases=True,
            random_seed=1,
            stability_stop=False,
            stability_check_interval=100,
            stability_min_samples=200,
            stability_window=3,
            ollama_num_thread=0,
            auto_manage_system=False,
            auto_manage_ollama=False,
            resume=False,
            checkpoint_every=100,
        )
        summary = MODULE.summarize_route([])
        route_payloads = {
            MODULE.ROUTE_SYSTEM: {"summary": summary, "wall_ms": 0.0},
            MODULE.ROUTE_OLLAMA: {"summary": summary, "wall_ms": 0.0},
            MODULE.ROUTE_LLAMACPP: {"summary": summary, "wall_ms": 0.0},
        }
        comparisons = {
            "quality_delta": {"delta": 0.0, "winner": "tie"},
            "latency_significance": {"test": "n/a"},
        }
        deps = MODULE.BenchDeps("urllib", False, False, False, False, False, False)
        progress_state = {
            "complete": False,
            "routes": {
                MODULE.ROUTE_SYSTEM: {"completed": 10, "expected": 30},
                MODULE.ROUTE_OLLAMA: {"completed": 0, "expected": 30},
                MODULE.ROUTE_LLAMACPP: {"completed": 0, "expected": 0},
            },
        }

        report = MODULE.build_report(
            args,
            [],
            [],
            [],
            [],
            route_payloads,
            comparisons,
            deps,
            {"ready": True, "action": "reuse"},
            {"ready": True, "action": "reuse"},
            {},
            {"selected": [], "loaded": [], "failed": [], "sourceReachable": True},
            {"files": [], "count": 0, "sharedCount": 0},
            progress_state,
        )

        self.assertIn("未完成快照", report)
        self.assertIn("不能据此判定模型能力为 0", report)


if __name__ == "__main__":
    unittest.main()

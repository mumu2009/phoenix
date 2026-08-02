import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).resolve().parent / "prof" / "offline_matrix.py"
SPEC = importlib.util.spec_from_file_location("phoenix_offline_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OfflineMatrixTests(unittest.TestCase):
    def test_load_plan_supports_json_presets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            plan = root / "plan.json"
            plan.write_text(
                """
[
  {
    "name": "functional-json",
    "backend": "ollama",
    "componentConfig": "configs/full.json",
    "benchArgs": ["--shared-local-qa", "--tests-dataset-limit", "16"]
  }
]
""".strip(),
                encoding="utf-8",
            )

            presets = MODULE.load_plan(plan)

        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["name"], "functional-json")
        self.assertEqual(presets[0]["backend"], "ollama")
        self.assertEqual(presets[0]["componentConfig"], "configs/full.json")

    def test_build_component_args_prefers_config_then_cli_overrides(self) -> None:
        preset = {
            "componentConfig": "test/prof/component_presets/full_functional.json",
            "components": "brain=structural,gnn=off",
        }

        args = MODULE.build_component_args(preset)

        self.assertIn("--component-config=test/prof/component_presets/full_functional.json", args)
        self.assertIn("--components=brain=structural,gnn=off", args)

    def test_build_hai_command_targets_baseline_suite(self) -> None:
        args = types.SimpleNamespace(
            system_url="http://127.0.0.1:5080/api/chat",
            system_token="local-dev",
            ollama_url="http://127.0.0.1:11434/api/chat",
            ollama_model="llama3.1:8b",
            timeout=120.0,
            max_tokens=160,
        )
        preset = {
            "runHai": True,
            "haiCasesFile": "test/intelligence/cases.baseline.json",
        }

        command = MODULE.build_hai_command(
            args,
            pathlib.Path("Python314/python.exe"),
            preset,
            pathlib.Path("build/hai_eval_report.md"),
            pathlib.Path("build/hai_eval_report.json"),
        )

        self.assertIn("test/intelligence/cases.baseline.json", command)
        self.assertIn("--output-md", command)
        self.assertIn("--output-json", command)

    def test_build_benchmark_command_embeds_system_launch_command(self) -> None:
        args = types.SimpleNamespace(
            system_url="http://127.0.0.1:5080/api/chat",
            system_token="",
            ollama_url="http://127.0.0.1:11434/api/chat",
            llamacpp_url="http://127.0.0.1:8082/v1/chat/completions",
            ollama_model="llama3.1:8b",
            rounds=1,
            concurrency=1,
            timeout=30.0,
            max_tokens=128,
            ollama_warmup_timeout=180.0,
            tests_dataset_limit=8,
            skip_direct_llamacpp=True,
        )
        preset = {
            "backend": "ollama",
            "componentConfig": "test/prof/component_presets/full_functional.json",
        }

        command = MODULE.build_benchmark_command(
            args,
            pathlib.Path("Python314/python.exe"),
            pathlib.Path("build/phoenix_main.exe"),
            preset,
            pathlib.Path("build/benchmark_report.md"),
            pathlib.Path("build/benchmark_report.json"),
        )

        self.assertIn("--system-launch-command-json", command)
        self.assertNotIn("--no-auto-manage-system", command)
        launch_index = command.index("--system-launch-command-json") + 1
        launch_command = json.loads(command[launch_index])
        self.assertEqual(pathlib.Path(launch_command[0]), pathlib.Path("build/phoenix_main.exe"))
        self.assertIn("--component-config=test/prof/component_presets/full_functional.json", launch_command)

    def test_build_benchmark_command_enables_direct_llamacpp_for_llamacpp_backend(self) -> None:
        args = types.SimpleNamespace(
            system_url="http://127.0.0.1:5080/api/chat",
            system_token="",
            ollama_url="http://127.0.0.1:11434/api/chat",
            llamacpp_url="http://127.0.0.1:8082/v1/chat/completions",
            ollama_model="llama3.1:8b",
            rounds=1,
            concurrency=1,
            timeout=30.0,
            max_tokens=128,
            ollama_warmup_timeout=180.0,
            tests_dataset_limit=8,
            skip_direct_llamacpp=False,
        )
        preset = {
            "backend": "llamacpp",
        }

        with mock.patch.object(MODULE, "direct_llamacpp_ready", return_value=True):
            command = MODULE.build_benchmark_command(
                args,
                pathlib.Path("Python314/python.exe"),
                pathlib.Path("build/phoenix_main.exe"),
                preset,
                pathlib.Path("build/benchmark_report.md"),
                pathlib.Path("build/benchmark_report.json"),
            )

        self.assertIn("--enable-llamacpp", command)

    def test_ensure_system_ready_for_hai_waits_for_chat_probe(self) -> None:
        args = types.SimpleNamespace(
            system_url="http://127.0.0.1:5080/api/chat",
            system_token="",
            startup_timeout=15.0,
            timeout=30.0,
        )
        process = mock.Mock()
        process.poll.return_value = None

        with mock.patch.object(MODULE, "wait_http_ready", return_value=None), mock.patch.object(
            MODULE,
            "wait_system_chat_ready",
            return_value=None,
        ) as chat_probe_mock:
            result_process, restarted, reason = MODULE.ensure_system_ready_for_hai(
                process,
                pathlib.Path("build/phoenix_main.exe"),
                {"backend": "ollama"},
                args,
            )

        self.assertIs(result_process, process)
        self.assertFalse(restarted)
        self.assertEqual(reason, "")
        chat_probe_mock.assert_called_once_with("http://127.0.0.1:5080/api/chat", "", 30.0)

    def test_extract_hai_summary_reads_score_and_coverage(self) -> None:
        payload = {
            "summary": {
                "total": 10,
                "transportPassed": 8,
                "transportFailed": 2,
                "transportPassRate": 80.0,
                "qualityPassed": 8,
                "qualityPassRate": 100.0,
                "scoreAvg": 39.06,
                "ollamaScoreAvg": 40.26,
                "scoreDeltaVsOllamaAvg": -1.2,
                "taskTypes": {
                    "coding": {
                        "total": 2,
                        "transportPassed": 2,
                        "qualityPassed": 1,
                        "qualityPassRate": 50.0,
                        "scoreAvg": 28.5,
                        "latencyAvgMs": 1234.0,
                    }
                },
            },
            "hai": {
                "overall": 39.07,
                "coverage": 83.33,
            },
            "gates": {
                "passed": True,
            },
        }

        summary = MODULE.extract_hai_summary(payload)

        self.assertAlmostEqual(summary["overall"], 39.07)
        self.assertAlmostEqual(summary["coverage"], 83.33)
        self.assertAlmostEqual(summary["scoreAvg"], 39.06)
        self.assertEqual(summary["transportPassed"], 8)
        self.assertIn("coding", summary["taskTypes"])
        self.assertTrue(summary["gatesPassed"])

    def test_summarize_component_config_reads_json_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            config = root / "functional.json"
            config.write_text(
                """
{
    "pipeline": {
        "gnn": true,
        "brain": {
            "profile": "functional"
        }
    },
    "learning": {
        "enabled": true
    }
}
""".strip(),
                encoding="utf-8",
            )

            summary = MODULE.summarize_component_config(str(config))

        self.assertIn("pipeline.gnn=true", summary)
        self.assertIn("pipeline.brain.profile=functional", summary)
        self.assertIn("learning.enabled=true", summary)

    def test_classify_result_status_distinguishes_regression_and_error(self) -> None:
        regression_status = MODULE.classify_result_status(
            1,
            {"routes": {"system": {}}},
            True,
            1,
            {"summary": {"scoreAvg": 10.0}},
        )
        error_status = MODULE.classify_result_status(1, {}, True, 1, {})

        self.assertEqual(regression_status, "regression")
        self.assertEqual(error_status, "error")

    def test_has_any_gguf_detects_manifest_backed_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir) / "GGUF_models"
            manifest_path = root / "manifests" / "registry.ollama.ai" / "library" / "llama3.1" / "8b"
            blob_path = root / "blobs" / "sha256-1234abcd"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "layers": [
                            {
                                "mediaType": "application/vnd.ollama.image.model",
                                "digest": "sha256:1234abcd",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            blob_path.write_bytes(b"GGUFpayload")

            self.assertTrue(MODULE.has_any_gguf(root))

    def test_should_skip_preset_accepts_manifest_backed_llamacpp_model(self) -> None:
        preset = {
            "backend": "llamacpp",
            "requireGguf": True,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = pathlib.Path(temp_dir)
            manifest_path = workspace / "GGUF_models" / "manifests" / "registry.ollama.ai" / "library" / "llama3.1" / "8b"
            blob_path = workspace / "GGUF_models" / "blobs" / "sha256-deadbeef"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "layers": [
                            {
                                "mediaType": "application/vnd.ollama.image.model",
                                "digest": "sha256:deadbeef",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            blob_path.write_bytes(b"GGUFmodel")

            with mock.patch.object(MODULE, "workspace_root", return_value=workspace):
                self.assertEqual(MODULE.should_skip_preset(preset), "")

    def test_write_aggregate_report_writes_root_alias_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir) / "build" / "offline_matrix"
            results = [
                {
                    "name": "phoenix-ollama-all-off-cli",
                    "backend": "ollama",
                    "status": "ok",
                    "reason": "",
                    "benchmarkMode": "shared-local-qa",
                    "componentLabel": "CLI:gnn=off,frontend-memory=off",
                    "systemArgs": ["--frontend-enabled=false"],
                    "qualitySource": "shared-local-qa://tests/GPT4all",
                    "qualityDeltaVsOllama": 1.25,
                    "haiEnabled": True,
                    "reportMarkdown": "build/offline_matrix/phoenix-ollama-all-off-cli/benchmark_report.md",
                    "reportJson": "build/offline_matrix/phoenix-ollama-all-off-cli/benchmark_report.json",
                    "haiReportMarkdown": "build/offline_matrix/phoenix-ollama-all-off-cli/hai_eval_report.md",
                    "haiReportJson": "build/offline_matrix/phoenix-ollama-all-off-cli/hai_eval_report.json",
                    "system": {
                        "success_rate": 100.0,
                        "quality_avg": 40.4,
                        "balanced_score": 55.0,
                        "task_types": {
                            "coding": {
                                "total": 2,
                                "transportPassed": 2,
                                "qualityPassed": 2,
                                "qualityPassRate": 100.0,
                                "scoreAvg": 32.75,
                                "latencyAvgMs": 11248.41,
                            }
                        },
                    },
                    "ollama": {
                        "quality_avg": 39.96,
                    },
                    "hai": {
                        "transportPassRate": 100.0,
                        "qualityPassRate": 100.0,
                        "overall": 40.4,
                        "scoreAvg": 40.4,
                        "scoreDeltaVsOllamaAvg": 0.44,
                        "gatesPassed": True,
                        "taskTypes": {
                            "coding": {
                                "total": 2,
                                "transportPassed": 2,
                                "qualityPassed": 2,
                                "qualityPassRate": 100.0,
                                "scoreAvg": 32.75,
                                "latencyAvgMs": 11248.41,
                            }
                        },
                    },
                }
            ]

            MODULE.write_aggregate_report(results, output_dir)

            self.assertTrue((output_dir / "offline_matrix_summary.md").exists())
            self.assertTrue((output_dir / "offline_matrix_summary.csv").exists())
            self.assertTrue((output_dir.parent / "offline_matrix_summary.md").exists())
            csv_text = (output_dir.parent / "offline_matrix_summary.csv").read_text(encoding="utf-8")
            md_text = (output_dir.parent / "offline_matrix_summary.md").read_text(encoding="utf-8")
            self.assertIn("phoenix-ollama-all-off-cli", csv_text)
            self.assertIn("## Full Matrix", md_text)
            self.assertIn("## HAI Task Matrix", md_text)

    def test_run_single_preset_restarts_system_before_hai_when_probe_fails(self) -> None:
        args = types.SimpleNamespace(
            system_url="http://127.0.0.1:5080/api/chat",
            ollama_url="http://127.0.0.1:11434/api/chat",
            llamacpp_url="http://127.0.0.1:8082/v1/chat/completions",
            ollama_model="llama3.1:8b",
            system_token="local-dev",
            startup_timeout=15.0,
            timeout=120.0,
            max_tokens=160,
            rounds=1,
            concurrency=1,
            tests_dataset_limit=8,
            ollama_warmup_timeout=60.0,
            skip_direct_llamacpp=True,
        )
        preset = {
            "name": "phoenix-ollama-minimal-cli",
            "backend": "ollama",
            "runHai": True,
        }

        class FakeProcess:
            def __init__(self) -> None:
                self.terminated = False

            def poll(self):
                return None

            def terminate(self) -> None:
                self.terminated = True

        first_process = FakeProcess()
        second_process = FakeProcess()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir) / "offline_matrix"
            with mock.patch.object(MODULE, "find_phoenix_executable", return_value=pathlib.Path("phoenix_main.exe")), \
                mock.patch.object(MODULE, "kill_named_processes"), \
                mock.patch.object(MODULE, "workspace_root", return_value=pathlib.Path(temp_dir)), \
                mock.patch.object(MODULE, "start_system_process", side_effect=[first_process, second_process]) as start_mock, \
                mock.patch.object(MODULE, "wait_http_ready", side_effect=[None, RuntimeError("down after benchmark"), None]), \
                mock.patch.object(MODULE, "wait_system_chat_ready", return_value=None), \
                mock.patch.object(MODULE, "run_command", side_effect=[0, 0]), \
                mock.patch.object(MODULE, "read_report_payload", side_effect=[{"routes": {"system": {}}}, {"summary": {"scoreAvg": 33.0}}]):
                result = MODULE.run_single_preset(args, pathlib.Path("Python314/python.exe"), preset, output_dir)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["haiSystemRestarted"])
        self.assertIn("down after benchmark", result["haiSystemRestartReason"])
        self.assertEqual(start_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
import importlib.util
import json
import pathlib
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).resolve().parent / "prof" / "main.py"
SPEC = importlib.util.spec_from_file_location("phoenix_prof_main", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProfMainTests(unittest.TestCase):
    def test_parse_system_launch_command_accepts_json_array(self) -> None:
        command = MODULE.parse_system_launch_command(json.dumps(["phoenix_main.exe", "--components=brain=off"]))

        self.assertEqual(command, ["phoenix_main.exe", "--components=brain=off"])

    def test_ensure_system_service_uses_custom_launch_command(self) -> None:
        launch_command = ["phoenix_main.exe", "--component-config=test/prof/component_presets/full_functional.json"]
        with mock.patch.object(MODULE, "is_tcp_port_open", return_value=False), \
             mock.patch.object(MODULE, "start_phoenix_service", return_value=(True, "started")) as start_mock, \
             mock.patch.object(MODULE, "wait_for_system_ready", return_value=(True, "ready")):
            result = MODULE.ensure_system_service(
                "http://127.0.0.1:5080/api/chat",
                True,
                15.0,
                launch_command,
            )

        self.assertTrue(result["ready"])
        self.assertEqual(result["action"], "start")
        start_mock.assert_called_once_with(launch_command)

    def test_run_route_case_passes_launch_command_to_recovery(self) -> None:
        launch_command = ["phoenix_main.exe", "--components=brain=off"]
        with mock.patch.object(
            MODULE,
            "post_json",
            side_effect=[
                (0, "[WinError 10061] connection refused"),
                (200, '{"ok": true, "result": {"reply": "ready"}}'),
            ],
        ), mock.patch.object(
            MODULE,
            "recover_system_service",
            return_value=(True, "recovered"),
        ) as recover_mock, mock.patch.object(MODULE.time, "sleep", return_value=None):
            result = MODULE.run_route_case(
                MODULE.ROUTE_SYSTEM,
                MODULE.BenchCase("case-1", "quality", "external", "ping", "pong", 0.0, ["test"]),
                1,
                5.0,
                "http://127.0.0.1:5080/api/chat",
                "",
                launch_command,
                "http://127.0.0.1:11434/api/chat",
                "llama3.1:8b",
                "http://127.0.0.1:8082/v1/chat/completions",
                "llamacpp",
                64,
                "urllib",
                None,
                1,
                True,
                15.0,
            )

        self.assertTrue(result.ok)
        recover_mock.assert_called_once_with(
            "http://127.0.0.1:5080/api/chat",
            "",
            5.0,
            "urllib",
            None,
            True,
            15.0,
            launch_command,
        )

    def test_probe_system_chat_retries_transient_llamacpp_status(self) -> None:
        with mock.patch.object(
            MODULE,
            "post_json",
            side_effect=[
                (503, '{"ok": false, "error": "llamacpp bad status"}'),
                (200, '{"ok": true, "result": {"reply": "ready"}}'),
            ],
        ) as post_mock, mock.patch.object(MODULE.time, "sleep", return_value=None):
            ready, err = MODULE.probe_system_chat(
                "http://127.0.0.1:5080/api/chat",
                "local-dev",
                2.0,
                "urllib",
                None,
            )

        self.assertTrue(ready)
        self.assertEqual(err, "")
        self.assertEqual(post_mock.call_count, 2)

    def test_probe_system_chat_does_not_retry_invalid_token(self) -> None:
        with mock.patch.object(
            MODULE,
            "post_json",
            return_value=(401, '{"ok": false, "error": "invalid-token"}'),
        ) as post_mock, mock.patch.object(MODULE.time, "sleep", return_value=None) as sleep_mock:
            ready, err = MODULE.probe_system_chat(
                "http://127.0.0.1:5080/api/chat",
                "bad-token",
                2.0,
                "urllib",
                None,
            )

        self.assertFalse(ready)
        self.assertEqual(err, "invalid-token")
        self.assertEqual(post_mock.call_count, 1)
        sleep_mock.assert_not_called()

    def test_recover_system_service_probes_chat_without_token(self) -> None:
        with mock.patch.object(
            MODULE,
            "ensure_system_service",
            return_value={"ready": True, "action": "start", "error": ""},
        ), mock.patch.object(
            MODULE,
            "probe_system_chat",
            return_value=(True, ""),
        ) as probe_mock:
            ok, message = MODULE.recover_system_service(
                "http://127.0.0.1:5080/api/chat",
                "",
                5.0,
                "urllib",
                None,
                True,
                15.0,
                ["phoenix_main.exe"],
            )

        self.assertTrue(ok)
        self.assertEqual(message, "recovered")
        probe_mock.assert_called_once_with(
            "http://127.0.0.1:5080/api/chat",
            "",
            30.0,
            "urllib",
            None,
        )


if __name__ == "__main__":
    unittest.main()
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "tools" / "computer_shell_bridge.py"


class ComputerShellBridgeTests(unittest.TestCase):
    def run_bridge(self, request, config=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_root = pathlib.Path(tmp_dir)
            request_file = temp_root / "request.json"
            response_file = temp_root / "response.json"
            payload = {
                "request": request,
                "config": config or {},
            }
            request_file.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(BRIDGE), "--request-file", str(request_file), "--response-file", str(response_file)],
                capture_output=True,
                text=True,
                cwd=str(ROOT),
            )
            self.assertTrue(response_file.exists(), msg=completed.stderr)
            result = json.loads(response_file.read_text(encoding="utf-8"))
            return completed.returncode, result

    def test_status_reports_allowed_operations(self):
        exit_code, result = self.run_bridge({"op": "status"}, {"readonly": True, "workingDir": str(ROOT)})
        self.assertEqual(exit_code, 0)
        self.assertTrue(result["ok"])
        self.assertTrue(result["readonly"])
        self.assertIn("list_dir", result["allowedOps"])
        self.assertNotIn("run", result["allowedOps"])

    def test_run_is_blocked_in_readonly_mode(self):
        exit_code, result = self.run_bridge({"op": "run", "command": "echo blocked", "shell": True}, {"readonly": True, "workingDir": str(ROOT)})
        self.assertNotEqual(exit_code, 0)
        self.assertFalse(result["ok"])
        self.assertIn("readonly", result["error"])

    def test_run_executes_argv_in_control_mode(self):
        exit_code, result = self.run_bridge(
            {"op": "run", "argv": [sys.executable, "-c", "print('bridge-ok')"]},
            {"readonly": False, "workingDir": str(ROOT), "timeoutMs": 5000, "maxOutputBytes": 2048},
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(result["ok"])
        self.assertIn("bridge-ok", result["stdout"])

    def test_list_dir_returns_entries(self):
        exit_code, result = self.run_bridge({"op": "list_dir", "path": ".", "limit": 8}, {"readonly": True, "workingDir": str(ROOT)})
        self.assertEqual(exit_code, 0)
        self.assertTrue(result["ok"])
        self.assertIsInstance(result["entries"], list)
        self.assertGreater(len(result["entries"]), 0)


if __name__ == "__main__":
    unittest.main()
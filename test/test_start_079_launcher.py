import importlib.util
import os
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "start_079_launcher.py"
SPEC = importlib.util.spec_from_file_location("start_079_launcher", MODULE_PATH)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class Start079LauncherTests(unittest.TestCase):
    def test_build_command_includes_mechanical_and_addons(self):
        values = launcher.default_option_values()
        values.update(
            {
                "mechanical_mind_enabled": True,
                "mechanical_mind_threshold": 0.61,
                "mechanical_mind_token_threshold": 0.67,
                "persistent_session_memory_enabled": False,
                "builtin_addons": "math",
                "addon_libraries": "addons/custom.dll",
                "computer_shell_enabled": True,
                "computer_shell_readonly": False,
                "extra_args": "--frontend-enabled=true --reasoning-planner-enabled=false",
            }
        )
        options = launcher.LaunchOptions(values)
        command = launcher.build_launch_command(options, ROOT)
        self.assertIn("--mechanical-mind-enabled=true", command)
        self.assertIn("--mechanical-mind-threshold=0.61", command)
        self.assertIn("--mechanical-mind-token-threshold=0.67", command)
        self.assertIn("--builtin-addons=math", command)
        self.assertIn("--persistent-session-memory=false", command)
        self.assertIn("--computer-shell-enabled=true", command)
        self.assertIn("--computer-shell-readonly=false", command)
        self.assertTrue(any(part.startswith("--addon-libraries=") for part in command))
        self.assertIn("--reasoning-planner-enabled=false", command)

    def test_serialize_builtin_addons_returns_none_when_empty(self):
        value = launcher.serialize_builtin_addons("")
        self.assertEqual(value, "none")

    def test_load_profile_merges_legacy_autonomy_args(self):
        original = os.environ.get("PHOENIX_AUTONOMY_ARGS")
        try:
            os.environ["PHOENIX_AUTONOMY_ARGS"] = "--computer-shell-enabled=true"
            options = launcher.load_profile(ROOT)
            self.assertIn("--computer-shell-enabled=true", options.values["extra_args"])
        finally:
            if original is None:
                os.environ.pop("PHOENIX_AUTONOMY_ARGS", None)
            else:
                os.environ["PHOENIX_AUTONOMY_ARGS"] = original


if __name__ == "__main__":
    unittest.main()
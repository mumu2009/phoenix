import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).resolve().parent.parent / "tools" / "external_model_adapter.py"
SPEC = importlib.util.spec_from_file_location("phoenix_external_model_adapter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def arg_value(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


class ExternalModelAdapterTests(unittest.TestCase):
    def test_build_cli_command_uses_10m_yarn_defaults(self) -> None:
        command = MODULE.build_cli_command(
            "llamacpp",
            pathlib.Path("llama-cli.exe"),
            "model.gguf",
            "hello",
            64,
            MODULE.RuntimeLaunchOptions(),
        )

        self.assertEqual(arg_value(command, "--ctx-size"), "10000000")
        self.assertEqual(arg_value(command, "--batch-size"), "512")
        self.assertEqual(arg_value(command, "--ubatch-size"), "128")
        self.assertEqual(arg_value(command, "--rope-scaling"), "yarn")
        self.assertEqual(arg_value(command, "--rope-freq-base"), "0")
        self.assertEqual(arg_value(command, "--rope-freq-scale"), "0")
        self.assertEqual(arg_value(command, "--yarn-orig-ctx"), "4096")
        self.assertEqual(arg_value(command, "--yarn-ext-factor"), "1")
        self.assertEqual(arg_value(command, "--yarn-attn-factor"), "1")
        self.assertEqual(arg_value(command, "--yarn-beta-fast"), "32")
        self.assertEqual(arg_value(command, "--yarn-beta-slow"), "1")
        self.assertIn("-cnv", command)

    def test_build_cli_command_preserves_custom_context_controls(self) -> None:
        options = MODULE.RuntimeLaunchOptions(
            ctx_size=123456,
            batch_size=96,
            ubatch_size=24,
            rope_scaling="linear",
            rope_freq_base=500000.0,
            rope_freq_scale=0.25,
            yarn_orig_ctx=8192,
            yarn_ext_factor=0.5,
            yarn_attn_factor=0.75,
            yarn_beta_fast=16.0,
            yarn_beta_slow=2.0,
        )

        command = MODULE.build_cli_command(
            "bitnet",
            pathlib.Path("llama-cli.exe"),
            "model.gguf",
            "hello",
            32,
            options,
        )

        self.assertEqual(arg_value(command, "--ctx-size"), "123456")
        self.assertEqual(arg_value(command, "--batch-size"), "96")
        self.assertEqual(arg_value(command, "--ubatch-size"), "24")
        self.assertEqual(arg_value(command, "--rope-scaling"), "linear")
        self.assertEqual(arg_value(command, "--rope-freq-base"), "500000")
        self.assertEqual(arg_value(command, "--rope-freq-scale"), "0.25")
        self.assertEqual(arg_value(command, "--yarn-orig-ctx"), "8192")
        self.assertEqual(arg_value(command, "--yarn-ext-factor"), "0.5")
        self.assertEqual(arg_value(command, "--yarn-attn-factor"), "0.75")
        self.assertEqual(arg_value(command, "--yarn-beta-fast"), "16")
        self.assertEqual(arg_value(command, "--yarn-beta-slow"), "2")
        self.assertEqual(command[-2:], ["-p", "hello"])

    def test_build_prompt_from_messages_preserves_world_model_blocks(self) -> None:
        messages = [
            {"role": "system", "content": "Keep the answer factual and concise."},
            {"role": "system", "content": "Phoenix guidance shell.\nworld_scene|summary: camera sees a cat near the fence\ncapture path: micro-mipi-csi\nworld_plan|goal: describe the most actionable event"},
            {"role": "user", "content": "What happened in the video?"},
        ]

        prompt = MODULE.build_prompt_from_messages(messages)

        self.assertIn("System instructions:", prompt)
        self.assertIn("Keep the answer factual and concise.", prompt)
        self.assertIn("World model context:", prompt)
        self.assertIn("world_scene|summary: camera sees a cat near the fence", prompt)
        self.assertIn("capture path: micro-mipi-csi", prompt)
        self.assertIn("User:\nWhat happened in the video?", prompt)
        self.assertTrue(prompt.rstrip().endswith("Assistant:"))

    def test_extract_prompt_and_max_tokens_support_openai_chat_payload(self) -> None:
        payload = {
            "messages": [
                {"role": "system", "content": "Phoenix guidance shell.\nworld_recent|vision: person opens the gate"},
                {"role": "user", "content": [{"type": "text", "text": "Summarize the scene."}]},
            ],
            "max_tokens": 48,
        }

        prompt = MODULE.extract_prompt(payload)
        max_tokens = MODULE.extract_max_tokens(payload)

        self.assertIn("World model context:", prompt)
        self.assertIn("world_recent|vision: person opens the gate", prompt)
        self.assertIn("User:\nSummarize the scene.", prompt)
        self.assertEqual(max_tokens, 48)

    def test_build_success_payload_matches_openai_chat_shape(self) -> None:
        payload = MODULE.build_success_payload("/v1/chat/completions", "llamacpp", "done")

        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "llamacpp")
        self.assertEqual(payload["choices"][0]["message"]["content"], "done")

    def test_build_success_payload_matches_adapter_chat_shape(self) -> None:
        payload = MODULE.build_success_payload("/api/chat", "llamacpp", "done")

        self.assertEqual(payload["model"], "llamacpp")
        self.assertEqual(payload["message"]["content"], "done")


if __name__ == "__main__":
    unittest.main()
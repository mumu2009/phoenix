import pathlib
import unittest


CONFIG_FILE = pathlib.Path(__file__).resolve().parent.parent / "main_hub_parts" / "001_struct_config.inc"


class LlamaCppContextDefaultTests(unittest.TestCase):
    def test_default_config_exposes_10m_context_controls(self) -> None:
        text = CONFIG_FILE.read_text(encoding="utf-8")

        self.assertIn("int llamaCppCtxSize{10000000};", text)
        self.assertIn("int llamaCppBatchSize{512};", text)
        self.assertIn("int llamaCppUbatchSize{128};", text)
        self.assertIn("std::string llamaCppRopeScaling{\"yarn\"};", text)
        self.assertIn("--ctx-size {ctx_size}", text)
        self.assertIn("--batch-size {batch_size}", text)
        self.assertIn("--ubatch-size {ubatch_size}", text)
        self.assertIn("--rope-scaling {rope_scaling}", text)
        self.assertIn("--rope-freq-base {rope_freq_base}", text)
        self.assertIn("--rope-freq-scale {rope_freq_scale}", text)
        self.assertIn("--yarn-orig-ctx {yarn_orig_ctx}", text)
        self.assertIn("--yarn-ext-factor {yarn_ext_factor}", text)
        self.assertIn("--yarn-attn-factor {yarn_attn_factor}", text)
        self.assertIn("--yarn-beta-fast {yarn_beta_fast}", text)
        self.assertIn("--yarn-beta-slow {yarn_beta_slow}", text)


if __name__ == "__main__":
    unittest.main()
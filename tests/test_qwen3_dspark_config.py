import unittest
from pathlib import Path

from tests.helpers import load_speculative_module

ROOT = Path(__file__).resolve().parents[1]


class Qwen3DSparkConfigTest(unittest.TestCase):

    def test_benchmark_defaults_to_qwen3_4b(self):
        bench_text = (ROOT / "bench.py").read_text()

        self.assertIn('DEFAULT_MODEL_REPO = "Qwen/Qwen3-4B"', bench_text)
        self.assertIn('return "~/huggingface/Qwen3-4B/"', bench_text)

    def test_dspark_defaults_match_deepspec_qwen3_4b(self):
        speculative = load_speculative_module()

        config = speculative.DSparkSpeculativeConfig()

        self.assertEqual(config.target_model, "Qwen/Qwen3-4B")
        self.assertEqual(config.draft_model, "deepseek-ai/dspark_qwen3_4b_block7")
        self.assertEqual(config.num_speculative_tokens, 7)
        self.assertEqual(config.target_layer_ids, (1, 9, 17, 25, 33))
        self.assertEqual(config.num_anchors, 512)
        self.assertEqual(config.markov_rank, 256)
        self.assertTrue(config.confidence_head_with_markov)

    def test_speculative_submodule_imports_without_llm_runtime_dependencies(self):
        import nanovllm.speculative as speculative

        self.assertEqual(speculative.DSparkSpeculativeConfig().draft_model, "deepseek-ai/dspark_qwen3_4b_block7")


if __name__ == "__main__":
    unittest.main()

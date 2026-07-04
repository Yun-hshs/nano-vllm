import unittest
from pathlib import Path

from tests.helpers import load_speculative_module


ROOT = Path(__file__).resolve().parents[1]


class EngineSpeculativeHookTest(unittest.TestCase):

    def test_runtime_config_defaults_to_disabled(self):
        speculative = load_speculative_module()

        config = speculative.SpeculativeRuntimeConfig()

        self.assertFalse(config.enabled)
        self.assertEqual(config.draft_type, "ngram")
        self.assertEqual(config.draft_model, "Qwen/Qwen3-0.6B")

    def test_runtime_config_accepts_nano_draft_type(self):
        speculative = load_speculative_module()

        config = speculative.SpeculativeRuntimeConfig(
            enabled=True,
            draft_type="nano",
            draft_model="/models/Qwen3-0.6B",
        )

        self.assertEqual(config.draft_type, "nano")
        self.assertEqual(config.draft_model, "/models/Qwen3-0.6B")

    def test_engine_step_keeps_default_non_speculative_path(self):
        source = (ROOT / "nanovllm" / "engine" / "llm_engine.py").read_text()

        self.assertIn("if self.speculative_config.enabled:", source)
        self.assertIn("return self._step_speculative()", source)
        self.assertIn("return self._step_regular()", source)

    def test_engine_speculative_path_is_experimental_but_implemented(self):
        source = (ROOT / "nanovllm" / "engine" / "llm_engine.py").read_text()

        self.assertIn("NGramDraftGenerator", source)
        self.assertIn("HFDraftGenerator", source)
        self.assertIn("NanoDraftGenerator", source)
        self.assertIn('draft_type == "hf"', source)
        self.assertIn('draft_type == "nano"', source)
        self.assertIn('self.model_runner.call("run_speculative"', source)
        self.assertIn("last_draft_rebuilds", source)
        self.assertIn("last_draft_cached_steps", source)
        self.assertIn("draft_propose_ms", source)
        self.assertIn("draft_prefill_ms", source)
        self.assertIn("draft_cache_extend_ms", source)
        self.assertIn("draft_decode_graph_ms", source)
        self.assertIn("draft_decode_eager_ms", source)
        self.assertIn("draft_token_select_ms", source)
        self.assertIn("target_verify_ms", source)
        self.assertIn("target_forward_ms", source)
        self.assertIn("target_argmax_ms", source)
        self.assertIn("target_compare_ms", source)
        self.assertIn("append_tokens_ms", source)
        self.assertIn("speculative_step_ms", source)
        self.assertIn("target_greedy_forwards", source)
        self.assertIn("self.speculative_stats.record_step", source)
        self.assertNotIn("raise NotImplementedError", source)

    def test_nano_draft_generator_uses_native_qwen_model_and_kv_cache(self):
        source = (ROOT / "nanovllm" / "engine" / "draft_runner.py").read_text()

        self.assertIn("class NanoDraftGenerator", source)
        self.assertIn("Qwen3ForCausalLM", source)
        self.assertIn("load_model", source)
        self.assertIn("set_context(True", source)
        self.assertIn("set_context(False", source)
        self.assertIn("k_cache", source)
        self.assertIn("def _last_logits", source)
        self.assertIn("logits.dim() == 3", source)
        self.assertIn("def _disable_torch_compile", source)
        self.assertIn("__wrapped__", source)
        self.assertIn("def _capture_decode_graph", source)
        self.assertIn("torch.cuda.CUDAGraph", source)
        self.assertIn("graph.replay()", source)
        self.assertIn("last_draft_graph_enabled", source)
        self.assertIn("last_draft_graph_replays", source)
        self.assertIn("last_draft_prefill_ms", source)
        self.assertIn("last_draft_cache_extend_ms", source)
        self.assertIn("last_draft_decode_graph_ms", source)
        self.assertIn("last_draft_decode_eager_ms", source)
        self.assertIn("last_draft_token_select_ms", source)

    def test_hf_draft_generator_uses_past_key_values_cache(self):
        source = (ROOT / "nanovllm" / "speculative.py").read_text()

        self.assertIn("past_key_values", source)
        self.assertIn("_rebuild_cache", source)
        self.assertIn("_extend_cache", source)
        self.assertIn("last_draft_cached_steps", source)

    def test_model_runner_has_target_batched_speculative_verification(self):
        source = (ROOT / "nanovllm" / "engine" / "model_runner.py").read_text()

        self.assertIn("@torch.inference_mode()\n    def run_speculative", source)
        self.assertIn("def run_speculative", source)
        self.assertIn("[seq.last_token] + draft_token_ids", source)
        self.assertIn("set_context(True", source)
        self.assertIn("hidden_states = self.model(input_ids, positions)", source)
        self.assertIn("reset_context()", source)
        self.assertIn("should_use_greedy_verification", source)
        self.assertIn("logits.float().argmax(dim=-1)", source)
        self.assertIn("self.sampler(logits, temperatures).tolist()", source)
        self.assertIn("last_speculative_forward_ms", source)
        self.assertIn("last_speculative_argmax_ms", source)

    def test_benchmark_can_enable_engine_speculative_mode(self):
        source = (ROOT / "bench.py").read_text()

        self.assertIn("--enable-engine-speculative", source)
        self.assertIn("--draft-type", source)
        self.assertIn('choices=("ngram", "hf", "nano")', source)
        self.assertIn("--gpu-memory-utilization", source)
        self.assertIn("--temperature", source)
        self.assertIn("draft_rebuilds", source)
        self.assertIn("draft_cached_steps", source)
        self.assertIn("draft_graph_enabled", source)
        self.assertIn("draft_graph_replays", source)
        self.assertIn("target_greedy_forwards", source)
        self.assertIn("target_verify_ms", source)
        self.assertIn("draft_decode_graph_ms", source)
        self.assertIn("target_forward_ms", source)
        self.assertIn("--speculative-greedy-temperature", source)
        self.assertIn("SpeculativeRuntimeConfig", source)
        self.assertIn("speculative_config=speculative_config", source)
        self.assertIn("llm.speculative_stats", source)

    def test_summarize_benchmarks_reports_k_sweep(self):
        source = (ROOT / "scripts" / "summarize_benchmarks.py").read_text()

        self.assertIn("benchmark_result_from_dict", source)
        self.assertIn("graph enabled", source)
        self.assertIn("graph replays", source)
        self.assertIn("target verify ms", source)
        self.assertIn("draft graph ms", source)
        self.assertIn("target forward ms", source)
        self.assertIn("greedy forwards", source)
        self.assertIn("speedup", source)


if __name__ == "__main__":
    unittest.main()

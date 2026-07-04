import unittest

from tests.helpers import load_speculative_module


class SpeculativeBenchmarkTest(unittest.TestCase):

    def test_comparison_reports_speedup_and_acceptance_rate(self):
        speculative = load_speculative_module()

        baseline = speculative.BenchmarkResult(
            mode="baseline",
            output_tokens=100,
            elapsed_seconds=10.0,
        )
        dspark = speculative.BenchmarkResult(
            mode="dspark",
            output_tokens=100,
            elapsed_seconds=4.0,
            proposed_tokens=80,
            accepted_tokens=60,
        )

        comparison = speculative.compare_benchmarks(baseline, dspark)

        self.assertEqual(comparison.baseline_tokens_per_second, 10.0)
        self.assertEqual(comparison.speculative_tokens_per_second, 25.0)
        self.assertEqual(comparison.speedup, 2.5)
        self.assertEqual(comparison.acceptance_rate, 0.75)

    def test_speculative_stats_accumulate_engine_metrics(self):
        speculative = load_speculative_module()

        stats = speculative.SpeculativeStats()
        stats.record_step(
            proposed_tokens=3,
            accepted_tokens=2,
            target_forwards=2,
            emitted_tokens=3,
            draft_forwards=3,
            draft_rebuilds=1,
            draft_cached_steps=10,
            draft_graph_enabled=True,
            draft_graph_replays=2,
            target_greedy_forwards=1,
            target_greedy_tokens=4,
            draft_propose_ms=1.5,
            draft_prefill_ms=0.2,
            draft_cache_extend_ms=0.3,
            draft_decode_graph_ms=0.4,
            draft_decode_eager_ms=0.5,
            draft_token_select_ms=0.1,
            target_verify_ms=2.5,
            target_forward_ms=1.1,
            target_argmax_ms=0.2,
            target_compare_ms=0.3,
            append_tokens_ms=0.25,
            speculative_step_ms=4.5,
        )
        stats.record_step(
            proposed_tokens=1,
            accepted_tokens=0,
            target_forwards=1,
            emitted_tokens=1,
            draft_forwards=1,
            draft_rebuilds=0,
            draft_cached_steps=14,
            draft_graph_enabled=False,
            draft_graph_replays=0,
            target_greedy_forwards=0,
            target_greedy_tokens=0,
            draft_propose_ms=0.5,
            draft_prefill_ms=0.1,
            draft_cache_extend_ms=0.2,
            draft_decode_graph_ms=0.3,
            draft_decode_eager_ms=0.4,
            draft_token_select_ms=0.05,
            target_verify_ms=1.0,
            target_forward_ms=0.7,
            target_argmax_ms=0.1,
            target_compare_ms=0.2,
            append_tokens_ms=0.10,
            speculative_step_ms=2.0,
        )

        self.assertEqual(stats.proposed_tokens, 4)
        self.assertEqual(stats.accepted_tokens, 2)
        self.assertEqual(stats.target_forwards, 3)
        self.assertEqual(stats.draft_forwards, 4)
        self.assertEqual(stats.draft_rebuilds, 1)
        self.assertEqual(stats.draft_cached_steps, 24)
        self.assertTrue(stats.draft_graph_enabled)
        self.assertEqual(stats.draft_graph_replays, 2)
        self.assertEqual(stats.target_greedy_forwards, 1)
        self.assertEqual(stats.target_greedy_tokens, 4)
        self.assertEqual(stats.draft_propose_ms, 2.0)
        self.assertAlmostEqual(stats.draft_prefill_ms, 0.3)
        self.assertAlmostEqual(stats.draft_cache_extend_ms, 0.5)
        self.assertAlmostEqual(stats.draft_decode_graph_ms, 0.7)
        self.assertAlmostEqual(stats.draft_decode_eager_ms, 0.9)
        self.assertAlmostEqual(stats.draft_token_select_ms, 0.15)
        self.assertEqual(stats.target_verify_ms, 3.5)
        self.assertAlmostEqual(stats.target_forward_ms, 1.8)
        self.assertAlmostEqual(stats.target_argmax_ms, 0.3)
        self.assertAlmostEqual(stats.target_compare_ms, 0.5)
        self.assertAlmostEqual(stats.append_tokens_ms, 0.35)
        self.assertEqual(stats.speculative_step_ms, 6.5)
        self.assertEqual(stats.emitted_tokens, 4)
        self.assertEqual(stats.acceptance_lengths, [2, 0])
        self.assertEqual(stats.acceptance_rate, 0.5)
        self.assertEqual(stats.mean_acceptance_length, 1.0)

        stats.reset()

        self.assertEqual(stats.proposed_tokens, 0)
        self.assertEqual(stats.accepted_tokens, 0)
        self.assertEqual(stats.target_forwards, 0)
        self.assertEqual(stats.draft_forwards, 0)
        self.assertEqual(stats.draft_rebuilds, 0)
        self.assertEqual(stats.draft_cached_steps, 0)
        self.assertFalse(stats.draft_graph_enabled)
        self.assertEqual(stats.draft_graph_replays, 0)
        self.assertEqual(stats.target_greedy_forwards, 0)
        self.assertEqual(stats.target_greedy_tokens, 0)
        self.assertEqual(stats.draft_propose_ms, 0.0)
        self.assertEqual(stats.draft_prefill_ms, 0.0)
        self.assertEqual(stats.draft_cache_extend_ms, 0.0)
        self.assertEqual(stats.draft_decode_graph_ms, 0.0)
        self.assertEqual(stats.draft_decode_eager_ms, 0.0)
        self.assertEqual(stats.draft_token_select_ms, 0.0)
        self.assertEqual(stats.target_verify_ms, 0.0)
        self.assertEqual(stats.target_forward_ms, 0.0)
        self.assertEqual(stats.target_argmax_ms, 0.0)
        self.assertEqual(stats.target_compare_ms, 0.0)
        self.assertEqual(stats.append_tokens_ms, 0.0)
        self.assertEqual(stats.speculative_step_ms, 0.0)
        self.assertEqual(stats.emitted_tokens, 0)
        self.assertEqual(stats.acceptance_lengths, [])

    def test_low_temperature_uses_greedy_speculative_verification(self):
        speculative = load_speculative_module()

        config = speculative.SpeculativeRuntimeConfig(greedy_verification_temperature=0.011)

        self.assertTrue(speculative.should_use_greedy_verification(0.01, config.greedy_verification_temperature))
        self.assertFalse(speculative.should_use_greedy_verification(0.6, config.greedy_verification_temperature))

    def test_runtime_config_supports_hf_draft_model(self):
        speculative = load_speculative_module()

        config = speculative.SpeculativeRuntimeConfig(
            enabled=True,
            draft_type="hf",
            draft_model="/models/Qwen3-0.6B",
            draft_device="cuda",
            draft_dtype="bfloat16",
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.draft_type, "hf")
        self.assertEqual(config.draft_model, "/models/Qwen3-0.6B")
        self.assertEqual(config.draft_device, "cuda")
        self.assertEqual(config.draft_dtype, "bfloat16")

    def test_accepts_confident_prefix_until_first_rejection(self):
        speculative = load_speculative_module()

        proposal = speculative.DraftProposal(token_ids=[10, 11, 12, 13], confidence=[0.9, 0.8, 0.2, 0.7])
        accepted = speculative.verify_draft_tokens(
            proposal,
            target_token_ids=[10, 11, 99, 13],
            confidence_threshold=0.5,
        )

        self.assertEqual(accepted.accepted_token_ids, [10, 11])
        self.assertEqual(accepted.rejected_token_id, 12)
        self.assertEqual(accepted.proposed_tokens, 4)
        self.assertEqual(accepted.accepted_tokens, 2)

    def test_ngram_draft_generator_proposes_from_repeated_prefix(self):
        speculative = load_speculative_module()

        generator = speculative.NGramDraftGenerator(ngram_size=2)
        proposal = generator.propose([1, 2, 3, 1, 2], num_tokens=2)

        self.assertEqual(proposal.token_ids, [3])
        self.assertEqual(proposal.confidence, [1.0])

    def test_hf_draft_generator_extracts_generated_suffix(self):
        speculative = load_speculative_module()

        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return list(self.values)

        class FakeGenerated:
            def __getitem__(self, key):
                self.last_key = key
                return FakeTensor([42, 43, 44])

        class FakeModel:
            generation_config = type("GenerationConfig", (), {"eos_token_id": 0})()

            def eval(self):
                return self

            def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id, attention_mask):
                self.max_new_tokens = max_new_tokens
                self.do_sample = do_sample
                self.attention_mask = attention_mask
                return FakeGenerated()

        fake_model = FakeModel()
        generator = speculative.HFDraftGenerator.from_model(fake_model)
        proposal = generator.propose([10, 11], num_tokens=3)

        self.assertTrue(hasattr(fake_model, "attention_mask"))
        self.assertEqual(generator.last_draft_forwards, 1)
        self.assertEqual(generator.last_draft_rebuilds, 1)
        self.assertEqual(generator.last_draft_cached_steps, 0)
        self.assertEqual(proposal.token_ids, [42, 43, 44])
        self.assertEqual(proposal.confidence, [1.0, 1.0, 1.0])

    def test_hf_draft_generator_uses_scalar_pad_token_when_eos_is_list(self):
        speculative = load_speculative_module()

        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return list(self.values)

        class FakeGenerated:
            def __getitem__(self, key):
                return FakeTensor([45])

        class FakeModel:
            generation_config = type("GenerationConfig", (), {"eos_token_id": [151645, 151643]})()

            def eval(self):
                return self

            def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id, attention_mask):
                self.pad_token_id = pad_token_id
                return FakeGenerated()

        fake_model = FakeModel()
        generator = speculative.HFDraftGenerator.from_model(fake_model)
        proposal = generator.propose([10, 11], num_tokens=1)

        self.assertEqual(fake_model.pad_token_id, 151645)
        self.assertEqual(proposal.token_ids, [45])

    def test_target_only_experiment_collects_acceptance_and_forward_stats(self):
        speculative = load_speculative_module()

        def target_fn(prefix, num_tokens):
            full_sequence = [1, 2, 3, 4, 5, 6]
            return full_sequence[len(prefix):len(prefix) + num_tokens]

        def draft_fn(prefix, num_tokens):
            return speculative.DraftProposal([3, 4, 9][:num_tokens], [1.0, 1.0, 1.0][:num_tokens])

        result = speculative.run_target_only_speculative_experiment(
            prefix_token_ids=[1, 2],
            max_tokens=4,
            draft_fn=draft_fn,
            target_fn=target_fn,
            config=speculative.DSparkSpeculativeConfig(num_speculative_tokens=3),
            elapsed_seconds=2.0,
        )

        self.assertEqual(result.output_token_ids, [3, 4, 5, 6])
        self.assertEqual(result.proposed_tokens, 4)
        self.assertEqual(result.accepted_tokens, 2)
        self.assertEqual(result.target_forwards, 2)
        self.assertEqual(result.acceptance_lengths, [2, 0])
        self.assertEqual(result.mean_acceptance_length, 1.0)
        self.assertEqual(result.tokens_per_second, 2.0)

    def test_ngram_target_only_experiment_uses_token_trace(self):
        speculative = load_speculative_module()

        result = speculative.run_ngram_target_only_experiment(
            token_trace=[1, 2, 3, 1, 2, 3, 4],
            prefix_len=5,
            max_tokens=2,
            ngram_size=2,
            config=speculative.DSparkSpeculativeConfig(num_speculative_tokens=2),
            elapsed_seconds=1.0,
        )

        self.assertEqual(result.output_token_ids, [3, 4])
        self.assertEqual(result.proposed_tokens, 2)
        self.assertEqual(result.accepted_tokens, 1)
        self.assertEqual(result.target_forwards, 2)
        self.assertEqual(result.tokens_per_second, 2.0)

    def test_formats_markdown_comparison_report(self):
        speculative = load_speculative_module()

        baseline = speculative.BenchmarkResult(mode="baseline", output_tokens=120, elapsed_seconds=12.0)
        dspark = speculative.BenchmarkResult(
            mode="dspark",
            output_tokens=120,
            elapsed_seconds=8.0,
            proposed_tokens=70,
            accepted_tokens=49,
        )

        report = speculative.format_benchmark_comparison(baseline, dspark)

        self.assertIn("| baseline | 120 | 12.00 | 10.00 |", report)
        self.assertIn("| dspark | 120 | 8.00 | 15.00 | 70.00% |", report)
        self.assertIn("Speedup: 1.50x", report)

    def test_benchmark_result_serializes_engine_speculative_stats(self):
        speculative = load_speculative_module()

        result = speculative.BenchmarkResult(
            mode="engine-speculative",
            output_tokens=12,
            elapsed_seconds=3.0,
            proposed_tokens=10,
            accepted_tokens=6,
            draft_forwards=4,
            draft_rebuilds=2,
            draft_cached_steps=128,
            draft_graph_enabled=True,
            draft_graph_replays=96,
            target_greedy_forwards=3,
            target_greedy_tokens=24,
            draft_propose_ms=1.25,
            draft_prefill_ms=0.25,
            draft_cache_extend_ms=0.5,
            draft_decode_graph_ms=0.75,
            draft_decode_eager_ms=1.0,
            draft_token_select_ms=0.125,
            target_verify_ms=3.5,
            target_forward_ms=2.0,
            target_argmax_ms=0.75,
            target_compare_ms=0.25,
            append_tokens_ms=0.5,
            speculative_step_ms=5.75,
            target_forwards=8,
            acceptance_lengths=[2, 4, 0],
        )

        data = result.to_dict()
        restored = speculative.benchmark_result_from_dict(data)

        self.assertEqual(data["target_forwards"], 8)
        self.assertEqual(data["draft_forwards"], 4)
        self.assertEqual(data["draft_rebuilds"], 2)
        self.assertEqual(data["draft_cached_steps"], 128)
        self.assertEqual(data["draft_graph_enabled"], True)
        self.assertEqual(data["draft_graph_replays"], 96)
        self.assertEqual(data["target_greedy_forwards"], 3)
        self.assertEqual(data["target_greedy_tokens"], 24)
        self.assertEqual(data["draft_propose_ms"], 1.25)
        self.assertEqual(data["draft_prefill_ms"], 0.25)
        self.assertEqual(data["draft_cache_extend_ms"], 0.5)
        self.assertEqual(data["draft_decode_graph_ms"], 0.75)
        self.assertEqual(data["draft_decode_eager_ms"], 1.0)
        self.assertEqual(data["draft_token_select_ms"], 0.125)
        self.assertEqual(data["target_verify_ms"], 3.5)
        self.assertEqual(data["target_forward_ms"], 2.0)
        self.assertEqual(data["target_argmax_ms"], 0.75)
        self.assertEqual(data["target_compare_ms"], 0.25)
        self.assertEqual(data["append_tokens_ms"], 0.5)
        self.assertEqual(data["speculative_step_ms"], 5.75)
        self.assertEqual(data["acceptance_lengths"], [2, 4, 0])
        self.assertEqual(result.mean_acceptance_length, 2.0)
        self.assertEqual(restored.draft_forwards, 4)
        self.assertEqual(restored.draft_rebuilds, 2)
        self.assertEqual(restored.draft_cached_steps, 128)
        self.assertTrue(restored.draft_graph_enabled)
        self.assertEqual(restored.draft_graph_replays, 96)
        self.assertEqual(restored.target_greedy_forwards, 3)
        self.assertEqual(restored.target_greedy_tokens, 24)
        self.assertEqual(restored.draft_propose_ms, 1.25)
        self.assertEqual(restored.draft_prefill_ms, 0.25)
        self.assertEqual(restored.draft_cache_extend_ms, 0.5)
        self.assertEqual(restored.draft_decode_graph_ms, 0.75)
        self.assertEqual(restored.draft_decode_eager_ms, 1.0)
        self.assertEqual(restored.draft_token_select_ms, 0.125)
        self.assertEqual(restored.target_verify_ms, 3.5)
        self.assertEqual(restored.target_forward_ms, 2.0)
        self.assertEqual(restored.target_argmax_ms, 0.75)
        self.assertEqual(restored.target_compare_ms, 0.25)
        self.assertEqual(restored.append_tokens_ms, 0.5)
        self.assertEqual(restored.speculative_step_ms, 5.75)
        self.assertEqual(restored.target_forwards, 8)
        self.assertEqual(restored.acceptance_lengths, [2, 4, 0])

    def test_formats_markdown_report_with_graph_stats(self):
        speculative = load_speculative_module()

        baseline = speculative.BenchmarkResult(mode="baseline", output_tokens=100, elapsed_seconds=1.0)
        graph = speculative.BenchmarkResult(
            mode="engine-speculative",
            output_tokens=100,
            elapsed_seconds=0.8,
            proposed_tokens=50,
            accepted_tokens=40,
            draft_graph_enabled=True,
            draft_graph_replays=60,
        )

        report = speculative.format_benchmark_comparison(baseline, graph)

        self.assertIn("graph enabled", report)
        self.assertIn("graph replays", report)
        self.assertIn("target verify ms", report)
        self.assertIn("draft graph ms", report)
        self.assertIn("target forward ms", report)
        self.assertIn("greedy forwards", report)
        self.assertIn("| engine-speculative |", report)
        self.assertIn("| yes | 60 |", report)


if __name__ == "__main__":
    unittest.main()

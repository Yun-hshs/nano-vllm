import os
import time
import argparse
import json
from random import randint, seed

from nanovllm.speculative import BenchmarkResult, DSparkSpeculativeConfig, SpeculativeRuntimeConfig


DEFAULT_MODEL_REPO = "Qwen/Qwen3-4B"


def default_model_path() -> str:
    return "~/huggingface/Qwen3-4B/"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Nano-vLLM with Qwen3-4B.")
    parser.add_argument("--model", default=default_model_path())
    parser.add_argument("--mode", choices=("baseline", "dspark"), default="baseline")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--draft-type", choices=("ngram", "hf", "nano"), default=SpeculativeRuntimeConfig().draft_type)
    parser.add_argument("--draft-model", default=SpeculativeRuntimeConfig().draft_model)
    parser.add_argument("--draft-device", default=SpeculativeRuntimeConfig().draft_device)
    parser.add_argument("--draft-dtype", default=SpeculativeRuntimeConfig().draft_dtype)
    parser.add_argument("--num-speculative-tokens", type=int, default=DSparkSpeculativeConfig().num_speculative_tokens)
    parser.add_argument("--enable-engine-speculative", action="store_true")
    parser.add_argument("--ngram-size", type=int, default=SpeculativeRuntimeConfig().ngram_size)
    parser.add_argument(
        "--speculative-greedy-temperature",
        type=float,
        default=SpeculativeRuntimeConfig().greedy_verification_temperature,
        help="Use argmax target verification when sampling temperature is at or below this value. Set 0 to disable.",
    )
    parser.add_argument("--out", help="Optional JSON output path for benchmark comparison.")
    return parser.parse_args()


def run_benchmark(args) -> BenchmarkResult:
    from nanovllm import LLM, SamplingParams

    seed(args.seed)
    path = os.path.expanduser(args.model)
    speculative_config = SpeculativeRuntimeConfig(
        enabled=args.enable_engine_speculative,
        draft_type=args.draft_type,
        draft_model=os.path.expanduser(args.draft_model),
        draft_device=args.draft_device,
        draft_dtype=args.draft_dtype,
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_size=args.ngram_size,
        greedy_verification_temperature=args.speculative_greedy_temperature,
    )
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    if args.mode == "dspark":
        spec_config = DSparkSpeculativeConfig(
            target_model=DEFAULT_MODEL_REPO,
            draft_model=args.draft_model,
            num_speculative_tokens=args.num_speculative_tokens,
        )
        print(
            "DSpark speculative config: "
            f"target={spec_config.target_model}, "
            f"draft={spec_config.draft_model}, "
            f"num_speculative_tokens={spec_config.num_speculative_tokens}"
        )
        print(
            "Note: this benchmark records DSpark settings and throughput in a "
            "comparable JSON format. Native ModelRunner execution of the DSpark "
            "drafter still needs a DSpark model adapter."
        )
    if args.enable_engine_speculative:
        print(
            "Engine speculative mode: enabled "
            f"(draft_type={args.draft_type}, "
            f"draft_model={args.draft_model}, "
            f"ngram_size={args.ngram_size}, "
            f"num_speculative_tokens={args.num_speculative_tokens})."
        )

    llm.generate(["Benchmark: "], SamplingParams())
    llm.speculative_stats.reset()
    start = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - start
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    stats = llm.speculative_stats
    mode = "engine-speculative" if args.enable_engine_speculative else args.mode
    return BenchmarkResult(
        mode=mode,
        output_tokens=total_tokens,
        elapsed_seconds=elapsed,
        proposed_tokens=stats.proposed_tokens,
        accepted_tokens=stats.accepted_tokens,
        draft_forwards=stats.draft_forwards,
        draft_rebuilds=stats.draft_rebuilds,
        draft_cached_steps=stats.draft_cached_steps,
        draft_graph_enabled=stats.draft_graph_enabled,
        draft_graph_replays=stats.draft_graph_replays,
        target_forwards=stats.target_forwards,
        target_greedy_forwards=stats.target_greedy_forwards,
        target_greedy_tokens=stats.target_greedy_tokens,
        draft_propose_ms=stats.draft_propose_ms,
        draft_prefill_ms=stats.draft_prefill_ms,
        draft_cache_extend_ms=stats.draft_cache_extend_ms,
        draft_decode_graph_ms=stats.draft_decode_graph_ms,
        draft_decode_eager_ms=stats.draft_decode_eager_ms,
        draft_token_select_ms=stats.draft_token_select_ms,
        target_verify_ms=stats.target_verify_ms,
        target_forward_ms=stats.target_forward_ms,
        target_argmax_ms=stats.target_argmax_ms,
        target_compare_ms=stats.target_compare_ms,
        append_tokens_ms=stats.append_tokens_ms,
        speculative_step_ms=stats.speculative_step_ms,
        acceptance_lengths=list(stats.acceptance_lengths),
    )


def main():
    args = parse_args()
    result = run_benchmark(args)
    print(
        f"Mode: {result.mode}, Total: {result.output_tokens}tok, "
        f"Time: {result.elapsed_seconds:.2f}s, "
        f"Throughput: {result.tokens_per_second:.2f}tok/s"
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

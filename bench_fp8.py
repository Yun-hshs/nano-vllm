#!/usr/bin/env python3
"""FP8 E4M3 KV cache benchmark.

For each precision (FP16 baseline vs FP8 E4M3) this reports:
  * num_kvcache_blocks / KV bytes          -> KV cache capacity
  * max no-preemption concurrency          -> num_blocks // ceil(max_model_len/block_size)
  * end-to-end output throughput           -> tok/s on the standard mixed workload

Each precision runs in its own subprocess so the GPU/cache state of one run
cannot contaminate the next, and the parent prints the ratios.

    python bench_fp8.py --model ~/huggingface/Qwen3-0.6B --mode all
    python bench_fp8.py --model ... --mode capacity      # fast
    python bench_fp8.py --model ... --mode throughput
"""

import argparse
import json
import math
import subprocess
import sys
import time
from random import randint, seed

RESULT_PREFIX = "NANOVLLM_RESULT "


def build_llm(args, fp8):
    from nanovllm import LLM
    return LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        fp8_kvcache=fp8,
        fp8_kv_calib_tokens=args.calib_tokens,
    )


def measure(args, fp8):
    from nanovllm import SamplingParams

    llm = build_llm(args, fp8)
    cfg = llm.model_runner.config
    block_size = cfg.kvcache_block_size
    blocks_per_seq = math.ceil(args.max_model_len / block_size)
    kv = llm.model_runner.kv_cache
    result = {
        "fp8": fp8,
        "num_kvcache_blocks": cfg.num_kvcache_blocks,
        "block_size": block_size,
        "blocks_per_seq": blocks_per_seq,
        "max_no_preempt_concurrency": cfg.num_kvcache_blocks // blocks_per_seq,
        "kv_cache_gib": kv.numel() * kv.element_size() / 2 ** 30,
    }
    if args.fp8:
        vsc = llm.model_runner.v_scale_cache
        result["v_scale_cache_mib"] = vsc.numel() * vsc.element_size() / 2 ** 20

    if args.with_throughput:
        seed(0)
        if args.fixed_lengths:
            # every sequence reaches max_model_len, so the KV limit is what
            # caps concurrency (random lengths leave capacity unused)
            prompts = [[randint(0, 10000) for _ in range(args.max_input_len)] for _ in range(args.num_seqs)]
            params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_output_len) for _ in range(args.num_seqs)]
        else:
            prompts = [[randint(0, 10000) for _ in range(randint(100, args.max_input_len))] for _ in range(args.num_seqs)]
            params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len)) for _ in range(args.num_seqs)]

        llm.generate(["Benchmark: "], SamplingParams())

        stats = {"max_running": 0}
        original_step = llm.step

        def instrumented_step():
            out = original_step()
            stats["max_running"] = max(stats["max_running"], len(llm.scheduler.running))
            return out

        llm.step = instrumented_step
        start = time.time()
        llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - start
        total_tokens = sum(p.max_tokens for p in params)
        result.update(
            total_tokens=total_tokens,
            elapsed_s=elapsed,
            throughput_tok_s=total_tokens / elapsed,
            max_running_observed=stats["max_running"],
            preemptions=llm.scheduler.num_preemptions,
        )

    print(RESULT_PREFIX + json.dumps(result))


def run_mode(args, fp8):
    cmd = [
        sys.executable, __file__,
        "--mode", "single",
        "--fp8", "1" if fp8 else "0",
        "--model", args.model,
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--calib-tokens", str(args.calib_tokens),
        "--num-seqs", str(args.num_seqs),
        "--max-input-len", str(args.max_input_len),
        "--max-output-len", str(args.max_output_len),
    ]
    if args.mode in ("throughput", "all"):
        cmd.append("--with-throughput")
    if args.fixed_lengths:
        cmd.append("--fixed-lengths")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    label = "fp8" if fp8 else "fp16"
    print(f"[bench] running {label} ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr)
        raise SystemExit(f"benchmark subprocess failed for fp8={fp8}")
    line = [l for l in proc.stdout.splitlines() if l.startswith(RESULT_PREFIX)][-1]
    print(f"[bench] {label} done", flush=True)
    return json.loads(line[len(RESULT_PREFIX):])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--mode", choices=["capacity", "throughput", "all", "single"], default="all")
    parser.add_argument("--fp8", type=int, choices=[0, 1], default=0)
    parser.add_argument("--with-throughput", action="store_true", help="internal: run the throughput workload")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--calib-tokens", type=int, default=512)
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--fixed-lengths", action="store_true",
                        help="use exactly max-input-len/max-output-len so all sequences reach max context")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    args.model = __import__("os").path.expanduser(args.model)

    if args.mode == "single":
        measure(args, bool(args.fp8))
        return

    base = run_mode(args, fp8=False)
    fp8 = run_mode(args, fp8=True)

    print("\n=== FP8 E4M3 KV cache benchmark ===")
    rows = [
        ("KV blocks", "num_kvcache_blocks", "{:d}"),
        ("KV cache (GiB)", "kv_cache_gib", "{:.2f}"),
        ("Max no-preempt concurrency", "max_no_preempt_concurrency", "{:d}"),
    ]
    if "throughput_tok_s" in base:
        rows += [
            ("Throughput (tok/s)", "throughput_tok_s", "{:.1f}"),
            ("Max running observed", "max_running_observed", "{:d}"),
            ("Preemptions", "preemptions", "{:d}"),
        ]
    print(f"{'metric':<30}{'FP16':>14}{'FP8':>14}{'ratio':>10}")
    for label, key, fmt in rows:
        a, b = base[key], fp8[key]
        ratio = (b / a) if a else float("nan")
        print(f"{label:<30}{fmt.format(a):>14}{fmt.format(b):>14}{ratio:>10.3f}")

    for key, label in [("num_kvcache_blocks", "KV capacity"), ("max_no_preempt_concurrency", "concurrency"), ("throughput_tok_s", "throughput")]:
        if key in base and base[key]:
            delta = (fp8[key] / base[key] - 1) * 100
            print(f"  {label}: {delta:+.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A/B benchmark: step-level exclusive prefill/decode vs. mixed chunked prefill.

Runs the same workload twice in separate processes:

  * off = legacy scheduler (`chunked_prefill=False`): a step is either a prefill
    step or a decode step, and prefill keeps priority, so a long prompt blocks
    the online decode stream;
  * on  = mixed scheduler (`chunked_prefill=True`): decode runs first through
    its CUDA graph and the leftover token budget is spent on an eager chunked
    prefill pass inside the same step.

Reports throughput, request latency, TTFT (time to first token) and TBT (time
between tokens) percentiles, peak memory and preemption counts. The run order
is alternated so clock/thermal drift does not favour one side.

Two arrival patterns:

  * `--arrival-interval-ms 0`  all requests submitted at once (offline batch);
  * `--arrival-interval-ms N`  one request every N ms (online serving). This is
    the pattern that exposes head-of-line blocking: long prompts keep arriving
    while other requests decode, and the exclusive scheduler serves the prefill
    first, stalling every in-flight decode.

`--budget` is the per-step token budget (`max_num_batched_tokens`); decode takes
one token per running sequence and prefill gets what is left. `--chunk-size`
additionally caps a single prefill pass (`chunked_prefill_size`, 0 = uncapped).

    python tools/bench_chunked_prefill.py --model ~/huggingface/Qwen3-0.6B \
        --num-seqs 256 --budget 1024 --repeat 3
    python tools/bench_chunked_prefill.py --model ... --arrival-interval-ms 5
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from time import perf_counter

# Allow the spawned worker (`python tools/bench_chunked_prefill.py --worker ...`)
# to import this checkout even when another nano-vllm copy is installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q / 100 * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def run_worker(args):
    import torch

    from nanovllm import LLM, SamplingParams

    random.seed(args.seed)
    if args.decode_wave:
        # Head-of-line-blocking probe: a first wave of tiny-prompt / long-output
        # requests fills the decode stream, then a second wave of long prompts
        # arrives while they are decoding.
        n_first = args.decode_wave
        input_lens = [4] * n_first + [args.max_input_len] * (args.num_seqs - n_first)
        output_lens = [args.max_output_len] * n_first + [8] * (args.num_seqs - n_first)
        offsets = [0.0] * n_first + [args.second_wave_delay_ms / 1000.0] * (args.num_seqs - n_first)
    elif args.fixed_lengths:
        input_lens = [args.max_input_len] * args.num_seqs
        output_lens = [args.max_output_len] * args.num_seqs
        offsets = [i * args.arrival_interval_ms / 1000.0 for i in range(args.num_seqs)]
    else:
        input_lens = [random.randint(args.min_input_len, args.max_input_len) for _ in range(args.num_seqs)]
        output_lens = [random.randint(args.min_output_len, args.max_output_len) for _ in range(args.num_seqs)]
        offsets = [i * args.arrival_interval_ms / 1000.0 for i in range(args.num_seqs)]

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.budget,
        chunked_prefill=bool(args.chunked),
        chunked_prefill_size=args.chunk_size or None,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.block_size,
    )

    # Warm up the allocator, the CUDA graph and any torch.compile work.
    warm_prompts = [[random.randint(0, 10000) for _ in range(8)] for _ in range(min(args.num_seqs, 64))]
    llm.generate(warm_prompts, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2), use_tqdm=False)
    torch.cuda.synchronize()
    del warm_prompts

    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in output_lens]
    pending = [([random.randint(0, 10000) for _ in range(n)], s) for n, s in zip(input_lens, sp)]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    base_reserved = torch.cuda.memory_reserved()

    step_times, decode_gaps = [], []
    arrival, finish_time, ttft = {}, {}, {}
    seen = set()
    total_tokens = 0
    last_decode_end = None
    t0 = perf_counter()
    next_arrival = 0
    while next_arrival < len(pending) or not llm.is_finished():
        now = perf_counter() - t0
        while next_arrival < len(pending) and offsets[next_arrival] <= now:
            prompt, sampling = pending[next_arrival]
            llm.add_request(prompt, sampling)
            arrival[llm.scheduler.waiting[-1].seq_id] = now
            next_arrival += 1
        if llm.is_finished():
            if next_arrival < len(pending):
                time.sleep(min(0.002, max(0.0, offsets[next_arrival] - (perf_counter() - t0))))
                continue
            break
        t = perf_counter()
        outputs, num_prefill_tokens, num_decode_tokens = llm.step()
        dt = perf_counter() - t
        now = perf_counter()
        step_times.append(dt)
        if num_decode_tokens > 0:
            # Wall time between this request's token and the previous one: a
            # prefill-only step lands entirely in this gap, which is exactly the
            # head-of-line blocking the mixed scheduler removes.
            decode_gaps.append(dt if last_decode_end is None else now - last_decode_end)
            last_decode_end = now
        for seq in llm.scheduler.running:
            if seq.seq_id not in seen:
                seen.add(seq.seq_id)
                ttft[seq.seq_id] = now - t0 - arrival.get(seq.seq_id, 0.0)
        for seq_id, token_ids in outputs:
            finish_time[seq_id] = now - t0
            total_tokens += len(token_ids)
    wall = perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    torch.cuda.synchronize()

    req = sorted(finish_time[sid] - arrival.get(sid, 0.0) for sid in finish_time)
    ttfts = sorted(ttft.values())
    steps = sorted(step_times)
    gaps = sorted(decode_gaps)
    result = {
        "chunked": bool(args.chunked),
        "num_seqs": args.num_seqs,
        "budget": args.budget,
        "chunk_size": args.chunk_size,
        "arrival_interval_ms": args.arrival_interval_ms,
        "decode_wave": args.decode_wave,
        "total_tokens": total_tokens,
        "wall_s": wall,
        "throughput": total_tokens / wall,
        "steps": len(step_times),
        "decode_steps": len(decode_gaps),
        "step_ms_p50": pct(steps, 50) * 1e3,
        "step_ms_p99": pct(steps, 99) * 1e3,
        "tbt_ms_p50": pct(gaps, 50) * 1e3,
        "tbt_ms_p99": pct(gaps, 99) * 1e3,
        "tbt_ms_p999": pct(gaps, 99.9) * 1e3,
        "ttft_ms_p50": pct(ttfts, 50) * 1e3,
        "ttft_ms_p99": pct(ttfts, 99) * 1e3,
        "ttft_ms_p999": pct(ttfts, 99.9) * 1e3,
        "request_ms_p50": pct(req, 50) * 1e3,
        "request_ms_p90": pct(req, 90) * 1e3,
        "request_ms_p99": pct(req, 99) * 1e3,
        "request_ms_p999": pct(req, 99.9) * 1e3,
        "peak_alloc_gb": peak / 2**30,
        "activation_peak_gb": (peak - base_alloc) / 2**30,
        "reserved_gb": reserved / 2**30,
        "activation_reserved_gb": (reserved - base_reserved) / 2**30,
        "preemptions": llm.scheduler.num_preemptions,
    }
    print("RESULT " + json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--budget", type=int, default=1024,
                        help="per-step token budget (max_num_batched_tokens)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="prefill pass cap per step (chunked_prefill_size); 0 = uncapped")
    parser.add_argument("--arrival-interval-ms", type=float, default=0.0,
                        help="online arrival: one request every N ms (0 = all at once)")
    parser.add_argument("--decode-wave", type=int, default=0,
                        help="first N requests are tiny-prompt/long-output; the rest arrive later "
                             "as long prompts (head-of-line-blocking probe)")
    parser.add_argument("--second-wave-delay-ms", type=float, default=300.0,
                        help="delay before the long-prompt wave arrives in --decode-wave mode")
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--fixed-lengths", action="store_true",
                        help="every sequence runs max_input+max_output tokens")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1, help="run the pair N times, report medians")
    parser.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs on both sides")
    # internal worker plumbing
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--chunked", type=int, default=0)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def spawn(chunked):
        cmd = [
            sys.executable, os.path.join(repo, "tools", "bench_chunked_prefill.py"),
            "--worker", "--model", args.model, "--chunked", str(chunked),
            "--num-seqs", str(args.num_seqs), "--chunk-size", str(args.chunk_size),
            "--budget", str(args.budget),
            "--arrival-interval-ms", str(args.arrival_interval_ms),
            "--decode-wave", str(args.decode_wave),
            "--second-wave-delay-ms", str(args.second_wave_delay_ms),
            "--min-input-len", str(args.min_input_len), "--max-input-len", str(args.max_input_len),
            "--min-output-len", str(args.min_output_len), "--max-output-len", str(args.max_output_len),
            "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--block-size", str(args.block_size), "--seed", str(args.seed),
        ]
        if args.fixed_lengths:
            cmd.append("--fixed-lengths")
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        if proc.returncode != 0:
            raise SystemExit("worker failed:\n" + proc.stdout[-4000:] + proc.stderr[-4000:])
        lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        assert lines, "worker produced no RESULT line:\n" + proc.stdout[-3000:]
        return json.loads(lines[-1][len("RESULT "):])

    off_runs, on_runs = [], []
    for i in range(args.repeat):
        # alternate the order so clock/thermal drift does not favour one side
        if i % 2 == 0:
            off_runs.append(spawn(0))
            on_runs.append(spawn(1))
        else:
            on_runs.append(spawn(1))
            off_runs.append(spawn(0))

    metrics = [
        ("throughput (tok/s)", "throughput", True),
        ("wall (s)", "wall_s", False),
        ("TTFT p50 (ms)", "ttft_ms_p50", False),
        ("TTFT p99 (ms)", "ttft_ms_p99", False),
        ("TTFT p999 (ms)", "ttft_ms_p999", False),
        ("TBT p50 (ms)", "tbt_ms_p50", False),
        ("TBT p99 (ms)", "tbt_ms_p99", False),
        ("TBT p999 (ms)", "tbt_ms_p999", False),
        ("request p50 (ms)", "request_ms_p50", False),
        ("request p90 (ms)", "request_ms_p90", False),
        ("request p99 (ms)", "request_ms_p99", False),
        ("request p999 (ms)", "request_ms_p999", False),
        ("step p50 (ms)", "step_ms_p50", False),
        ("step p99 (ms)", "step_ms_p99", False),
        ("activation peak (GiB)", "activation_peak_gb", False),
        ("total peak (GiB)", "peak_alloc_gb", False),
        ("total reserved (GiB)", "reserved_gb", False),
        ("activation reserved (GiB)", "activation_reserved_gb", False),
        ("preemptions", "preemptions", False),
    ]

    off0 = off_runs[0]
    print(f"workload  : {off0['num_seqs']} seqs, {off0['total_tokens']} tokens, "
          f"budget={off0['budget']}, chunk_size={off0['chunk_size']}, "
          f"arrival={off0['arrival_interval_ms']}ms, {off0['steps']} steps "
          f"({off0['decode_steps']} with decode), repeat={args.repeat}")
    print(f"{'metric':<24}{'exclusive':>14}{'chunked':>14}{'chunked/base':>14}{'delta':>10}"
          f"{'base min-max':>24}{'chunk min-max':>24}")

    def agg(runs, key):
        vals = [r[key] for r in runs]
        return statistics.median(vals), min(vals), max(vals)

    lines = {}
    for name, key, _ in metrics:
        o_med, o_lo, o_hi = agg(off_runs, key)
        f_med, f_lo, f_hi = agg(on_runs, key)
        r = f_med / o_med if o_med else float("nan")
        delta = (f_med - o_med) if key == "preemptions" else (r - 1) * 100
        lines[key] = {"exclusive": o_med, "chunked": f_med, "ratio": r, "delta_pct": delta}
        print(f"{name:<24}{o_med:>14.4f}{f_med:>14.4f}{r:>14.3f}{delta:>9.1f}%"
              f"{f'{o_lo:.3f}-{o_hi:.3f}':>24}{f'{f_lo:.3f}-{f_hi:.3f}':>24}")
    print("SUMMARY " + json.dumps({"by_metric": lines, "exclusive": off_runs, "chunked": on_runs}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end A/B of the fused Triton operators on a real model (eager).

Runs the same workload twice in separate processes -- fusion off (the
torch.compile chain) and fusion on (single-pass Triton kernels) -- and reports:

  * throughput (generated tokens / wall time),
  * inter-token latency percentiles over the decode steps,
  * per-request latency percentiles,
  * the *activation* peak (peak allocated during the run minus the memory held
    by the KV cache and weights, which is identical on both sides).

Eager mode is used on purpose: with CUDA graphs the launch overhead hides much
of the operator cost, while the task targets the eager path.

`--mode prefill` uses a few long prompts and a single output token, so the peak
allocation reflects the model's activation working set instead of the
logits/sampler buffers (which are identical on both sides and would otherwise
hide the difference).

    python tools/bench_fused_e2e.py --model ~/huggingface/Qwen3-0.6B --repeat 3
    python tools/bench_fused_e2e.py --model ... --mode prefill
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from time import perf_counter

# Allow the spawned worker (`python tools/bench_fused_e2e.py --worker ...`) to
# import this checkout even when another nano-vllm copy is installed.
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
    from nanovllm.layers import fused_ops

    ops = set() if args.ops == "none" else set(args.ops.split(","))
    # Patch the modules that were *not* selected back onto their torch.compile
    # chain, so a single run can fuse one operator at a time.
    if "silu" not in ops:
        from nanovllm.layers.activation import SiluAndMul
        SiluAndMul.forward = lambda self, x: self._compiled_forward(x)
    if "norm" not in ops:
        from nanovllm.layers.layernorm import RMSNorm
        RMSNorm.forward = lambda self, x, residual=None: (
            self.rms_forward(x) if residual is None else self.add_rms_forward(x, residual))
    if args.eager_baseline and not args.fusion:
        # Baseline = the original *unfused* torch chain (what the repo computes
        # before torch.compile), so the comparison isolates the fusion itself.
        from nanovllm.layers.activation import SiluAndMul
        from nanovllm.layers.layernorm import RMSNorm
        RMSNorm.forward = lambda self, x, residual=None: (
            fused_ops._torch_rms_norm(x, self.weight, self.eps) if residual is None
            else fused_ops._torch_add_rms_norm(x, residual, self.weight, self.eps))
        SiluAndMul.forward = lambda self, x: fused_ops._torch_silu_and_mul(x)

    random.seed(0)
    if args.mode == "prefill":
        input_lens = [args.max_input_len] * args.num_seqs
        output_lens = [1] * args.num_seqs
    elif args.fixed_lengths:
        input_lens = [args.max_input_len] * args.num_seqs
        output_lens = [args.max_output_len] * args.num_seqs
    else:
        input_lens = [random.randint(args.min_input_len, args.max_input_len) for _ in range(args.num_seqs)]
        output_lens = [random.randint(args.min_output_len, args.max_output_len) for _ in range(args.num_seqs)]

    llm = LLM(
        args.model,
        enforce_eager=not args.no_eager,
        triton_fusion=bool(ops),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.block_size,
    )
    assert fused_ops.fused_enabled() == bool(ops), "fusion flag did not take effect"

    # Warm up the allocator and the kernels with the *same batch size* the
    # benchmark uses, so any torch.compile work happens before the timer starts
    # on both sides.
    warm_prompts = [[random.randint(0, 10000) for _ in range(8)] for _ in range(args.num_seqs)]
    llm.generate(warm_prompts, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=2), use_tqdm=False)

    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=n) for n in output_lens]
    prompt_ids = [[random.randint(0, 10000) for _ in range(n)] for n in input_lens]
    for p, s in zip(prompt_ids, sp):
        llm.add_request(p, s)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()

    step_times, decode_times = [], []
    finish_time = {}
    total_tokens = 0
    t0 = perf_counter()
    while not llm.is_finished():
        t = perf_counter()
        outputs, num_prefill_tokens, num_decode_tokens = llm.step()
        dt = perf_counter() - t
        now = perf_counter()
        step_times.append(dt)
        if num_decode_tokens > 0:
            decode_times.append(dt)
        for seq_id, token_ids in outputs:
            finish_time[seq_id] = now - t0
            total_tokens += len(token_ids)
    wall = perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.synchronize()

    req = sorted(finish_time.values())
    steps = sorted(step_times)
    dec = sorted(decode_times)
    result = {
        "fusion": bool(args.fusion),
        "mode": args.mode,
        "num_seqs": args.num_seqs,
        "total_tokens": total_tokens,
        "wall_s": wall,
        "throughput": total_tokens / wall,
        "steps": len(step_times),
        "decode_steps": len(decode_times),
        "step_ms_p50": pct(steps, 50) * 1e3,
        "step_ms_p90": pct(steps, 90) * 1e3,
        "step_ms_p99": pct(steps, 99) * 1e3,
        "itl_ms_p50": pct(dec, 50) * 1e3,
        "itl_ms_p90": pct(dec, 90) * 1e3,
        "itl_ms_p99": pct(dec, 99) * 1e3,
        "request_ms_p50": pct(req, 50) * 1e3,
        "request_ms_p90": pct(req, 90) * 1e3,
        "request_ms_p99": pct(req, 99) * 1e3,
        "peak_alloc_gb": peak / 2**30,
        "activation_peak_gb": (peak - base_alloc) / 2**30,
        "preemptions": llm.scheduler.num_preemptions,
    }
    print("RESULT " + json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--fixed-lengths", action="store_true",
                        help="every sequence runs max_input+max_output tokens (fills the KV cache)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1, help="run the pair N times, report medians")
    parser.add_argument("--ops", default="norm,silu",
                        help="comma list of fused ops (norm, silu) or 'none'")
    parser.add_argument("--no-eager", action="store_true", help="use CUDA graphs")
    parser.add_argument("--eager-baseline", action="store_true",
                        help="baseline = unfused torch chain instead of torch.compile")
    # internal worker plumbing
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--fusion", type=int, default=0)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def spawn(fusion):
        cmd = [
            sys.executable, os.path.join(repo, "tools", "bench_fused_e2e.py"),
            "--worker", "--model", args.model, "--fusion", str(fusion),
            "--mode", args.mode, "--ops", args.ops,
            "--num-seqs", str(args.num_seqs),
            "--min-input-len", str(args.min_input_len), "--max-input-len", str(args.max_input_len),
            "--min-output-len", str(args.min_output_len), "--max-output-len", str(args.max_output_len),
            "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--block-size", str(args.block_size),
        ]
        if args.fixed_lengths:
            cmd.append("--fixed-lengths")
        if args.no_eager:
            cmd.append("--no-eager")
        if args.eager_baseline:
            cmd.append("--eager-baseline")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        if proc.returncode != 0:
            raise SystemExit("worker failed:\n" + proc.stdout[-3000:] + proc.stderr[-3000:])
        lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        assert lines, "worker produced no RESULT line:\n" + proc.stdout[-2000:]
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
        ("ITL p50 (ms)", "itl_ms_p50", False),
        ("ITL p99 (ms)", "itl_ms_p99", False),
        ("step p50 (ms)", "step_ms_p50", False),
        ("step p99 (ms)", "step_ms_p99", False),
        ("request p50 (ms)", "request_ms_p50", False),
        ("request p99 (ms)", "request_ms_p99", False),
        ("activation peak (GiB)", "activation_peak_gb", False),
        ("total peak (GiB)", "peak_alloc_gb", False),
        ("preemptions", "preemptions", False),
    ]

    def agg(runs, key):
        vals = [r[key] for r in runs]
        return statistics.median(vals), min(vals), max(vals)

    off0 = off_runs[0]
    print(f"mode       : {off0['mode']}, {off0['num_seqs']} seqs, {off0['total_tokens']} tokens, "
          f"{off0['decode_steps']} decode steps, repeat={args.repeat}, ops={args.ops}, "
          f"{'cuda-graph' if args.no_eager else 'eager'}, baseline={'eager-unfused' if args.eager_baseline else 'torch.compile'}")
    print(f"{'metric':<24}{'fusion off':>14}{'fusion on':>14}{'on/off':>10}{'delta':>10}"
          f"{'off min-max':>22}{'on min-max':>22}")
    summary = {"off": off_runs, "on": on_runs}
    lines = {}
    for name, key, _ in metrics:
        o_med, o_lo, o_hi = agg(off_runs, key)
        f_med, f_lo, f_hi = agg(on_runs, key)
        r = f_med / o_med if o_med else float("nan")
        delta = (f_med - o_med) if key == "preemptions" else (r - 1) * 100
        lines[key] = {"off": o_med, "on": f_med, "ratio": r, "delta_pct": delta}
        print(f"{name:<24}{o_med:>14.4f}{f_med:>14.4f}{r:>10.3f}{delta:>9.1f}%"
              f"{f'{o_lo:.3f}-{o_hi:.3f}':>22}{f'{f_lo:.3f}-{f_hi:.3f}':>22}")
    print("SUMMARY " + json.dumps({"by_metric": lines, "off": off_runs, "on": on_runs}))


if __name__ == "__main__":
    main()

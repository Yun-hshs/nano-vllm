#!/usr/bin/env python3
"""Microbenchmark: single-pass Triton operators vs. the torch.compile chain.

For each operator and shape it reports wall time, the *effective* bandwidth
(ideal bytes moved / time) and that bandwidth as a fraction of the device's
theoretical peak and of a measured device copy peak. The fused kernel is also
checked against the reference numerically so a fast-but-wrong kernel cannot win.

    python tools/bench_fused_ops.py
    python tools/bench_fused_ops.py --hidden 1024 --intermediate 3072 --rows 4096,16384
    python tools/bench_fused_ops.py --sweep
    python tools/bench_fused_ops.py --no-compile        # skip the baseline

Ideal bytes (single pass, weight counted once):
    add_rms_norm : 4 * M * N * itemsize + N * itemsize   (x, residual -> y, residual')
    rms_norm     : 2 * M * N * itemsize + N * itemsize   (x -> y)
    silu_and_mul : 3 * M * N * itemsize                  (gate, up -> y)
"""

import argparse
import itertools
import os
import sys

import torch

# Allow running this file directly (`python tools/bench_fused_ops.py`) even when
# another nano-vllm copy is installed in site-packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.layers import fused_ops  # noqa: E402
from nanovllm.layers.activation import SiluAndMul  # noqa: E402
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _time_raw(fn, iters, warmup, flush):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if flush is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    # Bracket every call with its own event pair so the L2 flush -- whose cost
    # is not constant once the cache state changes -- is excluded exactly
    # instead of being subtracted as an average.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def time_fn(fn, iters, warmup=5, flush=None):
    """Wall time per call, optionally evicting L2 before each iteration.
    Repeated iterations of a small working set otherwise measure L2 bandwidth,
    which dwarfs HBM and makes the ratios meaningless."""
    return _time_raw(fn, iters, warmup, flush)


def time_graph(fn, iters, flush=None):
    """Pure device time: capture `fn` in a CUDA graph and time replays, so the
    Python/Triton launch path is excluded."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    if flush is None:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def theoretical_peak_gbps():
    """clock(kHz) * bus(bytes) * 2 (DDR) -- the vendor number, when exposed."""
    props = torch.cuda.get_device_properties(0)
    clock = getattr(props, "memory_clock_rate", None)
    bus = getattr(props, "memory_bus_width", None)
    if not clock or not bus:
        return None
    return clock * 1e3 * (bus / 8) * 2 / 1e9


def measured_copy_gbps(iters=20, nbytes=512 << 20):
    """Practical read+write ceiling: a large contiguous copy."""
    n = nbytes // 4
    a = torch.empty(n, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    t = time_fn(lambda: b.copy_(a), iters)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n * 4 / (t / 1000) / 1e9


def make_case(op, m, hidden, inter, head_dim, num_heads, dtype, device, eps=1e-6):
    torch.manual_seed(0)
    if op == "add_rms_norm":
        x = torch.randn(m, hidden, dtype=dtype, device=device)
        residual = torch.randn(m, hidden, dtype=dtype, device=device)
        weight = torch.randn(hidden, dtype=dtype, device=device)
        return x, residual, weight, eps
    if op == "rms_norm":
        x = torch.randn(m, head_dim, dtype=dtype, device=device)
        weight = torch.randn(head_dim, dtype=dtype, device=device)
        return x, weight, eps
    if op == "silu_and_mul":
        x = torch.randn(m, 2 * inter, dtype=dtype, device=device)
        return x,
    raise ValueError(op)


def run_case(op, args, m, dtype, peak, copy_peak, flush):
    hidden, inter, head_dim = args.hidden, args.intermediate, args.head_dim
    device = "cuda"
    case = make_case(op, m, hidden, inter, head_dim, args.num_heads, dtype, device)

    if op == "add_rms_norm":
        norm = RMSNorm(hidden).to(device=device, dtype=dtype)
        n_cols = hidden
        call = lambda: norm(case[0], case[1])
        eager_call = lambda: fused_ops._torch_add_rms_norm(case[0], case[1], case[2], case[3])
        ideal_bytes = 4 * m * n_cols * dtype.itemsize + n_cols * dtype.itemsize
    elif op == "rms_norm":
        norm = RMSNorm(head_dim).to(device=device, dtype=dtype)
        n_cols = head_dim
        call = lambda: norm(case[0])
        eager_call = lambda: fused_ops._torch_rms_norm(case[0], case[1], case[2])
        ideal_bytes = 2 * m * n_cols * dtype.itemsize + n_cols * dtype.itemsize
    else:
        act = SiluAndMul().to(device=device, dtype=dtype)
        n_cols = inter
        call = lambda: act(case[0])
        eager_call = lambda: fused_ops._torch_silu_and_mul(case[0])
        ideal_bytes = 3 * m * n_cols * dtype.itemsize

    timer = time_graph if args.graph else time_fn
    # --- fused (Triton) ---------------------------------------------------
    fused_ops.set_fused_enabled(True)
    got = call()
    torch.cuda.synchronize()
    t_fused = timer(call, args.iters, flush=flush)

    # --- torch.compile reference -----------------------------------------
    t_base = float("nan")
    if not args.no_compile:
        fused_ops.set_fused_enabled(False)
        ref = call()
        torch.cuda.synchronize()
        max_err = max(float((a.float() - b.float()).abs().max()) for a, b in zip(
            got if isinstance(got, tuple) else (got,),
            ref if isinstance(ref, tuple) else (ref,),
        ))
        t_base = timer(call, args.iters, flush=flush)
        fused_ops.set_fused_enabled(True)
    else:
        max_err = float("nan")

    # --- plain unfused torch (the pre-torch.compile implementation) -------
    t_eager = float("nan")
    if args.eager_ref:
        eager_call()
        torch.cuda.synchronize()
        t_eager = timer(eager_call, args.iters, flush=flush)

    bw_fused = ideal_bytes / (t_fused / 1000) / 1e9
    bw_base = ideal_bytes / (t_base / 1000) / 1e9 if t_base == t_base else float("nan")
    bw_eager = ideal_bytes / (t_eager / 1000) / 1e9 if t_eager == t_eager else float("nan")
    pct_fused = 100 * bw_fused / peak if peak else float("nan")
    pct_base = 100 * bw_base / peak if (peak and bw_base == bw_base) else float("nan")
    pct_eager = 100 * bw_eager / peak if (peak and bw_eager == bw_eager) else float("nan")
    pct_copy = 100 * bw_fused / copy_peak if copy_peak else float("nan")

    label = f"{op:<13}{dtype_name(dtype):>5}{m:>7}"
    print(
        f"{label}{t_base:>11.4f}{t_eager:>10.4f}{t_fused:>10.4f}"
        f"{t_base / t_fused if t_base == t_base else float('nan'):>8.2f}x"
        f"{t_eager / t_fused if t_eager == t_eager else float('nan'):>8.2f}x"
        f"{bw_base:>10.1f}{bw_eager:>10.1f}{bw_fused:>10.1f}"
        f"{pct_base:>8.1f}{pct_eager:>8.1f}{pct_fused:>8.1f}"
        f"{max_err:>10.2e}"
    )
    return {"op": op, "m": m, "t_base": t_base, "t_eager": t_eager, "t_fused": t_fused,
            "bw_base": bw_base, "bw_eager": bw_eager, "bw_fused": bw_fused,
            "pct_peak_base": pct_base, "pct_peak_eager": pct_eager, "pct_peak_fused": pct_fused,
            "speedup": t_base / t_fused if t_base == t_base else float("nan"),
            "speedup_vs_eager": t_eager / t_fused if t_eager == t_eager else float("nan")}


def dtype_name(dtype):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(dtype, str(dtype))


def sweep(args, peak, copy_peak, flush):
    """Tune the fused kernels' launch geometry for the main Qwen3-0.6B shapes."""
    device = "cuda"
    dtype = DTYPES[args.dtype]
    print(f"\nsweep (device peak {peak:.1f} GB/s, copy peak {copy_peak:.1f} GB/s)" if peak
          else "\nsweep")
    print(f"{'op':<13}{'rows':>7}{'warps':>7}{'tile':>6}{'siluB':>7}{'ms':>10}{'GB/s':>9}{'%peak':>8}")

    def sweep_add_rms(m=16384, hidden=args.hidden):
        x = torch.randn(m, hidden, dtype=dtype, device=device)
        residual = torch.randn(m, hidden, dtype=dtype, device=device)
        weight = torch.randn(hidden, dtype=dtype, device=device)
        norm = RMSNorm(hidden).to(device=device, dtype=dtype)
        for tile, warps in itertools.product([512, 1024, 2048], [2, 4, 8]):
            fused_ops._TARGET_TILE = tile
            fused_ops._NUM_WARPS = warps
            fused_ops._norm_geometry.cache_clear()
            fused_ops.set_fused_enabled(True)
            t = time_fn(lambda: norm(x, residual), args.iters, flush=flush)
            ideal = 4 * m * hidden * dtype.itemsize + hidden * dtype.itemsize
            bw = ideal / (t / 1000) / 1e9
            _, rows = fused_ops._norm_geometry(hidden)
            print(f"{'add_rms_norm':<13}{rows:>7}{warps:>7}{tile:>6}{'-':>7}{t:>10.4f}{bw:>9.1f}{100 * bw / peak:>8.1f}")
        fused_ops._TARGET_TILE = 1024
        fused_ops._NUM_WARPS = 4
        fused_ops._norm_geometry.cache_clear()

    def sweep_silu(m=16384, inter=args.intermediate):
        x = torch.randn(m, 2 * inter, dtype=dtype, device=device)
        act = SiluAndMul().to(device=device, dtype=dtype)
        for block, warps in itertools.product([256, 512, 1024, 2048], [2, 4, 8]):
            fused_ops._SILU_BLOCK = block
            fused_ops._NUM_WARPS = warps
            fused_ops.set_fused_enabled(True)
            t = time_fn(lambda: act(x), args.iters, flush=flush)
            ideal = 3 * m * inter * dtype.itemsize
            bw = ideal / (t / 1000) / 1e9
            print(f"{'silu_and_mul':<13}{m // block:>7}{warps:>7}{'-':>6}{block:>7}{t:>10.4f}{bw:>9.1f}{100 * bw / peak:>8.1f}")
        fused_ops._SILU_BLOCK = 1024
        fused_ops._NUM_WARPS = 4

    sweep_add_rms()
    sweep_silu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024, help="Qwen3-0.6B hidden size")
    parser.add_argument("--intermediate", type=int, default=3072, help="Qwen3-0.6B intermediate size")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rows", default="1,64,512,2048,8192,16384",
                        help="token counts to sweep (comma separated)")
    parser.add_argument("--head-rows", default="16384,262144",
                        help="row counts for the head-dim RMSNorm (tokens*heads)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--no-compile", action="store_true", help="skip the torch.compile baseline")
    parser.add_argument("--no-eager-ref", dest="eager_ref", action="store_false",
                        help="skip the plain unfused torch baseline")
    parser.add_argument("--sweep", action="store_true", help="sweep launch geometry")
    parser.add_argument("--graph", action="store_true",
                        help="time CUDA-graph replays (pure device time, no launch overhead)")
    parser.add_argument("--no-flush", action="store_true",
                        help="do not evict L2 between iterations")
    parser.add_argument("--peak-gbps", type=float, default=None,
                        help="override the theoretical peak used for the %%pk columns")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    dtype = DTYPES[args.dtype]
    copy_peak = measured_copy_gbps()
    peak = args.peak_gbps or theoretical_peak_gbps() or copy_peak
    peak_label = "theoretical" if (args.peak_gbps or theoretical_peak_gbps()) else "copy (r+w)"
    # A 256 MiB fp32 buffer straddles the 4090's 72 MiB L2, so zeroing it evicts
    # the operator's working set and the measurement reflects HBM.
    flush = None
    if not args.no_flush:
        flush = torch.empty(64 << 20, dtype=torch.float32, device="cuda")

    print(f"device       : {torch.cuda.get_device_name(0)}")
    print(f"peak         : {peak:.1f} GB/s ({peak_label})")
    print(f"copy (r+w)   : {copy_peak:.1f} GB/s")
    print(f"dtype        : {args.dtype}, iters={args.iters}, "
          f"{'graph replay' if args.graph else 'eager launch'}, "
          f"{'L2 flushed' if flush is not None else 'L2 warm'}")
    print(f"{'op':<13}{'dtype':>5}{'rows':>7}{'cmp ms':>11}{'eag ms':>10}{'fus ms':>10}"
          f"{'vs cmp':>8}{'vs eag':>8}"
          f"{'cmp GB/s':>10}{'eag GB/s':>10}{'fus GB/s':>10}"
          f"{'cmp %pk':>9}{'eag %pk':>9}{'fus %pk':>9}{'maxerr':>10}")

    rows = [int(v) for v in args.rows.split(",")]
    head_rows = [int(v) for v in args.head_rows.split(",")]
    results = []
    for op in ("add_rms_norm", "rms_norm", "silu_and_mul"):
        sizes = head_rows if op == "rms_norm" else rows
        for m in sizes:
            results.append(run_case(op, args, m, dtype, peak, copy_peak, flush))

    if args.sweep:
        sweep(args, peak, copy_peak, flush)

    print("\nsummary (largest row per operator)")
    for op in ("add_rms_norm", "rms_norm", "silu_and_mul"):
        sel = [r for r in results if r["op"] == op]
        if not sel:
            continue
        big = max(sel, key=lambda r: r["m"])
        finite = [r for r in sel if r["speedup"] == r["speedup"]]
        best = max(finite, key=lambda r: r["speedup"]) if finite else big
        print(f"  {op:<14} M={big['m']:<7} "
              f"eager {big['pct_peak_eager']:5.1f}%  torch.compile {big['pct_peak_base']:5.1f}%  "
              f"fused {big['pct_peak_fused']:5.1f}% of {peak:.0f} GB/s  |  "
              f"speedup vs eager {big['speedup_vs_eager']:.2f}x, vs torch.compile {big['speedup']:.2f}x"
              + (f" (best {best['speedup']:.2f}x @ M={best['m']})" if best is not big else ""))


if __name__ == "__main__":
    main()

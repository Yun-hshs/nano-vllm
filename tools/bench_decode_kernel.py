#!/usr/bin/env python3
"""Microbenchmark: FP8 paged decode kernel vs flash_attn_with_kvcache.

Times the decode attention alone (no model), eagerly and under CUDA graph
replay, across batch sizes and context lengths, and sweeps the FP8 kernel's
BLOCK_N / num_warps so the shared-KV kernel can be tuned against the
flash-attention baseline.

    python tools/bench_decode_kernel.py
    python tools/bench_decode_kernel.py --sweep --batch 64 --ctx-len 4096
"""

import argparse
import itertools

import torch
import triton

from flash_attn import flash_attn_with_kvcache
from nanovllm.layers.attention import FP8_E4M3_DTYPE, fp8_decode_attention, store_kvcache_fp8
from nanovllm.utils.context import get_context, reset_context, set_context


def build_case(batch, ctx_len, kv_heads, group, head_dim, block_size, device="cuda"):
    num_heads = kv_heads * group
    max_blocks = triton.cdiv(ctx_len, block_size)
    num_blocks = max_blocks * batch
    slots = num_blocks * block_size

    torch.manual_seed(0)
    key = torch.randn(slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(slots, kv_heads, head_dim, device=device, dtype=torch.bfloat16)

    k_cache = torch.empty(num_blocks, block_size, kv_heads, head_dim, dtype=FP8_E4M3_DTYPE, device=device)
    v_cache = torch.empty_like(k_cache)
    v_scale = torch.empty(slots, kv_heads, dtype=torch.float16, device=device)
    k_scale = (key.float().abs().amax(dim=(0, 2)) / 448.0).clamp_min(1e-12)
    store_kvcache_fp8(key, value, k_cache, v_cache, k_scale, v_scale, torch.arange(slots, dtype=torch.int32, device=device))

    k_cache16 = key.view(num_blocks, block_size, kv_heads, head_dim).contiguous()
    v_cache16 = value.view(num_blocks, block_size, kv_heads, head_dim).contiguous()

    context_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=device)
    block_tables = torch.arange(max_blocks, dtype=torch.int32, device=device).repeat(batch, 1)
    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    return dict(q=q, k_cache=k_cache, v_cache=v_cache, k_scale=k_scale, v_scale=v_scale,
                k_cache16=k_cache16, v_cache16=v_cache16, context_lens=context_lens,
                block_tables=block_tables, scale=head_dim ** -0.5)


def time_fn(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def time_graph(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=0, help="0 = sweep several")
    parser.add_argument("--ctx-len", type=int, default=0, help="0 = sweep several")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--sweep", action="store_true", help="sweep BLOCK_N / num_warps / dot path")
    parser.add_argument("--use-dot", choices=["auto", "dot", "fma"], default="auto")
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()
    use_dot = None if args.use_dot == "auto" else (args.use_dot == "dot")

    batches = [args.batch] if args.batch else [1, 8, 32, 64, 128]
    ctx_lens = [args.ctx_len] if args.ctx_len else [512, 2048, 4096]

    print(f"{'batch':>6}{'ctx':>7}{'fp8 ms':>10}{'fp16 ms':>10}{'fp8 tok/s':>12}{'fp16 tok/s':>12}{'ratio':>8}")
    for ctx_len, batch in itertools.product(ctx_lens, batches):
        case = build_case(batch, ctx_len, args.kv_heads, args.group, args.head_dim, args.block_size)
        set_context(False, context_lens=case["context_lens"], block_tables=case["block_tables"])
        ctx = get_context()
        try:
            def fp8_call():
                return fp8_decode_attention(case["q"], case["k_cache"], case["v_cache"], case["k_scale"],
                                            case["v_scale"], ctx, case["scale"], use_dot=use_dot)

            def fp16_call():
                return flash_attn_with_kvcache(case["q"].unsqueeze(1), case["k_cache16"], case["v_cache16"],
                                               cache_seqlens=case["context_lens"], block_table=case["block_tables"],
                                               softmax_scale=case["scale"], causal=True)

            timer = time_fn if args.no_graph else time_graph
            t_fp8 = timer(fp8_call, args.iters)
            t_fp16 = timer(fp16_call, args.iters)
            tok8 = batch / (t_fp8 / 1000)
            tok16 = batch / (t_fp16 / 1000)
            print(f"{batch:>6}{ctx_len:>7}{t_fp8:>10.4f}{t_fp16:>10.4f}{tok8:>12.0f}{tok16:>12.0f}{t_fp8 / t_fp16:>8.3f}")
        finally:
            reset_context()

    if args.sweep:
        ctx_len = args.ctx_len or 4096
        batch = args.batch or 64
        case = build_case(batch, ctx_len, args.kv_heads, args.group, args.head_dim, args.block_size)
        set_context(False, context_lens=case["context_lens"], block_tables=case["block_tables"])
        ctx = get_context()
        print(f"\nsweep batch={batch} ctx={ctx_len}")
        print(f"{'path':>5}{'BLOCK_N':>8}{'warps':>7}{'ms':>10}{'tok/s':>12}")
        try:
            for dot, block_n, num_warps in itertools.product([False, True], [16, 32, 64, 128], [4, 8]):
                try:
                    fn = lambda: fp8_decode_attention(case["q"], case["k_cache"], case["v_cache"], case["k_scale"],
                                                      case["v_scale"], ctx, case["scale"],
                                                      block_n=block_n, num_warps=num_warps, use_dot=dot)
                    t = time_graph(fn, args.iters)
                    print(f"{'dot' if dot else 'fma':>5}{block_n:>8}{num_warps:>7}{t:>10.4f}{batch / (t / 1000):>12.0f}")
                except Exception as exc:  # pragma: no cover - tuning aid
                    print(f"{'dot' if dot else 'fma':>5}{block_n:>8}{num_warps:>7}  failed: {type(exc).__name__}: {exc}")
        finally:
            reset_context()


if __name__ == "__main__":
    main()

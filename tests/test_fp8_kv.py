"""FP8 E4M3 KV cache correctness tests.

Layout:
  * CPU tests  : host-side gather-slot mapping and capacity accounting math.
  * GPU tests  : Triton quantized-write / decode / gather parity against a
                 pure-torch reference that reads the *same* quantized cache, so
                 they isolate kernel bugs from quantization error.
  * GPU + model: end-to-end logits alignment between FP16 and FP8 KV cache.

Run everything:      pytest tests/test_fp8_kv.py -v
Run CPU only:        pytest tests/test_fp8_kv.py -v -m "not gpu"
Logits alignment:    NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B \
                     pytest tests/test_fp8_kv.py -v -m "logits"
"""

import os

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from nanovllm.layers.attention import (  # noqa: E402
    FP8_E4M3_DTYPE,
    FP8_E4M3_MAX,
    fp8_decode_attention,
    gather_dequant_kvcache,
    store_kvcache_fp8,
)
from nanovllm.utils.context import reset_context, set_context  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ===========================================================================
# CPU: host-side math
# ===========================================================================

class _BlockSizeStub:
    block_size = 4


def test_prefill_gather_slots_are_physical_positions():
    """The vectorised slot mapping must equal a naive per-token walk."""
    from nanovllm.engine.model_runner import ModelRunner

    cu_seqlens_k = torch.tensor([0, 5, 11], dtype=torch.int32)
    block_tables = torch.tensor([[10, 11, -1], [20, 21, 22]], dtype=torch.int32)
    slots = ModelRunner.prepare_prefill_gather_slots(_BlockSizeStub(), cu_seqlens_k, block_tables, 11)

    expected = []
    for seq, (start, end) in enumerate([(0, 5), (5, 11)]):
        for pos in range(end - start):
            blk = int(block_tables[seq, pos // 4])
            expected.append(blk * 4 + pos % 4)
    assert slots.tolist() == expected


def test_scale_accounting_gives_about_two_x_capacity():
    """FP8 + fp16 V-scale overhead must land at ~2x the FP16 block capacity."""
    layers, block_size, kv_heads, head_dim = 28, 256, 8, 128
    fp16_block_bytes = 2 * layers * block_size * kv_heads * head_dim * 2
    fp8_block_bytes = 2 * layers * block_size * kv_heads * head_dim * 1 + layers * block_size * kv_heads * 2
    assert 1.95 <= fp16_block_bytes / fp8_block_bytes <= 2.0


# ===========================================================================
# GPU: kernel parity
# ===========================================================================

def _torch_reference_decode(q, k_cache, v_cache, k_scale, v_scale_cache, block_tables, context_lens, block_size, scale):
    """Reference paged decode reading the same quantized cache/scales."""
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    group = num_heads // num_kv_heads
    flat_k = k_cache.reshape(-1, num_kv_heads, head_dim).float()
    flat_v = v_cache.reshape(-1, num_kv_heads, head_dim).float()
    out = torch.zeros(batch, num_heads, head_dim, dtype=torch.float32, device=q.device)
    for b in range(batch):
        length = int(context_lens[b])
        if length == 0:
            continue
        pos = torch.arange(length, device=q.device)
        blocks = block_tables[b, pos // block_size].long()
        slots = blocks * block_size + (pos % block_size)
        k = flat_k[slots] * k_scale[None, :, None]
        v = flat_v[slots] * v_scale_cache[slots].float()[:, :, None]
        for h in range(num_heads):
            kh = h // group
            scores = (q[b, h].float()[None, :] * k[:, kh, :]).sum(-1) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = (probs[:, None] * v[:, kh, :]).sum(0)
    return out


@requires_cuda
def test_quantized_write_roundtrip():
    torch.manual_seed(0)
    device = "cuda"
    num_blocks, block_size, kv_heads, head_dim = 4, 8, 4, 64
    k_cache = torch.empty(num_blocks, block_size, kv_heads, head_dim, dtype=FP8_E4M3_DTYPE, device=device)
    v_cache = torch.empty_like(k_cache)
    v_scale_cache = torch.empty(num_blocks * block_size, kv_heads, dtype=torch.float16, device=device)
    k_scale = torch.rand(kv_heads, device=device) * 0.05 + 0.01

    tokens = 12
    # Mirror the real model layout: K is rebuilt by RoPE so it is contiguous,
    # while V stays a strided view into the fused QKV output. The two row
    # strides differ -- a single-stride store silently reads the wrong memory
    # for every token after the first.
    q_size = 5 * head_dim
    qkv = torch.randn(tokens, q_size + 2 * kv_heads * head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = qkv[:, q_size + kv_heads * head_dim:].view(tokens, kv_heads, head_dim)
    value[:, 1] *= 1e-3  # exercise the per-token dynamic V scale
    assert key.stride(0) != value.stride(0), "test must exercise mismatched K/V strides"
    slots = torch.arange(tokens, dtype=torch.int32, device=device)

    store_kvcache_fp8(key, value, k_cache, v_cache, k_scale, v_scale_cache, slots)
    got_k, got_v = gather_dequant_kvcache(slots, k_cache, v_cache, k_scale, v_scale_cache, torch.float32)

    k_rel = (got_k - key.float()).norm() / key.float().norm()
    v_rel = (got_v - value.float()).norm() / value.float().norm()
    assert k_rel < 0.05, k_rel
    assert v_rel < 0.05, v_rel
    # the extreme-dynamic-range head must still round-trip within FP8 precision
    assert (got_v[:, 1] - value[:, 1].float()).norm() / value[:, 1].float().norm() < 0.05


@requires_cuda
def test_fp8_decode_matches_torch_reference():
    torch.manual_seed(0)
    device = "cuda"
    num_blocks, block_size, kv_heads, head_dim = 8, 8, 2, 64
    num_heads = kv_heads * 2
    scale = head_dim ** -0.5
    max_ctx = num_blocks * block_size

    k_cache = torch.empty(num_blocks, block_size, kv_heads, head_dim, dtype=FP8_E4M3_DTYPE, device=device)
    v_cache = torch.empty_like(k_cache)
    v_scale_cache = torch.empty(num_blocks * block_size, kv_heads, dtype=torch.float16, device=device)
    k_scale = torch.rand(kv_heads, device=device) * 0.05 + 0.01

    tokens = num_blocks * block_size
    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    slots = torch.arange(tokens, dtype=torch.int32, device=device)
    store_kvcache_fp8(key, value, k_cache, v_cache, k_scale, v_scale_cache, slots)

    batch = 3
    context_lens = torch.tensor([max_ctx, 5, 37], dtype=torch.int32, device=device)
    block_tables = torch.zeros(batch, num_blocks, dtype=torch.int32, device=device)
    for b in range(batch):
        for i in range(num_blocks):
            block_tables[b, i] = i

    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    set_context(False, context_lens=context_lens, block_tables=block_tables)
    try:
        got = fp8_decode_attention(q, k_cache, v_cache, k_scale, v_scale_cache, _ctx(), scale)
    finally:
        reset_context()
    ref = _torch_reference_decode(q, k_cache, v_cache, k_scale, v_scale_cache, block_tables, context_lens, block_size, scale)
    torch.testing.assert_close(got.float(), ref, atol=2e-3, rtol=2e-2)


def _ctx():
    from nanovllm.utils.context import get_context
    return get_context()


@requires_cuda
@pytest.mark.parametrize("use_dot", [False, True])
@pytest.mark.parametrize("group", [1, 2, 4, 7])
def test_decode_kernel_group_and_path_parity(group, use_dot):
    """Both the fp32-reduction and tensor-core decode paths must match the
    reference for padded and non-padded GQA groups."""
    torch.manual_seed(0)
    device = "cuda"
    kv_heads, head_dim, num_blocks, block_size = 2, 64, 8, 8
    num_heads = kv_heads * group
    scale = head_dim ** -0.5
    tokens = num_blocks * block_size

    k_cache = torch.empty(num_blocks, block_size, kv_heads, head_dim, dtype=FP8_E4M3_DTYPE, device=device)
    v_cache = torch.empty_like(k_cache)
    v_scale_cache = torch.empty(tokens, kv_heads, dtype=torch.float16, device=device)
    k_scale = torch.rand(kv_heads, device=device) * 0.05 + 0.01
    store_kvcache_fp8(
        torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16),
        torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16),
        k_cache, v_cache, k_scale, v_scale_cache, torch.arange(tokens, dtype=torch.int32, device=device),
    )

    batch = 3
    context_lens = torch.tensor([tokens, 5, 37], dtype=torch.int32, device=device)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).repeat(batch, 1)
    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    set_context(False, context_lens=context_lens, block_tables=block_tables)
    try:
        got = fp8_decode_attention(q, k_cache, v_cache, k_scale, v_scale_cache, _ctx(), scale, use_dot=use_dot)
    finally:
        reset_context()
    ref = _torch_reference_decode(q, k_cache, v_cache, k_scale, v_scale_cache, block_tables, context_lens, block_size, scale)
    torch.testing.assert_close(got.float(), ref, atol=2e-3, rtol=2e-2)


@requires_cuda
def test_fp8_decode_accuracy_vs_full_precision():
    """Quantization error (not kernel error) must stay small for logits parity."""
    torch.manual_seed(0)
    device = "cuda"
    num_blocks, block_size, kv_heads, head_dim = 16, 16, 2, 64
    num_heads = kv_heads * 2
    scale = head_dim ** -0.5
    tokens = num_blocks * block_size

    key = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.empty(num_blocks, block_size, kv_heads, head_dim, dtype=FP8_E4M3_DTYPE, device=device)
    v_cache = torch.empty_like(k_cache)
    v_scale_cache = torch.empty(tokens, kv_heads, dtype=torch.float16, device=device)
    # calibrate the static K scale exactly like ModelRunner does
    k_scale = (key.float().abs().amax(dim=(0, 2)) / FP8_E4M3_MAX).clamp_min(1e-12)
    slots = torch.arange(tokens, dtype=torch.int32, device=device)
    store_kvcache_fp8(key, value, k_cache, v_cache, k_scale, v_scale_cache, slots)

    context_lens = torch.tensor([tokens, 100, 7], dtype=torch.int32, device=device)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).repeat(3, 1)
    q = torch.randn(3, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    set_context(False, context_lens=context_lens, block_tables=block_tables)
    try:
        got = fp8_decode_attention(q, k_cache, v_cache, k_scale, v_scale_cache, _ctx(), scale)
    finally:
        reset_context()
    ref = _torch_reference_decode(q, k_cache, v_cache, k_scale, v_scale_cache, block_tables, context_lens, block_size, scale)
    # kernel output must match the quantized reference tightly ...
    torch.testing.assert_close(got.float(), ref, atol=2e-3, rtol=2e-2)
    # ... and the quantized result must stay close to a full-precision decode
    dense_k = key.float()
    dense_v = value.float()
    full = torch.zeros_like(ref)
    for b in range(3):
        length = int(context_lens[b])
        for h in range(num_heads):
            kh = h // 2
            scores = (q[b, h].float()[None, :] * dense_k[:length, kh, :]).sum(-1) * scale
            probs = torch.softmax(scores, dim=-1)
            full[b, h] = (probs[:, None] * dense_v[:length, kh, :]).sum(0)
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), full.flatten(), dim=0)
    assert cos > 0.999, cos


@requires_cuda
def test_decode_kernel_under_cuda_graph_dynamic_context():
    """The decode kernel must re-read context_lens/block_tables on every graph
    replay so the same captured graph serves different context lengths."""
    torch.manual_seed(0)
    device = "cuda"
    num_blocks, block_size, kv_heads, head_dim = 16, 16, 2, 64
    num_heads = kv_heads * 2
    scale = head_dim ** -0.5
    tokens = num_blocks * block_size

    k_cache = torch.empty(num_blocks, block_size, kv_heads, head_dim, dtype=FP8_E4M3_DTYPE, device=device)
    v_cache = torch.empty_like(k_cache)
    v_scale_cache = torch.empty(tokens, kv_heads, dtype=torch.float16, device=device)
    k_scale = torch.rand(kv_heads, device=device) * 0.05 + 0.01
    slots = torch.arange(tokens, dtype=torch.int32, device=device)
    store_kvcache_fp8(
        torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16),
        torch.randn(tokens, kv_heads, head_dim, device=device, dtype=torch.bfloat16),
        k_cache, v_cache, k_scale, v_scale_cache, slots,
    )

    batch = 2
    context_lens = torch.full((batch,), 32, dtype=torch.int32, device=device)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).repeat(batch, 1)
    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    set_context(False, context_lens=context_lens, block_tables=block_tables)
    try:
        # eager warmup compiles the Triton kernel and materialises the output
        # allocation in the graph pool before capture starts
        fp8_decode_attention(q, k_cache, v_cache, k_scale, v_scale_cache, _ctx(), scale)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = fp8_decode_attention(q, k_cache, v_cache, k_scale, v_scale_cache, _ctx(), scale)

        for lengths in ([32, 64], [256, 17], [1, 250]):
            context_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
            graph.replay()
            torch.cuda.synchronize()
            ref = _torch_reference_decode(
                q, k_cache, v_cache, k_scale, v_scale_cache, block_tables, context_lens, block_size, scale
            )
            torch.testing.assert_close(graph_out.float(), ref, atol=2e-3, rtol=2e-2)
    finally:
        reset_context()


# ===========================================================================
# GPU + real model: logits-level alignment
# ===========================================================================

@pytest.mark.gpu
@pytest.mark.logits
@requires_cuda
def test_logits_alignment_fp16_vs_fp8():
    """End-to-end logits alignment, including the prefix-cache gather path.

    Delegates to tools/logits_align.py, which runs each precision in its own
    process and forces the FP8 run to consume the FP16 run's greedy token
    stream, so every step compares identical inputs.
    """
    model_path = os.environ.get("NANOVLLM_TEST_MODEL")
    if not model_path:
        pytest.skip("set NANOVLLM_TEST_MODEL to run logits alignment")
    import json
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tool = os.path.join(repo, "tools", "logits_align.py")
    proc = subprocess.run(
        [sys.executable, tool, "--mode", "all", "--model", model_path],
        capture_output=True, text=True, cwd=repo,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    summary = json.loads([l for l in proc.stdout.splitlines() if l.startswith("SUMMARY ")][-1][len("SUMMARY "):])

    # phase 2 must actually have taken the gather + dequant prefill path
    assert summary["gather_calls"] > 0, "prefix-cache gather + dequant prefill path was not exercised"
    for phase in ("phase1", "phase2"):
        s = summary[phase]
        assert s["prefill_cosine"] > 0.99, (phase, s)
        assert s["worst_decode_cosine"] > 0.95, (phase, s)
        assert s["decode_top1_agreement"] >= 0.90, (phase, s)

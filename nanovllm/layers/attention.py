import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# FP8 E4M3 (torch.float8_e4m3fn) represents finite values in [-448, 448].
FP8_E4M3_MAX = 448.0
FP8_E4M3_DTYPE = torch.float8_e4m3fn


# ===========================================================================
# FP16/BF16 KV cache write (unchanged baseline path)
# ===========================================================================

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ===========================================================================
# FP8 E4M3 quantized KV cache write
#
#   K: static per-(layer, kv_head) scale, known before the token arrives, so the
#      write is a pure scale-and-saturate with no reduction.
#   V: dynamic per-(token, kv_head) scale, reduced from the token itself and
#      persisted next to the cache so decode/prefill can dequantize it later.
# ===========================================================================

@triton.jit
def store_kvcache_fp8_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    KV_HEADS_P2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return

    h = tl.arange(0, KV_HEADS_P2)
    d = tl.arange(0, BLOCK_D)
    mask = (h[:, None] < num_kv_heads) & (d[None, :] < head_dim)

    # K and V must use their own row strides: in a real model K is rebuilt by
    # RoPE (contiguous) while V stays a strided view into the fused QKV output,
    # so key_stride != value_stride.
    key_src = idx * key_stride + h[:, None] * head_dim + d[None, :]
    val_src = idx * value_stride + h[:, None] * head_dim + d[None, :]
    key = tl.load(key_ptr + key_src, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + val_src, mask=mask, other=0.0).to(tl.float32)

    # --- K: static per head scale -----------------------------------------
    k_scale = tl.load(k_scale_ptr + h, mask=h < num_kv_heads, other=1.0).to(tl.float32)
    k_q = key / k_scale[:, None]
    k_q = tl.minimum(tl.maximum(k_q, -FP8_MAX), FP8_MAX)

    # --- V: dynamic per token / kv head scale -----------------------------
    v_amax = tl.max(tl.abs(value), axis=1)
    v_scale = tl.maximum(v_amax, MIN_SCALE) / FP8_MAX
    v_q = value / v_scale[:, None]
    v_q = tl.minimum(tl.maximum(v_q, -FP8_MAX), FP8_MAX)

    D = num_kv_heads * head_dim
    dst = slot.to(tl.int64) * D + h[:, None] * head_dim + d[None, :]
    tl.store(k_cache_ptr + dst, k_q.to(k_cache_ptr.dtype.element_ty), mask=mask)
    tl.store(v_cache_ptr + dst, v_q.to(v_cache_ptr.dtype.element_ty), mask=mask)
    tl.store(
        v_scale_ptr + slot * num_kv_heads + h,
        v_scale.to(v_scale_ptr.dtype.element_ty),
        mask=h < num_kv_heads,
    )


def store_kvcache_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Quantize and write one token each to the paged FP8 cache.

    key/value:      [N, num_kv_heads, head_dim]
    k_cache/v_cache:[num_blocks, block_size, num_kv_heads, head_dim] (fp8)
    k_scale:        [num_kv_heads] (static)
    v_scale_cache:  [num_slots, num_kv_heads] (per token dynamic)
    """
    N, num_kv_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert slot_mapping.numel() == N
    store_kvcache_fp8_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, k_scale, v_scale_cache, slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        KV_HEADS_P2=triton.next_power_of_2(num_kv_heads),
        BLOCK_D=triton.next_power_of_2(head_dim),
        FP8_MAX=FP8_E4M3_MAX,
        MIN_SCALE=1e-6,
        num_warps=4,
    )


# ===========================================================================
# FP8 Paged Attention decode
#
# One program per (sequence, KV head). All query heads of the GQA group share
# a single K tile, V tile and V-scale load per iteration; the tiles are
# dequantized once and reused by every query head in the group.
# ===========================================================================

@triton.jit
def fp8_paged_attention_decode_kernel(
    out_ptr, q_ptr,
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    block_tables_ptr, context_lens_ptr,
    q_stride_m, q_stride_h,
    out_stride_m, out_stride_h,
    bt_stride_m,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    group_p: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    seq_id = tl.program_id(0)
    kv_head = tl.program_id(1)

    ctx_len = tl.load(context_lens_ptr + seq_id)
    if ctx_len <= 0:
        return

    D = num_kv_heads * head_dim
    g = tl.arange(0, group_p)
    d = tl.arange(0, BLOCK_D)
    g_mask = g < group_size
    d_mask = d < head_dim
    gd_mask = g_mask[:, None] & d_mask[None, :]

    q_off = seq_id * q_stride_m + (kv_head * group_size + g)[:, None] * q_stride_h + d[None, :]
    q_native = tl.load(q_ptr + q_off, mask=gd_mask, other=0.0)
    k_scale = tl.load(k_scale_ptr + kv_head)

    m_i = tl.full([group_p], float("-inf"), tl.float32)
    l_i = tl.zeros([group_p], tl.float32)
    acc = tl.zeros([group_p, BLOCK_D], tl.float32)

    # Iterate physical blocks, then contiguous sub-tiles inside each block. The
    # token index is a plain arange, so K/V rows are a regular stride apart and
    # the loads vectorise -- unlike a flat per-tile block-table gather. The inner
    # loop is static (block_size is constexpr), which keeps the outer dynamic
    # loop software-pipelineable.
    num_blocks = tl.cdiv(ctx_len, block_size)
    for blk_idx in range(0, num_blocks):
        physical = tl.load(block_tables_ptr + seq_id * bt_stride_m + blk_idx)
        base = physical * block_size
        block_start = blk_idx * block_size
        for sub in range(0, block_size, BLOCK_N):
            tok = sub + tl.arange(0, BLOCK_N)
            pos = block_start + tok
            valid = (tok < block_size) & (pos < ctx_len)
            kv_off = (base + tok).to(tl.int64)[:, None] * D + kv_head * head_dim + d[None, :]
            kv_mask = valid[:, None] & d_mask[None, :]

            # --- shared K tile for every query head of the GQA group ---------
            k_raw = tl.load(k_cache_ptr + kv_off, mask=kv_mask, other=0.0)
            if USE_DOT:
                # fp16 is the tensor-core input type here: e4m3 converts to fp16
                # natively on sm_89+, and every e4m3 value is exact in fp16. The
                # static K scale is applied to the scores in fp32.
                s = tl.dot(q_native.to(tl.float16), tl.trans(k_raw.to(tl.float16)), out_dtype=tl.float32)
                s = s * (k_scale * softmax_scale)
            else:
                q_eff = q_native.to(tl.float32) * (k_scale * softmax_scale)
                s = tl.sum(q_eff[:, None, :] * k_raw.to(tl.float32)[None, :, :], axis=2)
            s = tl.where(valid[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)

            # --- shared V tile, dequantized with the per-slot dynamic scale --
            v_scale = tl.load(v_scale_ptr + (base + tok) * num_kv_heads + kv_head, mask=valid, other=1.0).to(tl.float32)
            v_raw = tl.load(v_cache_ptr + kv_off, mask=kv_mask, other=0.0)
            v_deq = v_raw.to(tl.float32) * v_scale[:, None]
            if USE_DOT:
                acc = tl.dot(p.to(tl.float16), v_deq.to(tl.float16),
                             acc * alpha[:, None], out_dtype=tl.float32)
            else:
                acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v_deq[None, :, :], axis=1)
            m_i = m_new

    out = acc / tl.maximum(l_i, 1e-10)[:, None]
    out_off = seq_id * out_stride_m + (kv_head * group_size + g)[:, None] * out_stride_h + d[None, :]
    tl.store(out_ptr + out_off, out.to(out_ptr.dtype.element_ty), mask=gd_mask)


def fp8_decode_attention(q, k_cache, v_cache, k_scale, v_scale_cache, context, scale,
                         block_n=None, num_warps=None, use_dot=None, num_stages=None):
    """FP8 paged decode. q: [batch, num_heads, head_dim] -> same shape.

    `use_dot=True` gives the query-head tile a tensor-core matmul (the GQA group
    is padded to 16 rows); `use_dot=False` uses an fp32 broadcast reduction.
    The default picks the dot path only when the padding waste is small.
    """
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    group_size = num_heads // num_kv_heads
    BLOCK_D = triton.next_power_of_2(head_dim)
    if use_dot is None:
        # Tensor cores win even when the GQA group is padded up to 16 rows
        # (measured 2.3-8x faster than the fp32 reduction on sm_89).
        use_dot = BLOCK_D >= 16
    group_p = max(16, triton.next_power_of_2(group_size)) if use_dot else triton.next_power_of_2(group_size)
    if block_n is None:
        if use_dot:
            # keep the K/V tiles inside the shared-memory budget
            BLOCK_N = 64 if BLOCK_D <= 128 else 32
        else:
            BLOCK_N = max(16, min(64, 8192 // (group_p * BLOCK_D)))
    else:
        BLOCK_N = block_n
    BLOCK_N = max(16, triton.next_power_of_2(BLOCK_N))
    if num_warps is None:
        num_warps = 4 if (use_dot or group_p * BLOCK_N * BLOCK_D <= 4096) else 8
    extra = {} if num_stages is None else {"num_stages": num_stages}
    out = torch.empty_like(q)
    block_tables = context.block_tables
    grid = (batch, num_kv_heads)
    fp8_paged_attention_decode_kernel[grid](
        out, q,
        k_cache, v_cache,
        k_scale, v_scale_cache,
        block_tables, context.context_lens,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        group_p=group_p,
        head_dim=head_dim,
        block_size=block_size,
        softmax_scale=scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        USE_DOT=use_dot,
        num_warps=num_warps,
        **extra,
    )
    return out


# ===========================================================================
# FP8 prefill: gather requested slots + dequantize + dense FlashAttention
# ===========================================================================

@triton.jit
def gather_dequant_kvcache_kernel(
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    k_out_ptr, v_out_ptr,
    gather_slots_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    KV_HEADS_P2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(gather_slots_ptr + idx)

    h = tl.arange(0, KV_HEADS_P2)
    d = tl.arange(0, BLOCK_D)
    mask = (h[:, None] < num_kv_heads) & (d[None, :] < head_dim)

    D = num_kv_heads * head_dim
    src = slot.to(tl.int64) * D + h[:, None] * head_dim + d[None, :]
    k_scale = tl.load(k_scale_ptr + h, mask=h < num_kv_heads, other=1.0).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + slot * num_kv_heads + h, mask=h < num_kv_heads, other=1.0).to(tl.float32)
    k = tl.load(k_cache_ptr + src, mask=mask, other=0.0).to(tl.float32) * k_scale[:, None]
    v = tl.load(v_cache_ptr + src, mask=mask, other=0.0).to(tl.float32) * v_scale[:, None]

    dst = idx * D + h[:, None] * head_dim + d[None, :]
    tl.store(k_out_ptr + dst, k.to(k_out_ptr.dtype.element_ty), mask=mask)
    tl.store(v_out_ptr + dst, v.to(v_out_ptr.dtype.element_ty), mask=mask)


def gather_dequant_kvcache(gather_slots, k_cache, v_cache, k_scale, v_scale_cache, out_dtype):
    """Build dense [total_kv, num_kv_heads, head_dim] K/V for FlashAttention.

    Only the slots actually referenced by the scheduled sequences are gathered,
    so memory is O(sum(context_len)) instead of O(cache size).
    """
    total = gather_slots.numel()
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    device = k_cache.device
    k_out = torch.empty(total, num_kv_heads, head_dim, dtype=out_dtype, device=device)
    v_out = torch.empty(total, num_kv_heads, head_dim, dtype=out_dtype, device=device)
    if total == 0:
        return k_out, v_out
    gather_dequant_kvcache_kernel[(total,)](
        k_cache, v_cache, k_scale, v_scale_cache, k_out, v_out, gather_slots,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        KV_HEADS_P2=triton.next_power_of_2(num_kv_heads),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return k_out, v_out


# ===========================================================================
# Attention module
# ===========================================================================

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.fp8_kvcache = False
        self.k_scale = torch.tensor([])
        self.v_scale_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.fp8_kvcache:
                store_kvcache_fp8(k, v, k_cache, v_cache, self.k_scale, self.v_scale_cache, context.slot_mapping)
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                if self.fp8_kvcache:
                    return self._prefill_attention_fp8(q, context)
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if self.fp8_kvcache:
                o = fp8_decode_attention(q, k_cache, v_cache, self.k_scale, self.v_scale_cache, context, self.scale)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables,
                                            softmax_scale=self.scale, causal=True)
        return o

    def _prefill_attention_fp8(self, q, context):
        """Prefix-cache prefill: gather + dequantize only the referenced slots,
        then run dense FlashAttention.

        `cu_seqlens_k` counts full contexts, so with prefix caching a batch of
        many long-prefix-hit sequences can reference far more KV than the
        prefill token budget. `prefill_kv_groups` splits the batch so each
        dequantized buffer stays inside the gather budget."""
        groups = context.prefill_kv_groups
        if groups is None or len(groups) == 1:
            if groups is None:
                cu_q, cu_k = context.cu_seqlens_q, context.cu_seqlens_k
                slots = context.prefill_gather_slots
            else:
                _, _, k0, k1, cu_q, cu_k = groups[0]
                slots = context.prefill_gather_slots[k0:k1]
            k, v = gather_dequant_kvcache(slots, self.k_cache, self.v_cache, self.k_scale, self.v_scale_cache, q.dtype)
            return flash_attn_varlen_func(q, k, v,
                                          max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=cu_q,
                                          max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=cu_k,
                                          softmax_scale=self.scale, causal=True)
        o = torch.empty_like(q)
        for q0, q1, k0, k1, cu_q, cu_k in groups:
            k, v = gather_dequant_kvcache(
                context.prefill_gather_slots[k0:k1], self.k_cache, self.v_cache,
                self.k_scale, self.v_scale_cache, q.dtype,
            )
            o[q0:q1] = flash_attn_varlen_func(q[q0:q1], k, v,
                                              max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=cu_q,
                                              max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=cu_k,
                                              softmax_scale=self.scale, causal=True)
        return o

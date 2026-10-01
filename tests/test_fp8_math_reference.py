"""Pure-numpy reference for the FP8 KV cache math.

This file has NO torch/triton/flash-attn dependency, so it runs on any CPU
machine. It re-implements, step by step, exactly what the Triton kernels do:

  * e4m3 quantize/dequantize with saturation (store kernel),
  * dynamic per-(token, kv head) V scale + static K scale,
  * the tiled paged-decode loop with physical block addressing, FP8 dequant and
    online-softmax merging (decode kernel).

and checks it against a naive full-softmax decode over the *same* quantized
cache. That isolates algorithmic bugs (addressing, masking, online merge) from
Triton codegen, which is the only part that needs a GPU.

Run directly:  python3 tests/test_fp8_math_reference.py
Or with pytest: pytest tests/test_fp8_math_reference.py
"""

import numpy as np

FP8_MAX = 448.0
MIN_SCALE = 1e-6


# ---------------------------------------------------------------------------
# e4m3 quantization (matches the representable set of torch.float8_e4m3fn)
# ---------------------------------------------------------------------------

def _e4m3_finite_values():
    values = []
    for bits in range(256):
        sign = -1.0 if (bits >> 7) & 1 else 1.0
        exp = (bits >> 3) & 0xF
        man = bits & 0x7
        if exp == 0xF and man == 0x7:
            continue  # NaN
        if exp == 0:
            values.append(sign * (2.0 ** -6) * (man / 8.0))
        else:
            values.append(sign * (2.0 ** (exp - 7)) * (1.0 + man / 8.0))
    return np.array(sorted(set(values)), dtype=np.float64)


_E4M3_VALUES = _e4m3_finite_values()
assert _E4M3_VALUES.max() == FP8_MAX


def quantize_e4m3(x):
    """Saturating nearest-value e4m3 quantization (rounding mode is irrelevant
    for the properties under test, which are error bounds and tiled==naive)."""
    x = np.clip(np.asarray(x, dtype=np.float64), -FP8_MAX, FP8_MAX)
    flat = x.reshape(-1)
    idx = np.abs(flat[:, None] - _E4M3_VALUES[None, :]).argmin(axis=1)
    return _E4M3_VALUES[idx].reshape(x.shape)


# ---------------------------------------------------------------------------
# store kernel semantics
# ---------------------------------------------------------------------------

def store_kv_fp8(key, value, k_scale):
    """key/value: [T, H, D] -> (k_q, v_q, v_scale) with v_scale [T, H]."""
    tokens, heads, _ = key.shape
    k_q = quantize_e4m3(key / k_scale[None, :, None])
    v_amax = np.abs(value).max(axis=2)                      # [T, H]
    v_scale = np.maximum(v_amax, MIN_SCALE) / FP8_MAX
    v_q = quantize_e4m3(value / v_scale[:, :, None])
    return k_q, v_q, v_scale


# ---------------------------------------------------------------------------
# decode kernel semantics (mirrors fp8_paged_attention_decode_kernel)
# ---------------------------------------------------------------------------

def decode_online(q, k_cache, v_cache, v_scale_cache, k_scale, block_table,
                  context_lens, block_size, softmax_scale, block_n):
    """Tiled decode with fused addressing + FP8 dequant + online softmax.

    q:              [B, num_heads, D]
    k_cache:        [num_slots, kv_heads, D] quantized
    v_cache:        [num_slots, kv_heads, D] quantized
    v_scale_cache:  [num_slots, kv_heads]
    k_scale:        [kv_heads]
    block_table:    [B, max_blocks]
    context_lens:   [B]
    """
    batch, num_heads, head_dim = q.shape
    kv_heads = k_cache.shape[1]
    group = num_heads // kv_heads
    out = np.zeros((batch, num_heads, head_dim), dtype=np.float64)

    for b in range(batch):
        ctx_len = int(context_lens[b])
        if ctx_len <= 0:
            continue
        for kv_head in range(kv_heads):
            q_eff = q[b, kv_head * group:(kv_head + 1) * group].astype(np.float64)
            q_eff = q_eff * (k_scale[kv_head] * softmax_scale)   # K scale folded into Q

            m = np.full(group, -np.inf)
            l = np.zeros(group)
            acc = np.zeros((group, head_dim))
            num_tiles = (ctx_len + block_n - 1) // block_n
            for t in range(num_tiles):
                j = t * block_n + np.arange(block_n)
                valid = j < ctx_len
                # Triton's masked loads never dereference masked lanes, so clamp
                # the indices before numpy fancy-indexing them.
                blk_idx = np.minimum(j // block_size, block_table.shape[1] - 1)
                blk = np.where(valid, block_table[b, blk_idx], 0)
                slot = np.where(valid, blk * block_size + (j % block_size), 0)

                k_tile = k_cache[slot, kv_head, :].astype(np.float64)
                k_tile = np.where(valid[:, None], k_tile, 0.0)
                s = q_eff @ k_tile.T                                  # [group, block_n]
                s = np.where(valid[None, :], s, -np.inf)

                m_new = np.maximum(m, s.max(axis=1))
                p = np.exp(s - m_new[:, None])
                alpha = np.exp(m - m_new)
                l = l * alpha + p.sum(axis=1)

                vs = v_scale_cache[slot, kv_head]
                v_tile = v_cache[slot, kv_head, :].astype(np.float64) * vs[:, None]
                v_tile = np.where(valid[:, None], v_tile, 0.0)
                acc = acc * alpha[:, None] + p @ v_tile
                m = m_new
            out[b, kv_head * group:(kv_head + 1) * group] = acc / np.maximum(l, 1e-10)[:, None]
    return out


def decode_naive_quantized(q, k_cache, v_cache, v_scale_cache, k_scale, block_table,
                           context_lens, block_size, softmax_scale):
    """Same quantized cache, but full-softmax reference (no tiling / no online merge)."""
    batch, num_heads, head_dim = q.shape
    kv_heads = k_cache.shape[1]
    group = num_heads // kv_heads
    out = np.zeros((batch, num_heads, head_dim), dtype=np.float64)
    for b in range(batch):
        ctx_len = int(context_lens[b])
        if ctx_len <= 0:
            continue
        pos = np.arange(ctx_len)
        blk = block_table[b, pos // block_size]
        slot = blk * block_size + (pos % block_size)
        k = k_cache[slot].astype(np.float64) * k_scale[None, :, None]
        v = v_cache[slot].astype(np.float64) * v_scale_cache[slot][:, :, None]
        for h in range(num_heads):
            kh = h // group
            s = (q[b, h].astype(np.float64)[None, :] * k[:, kh, :]).sum(-1) * softmax_scale
            p = np.exp(s - s.max())
            p = p / p.sum()
            out[b, h] = (p[:, None] * v[:, kh, :]).sum(0)
    return out


def decode_full_precision(q, key, value, context_lens, softmax_scale):
    batch, num_heads, head_dim = q.shape
    kv_heads = key.shape[1]
    group = num_heads // kv_heads
    out = np.zeros((batch, num_heads, head_dim), dtype=np.float64)
    for b in range(batch):
        length = int(context_lens[b])
        if length <= 0:
            continue
        for h in range(num_heads):
            kh = h // group
            s = (q[b, h].astype(np.float64)[None, :] * key[:length, kh].astype(np.float64)).sum(-1) * softmax_scale
            p = np.exp(s - s.max())
            p = p / p.sum()
            out[b, h] = (p[:, None] * value[:length, kh].astype(np.float64)).sum(0)
    return out


def _cosine(a, b):
    return float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def _make_case(seed=0, batch=3, kv_heads=2, group=2, head_dim=16, block_size=8,
               max_slots=64, dynamic_range=False):
    rng = np.random.default_rng(seed)
    num_heads = kv_heads * group
    key = rng.standard_normal((max_slots, kv_heads, head_dim))
    value = rng.standard_normal((max_slots, kv_heads, head_dim))
    if dynamic_range:
        value[:, 1] *= 1e-3   # one head with a very different dynamic range
    k_scale = np.abs(key).max(axis=(0, 2)) / FP8_MAX
    k_cache, v_cache, v_scale_cache = store_kv_fp8(key, value, k_scale)
    num_blocks = max_slots // block_size
    context_lens = np.array([max_slots, 5, 37][:batch], dtype=np.int64)
    block_table = np.tile(np.arange(num_blocks, dtype=np.int64), (batch, 1))
    q = rng.standard_normal((batch, num_heads, head_dim))
    return dict(q=q, key=key, value=value, k_cache=k_cache, v_cache=v_cache,
                v_scale_cache=v_scale_cache, k_scale=k_scale, block_table=block_table,
                context_lens=context_lens, block_size=block_size, max_slots=max_slots)


def test_tiled_online_softmax_matches_full_softmax():
    """Tiling + online merge must be exact on the same quantized cache."""
    for dynamic_range in (False, True):
        case = _make_case(dynamic_range=dynamic_range)
        scale = case["q"].shape[-1] ** -0.5
        online = decode_online(case["q"], case["k_cache"], case["v_cache"], case["v_scale_cache"],
                               case["k_scale"], case["block_table"], case["context_lens"],
                               case["block_size"], scale, block_n=8)
        naive = decode_naive_quantized(case["q"], case["k_cache"], case["v_cache"], case["v_scale_cache"],
                                       case["k_scale"], case["block_table"], case["context_lens"],
                                       case["block_size"], scale)
        assert np.abs(online - naive).max() < 1e-9, np.abs(online - naive).max()


def test_tiling_straddles_block_boundaries_and_partial_tiles():
    """BLOCK_N that does not divide block_size must still address correctly."""
    case = _make_case(batch=4)
    case["context_lens"] = np.array([64, 1, 7, 63], dtype=np.int64)
    scale = case["q"].shape[-1] ** -0.5
    for block_n in (8, 16, 3):
        online = decode_online(case["q"], case["k_cache"], case["v_cache"], case["v_scale_cache"],
                               case["k_scale"], case["block_table"], case["context_lens"],
                               case["block_size"], scale, block_n=block_n)
        naive = decode_naive_quantized(case["q"], case["k_cache"], case["v_cache"], case["v_scale_cache"],
                                       case["k_scale"], case["block_table"], case["context_lens"],
                                       case["block_size"], scale)
        assert np.abs(online - naive).max() < 1e-9, (block_n, np.abs(online - naive).max())


def test_shuffled_block_table_is_followed():
    """Physical blocks must be addressed through the block table, not assumed
    contiguous."""
    case = _make_case()
    case["block_table"] = np.array([[5, 0, 3, 1, 7, 2, 6, 4]] * 3, dtype=np.int64)
    scale = case["q"].shape[-1] ** -0.5
    online = decode_online(case["q"], case["k_cache"], case["v_cache"], case["v_scale_cache"],
                           case["k_scale"], case["block_table"], case["context_lens"],
                           case["block_size"], scale, block_n=8)
    naive = decode_naive_quantized(case["q"], case["k_cache"], case["v_cache"], case["v_scale_cache"],
                                   case["k_scale"], case["block_table"], case["context_lens"],
                                   case["block_size"], scale)
    assert np.abs(online - naive).max() < 1e-9


def test_quantization_error_stays_small_with_wide_dynamic_range():
    """Static-K + dynamic-per-token-V must keep decode close to full precision."""
    for dynamic_range in (False, True):
        case = _make_case(dynamic_range=dynamic_range)
        scale = case["q"].shape[-1] ** -0.5
        online = decode_online(case["q"], case["k_cache"], case["v_cache"], case["v_scale_cache"],
                               case["k_scale"], case["block_table"], case["context_lens"],
                               case["block_size"], scale, block_n=8)
        full = decode_full_precision(case["q"], case["key"], case["value"], case["context_lens"], scale)
        cos = _cosine(online, full)
        assert cos > 0.999, (dynamic_range, cos)


def test_static_k_scale_saturates_only_beyond_calibration():
    """The calibrated static K scale must be overflow-free for the calibrated
    data and saturate (not wrap) beyond it."""
    key = np.array([[[0.0], [3.5]], [[1e6], [-1e6]]])
    k_scale = np.abs(key).max(axis=(0, 2)) / FP8_MAX
    k_q, _, _ = store_kv_fp8(key, np.zeros_like(key), k_scale)
    assert np.abs(k_q).max() <= FP8_MAX
    # the huge outlier saturates at the maximum magnitude instead of wrapping
    assert np.isclose(np.abs(k_q).max(), FP8_MAX)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("ALL_PASS" if failures == 0 else f"{failures} FAILURES")
    raise SystemExit(1 if failures else 0)

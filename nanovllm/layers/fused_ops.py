"""Single-pass Triton fused operators.

Two fusions, each replacing a chain of PyTorch ops (or the multi-kernel
schedule Inductor generates for that chain) with *one* kernel that reads every
input once and writes every output once:

``add_rms_norm(x, residual, weight, eps) -> (normed, new_residual)``
    Residual add + RMSNorm. The baseline materialises a float32 copy of
    ``x + residual``, a downcast residual, and a separate reduction pass, so the
    row crosses HBM three to four times. The kernel keeps the whole row in
    registers: load ``x`` + ``residual``, add, store the new residual, reduce
    ``sum(s^2)`` in-register, apply ``rsqrt`` and the weight, store the normed
    row.

``silu_and_mul(x) -> silu(gate) * up``
    SwiGLU activation over a fused gate/up projection. The baseline writes the
    ``silu(gate)`` intermediate (``[tokens, intermediate]``) back to HBM and
    reads it again for the multiply; the kernel consumes both halves of the row
    and emits only the product.

Everything is written so the common shapes (``hidden``/``head_dim`` powers of
two) compile with no masks at all, and both kernels are CUDA-graph capturable:
the launch grid depends only on the token count and every constexpr only on the
model geometry, so one captured graph replays for any batch size the graph was
captured for.

The torch reference implementations are kept as a fallback (and are what the
parity tests compare against) so the module imports and runs on a CPU-only box.
"""

import functools
import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _TRITON_OK = False


_ENV_VAR = "NANOVLLM_TRITON_FUSION"
_TRUTHY = ("1", "true", "yes", "on")
# Initial value; `ModelRunner` re-applies it from `Config.triton_fusion`.
_ENABLED = os.environ.get(_ENV_VAR, "1").strip().lower() not in ("0", "false", "no", "off")
_HAS_CUDA = _TRITON_OK and torch.cuda.is_available()


def set_fused_enabled(flag: bool) -> None:
    """Enable/disable the Triton path process-wide (used for A/B benchmarks)."""
    global _ENABLED
    _ENABLED = bool(flag)


def fused_enabled() -> bool:
    """True when the fused Triton kernels can actually be launched."""
    return _ENABLED and _HAS_CUDA


# ===========================================================================
# Kernels
# ===========================================================================

if _TRITON_OK:

    @triton.jit
    def _add_rms_norm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        residual_out_ptr,
        M,
        eps,
        stride_xm,
        stride_rm,
        stride_om,
        stride_rom,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROWS: tl.constexpr,
        NEED_ROW_MASK: tl.constexpr,
        NEED_COL_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_N)
        x_off = rows[:, None] * stride_xm + cols[None, :]
        r_off = rows[:, None] * stride_rm + cols[None, :]

        # Only the two shapes we actually meet are specialised: a full tile with
        # no masks, or a tail tile. Avoids predication in the hot path.
        if NEED_ROW_MASK and NEED_COL_MASK:
            mask = (rows < M)[:, None] & (cols < N)[None, :]
            x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            r = tl.load(residual_ptr + r_off, mask=mask, other=0.0)
        elif NEED_ROW_MASK:
            mask = (rows < M)[:, None] & (cols[None, :] >= 0)
            x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            r = tl.load(residual_ptr + r_off, mask=mask, other=0.0)
        elif NEED_COL_MASK:
            mask = (rows[:, None] >= 0) & (cols < N)[None, :]
            x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            r = tl.load(residual_ptr + r_off, mask=mask, other=0.0)
        else:
            mask = None
            x = tl.load(x_ptr + x_off)
            r = tl.load(residual_ptr + r_off)

        s = x.to(tl.float32) + r.to(tl.float32)

        if NEED_COL_MASK:
            w = tl.load(weight_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        else:
            w = tl.load(weight_ptr + cols).to(tl.float32)

        # The new residual is the *un-normalised* sum, stored in the activation
        # dtype exactly like the eager/Inductor path.
        ro_off = rows[:, None] * stride_rom + cols[None, :]
        tl.store(residual_out_ptr + ro_off, s.to(residual_out_ptr.dtype.element_ty), mask=mask)

        # One in-register reduction per row; the row never leaves the registers.
        var = tl.sum(s * s, axis=1) * (1.0 / N)
        rr = tl.rsqrt(var + eps)
        y = s * rr[:, None] * w[None, :]

        o_off = rows[:, None] * stride_om + cols[None, :]
        tl.store(out_ptr + o_off, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        M,
        eps,
        stride_xm,
        stride_om,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROWS: tl.constexpr,
        NEED_ROW_MASK: tl.constexpr,
        NEED_COL_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        cols = tl.arange(0, BLOCK_N)
        x_off = rows[:, None] * stride_xm + cols[None, :]

        if NEED_ROW_MASK and NEED_COL_MASK:
            mask = (rows < M)[:, None] & (cols < N)[None, :]
            x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        elif NEED_ROW_MASK:
            mask = (rows < M)[:, None] & (cols[None, :] >= 0)
            x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        elif NEED_COL_MASK:
            mask = (rows[:, None] >= 0) & (cols < N)[None, :]
            x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        else:
            mask = None
            x = tl.load(x_ptr + x_off)

        s = x.to(tl.float32)

        if NEED_COL_MASK:
            w = tl.load(weight_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        else:
            w = tl.load(weight_ptr + cols).to(tl.float32)

        var = tl.sum(s * s, axis=1) * (1.0 / N)
        rr = tl.rsqrt(var + eps)
        y = s * rr[:, None] * w[None, :]

        o_off = rows[:, None] * stride_om + cols[None, :]
        tl.store(out_ptr + o_off, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _silu_and_mul_kernel(
        x_ptr,
        out_ptr,
        stride_xm,
        stride_om,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        NEED_COL_MASK: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_base = row * stride_xm
        o_base = row * stride_om
        # Static column loop: `N` is model geometry, so the trip count folds at
        # compile time and the common case (N % BLOCK == 0) has no mask.
        for off in tl.static_range(0, N, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            if NEED_COL_MASK:
                cmask = cols < N
                gate = tl.load(x_ptr + x_base + cols, mask=cmask, other=0.0)
                up = tl.load(x_ptr + x_base + N + cols, mask=cmask, other=0.0)
            else:
                gate = tl.load(x_ptr + x_base + cols)
                up = tl.load(x_ptr + x_base + N + cols)
            g = gate.to(tl.float32)
            y = g * tl.sigmoid(g) * up.to(tl.float32)
            if NEED_COL_MASK:
                tl.store(out_ptr + o_base + cols, y.to(out_ptr.dtype.element_ty), mask=cmask)
            else:
                tl.store(out_ptr + o_base + cols, y.to(out_ptr.dtype.element_ty))


# ===========================================================================
# Geometry selection
# ===========================================================================

# Rows per program are picked so every program owns roughly this many elements:
# big enough to amortise the reduction, small enough to stay in registers.
_TARGET_TILE = 1024
_MAX_ROWS = 8
_SILU_BLOCK = 1024
_NUM_WARPS = 4


@functools.lru_cache(maxsize=None)
def _norm_geometry(n_cols: int) -> tuple[int, int]:
    block = triton.next_power_of_2(n_cols)
    # `tl.arange` needs a power-of-two extent, so clamp the row tile to one.
    rows = max(1, min(_MAX_ROWS, _TARGET_TILE // block))
    rows = min(_MAX_ROWS, triton.next_power_of_2(rows))
    return block, rows


def _flat_rows(t: torch.Tensor, n_cols: int) -> torch.Tensor:
    """View ``t`` as ``[rows, n_cols]`` with the last dim contiguous."""
    if t.shape[-1] != n_cols:
        raise ValueError(f"expected last dim {n_cols}, got {tuple(t.shape)}")
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t.reshape(-1, n_cols)


# ===========================================================================
# Torch reference implementations (fallback + parity reference)
# ===========================================================================

def _torch_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    # NB: `x.float()` aliases `x` when it is already fp32, so the add has to be
    # out-of-place or the caller's activation is silently clobbered.
    s = x.float() + residual.float()
    new_residual = s.to(orig_dtype)
    var = s.pow(2).mean(dim=-1, keepdim=True)
    y = (s * torch.rsqrt(var + eps)).to(orig_dtype) * weight
    return y, new_residual


def _torch_rms_norm(x, weight, eps):
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(orig_dtype) * weight


def _torch_silu_and_mul(x):
    gate, up = x.chunk(2, -1)
    return F.silu(gate) * up


# ===========================================================================
# Public API
# ===========================================================================

def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``residual + x`` followed by RMSNorm. Returns (normed, new_residual)."""
    if not fused_enabled():
        return _torch_add_rms_norm(x, residual, weight, eps)

    n_cols = x.shape[-1]
    x2 = _flat_rows(x, n_cols)
    r2 = _flat_rows(residual, n_cols)
    m = x2.shape[0]

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    residual_out = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    o2 = out.view(-1, n_cols)
    ro2 = residual_out.view(-1, n_cols)

    block, rows = _norm_geometry(n_cols)
    _add_rms_norm_kernel[(triton.cdiv(m, rows),)](
        x2, r2, weight, o2, ro2,
        m, float(eps),
        x2.stride(0), r2.stride(0), o2.stride(0), ro2.stride(0),
        N=n_cols, BLOCK_N=block, ROWS=rows,
        NEED_ROW_MASK=rows > 1, NEED_COL_MASK=block != n_cols,
        num_warps=_NUM_WARPS,
    )
    return out, residual_out


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Single-pass RMSNorm (no residual add)."""
    if not fused_enabled():
        return _torch_rms_norm(x, weight, eps)

    n_cols = x.shape[-1]
    x2 = _flat_rows(x, n_cols)
    m = x2.shape[0]

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    o2 = out.view(-1, n_cols)

    block, rows = _norm_geometry(n_cols)
    _rms_norm_kernel[(triton.cdiv(m, rows),)](
        x2, weight, o2,
        m, float(eps),
        x2.stride(0), o2.stride(0),
        N=n_cols, BLOCK_N=block, ROWS=rows,
        NEED_ROW_MASK=rows > 1, NEED_COL_MASK=block != n_cols,
        num_warps=_NUM_WARPS,
    )
    return out


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: ``silu(x[..., :N]) * x[..., N:]`` in a single pass."""
    if not fused_enabled():
        return _torch_silu_and_mul(x)

    last = x.shape[-1]
    if last % 2 != 0:
        raise ValueError(f"expected an even last dim, got {last}")
    n_cols = last // 2

    x2 = _flat_rows(x, last)
    m = x2.shape[0]

    out = torch.empty((*x.shape[:-1], n_cols), dtype=x.dtype, device=x.device)
    o2 = out.view(-1, n_cols)

    block = min(_SILU_BLOCK, triton.next_power_of_2(n_cols))
    _silu_and_mul_kernel[(m,)](
        x2, o2,
        x2.stride(0), o2.stride(0),
        N=n_cols, BLOCK=block, NEED_COL_MASK=(n_cols % block != 0),
        num_warps=_NUM_WARPS,
    )
    return out

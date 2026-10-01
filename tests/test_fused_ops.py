"""Fused Add+RMSNorm / SiluAndMul kernel correctness tests.

Layout:
  * CPU tests  : geometry selection, tensor-flattening helpers and the
                 module-level dispatch fallback (no CUDA needed).
  * GPU tests  : Triton kernel parity against the pure-torch reference for a
                 spread of shapes/dtypes, tail-mask handling, CUDA-graph
                 capture/replay, and the module-level A/B (fused vs. the
                 torch.compile path it replaces).
  * GPU + model: end-to-end greedy token agreement between fusion off and on.

Run everything:      pytest tests/test_fused_ops.py -v
Run CPU only:        pytest tests/test_fused_ops.py -v -m "not gpu"
Kernel + module:     pytest tests/test_fused_ops.py -v -m "not logits"
Logits alignment:    NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B \
                     pytest tests/test_fused_ops.py -v -m logits
"""

import os

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from nanovllm.layers import fused_ops  # noqa: E402
from nanovllm.layers.activation import SiluAndMul  # noqa: E402
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_TOL = {
    torch.float32: dict(rtol=1e-5, atol=1e-6),
    torch.float16: dict(rtol=2e-2, atol=2e-2),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}


# ===========================================================================
# CPU: geometry / helpers / dispatch fallback
# ===========================================================================

def test_norm_geometry_is_power_of_two_and_tiled():
    for n in (128, 1024, 3072, 4096, 5000):
        block, rows = fused_ops._norm_geometry(n)
        assert block == triton.next_power_of_2(n)
        assert block >= n
        assert 1 <= rows <= fused_ops._MAX_ROWS
        # tl.arange requires a power-of-two extent for the row tile
        assert rows & (rows - 1) == 0
        if block <= fused_ops._TARGET_TILE:
            # the tile never exceeds 2x the target, so register pressure is bounded
            assert block * rows <= 2 * fused_ops._TARGET_TILE


def test_flat_rows_views_and_copies():
    x = torch.randn(4, 8, 16, dtype=torch.float32)
    flat = fused_ops._flat_rows(x, 16)
    assert flat.shape == (32, 16)
    assert flat.stride(0) == 16
    # the view shares storage with the input
    assert flat.data_ptr() == x.data_ptr()

    # a non-contiguous last dim is materialised instead of mis-read
    w = torch.randn(8, 32, dtype=torch.float32)[:, ::2]
    assert w.stride(-1) == 2
    flat_w = fused_ops._flat_rows(w, 16)
    assert flat_w.stride(-1) == 1
    torch.testing.assert_close(flat_w, w.reshape(-1, 16))

    with pytest.raises(ValueError):
        fused_ops._flat_rows(x, 15)


def test_torch_reference_matches_naive_formula():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 8, dtype=torch.float32)
    residual = torch.randn(3, 5, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)
    eps = 1e-6

    x0, residual0 = x.clone(), residual.clone()
    got_y, got_r = fused_ops._torch_add_rms_norm(x, residual, weight, eps)
    # `x.float()` aliases an fp32 input, so the reference must not add in place
    torch.testing.assert_close(x, x0)
    torch.testing.assert_close(residual, residual0)
    s = x + residual
    var = s.pow(2).mean(-1, keepdim=True)
    ref_y = s * torch.rsqrt(var + eps) * weight
    torch.testing.assert_close(got_y, ref_y)
    torch.testing.assert_close(got_r, s)

    got = fused_ops._torch_rms_norm(x, weight, eps)
    var = x.pow(2).mean(-1, keepdim=True)
    torch.testing.assert_close(got, x * torch.rsqrt(var + eps) * weight)

    gate_up = torch.randn(3, 5, 8, dtype=torch.float32)
    got = fused_ops._torch_silu_and_mul(gate_up)
    gate, up = gate_up.chunk(2, -1)
    torch.testing.assert_close(got, torch.nn.functional.silu(gate) * up)


def test_modules_dispatch_to_torch_when_disabled():
    """With the Triton path disabled the modules must route to the torch path
    and keep the (normed, residual) contract the decoder relies on.

    The compiled methods are swapped for the plain torch reference so this runs
    without an Inductor toolchain; the real compiled baseline is exercised by
    the GPU module-parity test.
    """
    torch.manual_seed(0)
    hidden, eps = 64, 1e-6
    norm = RMSNorm(hidden, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden))

    was = fused_ops._ENABLED
    fused_ops.set_fused_enabled(False)
    try:
        # force the fallback branch and replace the compiled chain
        norm.rms_forward = lambda x: fused_ops._torch_rms_norm(x, norm.weight, eps)
        norm.add_rms_forward = lambda x, r: fused_ops._torch_add_rms_norm(x, r, norm.weight, eps)

        x = torch.randn(4, hidden)
        residual = torch.randn(4, hidden)
        y, r = norm(x, residual)
        s = x + residual
        var = s.pow(2).mean(-1, keepdim=True)
        torch.testing.assert_close(y, s * torch.rsqrt(var + eps) * norm.weight)
        torch.testing.assert_close(r, s)

        # residual=None path
        y = norm(x)
        var = x.pow(2).mean(-1, keepdim=True)
        torch.testing.assert_close(y, x * torch.rsqrt(var + eps) * norm.weight)

        act = SiluAndMul()
        act._compiled_forward = lambda g: fused_ops._torch_silu_and_mul(g)
        gate_up = torch.randn(4, 4 * hidden)
        got = act(gate_up)
        gate, up = gate_up.chunk(2, -1)
        torch.testing.assert_close(got, torch.nn.functional.silu(gate) * up)
    finally:
        fused_ops.set_fused_enabled(was)


# ===========================================================================
# GPU: kernel parity
# ===========================================================================

def _rand(shape, dtype, device):
    x = torch.randn(shape, dtype=torch.float32, device=device)
    if dtype == torch.bfloat16:
        return x.to(dtype)
    return x.to(dtype)


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n_cols", [128, 1024, 4096, 1000])
@pytest.mark.parametrize("m", [1, 7, 33, 128])
def test_add_rms_norm_kernel_parity(m, n_cols, dtype):
    """Fused kernel vs. the torch reference, including a non-power-of-two N
    (column tail mask) and M values that do not divide the row tiling."""
    torch.manual_seed(0)
    device = "cuda"
    x = _rand((m, n_cols), dtype, device)
    residual = _rand((m, n_cols), dtype, device)
    weight = _rand((n_cols,), dtype, device)

    fused_ops.set_fused_enabled(True)
    got_y, got_r = fused_ops.add_rms_norm(x, residual, weight, 1e-6)
    ref_y, ref_r = fused_ops._torch_add_rms_norm(x, residual, weight, 1e-6)

    assert got_y.shape == x.shape and got_y.dtype == x.dtype
    assert got_r.shape == residual.shape and got_r.dtype == residual.dtype
    assert not torch.isnan(got_y).any()
    torch.testing.assert_close(got_r, ref_r, **_TOL[dtype])
    torch.testing.assert_close(got_y, ref_y, **_TOL[dtype])


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_cols", [128, 1024, 1000])
@pytest.mark.parametrize("m", [1, 9, 64])
def test_rms_norm_kernel_parity(m, n_cols, dtype):
    torch.manual_seed(0)
    device = "cuda"
    x = _rand((m, n_cols), dtype, device)
    weight = _rand((n_cols,), dtype, device)

    fused_ops.set_fused_enabled(True)
    got = fused_ops.rms_norm(x, weight, 1e-6)
    ref = fused_ops._torch_rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(got, ref, **_TOL[dtype])


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_cols", [128, 1024, 3072, 4096, 1000])
@pytest.mark.parametrize("m", [1, 7, 128])
def test_silu_and_mul_kernel_parity(m, n_cols, dtype):
    torch.manual_seed(0)
    device = "cuda"
    x = _rand((m, 2 * n_cols), dtype, device)

    fused_ops.set_fused_enabled(True)
    got = fused_ops.silu_and_mul(x)
    ref = fused_ops._torch_silu_and_mul(x)
    assert got.shape == (m, n_cols) and got.dtype == x.dtype
    torch.testing.assert_close(got, ref, **_TOL[dtype])


@requires_cuda
@pytest.mark.gpu
def test_kernels_accept_3d_and_non_contiguous_inputs():
    """The decoder hands q/k to RMSNorm as [tokens, heads, head_dim]; q_norm is
    the hot 3-D case. Non-contiguous rows must be handled, not mis-read."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    tokens, heads, head_dim = 11, 4, 128

    q = _rand((tokens, heads, head_dim), dtype, device)
    weight = _rand((head_dim,), dtype, device)
    fused_ops.set_fused_enabled(True)
    got = fused_ops.rms_norm(q, weight, 1e-6)
    ref = fused_ops._torch_rms_norm(q, weight, 1e-6)
    assert got.shape == q.shape
    torch.testing.assert_close(got, ref, **_TOL[dtype])

    # Non-contiguous last dim: fused path must copy rather than mis-address.
    q_bad = q[:, :, ::2].contiguous()[:, :, ::2]
    assert q_bad.stride(-1) == 2
    n2 = q_bad.shape[-1]
    w2 = _rand((n2,), dtype, device)
    got = fused_ops.rms_norm(q_bad, w2, 1e-6)
    ref = fused_ops._torch_rms_norm(q_bad, w2, 1e-6)
    torch.testing.assert_close(got, ref, **_TOL[dtype])

    gate_up = _rand((tokens, 2 * 512), dtype, device)[:, ::2]
    assert gate_up.stride(-1) == 2
    got = fused_ops.silu_and_mul(gate_up)
    ref = fused_ops._torch_silu_and_mul(gate_up)
    torch.testing.assert_close(got, ref, **_TOL[dtype])


@requires_cuda
@pytest.mark.gpu
def test_module_parity_against_torch_compile_path():
    """The real modules, fused kernel vs. the torch.compile chain they replace."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    hidden, eps = 1024, 1e-6
    norm = RMSNorm(hidden, eps=eps).to(device=device, dtype=dtype)
    act = SiluAndMul().to(device=device, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(_rand((hidden,), dtype, device))

    x = _rand((257, hidden), dtype, device)
    residual = _rand((257, hidden), dtype, device)
    gate_up = _rand((257, 2 * 3072), dtype, device)

    was = fused_ops._ENABLED
    fused_ops.set_fused_enabled(False)
    try:
        ref_y, ref_r = norm(x, residual)
        ref_act = act(gate_up)
    finally:
        fused_ops.set_fused_enabled(was)

    fused_ops.set_fused_enabled(True)
    got_y, got_r = norm(x, residual)
    got_act = act(gate_up)

    torch.testing.assert_close(got_r, ref_r, **_TOL[dtype])
    torch.testing.assert_close(got_y, ref_y, **_TOL[dtype])
    torch.testing.assert_close(got_act, ref_act, **_TOL[dtype])


@requires_cuda
@pytest.mark.gpu
def test_kernels_are_cuda_graph_capturable():
    """Grid/constexpr depend only on model geometry and the token count, so a
    captured graph replays correctly after in-place input updates."""
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    m, hidden, inter = 64, 1024, 3072

    x = _rand((m, hidden), dtype, device)
    residual = _rand((m, hidden), dtype, device)
    weight = _rand((hidden,), dtype, device)
    gate_up = _rand((m, 2 * inter), dtype, device)

    fused_ops.set_fused_enabled(True)
    # eager warmup materialises outputs in the graph pool before capture
    fused_ops.add_rms_norm(x, residual, weight, 1e-6)
    fused_ops.silu_and_mul(gate_up)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_y, graph_r = fused_ops.add_rms_norm(x, residual, weight, 1e-6)
        graph_act = fused_ops.silu_and_mul(gate_up)

    for seed in (1, 2):
        torch.manual_seed(seed)
        x.copy_(_rand((m, hidden), dtype, device))
        residual.copy_(_rand((m, hidden), dtype, device))
        gate_up.copy_(_rand((m, 2 * inter), dtype, device))
        graph.replay()
        torch.cuda.synchronize()

        ref_y, ref_r = fused_ops._torch_add_rms_norm(x, residual, weight, 1e-6)
        ref_act = fused_ops._torch_silu_and_mul(gate_up)
        torch.testing.assert_close(graph_r, ref_r, **_TOL[dtype])
        torch.testing.assert_close(graph_y, ref_y, **_TOL[dtype])
        torch.testing.assert_close(graph_act, ref_act, **_TOL[dtype])


@requires_cuda
@pytest.mark.gpu
def test_disabled_path_never_launches_triton():
    """set_fused_enabled(False) must route to torch, so benchmarks that compare
    against the torch.compile baseline actually measure the baseline."""
    fused_ops.set_fused_enabled(False)
    try:
        assert not fused_ops.fused_enabled()
        x = _rand((8, 256), torch.bfloat16, "cuda")
        y, r = fused_ops.add_rms_norm(x, x, torch.ones(256, device="cuda"), 1e-6)
        assert y.device.type == "cuda"
    finally:
        fused_ops.set_fused_enabled(True)


# ===========================================================================
# GPU + real model: end-to-end token agreement
# ===========================================================================

@pytest.mark.gpu
@pytest.mark.logits
@requires_cuda
def test_logits_alignment_fusion_on_vs_off():
    """Greedy token streams must agree (near-)exactly with fusion off vs on.

    Delegates to tools/check_fusion_parity.py, which runs each setting in its
    own process and compares the generated token ids.
    """
    model_path = os.environ.get("NANOVLLM_TEST_MODEL")
    if not model_path:
        pytest.skip("set NANOVLLM_TEST_MODEL to run the end-to-end alignment")
    import json
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tool = os.path.join(repo, "tools", "check_fusion_parity.py")
    proc = subprocess.run(
        [sys.executable, tool, "--model", model_path, "--num-prompts", "5", "--max-tokens", "32"],
        capture_output=True, text=True, cwd=repo,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-3000:]
    summary = json.loads([l for l in proc.stdout.splitlines() if l.startswith("SUMMARY ")][-1][len("SUMMARY "):])

    # greedy argmax can flip on a bf16 ulp; the logits themselves must agree
    assert summary["token_match_rate"] >= 0.95, summary
    assert summary["worst_logit_cosine"] > 0.999, summary

<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, chunked prefill mixed scheduling, Tensor Parallelism, Torch compilation, CUDA graph, FP8 KV cache, fused Triton operators, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Chunked Prefill Mixed Scheduling

The baseline scheduler makes a step *either* a prefill step or a decode step, so
while a waiting prompt is being prefilled every in-flight decode stalls behind
it (head-of-line blocking). `chunked_prefill=True` (the default) removes the
step-level mutual exclusion:

* **Decode first, strictly.** Every running sequence is scheduled for exactly one
  token before any prefill is considered, and the decode batch runs through its
  captured CUDA graph. The decode cost of a step is therefore `len(running)`
  tokens no matter how long the waiting queue is.
* **Prefill on the leftover budget, eagerly.** The prefill pass then spends
  `max_num_batched_tokens - num_decode_tokens` on the waiting queue. A prompt
  that does not fit is split into a chunk and rotated behind the other waiters,
  so a single long prompt neither blocks decode nor idles the rest of the budget
  (the legacy rule chunked only the *first* sequence and stopped, wasting the
  remainder).
* **One step, two dispatches.** `ModelRunner.run(decode_seqs, prefill_seqs)`
  executes the graph-captured decode batch and then the eager prefill chunk. The
  context singleton is reset between the two so each phase sees its own
  `cu_seqlens` / `slot_mapping`, and only the decode path replays a graph.

`Scheduler.schedule()` returns a `Schedule(decode, prefill)`; `postprocess`
advances a chunked sequence's computed-token cursor without emitting a token
until its last chunk completes, and `LLMEngine.step()` reports prefill and
decode token counts separately.

```python
llm = LLM("/path/to/Qwen3-0.6B", max_num_batched_tokens=1024)    # token budget = chunk size
llm = LLM(..., chunked_prefill=True, chunked_prefill_size=1024)   # per-sequence chunk cap
llm = LLM(..., chunked_prefill=False)                             # legacy exclusive scheduler
```

`max_num_batched_tokens` is the per-step token budget; `chunked_prefill_size`
optionally caps how much of one sequence a step may take. The budget also bounds
the eager prefill activation working set: a full 16384-token prefill allocates
~0.58 GiB of activations, while capping it at 1024 drops that to ~0.07 GiB (about
2.5% of the ~20.6 GiB total peak, which is dominated by the KV cache).

### Correctness

```bash
pytest tests/test_chunked_prefill.py -v     # 12 host-side scheduler tests, CPU only
```

The tests pin the properties the refactor relies on: decode is emitted before
prefill and never starved, prefill gets only the leftover budget, a long prompt
is chunked without emitting a token until the last chunk, a sequence appears at
most once per step, prefix-cached prompts still advance by a positive chunk, and
the legacy scheduler (`chunked_prefill=False`) never mixes the two batches.

On GPU the mixed scheduler was checked for *greedy token equivalence* against the
legacy scheduler: with an argmax sampler, three prompts (300/50/260 tokens) run
under `chunked_prefill=False`, `chunked_prefill=True` and a forced per-sequence
chunk cap all produce **identical token streams**, while the mixed runs execute
decode and prefill in the same step (3 and 4 mixed steps respectively).

### Measured results

RTX 4090 24 GB, Qwen3-0.6B, bf16, CUDA graphs enabled. `tools/bench_chunked_prefill.py`
runs the exclusive and mixed schedulers in separate processes and alternates the
order. Static batch: 256 sequences, mixed 100-1024 token inputs/outputs (144,160
tokens), `max_num_batched_tokens=1024`, median of 3 repeats:

| Metric | Exclusive | Chunked | Delta |
|---|---|---|---|
| Throughput (tok/s) | 15130.9 | 15831.1 | **+4.6%** |
| Wall (s) | 9.53 | 9.11 | −4.4% |
| Request latency p50 (ms) | 7728.9 | 7207.6 | −6.7% |
| Request latency p99 (ms) | 9510.6 | 9099.6 | −4.3% |
| Request latency p999 (ms) | 9526.9 | 9105.6 | −4.4% |
| TTFT p999 (ms) | 4349.1 | 4301.2 | −1.1% |
| Step p50 (ms) | 5.65 | 5.42 | −4.1% |
| Activation peak (GiB) | 0.0725 | 0.0723 | −0.3% |

The mixed step does ~4% less work per step by folding the prefill chunk into the
decode step (1158 steps vs 1212), which is where the throughput gain comes from.
On a *batch* workload the tail is bounded by the total work, so the request-p999
gain is modest.

The tail-latency effect shows up when long prompts arrive while other requests
are already decoding — the case the refactor targets. `--decode-wave 32` starts
32 long-output decoders, then delivers 224 1024-token prompts 300 ms later:

| Metric | Exclusive | Chunked | Delta |
|---|---|---|---|
| TBT p999 (ms) | 2524.2 | 134.2 | **−94.7%** |
| TBT p99 (ms) | 7.6 | 29.2 | +286% |
| Request latency p999 (ms) | 6449.1 | 6633.1 | +2.9% |

Exclusive freezes the decode stream for ~2.5 s while the long-prompt wave is
prefilled; the mixed scheduler caps that stall at one prefill chunk (~134 ms), at
the cost of a higher *typical* inter-token gap (every step now carries a chunk)
and a slower first token for the long prompts themselves.

```bash
python tools/bench_chunked_prefill.py --model ~/huggingface/Qwen3-0.6B \
    --num-seqs 256 --budget 1024 --repeat 3
python tools/bench_chunked_prefill.py --model ~/huggingface/Qwen3-0.6B \
    --budget 1024 --decode-wave 32 --second-wave-delay-ms 300 --max-output-len 512
```

## FP8 E4M3 KV Cache

`fp8_kvcache=True` quantizes the paged KV cache to FP8 E4M3, which roughly
**doubles KV cache capacity** and therefore the maximum number of concurrent
sequences that fit on the GPU. Decode and prefill have dedicated Triton paths:

| Stage | Path |
|---|---|
| Write | Triton quantized store: **K** with a static per-(layer, KV head) scale, **V** with a dynamic per-(token, KV head) scale persisted beside the cache |
| Decode | Triton FP8 paged attention, grid = `batch × KV head`; every query head of a GQA group **shares one K tile, V tile and scale load** per iteration, with fused physical block addressing, tiled computation, FP8 dequant and online-softmax merging. The shared K/V tiles feed a **tensor-core `tl.dot`** (the GQA group is padded to 16 rows) |
| Prefill (prefix cache hit) | **Gather only the referenced slots + dequantize** into a dense buffer, then dense `flash_attn_varlen_func` — no full-cache `.to()` (which OOMs) |

Key details:

* **K static / V dynamic.** K's magnitude is stable across tokens, so one
  calibration pass (random tokens through the real model, before the cache is
  allocated) fixes a per-head scale and the write kernel needs no reduction.
  The calibrated range is widened by `fp8_kv_k_margin` (default 1.5) because a
  small calibration sample underestimates the real amax — without it up to 75%
  of (layer, head) pairs saturate on real prompts.
  V is reduced per token and stored as fp16 in a `[layers, num_slots, kv_heads]`
  side buffer — about 0.8% overhead, keeping the capacity gain at ~2x.
* **K and V need their own row stride.** In a real model RoPE rebuilds K (so it
  is contiguous) while V stays a strided view into the fused QKV output; a
  store that reuses one stride reads the wrong memory for every token after the
  first. The store kernel takes `key_stride` and `value_stride` separately.
* **CUDA Graph safe.** The decode kernel's loop bound is read from
  `context_lens` at execution time and the grid is static `(max_bs, kv_heads)`,
  so a single captured graph replays correctly for any context length and block
  table (dynamic context replay).

```python
llm = LLM("/path/to/Qwen3-0.6B", fp8_kvcache=True, max_model_len=4096)
```

`Config` knobs: `fp8_kvcache`, `fp8_kv_calib_tokens` (default 512),
`fp8_kv_k_margin` (default 1.5), `fp8_kv_k_scale` (skip calibration with an
explicit static K scale).
Requires an FP8-capable GPU (compute capability >= 8.9).

### Measured results

RTX 4090 24 GB, a Qwen2 0.5B-class model (24 layers, 2 KV heads, head_dim 64),
`max_model_len=4096`, block size 256, `bench_fp8.py --mode all`.

| Metric | FP16 | FP8 E4M3 | Ratio |
|---|---|---|---|
| KV cache blocks | 6527 | 12836 | **1.967** |
| KV cache size (GiB) | 19.12 | 18.80 | 0.983 |
| Max no-preemption concurrency (4096 ctx) | 407 | 802 | **+96.7%** |
| Decode attention, batch 512 / ctx 2048 | 0.270 ms | 0.252 ms | **0.93x** |
| Decode attention, batch 1024 / ctx 1024 | 0.281 ms | 0.247 ms | **0.88x** |

Max no-preemption concurrency is `num_kvcache_blocks // ceil(max_model_len /
block_size)`, so it can only double when the per-block bytes halve exactly. The
fp16 dynamic-V scale side buffer adds 0.8% to the FP8 block, giving 1.967x
(+96.7%); a "+108%" concurrency figure is not reachable from capacity alone.

Decode-kernel history on this machine: the first fp32-reduction implementation
ran at `0.416 ms` (10.3x *slower* than `flash_attn_with_kvcache`); switching the
shared K/V tiles to a tensor-core `tl.dot` and iterating physical blocks with
contiguous sub-tiles brought it to `0.040 ms` at batch 128 / ctx 1024 — a 10.4x
speedup that lands it at parity with the FP16 flash-attention baseline.

End-to-end throughput. The gain depends entirely on whether KV capacity is the
binding runtime constraint:

| Workload | FP16 tok/s | FP8 tok/s | FP16 preemptions | Delta |
|---|---|---|---|---|
| 256 seqs, mixed 100-1024 token I/O | 22097 | 24926 | 0 | +12.8% |
| 512 seqs, mixed 100-1024 token I/O | 26986 | 31058 | 0 | +15.1% |
| 768 seqs @ 4096 ctx (fixed 2048 in + 2048 out) | 14507 | 23052 | **361 vs 0** | **+58.9%** |
| same, both eager (no CUDA graph) | 13499 | 22978 | **361 vs 0** | **+70.2%** |

With mixed short sequences the KV limit never binds: both modes run every
sequence concurrently with zero preemptions, so the only gain is the kernel's
per-token speedup (+13-15%). Make every sequence reach `max_model_len` instead
and the capacity difference dominates — `num_kvcache_blocks` caps FP16 at 407
concurrent 4096-token sequences versus FP8's 802, so FP16 is forced to preempt
361 times (evicting and recomputing sequences) while FP8 never preempts, and
end-to-end throughput improves **+58.9%** (or **+70.2%** when both sides run
eager, which removes the CUDA-graph asymmetry of large batches). `bench_fp8.py
--fixed-lengths` reproduces it.

### Correctness

```bash
pytest tests/test_fp8_kv.py -v -m "not gpu"        # host-side math, CPU
pytest tests/test_fp8_kv.py -v                     # + Triton kernel parity & CUDA-graph replay
NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B \
  pytest tests/test_fp8_kv.py -v -m logits         # + logits-level alignment vs FP16
```

The kernel parity tests compare against a pure-torch reference that reads the
*same* quantized cache, isolating kernel bugs from quantization error; a
separate test bounds the quantization error itself with a cosine-similarity
check against a full-precision decode.

Logits-level alignment (`tools/logits_align.py`, which forces the FP8 run to
consume the FP16 run's greedy token stream so every step compares identical
inputs):

| Phase | Prefill cosine | Worst decode cosine | Decode top-1 agreement |
|---|---|---|---|
| Plain prefill + decode | 0.99996 | 0.988 | **100%** |
| Prefix-cache gather prefill | 0.996 | 0.984 | 96.7% |

### Benchmark

```bash
python bench_fp8.py --model ~/huggingface/Qwen3-0.6B --mode all
```

Runs FP16 and FP8 in separate subprocesses and reports KV blocks, KV cache
size, max no-preemption concurrency (`num_kvcache_blocks // ceil(max_model_len
/ block_size)`), end-to-end throughput and the FP8/FP16 ratios.

Two more harnesses live in `tools/`:

```bash
python tools/bench_decode_kernel.py --sweep    # decode kernel vs flash_attn, sweeps BLOCK_N/warps/dot
python tools/logits_align.py --model ...       # forced-token FP16 vs FP8 logits comparison
```

## Fused Triton Operators (Add+RMSNorm, SiluAndMul)

`nanovllm/layers/fused_ops.py` replaces the `torch.compile`'d chains behind
`RMSNorm` and `SiluAndMul` with one-pass Triton kernels. Each kernel reads every
input exactly once and writes every output exactly once, keeping the whole row
in registers so nothing round-trips through HBM:

| Operator | Baseline (Inductor) | Fused single-pass kernel |
|---|---|---|
| `add_rms_norm(x, residual, weight)` | residual add, downcast residual, reduce `mean(x²)`, normalise — the row crosses HBM 3-4x | load `x`+`residual`, add, store the new residual, reduce `Σx²` **in registers**, apply `rsqrt`+weight, store |
| `silu_and_mul(gate_up)` | materialises the `silu(gate)` intermediate and re-reads it for the multiply | load the gate and up halves, write only the product |

Design details:

* **Mask-free hot path.** `N` is model geometry, so it is a `constexpr`:
  power-of-two widths (`hidden=1024`, `head_dim=128`) compile to a single
  unmasked load/store per row, and `N=3072` (SwiGLU) unrolls into three
  full-width tiles. Tail masks only appear for genuinely non-power-of-two
  widths.
* **Row tiling.** Small widths (`head_dim=128` for `q_norm`/`k_norm`) process
  `ROWS=8` rows per program so each program still owns ~1024 elements; large
  widths use one row per program. Both keep the reduction/weight broadcast
  inside registers.
* **CUDA-graph safe.** The grid depends only on the token count and every
  `constexpr` only on the model geometry, so one captured decode graph replays
  correctly for any batch the graph was captured for (the same guarantee the FP8
  decode kernel relies on).
* **Precision-preserving.** The sum of squares and the normalisation are
  accumulated in fp32 and rounded once on store; the residual is stored in the
  activation dtype, matching the eager/Inductor semantics.
* **Fallback.** With `triton_fusion=False` (or no CUDA) the modules route to the
  original `torch.compile` methods, so the feature is A/B-testable and safe on a
  CPU-only box.

```python
llm = LLM("/path/to/Qwen3-0.6B", triton_fusion=True)   # default
```

### Correctness

```bash
pytest tests/test_fused_ops.py -v -m "not logits"     # CPU geometry + GPU kernel/module parity
NANOVLLM_TEST_MODEL=~/huggingface/Qwen3-0.6B \
  pytest tests/test_fused_ops.py -v -m logits          # end-to-end greedy token agreement
```

The GPU parity tests compare the kernels against the pure-torch reference over
power-of-two and tail widths, 2-D and 3-D (q/k norm) inputs, non-contiguous
strides, and under CUDA-graph capture/replay. The end-to-end test
(`tools/check_fusion_parity.py`) uses teacher forcing: fusion-on consumes the
fusion-off greedy token stream, so both runs see identical contexts and the
comparison is per-step logit fidelity plus greedy-decision agreement.

### Benchmark

```bash
python tools/bench_fused_ops.py            # per-operator time, effective GB/s, % of peak
python tools/bench_fused_ops.py --sweep    # tune ROWS / BLOCK / num_warps
python tools/bench_fused_e2e.py --model ~/huggingface/Qwen3-0.6B --repeat 6
python tools/bench_fused_e2e.py --model ... --mode prefill   # activation-peak comparison
python tools/bench_fused_e2e.py --model ... --eager-baseline # baseline = unfused torch
```

`bench_fused_ops.py` reports the *effective* bandwidth (ideal single-pass bytes
over time) as a fraction of both the device's theoretical peak and a measured
copy bandwidth, and verifies the fused result against the two baselines (the
`torch.compile` chain and the plain unfused torch chain). The e2e harness runs
the two sides in separate processes with alternating order and reports medians
over repeats: eager throughput, inter-token latency percentiles, per-request
latency percentiles and the activation peak (peak allocation minus the identical
KV-cache/weight footprint).

### Measured results

RTX 4090 24 GB, torch 2.5.1+cu124, triton 3.1.0, Qwen3-0.6B, bf16. Measured
device copy bandwidth (read+write) is **918 GB/s** and is used as the practical
peak; L2 is evicted between iterations so the numbers are HBM-representative.
Effective bandwidth = ideal single-pass bytes / time.

| Operator | M (rows) | eager unfused | `torch.compile` | **fused** | vs eager | vs compile |
|---|---|---|---|---|---|---|
| `add_rms_norm` | 16384 | 14.8% | 75.4% | **95.8%** | 6.46x | 1.27x |
| `add_rms_norm` | 8192 | 23.5% | 74.7% | **90.0%** | 3.83x | 1.20x |
| `add_rms_norm` | 2048 | 23.0% | 56.5% | **62.5%** | 2.72x | 1.11x |
| `rms_norm` | 262144 | 11.0% | 96.3% | **97.0%** | 8.85x | 1.01x |
| `silu_and_mul` | 16384 | 56.8% | 99.1% | **96.8%** | 1.70x | 0.98x |
| `silu_and_mul` | 8192 | 54.7% | 96.4% | **93.1%** | 1.70x | 0.96x |

End-to-end, eager, 256 sequences with mixed 100-1024 token inputs/outputs
(144k tokens, 1023 decode steps), medians over 6 alternating repeats:

| Metric | vs unfused eager | vs `torch.compile` |
|---|---|---|
| Throughput | **+5.8%** | +0.4% (noise) |
| Inter-token latency p50 | **−4.4%** | −0.9% (noise) |
| Inter-token latency p99 | **−8.4%** | −2.1% (noise) |
| Request latency p50 | **−7.5%** | −0.7% (noise) |
| Activation peak | **−17.2%** (0.702 → 0.582 GiB) | 0.0% |
| Prefill step time (8x2048 tok) | **−26.1%** | +3.5% (noise) |

What these numbers mean:

* The fused kernels land at **90-97% of achievable HBM bandwidth** for
  Add+RMSNorm and are **3.8-6.5x faster than the unfused eager chain**, whose
  float32 intermediate plus separate reduction/normalise/downcast passes put it
  at 15-24% of peak. `rms_norm` follows the same pattern (11% → 97%, up to
  8.9x).
* **The headroom against a working `torch.compile` is much smaller than the raw
  fusion ratio**: on this stack Inductor already emits one fused Triton kernel
  per operator and reaches 75% (Add+RMSNorm) to 99% (SwiGLU) of peak, so the
  fused kernel only wins 1.27x there and ties on the two operators that were
  already saturating. `SiluAndMul` in particular was *already* a single fused
  pointwise kernel in Inductor, so it is a wash (0.96-0.98x).
* Because `torch.compile` already avoids the fp32/SwiGLU intermediates, the
  activation-memory win also exists only against the unfused chain (−17.2%,
  vs. the 7% target); against Inductor the peak is identical.
* The e2e deltas are quoted with the run-to-run noise floor in mind: two
  identical configurations (`--ops none`, i.e. both sides on the compiled path)
  differ by ±2% on this shared machine, and alternating the run order is what
  makes the unfused-eager deltas reproducible.

So the "~20% → ~95% bandwidth, 4-5x" and "7% memory" figures describe the
comparison against the **unfused** implementation (and are reproduced above);
against the repository's default `torch.compile` baseline the same kernels are
bandwidth-optimal but the surplus is 1.0-1.27x and the e2e effect is neutral.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
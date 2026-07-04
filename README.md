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
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
hf download Qwen/Qwen3-4B \
  --local-dir ~/huggingface/Qwen3-4B/
```

For the Hugging Face draft-model speculative path, also download the first draft model:

```bash
hf download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/
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

## Benchmark

See `bench.py` for benchmark.

Baseline run:

```bash
python bench.py \
  --model ~/huggingface/Qwen3-4B/ \
  --mode baseline \
  --out outputs/bench-baseline.json
```

DSpark-configured run:

```bash
python bench.py \
  --model ~/huggingface/Qwen3-4B/ \
  --mode dspark \
  --draft-model deepseek-ai/dspark_qwen3_4b_block7 \
  --num-speculative-tokens 7 \
  --out outputs/bench-dspark.json
```

Experimental engine-level n-gram speculative run:

```bash
python bench.py \
  --model ~/huggingface/Qwen3-4B/ \
  --enable-engine-speculative \
  --num-speculative-tokens 7 \
  --ngram-size 4 \
  --out outputs/bench-engine-speculative.json
```

Experimental engine-level Hugging Face draft-model speculative run:

```bash
python bench.py \
  --model ~/huggingface/Qwen3-4B/ \
  --enable-engine-speculative \
  --draft-type hf \
  --draft-model ~/huggingface/Qwen3-0.6B/ \
  --draft-dtype bfloat16 \
  --temperature 0.01 \
  --num-speculative-tokens 7 \
  --out outputs/bench-hf-draft-speculative.json
```

The HF draft path keeps the draft model's `past_key_values` across speculative steps when the accepted prefix still matches the cached draft prefix. Partial rejection falls back to rebuilding the draft cache on the next proposal. The benchmark JSON reports `draft_forwards`, `draft_rebuilds`, and `draft_cached_steps` so cache behavior can be checked alongside acceptance.

Experimental engine-level nano-vLLM native draft-model speculative run:

```bash
python bench.py \
  --model ~/huggingface/Qwen3-4B/ \
  --enable-engine-speculative \
  --draft-type nano \
  --draft-model ~/huggingface/Qwen3-0.6B/ \
  --gpu-memory-utilization 0.75 \
  --temperature 0.01 \
  --speculative-greedy-temperature 0.011 \
  --num-speculative-tokens 5 \
  --out outputs/bench-nano-draft-speculative.json
```

The native draft path loads the draft model through nano-vLLM's Qwen3 implementation instead of Transformers `generate()`, with a separate draft KV cache and the same proposal/verify statistics. Low-temperature speculative verification can use an argmax fast path via `--speculative-greedy-temperature`; set it to `0` to force the original sampler path.

Sweep speculative lengths:

```bash
for k in 3 5 7; do
  python bench.py \
    --model ~/huggingface/Qwen3-4B/ \
    --enable-engine-speculative \
    --draft-type nano \
    --draft-model ~/huggingface/Qwen3-0.6B/ \
    --gpu-memory-utilization 0.75 \
    --temperature 0.01 \
    --speculative-greedy-temperature 0.011 \
    --num-speculative-tokens "$k" \
    --out "outputs/bench-nano-draft-k${k}.json"
done
```

Compare the before/after results:

```bash
python scripts/compare_benchmarks.py \
  outputs/bench-baseline.json \
  outputs/bench-hf-draft-speculative.json \
  --out outputs/bench-comparison.md
```

Summarize a k-sweep:

```bash
python scripts/summarize_benchmarks.py \
  outputs/bench-baseline.json \
  outputs/bench-nano-draft-k3.json \
  outputs/bench-nano-draft-k5.json \
  outputs/bench-nano-draft-k7.json \
  --out outputs/bench-nano-draft-sweep.md
```

Run the current target-only speculative verification experiment on a token trace:

```bash
python scripts/target_only_speculative.py \
  --token-trace 1,2,3,1,2,3,4 \
  --prefix-len 5 \
  --max-tokens 2 \
  --ngram-size 2 \
  --num-speculative-tokens 2
```

The comparison report includes throughput, acceptance rate, mean accepted length, draft forward counts, draft rebuild counts, cached draft steps, draft graph replays, target forward counts, greedy verification counts, and per-stage speculative timings (`draft_propose_ms`, `target_verify_ms`, `append_tokens_ms`, `speculative_step_ms`). The DSpark defaults follow DeepSeek DeepSpec's released `Qwen/Qwen3-4B` DSpark checkpoint (`deepseek-ai/dspark_qwen3_4b_block7`) and block-7 drafting setup. Native `ModelRunner` execution of the DSpark drafter still needs a DSpark model adapter. The current engine-level speculative paths support a local n-gram drafter, an experimental Hugging Face draft model, and an experimental nano-vLLM native draft model; all are disabled unless `--enable-engine-speculative` is passed.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-4B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Example Historical Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)

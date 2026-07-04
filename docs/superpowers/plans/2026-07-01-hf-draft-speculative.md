# HF Draft Model Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Hugging Face draft-model speculative decoding path for Qwen3-4B using Qwen3-0.6B as the first draft model.

**Architecture:** The engine keeps baseline decoding unchanged unless `SpeculativeRuntimeConfig.enabled` is true. A pluggable draft generator proposes token IDs, `ModelRunner.run_speculative()` verifies all proposal positions in one target forward, and `LLMEngine._step_speculative()` appends the accepted prefix plus the target token while recording stats.

**Tech Stack:** Python 3.12, PyTorch, Transformers, flash-attn, unittest, nano-vllm scheduler/model runner.

---

### Task 1: Config And Stats

**Files:**
- Modify: `nanovllm/speculative.py`
- Modify: `tests/test_speculative_benchmark.py`

- [ ] Add `draft_model`, `draft_device`, and `draft_dtype` fields to `SpeculativeRuntimeConfig`.
- [ ] Add `draft_forwards` to `SpeculativeStats` and `BenchmarkResult`.
- [ ] Update serialization and markdown comparison output.
- [ ] Add tests for the new config defaults and stats reset behavior.
- [ ] Run `python3 -m unittest tests.test_speculative_benchmark`.

### Task 2: HF Draft Generator

**Files:**
- Modify: `nanovllm/speculative.py`
- Modify: `tests/test_speculative_benchmark.py`

- [ ] Add `HFDraftGenerator` with lazy imports for `torch` and `transformers`.
- [ ] Implement greedy `generate()`-based proposal with `num_speculative_tokens`.
- [ ] Add a fake-model unit test that verifies generated suffix extraction.
- [ ] Run `python3 -m unittest tests.test_speculative_benchmark`.

### Task 3: Target Batched Verification

**Files:**
- Modify: `nanovllm/engine/model_runner.py`
- Modify: `tests/test_engine_speculative_hook.py`

- [ ] Add `ModelRunner.run_speculative(seq, draft_token_ids)` for tensor-parallel size 1.
- [ ] Build input IDs `[seq.last_token] + draft_token_ids`, positions, slot mapping, prefix block tables, and repeated temperatures.
- [ ] Return sampled target tokens for all verification positions.
- [ ] Add source-level tests for the method and its context setup.
- [ ] Run `python3 -m unittest tests.test_engine_speculative_hook`.

### Task 4: Engine Integration

**Files:**
- Modify: `nanovllm/engine/llm_engine.py`
- Modify: `tests/test_engine_speculative_hook.py`

- [ ] Instantiate `HFDraftGenerator` before target model allocation when `draft_type="hf"`.
- [ ] Keep `NGramDraftGenerator` for `draft_type="ngram"`.
- [ ] In `_step_speculative()`, fall back on waiting queue, multi-running sequence, tensor parallel, prefill, or KV block boundary crossing.
- [ ] Call `run_speculative` once, accept matching prefix, append emitted tokens, update `num_cached_tokens`, finish/deallocate if needed, and record stats.
- [ ] Run `python3 -m unittest tests.test_engine_speculative_hook`.

### Task 5: Benchmark And Docs

**Files:**
- Modify: `bench.py`
- Modify: `README.md`
- Modify: `tests/test_engine_speculative_hook.py`

- [ ] Add `--draft-type`, `--draft-device`, `--draft-dtype`, and `--temperature`.
- [ ] Include `draft_forwards` in benchmark JSON.
- [ ] Document Qwen3-0.6B download and HF draft benchmark commands.
- [ ] Run `python3 -m unittest discover -s tests`.
- [ ] On cloud, run a baseline and HF draft comparison with matching parameters.

## Self-Review

- The plan covers config, draft generation, target verification, engine integration, benchmark, docs, and tests.
- No placeholder steps remain.
- Names are consistent with the design: `SpeculativeRuntimeConfig`, `HFDraftGenerator`, `run_speculative`, `draft_forwards`.

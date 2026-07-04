# HF Draft Model Speculative Decoding Design

## Goal

Add an optional real draft-model speculative decoding path for Qwen3-4B in nano-vllm. The first draft model is `Qwen/Qwen3-0.6B`. The feature stays disabled by default and must not change baseline `generate()` behavior.

## Scope

This stage implements a conservative single-sequence decode path:

- Use a Hugging Face causal LM as the draft model.
- Generate `num_speculative_tokens` candidate tokens from the current sequence prefix.
- Verify those candidates with one target-model forward over `[last_token] + draft_tokens`.
- Accept the longest matching prefix and append one target-sampled token on rejection or after full acceptance when room remains.
- Record draft forwards, target forwards, proposed tokens, accepted tokens, acceptance lengths, and throughput.

The implementation falls back to regular decoding when:

- speculative decoding is disabled;
- the scheduler has waiting requests;
- more than one sequence is running;
- the sequence is still in prefill;
- tensor parallel size is greater than 1;
- the speculative verification would cross a KV-cache block boundary.

## Non-Goals

This stage does not implement Medusa, EAGLE, DSpark checkpoint loading, or full speculative sampling with probability correction. Medusa and EAGLE need trained heads or checkpoints. DSpark/DeepSpec integration remains the next checkpoint-specific stage.

## Architecture

`SpeculativeRuntimeConfig` gains `draft_type="hf"` and draft model options. `HFDraftGenerator` wraps `transformers.AutoModelForCausalLM.generate()` and returns draft token IDs. `ModelRunner.run_speculative()` performs one target forward for the verification window and samples target tokens for each verification position. `LLMEngine._step_speculative()` coordinates proposal, verification, sequence append, and stats.

The target-model KV invariant remains the same: after appending emitted tokens, the KV cache contains every token except the final last token, unless the sequence has finished and the cache is deallocated.

## Testing

Unit tests cover:

- runtime config supports HF draft settings;
- stats include draft forwards;
- HF generator can be tested with fake model/token tensors;
- engine source contains the target batched verification path and conservative fallback gates;
- benchmark exposes draft model and temperature controls.

Cloud verification uses Qwen3-4B as target and Qwen3-0.6B as draft, comparing baseline and `--draft-type hf`.

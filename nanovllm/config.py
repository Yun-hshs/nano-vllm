import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    # Per-step token budget. Decode consumes one token per running sequence
    # first; chunked prefill may then use the leftover budget, chunking a prompt
    # when it does not fit. With chunked_prefill=False this is also the legacy
    # prefill chunk size, since prefill and decode never share a step.
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    # Mixed scheduling: run the decode batch through its CUDA graph and then
    # spend the remaining token budget on an eager chunked-prefill pass, in the
    # same step. False restores step-level prefill/decode mutual exclusion.
    chunked_prefill: bool = True
    # Cap on how many tokens of a *single* sequence a prefill pass may take (the
    # chunked-prefill chunk size). The pass still fills the whole per-step token
    # budget by moving on to the next waiting sequence, so a long prompt is
    # chunked without idling the rest of the budget. None means "no per-sequence
    # cap beyond the remaining token budget".
    chunked_prefill_size: int | None = None
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # --- FP8 E4M3 KV cache -------------------------------------------------
    # Quantize the paged KV cache to FP8 E4M3. K uses a static per-(layer,
    # kv_head) scale calibrated once before the cache is allocated; V uses a
    # dynamic per-(token, kv_head) scale computed while the token is written.
    fp8_kvcache: bool = False
    # Number of tokens used for the one-shot static K scale calibration.
    fp8_kv_calib_tokens: int = 512
    # Safety margin on the calibrated static K scale. The calibration sample
    # underestimates the true K amax (random-token activations are lighter than
    # real ones), so scaling the range up by a small factor avoids saturation;
    # it costs a little relative precision (e4m3 has a 3-bit mantissa).
    fp8_kv_k_margin: float = 1.5
    # Optional override for the static K scale (uniform across all heads).
    fp8_kv_k_scale: float | None = None
    # --- Fused Triton operators --------------------------------------------
    # Replace the torch.compile'd Add+RMSNorm and SiluAndMul chains with
    # single-pass Triton kernels (see `nanovllm/layers/fused_ops.py`).
    triton_fusion: bool = True

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.fp8_kv_calib_tokens >= 0
        assert self.fp8_kv_k_margin > 0
        assert self.fp8_kv_k_scale is None or self.fp8_kv_k_scale > 0
        assert self.chunked_prefill_size is None or self.chunked_prefill_size > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

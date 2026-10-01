from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # FP8 prefill only: flat physical-slot index for every cached token of every
    # scheduled sequence, laid out in the same order as `cu_seqlens_k`. Used by
    # the gather + dequant prefill path so the paged FP8 cache can be fed to
    # FlashAttention without materialising a full-precision copy of the cache.
    prefill_gather_slots: torch.Tensor | None = None
    # FP8 prefill only: [(q_start, q_end, kv_start, kv_end, cu_seqlens_q_local,
    # cu_seqlens_k_local), ...] splitting the batch so each gather buffer stays
    # within the configured token budget.
    prefill_kv_groups: list | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, prefill_gather_slots=None, prefill_kv_groups=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, prefill_gather_slots, prefill_kv_groups)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

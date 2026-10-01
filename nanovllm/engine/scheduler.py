from collections import deque
from dataclasses import dataclass, field

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(slots=True)
class Schedule:
    """One scheduling step.

    `decode` and `prefill` are disjoint batches that a single step executes back
    to back: decode first (CUDA graph), then the prefill chunk (eager) with
    whatever token budget is left. The splitting is what keeps a long prompt
    from blocking the online decode stream: decode is scheduled first and always
    gets its token, prefill only ever consumes the leftover budget.
    """

    decode: list[Sequence] = field(default_factory=list)
    prefill: list[Sequence] = field(default_factory=list)

    @property
    def seqs(self) -> list[Sequence]:
        """Decode batch followed by the prefill chunk, in execution order."""
        return self.decode + self.prefill

    @property
    def is_empty(self) -> bool:
        return not self.decode and not self.prefill

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decode)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(seq.num_scheduled_tokens for seq in self.prefill)

    @property
    def num_batched_tokens(self) -> int:
        return self.num_decode_tokens + self.num_prefill_tokens


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.chunked_prefill = config.chunked_prefill
        self.chunked_prefill_size = config.chunked_prefill_size
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> Schedule:
        if self.chunked_prefill:
            return self._schedule_chunked_prefill()
        return self._schedule_exclusive()

    def _schedule_chunked_prefill(self) -> Schedule:
        """Decode strictly first, then chunked prefill on the leftover budget.

        Every running sequence decodes exactly one token, so the decode cost of
        a step is `len(running)` tokens no matter how long the waiting queue is.
        Prefill then gets `max_num_batched_tokens - num_decode_tokens` and may
        chunk any sequence to fit -- a single long prompt no longer forces the
        engine to choose between prefill and decode.
        """
        schedule = Schedule()
        num_batched_tokens = 0

        # --- 1. Decode: strict priority, one token per running sequence ------
        while self.running and len(schedule.decode) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                schedule.decode.append(seq)
                num_batched_tokens += 1
        self.running.extendleft(reversed(schedule.decode))

        # --- 2. Prefill: spend whatever the decode batch did not use ---------
        remaining = max(0, self.max_num_batched_tokens - num_batched_tokens)
        # `chunked_prefill_size` caps how much of one sequence a step may take;
        # the pass still fills the whole budget by moving on to the next waiter,
        # so chunking a long prompt does not idle the rest of the budget the way
        # the legacy "chunk only the first sequence" rule did.
        per_seq_cap = self.chunked_prefill_size if self.chunked_prefill_size is not None else remaining
        for _ in range(len(self.waiting)):
            if remaining <= 0 or len(schedule.decode) + len(schedule.prefill) >= self.max_num_seqs:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(num_tokens, remaining, per_seq_cap)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            remaining -= seq.num_scheduled_tokens
            schedule.prefill.append(seq)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                # Partial chunk: rotate it behind the other waiters. It is still
                # visited at most once per step, so it cannot appear twice in the
                # batch; its FIFO progress resumes on the next step.
                self.waiting.rotate(-1)

        assert not schedule.is_empty, "scheduler produced an empty step"
        return schedule

    def _schedule_exclusive(self) -> Schedule:
        """Legacy step-level prefill/decode mutual exclusion (`chunked_prefill=False`)."""
        schedule = Schedule()
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(schedule.prefill) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and schedule.prefill:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            schedule.prefill.append(seq)

        if schedule.prefill:
            return schedule

        # decode
        while self.running and len(schedule.decode) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                schedule.decode.append(seq)
        assert not schedule.is_empty, "scheduler produced an empty step"
        self.running.extendleft(reversed(schedule.decode))
        return schedule

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, schedule: Schedule, decode_token_ids: list[int], prefill_token_ids: list[int]):
        for seq, token_id in zip(schedule.decode, decode_token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.append_token(token_id)
            self._maybe_finish(seq)
        for seq, token_id in zip(schedule.prefill, prefill_token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                continue    # chunked prefill: nothing sampled yet
            seq.append_token(token_id)
            self._maybe_finish(seq)

    def _maybe_finish(self, seq: Sequence):
        if (not seq.ignore_eos and seq.last_token == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)

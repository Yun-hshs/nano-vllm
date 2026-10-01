"""Scheduler unit tests for mixed decode-first / chunked-prefill scheduling.

The scheduler is host-side logic, so these tests run on CPU: `nanovllm` imports
`LLM` lazily, and the scheduler/block-manager/sequence path only needs
transformers (for `Config`), numpy and xxhash.

Run: pytest tests/test_chunked_prefill.py -v
"""

import itertools

from nanovllm.engine.scheduler import Scheduler, Schedule
from nanovllm.engine.sequence import Sequence, SequenceStatus

BLOCK = 4
_PROMPT_IDS = itertools.count(1)


class DummyConfig:
    """Only the attributes the scheduler reads (keeps Config's model asserts out)."""

    def __init__(self, **kw):
        self.max_num_seqs = kw.pop("max_num_seqs", 16)
        self.max_num_batched_tokens = kw.pop("max_num_batched_tokens", 1024)
        self.chunked_prefill = kw.pop("chunked_prefill", True)
        self.chunked_prefill_size = kw.pop("chunked_prefill_size", None)
        self.eos = kw.pop("eos", -1)
        self.kvcache_block_size = kw.pop("kvcache_block_size", BLOCK)
        self.num_kvcache_blocks = kw.pop("num_kvcache_blocks", 256)
        assert not kw, f"unexpected config overrides: {kw}"


def make_scheduler(**kw):
    Sequence.block_size = kw.get("kvcache_block_size", BLOCK)
    return Scheduler(DummyConfig(**kw))


def prompt(n):
    """A unique prompt per call, so tests never hit the prefix cache by accident."""
    base = next(_PROMPT_IDS) * 10000
    return [base + i for i in range(n)]


def drive(sch, decode_token=7, prefill_token=8):
    """Run one full scheduling step, postprocessing with dummy sampled tokens."""
    schedule = sch.schedule()
    sch.postprocess(schedule, [decode_token] * len(schedule.decode), [prefill_token] * len(schedule.prefill))
    return schedule


def run_prefill(sch, seq):
    """Drive a sequence from WAITING to RUNNING through as many steps as needed."""
    while seq.status == SequenceStatus.WAITING:
        schedule = drive(sch)
        assert seq in schedule.prefill
    return seq


# ---------------------------------------------------------------------------
# Mixed step composition
# ---------------------------------------------------------------------------

def test_mixed_step_schedules_decode_before_prefill():
    sch = make_scheduler(max_num_batched_tokens=64)
    a = Sequence(prompt(10))
    sch.add(a)
    run_prefill(sch, a)
    assert a.status == SequenceStatus.RUNNING and a.num_tokens == 11

    b = Sequence(prompt(20))
    sch.add(b)
    schedule = sch.schedule()

    assert [s.seq_id for s in schedule.decode] == [a.seq_id]
    assert [s.seq_id for s in schedule.prefill] == [b.seq_id]
    assert schedule.seqs[0] is a, "decode must execute first inside the step"
    assert schedule.num_decode_tokens == 1
    assert schedule.num_prefill_tokens == 20
    assert not schedule.is_empty


def test_prefill_receives_only_the_leftover_token_budget():
    sch = make_scheduler(max_num_batched_tokens=16)
    running = []
    for _ in range(4):
        seq = Sequence(prompt(4))
        sch.add(seq)
        running.append(seq)
    schedule = drive(sch)
    assert len(schedule.prefill) == 4
    assert all(s.status == SequenceStatus.RUNNING for s in running)

    long = Sequence(prompt(100))
    sch.add(long)
    schedule = sch.schedule()
    assert len(schedule.decode) == 4
    assert schedule.num_prefill_tokens == 12          # 16 - 4 decoded tokens
    assert long.num_scheduled_tokens == 12
    assert schedule.num_batched_tokens == 16


def test_decode_is_never_starved_by_a_waiting_prefill():
    sch = make_scheduler(max_num_batched_tokens=8)
    running = []
    for _ in range(8):
        seq = Sequence(prompt(1))
        sch.add(seq)
        running.append(seq)
    schedule = drive(sch)
    assert len(schedule.prefill) == 8
    assert all(s.status == SequenceStatus.RUNNING for s in running)

    waiter = Sequence(prompt(100))
    sch.add(waiter)
    schedule = sch.schedule()
    assert len(schedule.decode) == 8        # the whole budget goes to decode
    assert schedule.num_prefill_tokens == 0
    assert waiter.status == SequenceStatus.WAITING


# ---------------------------------------------------------------------------
# Chunked prefill
# ---------------------------------------------------------------------------

def test_long_prompt_is_chunked_across_steps():
    sch = make_scheduler(max_num_batched_tokens=16)
    seq = Sequence(prompt(100))
    sch.add(seq)

    scheduled = 0
    while seq.status == SequenceStatus.WAITING:
        schedule = sch.schedule()
        assert schedule.decode == []
        assert len(schedule.prefill) == 1
        assert schedule.num_prefill_tokens == min(16, 100 - scheduled)
        scheduled += schedule.num_prefill_tokens
        assert seq.num_tokens == 100, "a partial chunk must not emit a token"
        sch.postprocess(schedule, [], [7])

    assert scheduled == 100
    assert seq.num_cached_tokens == 100
    assert seq.num_tokens == 101, "the completed chunk emits exactly one token"


def test_partial_chunk_rotates_behind_the_other_waiters():
    """Chunking one sequence must not block the rest of the waiting queue."""
    sch = make_scheduler(max_num_batched_tokens=16)
    long = Sequence(prompt(100))
    short = Sequence(prompt(4))
    sch.add(long)
    sch.add(short)

    first = sch.schedule()
    assert first.prefill == [long]
    assert long.num_scheduled_tokens == 16
    sch.postprocess(first, [], [7])
    assert list(sch.waiting)[0] is short, "the partially prefilled sequence rotates to the back"

    second = sch.schedule()
    assert second.prefill[0] is short
    assert long in second.prefill, "the leftover budget continues the long prompt"


def test_chunked_prefill_size_caps_the_prefill_pass():
    sch = make_scheduler(max_num_batched_tokens=1024, chunked_prefill_size=8)
    seq = Sequence(prompt(20))
    sch.add(seq)
    schedule = sch.schedule()
    assert schedule.decode == []
    assert schedule.prefill == [seq], "a sequence may be scheduled at most once per step"
    assert schedule.num_prefill_tokens == 8
    assert seq.num_scheduled_tokens == 8


def test_full_prefix_cache_completes_without_chunking():
    """A prompt whose blocks are already cached still prefills a positive chunk."""
    shared = [5] * 8
    sch = make_scheduler(max_num_batched_tokens=64)
    first = Sequence(shared)
    sch.add(first)
    run_prefill(sch, first)                     # hashes the full blocks

    second = Sequence(shared)
    sch.add(second)
    schedule = sch.schedule()
    assert second in schedule.prefill
    assert second.num_cached_tokens > 0         # prefix hit reused
    assert second.num_scheduled_tokens >= 1


# ---------------------------------------------------------------------------
# Legacy exclusive scheduling
# ---------------------------------------------------------------------------

def test_exclusive_scheduler_never_mixes_batches():
    sch = make_scheduler(chunked_prefill=False, max_num_batched_tokens=16)
    a = Sequence(prompt(4))
    sch.add(a)
    run_prefill(sch, a)

    b = Sequence(prompt(4))
    sch.add(b)
    schedule = sch.schedule()
    assert schedule.prefill and not schedule.decode, "legacy prefill keeps priority"

    schedule = drive(sch)
    schedule = sch.schedule()
    assert schedule.decode and not schedule.prefill


# ---------------------------------------------------------------------------
# Postprocess bookkeeping
# ---------------------------------------------------------------------------

def test_postprocess_advances_cache_and_emits_decode_token():
    sch = make_scheduler(max_num_batched_tokens=64)
    seq = Sequence(prompt(4))
    sch.add(seq)
    run_prefill(sch, seq)
    assert seq.num_cached_tokens == 4 and seq.num_tokens == 5

    schedule = sch.schedule()
    assert schedule.decode == [seq]
    sch.postprocess(schedule, [123], [])
    assert seq.last_token == 123
    assert seq.num_tokens == 6
    assert seq.num_cached_tokens == 5
    assert seq.num_scheduled_tokens == 0


def test_postprocess_finishes_on_eos_and_frees_blocks():
    sch = make_scheduler(max_num_batched_tokens=64, eos=99)
    seq = Sequence(prompt(4))
    sch.add(seq)
    run_prefill(sch, seq)

    schedule = sch.schedule()
    blocks = len(seq.block_table)                   # includes the block may_append just took
    free_after_schedule = len(sch.block_manager.free_block_ids)
    sch.postprocess(schedule, [99], [])
    assert seq.is_finished
    assert seq not in sch.running
    assert len(sch.block_manager.free_block_ids) == free_after_schedule + blocks


def test_preempt_requeues_at_the_front():
    sch = make_scheduler(num_kvcache_blocks=2)
    seq = Sequence(prompt(4))
    sch.add(seq)
    run_prefill(sch, seq)

    sch.preempt(seq)
    assert seq.status == SequenceStatus.WAITING
    assert seq.is_prefill
    assert seq.block_table == []
    assert sch.waiting[0] is seq
    assert sch.num_preemptions == 1


def test_schedule_batches_are_disjoint_and_ordered():
    sch = make_scheduler(max_num_batched_tokens=64)
    for _ in range(3):
        sch.add(Sequence(prompt(2)))
    drive(sch)
    sch.add(Sequence(prompt(30)))
    schedule = sch.schedule()

    ids = [s.seq_id for s in schedule.seqs]
    assert len(ids) == len(set(ids)), "a sequence may appear in at most one batch"
    assert schedule.seqs == schedule.decode + schedule.prefill
    assert isinstance(schedule, Schedule)

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiln.cache import BlockManager
from kiln.engine.scheduler import Scheduler
from kiln.engine.sequence import SeqStatus, Sequence
from kiln.sampling import SamplingParams


def seq(i, prompt_len, max_tokens=8):
    return Sequence(i, f"r{i}", list(range(prompt_len)), SamplingParams(max_tokens=max_tokens), None)


def run_step(sched, out, sampled_token=7):
    """Pretend the model ran: mark tokens computed and append a token to sequences that sampled.
    Returns the sequences that sampled this step."""
    sampled = []
    for s in out.scheduled:
        s.seq.num_computed += s.num_tokens
        if s.seq.num_computed == len(s.seq.tokens):
            s.seq.output_ids.append(sampled_token)
            sampled.append(s.seq)
    return sampled


def test_decode_before_prefill_and_budget():
    sched = Scheduler(BlockManager(64, 4), max_num_seqs=8, max_batched_tokens=10)
    a = seq(1, 6)
    sched.add(a)
    run_step(sched, sched.schedule())  # a prefills 6
    b = seq(2, 20)
    sched.add(b)
    out = sched.schedule()
    assert [(s.seq.seq_id, s.num_tokens) for s in out.scheduled] == [(1, 1), (2, 9)]  # b chunked to budget


def test_chunked_prefill_continues_next_step():
    sched = Scheduler(BlockManager(64, 4), max_num_seqs=8, max_batched_tokens=8)
    a = seq(1, 20)
    sched.add(a)
    chunks = []
    for _ in range(3):
        out = sched.schedule()
        chunks.append([(s.start, s.num_tokens) for s in out.scheduled])
        run_step(sched, out)
    assert chunks == [[(0, 8)], [(8, 8)], [(16, 4)]]
    assert a.output_ids == [7]  # sampled only once the whole prompt was computed


def test_max_num_seqs_and_fifo_no_skipping():
    sched = Scheduler(BlockManager(64, 4), max_num_seqs=2, max_batched_tokens=100)
    for i in range(1, 4):
        sched.add(seq(i, 4))
    out = sched.schedule()
    assert [s.seq.seq_id for s in out.scheduled] == [1, 2]

    sched2 = Scheduler(BlockManager(4, 4), max_num_seqs=8, max_batched_tokens=100)
    big, small = seq(1, 12, max_tokens=4), seq(2, 2, max_tokens=4)
    sched2.add(seq(0, 8, max_tokens=4))
    sched2.add(big)
    sched2.add(small)
    out = sched2.schedule()  # seq 0 takes 2 blocks; big needs 3 of the remaining 2 -> wait; small must NOT jump
    assert [s.seq.seq_id for s in out.scheduled] == [0]


def test_preempts_newest_running_and_requeues_at_front():
    bm = BlockManager(4, 4)  # 16 slots total
    sched = Scheduler(bm, max_num_seqs=8, max_batched_tokens=100)
    a, b = seq(1, 8, max_tokens=8), seq(2, 8, max_tokens=8)
    sched.add(a)
    sched.add(b)
    run_step(sched, sched.schedule())  # both prefill: pool full (4 blocks)
    out = sched.schedule()  # a needs a new block for its decode -> preempt b (newest)
    assert [s.seq.seq_id for s in out.scheduled] == [1]
    assert out.preempted == [b]
    assert b.status is SeqStatus.WAITING and b.num_computed == 0 and b.num_preemptions == 1
    assert sched.waiting[0] is b
    assert 2 not in bm.seq_ids


def test_abort_frees_blocks():
    bm = BlockManager(16, 4)
    sched = Scheduler(bm, max_num_seqs=8, max_batched_tokens=100)
    a, b = seq(1, 6), seq(2, 6)
    sched.add(a)
    run_step(sched, sched.schedule())
    sched.add(b)
    assert sched.abort(1) is a and sched.abort(2) is b
    assert sched.abort(1) is None  # already gone: no-op
    assert bm.num_free_blocks == 16 and not sched.has_unfinished()


def test_finish_frees_blocks():
    bm = BlockManager(16, 4)
    sched = Scheduler(bm, max_num_seqs=8, max_batched_tokens=100)
    a = seq(1, 6)
    sched.add(a)
    run_step(sched, sched.schedule())
    sched.finish(a, "length")
    assert a.status is SeqStatus.FINISHED and bm.num_free_blocks == 16 and not sched.has_unfinished()


@settings(max_examples=150, deadline=None)
@given(
    specs=st.lists(st.tuples(st.integers(1, 30), st.integers(1, 12), st.integers(0, 5)), min_size=1, max_size=12),
    num_blocks=st.integers(8, 24),
    budget=st.integers(4, 40),
    max_seqs=st.integers(1, 6),
)
def test_random_workloads_finish_without_leaks(specs, num_blocks, budget, max_seqs):
    bs = 4
    bm = BlockManager(num_blocks, bs)
    sched = Scheduler(bm, max_num_seqs=max_seqs, max_batched_tokens=budget)
    pending = [(arrive, seq(i, plen, mt)) for i, (plen, mt, arrive) in enumerate(specs)
               if plen + mt <= num_blocks * bs]  # longer ones are rejected by the engine before this
    pending.sort(key=lambda p: p[0])
    step = 0
    while pending or sched.has_unfinished():
        while pending and pending[0][0] <= step:
            sched.add(pending.pop(0)[1])
        out = sched.schedule()
        assert sum(s.num_tokens for s in out.scheduled) <= budget
        assert len(sched.running) <= max_seqs
        for sq in run_step(sched, out):
            if len(sq.output_ids) >= sq.params.max_tokens:
                sched.finish(sq, "length")
        step += 1
        assert step < 5000, "scheduler made no progress"
    assert bm.num_free_blocks == num_blocks


def test_sequence_too_long_for_pool_raises_on_add():
    sched = Scheduler(BlockManager(2, 4), max_num_seqs=8, max_batched_tokens=100)
    with pytest.raises(ValueError):
        sched.add(seq(1, 6, max_tokens=4))  # 10 tokens > 8 slots

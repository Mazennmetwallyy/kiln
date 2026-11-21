import math

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from kiln.cache.block_manager import BlockManager, NoFreeBlocksError


def test_allocate_rounds_up_blocks():
    bm = BlockManager(num_blocks=8, block_size=16)
    bm.allocate(1, 17)
    assert len(bm.block_table(1)) == 2
    assert bm.num_free_blocks == 6
    assert bm.seq_len(1) == 17


def test_exact_multiple_then_append_allocates():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate(1, 8)
    assert len(bm.block_table(1)) == 2
    bm.append_slots(1, 1)
    assert len(bm.block_table(1)) == 3
    assert bm.slots(1, 8, 1) == [bm.block_table(1)[2] * 4]


def test_out_of_blocks_is_atomic():
    bm = BlockManager(num_blocks=3, block_size=4)
    bm.allocate(1, 8)
    with pytest.raises(NoFreeBlocksError):
        bm.allocate(2, 8)
    assert bm.num_free_blocks == 1 and 2 not in bm.seq_ids
    with pytest.raises(NoFreeBlocksError):
        bm.append_slots(1, 9)
    assert bm.seq_len(1) == 8 and bm.num_free_blocks == 1
    assert not bm.can_allocate(5) and bm.can_allocate(4)
    assert bm.can_append(1, 4) and not bm.can_append(1, 5)


def test_double_free_raises():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate(1, 3)
    bm.free(1)
    with pytest.raises(KeyError):
        bm.free(1)
    with pytest.raises(KeyError):
        bm.free(99)
    assert bm.num_free_blocks == 4


def test_duplicate_seq_id_rejected():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate(1, 3)
    with pytest.raises(ValueError):
        bm.allocate(1, 3)


def test_fork_shares_and_cow_on_partial_block():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate(1, 6)  # blocks: [a (full), b (2 of 4 used)]
    bm.fork(1, 2)
    a, b = bm.block_table(1)
    assert bm.block_table(2) == [a, b] and bm.refcount(a) == 2 and bm.refcount(b) == 2

    copies = bm.append_slots(2, 1)  # child writes into the shared partial block -> copy it
    new_b = bm.block_table(2)[1]
    assert copies == [(b, new_b)] and new_b != b
    assert bm.refcount(b) == 1 and bm.refcount(new_b) == 1 and bm.refcount(a) == 2
    assert bm.append_slots(1, 1) == []  # parent now owns b alone: no copy


def test_rollback_across_block_boundary_frees():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate(1, 9)  # 3 blocks
    free_before = bm.num_free_blocks
    bm.rollback(1, 5)  # length 4 -> 1 block
    assert bm.seq_len(1) == 4 and len(bm.block_table(1)) == 1
    assert bm.num_free_blocks == free_before + 2
    with pytest.raises(ValueError):
        bm.rollback(1, 5)


def test_slots_are_block_offsets():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate(1, 6)
    a, b = bm.block_table(1)
    assert bm.slots(1, 0, 6) == [a * 4, a * 4 + 1, a * 4 + 2, a * 4 + 3, b * 4, b * 4 + 1]
    with pytest.raises(ValueError):
        bm.slots(1, 4, 3)  # past the sequence's length


class BlockManagerMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.bm = BlockManager(num_blocks=12, block_size=4)
        self.next_id = 0

    def _new_id(self):
        self.next_id += 1
        return self.next_id

    @rule(n=st.integers(1, 20))
    def allocate(self, n):
        if self.bm.can_allocate(n):
            self.bm.allocate(self._new_id(), n)
        else:
            with pytest.raises(NoFreeBlocksError):
                self.bm.allocate(self._new_id(), n)

    @precondition(lambda self: self.bm.seq_ids)
    @rule(data=st.data(), n=st.integers(1, 9))
    def append(self, data, n):
        sid = data.draw(st.sampled_from(sorted(self.bm.seq_ids)))
        before = self.bm.seq_len(sid)
        if self.bm.can_append(sid, n):
            self.bm.append_slots(sid, n)
            assert self.bm.seq_len(sid) == before + n
        else:
            with pytest.raises(NoFreeBlocksError):
                self.bm.append_slots(sid, n)
            assert self.bm.seq_len(sid) == before

    @precondition(lambda self: self.bm.seq_ids)
    @rule(data=st.data())
    def fork(self, data):
        self.bm.fork(data.draw(st.sampled_from(sorted(self.bm.seq_ids))), self._new_id())

    @precondition(lambda self: self.bm.seq_ids)
    @rule(data=st.data())
    def rollback(self, data):
        sid = data.draw(st.sampled_from(sorted(self.bm.seq_ids)))
        n = data.draw(st.integers(0, self.bm.seq_len(sid) - 1))
        self.bm.rollback(sid, n)

    @precondition(lambda self: self.bm.seq_ids)
    @rule(data=st.data())
    def free(self, data):
        self.bm.free(data.draw(st.sampled_from(sorted(self.bm.seq_ids))))

    @invariant()
    def refcounts_match_tables(self):
        counts = [0] * self.bm.num_blocks
        for sid in self.bm.seq_ids:
            table = self.bm.block_table(sid)
            assert len(table) == math.ceil(self.bm.seq_len(sid) / self.bm.block_size)
            for b in table:
                counts[b] += 1
        for b in range(self.bm.num_blocks):
            assert self.bm.refcount(b) == counts[b]
        assert self.bm.num_free_blocks == counts.count(0)

    def teardown(self):
        for sid in list(self.bm.seq_ids):
            self.bm.free(sid)
        assert self.bm.num_free_blocks == self.bm.num_blocks


TestBlockManagerMachine = BlockManagerMachine.TestCase
TestBlockManagerMachine.settings = settings(max_examples=300, stateful_step_count=40, deadline=None)

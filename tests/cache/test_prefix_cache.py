import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from kiln.cache.block_manager import BlockManager

BS = 4


def computed(bm, sid, tokens):
    """Simulate the engine: allocate, 'compute' every token, and register the full blocks."""
    cached = bm.allocate(sid, len(tokens), tokens)
    bm.register_computed(sid, tokens, len(tokens))
    return cached


def test_shared_prefix_is_reused():
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    system = list(range(100, 108))  # two full blocks
    assert computed(bm, 1, [*system, 1, 2, 3]) == 0
    a_blocks = bm.block_table(1)
    assert bm.allocate(2, 11, [*system, 7, 8, 9]) == 8
    assert bm.block_table(2)[:2] == a_blocks[:2] and bm.block_table(2)[2] != a_blocks[2]
    assert bm.refcount(a_blocks[0]) == 2


def test_partial_block_is_not_cached():
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    computed(bm, 1, [1, 2, 3, 4, 5, 6])  # one full block + 2 tokens
    assert bm.cached_prefix_len([1, 2, 3, 4, 5, 6, 7]) == 4


def test_leaves_at_least_one_token_to_compute():
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    computed(bm, 1, list(range(8)))
    assert bm.cached_prefix_len(list(range(8))) == 4  # not 8: the last token's logits are needed
    assert bm.cached_prefix_len(list(range(9))) == 8


def test_prefix_must_match_from_the_start():
    bm = BlockManager(16, BS, enable_prefix_caching=True)
    computed(bm, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert bm.cached_prefix_len([0, 2, 3, 4, 5, 6, 7, 8, 9]) == 0  # first block differs
    assert bm.cached_prefix_len([1, 2, 3, 4, 0, 6, 7, 8, 9]) == 4  # second differs: chain stops


def test_freed_blocks_stay_cached_until_evicted_lru():
    bm = BlockManager(4, BS, enable_prefix_caching=True)
    computed(bm, 1, [1, 1, 1, 1, 9])  # block hashed for [1,1,1,1]
    computed(bm, 2, [2, 2, 2, 2, 9])
    bm.free(1)
    bm.free(2)
    assert bm.num_free_blocks == 4  # evictable blocks count as available
    assert bm.cached_prefix_len([1, 1, 1, 1, 5]) == 4 and bm.cached_prefix_len([2, 2, 2, 2, 5]) == 4
    bm.allocate(3, 12, [7] * 12)  # needs 3 blocks: 2 free + evict the LRU cached block ([1,1,1,1])
    assert bm.cached_prefix_len([1, 1, 1, 1, 5]) == 0
    assert bm.cached_prefix_len([2, 2, 2, 2, 5]) == 4


def test_hit_on_evictable_block_takes_it_back():
    bm = BlockManager(4, BS, enable_prefix_caching=True)
    computed(bm, 1, [1, 1, 1, 1, 9])
    bm.free(1)
    assert bm.allocate(2, 5, [1, 1, 1, 1, 3]) == 4
    assert bm.num_free_blocks == 2
    bm.allocate(3, 8, [5] * 8)  # must not evict seq 2's block
    assert bm.block_table(2)[0] not in bm.block_table(3)


def test_disabled_by_default():
    bm = BlockManager(16, BS)
    computed(bm, 1, list(range(9)))
    assert bm.cached_prefix_len(list(range(9))) == 0


def test_can_allocate_counts_cached_blocks():
    bm = BlockManager(3, BS, enable_prefix_caching=True)
    computed(bm, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9])  # all 3 blocks in use
    assert not bm.can_allocate(9) and bm.can_allocate(9, [1, 2, 3, 4, 5, 6, 7, 8, 0]) is False
    bm.free(1)
    assert bm.can_allocate(12, [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0])


class PrefixMachine(RuleBasedStateMachine):
    """Random workloads over a 3-token alphabet, so prefixes collide constantly."""

    def __init__(self):
        super().__init__()
        self.bm = BlockManager(10, BS, enable_prefix_caching=True)
        self.tokens = {}
        self.next_id = 0

    @rule(toks=st.lists(st.integers(0, 2), min_size=1, max_size=14))
    def admit(self, toks):
        if self.bm.can_allocate(len(toks), toks):
            self.next_id += 1
            cached = self.bm.allocate(self.next_id, len(toks), toks)
            assert cached % BS == 0 and cached < len(toks)
            self.bm.register_computed(self.next_id, toks, len(toks))
            self.tokens[self.next_id] = list(toks)

    @precondition(lambda self: self.tokens)
    @rule(data=st.data(), extra=st.lists(st.integers(0, 2), min_size=1, max_size=6))
    def grow(self, data, extra):
        sid = data.draw(st.sampled_from(sorted(self.tokens)))
        if self.bm.can_append(sid, len(extra)):
            self.bm.append_slots(sid, len(extra))
            self.tokens[sid] += extra
            self.bm.register_computed(sid, self.tokens[sid], len(self.tokens[sid]))

    @precondition(lambda self: self.tokens)
    @rule(data=st.data())
    def free(self, data):
        sid = data.draw(st.sampled_from(sorted(self.tokens)))
        self.bm.free(sid)
        del self.tokens[sid]

    @invariant()
    def consistent(self):
        self.bm.check_invariants()
        for sid, toks in self.tokens.items():
            assert self.bm.seq_len(sid) == len(toks)
            for i, b in enumerate(self.bm.block_table(sid)):
                cached = self.bm.block_tokens(b)
                if cached is not None:  # a hashed block must hold exactly this sequence's tokens
                    assert cached == tuple(toks[i * BS:(i + 1) * BS])


TestPrefixMachine = PrefixMachine.TestCase
TestPrefixMachine.settings = settings(max_examples=300, stateful_step_count=40, deadline=None)


def test_free_unknown_still_raises():
    with pytest.raises(KeyError):
        BlockManager(4, BS, enable_prefix_caching=True).free(3)


def test_rollback_into_a_cached_block_uncaches_it():
    # The rolled-back block will be overwritten in place, so it must stop answering lookups for its old tokens.
    bm = BlockManager(8, 4, enable_prefix_caching=True)
    toks = [1, 2, 3, 4, 5]
    bm.allocate(1, 5, toks)
    bm.register_computed(1, toks, 5)
    assert bm.cached_prefix_len([1, 2, 3, 4, 9]) == 4
    bm.rollback(1, 2)  # length 3: block 0 is partial again and owned by this sequence alone
    assert bm.cached_prefix_len([1, 2, 3, 4, 9]) == 0
    bm.check_invariants()


def test_rollback_keeps_a_shared_cached_block():
    # Another sequence still reads the block, and a write would copy it first, so the cache entry stays valid.
    bm = BlockManager(8, 4, enable_prefix_caching=True)
    toks = [1, 2, 3, 4, 5]
    bm.allocate(1, 5, toks)
    bm.register_computed(1, toks, 5)
    bm.allocate(2, 5, [1, 2, 3, 4, 6])
    bm.rollback(1, 2)
    assert bm.cached_prefix_len([1, 2, 3, 4, 9]) == 4
    assert bm.append_slots(1, 1) == [(bm.block_table(2)[0], bm.block_table(1)[0])]  # copy-on-write
    bm.check_invariants()


def test_block_keys_are_sha256_of_the_whole_prefix():
    # A non-cryptographic hash lets a crafted prompt collide with another's parent key and then inherit
    # its children, serving K/V computed from a different context. Keys must be collision resistant.
    import hashlib
    from array import array

    bm = BlockManager(8, 4, enable_prefix_caching=True)
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bm.allocate(1, 9, toks)
    bm.register_computed(1, toks, 9)
    k0 = hashlib.sha256(b"" + array("q", [1, 2, 3, 4]).tobytes()).digest()
    k1 = hashlib.sha256(k0 + array("q", [5, 6, 7, 8]).tobytes()).digest()
    table = bm.block_table(1)
    assert bm.block_key(table[0]) == k0 and bm.block_key(table[1]) == k1


def test_same_block_tokens_under_different_parents_get_different_keys():
    bm = BlockManager(8, 4, enable_prefix_caching=True)
    a, b = [1, 1, 1, 1, 5, 6, 7, 8, 0], [2, 2, 2, 2, 5, 6, 7, 8, 0]
    computed(bm, 1, a)
    computed(bm, 2, b)
    assert bm.block_key(bm.block_table(1)[1]) != bm.block_key(bm.block_table(2)[1])
    assert bm.cached_prefix_len([2, 2, 2, 2, 5, 6, 7, 8, 3]) == 8

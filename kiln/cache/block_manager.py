"""Bookkeeping for the paged KV cache: which fixed-size blocks belong to which sequence.

This is the operating-system idea of paging applied to attention. A sequence's KV cache is a list of
blocks (its *block table*), which need not be contiguous in memory. Blocks are reference counted, so
sequences can share them (forks, and shared prompt prefixes). Writing into a shared, partially
filled block triggers copy-on-write. The manager only does bookkeeping. It never touches tensors.
Callers apply the copy pairs it returns.

Prefix caching (optional): once a *full* block's K/V has been computed it gets a content key. The key is
SHA-256 of (parent block's key, the block's 16 token ids), so it identifies the whole prefix up to that
block. A cryptographic hash matters here: with a weak one, a crafted prompt could collide with another
prompt's parent key and inherit its child blocks, whose K/V was computed from a different context.
A new request walks its prompt block by block and reuses every block already cached. A block
whose last user finishes isn't wiped. It becomes *evictable*: still cached, still available, and
reclaimed least-recently-used first when fresh blocks run out.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
from collections.abc import Sequence


class NoFreeBlocksError(RuntimeError):
    """Not enough free blocks; nothing was changed."""


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _block_key(parent: bytes | None, tokens: tuple[int, ...]) -> bytes:
    return hashlib.sha256((parent or b"") + array("q", tokens).tobytes()).digest()


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = False):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # pop() hands out low ids first
        self._evictable: OrderedDict[int, None] = OrderedDict()  # refcount 0 but still cached; LRU first
        self._ref = [0] * num_blocks
        self._tables: dict[int, list[int]] = {}
        self._lens: dict[int, int] = {}
        # prefix cache
        self._key_of: dict[int, bytes] = {}  # block -> content key
        self._block_of: dict[bytes, int] = {}  # content key -> block
        self._parent_key_of: dict[int, bytes | None] = {}
        self._tokens_of: dict[int, tuple[int, ...]] = {}
        self._seq_keys: dict[int, list[bytes]] = {}  # keys of each sequence's leading registered blocks

    # ---- queries -------------------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        """Blocks available to allocate: truly free plus cached-but-unused (evictable)."""
        return len(self._free) + len(self._evictable)

    @property
    def num_cached_blocks(self) -> int:
        return len(self._key_of)

    @property
    def seq_ids(self) -> set[int]:
        return set(self._tables)

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._tables[seq_id])

    def seq_len(self, seq_id: int) -> int:
        return self._lens[seq_id]

    def refcount(self, block_id: int) -> int:
        return self._ref[block_id]

    def block_tokens(self, block_id: int) -> tuple[int, ...] | None:
        """The token ids a cached block holds (None if the block isn't in the prefix cache)."""
        return self._tokens_of.get(block_id)

    def block_key(self, block_id: int) -> bytes | None:
        return self._key_of.get(block_id)

    def is_evictable(self, block_id: int) -> bool:
        return block_id in self._evictable

    def slots(self, seq_id: int, start: int, count: int) -> list[int]:
        """Flat slot ids (block * block_size + offset) for token positions [start, start + count)."""
        if start < 0 or start + count > self._lens[seq_id]:
            raise ValueError(f"positions [{start}, {start + count}) outside sequence of length "
                             f"{self._lens[seq_id]}")
        table, bs = self._tables[seq_id], self.block_size
        return [table[p // bs] * bs + p % bs for p in range(start, start + count)]

    def cached_prefix_len(self, token_ids: Sequence[int]) -> int:
        return len(self._cached_prefix_blocks(token_ids)) * self.block_size

    def can_allocate(self, num_tokens: int, token_ids: Sequence[int] | None = None) -> bool:
        cached = self._cached_prefix_blocks(token_ids, num_tokens) if token_ids is not None else []
        needed = _ceil_div(num_tokens, self.block_size) - len(cached)
        reclaimed = sum(1 for b in cached if b in self._evictable)  # reusing them makes them unavailable
        return needed <= self.num_free_blocks - reclaimed

    def can_append(self, seq_id: int, n: int) -> bool:
        return self._blocks_needed_to_append(seq_id, n) <= self.num_free_blocks

    # ---- mutations -----------------------------------------------------------------------------
    def allocate(self, seq_id: int, num_tokens: int, token_ids: Sequence[int] | None = None) -> int:
        """Reserve blocks for a new sequence's first num_tokens. Returns how many leading tokens were
        found in the prefix cache (their K/V is already there and must not be recomputed)."""
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already exists")
        if not self.can_allocate(num_tokens, token_ids):
            raise NoFreeBlocksError(f"not enough free blocks for {num_tokens} tokens")
        cached = self._cached_prefix_blocks(token_ids, num_tokens) if token_ids is not None else []
        for b in cached:  # take reused blocks out of the evictable pool *before* taking fresh ones
            if self._ref[b] == 0:
                del self._evictable[b]
            self._ref[b] += 1
        fresh = [self._take() for _ in range(_ceil_div(num_tokens, self.block_size) - len(cached))]
        self._tables[seq_id] = cached + fresh
        self._lens[seq_id] = num_tokens
        self._seq_keys[seq_id] = [self._key_of[b] for b in cached]
        return len(cached) * self.block_size

    def needs_register(self, seq_id: int, num_computed: int) -> bool:
        return self.enable_prefix_caching and num_computed // self.block_size > len(self._seq_keys[seq_id])

    def register_computed(self, seq_id: int, token_ids: Sequence[int], num_computed: int) -> None:
        """Add the sequence's newly completed full blocks (K/V computed) to the prefix cache."""
        if not self.enable_prefix_caching:
            return
        keys, table, bs = self._seq_keys[seq_id], self._tables[seq_id], self.block_size
        for i in range(len(keys), num_computed // bs):
            block, tokens = table[i], tuple(token_ids[i * bs:(i + 1) * bs])
            parent = keys[i - 1] if i else None
            key = _block_key(parent, tokens)
            keys.append(key)
            if block in self._key_of or key in self._block_of:
                continue  # already cached, or an identical block exists (e.g. computed concurrently)
            self._key_of[block], self._block_of[key] = key, block
            self._parent_key_of[block], self._tokens_of[block] = parent, tokens

    def append_slots(self, seq_id: int, n: int) -> list[tuple[int, int]]:
        """Grow the sequence by n tokens. Returns (src, dst) block copies the caller must perform."""
        needed = self._blocks_needed_to_append(seq_id, n)
        if needed > self.num_free_blocks:
            raise NoFreeBlocksError(f"need {needed} blocks, {self.num_free_blocks} free")
        table, length = self._tables[seq_id], self._lens[seq_id]
        copies: list[tuple[int, int]] = []
        if n > 0 and self._last_block_is_shared_partial(seq_id):
            old = table[-1]
            self._ref[old] -= 1  # still referenced by the other sharer(s), so never released here
            new = self._take()
            table[-1] = new
            copies.append((old, new))
        new_len = length + n
        while len(table) < _ceil_div(new_len, self.block_size):
            table.append(self._take())
        self._lens[seq_id] = new_len
        return copies

    def fork(self, parent: int, child: int) -> None:
        """The child shares all of the parent's blocks (copy-on-write)."""
        if child in self._tables:
            raise ValueError(f"sequence {child} already exists")
        table = self._tables[parent]
        for b in table:
            self._ref[b] += 1
        self._tables[child] = list(table)
        self._lens[child] = self._lens[parent]
        self._seq_keys[child] = list(self._seq_keys[parent])

    def rollback(self, seq_id: int, n: int) -> None:
        """Forget the last n tokens (e.g. rejected speculative tokens); release emptied blocks."""
        length = self._lens[seq_id]
        if not 0 <= n <= length:
            raise ValueError(f"cannot roll back {n} tokens from a sequence of length {length}")
        new_len = length - n
        table = self._tables[seq_id]
        while len(table) > _ceil_div(new_len, self.block_size):
            self._release(table.pop())
        full = new_len // self.block_size
        del self._seq_keys[seq_id][full:]
        if new_len % self.block_size and self._ref[table[-1]] == 1:
            self._uncache(table[-1])  # it will be overwritten in place: its cached content goes stale
        self._lens[seq_id] = new_len

    def free(self, seq_id: int) -> None:
        table = self._tables.pop(seq_id)  # KeyError for unknown or already-freed sequences
        del self._lens[seq_id]
        del self._seq_keys[seq_id]
        for b in table:
            self._release(b)

    def check_invariants(self) -> None:
        """Expensive consistency check, for tests."""
        counts = [0] * self.num_blocks
        for table in self._tables.values():
            for b in table:
                counts[b] += 1
        assert counts == self._ref, "refcounts don't match block tables"
        free, evictable = set(self._free), set(self._evictable)
        assert len(free) == len(self._free) and not free & evictable
        for b in range(self.num_blocks):
            if self._ref[b] == 0:
                assert (b in free) != (b in evictable), f"unused block {b} must be free xor evictable"
            else:
                assert b not in free and b not in evictable
        assert all(b in self._key_of for b in evictable), "evictable blocks must be cached"
        assert not free & set(self._key_of), "free blocks must not be cached"
        assert {self._block_of[k] for k in self._block_of} == set(self._key_of)

    # ---- internals -----------------------------------------------------------------------------
    def _cached_prefix_blocks(self, token_ids: Sequence[int] | None, num_tokens: int | None = None) -> list[int]:
        if not self.enable_prefix_caching or not token_ids:
            return []
        bs = self.block_size
        limit = (len(token_ids) - 1) // bs  # always leave >= 1 token to compute: we need its logits
        if num_tokens is not None:
            limit = min(limit, num_tokens // bs)
        blocks: list[int] = []
        parent: bytes | None = None
        for i in range(limit):
            tokens = tuple(token_ids[i * bs:(i + 1) * bs])
            key = _block_key(parent, tokens)
            b = self._block_of.get(key)
            # Verify rather than trust the hash: same parent and same tokens means the same prefix.
            if b is None or self._parent_key_of[b] != parent or self._tokens_of[b] != tokens:
                break
            blocks.append(b)
            parent = key
        return blocks

    def _take(self) -> int:
        if self._free:
            b = self._free.pop()
        else:
            b, _ = self._evictable.popitem(last=False)  # least recently used cached block
            self._uncache(b)
        self._ref[b] = 1
        return b

    def _release(self, block: int) -> None:
        self._ref[block] -= 1
        if self._ref[block] == 0:
            if block in self._key_of:
                self._evictable[block] = None  # most recently used goes to the end
            else:
                self._free.append(block)

    def _uncache(self, block: int) -> None:
        key = self._key_of.pop(block, None)
        if key is not None:
            del self._block_of[key]
            del self._parent_key_of[block]
            del self._tokens_of[block]

    def _last_block_is_shared_partial(self, seq_id: int) -> bool:
        table, length = self._tables[seq_id], self._lens[seq_id]
        return bool(table) and length % self.block_size != 0 and self._ref[table[-1]] > 1

    def _blocks_needed_to_append(self, seq_id: int, n: int) -> int:
        table, length = self._tables[seq_id], self._lens[seq_id]
        grow = _ceil_div(length + n, self.block_size) - len(table)
        cow = 1 if n > 0 and self._last_block_is_shared_partial(seq_id) else 0
        return grow + cow

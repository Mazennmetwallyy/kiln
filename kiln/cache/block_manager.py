"""Bookkeeping for the paged KV cache: which fixed-size blocks belong to which sequence.

This is the operating-system idea of paging applied to attention. A sequence's KV cache is a list of
blocks (its *block table*), which need not be contiguous in memory. Blocks are reference counted, so
sequences can share them (forks, and from M6 shared prompt prefixes). Writing into a shared,
partially filled block triggers copy-on-write. The manager only does bookkeeping. It never touches
tensors. Callers apply the copy pairs it returns.
"""

from __future__ import annotations


class NoFreeBlocksError(RuntimeError):
    """Not enough free blocks; nothing was changed."""


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))  # pop() hands out low ids first
        self._ref = [0] * num_blocks
        self._tables: dict[int, list[int]] = {}
        self._lens: dict[int, int] = {}

    # ---- queries -------------------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def seq_ids(self) -> set[int]:
        return set(self._tables)

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._tables[seq_id])

    def seq_len(self, seq_id: int) -> int:
        return self._lens[seq_id]

    def refcount(self, block_id: int) -> int:
        return self._ref[block_id]

    def slots(self, seq_id: int, start: int, count: int) -> list[int]:
        """Flat slot ids (block * block_size + offset) for token positions [start, start + count)."""
        if start < 0 or start + count > self._lens[seq_id]:
            raise ValueError(f"positions [{start}, {start + count}) outside sequence of length "
                             f"{self._lens[seq_id]}")
        table, bs = self._tables[seq_id], self.block_size
        return [table[p // bs] * bs + p % bs for p in range(start, start + count)]

    def can_allocate(self, num_tokens: int) -> bool:
        return _ceil_div(num_tokens, self.block_size) <= len(self._free)

    def can_append(self, seq_id: int, n: int) -> bool:
        return self._blocks_needed_to_append(seq_id, n) <= len(self._free)

    # ---- mutations -----------------------------------------------------------------------------
    def allocate(self, seq_id: int, num_tokens: int) -> None:
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already exists")
        needed = _ceil_div(num_tokens, self.block_size)
        if needed > len(self._free):
            raise NoFreeBlocksError(f"need {needed} blocks, {len(self._free)} free")
        self._tables[seq_id] = [self._take() for _ in range(needed)]
        self._lens[seq_id] = num_tokens

    def append_slots(self, seq_id: int, n: int) -> list[tuple[int, int]]:
        """Grow the sequence by n tokens. Returns (src, dst) block copies the caller must perform."""
        needed = self._blocks_needed_to_append(seq_id, n)
        if needed > len(self._free):
            raise NoFreeBlocksError(f"need {needed} blocks, {len(self._free)} free")
        table, length = self._tables[seq_id], self._lens[seq_id]
        copies: list[tuple[int, int]] = []
        if n > 0 and self._last_block_is_shared_partial(seq_id):
            old = table[-1]
            new = self._take()
            self._ref[old] -= 1
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

    def rollback(self, seq_id: int, n: int) -> None:
        """Forget the last n tokens (e.g. rejected speculative tokens); release emptied blocks."""
        length = self._lens[seq_id]
        if not 0 <= n <= length:
            raise ValueError(f"cannot roll back {n} tokens from a sequence of length {length}")
        new_len = length - n
        table = self._tables[seq_id]
        while len(table) > _ceil_div(new_len, self.block_size):
            self._release(table.pop())
        self._lens[seq_id] = new_len

    def free(self, seq_id: int) -> None:
        table = self._tables.pop(seq_id)  # KeyError for unknown or already-freed sequences
        del self._lens[seq_id]
        for b in table:
            self._release(b)

    # ---- internals -----------------------------------------------------------------------------
    def _take(self) -> int:
        b = self._free.pop()
        self._ref[b] = 1
        return b

    def _release(self, block: int) -> None:
        self._ref[block] -= 1
        if self._ref[block] == 0:
            self._free.append(block)

    def _last_block_is_shared_partial(self, seq_id: int) -> bool:
        table, length = self._tables[seq_id], self._lens[seq_id]
        return bool(table) and length % self.block_size != 0 and self._ref[table[-1]] > 1

    def _blocks_needed_to_append(self, seq_id: int, n: int) -> int:
        table, length = self._tables[seq_id], self._lens[seq_id]
        grow = _ceil_div(length + n, self.block_size) - len(table)
        cow = 1 if n > 0 and self._last_block_is_shared_partial(seq_id) else 0
        return grow + cow

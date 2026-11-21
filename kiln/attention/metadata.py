"""Per-step description of a flat (ragged) batch, built once and shared by every layer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from kiln.cache.block_manager import BlockManager


@dataclass
class AttentionMetadata:
    query_lens: list[int]  # new tokens per sequence this step, in flat order
    seq_lens: list[int]  # total length per sequence after this step
    slot_mapping: Tensor  # [T] pool slot for each new token
    block_tables: list[list[int]]
    block_size: int
    # Precomputed gather indices, so layers never rebuild them (and never sync with the device):
    context_slots: list[Tensor] = field(default_factory=list)  # per seq: slots for positions [0, len)
    decode_slots: Tensor | None = None  # [B, max_len] when every query_len == 1
    decode_mask: Tensor | None = None  # [B, 1, 1, max_len] True where the slot is real

    @property
    def is_pure_decode(self) -> bool:
        return all(q == 1 for q in self.query_lens)


def build_metadata(bm: BlockManager, seq_ids: list[int], query_lens: list[int],
                   device: str | torch.device) -> AttentionMetadata:
    """Call after the block manager has grown each sequence by its query_len."""
    bs = bm.block_size
    seq_lens = [bm.seq_len(s) for s in seq_ids]
    new_slots: list[int] = []
    context: list[list[int]] = []
    for sid, q, n in zip(seq_ids, query_lens, seq_lens, strict=True):
        all_slots = bm.slots(sid, 0, n)
        context.append(all_slots)
        new_slots += all_slots[n - q:]

    meta = AttentionMetadata(
        query_lens=list(query_lens), seq_lens=seq_lens,
        slot_mapping=torch.tensor(new_slots, dtype=torch.long, device=device),
        block_tables=[bm.block_table(s) for s in seq_ids], block_size=bs,
    )
    if meta.is_pure_decode:
        width = max(seq_lens)
        padded = [c + [0] * (width - len(c)) for c in context]
        meta.decode_slots = torch.tensor(padded, dtype=torch.long, device=device)
        lens = torch.tensor(seq_lens, device=device)
        meta.decode_mask = (torch.arange(width, device=device)[None, :] < lens[:, None])[:, None, None, :]
    else:
        meta.context_slots = [torch.tensor(c, dtype=torch.long, device=device) for c in context]
    return meta

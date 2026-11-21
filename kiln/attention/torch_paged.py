"""Paged attention in plain PyTorch: gather each sequence's K/V through its slots, then SDPA.

This is the readable reference backend. M8 replaces the decode path with a Metal kernel that reads
the paged pool directly instead of gathering into a temporary tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor

from kiln.attention.base import sdpa_gqa
from kiln.attention.metadata import AttentionMetadata
from kiln.cache.paged import PagedKVCache


class PagedContext:
    def __init__(self, cache: PagedKVCache, meta: AttentionMetadata):
        self.cache = cache
        self.meta = meta

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        meta = self.meta
        k_pool, v_pool = self.cache.k[layer], self.cache.v[layer]
        k_pool[meta.slot_mapping] = k
        v_pool[meta.slot_mapping] = v

        if meta.is_pure_decode:
            assert meta.decode_slots is not None
            keys = k_pool[meta.decode_slots].transpose(1, 2)  # [B, Hkv, L, D]
            values = v_pool[meta.decode_slots].transpose(1, 2)
            out = sdpa_gqa(q[:, :, None, :], keys, values, meta.decode_mask)  # [B, H, 1, D]
            return out[:, :, 0, :]

        outs = []
        start = 0
        for q_len, seq_len, slots in zip(meta.query_lens, meta.seq_lens, meta.context_slots, strict=True):
            qs = q[start:start + q_len].transpose(0, 1)  # [H, q, D]
            keys = k_pool[slots].transpose(0, 1)  # [Hkv, L, D]
            values = v_pool[slots].transpose(0, 1)
            # The q new tokens sit at positions [seq_len - q_len, seq_len); each sees keys up to itself.
            q_pos = torch.arange(seq_len - q_len, seq_len, device=q.device)
            mask = torch.arange(seq_len, device=q.device)[None, :] <= q_pos[:, None]
            outs.append(sdpa_gqa(qs, keys, values, mask).transpose(0, 1))
            start += q_len
        return torch.cat(outs)

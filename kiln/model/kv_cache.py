"""M1's simple KV cache: one contiguous [max_tokens, Hkv, D] buffer per layer, for one sequence.

Kept as the reference `AttentionContext`; the engine uses the paged cache (`kiln.cache`).
"""

from __future__ import annotations

import torch
from torch import Tensor

from kiln.attention.base import sdpa_gqa
from kiln.model.config import ModelConfig


class ContiguousKVCache:
    def __init__(self, config: ModelConfig, max_tokens: int, dtype: torch.dtype, device: str | torch.device):
        shape = (max_tokens, config.num_kv_heads, config.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_layers)]
        self.capacity = max_tokens
        self.device = device
        self._start = 0
        self._count = 0

    def set_positions(self, start: int, count: int) -> None:
        """The next forward pass writes `count` new tokens at positions [start, start + count)."""
        if start + count > self.capacity:
            raise ValueError(f"positions up to {start + count - 1} exceed cache capacity {self.capacity}")
        self._start, self._count = start, count

    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        start, end = self._start, self._start + self._count
        self.k[layer][start:end] = k
        self.v[layer][start:end] = v
        keys, values = self.k[layer][:end], self.v[layer][:end]
        # Causal mask for new tokens after a cached prefix: the query at position p sees keys 0..p.
        q_pos = torch.arange(start, end, device=q.device)
        mask = torch.arange(end, device=q.device)[None, :] <= q_pos[:, None]  # [T, end]
        out = sdpa_gqa(q.transpose(0, 1), keys.transpose(0, 1), values.transpose(0, 1), mask)
        return out.transpose(0, 1)

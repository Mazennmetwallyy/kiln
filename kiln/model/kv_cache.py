"""M1's simple KV cache: one contiguous [max_tokens, Hkv, D] buffer per layer, for one sequence.

M2 replaces it with a paged pool that has the same write/read interface.
"""

from __future__ import annotations

import torch
from torch import Tensor

from kiln.model.config import ModelConfig


class ContiguousKVCache:
    def __init__(self, config: ModelConfig, max_tokens: int, dtype: torch.dtype, device: str | torch.device):
        shape = (max_tokens, config.num_kv_heads, config.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_layers)]
        self.capacity = max_tokens

    def write(self, layer: int, positions: Tensor, k: Tensor, v: Tensor) -> None:
        """Store K/V for the given token positions. k, v: [T, Hkv, D]."""
        if int(positions.max()) >= self.capacity:
            raise ValueError(f"position {int(positions.max())} is beyond cache capacity {self.capacity}")
        self.k[layer][positions] = k
        self.v[layer][positions] = v

    def read(self, layer: int, length: int) -> tuple[Tensor, Tensor]:
        """K/V for positions [0, length)."""
        return self.k[layer][:length], self.v[layer][:length]

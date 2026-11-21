"""The physical KV pool: one flat tensor of token slots per layer, shared by every sequence."""

from __future__ import annotations

import torch

from kiln.model.config import ModelConfig


class PagedKVCache:
    def __init__(self, config: ModelConfig, num_blocks: int, block_size: int, dtype: torch.dtype,
                 device: str | torch.device):
        shape = (num_blocks * block_size, config.num_kv_heads, config.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_layers)]
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)

    def copy_blocks(self, pairs: list[tuple[int, int]]) -> None:
        """Copy-on-write: duplicate whole blocks (src -> dst) in every layer."""
        bs = self.block_size
        for src, dst in pairs:
            for pool in (*self.k, *self.v):
                pool[dst * bs:(dst + 1) * bs] = pool[src * bs:(src + 1) * bs]

    @staticmethod
    def bytes_per_block(config: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
        elem = torch.tensor([], dtype=dtype).element_size()
        return 2 * config.num_layers * block_size * config.num_kv_heads * config.head_dim * elem

    @staticmethod
    def blocks_for_memory(config: ModelConfig, bytes_budget: int, block_size: int, dtype: torch.dtype) -> int:
        return bytes_budget // PagedKVCache.bytes_per_block(config, block_size, dtype)

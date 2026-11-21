"""Grouped-query self-attention over a flat stream of new tokens plus the cached prefix."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kiln.model.config import ModelConfig
from kiln.model.kv_cache import ContiguousKVCache
from kiln.model.layers import RotaryEmbedding


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=bias)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, x: Tensor, positions: Tensor, cache: ContiguousKVCache, layer_idx: int,
                rope: RotaryEmbedding) -> Tensor:
        """x: [T, hidden] for T new tokens of one sequence at `positions` (ascending, contiguous)."""
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        q, k = rope(q, k, positions)

        cache.write(layer_idx, positions, k, v)
        length = int(positions[-1]) + 1
        k_all, v_all = cache.read(layer_idx, length)  # [L, Hkv, D]

        # GQA: each group of num_heads // num_kv_heads query heads shares one K/V head.
        group = self.num_heads // self.num_kv_heads
        k_all = k_all.repeat_interleave(group, dim=1)
        v_all = v_all.repeat_interleave(group, dim=1)

        # Causal mask for new tokens that follow a cached prefix: query at position p sees keys 0..p.
        mask = torch.arange(length, device=x.device)[None, :] <= positions[:, None]  # [T, L]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k_all.transpose(0, 1), v_all.transpose(0, 1), attn_mask=mask
        )  # [H, T, D]
        return self.o_proj(out.transpose(0, 1).reshape(t, self.num_heads * self.head_dim))

"""Self-attention projections. Storage and the attention computation live in the AttentionContext."""

from __future__ import annotations

from torch import Tensor, nn

from kiln.attention.base import AttentionContext
from kiln.model.config import ModelConfig
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

    def forward(self, x: Tensor, positions: Tensor, ctx: AttentionContext, layer_idx: int,
                rope: RotaryEmbedding) -> Tensor:
        """x: [T, hidden] for T new tokens (possibly from several sequences)."""
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        q, k = rope(q, k, positions)
        out = ctx.attend(layer_idx, q, k, v)  # [T, H, D]
        return self.o_proj(out.reshape(t, self.num_heads * self.head_dim))

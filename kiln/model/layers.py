"""Building blocks of a Qwen2/Llama decoder layer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square norm: rescale each vector to unit RMS, then by a learned gain.

    Unlike LayerNorm it does not subtract the mean, which saves work and is what these models were
    trained with. The statistic is computed in fp32 even for fp16 inputs, to avoid overflow in x².
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE), Hugging Face "rotate_half" layout.

    Each pair of dimensions (i, i + D/2) is rotated by angle pos * theta^(-2i/D). Rotating q and k by
    their own positions makes q·k depend only on the distance between them.
    """

    def __init__(self, head_dim: int, theta: float, max_positions: int):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)  # [P, D/2]
        angles = torch.cat([angles, angles], dim=-1)  # [P, D]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        # q: [T, H, D], k: [T, Hkv, D], positions: [T]
        cos = self.cos[positions].unsqueeze(1)  # [T, 1, D]
        sin = self.sin[positions].unsqueeze(1)
        q_out = q.float() * cos + self._rotate_half(q.float()) * sin
        k_out = k.float() * cos + self._rotate_half(k.float()) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)


class SwiGLU(nn.Module):
    """Gated feed-forward block: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

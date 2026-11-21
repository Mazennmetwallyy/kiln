"""The boundary between the model and wherever K/V live.

Attention computes q, k, v and applies RoPE, then hands them to an `AttentionContext`. The context
stores the new K/V, fetches the history it needs and computes attention. Swapping the context swaps
the memory layout (contiguous, paged) or the kernel (PyTorch, Metal) without touching the model.
"""

from __future__ import annotations

from typing import Protocol

import torch.nn.functional as F
from torch import Tensor


class AttentionContext(Protocol):
    def attend(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """q: [T, H, D]; k, v: [T, Hkv, D] for the T new tokens. Returns [T, H, D]."""
        ...


def sdpa_gqa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
    """Attention with grouped KV heads. q: [..., H, Tq, D]; k, v: [..., Hkv, Tk, D]; mask True = attend."""
    group = q.shape[-3] // k.shape[-3]
    if group > 1:
        k = k.repeat_interleave(group, dim=-3)
        v = v.repeat_interleave(group, dim=-3)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

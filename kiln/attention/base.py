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
    """Attention with grouped KV heads. q: [..., H, Tq, D]; k, v: [..., Hkv, Tk, D]; mask True = attend.

    Query head h uses KV head h // G (G = H / Hkv). Rather than copying each KV head G times
    (repeat_interleave), fold the G query heads of a group into the query-length dimension:
    [..., Hkv, G*Tq, D]. Every KV head is then read once. The mask doesn't depend on the head, so it is
    tiled G times along its query rows.
    """
    *lead, h, tq, d = q.shape
    hkv = k.shape[-3]
    group = h // hkv
    if group == 1:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    qf = q.reshape(*lead, hkv, group * tq, d)
    if mask is not None and mask.shape[-2] != 1:
        mask = mask.repeat(*([1] * (mask.dim() - 2)), group, 1)  # row g*Tq + t -> mask row t
    out = F.scaled_dot_product_attention(qf, k, v, attn_mask=mask)
    return out.reshape(*lead, h, tq, d)

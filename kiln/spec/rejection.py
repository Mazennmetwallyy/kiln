"""Verify draft tokens so the output follows the target model's distribution exactly.

Modified rejection sampling (Leviathan et al. 2023; Chen et al. 2023). For each draft token d_i,
drawn from the draft's distribution q_i:

    accept with probability min(1, p_i(d_i) / q_i(d_i))

At the first rejection, sample a replacement from the residual norm(max(0, p_i - q_i)) and stop. If all
k drafts are accepted, sample one bonus token from p_k, which the target computed anyway. Every
emitted token is then distributed exactly as if the target had sampled it alone: the accepted part
contributes min(p, q), and the residual contributes p - min(p, q).

Greedy decoding is the special case where p and q are one-hot: accept while the draft equals the
target's argmax, then emit the target's argmax.
"""

from __future__ import annotations

import torch
from torch import Tensor


def verify(drafts: list[int], q: Tensor | None, p: Tensor, greedy: bool,
           generator: torch.Generator | None) -> tuple[int, int]:
    """drafts: k proposed tokens. q: [k, V] draft probs (unused if greedy). p: [k + 1, V] target probs.

    Returns (j, token): the first j drafts are accepted and `token` follows them."""
    k = len(drafts)
    if greedy:
        target = p.argmax(dim=-1).tolist()
        j = 0
        while j < k and drafts[j] == target[j]:
            j += 1
        return j, target[j]

    u = torch.rand(k, generator=generator).tolist()
    for i, d in enumerate(drafts):
        assert q is not None
        pd, qd = float(p[i, d]), float(q[i, d])
        if qd > 0 and u[i] * qd <= pd:  # u < p/q, written without dividing
            continue
        residual = (p[i] - q[i]).clamp_min(0)
        total = float(residual.sum())
        # If p <= q everywhere the residual is empty; that has probability 0 of being reached unless
        # p == q, where any rejection is a float artefact and p itself is the right distribution.
        dist = residual / total if total > 1e-12 else p[i]
        return i, int(torch.multinomial(dist, 1, generator=generator))
    return k, int(torch.multinomial(p[k], 1, generator=generator))

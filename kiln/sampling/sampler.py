"""Turn a vector of logits into one token id."""

from __future__ import annotations

import torch
from torch import Tensor

from kiln.sampling.params import SamplingParams

GREEDY_BELOW = 1e-5


def probs(logits: Tensor, params: SamplingParams) -> Tensor:
    """The distribution `sample` draws from: temperature → top-k → top-p, renormalised. One-hot if greedy."""
    if params.temperature < GREEDY_BELOW:  # dividing by ~0 overflows to inf and softmax then gives NaN
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(0, torch.argmax(logits).view(1), 1.0)

    logits = logits.float() / params.temperature
    if params.top_k > 0 and params.top_k < logits.numel():
        kth = torch.topk(logits, params.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    p = torch.softmax(logits, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, order = torch.sort(p, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest prefix whose mass reaches top_p: drop a token if the mass *before* it
        # already reaches top_p. The first token is always kept.
        drop = (cumulative - sorted_probs) >= params.top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        p = torch.zeros_like(p).scatter(0, order, sorted_probs)
        p = p / p.sum()
    return p


def sample(logits: Tensor, params: SamplingParams, generator: torch.Generator | None) -> int:
    """logits: [vocab]. Greedy if temperature == 0, else temperature → top-k → top-p → draw."""
    if params.temperature < GREEDY_BELOW:
        return int(torch.argmax(logits))
    return int(torch.multinomial(probs(logits, params).cpu(), 1, generator=generator))

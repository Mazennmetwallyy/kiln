"""Single-sequence generation: prefill the prompt once, then decode one token per step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from kiln.model.config import ModelConfig
from kiln.model.kv_cache import ContiguousKVCache
from kiln.sampling import SamplingParams, sample


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    finish_reason: Literal["stop", "length"]


@torch.no_grad()
def generate(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
             tokenizer: Any = None, max_context: int | None = None) -> GenerationResult:
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if params.stop and tokenizer is None:
        raise ValueError("stop strings need a tokenizer")
    limit = max_context or config.max_position_embeddings
    total = len(prompt_ids) + params.max_tokens
    if total > limit:
        raise ValueError(f"prompt ({len(prompt_ids)}) + max_tokens ({params.max_tokens}) = {total} "
                         f"exceeds the context limit of {limit}")
    if params.max_tokens == 0:
        return GenerationResult([], "", "length")

    weight = next(model.parameters())
    device = weight.device
    cache = ContiguousKVCache(config, max_tokens=total, dtype=weight.dtype, device=device)
    generator = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    stop_ids = set(params.stop_token_ids)
    if not params.ignore_eos:
        stop_ids.update(config.eos_token_ids)

    out: list[int] = []
    ids = torch.tensor(prompt_ids, device=device)
    positions = torch.arange(len(prompt_ids), device=device)
    while True:
        logits = model(ids, positions, cache)[-1]
        token = sample(logits, params, generator)
        if token in stop_ids:
            return GenerationResult(out, _decode(tokenizer, out), "stop")
        out.append(token)
        if len(out) == params.max_tokens:
            return GenerationResult(out, _decode(tokenizer, out), "length")
        next_pos = len(prompt_ids) + len(out) - 1
        ids = torch.tensor([token], device=device)
        positions = torch.tensor([next_pos], device=device)


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids) if tokenizer is not None else ""

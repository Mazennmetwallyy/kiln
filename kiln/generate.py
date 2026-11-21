"""Single-sequence generation: prefill the prompt once, then decode one token per step."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from kiln.model.config import ModelConfig
from kiln.model.kv_cache import ContiguousKVCache
from kiln.sampling import SamplingParams, sample
from kiln.tokenizer import IncrementalDetokenizer, Tokenizer

FinishReason = Literal["stop", "length"]


@dataclass
class GenerationResult:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: FinishReason = "length"


def generate(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
             tokenizer: Tokenizer | None = None, max_context: int | None = None) -> GenerationResult:
    result = GenerationResult()
    for _ in _run(model, config, prompt_ids, params, tokenizer, max_context, result):
        pass
    return result


def stream_generate(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
                    tokenizer: Tokenizer | None = None, max_context: int | None = None) -> Iterator[str]:
    """Yield text deltas as they become final. Joined, they equal `generate(...).text`."""
    yield from _run(model, config, prompt_ids, params, tokenizer, max_context, GenerationResult())


def _run(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
         tokenizer: Tokenizer | None, max_context: int | None, result: GenerationResult) -> Iterator[str]:
    if params.stop and tokenizer is None:
        raise ValueError("stop strings need a tokenizer")
    tokens = _token_stream(model, config, prompt_ids, params, max_context)
    detok = IncrementalDetokenizer(tokenizer) if tokenizer is not None else None
    pending = ""  # decoded text not yet emitted, held back in case it starts a stop string
    try:
        while True:
            token = next(tokens)
            result.token_ids.append(token)
            if detok is None:
                continue
            pending += detok.push(token)
            hit = _find_stop(pending, params.stop)
            if hit is not None:
                result.text += pending[:hit]
                result.finish_reason = "stop"
                if hit:
                    yield pending[:hit]
                return
            keep = _partial_stop_suffix(pending, params.stop)
            ready, pending = pending[: len(pending) - keep], pending[len(pending) - keep:]
            if ready:
                result.text += ready
                yield ready
    except StopIteration as done:
        result.finish_reason = done.value
        if pending:  # generation ended; a partial stop string never completed, so it's real text
            result.text += pending
            yield pending


@torch.no_grad()
def _token_stream(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
                  max_context: int | None) -> Generator[int, None, FinishReason]:
    """Yield generated token ids (stop/EOS ids excluded); return the finish reason."""
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    limit = config.max_position_embeddings
    if max_context is not None:
        limit = min(limit, max_context)  # a deployment may lower the limit, never raise it past RoPE's table
    total = len(prompt_ids) + params.max_tokens
    if total > limit:
        raise ValueError(f"prompt ({len(prompt_ids)}) + max_tokens ({params.max_tokens}) = {total} "
                         f"exceeds the context limit of {limit}")
    if params.max_tokens == 0:
        return "length"

    weight = next(model.parameters())
    device = weight.device
    cache = ContiguousKVCache(config, max_tokens=total, dtype=weight.dtype, device=device)
    generator = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    stop_ids = set(params.stop_token_ids)
    if not params.ignore_eos:
        stop_ids.update(config.eos_token_ids)

    ids = torch.tensor(prompt_ids, device=device)
    positions = torch.arange(len(prompt_ids), device=device)
    for n in range(1, params.max_tokens + 1):
        token = sample(model(ids, positions, cache)[-1], params, generator)
        if token in stop_ids:
            return "stop"
        yield token
        ids = torch.tensor([token], device=device)
        positions = torch.tensor([len(prompt_ids) + n - 1], device=device)
    return "length"


def _find_stop(text: str, stops: list[str]) -> int | None:
    hits = [i for s in stops if (i := text.find(s)) != -1]
    return min(hits) if hits else None


def _partial_stop_suffix(text: str, stops: list[str]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of some stop string."""
    best = 0
    for s in stops:
        for n in range(min(len(s) - 1, len(text)), best, -1):
            if text.endswith(s[:n]):
                best = n
                break
    return best

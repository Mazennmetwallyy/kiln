"""One-request convenience API on top of the engine."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from torch import nn

from kiln.engine.engine import LLMEngine, RequestOutput
from kiln.model.config import ModelConfig
from kiln.sampling import SamplingParams
from kiln.tokenizer import Tokenizer

BLOCK_SIZE = 16


@dataclass
class GenerationResult:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str = "length"


def generate_stream(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
                    tokenizer: Tokenizer | None = None, max_context: int | None = None) -> Iterator[RequestOutput]:
    """Yield the engine's per-step outputs for a single request. Validation errors raise immediately."""
    num_blocks = max(1, -(-(len(prompt_ids) + params.max_tokens) // BLOCK_SIZE))
    engine = LLMEngine(model, config, tokenizer, num_blocks=num_blocks, block_size=BLOCK_SIZE,
                       max_batched_tokens=max(512, len(prompt_ids)), max_context=max_context)
    engine.add_request("0", prompt_ids, params)
    return _drain(engine)


def _drain(engine: LLMEngine) -> Iterator[RequestOutput]:
    while engine.has_unfinished():
        yield from engine.step()


def generate(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
             tokenizer: Tokenizer | None = None, max_context: int | None = None) -> GenerationResult:
    result = GenerationResult()
    for out in generate_stream(model, config, prompt_ids, params, tokenizer, max_context):
        result.token_ids += out.new_token_ids
        result.text += out.new_text
        if out.finished and out.finish_reason is not None:
            result.finish_reason = out.finish_reason
    return result


def stream_generate(model: nn.Module, config: ModelConfig, prompt_ids: list[int], params: SamplingParams,
                    tokenizer: Tokenizer | None = None, max_context: int | None = None) -> Iterator[str]:
    """Yield text deltas as they become final. Joined, they equal `generate(...).text`."""
    for out in generate_stream(model, config, prompt_ids, params, tokenizer, max_context):
        if out.new_text:
            yield out.new_text

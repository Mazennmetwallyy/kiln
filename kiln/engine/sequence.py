"""One request's state as it moves through the engine."""

from __future__ import annotations

import enum

import torch

from kiln.engine.text import TextStreamer
from kiln.sampling import SamplingParams


class SeqStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    def __init__(self, seq_id: int, request_id: str, prompt_ids: list[int], params: SamplingParams,
                 streamer: TextStreamer | None):
        self.seq_id = seq_id
        self.request_id = request_id
        self.prompt_ids = list(prompt_ids)
        self.params = params
        self.streamer = streamer
        self.output_ids: list[int] = []
        self.num_computed = 0  # tokens whose K/V are in the cache
        self.status = SeqStatus.WAITING
        self.finish_reason: str | None = None
        self.num_preemptions = 0
        # Each request owns its RNG, so its samples don't depend on what else is in the batch.
        self.generator = torch.Generator().manual_seed(params.seed) if params.seed is not None else None

    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def __repr__(self) -> str:
        return f"Sequence({self.seq_id}, {self.status.value}, computed={self.num_computed}/{self.num_tokens})"

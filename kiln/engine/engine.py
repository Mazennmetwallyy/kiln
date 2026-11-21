"""The engine loop: schedule → build one flat batch → one forward pass → sample → update.

Every call to `step()` runs the model once over a ragged batch that can mix prompt chunks (prefill)
and single next tokens (decode) from many requests. Requests join and leave between steps, so a
finished request's slot is reused at once. That's continuous batching, versus static batching where
the whole batch waits for its slowest member.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch
from torch import nn

from kiln.attention.metadata import build_metadata
from kiln.attention.torch_paged import PagedContext
from kiln.cache import BlockManager
from kiln.cache.paged import PagedKVCache
from kiln.engine.scheduler import Scheduler
from kiln.engine.sequence import SeqStatus, Sequence
from kiln.engine.text import TextStreamer
from kiln.model.config import ModelConfig
from kiln.sampling import SamplingParams, sample
from kiln.tokenizer import Tokenizer


@dataclass
class RequestOutput:
    request_id: str
    new_token_ids: list[int] = field(default_factory=list)
    new_text: str = ""
    finished: bool = False
    finish_reason: str | None = None


@dataclass
class EngineStats:
    num_running: int
    num_waiting: int
    num_free_blocks: int
    num_blocks: int
    num_preemptions_total: int


class LLMEngine:
    def __init__(self, model: nn.Module, config: ModelConfig, tokenizer: Tokenizer | None, num_blocks: int,
                 block_size: int = 16, max_num_seqs: int = 32, max_batched_tokens: int = 512,
                 max_context: int | None = None):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        weight = next(model.parameters())
        self.device = weight.device
        self.bm = BlockManager(num_blocks, block_size)
        self.cache = PagedKVCache(config, num_blocks, block_size, weight.dtype, self.device)
        self.scheduler = Scheduler(self.bm, max_num_seqs, max_batched_tokens)
        self.max_context = config.max_position_embeddings
        if max_context is not None:
            self.max_context = min(self.max_context, max_context)
        self._ids = itertools.count()
        self._live: dict[str, Sequence] = {}  # request_id -> sequence, until finished
        self._finished_now: list[RequestOutput] = []

    # ---- public API ----------------------------------------------------------------------------
    def add_request(self, request_id: str, prompt_ids: list[int], params: SamplingParams) -> None:
        """Validate and queue a request. Raises ValueError straight away, never mid-stream."""
        if request_id in self._live:
            raise ValueError(f"request {request_id!r} already exists")
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if params.stop and self.tokenizer is None:
            raise ValueError("stop strings need a tokenizer")
        if min(prompt_ids) < 0 or max(prompt_ids) >= self.config.vocab_size:
            raise ValueError(f"prompt token ids must be in [0, {self.config.vocab_size}) (the model's vocab)")
        total = len(prompt_ids) + params.max_tokens
        if total > self.max_context:
            raise ValueError(f"prompt ({len(prompt_ids)}) + max_tokens ({params.max_tokens}) = {total} "
                             f"exceeds the context limit of {self.max_context}")
        streamer = TextStreamer(self.tokenizer, params.stop) if self.tokenizer is not None else None
        seq = Sequence(next(self._ids), request_id, prompt_ids, params, streamer)
        if params.max_tokens == 0:
            seq.status, seq.finish_reason = SeqStatus.FINISHED, "length"
            self._finished_now.append(RequestOutput(request_id, finished=True, finish_reason="length"))
            return
        self.scheduler.add(seq)  # raises if it could never fit in the KV pool
        self._live[request_id] = seq

    def abort(self, request_id: str) -> None:
        seq = self._live.pop(request_id, None)
        if seq is not None:
            self.scheduler.abort(seq.seq_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished() or bool(self._finished_now)

    def stats(self) -> EngineStats:
        return EngineStats(len(self.scheduler.running), len(self.scheduler.waiting), self.bm.num_free_blocks,
                           self.bm.num_blocks, self.scheduler.num_preemptions)

    @torch.no_grad()
    def step(self) -> list[RequestOutput]:
        outputs, self._finished_now = self._finished_now, []
        plan = self.scheduler.schedule()
        if not plan.scheduled:
            return outputs
        if plan.copies:
            self.cache.copy_blocks(plan.copies)

        # Flatten every scheduled sequence's new tokens into one stream.
        ids: list[int] = []
        positions: list[int] = []
        to_sample: list[tuple[int, Sequence]] = []  # (row in the flat batch, sequence)
        for s in plan.scheduled:
            seq = s.seq
            ids += seq.tokens[s.start:s.start + s.num_tokens]
            positions += range(s.start, s.start + s.num_tokens)
            seq.num_computed += s.num_tokens
            if seq.num_computed == seq.num_tokens:  # prompt fully processed (or a decode step)
                to_sample.append((len(ids) - 1, seq))

        meta = build_metadata(self.bm, [s.seq.seq_id for s in plan.scheduled],
                              [s.num_tokens for s in plan.scheduled], self.device)
        rows = torch.tensor([r for r, _ in to_sample], dtype=torch.long, device=self.device)
        logits = self.model(torch.tensor(ids, device=self.device), torch.tensor(positions, device=self.device),
                            PagedContext(self.cache, meta), logits_indices=rows)
        for (_, seq), token in zip(to_sample, self._sample_all(logits, [s for _, s in to_sample]), strict=True):
            outputs.append(self._process_token(seq, token))
        return outputs

    # ---- internals -----------------------------------------------------------------------------
    def _sample_all(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int | None]:
        """One device→host transfer per step. Greedy rows are argmax'd on the device, so only their token
        ids cross over; rows that sample randomly bring their full logits to the CPU."""
        greedy = [s.params.temperature == 0 for s in seqs]
        tokens: list[int | None] = [None] * len(seqs)
        if any(greedy):
            idx = [i for i, g in enumerate(greedy) if g]
            for i, t in zip(idx, logits[idx].argmax(dim=-1).tolist(), strict=True):
                tokens[i] = t
        if not all(greedy):
            idx = [i for i, g in enumerate(greedy) if not g]
            rows = logits[idx].cpu()
            for j, i in enumerate(idx):
                try:
                    tokens[i] = sample(rows[j], seqs[i].params, seqs[i].generator)
                except Exception:  # one bad request must not take the whole batch down
                    tokens[i] = None
        return tokens

    def _process_token(self, seq: Sequence, token: int | None) -> RequestOutput:
        out = RequestOutput(seq.request_id)
        if token is None:
            return self._finish(seq, out, "error")
        params = seq.params
        stop_ids = set(params.stop_token_ids)
        if not params.ignore_eos:
            stop_ids.update(self.config.eos_token_ids)
        if token in stop_ids:
            return self._finish(seq, out, "stop")

        seq.output_ids.append(token)
        out.new_token_ids = [token]
        if seq.streamer is not None:
            out.new_text, hit = seq.streamer.push(token)
            if hit:
                return self._finish(seq, out, "stop")
        if len(seq.output_ids) >= params.max_tokens:
            return self._finish(seq, out, "length")
        return out

    def _finish(self, seq: Sequence, out: RequestOutput, reason: str) -> RequestOutput:
        if reason == "length" and seq.streamer is not None:
            out.new_text += seq.streamer.flush()
        self.scheduler.finish(seq, reason)
        self._live.pop(seq.request_id, None)
        out.finished, out.finish_reason = True, reason
        return out

"""The engine loop: schedule → build one flat batch → one forward pass → sample → update.

Every call to `step()` runs the model once over a ragged batch that can mix prompt chunks (prefill)
and single next tokens (decode) from many requests. Requests join and leave between steps, so a
finished request's slot is reused at once. That's continuous batching, versus static batching where
the whole batch waits for its slowest member.

With a draft model (speculative decoding), a step where every request is decoding becomes: the draft
proposes k tokens per request (k cheap forwards), the target scores them all in one forward, and
rejection sampling keeps the longest acceptable prefix plus one token from the target. The draft has
its own KV pool addressed by the *same* block tables, so allocation, copy-on-write, preemption,
rollback and prefix caching cover both caches at once.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from kiln.attention import metal
from kiln.attention.metadata import build_metadata
from kiln.attention.torch_paged import PagedContext
from kiln.cache import BlockManager
from kiln.cache.paged import PagedKVCache
from kiln.engine.scheduler import Scheduler
from kiln.engine.sequence import SeqStatus, Sequence
from kiln.engine.text import TextStreamer
from kiln.model.config import ModelConfig
from kiln.sampling import SamplingParams, probs, sample
from kiln.spec.rejection import verify
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
    prefix_cache_query_tokens: int = 0
    prefix_cache_hit_tokens: int = 0
    num_cached_blocks: int = 0
    spec_proposed_tokens: int = 0  # draft tokens offered to the target
    spec_accepted_tokens: int = 0  # of which the target accepted
    num_target_forwards: int = 0


class LLMEngine:
    def __init__(self, model: nn.Module, config: ModelConfig, tokenizer: Tokenizer | None, num_blocks: int,
                 block_size: int = 16, max_num_seqs: int = 32, max_batched_tokens: int = 512,
                 max_context: int | None = None, enable_prefix_caching: bool = True,
                 draft_model: nn.Module | None = None, draft_config: ModelConfig | None = None,
                 num_speculative_tokens: int = 0, attention_backend: str = "torch"):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        weight = next(model.parameters())
        self.device = weight.device
        self.bm = BlockManager(num_blocks, block_size, enable_prefix_caching=enable_prefix_caching)
        self.cache = PagedKVCache(config, num_blocks, block_size, weight.dtype, self.device)
        if attention_backend not in ("torch", "metal"):
            raise ValueError(f"unknown attention backend {attention_backend!r}")
        if attention_backend == "metal" and not all(
                metal.supported(weight.dtype, c.head_dim) and weight.device.type == "mps"
                for c in [config, *([draft_config] if draft_config is not None else [])]):
            raise ValueError("the metal backend needs an MPS device, fp16/fp32 and head_dim a multiple of 32")
        self.backend = attention_backend
        self.draft = draft_model
        self.k = num_speculative_tokens
        self.draft_cache: PagedKVCache | None = None
        if draft_model is not None:
            if draft_config is None or self.k < 1:
                raise ValueError("a draft model needs its config and num_speculative_tokens >= 1")
            if draft_config.vocab_size != config.vocab_size:
                raise ValueError(f"draft vocab ({draft_config.vocab_size}) must equal the target's "
                                 f"({config.vocab_size}): both models must use the same tokenizer")
            dweight = next(draft_model.parameters())
            self.draft_cache = PagedKVCache(draft_config, num_blocks, block_size, dweight.dtype, self.device)
        self.spec_proposed = self.spec_accepted = self.target_forwards = 0
        self.scheduler = Scheduler(self.bm, max_num_seqs, max_batched_tokens)
        self.max_context = config.max_position_embeddings
        if max_context is not None:
            self.max_context = min(self.max_context, max_context)
        if draft_config is not None and draft_model is not None:
            # The draft sees every position the target does, so its context window limits requests too.
            self.max_context = min(self.max_context, draft_config.max_position_embeddings)
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
        sch = self.scheduler
        return EngineStats(len(sch.running), len(sch.waiting), self.bm.num_free_blocks, self.bm.num_blocks,
                           sch.num_preemptions, sch.prefix_query_tokens, sch.prefix_hit_tokens,
                           self.bm.num_cached_blocks, self.spec_proposed, self.spec_accepted,
                           self.target_forwards)

    def kv_snapshot(self) -> dict[str, Any]:
        """What the KV visualiser draws: every block's state and owner, and every live request.
        Call it on the engine thread (it reads the block manager without locks)."""
        snap = self.bm.snapshot()
        refcount: list[int] = snap["refcount"]  # type: ignore[assignment]
        cached: list[bool] = snap["cached"]  # type: ignore[assignment]
        tables: dict[int, list[int]] = snap["tables"]  # type: ignore[assignment]
        owner = [-1] * self.bm.num_blocks
        for sid, table in tables.items():
            for b in table:
                if owner[b] == -1:
                    owner[b] = sid
        state = []
        for b in range(self.bm.num_blocks):
            if refcount[b] > 1:
                state.append(2)  # shared (prefix cache hit, or fork)
            elif refcount[b] == 1:
                state.append(1)  # used by one request
            else:
                state.append(3 if cached[b] else 0)  # 3: cached and evictable, 0: free
        seqs = [*self.scheduler.running, *self.scheduler.waiting]
        s = self.stats()
        return {
            "num_blocks": self.bm.num_blocks, "block_size": self.bm.block_size, "state": state, "owner": owner,
            "requests": [{"id": q.request_id, "seq": q.seq_id, "status": q.status.value, "prompt": len(q.prompt_ids),
                          "output": len(q.output_ids), "blocks": len(tables.get(q.seq_id, [])),
                          "preemptions": q.num_preemptions} for q in seqs],
            "stats": {"running": s.num_running, "waiting": s.num_waiting, "preemptions": s.num_preemptions_total,
                      "prefix_query": s.prefix_cache_query_tokens, "prefix_hit": s.prefix_cache_hit_tokens,
                      "spec_proposed": s.spec_proposed_tokens, "spec_accepted": s.spec_accepted_tokens},
        }

    @torch.no_grad()
    def step(self) -> list[RequestOutput]:
        outputs, self._finished_now = self._finished_now, []
        plan = self.scheduler.schedule()
        if not plan.scheduled:
            return outputs
        self._copy_blocks(plan.copies)
        self.target_forwards += 1
        if self.draft is not None and all(s.num_tokens == 1 and s.start == s.seq.num_tokens - 1
                                          for s in plan.scheduled):
            return outputs + self._speculative_step([s.seq for s in plan.scheduled])

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
                            PagedContext(self.cache, meta, self.backend), logits_indices=rows)
        if self.draft is not None:
            self._draft_catch_up([s.seq for s in plan.scheduled])
        for s in plan.scheduled:  # newly completed full blocks join the prefix cache
            self._register(s.seq)
        for (_, seq), token in zip(to_sample, self._sample_all(logits, [s for _, s in to_sample]), strict=True):
            outputs.append(self._process_token(seq, token))
        return outputs

    # ---- speculative decoding ------------------------------------------------------------------
    def _draft_catch_up(self, seqs: list[Sequence]) -> None:
        """Run the draft over every token the target just computed that the draft hasn't seen, so both
        caches hold the same positions. No logits are needed."""
        ids: list[int] = []
        positions: list[int] = []
        live: list[Sequence] = []
        for seq in seqs:
            if seq.draft_computed < seq.num_computed:
                ids += seq.tokens[seq.draft_computed:seq.num_computed]
                positions += range(seq.draft_computed, seq.num_computed)
                live.append(seq)
        if not live:
            return
        meta = build_metadata(self.bm, [s.seq_id for s in live], [s.num_computed - s.draft_computed for s in live],
                              self.device, seq_lens=[s.num_computed for s in live])
        assert self.draft is not None and self.draft_cache is not None
        self.draft(torch.tensor(ids, device=self.device), torch.tensor(positions, device=self.device),
                   PagedContext(self.draft_cache, meta, self.backend), logits_indices=torch.empty(0, dtype=torch.long,
                                                                                    device=self.device))
        for seq in live:
            seq.draft_computed = seq.num_computed

    def _speculative_step(self, seqs: list[Sequence]) -> list[RequestOutput]:
        """Every sequence is decoding: the scheduler reserved one slot for its last token (position L-1).
        Reserve k more, draft k tokens, score [last, d1..dk] with the target in one pass, keep what verifies."""
        assert self.draft is not None and self.draft_cache is not None
        ks: list[int] = []
        for seq in seqs:
            remaining = seq.params.max_tokens - len(seq.output_ids)  # >= 1, or it would have finished
            k = min(self.k, remaining - 1)  # all k accepted + the bonus token must fit in max_tokens
            if k > 0 and self.bm.can_append(seq.seq_id, k):
                self._copy_blocks(self.bm.append_slots(seq.seq_id, k))
            else:
                k = 0
            ks.append(k)
        failed: set[int] = set()

        # Draft: k passes. Pass 0 also feeds any tokens the draft hasn't seen yet (it can trail by one).
        drafts: list[list[int]] = [[] for _ in seqs]
        draft_probs: list[list[torch.Tensor]] = [[] for _ in seqs]
        for i in range(max(ks)):
            active = [b for b, k in enumerate(ks) if k > i]
            ids: list[int] = []
            positions: list[int] = []
            qlens: list[int] = []
            for b in active:
                seq, L = seqs[b], seqs[b].num_tokens
                if i == 0:
                    new = seq.tokens[seq.draft_computed:L]
                    positions += range(seq.draft_computed, L)
                else:
                    new = [drafts[b][-1]]
                    positions.append(L - 1 + i)
                ids += new
                qlens.append(len(new))
            visible = [seqs[b].num_tokens + i for b in active]
            meta = build_metadata(self.bm, [seqs[b].seq_id for b in active], qlens, self.device, seq_lens=visible)
            rows = torch.tensor(list(itertools.accumulate(qlens)), device=self.device) - 1
            logits = self.draft(torch.tensor(ids, device=self.device), torch.tensor(positions, device=self.device),
                                PagedContext(self.draft_cache, meta, self.backend), logits_indices=rows)
            results = self._propose(logits, [seqs[b] for b in active], [drafts[b] for b in active],
                                    [draft_probs[b] for b in active])
            for b, ok in zip(active, results, strict=True):
                if not ok:  # stop drafting for it and give back the unused slots; it finishes with "error"
                    failed.add(b)
                    self.bm.rollback(seqs[b].seq_id, ks[b] - len(drafts[b]))
                    ks[b] = len(drafts[b])
        for seq, k in zip(seqs, ks, strict=True):
            if k:
                seq.draft_computed = seq.num_tokens - 1 + k  # it saw last, d1 .. d(k-1)

        # Target: one pass over [last, d1..dk] for every sequence.
        ids, positions = [], []
        for seq, d in zip(seqs, drafts, strict=True):
            L = seq.num_tokens
            ids += [seq.tokens[-1], *d]
            positions += range(L - 1, L + len(d))
        meta = build_metadata(self.bm, [s.seq_id for s in seqs], [k + 1 for k in ks], self.device)
        logits = self.model(torch.tensor(ids, device=self.device), torch.tensor(positions, device=self.device),
                            PagedContext(self.cache, meta, self.backend))

        outputs: list[RequestOutput] = []
        offsets = [0, *itertools.accumulate(k + 1 for k in ks)]
        greedy_rows = logits.argmax(dim=-1).tolist() if any(s.params.temperature == 0 for s in seqs) else []
        for b, (seq, k) in enumerate(zip(seqs, ks, strict=True)):
            lo, hi = offsets[b], offsets[b + 1]
            try:
                if b in failed:
                    raise RuntimeError("draft proposal failed")
                if seq.params.temperature == 0:
                    target = greedy_rows[lo:hi]
                    j = next((i for i in range(k) if drafts[b][i] != target[i]), k)
                    token: int | None = target[j]
                else:
                    p = torch.stack([probs(r, seq.params) for r in logits[lo:hi].cpu()])
                    q = torch.stack(draft_probs[b]) if k else None
                    j, token = verify(drafts[b], q, p, greedy=False, generator=seq.generator)
            except Exception:  # one bad request must not take the whole batch down
                j, token = 0, None
            self.spec_proposed += k
            self.spec_accepted += j
            L = seq.num_tokens
            self.bm.rollback(seq.seq_id, k - j)  # forget the rejected draft positions in both caches
            seq.num_computed = L + j
            seq.draft_computed = min(seq.draft_computed, seq.num_computed)
            out = RequestOutput(seq.request_id)
            for t in [*drafts[b][:j], token]:
                self._process_token(seq, t, out)
                if out.finished:
                    break
            if not out.finished:
                self._register(seq)
            outputs.append(out)
        return outputs

    def _propose(self, logits: torch.Tensor, seqs: list[Sequence], drafts: list[list[int]],
                 draft_probs: list[list[torch.Tensor]]) -> list[bool]:
        """Draw one draft token per sequence. Sampling requests keep the draft distribution q for verify.
        Returns, per sequence, whether it worked: one bad request must not take the whole batch down."""
        greedy = [s.params.temperature == 0 for s in seqs]
        argmax = logits.argmax(dim=-1).tolist() if any(greedy) else []
        rows = logits.cpu() if not all(greedy) else None
        ok = [True] * len(seqs)
        for i, seq in enumerate(seqs):
            if greedy[i]:
                drafts[i].append(argmax[i])
                continue
            assert rows is not None
            try:
                q = probs(rows[i], seq.params)
                token = int(torch.multinomial(q, 1, generator=seq.generator))
            except Exception:
                ok[i] = False
                continue
            drafts[i].append(token)
            draft_probs[i].append(q)
        return ok

    def _register(self, seq: Sequence) -> None:
        """Add newly completed full blocks to the prefix cache, once both caches hold their K/V."""
        done = seq.num_computed if self.draft is None else min(seq.num_computed, seq.draft_computed)
        if seq.seq_id in self.bm.seq_ids and self.bm.needs_register(seq.seq_id, done):
            self.bm.register_computed(seq.seq_id, seq.tokens, done)

    def _copy_blocks(self, copies: list[tuple[int, int]]) -> None:
        if copies:
            self.cache.copy_blocks(copies)
            if self.draft_cache is not None:
                self.draft_cache.copy_blocks(copies)

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

    def _process_token(self, seq: Sequence, token: int | None, out: RequestOutput | None = None) -> RequestOutput:
        """Append one token to the sequence and to `out` (a new output if None); finish if it stops."""
        out = out if out is not None else RequestOutput(seq.request_id)
        if token is None:
            return self._finish(seq, out, "error")
        params = seq.params
        stop_ids = set(params.stop_token_ids)
        if not params.ignore_eos:
            stop_ids.update(self.config.eos_token_ids)
        if token in stop_ids:
            return self._finish(seq, out, "stop")

        seq.output_ids.append(token)
        out.new_token_ids.append(token)
        if seq.streamer is not None:
            text, hit = seq.streamer.push(token)
            out.new_text += text
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

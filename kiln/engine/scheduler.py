"""Iteration-level scheduling: every engine step decides afresh which sequences run and how far.

Rules, in order:
1. Running sequences go first, oldest first. Each gets its next token (decode) or its next prefill
   chunk, within the step's token budget.
2. If a running sequence needs a KV block and none is free, the most recently admitted running
   sequence is preempted: its blocks are freed, it rejoins the *front* of the queue, and it will
   recompute its K/V later. That may be the sequence itself.
3. Waiting sequences are admitted strictly first come, first served (a big request is never starved
   by small ones jumping ahead), while there is token budget, a free sequence slot and memory. A long
   prompt is chunked to the remaining budget, so it never stalls everyone else's decoding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from kiln.cache import BlockManager
from kiln.engine.sequence import SeqStatus, Sequence


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_tokens: int  # tokens to compute this step
    start: int  # position of the first of them


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    copies: list[tuple[int, int]] = field(default_factory=list)  # copy-on-write block copies to apply
    preempted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)


class Scheduler:
    def __init__(self, bm: BlockManager, max_num_seqs: int, max_batched_tokens: int):
        self.bm = bm
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        capacity = self.bm.num_blocks * self.bm.block_size
        if len(seq.prompt_ids) + seq.params.max_tokens > capacity:
            raise ValueError(f"request needs up to {len(seq.prompt_ids) + seq.params.max_tokens} tokens of "
                             f"KV cache but the whole pool holds {capacity}")
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.max_batched_tokens

        idx = 0
        while idx < len(self.running) and budget > 0:
            seq = self.running[idx]
            n = min(seq.num_tokens - seq.num_computed, budget)
            while not self.bm.can_append(seq.seq_id, n):
                victim = self.running[-1]  # newest; never one already scheduled this step
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    break
            if seq.status is SeqStatus.WAITING:  # it preempted itself; everything after it is gone too
                break
            out.copies += self.bm.append_slots(seq.seq_id, n)
            out.scheduled.append(ScheduledSeq(seq, n, seq.num_computed))
            budget -= n
            idx += 1

        # Don't re-admit in the same step we had to preempt: memory is the bottleneck right now.
        while (not out.preempted and self.waiting and budget > 0
               and len(self.running) < self.max_num_seqs):
            seq = self.waiting[0]
            n = min(seq.num_tokens - seq.num_computed, budget)
            if not self.bm.can_allocate(n):
                break  # strict FIFO: don't let smaller requests jump the queue
            self.waiting.popleft()
            self.bm.allocate(seq.seq_id, n)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            out.scheduled.append(ScheduledSeq(seq, n, 0))
            budget -= n
        return out

    def finish(self, seq: Sequence, reason: str) -> None:
        self.running.remove(seq)
        self.bm.free(seq.seq_id)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason

    def abort(self, seq_id: int) -> Sequence | None:
        for queue in (self.running, self.waiting):
            for seq in queue:
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    if seq_id in self.bm.seq_ids:
                        self.bm.free(seq_id)
                    seq.status = SeqStatus.FINISHED
                    seq.finish_reason = "abort"
                    return seq
        return None

    def _preempt(self, seq: Sequence) -> None:
        """Preemption by recomputation: drop the K/V and redo the prefill (prompt + output) later."""
        self.running.remove(seq)
        self.bm.free(seq.seq_id)
        seq.num_computed = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

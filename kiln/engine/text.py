"""Turns generated token ids into final text deltas, honouring stop strings."""

from __future__ import annotations

from kiln.tokenizer import IncrementalDetokenizer, Tokenizer


class TextStreamer:
    """Streams text without ever emitting half a character or the start of a stop string.

    Text that could be the beginning of a stop string is held back until it's clear either way. A stop
    match can therefore never begin inside text that was already emitted.
    """

    def __init__(self, tokenizer: Tokenizer, stop: list[str]):
        self._detok = IncrementalDetokenizer(tokenizer)
        self._stop = stop
        self._pending = ""

    def push(self, token_id: int) -> tuple[str, bool]:
        """Returns (text that is now final, whether a stop string was hit)."""
        self._pending += self._detok.push(token_id)
        hits = [i for s in self._stop if (i := self._pending.find(s)) != -1]
        if hits:
            out, self._pending = self._pending[:min(hits)], ""
            return out, True
        keep = self._partial_stop_suffix()
        cut = len(self._pending) - keep
        out, self._pending = self._pending[:cut], self._pending[cut:]
        return out, False

    def flush(self) -> str:
        """Generation ended without a stop hit: held-back text was real text after all."""
        out, self._pending = self._pending, ""
        return out

    def _partial_stop_suffix(self) -> int:
        """Length of the longest suffix of pending text that is a proper prefix of some stop string."""
        best = 0
        for s in self._stop:
            for n in range(min(len(s) - 1, len(self._pending)), best, -1):
                if self._pending.endswith(s[:n]):
                    best = n
                    break
        return best

"""Live view of the paged KV pool, for the visualiser page (/viz).

The engine thread takes a snapshot after a step, at most every `min_interval` seconds (always when a
request finishes, so the final state is never skipped). HTTP handlers only read the latest snapshot,
so watching the pool never slows the engine down by more than one snapshot per interval.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from kiln.engine.engine import EngineStats, LLMEngine, RequestOutput


class KVView:
    def __init__(self, engine: LLMEngine, min_interval: float = 0.1):
        self.engine = engine
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._snapshot = engine.kv_snapshot()
        self._version = 0
        self._last = 0.0

    def on_step(self, outputs: list[RequestOutput], _: EngineStats) -> None:
        """Runs on the engine thread after every step."""
        now = time.monotonic()
        if now - self._last < self.min_interval and not any(o.finished for o in outputs):
            return
        snapshot = self.engine.kv_snapshot()
        with self._lock:
            self._snapshot, self._version, self._last = snapshot, self._version + 1, now

    def latest(self) -> tuple[int, dict[str, Any]]:
        with self._lock:
            return self._version, self._snapshot

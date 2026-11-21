from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_STOP_STRINGS = 16
MAX_STOP_LENGTH = 256


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0 means greedy (argmax)
    top_k: int = 0  # 0 = off
    top_p: float = 1.0  # 1.0 = off
    seed: int | None = None
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and math.isfinite(self.top_p)):
            raise ValueError("temperature and top_p must be finite numbers")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens must be >= 0, got {self.max_tokens}")
        if any(s == "" for s in self.stop):
            raise ValueError("stop strings must be non-empty")
        # Stop strings are matched on the shared engine thread every token: bound the work per request.
        if len(self.stop) > MAX_STOP_STRINGS or any(len(s) > MAX_STOP_LENGTH for s in self.stop):
            raise ValueError(f"at most {MAX_STOP_STRINGS} stop strings of at most {MAX_STOP_LENGTH} characters")

"""OpenAI-compatible request schemas (the subset Kiln supports) and their mapping to SamplingParams."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kiln.sampling import SamplingParams


class StreamOptions(BaseModel):
    include_usage: bool = False


MAX_API_STOP_STRINGS = 4  # OpenAI's limit


class _Common(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    model: str | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # extension
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = 1
    ignore_eos: bool = False  # extension, used by benchmarks

    def sampling_params(self, max_tokens: int) -> SamplingParams:
        """Raises ValueError for anything Kiln can't do or that's out of range."""
        if self.n != 1:
            raise ValueError("only n=1 is supported")
        stop = [self.stop] if isinstance(self.stop, str) else list(self.stop or [])
        if len(stop) > MAX_API_STOP_STRINGS:
            raise ValueError(f"at most {MAX_API_STOP_STRINGS} stop strings are allowed")
        return SamplingParams(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, seed=self.seed,
                              max_tokens=max_tokens, stop=stop, ignore_eos=self.ignore_eos)

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)


class CompletionRequest(_Common):
    prompt: str | list[int]
    max_tokens: int = 16  # OpenAI's default for completions


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(_Common):
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None

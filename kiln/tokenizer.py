"""Tokenizer wrapper, chat templates, and streaming-safe incremental detokenisation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer as _HFTokenizer

_SPECIAL_KEYS = ("bos_token", "eos_token", "unk_token", "pad_token")


def _token_str(value: Any) -> str | None:
    if isinstance(value, dict):  # older configs store {"content": "<s>", ...}
        return value.get("content")
    return value


class Tokenizer:
    def __init__(self, tokenizer: _HFTokenizer, chat_template: str | None, special: dict[str, str]):
        self._tok = tokenizer
        self._special = special
        self._template = _compile(chat_template) if chat_template else None
        eos = special.get("eos_token")
        self.eos_token_id: int | None = tokenizer.token_to_id(eos) if eos else None

    @classmethod
    def from_hf_dir(cls, path: Path | str) -> Tokenizer:
        path = Path(path)
        tok = _HFTokenizer.from_file(str(path / "tokenizer.json"))
        cfg_path = path / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        special = {k: s for k in _SPECIAL_KEYS if (s := _token_str(cfg.get(k)))}

        template = cfg.get("chat_template")
        if isinstance(template, list):  # named templates: use "default"
            template = next((t["template"] for t in template if t.get("name") == "default"), None)
        jinja_file = path / "chat_template.jinja"
        if template is None and jinja_file.exists():
            template = jinja_file.read_text()
        return cls(tok, template, special)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        if self._template is None:
            raise ValueError("this tokenizer has no chat template")
        return self._template.render(messages=messages, add_generation_prompt=add_generation_prompt,
                                     **self._special)


def _compile(source: str) -> jinja2.Template:
    # Same environment settings Hugging Face uses, so templates render byte-identically.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

    def raise_exception(message: str) -> None:
        raise jinja2.TemplateError(message)

    env.filters["tojson"] = lambda x, indent=None, **_: json.dumps(x, ensure_ascii=False, indent=indent)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(source)


class IncrementalDetokenizer:
    """Turns a stream of token ids into text deltas without ever emitting half a character.

    Byte-level BPE can split one UTF-8 character (an emoji, say) across several tokens; decoding a
    partial sequence yields U+FFFD. We keep a small window of recent tokens, re-decode it on each push,
    and only release text once the decode no longer ends in U+FFFD. Re-decoding a window (rather than
    single tokens) also keeps tokenizer-dependent spacing right.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self._ids: list[int] = []
        self._prefix_offset = 0  # start of the window we re-decode
        self._read_offset = 0  # tokens before this have been emitted

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        prefix_text = self._tok.decode(self._ids[self._prefix_offset:self._read_offset])
        new_text = self._tok.decode(self._ids[self._prefix_offset:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)
            return new_text[len(prefix_text):]
        return ""

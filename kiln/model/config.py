"""Model hyper-parameters, read from a Hugging Face `config.json`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ARCHES: dict[str, Literal["qwen2", "llama"]] = {
    "Qwen2ForCausalLM": "qwen2",
    "LlamaForCausalLM": "llama",
}


class UnsupportedModelError(ValueError):
    """The checkpoint uses an architecture or feature Kiln does not implement."""


@dataclass(frozen=True)
class ModelConfig:
    arch: Literal["qwen2", "llama"]
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    eos_token_ids: tuple[int, ...]

    @classmethod
    def from_hf_dir(cls, path: Path | str) -> ModelConfig:
        path = Path(path)
        raw = json.loads((path / "config.json").read_text())

        archs = raw.get("architectures") or []
        arch = next((_ARCHES[a] for a in archs if a in _ARCHES), None)
        if arch is None:
            raise UnsupportedModelError(f"unsupported architecture {archs}; Kiln supports {list(_ARCHES)}")
        if raw.get("rope_scaling"):
            raise UnsupportedModelError(f"rope_scaling={raw['rope_scaling']!r} is not implemented")

        num_heads = raw["num_attention_heads"]
        head_dim = raw.get("head_dim") or raw["hidden_size"] // num_heads
        attention_bias = True if arch == "qwen2" else bool(raw.get("attention_bias", False))

        eos: list[int] = []
        sources = [raw.get("eos_token_id")]
        gen_path = path / "generation_config.json"
        if gen_path.exists():
            sources.append(json.loads(gen_path.read_text()).get("eos_token_id"))
        for src in sources:
            ids = src if isinstance(src, list) else [src] if src is not None else []
            eos.extend(i for i in ids if i not in eos)

        return cls(
            arch=arch,
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_layers=raw["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=raw.get("num_key_value_heads", num_heads),
            head_dim=head_dim,
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            max_position_embeddings=raw["max_position_embeddings"],
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            attention_bias=attention_bias,
            eos_token_ids=tuple(eos),
        )

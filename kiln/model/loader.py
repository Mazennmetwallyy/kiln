"""Load Hugging Face safetensors checkpoints into a Kiln Transformer."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from kiln.model.config import ModelConfig
from kiln.model.layers import RotaryEmbedding
from kiln.model.transformer import Transformer


class WeightLoadError(RuntimeError):
    """The checkpoint's tensors don't match the model's parameters."""


def load_model(path: Path | str, dtype: torch.dtype = torch.float32,
               device: str = "cpu") -> tuple[Transformer, ModelConfig]:
    path = Path(path)
    config = ModelConfig.from_hf_dir(path)

    # Build on the meta device (no memory), then attach real tensors, so we never hold two copies.
    with torch.device("meta"):
        model = Transformer(config)
    expected = set(model.state_dict().keys())
    if config.tie_word_embeddings:
        expected.discard("lm_head.weight")

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise WeightLoadError(f"no *.safetensors files in {path}")
    state: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as fh:
            for name in fh.keys():  # noqa: SIM118 (safe_open handle is not a dict)
                if name.endswith("rotary_emb.inv_freq"):
                    continue  # old checkpoints store this buffer; we recompute it
                if config.tie_word_embeddings and name == "lm_head.weight":
                    continue
                state[name] = fh.get_tensor(name).to(dtype=dtype, device=device)

    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected)
    if missing or unexpected:
        raise WeightLoadError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")

    model.load_state_dict(state, strict=False, assign=True)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    rope = RotaryEmbedding(config.head_dim, config.rope_theta, config.max_position_embeddings)
    model.rotary = rope.to(device)
    return model.eval(), config

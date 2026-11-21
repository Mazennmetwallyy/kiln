"""The Qwen2/Llama decoder. Module names mirror Hugging Face so checkpoints load one to one."""

from __future__ import annotations

from torch import Tensor, nn

from kiln.model.attention import Attention
from kiln.model.config import ModelConfig
from kiln.model.kv_cache import ContiguousKVCache
from kiln.model.layers import RMSNorm, RotaryEmbedding, SwiGLU


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor, positions: Tensor, cache: ContiguousKVCache, layer_idx: int,
                rope: RotaryEmbedding) -> Tensor:
        # Pre-norm residual blocks: x + attn(norm(x)), then x + mlp(norm(x)).
        x = x + self.self_attn(self.input_layernorm(x), positions, cache, layer_idx, rope)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Body(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = _Body(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta, config.max_position_embeddings)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: Tensor, positions: Tensor, cache: ContiguousKVCache) -> Tensor:
        """input_ids, positions: [T] (flat, no batch dim). Returns fp32 logits [T, vocab]."""
        x = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, positions, cache, i, self.rotary)
        return self.lm_head(self.model.norm(x)).float()

"""Tiny randomly initialised Hugging Face models, saved to disk, for CI-friendly reference tests."""

from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

_COMMON = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "vocab_size": 256, "max_position_embeddings": 128, "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6, "eos_token_id": 1, "bos_token_id": 0,
}


def make_tiny_hf_model(arch: str, tmp_path: Path, tied: bool = True, seed: int = 0):
    torch.manual_seed(seed)
    if arch == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(tie_word_embeddings=tied, **_COMMON))
    elif arch == "llama":
        model = LlamaForCausalLM(LlamaConfig(tie_word_embeddings=tied, **_COMMON))
    else:
        raise ValueError(arch)
    model = model.float().eval()
    # Random init leaves norm gains at 1 and biases at 0; perturb them so tests exercise the weights.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name or name.endswith("bias"):
                p.add_(torch.randn_like(p) * 0.1)
    model.save_pretrained(tmp_path, safe_serialization=True)
    return tmp_path, model

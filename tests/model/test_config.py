import json

import pytest

from kiln.model.config import ModelConfig, UnsupportedModelError

QWEN = {
    "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 896,
    "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14,
    "num_key_value_heads": 2, "vocab_size": 151936, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6,
    "max_position_embeddings": 32768, "tie_word_embeddings": True, "eos_token_id": 151645,
    "rope_scaling": None,
}
SMOL = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 576,
    "intermediate_size": 1536, "num_hidden_layers": 30, "num_attention_heads": 9,
    "num_key_value_heads": 3, "vocab_size": 49152, "rope_theta": 100000.0, "rms_norm_eps": 1e-5,
    "max_position_embeddings": 8192, "tie_word_embeddings": True, "eos_token_id": 2,
}


def write(tmp_path, cfg, gen=None):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    if gen is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(gen))
    return tmp_path


def test_qwen2_config(tmp_path):
    c = ModelConfig.from_hf_dir(write(tmp_path, QWEN))
    assert c.arch == "qwen2"
    assert (c.hidden_size, c.num_layers, c.num_heads, c.num_kv_heads) == (896, 24, 14, 2)
    assert c.head_dim == 64
    assert c.attention_bias is True
    assert c.tie_word_embeddings is True
    assert c.rope_theta == 1000000.0 and c.rms_norm_eps == 1e-6
    assert c.eos_token_ids == (151645,)


def test_llama_config(tmp_path):
    c = ModelConfig.from_hf_dir(write(tmp_path, SMOL))
    assert c.arch == "llama"
    assert c.head_dim == 64
    assert c.attention_bias is False
    assert c.eos_token_ids == (2,)


def test_eos_list_and_generation_config(tmp_path):
    cfg = dict(QWEN, eos_token_id=[151645, 151643])
    c = ModelConfig.from_hf_dir(write(tmp_path, cfg, gen={"eos_token_id": [151643, 7]}))
    assert c.eos_token_ids == (151645, 151643, 7)


def test_rope_scaling_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError):
        ModelConfig.from_hf_dir(write(tmp_path, dict(SMOL, rope_scaling={"type": "linear", "factor": 2.0})))


def test_unsupported_arch(tmp_path):
    with pytest.raises(UnsupportedModelError):
        ModelConfig.from_hf_dir(write(tmp_path, dict(SMOL, architectures=["GPT2LMHeadModel"])))


def test_rope_parameters_format(tmp_path):
    # transformers >= 5 writes rope settings as a nested dict instead of a top-level rope_theta.
    cfg = {k: v for k, v in SMOL.items() if k != "rope_theta"}
    cfg["rope_parameters"] = {"rope_theta": 500000.0, "rope_type": "default"}
    assert ModelConfig.from_hf_dir(write(tmp_path, cfg)).rope_theta == 500000.0


def test_rope_parameters_scaled_type_rejected(tmp_path):
    cfg = dict(SMOL, rope_parameters={"rope_theta": 500000.0, "rope_type": "llama3", "factor": 8.0})
    with pytest.raises(UnsupportedModelError):
        ModelConfig.from_hf_dir(write(tmp_path, cfg))


def test_missing_rope_theta_rejected(tmp_path):
    cfg = {k: v for k, v in SMOL.items() if k != "rope_theta"}
    with pytest.raises(UnsupportedModelError):
        ModelConfig.from_hf_dir(write(tmp_path, cfg))


def test_sliding_window_rejected(tmp_path):
    with pytest.raises(UnsupportedModelError):
        ModelConfig.from_hf_dir(write(tmp_path, dict(QWEN, use_sliding_window=True, sliding_window=4096)))

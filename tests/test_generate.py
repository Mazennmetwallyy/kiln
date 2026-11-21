from pathlib import Path

import pytest
import torch
from torch import nn

from kiln.generate import generate, stream_generate
from kiln.model.config import ModelConfig
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams
from kiln.tokenizer import Tokenizer
from tests.model.tiny import make_tiny_hf_model

PROMPT = [5, 17, 42, 99, 3, 250, 7]


@pytest.mark.parametrize("arch", ["qwen2", "llama"])
def test_greedy_matches_hf_generate(arch, tmp_path):
    path, hf_model = make_tiny_hf_model(arch, tmp_path)
    model, config = load_model(path)
    hf_model.generation_config.eos_token_id = None
    with torch.no_grad():
        ref = hf_model.generate(torch.tensor([PROMPT]), do_sample=False, max_new_tokens=32,
                                pad_token_id=0)[0, len(PROMPT):].tolist()
    out = generate(model, config, PROMPT, SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))
    assert out.token_ids == ref
    assert out.finish_reason == "length"


def test_seeded_sampling_reproducible(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    p = SamplingParams(temperature=1.0, top_p=0.9, seed=3, max_tokens=20, ignore_eos=True)
    assert generate(model, config, PROMPT, p).token_ids == generate(model, config, PROMPT, p).token_ids


def test_context_overflow_raises(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)  # max_position_embeddings = 128
    model, config = load_model(path)
    with pytest.raises(ValueError, match="context"):
        generate(model, config, list(range(100)), SamplingParams(max_tokens=100))


def test_max_context_argument_overrides(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    with pytest.raises(ValueError, match="context"):
        generate(model, config, PROMPT, SamplingParams(max_tokens=10), max_context=12)


def test_max_tokens_zero(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    out = generate(model, config, PROMPT, SamplingParams(max_tokens=0))
    assert out.token_ids == [] and out.text == "" and out.finish_reason == "length"


class _ForceToken(nn.Module):
    """Wraps a model so one token id always wins the argmax."""

    def __init__(self, inner, token_id):
        super().__init__()
        self.inner, self.token_id = inner, token_id

    def forward(self, ids, positions, cache, logits_indices=None):
        logits = self.inner(ids, positions, cache, logits_indices=logits_indices)
        logits[:, self.token_id] += 1e4
        return logits


def test_eos_first_token(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    eos = config.eos_token_ids[0]
    out = generate(_ForceToken(model, eos), config, PROMPT, SamplingParams(temperature=0.0, max_tokens=10))
    assert out.token_ids == [] and out.finish_reason == "stop"


def test_stop_token_ids(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    out = generate(_ForceToken(model, 33), config, PROMPT,
                   SamplingParams(temperature=0.0, max_tokens=10, stop_token_ids=[33]))
    assert out.token_ids == [] and out.finish_reason == "stop"


def test_stop_strings_need_tokenizer(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    with pytest.raises(ValueError, match="tokenizer"):
        generate(model, config, PROMPT, SamplingParams(stop=["\n"]))


def test_empty_prompt_raises(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    with pytest.raises(ValueError):
        generate(model, config, [], SamplingParams())


# ---- stop strings and streaming (need a real tokenizer) ----


FIXTURE = Path(__file__).parent / "fixtures" / "smollm2"


class _Scripted(nn.Module):
    """A fake model that emits a fixed token sequence, one per forward call."""

    def __init__(self, script, vocab):
        super().__init__()
        self.script, self.vocab, self.calls = script, vocab, 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, ids, positions, cache, logits_indices=None):
        logits = torch.zeros(len(ids), self.vocab)
        logits[-1, self.script[min(self.calls, len(self.script) - 1)]] = 1.0
        self.calls += 1
        return logits if logits_indices is None else logits[logits_indices]


def _fake_config(vocab):
    return ModelConfig(arch="llama", vocab_size=vocab, hidden_size=4, intermediate_size=4, num_layers=1,
                       num_heads=1, num_kv_heads=1, head_dim=4, rope_theta=1e4, rms_norm_eps=1e-6,
                       max_position_embeddings=512, tie_word_embeddings=True, attention_bias=False,
                       eos_token_ids=(2,))


def test_stop_string_across_tokens():
    tok = Tokenizer.from_hf_dir(FIXTURE)
    script = tok.encode("Hello world, how are you", add_special_tokens=False)
    # Pick a stop string that starts inside one token and ends inside the next.
    assert tok.decode(script[:1]) == "Hello" and tok.decode(script[1:2]) == " world"
    params = SamplingParams(temperature=0.0, max_tokens=20, stop=["o w"])
    model = _Scripted(script, 49152)
    out = generate(model, _fake_config(49152), [1], params, tokenizer=tok)
    assert out.text == "Hell"
    assert out.finish_reason == "stop"


def test_stream_deltas_hold_back_partial_stop():
    tok = Tokenizer.from_hf_dir(FIXTURE)
    script = tok.encode("Hello world, how are you", add_special_tokens=False)
    params = SamplingParams(temperature=0.0, max_tokens=20, stop=["o w"])
    deltas = list(stream_generate(_Scripted(script, 49152), _fake_config(49152), [1], params, tokenizer=tok))
    assert "".join(deltas) == "Hell"
    assert all("o w" not in d for d in deltas)


def test_stream_matches_generate_without_stop():
    tok = Tokenizer.from_hf_dir(FIXTURE)
    script = [*tok.encode("Héllo 👋🏽 world", add_special_tokens=False), 2]
    params = SamplingParams(temperature=0.0, max_tokens=40)
    cfg = _fake_config(49152)
    full = generate(_Scripted(script, 49152), cfg, [1], params, tokenizer=tok)
    deltas = list(stream_generate(_Scripted(script, 49152), cfg, [1], params, tokenizer=tok))
    assert "".join(deltas) == full.text == "Héllo 👋🏽 world"
    assert full.finish_reason == "stop"


def test_max_context_above_model_limit_is_capped(tmp_path):
    # Review finding: max_context must not lift the model's own position limit (128 for the tiny model).
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    with pytest.raises(ValueError, match="context"):
        generate(model, config, list(range(120)), SamplingParams(max_tokens=20), max_context=1000)


def test_max_context_zero_is_not_unset(tmp_path):
    path, _ = make_tiny_hf_model("llama", tmp_path)
    model, config = load_model(path)
    with pytest.raises(ValueError, match="context"):
        generate(model, config, PROMPT, SamplingParams(max_tokens=1), max_context=0)

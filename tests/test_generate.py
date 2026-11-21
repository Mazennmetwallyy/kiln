import pytest
import torch
from torch import nn

from kiln.generate import generate
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams
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

    def forward(self, ids, positions, cache):
        logits = self.inner(ids, positions, cache)
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

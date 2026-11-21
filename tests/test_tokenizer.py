from pathlib import Path

import pytest
from transformers import AutoTokenizer

from kiln.tokenizer import IncrementalDetokenizer, Tokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "smollm2"
TEXTS = ["Hello, world!", "héllo 👋🏽 世界", "def f(x):\n    return x * 2\n", "naïve café — résumé"]


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_hf_dir(FIXTURE)


@pytest.mark.parametrize("text", TEXTS)
def test_encode_decode_roundtrip(tok, text):
    assert tok.decode(tok.encode(text, add_special_tokens=False)) == text


def test_encode_matches_hf(tok):
    hf = AutoTokenizer.from_pretrained(FIXTURE)
    for text in TEXTS:
        assert tok.encode(text) == hf.encode(text)


def test_eos_token_id(tok):
    assert tok.eos_token_id == 2  # <|im_end|>


@pytest.mark.parametrize("text", TEXTS)
def test_incremental_detok_multibyte(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    detok = IncrementalDetokenizer(tok)
    deltas = [detok.push(i) for i in ids]
    assert all("�" not in d for d in deltas)
    assert "".join(deltas) == tok.decode(ids)


def test_emoji_is_split_across_tokens(tok):
    # Guard: the multibyte test is only meaningful if some character really spans several tokens.
    ids = tok.encode("👋🏽", add_special_tokens=False)
    assert len(ids) > 1 and any("�" in tok.decode([i]) for i in ids)


@pytest.mark.parametrize("add_gen", [True, False])
def test_chat_template_matches_hf(tok, add_gen):
    msgs = [{"role": "user", "content": "Hi there"}, {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Name a planet"}]
    hf = AutoTokenizer.from_pretrained(FIXTURE)
    expected = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen)
    assert tok.apply_chat_template(msgs, add_generation_prompt=add_gen) == expected

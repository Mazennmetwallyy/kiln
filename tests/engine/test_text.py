from pathlib import Path

from kiln.engine.text import TextStreamer
from kiln.tokenizer import Tokenizer

FIXTURE = Path(__file__).parents[1] / "fixtures" / "smollm2"


def run(text, stop):
    tok = Tokenizer.from_hf_dir(FIXTURE)
    s = TextStreamer(tok, stop)
    out, hit = [], False
    for t in tok.encode(text, add_special_tokens=False):
        delta, hit = s.push(t)
        out.append(delta)
        if hit:
            break
    if not hit:
        out.append(s.flush())
    return out, hit


def test_stop_across_tokens_trims_and_holds_back():
    out, hit = run("Hello world, how are you", ["o w"])
    assert hit and "".join(out) == "Hell" and all("o w" not in d for d in out)


def test_no_stop_flushes_held_back_prefix():
    out, hit = run("Hello", ["lo!"])  # "lo" is held back as a possible stop prefix, then flushed
    assert not hit and "".join(out) == "Hello"


def test_multibyte_never_emits_replacement_char():
    out, hit = run("héllo 👋🏽 世界", [])
    assert not hit and "".join(out) == "héllo 👋🏽 世界" and all("�" not in d for d in out)

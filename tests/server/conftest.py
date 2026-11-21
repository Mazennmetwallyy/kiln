import shutil
from pathlib import Path

import pytest

from tests.model.tiny import make_tiny_hf_model

FIXTURE = Path(__file__).parents[1] / "fixtures" / "smollm2"


@pytest.fixture(scope="session")
def tiny_text_model_dir(tmp_path_factory):
    """A tiny random Llama with SmolLM2's real tokenizer, vocab and chat template. Outputs are gibberish,
    but every text path (chat template, detokenisation, stop strings) is real."""
    path = tmp_path_factory.mktemp("tiny_text")
    make_tiny_hf_model("llama", path, vocab_size=49152, eos_token_id=2, bos_token_id=1,
                       max_position_embeddings=512)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(FIXTURE / f, path / f)
    return path

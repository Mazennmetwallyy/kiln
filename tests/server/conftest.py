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


@pytest.fixture(scope="module")
def server(tiny_text_model_dir):
    """A real uvicorn server on a free port, serving the tiny model. Yields (base_url, async_engine)."""
    import socket
    import threading
    import time

    import uvicorn

    from kiln.engine.engine import LLMEngine
    from kiln.model.loader import load_model
    from kiln.server.app import create_app
    from kiln.server.async_engine import AsyncEngine
    from kiln.tokenizer import Tokenizer

    model, config = load_model(tiny_text_model_dir)
    tok = Tokenizer.from_hf_dir(tiny_text_model_dir)
    ae = AsyncEngine(LLMEngine(model, config, tok, num_blocks=128, block_size=16))
    app = create_app(ae, tok, model_name="tiny")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", ae
    srv.should_exit = True
    thread.join(timeout=10)

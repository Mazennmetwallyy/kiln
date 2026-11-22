"""The hosted Kiln demo: the OpenAI-compatible server, the live KV-cache view, and a chat page.

Runs Qwen2.5-0.5B-Instruct on CPU with the PyTorch attention path (the Metal kernel needs an Apple
GPU). The KV pool is deliberately small, so a handful of concurrent requests visibly fills it.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import uvicorn
from fastapi.responses import HTMLResponse, JSONResponse

from kiln.cli import resolve_model_dir
from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.server.app import create_app
from kiln.server.async_engine import AsyncEngine
from kiln.tokenizer import Tokenizer

MODEL = os.environ.get("KILN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("KILN_DTYPE", "fp32")]
NUM_BLOCKS = int(os.environ.get("KILN_NUM_BLOCKS", "320"))  # x 16 tokens
MAX_IN_FLIGHT = 24  # a public demo on 2 vCPUs: refuse rather than queue for minutes
PAGE = (Path(__file__).parent / "index.html").read_text()


class InFlightLimit:
    """Refuse POSTs to /v1/ beyond `limit` at once. Pure ASGI, so the slot is released in a `finally` however
    the request ends: finished, client gone before or during streaming, or an error."""

    def __init__(self, app, limit: int):
        self.app, self.limit, self.in_flight = app, limit, 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        if self.in_flight >= self.limit:
            busy = JSONResponse({"error": {"message": "the demo is busy, try again in a moment",
                                           "type": "server_busy", "code": None}}, status_code=503)
            return await busy(scope, receive, send)
        self.in_flight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.in_flight -= 1


def build():
    torch.set_num_threads(os.cpu_count() or 2)
    path = resolve_model_dir(MODEL)
    model, config = load_model(path, dtype=DTYPE, device="cpu")
    tok = Tokenizer.from_hf_dir(path)
    engine = LLMEngine(model, config, tok, num_blocks=NUM_BLOCKS, block_size=16, max_num_seqs=16,
                       max_batched_tokens=256, max_context=1024)
    app = create_app(AsyncEngine(engine), tok, MODEL, enable_viz=True)
    app.add_middleware(InFlightLimit, limit=MAX_IN_FLIGHT)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE

    return app


if __name__ == "__main__":
    uvicorn.run(build(), host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), log_level="warning")

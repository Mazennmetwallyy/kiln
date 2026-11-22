"""The hosted Kiln demo: the OpenAI-compatible server, the live KV-cache view, and a chat page.

Runs Qwen2.5-0.5B-Instruct on CPU with the PyTorch attention path (the Metal kernel needs an Apple
GPU). The KV pool is deliberately small, so a handful of concurrent requests visibly fills it.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import uvicorn
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from kiln.cli import resolve_model_dir
from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.server.app import create_app
from kiln.server.async_engine import AsyncEngine
from kiln.tokenizer import Tokenizer

MODEL = os.environ.get("KILN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
NUM_BLOCKS = int(os.environ.get("KILN_NUM_BLOCKS", "320"))  # x 16 tokens
MAX_IN_FLIGHT = 24  # a public demo on 2 vCPUs: refuse rather than queue for minutes
PAGE = (Path(__file__).parent / "index.html").read_text()


def build():
    torch.set_num_threads(os.cpu_count() or 2)
    path = resolve_model_dir(MODEL)
    model, config = load_model(path, dtype=torch.float32, device="cpu")
    tok = Tokenizer.from_hf_dir(path)
    engine = LLMEngine(model, config, tok, num_blocks=NUM_BLOCKS, block_size=16, max_num_seqs=16,
                       max_batched_tokens=256, max_context=1024)
    app = create_app(AsyncEngine(engine), tok, MODEL, enable_viz=True)
    in_flight = 0

    @app.middleware("http")
    async def limit(request: Request, call_next):
        nonlocal in_flight
        if not request.url.path.startswith("/v1/") or request.method != "POST":
            return await call_next(request)
        if in_flight >= MAX_IN_FLIGHT:
            return JSONResponse({"error": {"message": "the demo is busy, try again in a moment",
                                           "type": "server_busy", "code": None}}, status_code=503)
        in_flight += 1
        try:
            response = await call_next(request)
        except BaseException:
            in_flight -= 1
            raise
        # Streaming bodies run after call_next returns: release the slot when the body is done.
        body = response.body_iterator

        async def release():
            nonlocal in_flight
            try:
                async for chunk in body:
                    yield chunk
            finally:
                in_flight -= 1

        response.body_iterator = release()
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return PAGE

    return app


if __name__ == "__main__":
    uvicorn.run(build(), host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), log_level="warning")

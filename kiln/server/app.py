"""The OpenAI-compatible HTTP API."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from kiln.engine.engine import RequestOutput
from kiln.server.async_engine import AsyncEngine
from kiln.server.kv_view import KVView
from kiln.server.metrics import Metrics
from kiln.server.protocol import ChatCompletionRequest, CompletionRequest
from kiln.tokenizer import Tokenizer

DEFAULT_CHAT_MAX_TOKENS = 512


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "code": None}},
                        status_code=status)


def create_app(engine: AsyncEngine, tokenizer: Tokenizer, model_name: str, enable_viz: bool = False) -> FastAPI:
    metrics = Metrics()
    engine.on_step.append(metrics.on_step)
    if enable_viz:
        kv_view = KVView(engine.engine)
        engine.on_step.append(kv_view.on_step)
    max_context = engine.engine.max_context

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        engine.start()
        yield
        engine.stop()

    app = FastAPI(title="Kiln", lifespan=lifespan)
    app.state.metrics = metrics

    @app.exception_handler(RequestValidationError)
    async def _bad_schema(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(f"invalid request: {exc.errors()}")

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return _error(str(exc))

    if enable_viz:  # opt-in: who shares which blocks is a side channel between users (see ADR 0004)
        @app.get("/viz", response_class=HTMLResponse)
        async def viz() -> str:
            """The live KV-pool visualiser."""
            return resources.files("kiln.server").joinpath("static/viz.html").read_text()

        @app.get("/kv")
        async def kv() -> dict[str, Any]:
            return kv_view.latest()[1]

        @app.get("/kv/stream")
        async def kv_stream(request: Request) -> StreamingResponse:
            """Server-Sent Events: a new snapshot whenever the pool changes (at most ~10 per second)."""

            async def events() -> AsyncIterator[str]:
                seen = -1
                while not await request.is_disconnected():
                    version, snapshot = kv_view.latest()
                    if version != seen:
                        seen = version
                        yield f"data: {json.dumps(snapshot, separators=(',', ':'))}\n\n"
                    await asyncio.sleep(0.1)

            return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "kiln"}]}

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    async def _start(prefix: str, prompt_ids: list[int], req: Any, max_tokens: int):
        params = req.sampling_params(max_tokens)
        rid = f"{prefix}-{uuid.uuid4().hex}"
        stream = await engine.submit(rid, prompt_ids, params)  # ValueError -> 400, before any output
        metrics.requests.inc()
        metrics.prompt_tokens.inc(len(prompt_ids))
        return rid, _timed(stream, metrics)

    @app.post("/v1/completions", response_model=None)
    async def completions(req: CompletionRequest) -> Response:
        prompt_ids = tokenizer.encode(req.prompt) if isinstance(req.prompt, str) else list(req.prompt)
        rid, stream = await _start("cmpl", prompt_ids, req, req.max_tokens)
        created = int(time.time())

        def chunk(text: str, finish: str | None) -> dict[str, Any]:
            return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}

        if not req.stream:
            text, n, finish = await _collect(stream)
            body = chunk(text, finish)
            body["usage"] = _usage(len(prompt_ids), n)
            return JSONResponse(body)

        async def sse() -> AsyncGenerator[str, None]:
            n = 0
            try:
                async for out in stream:
                    n += len(out.new_token_ids)
                    if out.new_text or out.finished:
                        yield _sse(chunk(out.new_text, out.finish_reason if out.finished else None))
                if req.include_usage:
                    yield _sse({**chunk("", None), "choices": [], "usage": _usage(len(prompt_ids), n)})
                yield "data: [DONE]\n\n"
            finally:
                await stream.aclose()

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/chat/completions", response_model=None)
    async def chat(req: ChatCompletionRequest) -> Response:
        prompt = tokenizer.apply_chat_template([m.model_dump() for m in req.messages])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        max_tokens = req.max_completion_tokens or req.max_tokens
        if max_tokens is None:
            max_tokens = max(1, min(DEFAULT_CHAT_MAX_TOKENS, max_context - len(prompt_ids)))
        rid, stream = await _start("chatcmpl", prompt_ids, req, max_tokens)
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish: str | None) -> dict[str, Any]:
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}

        if not req.stream:
            text, n, finish = await _collect(stream)
            return JSONResponse({
                "id": rid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "logprobs": None, "finish_reason": finish}],
                "usage": _usage(len(prompt_ids), n),
            })

        async def sse() -> AsyncGenerator[str, None]:
            n = 0
            try:
                yield _sse(chunk({"role": "assistant", "content": ""}, None))
                async for out in stream:
                    n += len(out.new_token_ids)
                    if out.new_text:
                        yield _sse(chunk({"content": out.new_text}, None))
                    if out.finished:
                        yield _sse(chunk({}, out.finish_reason))
                if req.include_usage:
                    yield _sse({**chunk({}, None), "choices": [], "usage": _usage(len(prompt_ids), n)})
                yield "data: [DONE]\n\n"
            finally:
                await stream.aclose()

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


async def _timed(stream: AsyncGenerator[RequestOutput, None], metrics: Metrics) -> AsyncGenerator[RequestOutput, None]:
    """Pass outputs through while recording time to first token, inter-token latency and total latency."""
    start = last = time.perf_counter()
    first = True
    try:
        async for out in stream:
            now = time.perf_counter()
            if out.new_token_ids:
                (metrics.ttft if first else metrics.itl).observe(now - last if not first else now - start)
                first, last = False, now
            if out.finished:
                metrics.e2e.observe(now - start)
            yield out
    finally:
        await stream.aclose()


async def _collect(stream: AsyncGenerator[RequestOutput, None]) -> tuple[str, int, str | None]:
    text, n, finish = "", 0, None
    async for out in stream:
        text += out.new_text
        n += len(out.new_token_ids)
        finish = out.finish_reason or finish
    return text, n, finish


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

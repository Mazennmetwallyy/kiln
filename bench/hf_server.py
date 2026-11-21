"""Baselines: Hugging Face transformers behind the same streaming /v1/completions API Kiln serves.

    python -m bench.hf_server --mode sequential --port 8001
    python -m bench.hf_server --mode static --max-batch 16 --port 8002

sequential: one request at a time, first come first served, tokens streamed as generated. This is
            what you get by wrapping `model.generate` in a web server.
static:     classic static batching. Queued requests (up to --max-batch) are padded into one batch and
            run with a single `generate` call until the longest one is done. Nothing is returned until
            the batch finishes, and short requests wait for long ones.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.generation.streamers import BaseStreamer

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def _clean_config(pad_id: int) -> GenerationConfig:
    # Greedy, no EOS (benchmarks generate exactly max_tokens), and none of the checkpoint's sampling defaults.
    return GenerationConfig(do_sample=False, eos_token_id=None, pad_token_id=pad_id)


@torch.no_grad()
def static_batch_generate(model: Any, prompts: list[list[int]], out_lens: list[int], pad_id: int) -> list[list[int]]:
    """Left-pad the prompts into one batch, generate to the longest requested length, then truncate."""
    width = max(len(p) for p in prompts)
    device = model.device
    ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max(out_lens),
                         generation_config=_clean_config(pad_id))
    return [out[i, width:width + n].tolist() for i, n in enumerate(out_lens)]


class _QueueStreamer(BaseStreamer):
    """Forwards each newly generated token id to an asyncio queue (the first put() is the prompt)."""

    def __init__(self, loop: asyncio.AbstractEventLoop, q: asyncio.Queue[int | None]):
        self.loop, self.q, self.seen_prompt = loop, q, False

    def put(self, value: torch.Tensor) -> None:
        if not self.seen_prompt:
            self.seen_prompt = True
            return
        for t in value.reshape(-1).tolist():
            self.loop.call_soon_threadsafe(self.q.put_nowait, t)

    def end(self) -> None:
        self.loop.call_soon_threadsafe(self.q.put_nowait, None)


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chunk(text: str, finish: str | None = None) -> dict[str, Any]:
    return {"object": "text_completion", "choices": [{"index": 0, "text": text, "finish_reason": finish}]}


def create_app(model: Any, mode: str, max_batch: int = 16, batch_wait_s: float = 0.05, pad_id: int = 0) -> FastAPI:
    lock = asyncio.Lock()  # sequential mode: one generate() at a time
    pending: asyncio.Queue[tuple[list[int], int, asyncio.Future[list[int]]]] = asyncio.Queue()

    async def batcher() -> None:
        while True:
            batch = [await pending.get()]
            deadline = asyncio.get_running_loop().time() + batch_wait_s
            while len(batch) < max_batch:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(pending.get(), timeout))
                except asyncio.TimeoutError:  # not the builtin TimeoutError before Python 3.11
                    break
            try:
                outs = await asyncio.to_thread(static_batch_generate, model, [b[0] for b in batch],
                                               [b[1] for b in batch], pad_id)
                for (_, _, fut), out in zip(batch, outs, strict=True):
                    fut.set_result(out)
            except Exception as e:
                for _, _, fut in batch:
                    fut.set_exception(e)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(batcher()) if mode == "static" else None
        yield
        if task:
            task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request) -> StreamingResponse:
        body = await request.json()
        prompt, n = list(body["prompt"]), int(body["max_tokens"])

        async def sequential() -> AsyncIterator[str]:
            async with lock:
                loop = asyncio.get_running_loop()
                q: asyncio.Queue[int | None] = asyncio.Queue()
                ids = torch.tensor([prompt], device=model.device)

                def work() -> None:
                    with torch.no_grad():
                        model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                                       generation_config=_clean_config(pad_id), streamer=_QueueStreamer(loop, q))

                thread = threading.Thread(target=work, daemon=True)
                thread.start()
                count = 0
                while (tok := await q.get()) is not None:
                    count += 1
                    yield _sse(_chunk(f"<{tok}>"))
                thread.join()
            yield _sse({**_chunk("", "length"), "usage": {"completion_tokens": count}})
            yield "data: [DONE]\n\n"

        async def static() -> AsyncIterator[str]:
            fut: asyncio.Future[list[int]] = asyncio.get_running_loop().create_future()
            await pending.put((prompt, n, fut))
            out = await fut
            yield _sse({**_chunk("".join(f"<{t}>" for t in out), "length"),
                        "usage": {"completion_tokens": len(out)}})
            yield "data: [DONE]\n\n"

        return StreamingResponse(sequential() if mode == "sequential" else static(), media_type="text/event-stream")

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", choices=["sequential", "static"], required=True)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=DTYPES[a.dtype]).to(a.device).eval()
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
    uvicorn.run(create_app(model, a.mode, a.max_batch, pad_id=pad), host="127.0.0.1", port=a.port,
                log_level="warning")


if __name__ == "__main__":
    main()

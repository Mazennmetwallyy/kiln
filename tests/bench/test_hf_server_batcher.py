"""The static-batching baseline must answer a batch that never fills (fewer requests than --max-batch)."""

import asyncio

import httpx
import torch

from bench import hf_server


class _EchoModel:
    device = torch.device("cpu")


def test_partial_batch_completes(monkeypatch):
    # Stand in for model.generate: return token 7 repeated n times per request.
    monkeypatch.setattr(hf_server, "static_batch_generate",
                        lambda model, prompts, out_lens, pad_id: [[7] * n for n in out_lens])
    app = hf_server.create_app(_EchoModel(), "static", max_batch=16, batch_wait_s=0.01)

    async def go() -> list[str]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
                rs = await asyncio.wait_for(asyncio.gather(
                    *[c.post("/v1/completions", json={"prompt": [1, 2], "max_tokens": 3}) for _ in range(2)]), 5)
                return [r.text for r in rs]

    for body in asyncio.run(go()):
        assert "<7><7><7>" in body

"""Open-loop load generator for OpenAI-compatible /v1/completions servers (streaming).

Requests arrive as a Poisson process at a given rate (or all at once for rate=inf), independently of
how fast the server answers: that's how real traffic behaves, and it exposes queueing.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    ttft: float = math.nan  # seconds to the first streamed text
    itls: list[float] = field(default_factory=list)  # gaps between later text chunks
    e2e: float = math.nan
    output_tokens: int = 0
    error: str | None = None


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (numpy's default), without needing numpy."""
    if not xs:
        return math.nan
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def poisson_arrivals(n: int, rate: float, seed: int = 0) -> list[float]:
    """Send times (s from start). Exponential gaps with mean 1/rate; the first request goes at t=0."""
    if math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        if i:
            t += rng.expovariate(rate)
        out.append(t)
    return out


def summarize(results: list[RequestResult], duration: float) -> dict[str, float]:
    ok = [r for r in results if r.error is None]
    itls = [x for r in ok for x in r.itls]
    return {
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "duration_s": duration,
        "request_throughput": len(ok) / duration,
        "output_throughput": sum(r.output_tokens for r in ok) / duration,
        "ttft_p50": percentile([r.ttft for r in ok], 50),
        "ttft_p99": percentile([r.ttft for r in ok], 99),
        "itl_p50": percentile(itls, 50),
        "itl_p99": percentile(itls, 99),
        "e2e_p50": percentile([r.e2e for r in ok], 50),
        "e2e_p99": percentile([r.e2e for r in ok], 99),
    }


async def _one(client: httpx.AsyncClient, url: str, prompt_ids: list[int], output_len: int) -> RequestResult:
    body = {"model": "bench", "prompt": prompt_ids, "max_tokens": output_len, "temperature": 0, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    res = RequestResult(output_tokens=output_len)
    start = last = time.perf_counter()
    try:
        async with client.stream("POST", f"{url}/v1/completions", json=body) as r:
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    res.output_tokens = chunk["usage"]["completion_tokens"]
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    now = time.perf_counter()
                    if math.isnan(res.ttft):
                        res.ttft = now - start
                    else:
                        res.itls.append(now - last)
                    last = now
        res.e2e = time.perf_counter() - start
    except Exception as e:  # recorded, not raised: one failure shouldn't end the run
        res.error = repr(e)
    return res


async def run(url: str, requests: list[dict], rate: float, seed: int = 0) -> dict[str, float]:
    arrivals = poisson_arrivals(len(requests), rate, seed)
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600), limits=limits) as client:
        start = time.perf_counter()

        async def fire(at: float, req: dict) -> RequestResult:
            await asyncio.sleep(max(0.0, at - (time.perf_counter() - start)))
            return await _one(client, url, req["prompt_ids"], req["output_len"])

        results = await asyncio.gather(*(fire(a, r) for a, r in zip(arrivals, requests, strict=True)))
        duration = time.perf_counter() - start
    return summarize(list(results), duration)

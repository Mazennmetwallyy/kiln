"""Drive a running Kiln server with traffic that makes the KV visualiser interesting (for the demo GIF).

Requests share one of three system prompts (prefix-cache hits: shared, orange blocks), arrive faster
than a deliberately small pool can hold (preemption), and finish at different times (cached,
evictable blocks).

    kiln serve --model Qwen/Qwen2.5-0.5B-Instruct --kv-memory-gb 0.025 \
        --attention-backend metal --enable-viz --port 8000
    python -m bench.demo_viz --url http://127.0.0.1:8000 --requests 40 --rate 3
    open http://127.0.0.1:8000/viz
"""

from __future__ import annotations

import argparse
import asyncio
import random

import httpx

SYSTEMS = [
    "You are a patient maths tutor. Explain each step, check the student's reasoning, and end with one "
    "practice question. Prefer worked examples over definitions, and keep notation consistent.",
    "You are a senior code reviewer. Point out bugs first, then risky patterns, then style. Quote the exact "
    "line you mean, suggest a concrete fix, and say how confident you are in each comment.",
    "You are a travel planner for rail trips in Europe. Suggest routes with approximate durations, mention "
    "where seat reservations are required, and keep each day's plan realistic.",
]
QUESTIONS = [
    "How do I solve 3x + 7 = 22?", "What is a derivative, intuitively?", "Review: for i in range(len(a)): print(a[i])",
    "Is it bad to catch Exception everywhere?", "Plan three days from Paris to Milan.", "Why is 0.1 + 0.2 != 0.3?",
    "Explain recursion with an example.", "Review: if x == None: return", "Best way from Vienna to Prague?",
    "What's the area of a circle of radius 3?", "Review: password = request.args['pw']", "Two days in Switzerland?",
]


async def one(client: httpx.AsyncClient, rng: random.Random) -> None:
    messages = [{"role": "system", "content": rng.choice(SYSTEMS)}, {"role": "user", "content": rng.choice(QUESTIONS)}]
    body = {"model": "kiln", "messages": messages, "max_tokens": rng.randint(40, 220), "temperature": 0.7,
            "stream": True}
    async with client.stream("POST", "/v1/chat/completions", json=body) as r:
        async for _ in r.aiter_lines():
            pass


async def main(url: str, n: int, rate: float, seed: int) -> None:
    rng = random.Random(seed)
    async with httpx.AsyncClient(base_url=url, timeout=600) as client:
        tasks = []
        for _ in range(n):
            tasks.append(asyncio.create_task(one(client, rng)))
            await asyncio.sleep(rng.expovariate(rate))
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--rate", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    asyncio.run(main(a.url, a.requests, a.rate, a.seed))

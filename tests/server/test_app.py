import asyncio
import time

import httpx
import openai
import pytest

GREEDY = {"temperature": 0, "extra_body": {"ignore_eos": True}}


@pytest.fixture
def client(server):
    url, _ = server
    return openai.OpenAI(base_url=f"{url}/v1", api_key="unused")


def test_models_and_health(server, client):
    assert [m.id for m in client.models.list().data] == ["tiny"]
    assert httpx.get(f"{server[0]}/health").status_code == 200


def test_completion_non_streaming(client):
    r = client.completions.create(model="tiny", prompt="Hello there", max_tokens=8, **GREEDY)
    assert r.object == "text_completion" and len(r.choices) == 1
    assert isinstance(r.choices[0].text, str) and r.choices[0].finish_reason == "length"
    assert r.usage.completion_tokens == 8 and r.usage.prompt_tokens > 0
    assert r.usage.total_tokens == r.usage.prompt_tokens + 8


def test_completion_stream_matches_non_stream(client):
    full = client.completions.create(model="tiny", prompt="Once upon", max_tokens=12, **GREEDY)
    chunks = list(client.completions.create(model="tiny", prompt="Once upon", max_tokens=12, stream=True, **GREEDY))
    assert "".join(c.choices[0].text for c in chunks) == full.choices[0].text
    assert chunks[-1].choices[0].finish_reason == "length"
    assert all(c.choices[0].finish_reason is None for c in chunks[:-1])


def test_chat_non_streaming(client):
    r = client.chat.completions.create(model="tiny", messages=[{"role": "user", "content": "Hi"}], max_tokens=6,
                                       **GREEDY)
    assert r.object == "chat.completion"
    assert r.choices[0].message.role == "assistant" and isinstance(r.choices[0].message.content, str)
    assert r.choices[0].finish_reason == "length" and r.usage.completion_tokens == 6


def test_chat_stream_with_usage(client):
    msgs = [{"role": "user", "content": "Tell me a story"}]
    full = client.chat.completions.create(model="tiny", messages=msgs, max_tokens=10, **GREEDY)
    chunks = list(client.chat.completions.create(model="tiny", messages=msgs, max_tokens=10, stream=True,
                                                 stream_options={"include_usage": True}, **GREEDY))
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == full.choices[0].message.content
    finals = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert [c.choices[0].finish_reason for c in finals] == ["length"]
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 10 and chunks[-1].choices == []


def test_stop_string(client):
    full = client.completions.create(model="tiny", prompt="The answer is", max_tokens=24, **GREEDY).choices[0].text
    stop = full[len(full) // 2: len(full) // 2 + 3]
    assert stop.strip(), "need a non-blank stop string"
    r = client.completions.create(model="tiny", prompt="The answer is", max_tokens=24, stop=[stop], **GREEDY)
    assert r.choices[0].text == full[: full.index(stop)] and r.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("kwargs", [
    {"n": 2},
    {"max_tokens": 5000},
    {"temperature": -1.0},
    {"top_p": 0.0},
    {"stop": ["a", "b", "c", "d", "e"]},
    {"prompt": [1, 2, 10**9]},
])
def test_bad_requests_are_400(client, kwargs):
    with pytest.raises(openai.BadRequestError):
        client.completions.create(model="tiny", **({"prompt": "x", "max_tokens": 4} | kwargs))


def test_client_disconnect_frees_blocks(server):
    url, ae = server
    body = {"model": "tiny", "prompt": "Go", "max_tokens": 400, "stream": True, "ignore_eos": True}
    with httpx.stream("POST", f"{url}/v1/completions", json=body, timeout=30) as r:
        lines = r.iter_lines()
        got = [next(lines) for _ in range(6)]
        assert any(line.startswith("data: ") for line in got)
    deadline = time.time() + 2
    while time.time() < deadline:
        stats = asyncio.run(ae.stats())
        if stats.num_running == 0 and stats.num_free_blocks == stats.num_blocks:
            return
        time.sleep(0.05)
    raise AssertionError(f"KV blocks not freed after disconnect: {stats}")


def test_metrics(server, client):
    client.completions.create(model="tiny", prompt="metrics", max_tokens=3, **GREEDY)
    text = httpx.get(f"{server[0]}/metrics").text
    for name in ["kiln_requests_total", "kiln_generation_tokens_total", "kiln_prompt_tokens_total",
                 "kiln_time_to_first_token_seconds_bucket", "kiln_kv_blocks_total", "kiln_running_requests",
                 "kiln_prefix_cache_hit_tokens", "kiln_prefix_cache_query_tokens", "kiln_kv_blocks_cached"]:
        assert name in text, name
    gen = [line for line in text.splitlines() if line.startswith("kiln_generation_tokens_total ")]
    assert gen and float(gen[0].split()[1]) >= 3


def test_non_finite_temperature_is_400(server):
    body = '{"model": "tiny", "prompt": "x", "max_tokens": 2, "temperature": Infinity}'
    r = httpx.post(f"{server[0]}/v1/completions", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 400

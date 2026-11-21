import asyncio

import pytest

from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams
from kiln.server.async_engine import AsyncEngine
from kiln.tokenizer import Tokenizer


@pytest.fixture
def async_engine(tiny_text_model_dir):
    model, config = load_model(tiny_text_model_dir)
    tok = Tokenizer.from_hf_dir(tiny_text_model_dir)
    ae = AsyncEngine(LLMEngine(model, config, tok, num_blocks=64, block_size=16))
    ae.start()
    yield ae
    ae.stop()


async def collect(ae, rid, prompt, params):
    toks, finish = [], None
    async for out in await ae.submit(rid, prompt, params):
        toks += out.new_token_ids
        finish = out.finish_reason or finish
    return toks, finish


def test_concurrent_requests_complete(async_engine):
    async def main():
        params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
        results = await asyncio.gather(*(collect(async_engine, f"r{i}", [5 + i, 6, 7], params) for i in range(6)))
        assert all(len(t) == 12 and f == "length" for t, f in results)
        alone = await collect(async_engine, "solo", [5, 6, 7], params)
        assert alone[0] == results[0][0]

    asyncio.run(main())


def test_validation_error_raises_before_iteration(async_engine):
    async def main():
        with pytest.raises(ValueError, match="prompt"):
            await async_engine.submit("bad", [], SamplingParams(max_tokens=4))

    asyncio.run(main())


def test_leaving_early_aborts_and_frees_blocks(async_engine):
    async def main():
        stream = await async_engine.submit("long", [5, 6, 7], SamplingParams(max_tokens=400, ignore_eos=True))
        n = 0
        async for _ in stream:
            n += 1
            if n == 3:
                break
        await stream.aclose()
        for _ in range(100):
            stats = await async_engine.stats()
            if stats.num_running == 0 and stats.num_free_blocks == stats.num_blocks:
                return
            await asyncio.sleep(0.02)
        raise AssertionError(f"blocks not freed: {stats}")

    asyncio.run(main())

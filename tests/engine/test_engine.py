import pytest
import torch

from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams
from tests.model.tiny import make_tiny_hf_model

torch.set_grad_enabled(False)

LENGTHS = [1, 3, 9, 17, 30, 5, 40, 12]
PROMPTS = {f"r{i}": [(7 * i + j) % 250 + 3 for j in range(n)] for i, n in enumerate(LENGTHS)}
MAX_TOKENS = {f"r{i}": m for i, m in enumerate([5, 20, 8, 12, 6, 15, 10, 18])}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path, hf = make_tiny_hf_model("qwen2", tmp_path_factory.mktemp("tiny"))
    hf.generation_config.eos_token_id = None
    model, config = load_model(path)
    return model, config, hf


def hf_greedy(hf, prompt, n):
    out = hf.generate(torch.tensor([prompt]), do_sample=False, max_new_tokens=n, pad_token_id=0)
    return out[0, len(prompt):].tolist()


def run_all(engine, requests, params_for):
    for rid, prompt in requests.items():
        engine.add_request(rid, prompt, params_for(rid))
    tokens = {rid: [] for rid in requests}
    finish = {}
    steps = 0
    while engine.has_unfinished():
        for out in engine.step():
            tokens[out.request_id] += out.new_token_ids
            if out.finished:
                finish[out.request_id] = out.finish_reason
        steps += 1
        assert steps < 2000
    return tokens, finish


def greedy(rid):
    return SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS[rid], ignore_eos=True)


def test_batched_greedy_matches_hf(tiny):
    model, config, hf = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4)
    tokens, finish = run_all(engine, PROMPTS, greedy)
    for rid, prompt in PROMPTS.items():
        assert tokens[rid] == hf_greedy(hf, prompt, MAX_TOKENS[rid]), rid
        assert finish[rid] == "length"
    assert engine.stats().num_free_blocks == 64


def test_preempted_requests_resume_identically(tiny):
    model, config, hf = tiny
    engine = LLMEngine(model, config, None, num_blocks=16, block_size=4)  # 64 slots: forces preemption
    tokens, _ = run_all(engine, PROMPTS, greedy)
    assert engine.stats().num_preemptions_total > 0
    for rid, prompt in PROMPTS.items():
        assert tokens[rid] == hf_greedy(hf, prompt, MAX_TOKENS[rid]), rid


def test_chunked_prefill_matches(tiny):
    model, config, hf = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4, max_batched_tokens=7)
    tokens, _ = run_all(engine, PROMPTS, greedy)
    for rid, prompt in PROMPTS.items():
        assert tokens[rid] == hf_greedy(hf, prompt, MAX_TOKENS[rid]), rid


def test_requests_added_between_steps(tiny):
    model, config, hf = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4)
    engine.add_request("a", PROMPTS["r3"], greedy("r3"))
    got: dict[str, list[int]] = {"a": [], "b": []}
    for i in range(200):
        if i == 4:
            engine.add_request("b", PROMPTS["r6"], greedy("r6"))
        for out in engine.step():
            got[out.request_id] += out.new_token_ids
        if i > 4 and not engine.has_unfinished():
            break
    assert got["a"] == hf_greedy(hf, PROMPTS["r3"], MAX_TOKENS["r3"])
    assert got["b"] == hf_greedy(hf, PROMPTS["r6"], MAX_TOKENS["r6"])


def test_seeded_sampling_batched_equals_alone(tiny):
    model, config, _ = tiny

    def sampled(rid):
        return SamplingParams(temperature=1.0, top_p=0.9, seed=11, max_tokens=MAX_TOKENS[rid], ignore_eos=True)

    together, _ = run_all(LLMEngine(model, config, None, num_blocks=64, block_size=4), PROMPTS, sampled)
    for rid in ["r1", "r4"]:
        alone, _ = run_all(LLMEngine(model, config, None, num_blocks=64, block_size=4), {rid: PROMPTS[rid]}, sampled)
        assert alone[rid] == together[rid]


def test_request_too_long_for_pool_rejected(tiny):
    model, config, _ = tiny
    engine = LLMEngine(model, config, None, num_blocks=4, block_size=4)  # 16 slots
    with pytest.raises(ValueError, match="pool"):
        engine.add_request("x", list(range(10)), SamplingParams(max_tokens=10))
    assert not engine.has_unfinished()


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"prompt_ids": []}, "prompt"),
    ({"prompt_ids": list(range(100)), "params": SamplingParams(max_tokens=100)}, "context"),
    ({"params": SamplingParams(stop=["x"], max_tokens=4)}, "tokenizer"),
    ({"prompt_ids": [1, 2, 256]}, "vocab"),
    ({"prompt_ids": [-1, 2]}, "vocab"),
])
def test_add_request_validates_eagerly(tiny, kwargs, match):
    model, config, _ = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4)
    args = {"prompt_ids": [1, 2, 3], "params": SamplingParams(max_tokens=4)} | kwargs
    with pytest.raises(ValueError, match=match):
        engine.add_request("x", args["prompt_ids"], args["params"])


def test_duplicate_request_id_rejected(tiny):
    model, config, _ = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4)
    engine.add_request("x", [1, 2], SamplingParams(max_tokens=2))
    with pytest.raises(ValueError, match="already"):
        engine.add_request("x", [1, 2], SamplingParams(max_tokens=2))


def test_abort_waiting_running_finished_unknown(tiny):
    model, config, _ = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4, max_num_seqs=1)
    engine.add_request("run", [1, 2, 3], SamplingParams(max_tokens=50, ignore_eos=True))
    engine.add_request("wait", [4, 5, 6], SamplingParams(max_tokens=50, ignore_eos=True))
    engine.step()
    assert engine.stats().num_running == 1 and engine.stats().num_waiting == 1
    engine.abort("run")
    engine.abort("wait")
    engine.abort("never-existed")
    assert not engine.has_unfinished() and engine.stats().num_free_blocks == 64
    engine.add_request("done", [1], SamplingParams(max_tokens=1, ignore_eos=True))
    while engine.has_unfinished():
        engine.step()
    engine.abort("done")  # finished already: no-op


def test_max_tokens_zero(tiny):
    model, config, _ = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4)
    engine.add_request("z", [1, 2, 3], SamplingParams(max_tokens=0))
    outs = engine.step()
    assert [(o.request_id, o.new_token_ids, o.finished, o.finish_reason) for o in outs] == [("z", [], True, "length")]
    assert not engine.has_unfinished()


def test_one_failing_request_does_not_fail_the_batch(tiny, monkeypatch):
    import kiln.engine.engine as engine_mod

    model, config, hf = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4)
    real_sample = engine_mod.sample

    def flaky(logits, params, generator):
        if params.seed == 666:
            raise RuntimeError("boom")
        return real_sample(logits, params, generator)

    monkeypatch.setattr(engine_mod, "sample", flaky)
    engine.add_request("bad", [1, 2, 3], SamplingParams(temperature=1.0, seed=666, max_tokens=5, ignore_eos=True))
    engine.add_request("good", PROMPTS["r2"], greedy("r2"))
    got, finish = {"bad": [], "good": []}, {}
    while engine.has_unfinished():
        for out in engine.step():
            got[out.request_id] += out.new_token_ids
            if out.finished:
                finish[out.request_id] = out.finish_reason
    assert finish == {"bad": "error", "good": "length"}
    assert got["good"] == hf_greedy(hf, PROMPTS["r2"], MAX_TOKENS["r2"])
    assert engine.stats().num_free_blocks == 64


SYSTEM = [(11 * j) % 200 + 20 for j in range(21)]  # a 21-token "system prompt" (5 full blocks of 4)
SHARED = {f"s{i}": [*SYSTEM, *[(5 * i + j) % 90 + 100 for j in range(i + 1)]] for i in range(8)}


def shared_greedy(rid):
    return SamplingParams(temperature=0.0, max_tokens=6 + int(rid[1:]), ignore_eos=True)


@pytest.mark.parametrize("num_blocks", [64, 14])  # 14 blocks = 56 slots: forces eviction and preemption
def test_prefix_caching_shared_system_prompt(tiny, num_blocks):
    model, config, hf = tiny
    engine = LLMEngine(model, config, None, num_blocks=num_blocks, block_size=4, max_num_seqs=4)
    tokens, _ = run_all(engine, SHARED, shared_greedy)
    for rid, prompt in SHARED.items():
        assert tokens[rid] == hf_greedy(hf, prompt, 6 + int(rid[1:])), rid
    stats = engine.stats()
    assert stats.prefix_cache_hit_tokens >= 20 * (len(SHARED) - 4)  # later admissions hit the 20 cached tokens
    assert stats.num_free_blocks == num_blocks


def test_prefix_caching_off_has_no_hits(tiny):
    model, config, _ = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4, enable_prefix_caching=False)
    run_all(engine, SHARED, shared_greedy)
    assert engine.stats().prefix_cache_hit_tokens == 0


def test_identical_prompts_reuse_all_but_last_token(tiny):
    model, config, hf = tiny
    engine = LLMEngine(model, config, None, num_blocks=64, block_size=4)
    prompt = SYSTEM[:17]
    run_all(engine, {"a": prompt}, lambda _: SamplingParams(temperature=0.0, max_tokens=5, ignore_eos=True))
    tokens, _ = run_all(engine, {"b": prompt}, lambda _: SamplingParams(temperature=0.0, max_tokens=5,
                                                                         ignore_eos=True))
    assert tokens["b"] == hf_greedy(hf, prompt, 5)
    assert engine.stats().prefix_cache_hit_tokens == 16


def test_kv_snapshot_shows_ownership_sharing_and_cached_blocks(tiny):
    model, config, _ = tiny
    engine = LLMEngine(model, config, None, num_blocks=32, block_size=4)
    shared = list(range(3, 15))  # 3 full blocks
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    engine.add_request("a", [*shared, 40], params)
    engine.step()  # a is prefilled; its 3 full blocks are now in the prefix cache
    engine.add_request("b", [*shared, 41], params)
    engine.step()  # b reuses a's 3 blocks
    snap = engine.kv_snapshot()
    assert snap["num_blocks"] == 32 and len(snap["state"]) == 32 == len(snap["owner"])
    assert snap["state"].count(2) == 3  # shared by a and b
    assert {r["id"] for r in snap["requests"]} == {"a", "b"}
    a_seq = next(r["seq"] for r in snap["requests"] if r["id"] == "a")
    assert all(snap["owner"][b] == a_seq for b in range(32) if snap["state"][b] == 2)
    while engine.has_unfinished():
        engine.step()
    snap = engine.kv_snapshot()
    assert snap["requests"] == [] and snap["state"].count(1) == snap["state"].count(2) == 0
    assert snap["state"].count(3) >= 3  # finished requests' full blocks stay cached (evictable)

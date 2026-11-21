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

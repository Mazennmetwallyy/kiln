"""Speculative decoding inside the continuous-batching engine must never change greedy output."""

import pytest
import torch

from kiln.engine.engine import LLMEngine
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams
from tests.engine.test_engine import MAX_TOKENS, PROMPTS, greedy, run_all
from tests.model.tiny import make_tiny_hf_model

torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    path, _ = make_tiny_hf_model("qwen2", tmp_path_factory.mktemp("target"))
    target, config = load_model(path)
    same, _ = load_model(path)
    # A draft that agrees with the target only some of the time: the same weights, slightly perturbed.
    near, _ = load_model(path)
    g = torch.Generator().manual_seed(1)
    for p in near.parameters():
        p.add_(torch.randn(p.shape, generator=g) * 0.05 * p.abs().mean())
    # A draft with a different architecture size that has nothing to do with the target.
    other_path, _ = make_tiny_hf_model("qwen2", tmp_path_factory.mktemp("other"), seed=7, num_hidden_layers=1,
                                       hidden_size=32, num_attention_heads=2, num_key_value_heads=1)
    other, other_config = load_model(other_path)
    return {"target": (target, config), "same": (same, config), "near": (near, config),
            "other": (other, other_config)}


def spec_engine(models, draft, k, **kw):
    target, config = models["target"]
    d, dconfig = models[draft]
    args = {"num_blocks": 64, "block_size": 4} | kw
    return LLMEngine(target, config, None, draft_model=d, draft_config=dconfig, num_speculative_tokens=k, **args)


@pytest.fixture(scope="module")
def reference(models):
    target, config = models["target"]
    tokens, _ = run_all(LLMEngine(target, config, None, num_blocks=64, block_size=4), PROMPTS, greedy)
    return tokens


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("draft", ["near", "other", "same"])
def test_greedy_output_identical_to_target_alone(models, reference, draft, k):
    engine = spec_engine(models, draft, k)
    tokens, finish = run_all(engine, PROMPTS, greedy)
    assert tokens == reference
    assert all(f == "length" for f in finish.values())
    s = engine.stats()
    assert s.num_free_blocks == 64  # nothing leaked, including the rolled-back speculative slots
    assert s.spec_proposed_tokens > 0
    if draft == "same":
        assert s.spec_accepted_tokens == s.spec_proposed_tokens
    if draft == "near":
        assert 0 < s.spec_accepted_tokens < s.spec_proposed_tokens


def test_fewer_target_forwards_when_the_draft_agrees(models, reference):
    engine = spec_engine(models, "same", 4)
    engine.add_request("a", PROMPTS["r3"], greedy("r3"))
    steps = 0
    while engine.has_unfinished():
        engine.step()
        steps += 1
    # 1 prefill step, then each target forward yields up to 5 tokens.
    assert steps <= 1 + -(-MAX_TOKENS["r3"] // 5) + 1


def test_preemption_with_speculation(models, reference):
    engine = spec_engine(models, "near", 3, num_blocks=16)  # 64 slots: forces preemption
    tokens, _ = run_all(engine, PROMPTS, greedy)
    assert engine.stats().num_preemptions_total > 0
    assert tokens == reference


def test_chunked_prefill_with_speculation(models, reference):
    tokens, _ = run_all(spec_engine(models, "near", 3, max_batched_tokens=7), PROMPTS, greedy)
    assert tokens == reference


def test_prefix_caching_with_speculation(models):
    target, config = models["target"]
    shared = list(range(3, 23))
    prompts = {f"s{i}": [*shared, 30 + i, 40 + i] for i in range(6)}

    def params(_):
        return SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)

    ref, _ = run_all(LLMEngine(target, config, None, num_blocks=64, block_size=4), prompts, params)
    engine = spec_engine(models, "near", 3)
    got: dict[str, list[int]] = {}
    for rid in prompts:  # one at a time, so later requests hit the cache
        out, _ = run_all(engine, {rid: prompts[rid]}, params)
        got |= out
    assert got == ref
    assert engine.stats().prefix_cache_hit_tokens > 0


def test_stops_mid_acceptance(models, reference):
    # Stop on a token the target emits mid-stream: the accepted drafts after it must not leak out.
    rid = "r6"
    stop_at = reference[rid][4]
    first = reference[rid].index(stop_at)
    engine = spec_engine(models, "same", 5)
    engine.add_request(rid, PROMPTS[rid], SamplingParams(temperature=0, max_tokens=MAX_TOKENS[rid],
                                                         ignore_eos=True, stop_token_ids=[stop_at]))
    got, reason = [], None
    while engine.has_unfinished():
        for out in engine.step():
            got += out.new_token_ids
            reason = out.finish_reason or reason
    assert got == reference[rid][:first] and reason == "stop"
    assert engine.stats().num_free_blocks == 64


def test_max_tokens_respected_exactly(models):
    engine = spec_engine(models, "same", 5)
    for n in [1, 2, 3, 7]:
        engine.add_request(f"m{n}", PROMPTS["r2"], SamplingParams(temperature=0, max_tokens=n, ignore_eos=True))
    got = {f"m{n}": 0 for n in [1, 2, 3, 7]}
    while engine.has_unfinished():
        for out in engine.step():
            got[out.request_id] += len(out.new_token_ids)
    assert got == {"m1": 1, "m2": 2, "m3": 3, "m7": 7}


def test_seeded_sampling_is_reproducible(models):
    def sampled(rid):
        return SamplingParams(temperature=0.8, top_k=50, seed=3, max_tokens=MAX_TOKENS[rid], ignore_eos=True)

    a, _ = run_all(spec_engine(models, "near", 3), PROMPTS, sampled)
    b, _ = run_all(spec_engine(models, "near", 3), PROMPTS, sampled)
    assert a == b
    assert all(len(a[r]) == MAX_TOKENS[r] for r in PROMPTS)


def test_draft_must_share_the_vocab(models, tmp_path):
    path, _ = make_tiny_hf_model("qwen2", tmp_path, vocab_size=300)
    draft, dconfig = load_model(path)
    target, config = models["target"]
    with pytest.raises(ValueError, match="vocab"):
        LLMEngine(target, config, None, num_blocks=8, draft_model=draft, draft_config=dconfig,
                  num_speculative_tokens=2)


def test_a_failing_draft_proposal_only_fails_that_request(models, monkeypatch):
    import kiln.engine.engine as engine_mod

    real = engine_mod.probs

    def flaky(logits, params):
        if params.top_k == 7:  # marks the request whose sampling blows up (e.g. NaN logits)
            raise RuntimeError("probability tensor contains inf or nan")
        return real(logits, params)

    monkeypatch.setattr(engine_mod, "probs", flaky)
    engine = spec_engine(models, "near", 3)
    engine.add_request("bad", PROMPTS["r3"], SamplingParams(temperature=1.0, top_k=7, max_tokens=10, ignore_eos=True))
    engine.add_request("ok", PROMPTS["r5"], SamplingParams(temperature=1.0, top_k=8, seed=1, max_tokens=10,
                                                           ignore_eos=True))
    finish, counts = {}, {"bad": 0, "ok": 0}
    while engine.has_unfinished():
        for out in engine.step():
            counts[out.request_id] += len(out.new_token_ids)
            if out.finished:
                finish[out.request_id] = out.finish_reason
    assert finish == {"bad": "error", "ok": "length"} and counts["ok"] == 10
    assert engine.stats().num_free_blocks == 64


def test_context_limit_includes_the_draft_model(models, tmp_path):
    path, _ = make_tiny_hf_model("qwen2", tmp_path, max_position_embeddings=32)
    draft, dconfig = load_model(path)
    target, config = models["target"]  # 128 positions
    engine = LLMEngine(target, config, None, num_blocks=64, block_size=4, draft_model=draft, draft_config=dconfig,
                       num_speculative_tokens=2)
    with pytest.raises(ValueError, match="context"):
        engine.add_request("long", list(range(3, 30)), SamplingParams(max_tokens=10))

"""Exact-match correctness against Hugging Face transformers on real checkpoints.

Run locally: pytest --runslow --runmps tests/integration -v -s
"""

import functools
import gc
import json
import time
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, GenerationConfig

from kiln.engine.engine import LLMEngine
from kiln.generate import generate
from kiln.model.kv_cache import ContiguousKVCache
from kiln.model.loader import load_model
from kiln.sampling import SamplingParams
from kiln.tokenizer import Tokenizer

PROMPTS = json.loads((Path(__file__).parent / "prompts.json").read_text())
NEW_TOKENS = 64
NEAR_TIE = 1e-3  # spec 5.10: a divergence is allowed only where the reference's top-2 margin is below this
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"]
PATTERNS = ["*.json", "*.safetensors", "*.jinja"]


def _prompt_ids(tok: Tokenizer, text: str) -> list[int]:
    chat = tok.apply_chat_template([{"role": "user", "content": text}])
    return tok.encode(chat, add_special_tokens=False)


@functools.cache
def _references(repo):
    """HF greedy continuations for every prompt, computed once per model and then freed."""
    path = Path(snapshot_download(repo, allow_patterns=PATTERNS))
    tok = Tokenizer.from_hf_dir(path)
    prompts = [_prompt_ids(tok, p) for p in PROMPTS]
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    # A clean config: checkpoints ship sampling defaults (Qwen sets repetition_penalty=1.1), and HF
    # applies a repetition penalty even when do_sample=False, so the reference would not be pure greedy.
    hf.generation_config = GenerationConfig(eos_token_id=None, pad_token_id=0)
    with torch.no_grad():
        refs = [hf.generate(torch.tensor([ids]), do_sample=False, max_new_tokens=NEW_TOKENS,
                            pad_token_id=0)[0, len(ids):].tolist() for ids in prompts]
    return path, prompts, refs, hf


def _check(repo, outs):
    """Exact match, except a divergence is allowed where HF's top-2 logit margin is a near-tie."""
    _, prompts, refs, hf = _references(repo)
    report = []
    for text, ids, ref, out in zip(PROMPTS, prompts, refs, outs, strict=True):
        if out == ref:
            continue
        i = next(j for j, (a, b) in enumerate(zip(out, ref, strict=True)) if a != b)
        with torch.no_grad():
            logits = hf(torch.tensor([ids + ref[:i]])).logits[0, -1]
        top2 = torch.topk(logits, 2).values
        margin = float(top2[0] - top2[1])
        report.append((text[:40], i, margin))
        assert margin < NEAR_TIE, f"{repo}: prompt {text[:40]!r} diverged at token {i} (margin {margin:.4g})"
    exact = len(PROMPTS) - len(report)
    print(f"\n{repo}: {exact}/{len(PROMPTS)} prompts exact; near-tie divergences: {report}")


@pytest.mark.slow
@pytest.mark.parametrize("repo", MODELS)
def test_fp32_greedy_exact(repo):
    path, prompts, _, _ = _references(repo)
    model, config = load_model(path, dtype=torch.float32, device="cpu")
    params = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)
    _check(repo, [generate(model, config, ids, params).token_ids for ids in prompts])


@pytest.mark.slow
@pytest.mark.parametrize("repo", MODELS)
def test_fp32_engine_all_prompts_at_once(repo):
    """All 12 prompts go through the engine together: ragged prefill + batched decode."""
    path, prompts, _, _ = _references(repo)
    model, config = load_model(path, dtype=torch.float32, device="cpu")
    engine = LLMEngine(model, config, None, num_blocks=128, block_size=16, max_batched_tokens=256)
    params = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)
    for i, ids in enumerate(prompts):
        engine.add_request(str(i), ids, params)
    got: dict[str, list[int]] = {str(i): [] for i in range(len(prompts))}
    while engine.has_unfinished():
        for out in engine.step():
            got[out.request_id] += out.new_token_ids
    _check(repo, [got[str(i)] for i in range(len(prompts))])
    del model
    gc.collect()


@pytest.mark.mps
@pytest.mark.slow
def test_qwen_fp16_mps_logits_close():
    path = Path(snapshot_download(MODELS[0], allow_patterns=PATTERNS))
    tok = Tokenizer.from_hf_dir(path)
    prompts = [_prompt_ids(tok, p) for p in PROMPTS]

    ref_model, config = load_model(path, dtype=torch.float32, device="cpu")
    def full(model, ids, dtype, device):
        cache = ContiguousKVCache(config, len(ids), dtype, device)
        cache.set_positions(0, len(ids))
        with torch.no_grad():
            return model(torch.tensor(ids, device=device), torch.arange(len(ids), device=device), cache)[-1]

    refs = [full(ref_model, ids, torch.float32, "cpu") for ids in prompts]
    del ref_model
    gc.collect()

    model, config = load_model(path, dtype=torch.float16, device="mps")
    worst = 0.0
    for ids, ref in zip(prompts, refs, strict=True):
        out = full(model, ids, torch.float16, "mps").cpu()
        worst = max(worst, float((out - ref).abs().max()))
        assert int(out.argmax()) == int(ref.argmax())
    print(f"\nMPS fp16 vs CPU fp32: worst max-abs logit diff {worst:.4f}; "
          f"MPS memory {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB")
    assert worst < 0.1

    # Single-stream decode speed, for the README (measured, not claimed).
    params = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    generate(model, config, prompts[0], SamplingParams(temperature=0.0, max_tokens=8))  # warm-up
    t = time.perf_counter()
    generate(model, config, prompts[0], params)
    torch.mps.synchronize()
    print(f"Qwen2.5-0.5B fp16 MPS single-stream: {128 / (time.perf_counter() - t):.1f} tokens/s")

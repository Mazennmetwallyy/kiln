import pytest
import torch

from kiln.model.kv_cache import ContiguousKVCache
from kiln.model.loader import load_model
from tests.model.tiny import make_tiny_hf_model


def full_forward(model, config, ids):
    cache = ContiguousKVCache(config, max_tokens=len(ids), dtype=torch.float32, device="cpu")
    cache.set_positions(0, len(ids))
    return model(ids, torch.arange(len(ids)), cache)


@pytest.mark.parametrize("arch", ["qwen2", "llama"])
def test_logits_match_hf(arch, tmp_path):
    path, hf_model = make_tiny_hf_model(arch, tmp_path)
    model, config = load_model(path)
    ids = torch.randint(0, 256, (20,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = hf_model(ids[None]).logits[0]
        out = full_forward(model, config, ids)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)


@pytest.mark.parametrize("arch", ["qwen2", "llama"])
def test_incremental_matches_full(arch, tmp_path):
    path, _ = make_tiny_hf_model(arch, tmp_path)
    model, config = load_model(path)
    ids = torch.randint(0, 256, (20,), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        full = full_forward(model, config, ids)
        cache = ContiguousKVCache(config, max_tokens=20, dtype=torch.float32, device="cpu")
        cache.set_positions(0, 12)
        rows = [model(ids[:12], torch.arange(12), cache)]
        for t in range(12, 20):
            cache.set_positions(t, 1)
            rows.append(model(ids[t : t + 1], torch.tensor([t]), cache))
    torch.testing.assert_close(torch.cat(rows), full, atol=1e-5, rtol=0)


def test_untied_model_loads(tmp_path):
    path, hf_model = make_tiny_hf_model("qwen2", tmp_path, tied=False)
    model, config = load_model(path)
    assert not config.tie_word_embeddings
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    ids = torch.arange(10)
    with torch.no_grad():
        torch.testing.assert_close(full_forward(model, config, ids), hf_model(ids[None]).logits[0],
                                   atol=1e-5, rtol=0)


def test_logits_indices_selects_rows(tmp_path):
    path, _ = make_tiny_hf_model("qwen2", tmp_path)
    model, config = load_model(path)
    ids = torch.arange(3, 15)
    with torch.no_grad():
        full = full_forward(model, config, ids)
        cache = ContiguousKVCache(config, max_tokens=len(ids), dtype=torch.float32, device="cpu")
        cache.set_positions(0, len(ids))
        picked = model(ids, torch.arange(len(ids)), cache, logits_indices=torch.tensor([4, 11]))
    assert picked.shape == (2, config.vocab_size)
    torch.testing.assert_close(picked, full[[4, 11]], atol=1e-6, rtol=0)

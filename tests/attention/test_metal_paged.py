"""The Metal decode kernel must match the PyTorch paged-attention reference."""

import pytest
import torch

from kiln.attention.base import sdpa_gqa

pytestmark = pytest.mark.mps


def _reference(q, k_pool, v_pool, tables, lens, bs):
    outs = []
    for b, n in enumerate(lens):
        slots = torch.tensor([tables[b][p // bs] * bs + p % bs for p in range(n)], device=q.device)
        keys, values = k_pool[slots].transpose(0, 1).float(), v_pool[slots].transpose(0, 1).float()
        outs.append(sdpa_gqa(q[b][:, None, :].float(), keys, values, None)[:, 0, :])
    return torch.stack(outs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(("heads", "kv_heads", "head_dim"), [(4, 4, 64), (8, 2, 64), (12, 2, 128), (14, 2, 64)])
def test_matches_reference(dtype, heads, kv_heads, head_dim):
    from kiln.attention.metal import paged_decode_attention

    g = torch.Generator().manual_seed(0)
    bs, num_blocks = 16, 40
    lens = [1, 3, 16, 17, 100, 250]
    perm = torch.randperm(num_blocks, generator=g).tolist()  # scattered, out-of-order blocks
    tables, used = [], 0
    for n in lens:
        need = -(-n // bs)
        tables.append(perm[used:used + need])
        used += need
    width = max(len(t) for t in tables)
    pool_shape = (num_blocks * bs, kv_heads, head_dim)
    k_pool = torch.randn(pool_shape, generator=g).to("mps", dtype)
    v_pool = torch.randn(pool_shape, generator=g).to("mps", dtype)
    q = torch.randn(len(lens), heads, head_dim, generator=g).to("mps", dtype)
    t = torch.tensor([tb + [0] * (width - len(tb)) for tb in tables], dtype=torch.int32, device="mps")
    n = torch.tensor(lens, dtype=torch.int32, device="mps")
    got = paged_decode_attention(q, k_pool, v_pool, t, n, bs).float()
    want = _reference(q, k_pool, v_pool, tables, lens, bs)
    tol = 1e-5 if dtype == torch.float32 else 2e-3
    assert torch.allclose(got, want, atol=tol, rtol=tol), (got - want).abs().max()


def test_engine_greedy_identical_with_metal_backend(tmp_path):
    from kiln.engine.engine import LLMEngine
    from kiln.model.loader import load_model
    from tests.engine.test_engine import PROMPTS, greedy, run_all
    from tests.model.tiny import make_tiny_hf_model

    torch.set_grad_enabled(False)
    # head_dim 64 (the kernel needs a multiple of 32), grouped-query attention 4:2
    path, _ = make_tiny_hf_model("qwen2", tmp_path, hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
    model, config = load_model(path, dtype=torch.float32, device="mps")
    results = {}
    for backend in ["torch", "metal"]:
        for blocks in [64, 16]:  # 16 blocks forces preemption
            engine = LLMEngine(model, config, None, num_blocks=blocks, block_size=4, attention_backend=backend)
            results[backend, blocks], _ = run_all(engine, PROMPTS, greedy)
        engine = LLMEngine(model, config, None, num_blocks=64, block_size=4, attention_backend=backend,
                           draft_model=model, draft_config=config, num_speculative_tokens=3)
        results[backend, "spec"], _ = run_all(engine, PROMPTS, greedy)
    ref = results["torch", 64]
    assert len({len(set(v)) for v in ref.values()}) > 1  # outputs vary, so wrong K/V would show
    for key, got in results.items():
        assert got == ref, key


def test_metal_backend_rejects_unsupported_setups(tmp_path):
    from kiln.engine.engine import LLMEngine
    from kiln.model.loader import load_model
    from tests.model.tiny import make_tiny_hf_model

    path, _ = make_tiny_hf_model("qwen2", tmp_path)  # head_dim 16
    model, config = load_model(path, dtype=torch.float32, device="mps")
    with pytest.raises(ValueError, match="metal"):
        LLMEngine(model, config, None, num_blocks=8, attention_backend="metal")

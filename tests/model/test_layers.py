import torch
from transformers import Qwen2Config
from transformers.models.qwen2 import modeling_qwen2 as hf

from kiln.model.layers import RMSNorm, RotaryEmbedding, SwiGLU

torch.manual_seed(0)


def test_rmsnorm_matches_formula():
    norm = RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
    x = torch.randn(5, 16)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=0)


def test_rmsnorm_fp16_input_returns_fp16():
    norm = RMSNorm(16, eps=1e-6).half()
    x = torch.randn(3, 16).half()
    out = norm(x)
    assert out.dtype == torch.float16
    ref = RMSNorm(16, eps=1e-6)(x.float())
    torch.testing.assert_close(out.float(), ref, atol=2e-3, rtol=0)


def _qk(t=6, h=4, hkv=2, d=32):
    return torch.randn(t, h, d), torch.randn(t, hkv, d)


def test_rope_position_zero_is_identity():
    rope = RotaryEmbedding(32, theta=10000.0, max_positions=64)
    q, k = _qk(t=1)
    q2, k2 = rope(q, k, torch.tensor([0]))
    torch.testing.assert_close(q2, q)
    torch.testing.assert_close(k2, k)


def test_rope_preserves_norm():
    rope = RotaryEmbedding(32, theta=10000.0, max_positions=64)
    q, k = _qk()
    q2, _ = rope(q, k, torch.arange(6) * 7)
    torch.testing.assert_close(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=0)


def test_rope_relative():
    rope = RotaryEmbedding(32, theta=10000.0, max_positions=64)
    q = torch.randn(1, 1, 32)
    k = torch.randn(1, 1, 32)

    def score(m, n):
        qm, _ = rope(q, k, torch.tensor([m]))
        _, kn = rope(q, k, torch.tensor([n]))
        return (qm * kn).sum()

    torch.testing.assert_close(score(3, 1), score(12, 10), atol=1e-5, rtol=0)


def test_layers_match_hf():
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, rope_theta=10000.0, rms_norm_eps=1e-6)
    # RMSNorm
    hf_norm = hf.Qwen2RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        hf_norm.weight.copy_(torch.randn(64))
    norm = RMSNorm(64, eps=1e-6)
    norm.load_state_dict(hf_norm.state_dict())
    x = torch.randn(7, 64)
    torch.testing.assert_close(norm(x), hf_norm(x), atol=1e-6, rtol=0)

    # MLP
    hf_mlp = hf.Qwen2MLP(cfg)
    mlp = SwiGLU(64, 128)
    mlp.load_state_dict(hf_mlp.state_dict())
    torch.testing.assert_close(mlp(x), hf_mlp(x), atol=1e-6, rtol=0)

    # RoPE: HF works on [B, H, T, D]
    hf_rope = hf.Qwen2RotaryEmbedding(cfg)
    q, k = _qk(t=7, h=4, hkv=2, d=16)
    pos = torch.tensor([0, 1, 2, 5, 9, 30, 63])
    cos, sin = hf_rope(x[None], pos[None])
    hq, hk = hf.apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
    rope = RotaryEmbedding(16, theta=10000.0, max_positions=64)
    q2, k2 = rope(q, k, pos)
    torch.testing.assert_close(q2, hq[0].transpose(0, 1), atol=1e-6, rtol=0)
    torch.testing.assert_close(k2, hk[0].transpose(0, 1), atol=1e-6, rtol=0)

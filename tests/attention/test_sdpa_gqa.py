import pytest
import torch
import torch.nn.functional as F

from kiln.attention.base import sdpa_gqa


def reference(q, k, v, mask):
    g = q.shape[-3] // k.shape[-3]
    return F.scaled_dot_product_attention(q, k.repeat_interleave(g, -3), v.repeat_interleave(g, -3), attn_mask=mask)


@pytest.mark.parametrize("heads", [(14, 2), (9, 3), (4, 4)])
def test_prefill_shape_with_causal_mask(heads):
    h, hkv = heads
    torch.manual_seed(0)
    q, k, v = torch.randn(h, 7, 16), torch.randn(hkv, 12, 16), torch.randn(hkv, 12, 16)
    pos = torch.arange(5, 12)
    mask = torch.arange(12)[None, :] <= pos[:, None]
    torch.testing.assert_close(sdpa_gqa(q, k, v, mask), reference(q, k, v, mask), atol=1e-5, rtol=0)


@pytest.mark.parametrize("heads", [(14, 2), (9, 3)])
def test_batched_decode_with_length_mask(heads):
    h, hkv = heads
    torch.manual_seed(1)
    q, k, v = torch.randn(5, h, 1, 16), torch.randn(5, hkv, 20, 16), torch.randn(5, hkv, 20, 16)
    lens = torch.tensor([3, 20, 7, 1, 15])
    mask = (torch.arange(20)[None, :] < lens[:, None])[:, None, None, :]
    torch.testing.assert_close(sdpa_gqa(q, k, v, mask), reference(q, k, v, mask), atol=1e-5, rtol=0)


def test_non_contiguous_query():
    torch.manual_seed(2)
    q = torch.randn(7, 14, 16).transpose(0, 1)  # [14, 7, 16], non-contiguous, as attention passes it
    k, v = torch.randn(2, 7, 16), torch.randn(2, 7, 16)
    mask = torch.ones(7, 7, dtype=torch.bool).tril()
    torch.testing.assert_close(sdpa_gqa(q, k, v, mask), reference(q, k, v, mask), atol=1e-5, rtol=0)

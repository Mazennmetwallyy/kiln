import pytest
import torch

from kiln.sampling import SamplingParams, sample

LOGITS = torch.tensor([1.0, 3.0, 2.0, -1.0, 0.5])


def test_greedy_is_argmax():
    assert sample(LOGITS, SamplingParams(temperature=0.0), None) == 1


def test_top_k_1_equals_greedy():
    g = torch.Generator().manual_seed(0)
    assert all(sample(LOGITS, SamplingParams(top_k=1), g) == 1 for _ in range(50))


def test_top_p_keeps_at_least_one():
    g = torch.Generator().manual_seed(0)
    peaked = torch.tensor([10.0, 0.0, 0.0])  # p(0) ≈ 0.9999 > top_p
    assert all(sample(peaked, SamplingParams(top_p=0.01), g) == 0 for _ in range(50))


def test_seed_reproducible():
    def draws(seed):
        g = torch.Generator().manual_seed(seed)
        return [sample(LOGITS, SamplingParams(), g) for _ in range(50)]

    assert draws(7) == draws(7)
    assert draws(7) != draws(8)


def test_top_p_distribution():
    logits = torch.tensor([0.5, 0.3, 0.2]).log()
    g = torch.Generator().manual_seed(0)
    draws = [sample(logits, SamplingParams(top_p=0.6), g) for _ in range(2000)]
    assert set(draws) == {0, 1}
    assert abs(draws.count(0) / 2000 - 0.625) < 0.05


def test_temperature_sharpens():
    g = torch.Generator().manual_seed(0)
    cold = [sample(LOGITS, SamplingParams(temperature=0.05), g) for _ in range(200)]
    assert cold.count(1) > 195


@pytest.mark.parametrize("kw", [{"temperature": -0.1}, {"top_p": 0.0}, {"top_p": 1.5}, {"top_k": -1},
                                {"max_tokens": -1}])
def test_invalid_params_raise(kw):
    with pytest.raises(ValueError):
        SamplingParams(**kw)

import torch

from kiln.sampling import SamplingParams, probs


def test_probs_is_normalised_softmax_at_temperature():
    logits = torch.tensor([1.0, 2.0, 3.0])
    p = probs(logits, SamplingParams(temperature=0.5))
    assert torch.allclose(p, torch.softmax(logits / 0.5, -1))


def test_probs_top_k_and_top_p_renormalise():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    p = probs(logits, SamplingParams(temperature=1.0, top_k=2))
    assert p[:2].sum() == 0 and torch.isclose(p.sum(), torch.tensor(1.0))
    p = probs(logits, SamplingParams(temperature=1.0, top_p=0.5))
    assert torch.isclose(p.sum(), torch.tensor(1.0)) and p[3] > 0


def test_probs_greedy_is_one_hot():
    p = probs(torch.tensor([0.1, 5.0, 0.2]), SamplingParams(temperature=0))
    assert p.tolist() == [0.0, 1.0, 0.0]

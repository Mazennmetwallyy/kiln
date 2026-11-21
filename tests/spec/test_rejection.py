"""Modified rejection sampling must reproduce the target distribution exactly (Leviathan et al. 2023)."""

import pytest
import torch

from kiln.spec.rejection import verify

V = 6


def test_greedy_accepts_matching_prefix_then_corrects():
    p = torch.eye(V)[[2, 4, 1, 5]]  # target argmax at each of the k+1 = 4 positions
    j, tok = verify([2, 4, 3], None, p, greedy=True, generator=None)
    assert (j, tok) == (2, 1)  # d1, d2 accepted; d3 = 3 rejected; the target's own token 1 follows


def test_greedy_all_accepted_takes_bonus_token():
    p = torch.eye(V)[[2, 4, 1, 5]]
    assert verify([2, 4, 1], None, p, greedy=True, generator=None) == (3, 5)


def test_greedy_first_rejected():
    p = torch.eye(V)[[2, 4]]
    assert verify([0], None, p, greedy=True, generator=None) == (0, 2)


def _chi_square(counts: torch.Tensor, expected: torch.Tensor) -> float:
    mask = expected > 0
    assert counts[~mask].sum() == 0, "sampled a token the target gives zero probability"
    return float(((counts[mask] - expected[mask]) ** 2 / expected[mask]).sum())


@pytest.mark.parametrize("case", ["random", "draft_misses_support", "target_sparse"])
def test_first_emitted_token_follows_the_target(case):
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(3, V, generator=g) * 1.5, -1)  # k = 2
    q = torch.softmax(torch.randn(2, V, generator=g) * 1.5, -1)
    if case == "draft_misses_support":
        q[:, :2] = 0  # the draft can never propose tokens 0 and 1, which the target likes
        q /= q.sum(-1, keepdim=True)
    if case == "target_sparse":
        p[:, 3:] = 0  # the draft proposes tokens the target never allows
        p /= p.sum(-1, keepdim=True)
    n = 20000
    first = torch.zeros(V)
    second = torch.zeros(V)
    n_second = 0
    for _ in range(n):
        drafts = [int(torch.multinomial(q[i], 1, generator=g)) for i in range(2)]
        j, tok = verify(drafts, q, p, greedy=False, generator=g)
        emitted = [*drafts[:j], tok]
        first[emitted[0]] += 1
        if len(emitted) > 1:  # d1 was accepted; the next token must again follow the target
            second[emitted[1]] += 1
            n_second += 1
    # 5 degrees of freedom: chi-square 99.9th percentile is 20.5
    assert _chi_square(first, p[0] * n) < 20.5
    assert n_second > 1000
    assert _chi_square(second, p[1] * n_second) < 20.5


def test_no_drafts_samples_from_the_target():
    # k = 0 happens on a request's last token: there is no room to speculate, so just sample from p.
    p = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    assert verify([], None, p, greedy=False, generator=torch.Generator().manual_seed(0)) == (0, 2)

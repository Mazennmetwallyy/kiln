import itertools
import math

from bench.loadgen import RequestResult, percentile, poisson_arrivals, summarize


def test_percentile():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 100) == 5.0
    assert percentile(xs, 0) == 1.0
    assert math.isclose(percentile(xs, 90), 4.6)
    assert math.isnan(percentile([], 50))


def test_poisson_arrivals_deterministic_and_rate():
    a = poisson_arrivals(2000, rate=4.0, seed=0)
    assert a == poisson_arrivals(2000, rate=4.0, seed=0)
    assert a[0] == 0.0 and all(x <= y for x, y in itertools.pairwise(a))
    assert abs(a[-1] / 1999 - 0.25) < 0.02  # mean gap ≈ 1/rate


def test_infinite_rate_sends_everything_at_once():
    assert poisson_arrivals(5, rate=math.inf, seed=0) == [0.0] * 5


def test_summarize():
    rs = [RequestResult(ttft=0.1, itls=[0.02, 0.04], e2e=1.0, output_tokens=10),
          RequestResult(ttft=0.3, itls=[0.06], e2e=2.0, output_tokens=30)]
    s = summarize(rs, duration=4.0)
    assert s["completed"] == 2 and s["request_throughput"] == 0.5 and s["output_throughput"] == 10.0
    assert s["ttft_p50"] == 0.2 and math.isclose(s["itl_p50"], 0.04) and s["e2e_p50"] == 1.5

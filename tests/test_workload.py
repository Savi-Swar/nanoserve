"""Tests for bench.workload.build_requests (Poisson open-loop load gen)."""
from bench.workload import build_requests


def test_returns_exactly_n_requests():
    reqs, offsets = build_requests(n=50, rate=10, max_tokens=64, seed=1)
    assert len(reqs) == 50
    assert len(offsets) == 50


def test_same_seed_is_deterministic():
    reqs_a, off_a = build_requests(n=100, rate=10, max_tokens=64, seed=7)
    reqs_b, off_b = build_requests(n=100, rate=10, max_tokens=64, seed=7)
    assert off_a == off_b
    assert [r.prompt for r in reqs_a] == [r.prompt for r in reqs_b]
    assert [r.sampling.max_tokens for r in reqs_a] == [
        r.sampling.max_tokens for r in reqs_b
    ]


def test_different_seed_differs():
    _, off_a = build_requests(n=100, rate=10, max_tokens=64, seed=1)
    _, off_b = build_requests(n=100, rate=10, max_tokens=64, seed=2)
    assert off_a != off_b


def test_offsets_strictly_increasing():
    _, offsets = build_requests(n=200, rate=10, max_tokens=64, seed=3)
    for prev, cur in zip(offsets, offsets[1:]):
        assert cur > prev


def test_jitter_tokens_within_bounds():
    mt = 64
    reqs, _ = build_requests(
        n=500, rate=10, max_tokens=mt, seed=4, jitter_tokens=True
    )
    lo = max(8, int(mt * 0.5))
    hi = int(mt * 1.5)
    for r in reqs:
        assert lo <= r.sampling.max_tokens <= hi


def test_no_jitter_uses_exact_max_tokens():
    mt = 64
    reqs, _ = build_requests(
        n=50, rate=10, max_tokens=mt, seed=5, jitter_tokens=False
    )
    assert all(r.sampling.max_tokens == mt for r in reqs)


def test_arrival_rate_matches_roughly():
    n, rate = 2000, 10
    _, offsets = build_requests(n=n, rate=rate, max_tokens=64, seed=6)
    # mean inter-arrival gap should approximate 1/rate for a Poisson process
    gaps = [b - a for a, b in zip([0.0] + offsets[:-1], offsets)]
    mean_gap = sum(gaps) / len(gaps)
    assert abs(mean_gap - 1.0 / rate) < 0.02


def test_temperature_propagates_to_sampling():
    reqs, _ = build_requests(
        n=10, rate=10, max_tokens=64, seed=8, temperature=0.7
    )
    assert all(r.sampling.temperature == 0.7 for r in reqs)

"""Tests for server.model.sample (greedy + stochastic token selection)."""
import torch

from server.model import sample
from server.request import SamplingParams


def test_greedy_returns_argmax_deterministically():
    logits = torch.tensor([[0.1, 3.7, 0.2, 1.5, 0.0]])
    params = SamplingParams(temperature=0.0)
    expected = int(logits.argmax(-1))
    # greedy must be stable across repeated calls
    results = {sample(logits, params) for _ in range(20)}
    assert results == {expected}
    assert expected == 1


def test_greedy_property_true_for_zero_temperature():
    assert SamplingParams(temperature=0.0).greedy is True
    assert SamplingParams(temperature=-1.0).greedy is True
    assert SamplingParams(temperature=1.0).greedy is False


def test_one_hot_huge_logit_dominates_sampling():
    # A single enormous logit makes the softmax ~one-hot, so even stochastic
    # sampling (temperature=1.0) should return that index.
    vocab = 32
    logits = torch.full((1, vocab), -50.0)
    target = 17
    logits[0, target] = 100.0
    params = SamplingParams(temperature=1.0)
    for _ in range(20):
        assert sample(logits, params) == target


def test_one_hot_with_top_p_still_returns_target():
    vocab = 16
    logits = torch.full((1, vocab), -50.0)
    target = 5
    logits[0, target] = 100.0
    params = SamplingParams(temperature=1.0, top_p=0.9)
    for _ in range(20):
        assert sample(logits, params) == target

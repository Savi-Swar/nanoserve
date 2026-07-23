"""Tests for bench.metrics.pct (linear-interpolated percentile)."""
from bench.metrics import pct


def test_p50_linear_interpolation():
    # sorted [10,20,30,40], p50 -> k=(4-1)*0.5=1.5 -> 20 + (30-20)*0.5 = 25
    assert pct([10, 20, 30, 40], 50) == 25.0


def test_p50_unsorted_input():
    # pct sorts internally, so order shouldn't matter
    assert pct([40, 10, 30, 20], 50) == 25.0


def test_p0_is_min():
    assert pct([10, 20, 30, 40], 0) == 10.0


def test_p100_is_max():
    assert pct([10, 20, 30, 40], 100) == 40.0


def test_single_element_list():
    assert pct([42.0], 50) == 42.0
    assert pct([42.0], 0) == 42.0
    assert pct([42.0], 100) == 42.0


def test_empty_list_returns_zero():
    assert pct([], 50) == 0.0
    assert pct([], 99) == 0.0


def test_intermediate_percentile():
    # sorted [0,1,2,3,4], p90 -> k=(5-1)*0.9=3.6 -> 3 + (4-3)*0.6 = 3.6
    assert pct([0, 1, 2, 3, 4], 90) == 3.6

"""Tests for server.request.Request latency properties."""
import pytest

from server.request import Request, SamplingParams


def make_request():
    return Request(id=0, prompt="hi", sampling=SamplingParams())


def test_latency_properties_compute_correctly():
    r = make_request()
    r.arrival_time = 100.0
    r.schedule_time = 100.5
    r.first_token_time = 101.0
    r.finish_time = 105.0
    # 5 generated tokens
    r.output_tokens = [1, 2, 3, 4, 5]

    assert r.num_output == 5
    assert r.ttft == pytest.approx(1.0)          # first_token - arrival
    assert r.queue_delay == pytest.approx(0.5)   # schedule - arrival
    assert r.e2e == pytest.approx(5.0)           # finish - arrival
    # decode_tps: (num_output - 1) / (finish - first_token) = 4 / 4.0 = 1.0
    assert r.decode_tps == pytest.approx(1.0)


def test_decode_tps_zero_when_single_token():
    r = make_request()
    r.arrival_time = 0.0
    r.first_token_time = 1.0
    r.finish_time = 2.0
    r.output_tokens = [1]  # n - 1 == 0
    assert r.decode_tps == 0.0


def test_decode_tps_zero_when_no_gen_time():
    r = make_request()
    r.arrival_time = 0.0
    r.first_token_time = 1.0
    r.finish_time = 1.0  # gen == 0
    r.output_tokens = [1, 2, 3]
    assert r.decode_tps == 0.0


def test_num_output_empty():
    r = make_request()
    assert r.num_output == 0


def test_sampling_greedy_flag():
    assert make_request().sampling.greedy is True


def test_tpot_and_meets_slo():
    r = make_request()
    r.arrival_time = 0.0
    r.first_token_time = 0.2       # TTFT = 200 ms
    r.finish_time = 0.6            # 4 decode intervals over 0.4s -> 100 ms/token
    r.output_tokens = [1, 2, 3, 4, 5]
    assert r.tpot == pytest.approx(0.1)          # 0.4s / 4 tokens
    assert r.meets_slo(ttft_s=0.5, tpot_s=0.15)  # both under SLO
    assert not r.meets_slo(ttft_s=0.1, tpot_s=0.15)  # TTFT over
    assert not r.meets_slo(ttft_s=0.5, tpot_s=0.05)  # TPOT over

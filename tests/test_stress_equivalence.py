"""Adversarial equivalence: try to BREAK batched/paged decoding against the
naive single-sequence baseline. Batched inference is a minefield: left padding,
attention-mask offsets, KV-cache write positions, RoPE positions, block
boundaries. These cases are the ones most likely to expose an offset bug:
1-token prompts, permutation across batch positions, extreme length skew,
duplicates, and paged execution with block_size down to 1.

Why this asserts token-identity, not bit-identity of logits: under greedy
(temperature 0) the emitted token is argmax(logits). Batching changes the float
reduction order, so logits differ in the last ~1e-4. That flips the argmax only
on a near-tie. These prompts don't hit a near-tie, so identity holds; on a
genuinely tied logit, batched and naive may diverge by a token. Batching is
exact up to float non-associativity, not bit-identical.

Guarded (loads the model): RUN_SLOW=1 python -m pytest tests/test_stress_equivalence.py -q
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 (loads the model)"
)

MODEL = "Qwen/Qwen2.5-0.5B"
N = 16
POOL = ["x", ".", "Hi", "Hi.", "1+1=", "The cat", "A B C D", "Once upon a time",
        "print('hi')", "Red green blue.", "Yes or no? Answer:", "2 3 5 7 11",
        "What is the capital of France?", "Explain gravity in one sentence please."]


@pytest.fixture(scope="module")
def ctx():
    from server.model import ModelRunner, sample
    from server.request import Request, SamplingParams
    m = ModelRunner(MODEL, device="cpu")
    m.warmup()

    def naive(prompt):
        sp = SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True)
        ids = m.encode(prompt)
        logits, kv, cur = m.prefill(ids)
        toks = [sample(logits, sp)]
        while len(toks) < N:
            logits, kv, cur = m.decode(toks[-1], kv, cur)
            toks.append(sample(logits, sp))
        return toks

    ref = {p: naive(p) for p in POOL}
    return m, Request, SamplingParams, ref


def _run(m, Request, SamplingParams, prompts, state):
    reqs = [Request(i, p, SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True))
            for i, p in enumerate(prompts)]
    state.add(reqs)
    while state.any_active:
        state.step()
    return reqs


def test_short_prompts_and_skew_contiguous(ctx):
    from server.batched import BatchState
    m, Request, SamplingParams, ref = ctx
    reqs = _run(m, Request, SamplingParams, POOL, BatchState(m))
    for r in reqs:
        assert r.output_tokens == ref[r.prompt], f"{r.prompt!r} len={len(m.encode(r.prompt))}"


def test_permutation_invariance(ctx):
    from server.batched import BatchState
    m, Request, SamplingParams, ref = ctx
    reqs = _run(m, Request, SamplingParams, POOL[::-1], BatchState(m))
    for r in reqs:
        assert r.output_tokens == ref[r.prompt], f"position-dependent output for {r.prompt!r}"


def test_paged_tiny_blocks(ctx):
    from server.paged_exec import PagedBatchState
    m, Request, SamplingParams, ref = ctx
    for bs in (1, 2, 3, 4):
        reqs = _run(m, Request, SamplingParams, POOL, PagedBatchState(m, num_blocks=4096, block_size=bs))
        for r in reqs:
            assert r.output_tokens == ref[r.prompt], f"block_size={bs} {r.prompt!r}"

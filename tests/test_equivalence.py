"""Equivalence oracle: every batched/paged engine must decode token-for-token
identically to naive single-sequence decoding under greedy. Makes "batching is
an optimization, not an approximation" a checked claim.

Loads the model, so it's guarded. Run with:
    RUN_SLOW=1 python -m pytest tests/test_equivalence.py -q
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 (loads the model)"
)

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPTS = [
    "What is the capital of Japan and roughly how many people live there?",
    "List three prime numbers.",
    "Write a haiku about winter.",
]
N = 20


def _reference(m, sample, SamplingParams):
    out = []
    for p in PROMPTS:
        sp = SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True)
        ids = m.encode(p)
        logits, kv, cur = m.prefill(ids)
        toks = [sample(logits, sp)]
        while len(toks) < N:
            logits, kv, cur = m.decode(toks[-1], kv, cur)
            toks.append(sample(logits, sp))
        out.append(toks)
    return out


def _mkreqs(Request, SamplingParams):
    return [Request(i, p, SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True))
            for i, p in enumerate(PROMPTS)]


@pytest.fixture(scope="module")
def ctx():
    from server.model import ModelRunner, sample
    from server.request import Request, SamplingParams
    m = ModelRunner(MODEL, device="cpu")
    m.warmup()
    return m, sample, Request, SamplingParams, _reference(m, sample, SamplingParams)


def test_contiguous_batch_equals_naive(ctx):
    from server.batched import BatchState
    m, sample, Request, SamplingParams, ref = ctx
    st = BatchState(m)
    reqs = _mkreqs(Request, SamplingParams)
    st.add(reqs)
    while st.any_active:
        st.step()
    for i in range(len(PROMPTS)):
        assert reqs[i].output_tokens == ref[i]


def test_midstream_admission_equals_naive(ctx):
    from server.batched import BatchState
    m, sample, Request, SamplingParams, ref = ctx
    st = BatchState(m)
    reqs = _mkreqs(Request, SamplingParams)
    st.add(reqs[:2])
    for _ in range(5):
        st.step()
    st.add([reqs[2]])
    while st.any_active:
        st.step()
    for i in range(len(PROMPTS)):
        assert reqs[i].output_tokens == ref[i]


def test_paged_execution_equals_naive(ctx):
    from server.paged_exec import PagedBatchState
    m, sample, Request, SamplingParams, ref = ctx
    st = PagedBatchState(m, num_blocks=256, block_size=8)
    reqs = _mkreqs(Request, SamplingParams)
    st.add(reqs)
    while st.any_active:
        st.step()
    for i in range(len(PROMPTS)):
        assert reqs[i].output_tokens == ref[i]


def test_speculative_equals_naive(ctx):
    """Speculative decoding is exact: it changes forward-pass count, not output."""
    from server.speculative import SpeculativeEngine
    m, sample, Request, SamplingParams, ref = ctx
    eng = SpeculativeEngine(m, ngram=3, draft=8)
    for i, p in enumerate(PROMPTS):
        req = Request(i, p, SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True))
        eng._process(req)  # synchronous
        assert req.output_tokens == ref[i], f"spec != naive for {p!r}"


def test_batched_spec_equals_naive(ctx):
    """Speculative decoding inside a continuous batch stays token-exact."""
    from server.spec_batched import SpecPagedState
    m, sample, Request, SamplingParams, ref = ctx
    st = SpecPagedState(m, num_blocks=4096, block_size=16, ngram=3, draft=8)
    reqs = _mkreqs(Request, SamplingParams)
    st.add(reqs)
    while st.any_active:
        st.step()
    for i in range(len(PROMPTS)):
        assert reqs[i].output_tokens == ref[i], f"spec_cont != naive for {PROMPTS[i]!r}"


def test_prefix_cache_equals_naive(ctx):
    """Reusing a shared prefix's KV must not change the output."""
    from server.prefix_cache import PrefixCache
    m, sample, Request, SamplingParams, ref = ctx
    cache = PrefixCache(m)
    # prime the cache with prompt 0, then a prompt that shares its whole prefix
    shared = [PROMPTS[0], PROMPTS[0] + " Also, briefly, why?"]
    naive_ref = {}
    for p in shared:
        ids = m.encode(p)
        logits, kv, cur = m.prefill(ids)
        toks = [int(logits.argmax(-1))]
        while len(toks) < N:
            logits, kv, cur = m.decode(toks[-1], kv, cur)
            toks.append(int(logits.argmax(-1)))
        naive_ref[p] = toks
    for p in shared:
        ids = m.encode(p)
        logits, kv, cur = cache.prefill(ids)
        toks = [int(logits.argmax(-1))]
        while len(toks) < N:
            logits, kv, cur = m.decode(toks[-1], kv, cur)
            toks.append(int(logits.argmax(-1)))
        assert toks == naive_ref[p], f"prefix-cached != naive for {p!r}"
    assert cache.hits >= 1  # the second prompt must have reused the prefix

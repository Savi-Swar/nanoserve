"""CUDA-graph decode vs the eager fused path: same tokens, live state.

CUDA only (graphs don't exist off-GPU; on CPU the graph engine already
degrades to the fused step, covered by the CPU suites). The test drives real
generation far enough that sequences cross block boundaries and rows leave
and re-enter buckets, which is where stale static buffers would show."""
import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a small town",
]
N = 40


@pytest.fixture(scope="module")
def runner():
    from server.model import ModelRunner
    return ModelRunner(MODEL, device="cuda")


def batch_tokens(runner, graphs, prompts=PROMPTS, n=N):
    from server.kernels.paged_attention_triton import (restore_attention,
                                                       use_triton_attention)
    from server.paged_exec import PagedBatchState
    from server.request import Request, SamplingParams

    prev = use_triton_attention(runner.model)
    try:
        state = PagedBatchState(runner, num_blocks=512, block_size=16,
                                fused=True, graphs=graphs,
                                graph_buckets=[1, 2, 4])
        reqs = [Request(i, p, SamplingParams(max_tokens=n, temperature=0.0,
                                             ignore_eos=True))
                for i, p in enumerate(prompts)]
        state.add(reqs)
        while state.any_active:
            finished = state.step()
            if finished:
                state.evict(finished)
    finally:
        restore_attention(runner.model, prev)
    return [r.output_tokens for r in reqs]


def test_graphed_matches_fused(runner):
    want = batch_tokens(runner, graphs=False)
    got = batch_tokens(runner, graphs=True)
    for p, a, b in zip(PROMPTS, want, got):
        assert a == b, f"graph divergence on {p!r}:\n fused {a}\n graph {b}"


def test_graphed_midstream_admit(runner):
    """Rows joining mid-decode change the bucket; the replayed graph must see
    the new row's tables/lens through the static buffers."""
    from server.kernels.paged_attention_triton import (restore_attention,
                                                       use_triton_attention)
    from server.paged_exec import PagedBatchState
    from server.request import Request, SamplingParams

    sp = lambda: SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True)
    want = batch_tokens(runner, graphs=False, prompts=PROMPTS[:2], n=N)

    prev = use_triton_attention(runner.model)
    try:
        state = PagedBatchState(runner, num_blocks=512, block_size=16,
                                fused=True, graphs=True, graph_buckets=[1, 2, 4])
        ra = Request(0, PROMPTS[0], sp())
        state.add([ra])
        for _ in range(5):
            state.step()
        rb = Request(1, PROMPTS[1], sp())
        state.add([rb])
        while state.any_active:
            finished = state.step()
            if finished:
                state.evict(finished)
    finally:
        restore_attention(runner.model, prev)
    assert ra.output_tokens == want[0]
    assert rb.output_tokens == want[1]

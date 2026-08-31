"""Fused (no-gather) paged decode vs naive: token-exact through the real model.

Runs on CPU: the fused path's plumbing (NanoPagedCache pool writes, block
tables, lengths, positions, the attention handle) is identical on CPU and
CUDA; only the attention math swaps (gather+SDPA fallback vs Triton kernel).
So this test pins the whole M2 integration except the kernel itself, which
tests/test_kernel_equivalence.py covers on GPU.

Guarded (loads the model): RUN_SLOW=1 python -m pytest tests/test_fused_paged.py -q
"""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 (loads the model)"
)

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time in a small town",
    "The three laws of thermodynamics state",
]
N = 24


@pytest.fixture(scope="module")
def runner():
    from server.model import ModelRunner
    return ModelRunner(MODEL, device="cpu")


def naive_greedy(runner, prompt, n):
    from server.model import sample
    from server.request import SamplingParams
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    ids = runner.encode(prompt)
    logits, kv, cur = runner.prefill(ids)
    tok = sample(logits, sp)
    out = [tok]
    for _ in range(n - 1):
        logits, kv, cur = runner.decode(tok, kv, cur)
        tok = sample(logits, sp)
        out.append(tok)
    return out


def fused_batch_tokens(runner, prompts, n):
    from server.kernels.paged_attention_triton import use_triton_attention, restore_attention
    from server.paged_exec import PagedBatchState
    from server.request import Request, SamplingParams

    prev = use_triton_attention(runner.model)
    try:
        state = PagedBatchState(runner, num_blocks=512, block_size=16, fused=True)
        reqs = [Request(i, p, SamplingParams(max_tokens=n, temperature=0.0,
                                             ignore_eos=True))
                for i, p in enumerate(prompts)]
        state.add(reqs)
        while state.any_active:
            finished = state.step()
            if finished:                      # engine contract: evict promptly
                state.evict(finished)
        return [r.output_tokens for r in reqs]
    finally:
        restore_attention(runner.model, prev)


def test_fused_paged_matches_naive(runner):
    want = [naive_greedy(runner, p, N) for p in PROMPTS]
    got = fused_batch_tokens(runner, PROMPTS, N)
    for p, a, b in zip(PROMPTS, want, got):
        assert a == b, f"fused divergence on {p!r}:\n naive {a}\n fused {b}"


def test_fused_midstream_admit(runner):
    """Admitting a request while others are mid-decode must not disturb
    anyone's tokens (block tables grow, batch reshapes)."""
    from server.kernels.paged_attention_triton import use_triton_attention, restore_attention
    from server.paged_exec import PagedBatchState
    from server.request import Request, SamplingParams

    sp = lambda: SamplingParams(max_tokens=N, temperature=0.0, ignore_eos=True)
    want_a = naive_greedy(runner, PROMPTS[0], N)
    want_b = naive_greedy(runner, PROMPTS[1], N)

    prev = use_triton_attention(runner.model)
    try:
        state = PagedBatchState(runner, num_blocks=512, block_size=16, fused=True)
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

    assert ra.output_tokens == want_a
    assert rb.output_tokens == want_b


def test_cpp_alloc_backend_matches_naive(runner):
    """Same tokens with block bookkeeping in nanoserve_core: the served-token
    counterpart of the replay harness's decision-hash equivalence."""
    pytest.importorskip("nanoserve_core")
    from server.kernels.paged_attention_triton import (restore_attention,
                                                       use_triton_attention)
    from server.paged_exec import PagedBatchState
    from server.request import Request, SamplingParams

    want = [naive_greedy(runner, p, N) for p in PROMPTS]
    prev = use_triton_attention(runner.model)
    try:
        state = PagedBatchState(runner, num_blocks=512, block_size=16,
                                fused=True, alloc_backend="cpp")
        reqs = [Request(i, p, SamplingParams(max_tokens=N, temperature=0.0,
                                             ignore_eos=True))
                for i, p in enumerate(PROMPTS)]
        state.add(reqs)
        while state.any_active:
            finished = state.step()
            if finished:
                state.evict(finished)
    finally:
        restore_attention(runner.model, prev)
    got = [r.output_tokens for r in reqs]
    for p, a, b in zip(PROMPTS, want, got):
        assert a == b, f"cpp-alloc divergence on {p!r}:\n naive {a}\n cpp {b}"
    assert state.alloc.num_free == 512

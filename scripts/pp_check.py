"""Token-exactness through the pipeline cut, then the serving numbers.

The 7B model spans two GPUs; every claim about serving it rests on the
sharded paged-fused path emitting exactly what stock HF generation emits.
Check that first with greedy prompts, then measure what continuous batching
buys over the naive 14.6 tok/s the probe measured.

    python scripts/pp_check.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from server.kernels.paged_attention_triton import (restore_attention,
                                                   use_triton_attention)
from server.paged_exec import PagedBatchState
from server.request import Request, SamplingParams

MODEL = "Qwen/Qwen2.5-7B"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "The three laws of thermodynamics state",
]
N = 32


def main():
    from server.model import ModelRunner
    m = ModelRunner(MODEL, device="pipeline")
    print(f"pipeline over {torch.cuda.device_count()} GPUs; "
          f"input device {m.device}")
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i} {torch.cuda.memory_allocated(i)/2**30:.2f} GiB weights+")

    # ground truth straight through HF, greedy
    want = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = torch.tensor([m.encode(p)], device=m.device)
            out = m.model.generate(ids, max_new_tokens=N, do_sample=False)
            want.append(out[0][ids.shape[1]:].tolist())

    # the sharded fused paged path
    prev = use_triton_attention(m.model)
    try:
        state = PagedBatchState(m, num_blocks=2048, block_size=16, fused=True)
        reqs = [Request(i, p, SamplingParams(max_tokens=N, temperature=0.0,
                                             ignore_eos=True))
                for i, p in enumerate(PROMPTS)]
        state.add(reqs)
        m.sync()
        t0 = time.perf_counter()
        steps = 0
        while state.any_active:
            fin = state.step()
            steps += 1
            if fin:
                state.evict(fin)
        m.sync()
        dt = time.perf_counter() - t0
    finally:
        restore_attention(m.model, prev)

    ok = True
    for p, a, r in zip(PROMPTS, want, reqs):
        match = a == r.output_tokens
        ok = ok and match
        print(f"  {'exact' if match else 'DIVERGED'}: {p!r}")
        if not match:
            print(f"    hf    {a[:12]}")
            print(f"    paged {r.output_tokens[:12]}")
    toks = sum(len(r.output_tokens) for r in reqs) - len(reqs)  # first via prefill
    print(f"decode: {toks} tokens in {dt:.2f}s = {toks/dt:.1f} tok/s "
          f"(B={len(PROMPTS)}, fused paged, sharded)")
    if not ok:
        raise SystemExit("pipeline path diverged from hf generation")
    print("token-exact through the cut")

    # the per-stage graph path: same prompts, same tokens required
    prev = use_triton_attention(m.model)
    try:
        t0 = time.perf_counter()
        state = PagedBatchState(m, num_blocks=2048, block_size=16, fused=True,
                                graphs=True, graph_buckets=[1, 2, 4, 8, 16])
        m.sync()
        print(f"stage graphs captured in {time.perf_counter()-t0:.1f}s")
        reqs2 = [Request(10 + i, p, SamplingParams(max_tokens=N,
                                                   temperature=0.0,
                                                   ignore_eos=True))
                 for i, p in enumerate(PROMPTS)]
        state.add(reqs2)
        m.sync()
        t0 = time.perf_counter()
        while state.any_active:
            fin = state.step()
            if fin:
                state.evict(fin)
        m.sync()
        dt = time.perf_counter() - t0
    finally:
        restore_attention(m.model, prev)
    ok = True
    for p, a, r in zip(PROMPTS, want, reqs2):
        match = a == r.output_tokens
        ok = ok and match
        print(f"  {'exact' if match else 'DIVERGED'} (graphed): {p!r}")
        if not match:
            print(f"    hf      {a[:12]}")
            print(f"    graphed {r.output_tokens[:12]}")
    toks = sum(len(r.output_tokens) for r in reqs2) - len(reqs2)
    print(f"decode: {toks} tokens in {dt:.2f}s = {toks/dt:.1f} tok/s "
          f"(B={len(PROMPTS)}, stage graphs, sharded)")
    if not ok:
        raise SystemExit("stage-graph path diverged from hf generation")
    print("stage graphs token-exact through the cut")


if __name__ == "__main__":
    main()

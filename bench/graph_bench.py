"""Decode step time: eager fused vs CUDA-graph replay, per batch size.

The graph's whole value is killing launch overhead, so the honest comparison
is the full step (input copies + replay + sampling bookkeeping) at the batch
sizes where launches dominate. Contexts short and long, because the win
should shrink as real GPU work grows.

    python -m bench.graph_bench --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from server.model import ModelRunner
from server.request import Request, SamplingParams


def make_state(runner, B, ctx, graphs):
    from server.paged_exec import PagedBatchState
    ids = runner.encode(
        "The quick brown fox jumps over the lazy dog and keeps running. " * 200
    )[:ctx]
    reqs = [Request(i, "", SamplingParams(max_tokens=512, temperature=0.0,
                                          ignore_eos=True),
                    prompt_ids=list(ids)) for i in range(B)]
    state = PagedBatchState(runner, num_blocks=8192, block_size=16,
                            fused=True, graphs=graphs)
    state.add(reqs)
    return state


@torch.no_grad()
def time_step(runner, state, iters=50):
    for _ in range(5):
        state.step()
    runner.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        state.step()
    runner.sync()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--ctxs", nargs="+", type=int, default=[128, 1024])
    p.add_argument("--out", default="results/graph_bench.json")
    a = p.parse_args()

    from server.kernels.paged_attention_triton import (restore_attention,
                                                       use_triton_attention)
    m = ModelRunner(a.model, device=a.device)
    m.warmup()
    prev = use_triton_attention(m.model)
    rows = []
    try:
        print(f"{'ctx':>5} {'B':>3} {'eager ms':>9} {'graph ms':>9} {'speedup':>8}")
        for ctx in a.ctxs:
            for B in a.batches:
                se = make_state(m, B, ctx, graphs=False)
                ms_e = time_step(m, se)
                del se
                sg = make_state(m, B, ctx, graphs=True)
                ms_g = time_step(m, sg)
                del sg
                torch.cuda.empty_cache()
                rows.append({"ctx": ctx, "batch": B, "eager_ms": ms_e,
                             "graph_ms": ms_g, "speedup": ms_e / ms_g})
                print(f"{ctx:>5} {B:>3} {ms_e:>9.2f} {ms_g:>9.2f} "
                      f"{ms_e / ms_g:>7.2f}x")
    finally:
        restore_attention(m.model, prev)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

"""Phase-2 gate: how much of a decode step is NOT the model forward?

The C++ rewrite of the scheduler hot path only moves end-to-end tail latency if
the Python around the forward is a real fraction of the step. Measure it before
porting anything:

    share = (t_step - t_forward) / t_step

per engine state at B in {1, 8, 16}. t_step times state.step() as the engine
runs it; t_forward times just the model call with identical inputs. Decision
rule (written before measuring): share >= 15% means the end-to-end p99 framing
for the C++ port is viable; below that, the port's story is the allocator
microbenchmarks, determinism, and cancel-storm tails instead.

    python -m bench.overhead_gate --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from server.model import ModelRunner
from server.request import Request, SamplingParams


def make_state(runner, kind, B, ctx_tokens):
    from server.batched import BatchState
    from server.paged_exec import PagedBatchState
    prompt_ids = runner.encode(
        "The quick brown fox jumps over the lazy dog and keeps running. " * 40
    )[:ctx_tokens]
    reqs = [Request(i, "", SamplingParams(max_tokens=10_000, temperature=0.0,
                                          ignore_eos=True),
                    prompt_ids=list(prompt_ids)) for i in range(B)]
    if kind == "paged":
        state = PagedBatchState(runner, num_blocks=8192, block_size=16)
    else:
        state = BatchState(runner)
    state.add(reqs)
    return state


@torch.no_grad()
def time_step(runner, state, iters=30):
    for _ in range(3):
        state.step()
    runner.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        state.step()
    runner.sync()
    return (time.perf_counter() - t0) / iters


@torch.no_grad()
def time_forward_only(runner, state, iters=30):
    """The pure model call with the same shapes the step would use, cache held
    fixed (crop back each iteration is unnecessary: we time a single-token
    forward against a snapshot of the current cache)."""
    from transformers import DynamicCache
    dev = runner.device
    B = state.size
    if hasattr(state, "store"):    # paged: gather once, then time forward alone
        tables = [state.alloc.tables[state.sids[i]] for i in range(B)]
        T = max(state.true_len)
        keys, vals, mask = state.store.gather_batch(tables, state.true_len, T)
        base_k = [k.clone() for k in keys]
        base_v = [v.clone() for v in vals]
        last = torch.tensor(state.last_tok, device=dev).unsqueeze(1)
        pos = torch.tensor(state.true_len, device=dev).unsqueeze(1)
        full_mask = torch.cat([mask, torch.ones(B, 1, device=dev, dtype=torch.long)], 1)

        def one():
            cache = DynamicCache()
            for li in range(state.store.n_layers):
                cache.update(base_k[li], base_v[li], li)
            runner.model(input_ids=last, attention_mask=full_mask,
                         position_ids=pos, past_key_values=cache, use_cache=True,
                         cache_position=torch.tensor([T], device=dev))
    else:                           # contiguous: snapshot the live cache
        T = state.T
        last = state.last_tok.clone()
        pos = state.true_len.unsqueeze(1).clone()
        mask = torch.cat([state.mask,
                          torch.ones(B, 1, device=dev, dtype=state.mask.dtype)], 1)
        base_k = [l.keys.clone() for l in state.cache.layers]
        base_v = [l.values.clone() for l in state.cache.layers]

        def one():
            cache = DynamicCache()
            for li in range(len(base_k)):
                cache.update(base_k[li], base_v[li], li)
            runner.model(input_ids=last, attention_mask=mask,
                         position_ids=pos, past_key_values=cache, use_cache=True,
                         cache_position=torch.tensor([T], device=dev))

    for _ in range(3):
        one()
    runner.sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        one()
    runner.sync()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 16])
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--out", default="results/overhead_gate.json")
    a = p.parse_args()

    m = ModelRunner(a.model, device=a.device)
    m.warmup()
    rows = []
    print(f"{'state':>7} {'B':>3} {'step ms':>9} {'fwd ms':>9} {'overhead':>9}")
    for kind in ("batched", "paged"):
        for B in a.batches:
            state = make_state(m, kind, B, a.ctx)
            ms_step = time_step(m, state) * 1e3
            ms_fwd = time_forward_only(m, state) * 1e3
            share = max(0.0, 1 - ms_fwd / ms_step)
            rows.append({"state": kind, "batch": B, "step_ms": ms_step,
                         "forward_ms": ms_fwd, "overhead_share": share})
            print(f"{kind:>7} {B:>3} {ms_step:>9.2f} {ms_fwd:>9.2f} {share:>8.1%}")

    worst = max(r["overhead_share"] for r in rows)
    verdict = "C++ end-to-end p99 framing VIABLE" if worst >= 0.15 else (
        "overhead below 15%: C++ story = allocator microbench + determinism + "
        "cancel-storm tails")
    print(f"\nmax overhead share: {worst:.1%} -> {verdict}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"rows": rows, "max_overhead_share": worst,
                   "verdict": verdict}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

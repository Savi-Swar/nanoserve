"""Where do the engine's 100 ms steps go, when state.step alone costs 23?

The micro loop steps the graphed sharded state at ~23 ms; the serving
ladder's arithmetic says the engine averages ~100 ms per step, for eager
and graphed engines alike. This instruments a real engine run: every
forward_async and finish_async is timed, the graphed branch counts its
hits, and admissions are timed separately, so the gap gets an owner.

    python scripts/pp_serve_diag.py
"""
from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from server.engine import ENGINES  # noqa: E402
from server.request import Request, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B"
N = 12
MT = 64


def main():
    from server.model import ModelRunner
    m = ModelRunner(MODEL, device="pipeline")

    done = threading.Event()
    left = [N]

    def fin(_r):
        left[0] -= 1
        if left[0] == 0:
            done.set()

    eng = ENGINES["paged_fused_graph"](m, on_finish=fin, max_batch=16,
                                       num_blocks=2048)
    st = eng.state

    fwd_times, fin_times, add_times, sizes = [], [], [], []
    graph_hits = [0]

    g = st._graphed
    if g is not None:
        orig_gstep = g.step

        def counted(*a, **k):
            graph_hits[0] += 1
            return orig_gstep(*a, **k)
        g.step = counted

    orig_fwd, orig_fin, orig_add = st.forward_async, st._finish_step, st.add

    def timed_fwd():
        t0 = time.perf_counter()
        r = orig_fwd()
        fwd_times.append(time.perf_counter() - t0)
        sizes.append(st.size)
        return r

    def timed_fin(logits):
        t0 = time.perf_counter()
        r = orig_fin(logits)
        fin_times.append(time.perf_counter() - t0)
        return r

    def timed_add(reqs):
        t0 = time.perf_counter()
        r = orig_add(reqs)
        add_times.append(time.perf_counter() - t0)
        return r

    st.forward_async, st._finish_step, st.add = timed_fwd, timed_fin, timed_add

    eng.start()
    t0 = time.perf_counter()
    for i in range(N):
        eng.submit(Request(i, "The three laws of thermodynamics state that",
                           SamplingParams(max_tokens=MT, temperature=0.0,
                                          ignore_eos=True)))
    done.wait(600)
    wall = time.perf_counter() - t0
    eng.stop()

    toks = N * MT
    steps = len(fwd_times)
    import statistics as stats

    def rep(name, xs):
        if not xs:
            print(f"{name}: none")
            return
        xs_ms = sorted(x * 1e3 for x in xs)
        print(f"{name}: n={len(xs)} p50={stats.median(xs_ms):.1f}ms "
              f"p90={xs_ms[int(0.9 * len(xs_ms)) - 1]:.1f}ms "
              f"max={xs_ms[-1]:.1f}ms total={sum(xs_ms) / 1e3:.2f}s")

    print(f"wall {wall:.2f}s, {toks} tokens = {toks / wall:.1f} tok/s, "
          f"{steps} decode steps, mean batch "
          f"{sum(sizes) / max(1, len(sizes)):.1f}")
    print(f"graphed-branch hits: {graph_hits[0]} of {steps} steps")
    rep("forward_async", fwd_times)
    rep("finish_step  ", fin_times)
    rep("add (prefill)", add_times)
    accounted = sum(fwd_times) + sum(fin_times) + sum(add_times)
    print(f"accounted {accounted:.2f}s of {wall:.2f}s wall "
          f"({(wall - accounted):.2f}s unexplained: engine loop + queue)")


if __name__ == "__main__":
    main()

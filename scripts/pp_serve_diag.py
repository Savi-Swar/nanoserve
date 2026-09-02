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
N = 32
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

    engine_name = sys.argv[1] if len(sys.argv) > 1 else "paged_fused_graph"
    eng = ENGINES[engine_name](m, on_finish=fin, max_batch=16,
                               num_blocks=2048)
    states = list(getattr(eng, "states", [])) or [eng.state]

    fwd_times, fin_times, add_times, sizes = [], [], [], []
    per_state = {i: {"fwd": [], "fin": []} for i in range(len(states))}
    graph_hits = [0]

    for i, st in enumerate(states):
        g = st._graphed
        if g is not None:
            def counted(*a, __orig=g.step, **k):
                graph_hits[0] += 1
                return __orig(*a, **k)
            g.step = counted

        def timed_fwd(__orig=st.forward_async, __st=st, __i=i):
            t0 = time.perf_counter()
            r = __orig()
            dt = time.perf_counter() - t0
            fwd_times.append(dt)
            per_state[__i]["fwd"].append(dt)
            sizes.append(__st.size)
            return r

        def timed_fin(logits, __orig=st._finish_step, __i=i):
            t0 = time.perf_counter()
            r = __orig(logits)
            dt = time.perf_counter() - t0
            fin_times.append(dt)
            per_state[__i]["fin"].append(dt)
            return r

        def timed_add(reqs, __orig=st.add):
            t0 = time.perf_counter()
            r = __orig(reqs)
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

    print(f"[{engine_name}] wall {wall:.2f}s, {toks} tokens = "
          f"{toks / wall:.1f} tok/s, {steps} decode steps, mean batch "
          f"{sum(sizes) / max(1, len(sizes)):.1f}")
    for i, d in per_state.items():
        if d["fwd"]:
            import statistics as _st
            print(f"  state{i}: fwd p50 "
                  f"{_st.median(d['fwd']) * 1e3:.1f}ms  fin p50 "
                  f"{_st.median(d['fin']) * 1e3:.1f}ms  n={len(d['fwd'])}")
    print(f"graphed-branch hits: {graph_hits[0]} of {steps} steps")
    rep("forward_async", fwd_times)
    rep("finish_step  ", fin_times)
    rep("add (prefill)", add_times)
    accounted = sum(fwd_times) + sum(fin_times) + sum(add_times)
    print(f"accounted {accounted:.2f}s of {wall:.2f}s wall "
          f"({(wall - accounted):.2f}s unexplained: engine loop + queue)")


if __name__ == "__main__":
    main()

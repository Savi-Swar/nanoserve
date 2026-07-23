"""Does speculative decoding survive batching? Measure spec-in-batch (spec_cont)
vs plain continuous batching (paged) at increasing batch size, on generic
(low-acceptance) vs grounded (high-acceptance) prompts.

The cost model (bench/spec_cost.py) predicts speedup crosses 1 at B = a*B*, so
on generic traffic speculation should turn into a net loss almost immediately
(B>=2), while on grounded traffic it should stay a win much longer. This
measures the real crossover to check that prediction.

    python -m bench.spec_batched_study --device cuda --batches 1 2 4 8 16 32

Only meaningful on a GPU (the compute-bound regime past the roofline crossover
is where drafts start costing real time); on CPU it's a smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from server.model import ModelRunner
from server.paged_exec import PagedBatchState
from server.request import Request, SamplingParams
from server.spec_batched import SpecPagedState

GENERIC = [
    "Explain the theory of general relativity in a detailed paragraph.",
    "Describe how photosynthesis works and why it matters for life on Earth.",
    "Write a short essay about the causes of the First World War.",
    "Summarize the plot of a novel about a long sea voyage.",
]
GROUNDED = [
    "Repeat exactly: alpha beta gamma delta alpha beta gamma delta alpha beta gamma",
    "Copy this: red red blue blue green green red red blue blue green green red",
    "x = 1\ny = 2\nz = 3\nx = 1\ny = 2\nz = 3\nx = 1\ny = 2\nz =",
    "one two three four one two three four one two three four one two three",
]


def measure(m, StateCls, prompts, B, steps, spec):
    reqs = [Request(i, prompts[i % len(prompts)],
                    SamplingParams(max_tokens=steps * 10 + 4, temperature=0.0, ignore_eos=True))
            for i in range(B)]
    st = StateCls(m) if not spec else StateCls(m)
    st.add(reqs)                      # prefill (not timed)
    base = sum(r.num_output for r in reqs)
    m.sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        if not st.any_active:
            break
        st.step()
    m.sync()
    wall = time.perf_counter() - t0
    committed = sum(r.num_output for r in reqs) - base
    tpf = st.stats()["tokens_per_forward"] if spec else 1.0
    return committed / wall if wall else 0.0, tpf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--out", default="results/spec_batched.json")
    a = p.parse_args()

    m = ModelRunner(a.model, device=a.device)
    print(f"loaded on {m.device}; warming up...")
    m.warmup()

    out = {"device": m.device, "workloads": {}}
    for name, prompts in [("generic", GENERIC), ("grounded", GROUNDED)]:
        print(f"\n=== {name} workload ===")
        print(f"{'B':>4}{'cont tok/s':>13}{'spec tok/s':>13}{'ratio':>9}{'tok/fwd':>9}")
        rows = []
        crossover = None
        for B in a.batches:
            cont, _ = measure(m, PagedBatchState, prompts, B, a.steps, spec=False)
            spec, tpf = measure(m, SpecPagedState, prompts, B, a.steps, spec=True)
            ratio = spec / cont if cont else 0.0
            rows.append({"batch": B, "cont_tps": cont, "spec_tps": spec,
                         "ratio": ratio, "tokens_per_forward": tpf})
            if crossover is None and ratio < 1.0 and B > 1:
                crossover = B
            print(f"{B:>4}{cont:>13.1f}{spec:>13.1f}{ratio:>9.2f}{tpf:>9.2f}")
        out["workloads"][name] = {"rows": rows, "measured_crossover_batch": crossover}
        if crossover:
            print(f"-> speculation becomes a net LOSS at batch >= {crossover}")
        else:
            print("-> speculation stayed a win across the tested batch sizes")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

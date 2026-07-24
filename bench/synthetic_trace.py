"""A second, deliberately different workload: a long-context regime.

The Azure conversational trace (`bench/trace.py`) is a real heavy-tailed
workload, but heavy-tailed around a short context (p50 ~1020 tokens) with
meaningful generated lengths. That trace is decode-dominated: most wall-clock
time is in the autoregressive decode loop, so it stresses the scheduler's
steady-state batching, TPOT, and decode throughput.

Auditing against a single trace risks "one trace = one workload": whatever it
exercises is all you measure. Real deployments also serve a different shape,
RAG / doc-QA / summarization, where each request drags a large retrieved context
(thousands of tokens) and emits a short answer. That regime is prefill-dominated
and KV-pressure-heavy, and stresses the parts the Azure trace barely touches:

  * Prefill cost. A 2000-token prompt is ~2000 tokens of attention/FFN work
    before the first output token, so TTFT is gated by prefill, not queueing.
    This is where chunked prefill earns its keep: break a giant prompt into
    pieces so it interleaves with ongoing decodes instead of stalling the batch
    head-of-line.
  * KV-cache footprint. KV memory scales with context length, so long contexts
    fill the cache far faster than short conversational turns. This is where a
    paged KV cache matters most: without paging, a few long prompts fragment and
    waste the block pool; with it, the same pool admits many more concurrent long
    requests. High KV pressure also hits admission-control / preemption paths the
    short Azure trace rarely reaches.
  * Output-bound-ness inverts. Short outputs (mean ~64) mean a brief decode loop,
    so the ratio of prefill to decode work is high, the mirror image of the
    conversational trace.

Contexts are drawn heavy-tailed (lognormal) around `mean_ctx`, output lengths
heavy-tailed around a small `mean_out`, arrivals Poisson at `rate`. Prompt
content is irrelevant to serving performance (only lengths and arrival timing
matter), so, like `trace.py`, each prompt is synthesized as
`[FILLER_TOKEN] * ctx_len`.

The public surface mirrors `bench/trace.py` so this drops into the existing
`replay()` / engine harness:

    reqs, offsets = build_longcontext_requests(n=256, mean_ctx=2000, mean_out=64)
    # ... same (list[Request], offsets) shape as build_trace_requests(...)

Run standalone to inspect the generated distribution:

    python -m bench.synthetic_trace --n 256 --mean-ctx 2000 --mean-out 64
"""
from __future__ import annotations

import argparse
import math
import random

from server.request import Request, SamplingParams

FILLER_TOKEN = 1000  # any benign in-vocab id; content is irrelevant to timing


def _lognormal_lengths(rng: random.Random, n: int, mean: float, spread: float,
                       lo: int = 1) -> list[int]:
    """`n` positive integer lengths, heavy-tailed (lognormal) with the given
    arithmetic `mean`. `spread` is the sigma of the underlying log-space normal:
    larger sigma => heavier tail. Solve for mu so the lognormal's mean equals the
    requested `mean` (E[X] = exp(mu + sigma^2/2)).
    """
    sigma = max(1e-6, spread)
    mu = math.log(max(1e-9, mean)) - 0.5 * sigma * sigma
    return [max(lo, round(rng.lognormvariate(mu, sigma))) for _ in range(n)]


def build_longcontext_requests(
    n: int,
    mean_ctx: int = 2000,
    ctx_spread: float = 0.8,
    mean_out: int = 64,
    out_spread: float = 0.6,
    rate: float = 8.0,
    seed: int = 0,
) -> tuple[list[Request], list[float]]:
    """Return (requests, arrival_offsets_seconds) for a synthetic long-context
    (RAG / doc-QA) workload.

    Parameters
    ----------
    n           number of requests to generate.
    mean_ctx    arithmetic mean prompt/context length in tokens (heavy-tailed
                around it via lognormal). Default 2000, far above the Azure
                conversational p50 (~1020).
    ctx_spread  sigma of the log-space normal for contexts; larger => heavier
                tail (a few very long documents).
    mean_out    arithmetic mean generated (output) length; short by design.
    out_spread  sigma of the log-space normal for outputs.
    rate        Poisson arrival rate (requests/sec); exponential inter-arrival
                gaps, exactly like the uniform synthetic and the trace's timing.
    seed        RNG seed for reproducibility.

    The return shape matches `bench.trace.build_trace_requests`: offsets are
    seconds relative to the first arrival (offsets[0] == 0.0), so it feeds the
    same `replay()` / engine harness with no changes.
    """
    rng = random.Random(seed)
    ctx_lens = _lognormal_lengths(rng, n, mean_ctx, ctx_spread, lo=1)
    out_lens = _lognormal_lengths(rng, n, mean_out, out_spread, lo=1)

    reqs, offsets = [], []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(rate)  # exponential gap => Poisson arrivals
        reqs.append(Request(
            id=i, prompt="",
            sampling=SamplingParams(
                max_tokens=out_lens[i], temperature=0.0, ignore_eos=True),
            prompt_ids=[FILLER_TOKEN] * ctx_lens[i],
        ))
        offsets.append(t)
    base = offsets[0] if offsets else 0.0
    return reqs, [o - base for o in offsets]


def effective_rate(offsets: list[float]) -> float:
    """Realized arrival rate (requests/sec); same helper as `trace.py`."""
    span = offsets[-1] - offsets[0]
    return len(offsets) / span if span > 0 else float(len(offsets))


def _percentiles(values: list[int], ps=(50, 90, 99)) -> dict[int, float]:
    if not values:
        return {p: 0.0 for p in ps}
    s = sorted(values)
    out = {}
    for p in ps:
        # nearest-rank on a 0..1 fractional index
        idx = min(len(s) - 1, max(0, round((p / 100.0) * (len(s) - 1))))
        out[p] = float(s[idx])
    return out


def describe(requests: list[Request]) -> dict:
    """Summarize the generated length distribution.

    Returns context/output length percentiles (p50/p90/p99), means, and max: the
    same numbers the CLI prints, exposed programmatically so the audit harness
    can assert on the regime it's testing.
    """
    ctx = [len(r.prompt_ids) if r.prompt_ids is not None else 0 for r in requests]
    out = [r.sampling.max_tokens for r in requests]
    n = len(requests)

    def _stats(vals):
        pct = _percentiles(vals)
        return {
            "p50": pct[50], "p90": pct[90], "p99": pct[99],
            "mean": (sum(vals) / len(vals)) if vals else 0.0,
            "max": max(vals) if vals else 0,
        }

    return {"n": n, "context": _stats(ctx), "output": _stats(out)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate and inspect a synthetic long-context "
                    "(RAG / doc-QA) workload: a prefill-dominated, "
                    "KV-pressure-heavy regime distinct from the decode-dominated "
                    "Azure conversational trace.")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--mean-ctx", type=int, default=2000)
    ap.add_argument("--ctx-spread", type=float, default=0.8)
    ap.add_argument("--mean-out", type=int, default=64)
    ap.add_argument("--out-spread", type=float, default=0.6)
    ap.add_argument("--rate", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reqs, offsets = build_longcontext_requests(
        n=args.n, mean_ctx=args.mean_ctx, ctx_spread=args.ctx_spread,
        mean_out=args.mean_out, out_spread=args.out_spread,
        rate=args.rate, seed=args.seed,
    )
    d = describe(reqs)
    c, o = d["context"], d["output"]

    print("long-context synthetic workload (RAG / doc-QA regime)")
    print(f"  requests           : {d['n']}")
    print(f"  target arrival rate : {args.rate:.2f} req/s "
          f"(realized {effective_rate(offsets):.2f} req/s)")
    print(f"  span               : {offsets[-1]:.2f} s")
    print("  context tokens (prompt), heavy-tailed, prefill-dominated:")
    print(f"    p50={c['p50']:.0f}  p90={c['p90']:.0f}  p99={c['p99']:.0f}  "
          f"mean={c['mean']:.0f}  max={c['max']:.0f}")
    print("  output tokens (generated), short by design:")
    print(f"    p50={o['p50']:.0f}  p90={o['p90']:.0f}  p99={o['p99']:.0f}  "
          f"mean={o['mean']:.0f}  max={o['max']:.0f}")


if __name__ == "__main__":
    main()

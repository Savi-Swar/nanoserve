"""Inter-token latency (ITL) tails per engine, measured the defensible way.

Open-loop Poisson load (no coordinated omission: arrivals don't wait for the
server), per-token timestamps via the engine's on_token hook, ITL = the gap
between consecutive tokens of one request. Pooled across requests and runs;
p99.9 is only quoted when the pool holds >= 100k samples (below that the tail
estimate is the k-th worst sample wearing a suit). Multiple runs give a CI: a
comparison counts only if the p99 ranges don't overlap.

    python -m bench.latency_study --engines paged paged_fused --device cuda
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time

import numpy as np

from server.engine import ENGINES
from server.model import ModelRunner

from .workload import build_requests, replay


def one_run(model, engine_name, rate, n, max_tokens, seed, cfg):
    reqs, offsets = build_requests(n=n, rate=rate, max_tokens=max_tokens, seed=seed)
    stamps: dict[int, list[float]] = {r.id: [] for r in reqs}
    done = threading.Event()
    left = [len(reqs)]

    def on_finish(_r):
        left[0] -= 1
        if left[0] == 0:
            done.set()

    def on_token(req, _tok):
        stamps[req.id].append(time.perf_counter())

    eng = ENGINES[engine_name](model, on_finish=on_finish, on_token=on_token, **cfg)
    eng.start()
    replay(eng, reqs, offsets)
    done.wait()
    eng.stop()

    itls = []
    for ts in stamps.values():
        if len(ts) > 1:
            d = np.diff(np.asarray(ts))
            itls.append(d)
    out = np.concatenate(itls) * 1e3   # ms
    del eng
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return out


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--engines", nargs="+", default=["paged", "paged_fused"])
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=8192)
    p.add_argument("--out", default="results/latency.json")
    a = p.parse_args()

    model = ModelRunner(a.model, device=a.device)
    model.warmup()
    print(f"open-loop rate={a.rate}/s n={a.n} max_tokens={a.max_tokens} "
          f"runs={a.runs} ({model.device})")

    out = {"config": vars(a).copy(), "engines": {}}
    out["config"].pop("out", None)
    print(f"{'engine':>17} {'samples':>8} {'p50':>7} {'p90':>7} {'p99':>8} "
          f"{'p99.9':>8} {'max':>8}   (ITL ms, pooled)")
    for name in a.engines:
        cfg = {"max_batch": a.max_batch}
        if name in ("paged", "paged_fused", "paged_fused_cpp", "paged_fused_graph", "paged_fused_pp2", "paged_fused_pp2t", "paged_fused_pp2g"):
            cfg["num_blocks"] = a.num_blocks
        pools, p99s = [], []
        for r in range(a.runs):
            itl = one_run(model, name, a.rate, a.n, a.max_tokens, seed=r, cfg=cfg)
            pools.append(itl)
            p99s.append(pct(itl, 99))
        pool = np.concatenate(pools)
        row = {
            "samples": int(pool.size),
            "p50": pct(pool, 50), "p90": pct(pool, 90), "p99": pct(pool, 99),
            "p999": pct(pool, 99.9) if pool.size >= 100_000 else None,
            "max": float(pool.max()),
            "p99_per_run": p99s,
            "p99_range": [min(p99s), max(p99s)],
        }
        out["engines"][name] = row
        p999s = f"{row['p999']:8.2f}" if row["p999"] is not None else "   (n<100k)"
        print(f"{name:>17} {row['samples']:>8} {row['p50']:>7.2f} "
              f"{row['p90']:>7.2f} {row['p99']:>8.2f} {p999s} "
              f"{row['max']:>8.1f}")
        print(f"{'':>17} p99 per-run range: "
              f"[{min(p99s):.2f}, {max(p99s):.2f}] over {a.runs} runs")

    # non-overlap verdicts for adjacent engine pairs
    names = list(out["engines"])
    for x, y in zip(names, names[1:]):
        rx = out["engines"][x]["p99_range"]
        ry = out["engines"][y]["p99_range"]
        sep = rx[1] < ry[0] or ry[1] < rx[0]
        out["engines"][y][f"p99_vs_{x}"] = "distinguishable" if sep else "within noise"
        print(f"p99 {x} vs {y}: "
              + ("DISTINGUISHABLE (ranges don't overlap)" if sep else "within noise"))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

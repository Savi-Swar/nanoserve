"""Goodput, the metric production serving actually optimizes for.

Peak throughput is vanity: a server can post big tok/s while most requests miss
their latency target. Goodput counts only requests that meet BOTH an SLO on
TTFT (time to first token) and on TPOT (per-output-token latency), and reports
requests/sec of those. Sweeping offered load, goodput rises with rate, then
saturates and falls once the server can't keep the tail under SLO — the peak is
the sustainable capacity.

    python -m bench.goodput_study --engines naive static continuous paged \
        --rates 2 4 8 16 32 --ttft-slo 1000 --tpot-slo 100

On a GPU, tighten the SLOs (e.g. --ttft-slo 500 --tpot-slo 50).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from types import SimpleNamespace

from server.engine import ENGINES
from server.model import ModelRunner

from .workload import build_requests, replay


def _cfg(engine, a):
    if engine == "static":
        return dict(batch_size=a.batch_size, max_wait=a.max_wait)
    if engine == "continuous":
        return dict(max_batch=a.max_batch)
    if engine == "paged":
        return dict(max_batch=a.max_batch, num_blocks=a.num_blocks)
    return {}


def run_one(model, engine, rate, a, ttft_s, tpot_s):
    reqs, offsets = build_requests(n=a.n, rate=rate, max_tokens=a.max_tokens, seed=a.seed)
    done = threading.Event()
    finished = []

    def on_finish(r):
        finished.append(r)
        if len(finished) == len(reqs):
            done.set()

    eng = ENGINES[engine](model, on_finish=on_finish, **_cfg(engine, a))
    eng.start()
    t0 = time.perf_counter()
    replay(eng, reqs, offsets)
    done.wait()
    wall = time.perf_counter() - t0
    eng.stop()

    good = sum(1 for r in reqs if r.meets_slo(ttft_s, tpot_s))
    out_tokens = sum(r.num_output for r in reqs)
    return {"rate": rate, "goodput_qps": good / wall, "good_frac": good / len(reqs),
            "throughput": out_tokens / wall}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--engines", nargs="+", default=["naive", "static", "continuous", "paged"])
    p.add_argument("--rates", nargs="+", type=float, default=[2, 4, 8, 16])
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ttft-slo", type=float, default=1000.0, help="TTFT SLO (ms)")
    p.add_argument("--tpot-slo", type=float, default=100.0, help="per-output-token SLO (ms)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-wait", type=float, default=0.05)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=4096)
    p.add_argument("--out", default="results/goodput.json")
    a = p.parse_args()

    ttft_s, tpot_s = a.ttft_slo / 1e3, a.tpot_slo / 1e3
    model = ModelRunner(a.model, device=a.device)
    print(f"loaded on {model.device}; SLO: TTFT<={a.ttft_slo:.0f}ms, TPOT<={a.tpot_slo:.0f}ms; warming up...")
    model.warmup()

    out = {"ttft_slo_ms": a.ttft_slo, "tpot_slo_ms": a.tpot_slo, "runs": {}}
    print(f"\n{'engine':<12}" + "".join(f"{'r=' + str(r):>10}" for r in a.rates) + f"{'MAX':>10}")
    for engine in a.engines:
        row = [run_one(model, engine, r, a, ttft_s, tpot_s) for r in a.rates]
        out["runs"][engine] = row
        peak = max(row, key=lambda x: x["goodput_qps"])
        cells = "".join(f"{x['goodput_qps']:>10.1f}" for x in row)
        print(f"{engine:<12}{cells}{peak['goodput_qps']:>10.1f}")
    print("-" * (12 + 10 * (len(a.rates) + 1)))
    print("cells = goodput (req/s meeting both SLOs). MAX = sustainable capacity under SLO.")
    caps = {e: max(x["goodput_qps"] for x in out["runs"][e]) for e in a.engines}
    if caps.get("naive"):
        best = max(caps, key=caps.get)
        print(f"{best} sustains {caps[best]/caps['naive']:.1f}x the goodput of naive under this SLO.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

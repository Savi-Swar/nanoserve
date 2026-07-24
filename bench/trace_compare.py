"""Run every engine on the real Azure trace at natural load and under a burst,
print both tables, write results/trace.json.

    python -m bench.trace_compare --device cpu --n 32 --len-scale 16

On a GPU: --device cuda --len-scale 1 (full lengths) and a larger --n.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from types import SimpleNamespace

from server.model import ModelRunner

from .run_bench import run

TRACE = "data/azure_llm_conv.csv"


def one(model, engine, n, len_scale, trace_scale, max_batch, num_blocks):
    args = SimpleNamespace(
        model=model.model_name, engine=engine, rate=0.0, n=n, max_tokens=64,
        temperature=0.0, seed=0, device=model.device, batch_size=8, max_wait=0.05,
        max_batch=max_batch, num_blocks=num_blocks, out=None,
        trace=TRACE, trace_start=0, len_scale=len_scale, trace_scale=trace_scale,
    )
    return run(args, model=model)


def table(title, rows):
    print(f"\n{title}")
    print(f"{'engine':<12}{'throughput':>12}{'TTFT p99':>12}")
    for r in rows:
        print(f"{r['engine']:<12}{r['throughput']:>9.1f} t/s{r['ttft']['p99']*1e3:>9.0f}ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--len-scale", type=float, default=16.0)
    p.add_argument("--engines", nargs="+", default=["naive", "static", "continuous", "paged"])
    p.add_argument("--out", default="results/trace.json")
    a = p.parse_args()

    if not os.path.exists(TRACE):
        raise SystemExit(f"missing {TRACE}; see data/README.md to download the Azure trace")

    model = ModelRunner(a.model, device=a.device)
    print(f"loaded {a.model} on {model.device}; warming up...")
    model.warmup()

    out = {"natural": [], "burst_10x": []}
    for scale_name, tscale in (("natural", 1.0), ("burst_10x", 0.1)):
        for eng in a.engines:
            rep = one(model, eng, a.n, a.len_scale, tscale, max_batch=16, num_blocks=8192)
            out[scale_name].append(rep.to_dict(a.model, rep.throughput, 64))
    table("NATURAL arrival rate:", out["natural"])
    table("BURST (10x compressed arrivals):", out["burst_10x"])

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

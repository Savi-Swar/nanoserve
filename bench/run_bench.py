"""Wire load generator -> engine -> metrics and print one report.

    python -m bench.run_bench --engine naive --rate 4 --n 40 --max-tokens 64

Every optimization in later weeks is validated by re-running this exact command
with a different --engine and comparing reports on identical workloads.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time

from server.engine import ENGINES
from server.model import ModelRunner

from .gpu import GpuSampler
from .metrics import build_report
from .workload import build_requests, replay


def run(args, model: ModelRunner | None = None):
    """Run one benchmark. Pass an existing `model` to reuse it across a sweep.
    Returns the Report."""
    if model is None:
        model = ModelRunner(args.model, device=args.device)
        print(f"loaded {args.model} on {model.device} ({model.dtype})")
        print("warming up...")
        model.warmup()

    if getattr(args, "trace", None):
        from .trace import build_trace_requests, effective_rate
        reqs, offsets = build_trace_requests(
            args.trace, n=args.n, start=getattr(args, "trace_start", 0),
            len_scale=getattr(args, "len_scale", 1.0),
            time_scale=getattr(args, "trace_scale", 1.0),
        )
        args.rate = effective_rate(offsets)
        print(f"replaying {args.n} real requests from {args.trace} "
              f"(~{args.rate:.1f} req/s effective, len/{getattr(args,'len_scale',1.0)})")
    else:
        reqs, offsets = build_requests(
            n=args.n,
            rate=args.rate,
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
        )

    done = threading.Event()
    finished = []

    def on_finish(r):
        finished.append(r)
        if len(finished) == len(reqs):
            done.set()

    cfg = {}
    if args.engine == "static":
        cfg = dict(batch_size=args.batch_size, max_wait=args.max_wait)
    elif args.engine == "continuous":
        cfg = dict(max_batch=args.max_batch)
    elif args.engine == "paged":
        cfg = dict(max_batch=args.max_batch, num_blocks=getattr(args, "num_blocks", 4096))
    engine = ENGINES[args.engine](model, on_finish=on_finish, **cfg)
    engine.start()

    print(f"replaying {args.n} requests at {args.rate} req/s (Poisson)...")
    with GpuSampler() as gpu:
        t0 = time.perf_counter()
        replay(engine, reqs, offsets)
        done.wait()
        wall = time.perf_counter() - t0
    engine.stop()

    report = build_report(args.engine, model.device, reqs, wall, gpu.summary())
    print("\n" + "=" * 56)
    print(report.render())
    print("=" * 56)

    if getattr(args, "out", None):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report.to_dict(args.model, args.rate, args.max_tokens), f, indent=2)
        print(f"wrote {args.out}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--engine", default="naive", choices=list(ENGINES))
    p.add_argument("--rate", type=float, default=4.0, help="arrivals/sec (Poisson)")
    p.add_argument("--n", type=int, default=40, help="number of requests")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset)")
    p.add_argument("--batch-size", type=int, default=8, help="static: max batch")
    p.add_argument("--max-wait", type=float, default=0.05, help="static: batch fill wait (s)")
    p.add_argument("--max-batch", type=int, default=16, help="continuous/paged: max concurrent seqs")
    p.add_argument("--num-blocks", type=int, default=4096, help="paged: KV block pool size")
    p.add_argument("--out", default=None, help="write results JSON to this path")
    p.add_argument("--trace", default=None, help="replay a real trace CSV instead of synthetic")
    p.add_argument("--trace-start", type=int, default=0, help="trace: first row")
    p.add_argument("--len-scale", type=float, default=1.0, help="trace: divide token lengths (CPU)")
    p.add_argument("--trace-scale", type=float, default=1.0, help="trace: scale inter-arrival gaps")
    run(p.parse_args())


if __name__ == "__main__":
    main()

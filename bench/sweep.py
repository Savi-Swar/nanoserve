"""Run the full engine x arrival-rate grid on ONE loaded model and write
results/sweep.json for bench.plot. This is the command that produces the
headline comparison.

    python -m bench.sweep --engines naive static continuous \
        --rates 2 4 8 16 --n 64 --max-tokens 64

On CPU keep --n and --rates small; on a CUDA GPU push them up.
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

from server.model import ModelRunner

from .run_bench import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--engines", nargs="+", default=["naive", "static", "continuous"])
    p.add_argument("--rates", nargs="+", type=float, default=[2, 4, 8])
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-wait", type=float, default=0.05)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--out", default="results/sweep.json")
    a = p.parse_args()

    model = ModelRunner(a.model, device=a.device)
    print(f"loaded {a.model} on {model.device} ({model.dtype}); warming up...")
    model.warmup()

    runs = []
    for engine in a.engines:
        for rate in a.rates:
            print(f"\n>>> engine={engine} rate={rate}")
            args = SimpleNamespace(
                model=a.model, engine=engine, rate=rate, n=a.n,
                max_tokens=a.max_tokens, temperature=a.temperature, seed=a.seed,
                device=a.device, batch_size=a.batch_size, max_wait=a.max_wait,
                max_batch=a.max_batch, out=None,
            )
            report = run(args, model=model)
            runs.append(report.to_dict(a.model, rate, a.max_tokens))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"runs": runs}, f, indent=2)
    print(f"\nwrote {len(runs)} runs -> {a.out}")
    print(f"now plot:  python -m bench.plot --results {a.out} --out results/")


if __name__ == "__main__":
    main()

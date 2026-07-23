"""Statistical hygiene: run a config N times and report mean +/- 95% CI and the
noise floor, so an "improvement" smaller than the noise floor is correctly
called nothing. This is the instrument that makes Month-3 audit claims
defensible — we watched continuous batching swing 14->23 tok/s run-to-run on
CPU, so no single number means anything without this.

    # one config, N runs, with error bars + noise floor
    python -m bench.repeat --engine continuous --runs 5 --rate 8 --n 24

    # compare two engines and rule on whether the gap beats the noise
    python -m bench.repeat --compare continuous paged --runs 5 --rate 8 --n 24
"""
from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

from server.model import ModelRunner

from .run_bench import run

# two-sided 95% t critical values by degrees of freedom (n-1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def _tcrit(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    for k in keys:
        if k >= df:
            return _T95[k]
    return 1.96  # large-df normal approx


class Stat:
    def __init__(self, xs: list[float]):
        self.xs = xs
        self.n = len(xs)
        self.mean = sum(xs) / self.n
        var = sum((x - self.mean) ** 2 for x in xs) / (self.n - 1) if self.n > 1 else 0.0
        self.std = math.sqrt(var)
        self.ci = _tcrit(self.n - 1) * self.std / math.sqrt(self.n) if self.n > 1 else 0.0

    @property
    def cv(self) -> float:  # coefficient of variation
        return self.std / self.mean if self.mean else 0.0

    @property
    def ci_pct(self) -> float:
        return self.ci / self.mean if self.mean else 0.0

    def fmt(self, unit=""):
        return f"{self.mean:.1f} +/- {self.ci:.1f}{unit} (+/-{self.ci_pct*100:.1f}%, n={self.n})"


def _args(base, engine):
    d = dict(model=base.model, engine=engine, rate=base.rate, n=base.n,
             max_tokens=base.max_tokens, temperature=0.0, seed=base.seed,
             device=base.device, batch_size=8, max_wait=0.05, max_batch=16,
             num_blocks=4096, out=None, trace=base.trace, trace_start=0,
             len_scale=base.len_scale, trace_scale=base.trace_scale)
    return SimpleNamespace(**d)


def measure(model, base, engine, runs):
    tput, ttft = [], []
    for i in range(runs):
        a = _args(base, engine)
        a.seed = base.seed + i  # vary the workload draw across runs
        rep = run(a, model=model)
        tput.append(rep.throughput)
        ttft.append(rep.ttft["p99"] * 1e3)
    return Stat(tput), Stat(ttft)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--engine", default="continuous")
    p.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--trace", default=None)
    p.add_argument("--len-scale", type=float, default=1.0)
    p.add_argument("--trace-scale", type=float, default=1.0)
    a = p.parse_args()

    model = ModelRunner(a.model, device=a.device)
    print(f"loaded on {model.device}; warming up...")
    model.warmup()

    engines = a.compare if a.compare else [a.engine]
    results = {}
    for eng in engines:
        print(f"\n>>> {eng}: {a.runs} runs")
        results[eng] = measure(model, a, eng, a.runs)

    print("\n" + "=" * 60)
    for eng in engines:
        tp, tt = results[eng]
        print(f"{eng:<12} throughput {tp.fmt(' tok/s')}")
        print(f"{'':<12} TTFT p99   {tt.fmt(' ms')}")
    floor = max(results[e][0].ci_pct for e in engines)
    print("-" * 60)
    print(f"noise floor (max throughput 95% CI): +/-{floor*100:.1f}%")

    if a.compare:
        A, B = a.compare
        ta, tb = results[A][0], results[B][0]
        diff = tb.mean - ta.mean
        diff_pct = diff / ta.mean * 100
        # distinguishable only if the 95% CIs do not overlap
        overlap = abs(diff) <= (ta.ci + tb.ci)
        verdict = "WITHIN NOISE (not distinguishable)" if overlap else "DISTINGUISHABLE"
        print(f"{B} vs {A}: {diff_pct:+.1f}% throughput -> {verdict}")
        if overlap:
            print(f"  |{diff:.1f}| <= CI_A+CI_B ({ta.ci:.1f}+{tb.ci:.1f}) — do not claim a winner")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""The matched-workload percent-of-vLLM, with error bars on both sides.

The single-run 81% needs the same discipline every other headline got: five
runs of each system on the identical workload, a 95% CI each, and a ratio
quoted as an interval. Reads results/repeat.json (nanoserve side, from
bench.repeat) and results/vllm_m*.json (five vllm_ref runs).

    python -m bench.match_compare
"""
from __future__ import annotations

import glob
import json
import math


def ci95(xs):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mean, half


def main():
    with open("results/repeat.json") as f:
        rep = json.load(f)
    name, d = next(iter(rep["engines"].items()))
    xs = d["throughputs"]
    om, oh = ci95(xs)

    vs = []
    for p in sorted(glob.glob("results/vllm_m*.json")):
        with open(p) as f:
            vs.append(json.load(f)["throughput"])
    vm, vh = ci95(vs)

    lo = (om - oh) / (vm + vh)
    hi = (om + oh) / (vm - vh)
    print(f"{name}: {om:.1f} +/- {oh:.1f} tok/s (n={len(xs)})")
    print(f"vllm:   {vm:.1f} +/- {vh:.1f} tok/s (n={len(vs)})")
    print(f"matched-workload ratio: {om/vm*100:.0f}% "
          f"(interval [{lo*100:.0f}%, {hi*100:.0f}%])")
    out = {"nanoserve": {"engine": name, "mean": om, "ci": oh, "runs": xs},
           "vllm": {"mean": vm, "ci": vh, "runs": vs},
           "ratio": om / vm, "ratio_interval": [lo, hi]}
    with open("results/match.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/match.json")


if __name__ == "__main__":
    main()

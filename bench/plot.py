"""Turn a benchmark sweep JSON into publication-quality PNG charts.

    python -m bench.plot --results results/sweep.json --out results/

Reads the JSON produced by sweeping ``run_bench`` across engines/rates and
emits a small set of comparison figures (matplotlib only, no seaborn).

Input schema (see repo README / run_bench)::

    {"runs": [
        {"engine": "naive", "device": "cuda", "model": "...",
         "rate": 4.0, "n": 40, "max_tokens": 64,
         "wall": 12.3, "out_tokens": 2200,
         "throughput": 178.9, "per_req_decode_tps": 42.1,
         "ttft": {"p50": .., "p90": .., "p99": ..},
         "e2e":  {"p50": .., "p90": .., "p99": ..},
         "queue":{"p50": .., "p90": .., "p99": ..},
         "gpu":  {"mean": 71.0, "peak": 92.0}},   # or null
        ...
    ]}
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

# Canonical engine ordering + a colorblind-friendly (Okabe-Ito) palette so a
# given engine keeps the same color across every figure.
ENGINE_ORDER = ["naive", "static", "continuous", "paged"]
ENGINE_COLORS = {
    "naive": "#0072B2",       # blue
    "static": "#E69F00",      # orange
    "continuous": "#009E73",  # green
    "paged": "#CC79AC",       # purple/pink
}
_FALLBACK_COLORS = ["#56B4E9", "#D55E00", "#F0E442", "#999999", "#000000"]

FIGSIZE = (7, 4.5)
DPI = 150


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(path):
    with open(path) as f:
        data = json.load(f)
    runs = data.get("runs", []) if isinstance(data, dict) else data
    return [r for r in runs if isinstance(r, dict)]


def _engines_in(runs):
    """Distinct engines present, in canonical order (unknowns appended)."""
    present = {r.get("engine") for r in runs if r.get("engine") is not None}
    ordered = [e for e in ENGINE_ORDER if e in present]
    ordered += sorted(e for e in present if e not in ENGINE_ORDER)
    return ordered


def _color_for(engine, idx=0):
    if engine in ENGINE_COLORS:
        return ENGINE_COLORS[engine]
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def _by_engine(runs):
    d = defaultdict(list)
    for r in runs:
        if r.get("engine") is not None:
            d[r["engine"]].append(r)
    return d


def _get(run, *path):
    """Nested lookup that tolerates missing/None, e.g. _get(r,'ttft','p99')."""
    cur = run
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _finish(fig, ax, out_dir, name):
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_throughput_by_engine(runs, out_dir):
    """Bar of peak throughput per engine, annotated with x-over-naive speedup."""
    engines = _engines_in(runs)
    by_eng = _by_engine(runs)

    best = {}
    for e in engines:
        vals = [r for r in by_eng[e] if _get(r, "throughput") is not None]
        if vals:
            best[e] = max(vals, key=lambda r: r["throughput"])
    engines = [e for e in engines if e in best]
    if not engines:
        print("skip throughput_by_engine: no throughput data")
        return None

    tput = [best[e]["throughput"] for e in engines]
    naive = best.get("naive", {}).get("throughput")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(engines, tput,
                  color=[_color_for(e, i) for i, e in enumerate(engines)])
    top = max(tput)
    for bar, e, v in zip(bars, engines, tput):
        label = f"{v:.0f}"
        if naive and naive > 0:
            label += f"\n{v / naive:.2f}x"
        ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.01, label,
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xlabel("Engine")
    ax.set_title("Peak throughput by engine (higher is better)")
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis="y", ls=":", alpha=0.4)
    return _finish(fig, ax, out_dir, "throughput_by_engine.png")


def _representative_rate(by_eng, engines):
    """Highest rate common to all engines; else None (caller falls back)."""
    rate_sets = []
    for e in engines:
        rs = {r["rate"] for r in by_eng[e] if _get(r, "rate") is not None}
        if rs:
            rate_sets.append(rs)
    if len(rate_sets) != len(engines) or not rate_sets:
        return None
    common = set.intersection(*rate_sets)
    return max(common) if common else None


def fig_ttft_p99_by_engine(runs, out_dir):
    """Bar of TTFT p99 (ms) at a representative rate (lower is better)."""
    engines = _engines_in(runs)
    by_eng = _by_engine(runs)
    common_rate = _representative_rate(by_eng, engines)

    labels, vals, used_common = [], [], common_rate is not None
    for i, e in enumerate(engines):
        run = None
        if used_common:
            for r in by_eng[e]:
                if r.get("rate") == common_rate:
                    run = r
                    break
        else:
            cand = [r for r in by_eng[e] if _get(r, "rate") is not None]
            if cand:
                run = max(cand, key=lambda r: r["rate"])
        p99 = _get(run, "ttft", "p99") if run else None
        if p99 is not None:
            labels.append(e)
            vals.append(p99 * 1000.0)  # s -> ms

    if not labels:
        print("skip ttft_p99_by_engine: no TTFT p99 data")
        return None

    if used_common:
        sub = f"at common rate = {common_rate:g} req/s"
    else:
        sub = "at each engine's max rate (no common rate)"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, vals,
                  color=[_color_for(e, i) for i, e in enumerate(labels)])
    top = max(vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.01,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("TTFT p99 (ms)")
    ax.set_xlabel("Engine")
    ax.set_title(f"Tail time-to-first-token by engine (lower is better)\n{sub}")
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis="y", ls=":", alpha=0.4)
    return _finish(fig, ax, out_dir, "ttft_p99_by_engine.png")


def fig_throughput_vs_rate(runs, out_dir):
    """Line: x=arrival rate, y=throughput, one series per engine."""
    engines = _engines_in(runs)
    by_eng = _by_engine(runs)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = False
    for i, e in enumerate(engines):
        pts = [(r["rate"], r["throughput"]) for r in by_eng[e]
               if _get(r, "rate") is not None and _get(r, "throughput") is not None]
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=e, color=_color_for(e, i))
        plotted = True

    if not plotted:
        print("skip throughput_vs_rate: no rate/throughput data")
        plt.close(fig)
        return None

    ax.set_xlabel("Arrival rate (req/s)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs. offered load")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(title="Engine")
    return _finish(fig, ax, out_dir, "throughput_vs_rate.png")


def fig_latency_throughput(runs, out_dir):
    """Latency-throughput tradeoff: x=throughput, y=TTFT p99 (ms)."""
    engines = _engines_in(runs)
    by_eng = _by_engine(runs)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = False
    for i, e in enumerate(engines):
        pts = []
        for r in by_eng[e]:
            tp = _get(r, "throughput")
            p99 = _get(r, "ttft", "p99")
            rate = _get(r, "rate")
            if tp is not None and p99 is not None:
                pts.append((tp, p99 * 1000.0, rate))
        if not pts:
            continue
        pts.sort()  # by throughput, so the connecting line reads left->right
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=e, color=_color_for(e, i))
        plotted = True

    if not plotted:
        print("skip latency_throughput: no throughput/TTFT data")
        plt.close(fig)
        return None

    ax.set_xlabel("Throughput (tokens/s)")
    ax.set_ylabel("TTFT p99 (ms)")
    ax.set_title("Latency-throughput tradeoff (up-and-to-the-left is worse)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(title="Engine")
    return _finish(fig, ax, out_dir, "latency_throughput.png")


def fig_gpu_util_by_engine(runs, out_dir):
    """Bar of mean GPU util % per engine; skipped if no gpu data anywhere."""
    engines = _engines_in(runs)
    by_eng = _by_engine(runs)

    if not any(_get(r, "gpu", "mean") is not None for r in runs):
        print("skip gpu_util_by_engine: no non-null gpu data in any run")
        return None

    labels, vals = [], []
    for e in engines:
        # pair GPU util with the peak-throughput run for a like-for-like view
        cand = [r for r in by_eng[e]
                if _get(r, "gpu", "mean") is not None
                and _get(r, "throughput") is not None]
        if cand:
            run = max(cand, key=lambda r: r["throughput"])
        else:
            cand = [r for r in by_eng[e] if _get(r, "gpu", "mean") is not None]
            run = cand[0] if cand else None
        if run is not None:
            labels.append(e)
            vals.append(_get(run, "gpu", "mean"))

    if not labels:
        print("skip gpu_util_by_engine: no per-engine gpu means")
        return None

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, vals,
                  color=[_color_for(e, i) for i, e in enumerate(labels)])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean GPU utilization (%)")
    ax.set_xlabel("Engine")
    ax.set_title("GPU utilization by engine (at peak throughput)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", ls=":", alpha=0.4)
    return _finish(fig, ax, out_dir, "gpu_util_by_engine.png")


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def make_all(results_path, out_dir):
    runs = _load(results_path)
    if not runs:
        print(f"no runs found in {results_path}; nothing to plot")
        return []
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for fn in (
        fig_throughput_by_engine,
        fig_ttft_p99_by_engine,
        fig_throughput_vs_rate,
        fig_latency_throughput,
        fig_gpu_util_by_engine,
    ):
        try:
            path = fn(runs, out_dir)
        except Exception as exc:  # never let one bad figure kill the rest
            print(f"skip {fn.__name__}: {exc}")
            path = None
        if path:
            written.append(path)
    return written


def main():
    p = argparse.ArgumentParser(description="Plot benchmark sweep results.")
    p.add_argument("--results", default="results/sweep.json",
                   help="path to the sweep results JSON")
    p.add_argument("--out", default="results/",
                   help="output directory for PNG figures")
    args = p.parse_args()
    written = make_all(args.results, args.out)
    print(f"\n{len(written)} figure(s) written to {args.out}")


if __name__ == "__main__":
    main()

"""Cancellation storms: what aborts do to the requests that stay.

Nobody's benchmark injects client disconnects open-loop and measures the
survivors' inter-token tails; servers keep shipping abort-path bugs (leaked
blocks, wedged schedulers) precisely because that path is never under load
in CI. This harness drives the real engine with seeded storms and reports
the survivors' ITL percentiles by phase, then asserts the accounting:

  baseline    same load, no cancels (the reference tails)
  disconnect  a fraction of arrivals are chaff, each cancelled after
              reading k tokens, k ~ U(1, 32) (client-gone-mid-stream)
  burst       all chaff cancelled at once mid-run (thundering abort)
  swizzle     chaff cancelled one by one in random order over the run

Invariants at quiesce, every scenario: every block back in the pool, every
request in a terminal status, every survivor got its full token budget.
Storms are generated from (seed, scenario), so any anomaly is a one-command
repro.

    python -m bench.storm_study --device cuda --engine paged_fused
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import threading
import time

import numpy as np

from server.engine import ENGINES
from server.model import ModelRunner

from .workload import build_requests, replay


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


_SCENARIO_SALT = {"baseline": 0, "disconnect": 1, "burst": 2, "swizzle": 3}


def run_storm(model, engine_name, scenario, seed, a):
    # python's str hash is randomized per process; a stable salt keeps
    # (seed, scenario) a reproducible storm across runs and machines
    rng = random.Random(seed * 1000 + _SCENARIO_SALT.get(scenario, 9))
    reqs, offsets = build_requests(n=a.n, rate=a.rate, max_tokens=a.max_tokens,
                                   seed=seed)
    if scenario == "baseline":
        chaff = set()
    else:
        k = int(len(reqs) * a.cancel_frac)
        chaff = set(r.id for r in rng.sample(reqs, k))
    cancel_after = {rid: rng.randint(1, 32) for rid in chaff}

    stamps = {r.id: [] for r in reqs if r.id not in chaff}
    counts = dict.fromkeys(chaff, 0)
    done = threading.Event()
    left = [len(reqs)]
    lock = threading.Lock()
    eng_box = {}
    storm_win = [None, None]  # [first cancel t, last cancel t]

    def mark_cancel():
        t = time.perf_counter()
        if storm_win[0] is None:
            storm_win[0] = t
        storm_win[1] = t

    def on_finish(_r):
        with lock:
            left[0] -= 1
            if left[0] == 0:
                done.set()

    def on_token(req, _tok):
        if req.id in chaff:
            if scenario == "disconnect":
                counts[req.id] += 1
                if counts[req.id] == cancel_after[req.id]:
                    mark_cancel()
                    eng_box["eng"].cancel(req.id)
        else:
            stamps[req.id].append(time.perf_counter())

    cfg = {"max_batch": a.max_batch}
    if engine_name.startswith("paged"):
        cfg["num_blocks"] = a.num_blocks
    eng = ENGINES[engine_name](model, on_finish=on_finish, on_token=on_token,
                               **cfg)
    eng_box["eng"] = eng
    eng.start()

    storm_thread = None
    if scenario in ("burst", "swizzle"):
        order = sorted(chaff)
        rng.shuffle(order)
        total_s = len(reqs) / a.rate

        def storm():
            if scenario == "burst":
                time.sleep(total_s * 0.5)
                mark_cancel()
                for rid in order:
                    eng.cancel(rid)
                mark_cancel()
            else:
                gap = (total_s * 0.7) / max(1, len(order))
                time.sleep(total_s * 0.15)
                for rid in order:
                    mark_cancel()
                    eng.cancel(rid)
                    time.sleep(gap)

        storm_thread = threading.Thread(target=storm, daemon=True)
        storm_thread.start()

    replay(eng, reqs, offsets)
    done.wait()
    if storm_thread:
        storm_thread.join()
    eng.stop()

    # --- invariants (the point of the exercise) ------------------------
    violations = []
    free = getattr(getattr(eng.state, "alloc", None), "num_free", None)
    expected = a.num_blocks - getattr(eng.state, "reserved_blocks", 0)
    if free is not None and free != expected:
        violations.append(f"blocks leaked: {expected - free}")
    for r in reqs:
        if r.status not in ("done", "cancelled", "rejected", "timeout"):
            violations.append(f"req {r.id} non-terminal: {r.status}")
        if r.id not in chaff and r.status == "done" \
                and len(r.output_tokens) != r.sampling.max_tokens:
            violations.append(f"survivor {r.id} short: {len(r.output_tokens)}")

    # --- survivor ITLs by phase ----------------------------------------
    gaps, ts = [], []
    for arr in stamps.values():
        if len(arr) > 1:
            v = np.asarray(arr)
            gaps.append(np.diff(v) * 1e3)
            ts.append(v[1:])
    gaps = np.concatenate(gaps) if gaps else np.array([])
    ts = np.concatenate(ts) if ts else np.array([])

    phases = {"all": gaps}
    if storm_win[0] is not None:
        lo, hi = storm_win
        phases["before"] = gaps[ts < lo]
        phases["during"] = gaps[(ts >= lo) & (ts <= hi + 0.5)]
        phases["after"] = gaps[ts > hi + 0.5]

    row = {"scenario": scenario, "seed": seed, "chaff": len(chaff),
           "survivors": len(stamps), "violations": violations,
           "phases": {k: {"n": int(v.size), "p50": pct(v, 50),
                          "p99": pct(v, 99), "max": float(v.max()) if v.size else None}
                      for k, v in phases.items()}}
    # each run allocates a ~1.5 GB KV pool; the engine/state/closure cycle
    # keeps dead pools alive until a gc pass, and ten of them OOM a T4
    eng_box.clear()
    del eng
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--engine", default="paged_fused")
    p.add_argument("--scenarios", nargs="+",
                   default=["baseline", "disconnect", "burst", "swizzle"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--cancel-frac", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=8192)
    p.add_argument("--out", default="results/storm.json")
    a = p.parse_args()

    model = ModelRunner(a.model, device=a.device)
    model.warmup()
    print(f"engine={a.engine} rate={a.rate}/s n={a.n} "
          f"cancel_frac={a.cancel_frac} seeds={a.seeds} ({model.device})")

    rows, bad = [], 0
    for scenario in a.scenarios:
        for seed in a.seeds:
            r = run_storm(model, a.engine, scenario, seed, a)
            rows.append(r)
            ph = r["phases"]
            parts = [f"{k} p99 {v['p99']:.1f}ms (n={v['n']})"
                     for k, v in ph.items() if k != "all" and v["n"]]
            detail = "  ".join(parts) if parts else \
                f"p99 {ph['all']['p99']:.1f}ms (n={ph['all']['n']})"
            flag = ""
            if r["violations"]:
                bad += 1
                flag = "  INVARIANT VIOLATED: " + "; ".join(r["violations"])
            print(f"{scenario:>11} seed {seed}: {r['chaff']} cancelled, "
                  f"{r['survivors']} survived  {detail}{flag}")

    print(f"\n{bad} runs with invariant violations out of {len(rows)}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"config": {k: v for k, v in vars(a).items() if k != "out"},
                   "rows": rows}, f, indent=2)
    print(f"wrote {a.out}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

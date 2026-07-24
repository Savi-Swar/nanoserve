"""Performance-regression gate for CI.

CI runners have no GPU and pulling a ~1 GB model per PR is wasteful, so the gate
doesn't run the real engine. It runs a fast, deterministic proxy built from the
two things that need no model:

  1. Fragmentation ablation metrics (`bench.memory_study`): paged KV
     fragmentation fraction and paged capacity under a fixed budget. Pure
     functions of a seeded simulated workload + the real BlockAllocator, so
     they're bit-for-bit reproducible across machines: a correctness signal for
     the paged allocator's packing behaviour.

  2. A micro-benchmark of the pure-Python BlockAllocator hot path
     (add_seq / append_token / free_seq): allocator throughput in ops/sec. A
     timing signal, so it's noisy on shared runners and gets a wider band.

The gate compares the current proxy against a committed baseline and fails if a
metric regresses beyond its threshold:

  * fragmentation increased  > 3 %  relative   (tight, deterministic)
  * paged capacity dropped   > 3 %             (tight, deterministic)
  * alloc ops/sec dropped    > 15 %            (loose, timing is noisy)

Usage:
    python -m bench.regression --update-baseline   # write results/baseline.json
    python -m bench.regression --check             # gate against the baseline

On the deterministic metrics: `bench.memory_study.study()` sizes its pool at
num_blocks=1e9, allocating a billion-entry free list (tens of GB, tens of
seconds): fine on a big dev box, an OOM risk on a 16 GB CI runner. The
fragmentation numbers don't depend on the pool being that large, only on it
never running dry, so we recompute the same metric with a right-sized allocator.
Identical to memory_study's output (verified against results/memory.json) while
staying instant and memory-cheap.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time

from bench.memory_study import capacity_under_budget, simulate_lengths
from server.paged_cache import BlockAllocator

BASELINE_PATH = "results/baseline.json"
MIB = 1024 * 1024

# --- proxy workload knobs (fixed so the proxy is reproducible) --------------
# These mirror bench.memory_study's defaults so the fragmentation/capacity
# numbers line up with results/memory.json.
N_SEQS = 64
BLOCK_SIZE = 16
MAX_LEN = 512
MEAN_OUT = 128
BYTES_PER_TOKEN = 12288      # Qwen2.5-0.5B fp16 KV
BUDGET_MIB = 256
SEED = 0

# --- micro-benchmark knobs --------------------------------------------------
MICRO_OPS = 200_000          # target allocator ops per timed repeat
MICRO_REPEATS = 5            # median over repeats to damp timing noise

# --- regression thresholds --------------------------------------------------
FRAG_TOL = 0.03              # frag may rise at most 3% relative (deterministic)
CAP_TOL = 0.03              # capacity may drop at most 3% (deterministic)
OPS_TOL = 0.15              # ops/sec may drop at most 15% (noisy timing)


# ---------------------------------------------------------------------------
# Deterministic correctness metrics (identical to bench.memory_study output)
# ---------------------------------------------------------------------------
def paged_fragmentation() -> float:
    """Paged KV fragmentation fraction on the seeded length-skewed batch.

    Recomputes memory_study.study()['strategies']['paged']['frag'] with a
    right-sized pool: bit-identical, without the 1e9-block allocation."""
    lens = simulate_lengths(N_SEQS, SEED, MEAN_OUT, MAX_LEN)
    stored = sum(lens)
    num_blocks = sum(max(1, math.ceil(l / BLOCK_SIZE)) for l in lens)
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    for i, l in enumerate(lens):
        alloc.add_seq(i, l)
    paged_slots = alloc.num_used * BLOCK_SIZE
    return (paged_slots - stored) / paged_slots if paged_slots else 0.0


def paged_capacity() -> int:
    """Sequences the paged layout fits in a fixed KV budget (deterministic)."""
    cap = capacity_under_budget(
        BUDGET_MIB * MIB, BLOCK_SIZE, MAX_LEN, BYTES_PER_TOKEN, MEAN_OUT, SEED
    )
    return int(cap["paged"])


# ---------------------------------------------------------------------------
# Timing metric: pure-Python allocator throughput
# ---------------------------------------------------------------------------
def _alloc_workload(n_ops: int, seed: int) -> int:
    """Drive the BlockAllocator hot path for ~n_ops operations, deterministically.

    Mixes add_seq / append_token / free_seq the way the scheduler does: admit a
    sequence, grow it a few tokens, and retire the oldest once the live set is
    full (so the pool churns and the free list stays exercised). Returns the
    exact op count performed."""
    rng = random.Random(seed)
    # Right-sized, churning pool: a few thousand blocks is plenty since we free.
    alloc = BlockAllocator(num_blocks=8192, block_size=BLOCK_SIZE)
    live: list[int] = []
    seq_id = 0
    ops = 0
    while ops < n_ops:
        prompt = rng.randint(8, 48)
        alloc.add_seq(seq_id, prompt)
        ops += 1
        for _ in range(rng.randint(1, 20)):
            alloc.append_token(seq_id)
            ops += 1
        live.append(seq_id)
        seq_id += 1
        if len(live) > 32:                 # retire oldest -> free-list churn
            alloc.free_seq(live.pop(0))
            ops += 1
    for s in live:                         # drain
        alloc.free_seq(s)
        ops += 1
    return ops


def alloc_ops_per_sec() -> float:
    """Median allocator throughput (ops/sec) over MICRO_REPEATS timed runs."""
    rates = []
    for r in range(MICRO_REPEATS):
        t0 = time.perf_counter()
        ops = _alloc_workload(MICRO_OPS, seed=SEED + r)
        dt = time.perf_counter() - t0
        rates.append(ops / dt if dt > 0 else 0.0)
    return statistics.median(rates)


# ---------------------------------------------------------------------------
# Proxy + gate
# ---------------------------------------------------------------------------
def run_proxy() -> dict:
    """Run the full deterministic proxy and return the key metrics."""
    return {
        "paged_frag": paged_fragmentation(),
        "paged_capacity": paged_capacity(),
        "alloc_ops_per_sec": alloc_ops_per_sec(),
        # provenance so a stale baseline is easy to spot
        "config": {
            "n_seqs": N_SEQS,
            "block_size": BLOCK_SIZE,
            "max_len": MAX_LEN,
            "mean_out": MEAN_OUT,
            "bytes_per_token": BYTES_PER_TOKEN,
            "budget_mib": BUDGET_MIB,
            "seed": SEED,
            "micro_ops": MICRO_OPS,
            "micro_repeats": MICRO_REPEATS,
        },
    }


def update_baseline(path: str = BASELINE_PATH) -> dict:
    metrics = run_proxy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote baseline -> {path}")
    print(f"  paged_frag         {metrics['paged_frag']:.6f}")
    print(f"  paged_capacity     {metrics['paged_capacity']}")
    print(f"  alloc_ops_per_sec  {metrics['alloc_ops_per_sec']:,.0f}")
    return metrics


def check(path: str = BASELINE_PATH) -> int:
    if not os.path.exists(path):
        print(f"FAIL: no baseline at {path}; run --update-baseline first.")
        return 1

    with open(path) as f:
        base = json.load(f)
    cur = run_proxy()

    # (label, baseline, current, direction, tolerance)
    #   direction "up"   -> a regression is the metric going UP  (fragmentation)
    #   direction "down" -> a regression is the metric going DOWN (capacity, ops)
    rows = [
        ("frag (paged)", base["paged_frag"], cur["paged_frag"], "up", FRAG_TOL),
        ("capacity (paged)", base["paged_capacity"], cur["paged_capacity"], "down", CAP_TOL),
        ("alloc ops/sec", base["alloc_ops_per_sec"], cur["alloc_ops_per_sec"], "down", OPS_TOL),
    ]

    print("=" * 74)
    print(f"{'metric':<18}{'baseline':>14}{'current':>14}{'Δ%':>9}{'limit':>9}  verdict")
    print("-" * 74)
    failed = False
    for label, b, c, direction, tol in rows:
        b = float(b)
        c = float(c)
        rel = (c - b) / b if b else 0.0            # signed relative change
        if direction == "up":
            regressed = rel > tol                   # went up too much
            limit = f"+{tol*100:.0f}%"
        else:
            regressed = rel < -tol                  # dropped too much
            limit = f"-{tol*100:.0f}%"
        verdict = "FAIL" if regressed else "ok"
        failed = failed or regressed
        print(f"{label:<18}{b:>14.4g}{c:>14.4g}{rel*100:>+8.1f}%{limit:>9}  {verdict}")
    print("=" * 74)

    if failed:
        print("PERF REGRESSION: a metric moved past its threshold. See table above.")
        return 1
    print("PASS: no regression beyond thresholds.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--update-baseline", action="store_true",
                   help="run the proxy and (over)write results/baseline.json")
    g.add_argument("--check", action="store_true",
                   help="run the proxy and gate against the committed baseline")
    p.add_argument("--baseline", default=BASELINE_PATH, help="baseline JSON path")
    a = p.parse_args()

    if a.update_baseline:
        update_baseline(a.baseline)
        return 0
    return check(a.baseline)


if __name__ == "__main__":
    raise SystemExit(main())

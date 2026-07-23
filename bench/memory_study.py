"""Fragmentation ablation: how much KV memory each cache layout wastes on a
realistic, length-skewed batch — and how many more sequences paging lets you
hold under a fixed budget (which is what turns into throughput).

Runs standalone (no GPU/model needed): sequence lengths are simulated and the
paged column uses the *real* BlockAllocator, not a formula. Defaults to
Qwen2.5-0.5B fp16 KV size (12,288 B/token); pass --bytes-per-token to change.

    python -m bench.memory_study --n 64 --block-size 16 --mean-out 128
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random

from server.paged_cache import BlockAllocator

KIB = 1024
MIB = 1024 * 1024


def simulate_lengths(n: int, seed: int, mean_out: int, max_len: int) -> list[int]:
    """Current length of each concurrent sequence: a short prompt plus a
    partially-generated, exponentially-distributed output (heavy length skew —
    the regime where layout actually matters)."""
    rng = random.Random(seed)
    lens = []
    for _ in range(n):
        prompt = rng.randint(8, 48)
        out = int(rng.expovariate(1.0 / mean_out))
        lens.append(min(prompt + out, max_len))
    return lens


def study(lens: list[int], block_size: int, max_len: int, bpt: int) -> dict:
    n = len(lens)
    stored = sum(lens)  # tokens that genuinely need KV

    reserve_slots = n * max_len          # reserve-to-max (pre-paging systems)
    padded_slots = n * max(lens)         # pad to the batch's longest (static/cont.)

    # size the pool to exactly what this batch needs (never runs dry); a huge
    # num_blocks would materialize a huge free list for no reason.
    needed = sum(math.ceil(l / block_size) for l in lens)
    alloc = BlockAllocator(num_blocks=needed, block_size=block_size)
    for i, l in enumerate(lens):
        alloc.add_seq(i, l)
    paged_slots = alloc.num_used * block_size
    paged_frag = alloc.internal_frag_tokens()

    def frag(slots):
        return (slots - stored) / slots if slots else 0.0

    return {
        "n": n,
        "block_size": block_size,
        "max_len": max_len,
        "bytes_per_token": bpt,
        "stored_tokens": stored,
        "strategies": {
            "reserve_max": {"slots": reserve_slots, "bytes": reserve_slots * bpt, "frag": frag(reserve_slots)},
            "padded_batch": {"slots": padded_slots, "bytes": padded_slots * bpt, "frag": frag(padded_slots)},
            "paged": {"slots": paged_slots, "bytes": paged_slots * bpt, "frag": frag(paged_slots),
                      "internal_frag_tokens": paged_frag},
        },
    }


def capacity_under_budget(budget_bytes: int, block_size: int, max_len: int,
                          bpt: int, mean_out: int, seed: int) -> dict:
    """How many sequences fit in a fixed KV budget under each layout."""
    reserve_cap = budget_bytes // (max_len * bpt)
    padded_cap = reserve_cap  # padded still can't exceed reserve in the worst case

    num_blocks = budget_bytes // (block_size * bpt)
    alloc = BlockAllocator(num_blocks=int(num_blocks), block_size=block_size)
    rng = random.Random(seed)
    admitted = 0
    while True:
        prompt = rng.randint(8, 48)
        out = int(rng.expovariate(1.0 / mean_out))
        need = min(prompt + out, max_len)
        if not alloc.can_admit(need):
            break
        alloc.add_seq(admitted, need)
        admitted += 1
    return {"reserve_max": int(reserve_cap), "padded_batch": int(padded_cap), "paged": admitted}


def fmt_bytes(b: int) -> str:
    if b >= MIB:
        return f"{b / MIB:.1f} MiB"
    return f"{b / KIB:.1f} KiB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64, help="concurrent sequences")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=512, help="max context (reserve target)")
    p.add_argument("--mean-out", type=int, default=128, help="mean output length")
    p.add_argument("--bytes-per-token", type=int, default=12288, help="KV B/token (Qwen2.5-0.5B fp16)")
    p.add_argument("--budget-mib", type=int, default=256, help="fixed KV budget for capacity test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/memory.json")
    a = p.parse_args()

    lens = simulate_lengths(a.n, a.seed, a.mean_out, a.max_len)
    rep = study(lens, a.block_size, a.max_len, a.bytes_per_token)
    cap = capacity_under_budget(a.budget_mib * MIB, a.block_size, a.max_len,
                                a.bytes_per_token, a.mean_out, a.seed)
    rep["capacity_under_budget"] = {"budget_mib": a.budget_mib, **cap}

    s = rep["strategies"]
    paged_b = s["paged"]["bytes"]
    print("=" * 60)
    print(f"KV memory for {a.n} length-skewed sequences (block_size={a.block_size})")
    print(f"tokens actually stored: {rep['stored_tokens']}")
    print("-" * 60)
    print(f"{'layout':<14}{'KV bytes':>12}{'waste':>9}{'vs paged':>10}")
    for name in ("reserve_max", "padded_batch", "paged"):
        st = s[name]
        print(f"{name:<14}{fmt_bytes(st['bytes']):>12}{st['frag']*100:>8.0f}%"
              f"{st['bytes']/paged_b:>9.1f}x")
    print("-" * 60)
    print(f"sequences fittable in {a.budget_mib} MiB:")
    print(f"  reserve-to-max {cap['reserve_max']:>5}   paged {cap['paged']:>5}"
          f"   ({cap['paged']/max(cap['reserve_max'],1):.1f}x more)")
    print("=" * 60)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

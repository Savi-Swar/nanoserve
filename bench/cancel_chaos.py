"""Cancellation chaos harness — prove the KV block pool never leaks when
clients hang up mid-generation.

Two layers:

1. `alloc_stress` hammers the BlockAllocator directly: spin up random
   sequences, grow them, abort a random subset partway, free the rest, and
   after every cycle assert the free list is whole again. A leaked block (one
   grabbed on admission and never returned on abort) shows up immediately as
   `num_free < num_blocks`, and a partition check catches a block that is ever
   in two places or none. Runs thousands of cycles with no model, so it's the
   rigorous proof.

2. `engine_chaos` runs the real PagedContinuousEngine: submit a batch, cancel a
   random subset mid-stream via `engine.cancel()`, wait for everything to
   settle, and assert the pool is fully reclaimed and no sequence is left
   running. This proves the cancel -> reap -> evict -> free_seq wiring end to
   end, not just the allocator.

    python -m bench.cancel_chaos                        # alloc stress only (fast)
    python -m bench.cancel_chaos --engine --engine-cycles 30   # + real engine

Exit code is non-zero if any block leaks.
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
import time

from server.paged_cache import BlockAllocator, OutOfBlocks


def _assert_partition(alloc: BlockAllocator):
    """Every block is either free or in exactly one sequence's table — always.
    A leak (block in neither) or a double-alloc (block in two) fails here."""
    used = [b for t in alloc.tables.values() for b in t]
    assert len(used) == len(set(used)), "a block is allocated to two sequences"
    everything = set(used) | set(alloc.free)
    assert everything == set(range(alloc.num_blocks)), "a block went missing"
    assert len(used) + len(alloc.free) == alloc.num_blocks, "block count drifted"


def alloc_stress(num_blocks=256, block_size=16, cycles=2000, seed=0):
    rng = random.Random(seed)
    alloc = BlockAllocator(num_blocks, block_size)
    max_used = 0
    aborted = 0
    for c in range(cycles):
        live: dict[int, bool] = {}
        sid_base = c * 1000
        # admit a random set of sequences, each reserving prompt+gen span
        for k in range(rng.randint(1, 14)):
            need = rng.randint(1, 10) * block_size
            if not alloc.can_admit(need):
                continue
            alloc.add_seq(sid_base + k, need)
            live[sid_base + k] = True
        # grow them; abort a random subset partway (the "client hung up" event)
        for _ in range(rng.randint(0, 25)):
            for s in list(live):
                if rng.random() < 0.18:
                    alloc.free_seq(s)
                    live.pop(s)
                    aborted += 1
                    continue
                try:
                    alloc.append_token(s)
                except OutOfBlocks:
                    alloc.free_seq(s)
                    live.pop(s)
            max_used = max(max_used, alloc.num_used)
            _assert_partition(alloc)
        # everyone else finishes normally
        for s in list(live):
            alloc.free_seq(s)
        _assert_partition(alloc)
        assert alloc.num_free == num_blocks, (
            f"leak after cycle {c}: {num_blocks - alloc.num_free} block(s) lost")
    return {"cycles": cycles, "num_blocks": num_blocks, "aborts": aborted,
            "max_used": max_used, "leaked": num_blocks - alloc.num_free}


def engine_chaos(cycles=30, per_cycle=8, num_blocks=512, block_size=16,
                 max_tokens=12, kill_frac=0.6, seed=0, model_name="Qwen/Qwen2.5-0.5B",
                 device=None):
    from server.engine import PagedContinuousEngine
    from server.model import ModelRunner
    from server.request import Request, SamplingParams

    rng = random.Random(seed)
    print(f"loading {model_name} ...")
    model = ModelRunner(model_name, device=device)

    done = {"n": 0}
    cv = threading.Condition()

    def on_finish(_r):
        with cv:
            done["n"] += 1
            cv.notify_all()

    eng = PagedContinuousEngine(model, on_finish=on_finish, max_batch=per_cycle,
                                num_blocks=num_blocks, block_size=block_size)
    eng.start()

    prompts = ["The quick brown fox jumps over", "In a distant galaxy there lived",
               "Once upon a time in a small town", "The meaning of life is probably"]
    rid = 0
    killed = 0
    leaked_cycles = 0
    try:
        for c in range(cycles):
            with cv:
                base = done["n"]
            ids = []
            for _ in range(per_cycle):
                eng.submit(Request(rid, rng.choice(prompts),
                                   SamplingParams(max_tokens=max_tokens, temperature=0.0,
                                                  ignore_eos=True)))
                ids.append(rid)
                rid += 1
            time.sleep(0.05)  # let them start generating, then pull the plug
            kill = rng.sample(ids, int(len(ids) * kill_frac))
            for k in kill:
                eng.cancel(k)
            killed += len(kill)
            with cv:
                ok = cv.wait_for(lambda: done["n"] - base >= per_cycle, timeout=120)
            free, size = eng.state.alloc.num_free, eng.state.size
            leaked = (not ok) or free != num_blocks or size != 0
            if leaked:
                leaked_cycles += 1
                print(f"  [cycle {c}] LEAK free={free}/{num_blocks} size={size} settled={ok}")
            elif c == 0 or (c + 1) % 10 == 0:
                print(f"  [cycle {c + 1}/{cycles}] killed {len(kill)}/{per_cycle} "
                      f"mid-stream, pool reclaimed ({free}/{num_blocks} free)")
    finally:
        eng.stop()
    return {"cycles": cycles, "killed_midstream": killed,
            "leaked_cycles": leaked_cycles, "num_blocks": num_blocks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=2000, help="allocator-stress cycles")
    p.add_argument("--num-blocks", type=int, default=256)
    p.add_argument("--engine", action="store_true", help="also run the real-engine chaos")
    p.add_argument("--engine-cycles", type=int, default=30)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    print("=" * 60)
    print(f"allocator chaos: {a.cycles} cycles, {a.num_blocks}-block pool")
    r1 = alloc_stress(num_blocks=a.num_blocks, cycles=a.cycles)
    print(f"  {r1['aborts']} sequences aborted mid-stream, peak {r1['max_used']} "
          f"blocks used")
    print(f"  leaked blocks: {r1['leaked']}  ->  {'OK' if r1['leaked'] == 0 else 'LEAK'}")

    r2 = None
    if a.engine:
        print("=" * 60)
        print(f"engine chaos: {a.engine_cycles} cycles on the real paged engine")
        r2 = engine_chaos(cycles=a.engine_cycles, device=a.device)
        print(f"  killed {r2['killed_midstream']} sequences mid-stream across "
              f"{r2['cycles']} cycles")
        print(f"  cycles with a leak: {r2['leaked_cycles']}  ->  "
              f"{'OK' if r2['leaked_cycles'] == 0 else 'LEAK'}")

    print("=" * 60)
    leaked = r1["leaked"] != 0 or (r2 is not None and r2["leaked_cycles"] != 0)
    print("RESULT:", "block leak detected" if leaked else "zero block leakage")
    sys.exit(1 if leaked else 0)


if __name__ == "__main__":
    main()

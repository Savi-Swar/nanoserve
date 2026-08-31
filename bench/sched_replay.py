"""Deterministic scheduler replay: prove Python and C++ make identical
decisions, then compare their bookkeeping cost.

Scheduling decisions here don't depend on token values (finish = token budget,
eos handled as a mask), so a trace of (arrival_step, prompt_len, out_len) fully
determines every decision: what admits when, which slot each token lands in,
what finishes and frees. The harness drives both implementations through the
same trace, logs every decision, and hashes the log:

  - same implementation twice -> identical hash        (determinism, N6)
  - python vs c++             -> identical hash        (drop-in equivalence)
  - steps/sec per impl        -> the bookkeeping cost  (one crossing per step)

No model, no wall clock, no threads: virtual time is the step counter.

    python -m bench.sched_replay --n 500 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

MAX_BATCH = 16
NUM_BLOCKS = 2048
BLOCK_SIZE = 16


def build_trace(n, seed):
    """(arrival_step, prompt_len, out_len) per request; arrivals bunch so
    admission pressure and mid-decode admits both happen."""
    rng = random.Random(seed)
    trace, step = [], 0
    for _ in range(n):
        step += rng.choice([0, 0, 1, 1, 2, 5])
        trace.append((step, rng.randint(4, 200), rng.randint(1, 96)))
    return trace


class PyScheduler:
    """The Python bookkeeping, same contract as nano::Scheduler."""

    def __init__(self, num_blocks, block_size):
        from server.paged_cache import BlockAllocator
        self.alloc = BlockAllocator(num_blocks, block_size)
        self.bs = block_size
        self.rows = []          # [rid, sid, len, emitted, max_new]
        self._sid = 0

    @property
    def size(self):
        return len(self.rows)

    @property
    def num_free_blocks(self):
        return self.alloc.num_free

    def can_admit(self, prompt_len, max_new):
        return self.alloc.can_admit(prompt_len + max_new)

    def admit(self, rid, prompt_len, max_new):
        sid = self._sid
        self._sid += 1
        self.alloc.add_seq(sid, prompt_len + max_new)
        self.rows.append([rid, sid, prompt_len, 1, max_new])
        return sid

    def table(self, sid):
        return self.alloc.tables[sid]

    def plan_step(self):
        slots, lens = [], []
        for _, sid, ln, _, _ in self.rows:
            t = self.alloc.tables[sid]
            slots.append(t[ln // self.bs] * self.bs + ln % self.bs)
            lens.append(ln)
        return slots, lens

    def commit_step(self, eos_hit):
        finished = []
        for i, row in enumerate(self.rows):
            row[2] += 1
            row[3] += 1
            if row[3] >= row[4] or (eos_hit and eos_hit[i]):
                finished.append(i)
        for i in finished:
            self.alloc.free_seq(self.rows[i][1])
        drop = set(finished)
        self.rows = [r for i, r in enumerate(self.rows) if i not in drop]
        return finished


def run_trace(sched, trace):
    """Drive one scheduler through the trace. Returns (decision_log_hash,
    steps, wall_seconds). The log records admissions (with block tables),
    per-step plans, and finishes: any divergence in any decision changes it."""
    h = hashlib.sha256()
    waiting = list(range(len(trace)))
    step = 0
    t0 = time.perf_counter()
    steps = 0
    while waiting or sched.size > 0:
        # admit everything that has arrived, in order, while it fits
        while waiting:
            idx = waiting[0]
            arr, plen, olen = trace[idx]
            if arr > step:
                break
            if sched.size >= MAX_BATCH or not sched.can_admit(plen, olen):
                # force progress when idle, exactly like the engine
                if sched.size == 0 and sched.can_admit(plen, olen):
                    pass
                else:
                    break
            sid = sched.admit(idx, plen, olen)
            h.update(b"A%d:%d:%s" % (idx, sid, bytes(str(list(sched.table(sid))), "ascii")))
            waiting.pop(0)
        if sched.size > 0:
            slots, lens = sched.plan_step()
            h.update(b"P" + bytes(str(list(slots)), "ascii"))
            fin = sched.commit_step([])
            if fin:
                h.update(b"F" + bytes(str(list(fin)), "ascii"))
            steps += 1
        step += 1
    return h.hexdigest(), steps, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out", default="results/sched_replay.json")
    a = p.parse_args()

    try:
        import nanoserve_core as nc
        have_cpp = True
    except ImportError:
        have_cpp = False
        print("[!] nanoserve_core not built (make cpp); python-only run")

    results = []
    for seed in a.seeds:
        trace = build_trace(a.n, seed)
        h_py1, steps, t_py = run_trace(PyScheduler(NUM_BLOCKS, BLOCK_SIZE), trace)
        h_py2, _, _ = run_trace(PyScheduler(NUM_BLOCKS, BLOCK_SIZE), trace)
        det = h_py1 == h_py2
        row = {"seed": seed, "steps": steps, "hash": h_py1[:16],
               "py_deterministic": det, "py_s": t_py}
        line = (f"seed {seed}: {steps} steps  hash {h_py1[:16]}  "
                f"py {steps / t_py:,.0f} steps/s")
        if have_cpp:
            h_cc1, _, t_cc = run_trace(nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE), trace)
            h_cc2, _, _ = run_trace(nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE), trace)
            row.update({"cpp_matches_py": h_cc1 == h_py1,
                        "cpp_deterministic": h_cc1 == h_cc2, "cpp_s": t_cc})
            line += (f"  c++ {steps / t_cc:,.0f} steps/s "
                     f"({t_py / t_cc:.1f}x)  match={h_cc1 == h_py1}")
        results.append(row)
        print(line)
        assert det, "python scheduler is nondeterministic"
        if have_cpp:
            assert row["cpp_deterministic"], "c++ scheduler is nondeterministic"
            assert row["cpp_matches_py"], "c++ decisions diverge from python"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"max_batch": MAX_BATCH, "num_blocks": NUM_BLOCKS,
                   "rows": results}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

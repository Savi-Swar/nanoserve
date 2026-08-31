# The C++ hot path

Phase 2 moves the scheduler's bookkeeping (block allocator, admission, per-step
slot planning, eviction) into C++. This doc records why, the one measurement
that set the design, and what equivalence with the Python engine means here.

## The gate came first

Rewriting things in C++ because C++ is fast is how people end up with a fast
implementation of the wrong bottleneck. So the first artifact was a decision
rule, written down before measuring:

    share = (t_step - t_forward) / t_step

per engine state at B in {1, 8, 16}. If the Python around the model forward is
at least 15% of a decode step, the port gets an end-to-end tail-latency
framing. Below that, the port's story is the microbenchmarks, determinism, and
cancel-storm tails, and it says so honestly.

On the T4: the plain batched state measures ~0% (the forward is the step).
The paged state measures 11.1% at B=1 and 16.0% at B=8. The paged path
crosses the line: block-table bookkeeping, slot math, and gather assembly are
real end-to-end costs, and that is exactly the part the C++ scheduler owns.

## The boundary costs more than the work

First version: bind the C++ allocator with pybind11, call it from Python
per operation, same call pattern as the Python allocator. Measured throughput
on the append-token hot loop:

| implementation             | M ops/s |
|----------------------------|---------|
| Python allocator           | 10.7    |
| C++ via pybind11, per op   | 10.0    |
| C++ via pybind11, batched  | 30.9    |

The per-op C++ allocator is slower than the Python it replaces. The work is
tens of nanoseconds; the Python-to-C++ crossing is a few hundred. Every fast
data structure behind a chatty boundary is a slow data structure.

That measurement is the design rule for everything after it: one crossing per
decode step. The scheduler exposes exactly two hot-path entry points,

    plan_step()          -> (write slot, current length) per row
    commit_step(eos_hit) -> finished row indices, frees their blocks, compacts

and nothing else is called while decoding. Admission (admit / can_admit) is
off the per-step path.

## Two freelists, one contract

The allocator's free list has a mutex version and a lock-free version, both
behind the same interface, both matching the Python allocator's LIFO pop order
so all three can be driven in lockstep and compared exactly.

The lock-free version is a Treiber stack over block indices with the classic
ABA fix: head is a single `atomic<uint64_t>` packing a 32-bit index and a
32-bit generation tag, and the tag bumps on every successful push and pop. A
stale CAS that would land on a recycled index fails on the tag instead.
Because the stack holds indices into a fixed pool rather than heap nodes,
there is no reclamation problem and no need for hazard pointers, which is the
part of lock-free programming that actually hurts.

The differential test drives Python, mutex C++, and Treiber C++ through the
same randomized add/append/free schedule and asserts identical tables,
identical free counts, and identical rollback behavior on exhaustion (add_seq
that fails halfway must return every block it took).

## Equivalence you can hash

"The C++ scheduler does the same thing" is a claim that decays unless it is
mechanically checked. Scheduling decisions here do not depend on token values
(finish is a token budget; EOS arrives as a mask), so a trace of
(arrival_step, prompt_len, out_len) fully determines every decision: what
admits when, which pool slot every token lands in, what finishes and frees.

The replay harness (`bench/sched_replay.py`) drives both implementations
through the same trace with virtual time (the step counter), logs every
admission with its block table, every per-step slot plan, and every finish,
and hashes the log. Three seeds, ~2400 steps each:

| seed | steps | decision hash    | python steps/s | c++ steps/s |
|------|-------|------------------|----------------|-------------|
| 0    | 2406  | 56271d35ef6b1a37 | 267k           | 475k (1.8x) |
| 1    | 2442  | 519cf8ca91ab9fa5 | 255k           | 486k (1.9x) |
| 2    | 2463  | 045493b50dc0c2d9 | 254k           | 477k (1.9x) |

Same implementation twice gives the same hash (determinism). Python and C++
give the same hash (drop-in equivalence, bit for bit, across every decision).
And the C++ bookkeeping is 1.8-1.9x faster at one crossing per step. That
ratio is a floor: the harness spends the same hashing and trace-driving cost
on both sides, so the pure bookkeeping gap is larger than what the loop shows.

## Contention, and the collapse that fixed itself

`cpp/bench_contention.cpp` hammers one shared freelist with alloc/free pairs
from 1 to 8 threads (fixed total work, median of 5). The first run produced
the most useful negative result of the pillar:

| threads | mutex Mops/s | naive CAS Mops/s | treiber+backoff Mops/s |
|---------|--------------|------------------|------------------------|
| 1       | 234          | 428              | 426                    |
| 2       | 129          | 53               | 386                    |
| 4       | 80           | 15               | 396                    |
| 8       | 68           | 4.1              | 366                    |

The naive CAS loop, the thing people mean when they say "lock-free is fast",
is 2x the mutex alone and then collapses ~100x at 8 threads. Every failed CAS
immediately re-reads the same head cache line every other thread is fighting
over, so the coherence traffic grows with contention while the mutex quietly
serializes waiters off the line. Lock-free means progress guarantees, not
speed.

The fix is exponential backoff on CAS failure: spin away from the line for a
doubling number of pause instructions before retrying. That version matches
the naive one uncontended exactly (an uncontended CAS never fails, so the
backoff path never runs) and holds nearly flat under contention, 5.4x the
mutex at 8 threads. It was promoted into `TreiberFreeList` on those numbers;
the naive loop stays in the bench so the collapse remains reproducible. The
differential tests pass unchanged, since backoff alters timing, not order.

The latency harness (`bench/latency_study.py`) follows the rules that the
benchmarking literature keeps having to re-teach:

- Open loop. Arrivals come from a Poisson process that does not wait for the
  server. Closed-loop load generators hide queueing delay behind their own
  backpressure (coordinated omission) and make p99 look better than any real
  client would see.
- Raw samples, pooled. Every inter-token gap is kept; percentiles come from
  the pool, not from averaging per-run percentiles.
- No p99.9 without the samples. The harness refuses to print p99.9 below
  100k pooled ITLs. With fewer, that number is just the k-th worst sample
  wearing a suit.
- Noise gets a vote. Each engine runs 5 times; two engines are called
  different at p99 only if their per-run p99 ranges do not overlap.

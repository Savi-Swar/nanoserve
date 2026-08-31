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

The full table from the T4 (Qwen2.5-0.5B, 256-token context, 30 timed steps):

| state       | B=1   | B=8   | B=16  |
|-------------|-------|-------|-------|
| batched     | 0.0%  | 3.4%  | 0.0%  |
| paged       | 12.8% | 9.5%  | 15.8% |
| paged_fused | 0.2%  | 2.3%  | 3.2%  |

Two conclusions, one of them uncomfortable. The gather-paged path crosses the
line at B=16: block-table bookkeeping, slot math, and gather assembly are
real end-to-end costs. But the fused path, which replaced the per-step gather
with persistent device tensors and the Triton kernel, already ate almost all
of that overhead in Python. The C++ port's end-to-end tail claim applies to
the path we no longer ship as the best engine. So the port is framed by what
the gate actually licenses: the bookkeeping speedup is measured at the
microbench level, the equivalence is proven by replay, and the contention and
cancel-storm behavior carry the systems story. Writing the decision rule down
before measuring is what keeps this honest.

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

## Storms

The abort path is the least-tested path in every inference server; vLLM and
SGLang both shipped bugs where disconnected clients kept decoding or leaked
their memory, and no published benchmark injects client disconnects
open-loop and measures what they do to the requests that stay. So that is
what `bench/storm_study.py` does: seeded storms against the live engine,
survivor inter-token tails split by phase, and hard accounting at quiesce.

T4, rate 8 req/s, 150 requests, 30% chaff, three seeds per scenario:

- Baseline survivor p99: 98-99 ms.
- Disconnect storm (each chaff request cancelled after reading 1-32 tokens):
  survivor p99 during the storm 97-102 ms. Indistinguishable from baseline.
  Mid-stream eviction and block reclaim cost the batch nothing measurable.
- Swizzle (chaff cancelled one by one in random order): flat, 96-98 ms.
- Burst (all 45 chaff aborted in one instant): the one real signature.
  During-window survivor p99 hit 160 ms in one of three seeds (94-96 in the
  others) - a single step that evicts 45 rows and compacts the batch is
  visibly not free, roughly a p99 doubling for the requests unlucky enough
  to share it.
- Invariants: 12 of 12 runs clean. Every block back in the pool, every
  request terminal, every survivor got its full token budget.

Writing the harness found a wedge before it ever ran: the head-of-line
poison scenario (a request whose reservation exceeds the whole pool)
force-admitted into an allocator exception that killed the engine thread
and starved everything behind it - the same failure class as vLLM issue
39734. Rejected at admission now, with a regression test.

## Measuring the fused engines' tails

First results from the T4 (rate 8 req/s, 250 requests, 96 tokens, 5 runs,
117k pooled ITLs per engine):

| engine           | p50  | p90  | p99   | p99.9 | max    |
|------------------|------|------|-------|-------|--------|
| continuous       | 32.6 | 72.5 | 77.3  | 90.5  | 233    |
| continuous_fused | 31.0 | 65.8 | 70.4  | 82.4  | 2045   |
| paged            | 43.3 | 84.2 | 123.2 | 132.4 | 176    |
| paged_fused      | 34.5 | 70.2 | 94.8  | 110.8 | 1380   |

(ms). The kernel's tail claim survives the noise rule for the continuous
pair: p99 ranges [69.6, 70.9] vs [76.7, 78.4] do not overlap, a 9% p99 cut.
The paged pair's ranges overlap, so that comparison stays unclaimed. A
second independent run reproduced the claim (67.5 vs 73.6 ms, ranges again
disjoint) and put p99.9 at 74.7 vs 84.2.

And the max column caught a real bug. Both fused engines showed a 1.4-2.0s
worst gap on an otherwise sub-111ms p99.9: the Triton JIT compiling a new
NUM_SPLITS specialization the first time a request's length crossed a split
threshold, inside that request's inter-token latency. The fix compiles every
variant at engine construction (warm_decode_kernels); batch size is not a
specialization key, so a handful of tiny launches covers the space. The
re-run confirms it: paged_fused max fell from 1380 to 155 ms (now below the
gather engine's own max) and continuous_fused from 2045 to 456. This is the
argument for reporting max alongside percentiles: p99.9 absorbed the spike
and said nothing.

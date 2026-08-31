# The decode-attention kernel

The 16%-of-vLLM gap in the original numbers had one owner: decode attention ran
as `repeat_kv` copies plus unfused SDPA over a contiguous gather of the paged
pool, all orchestrated from Python. This writes that path out of existence: a
fused decode-attention kernel in Triton, one for contiguous KV and one that
indexes the paged pool through the block table directly, with split-K
flash-decoding for the small-batch regime.

## Shape of the kernel

One program per (sequence, head). The program holds the single query row,
streams K/V tiles of `BLOCK_L` tokens, and maintains an online softmax: running
max `m`, denominator `l`, and an fp32 accumulator, so the full score row never
exists in memory. GQA is native: head `h` reads kv-head `h // (H/H_kv)`, which
deletes the `repeat_kv` materialization entirely (a 7x copy for Qwen2.5-0.5B's
14:2 head ratio). The paged variant computes each token's pool slot as
`table[pos // block_size] * block_size + pos % block_size` and loads straight
from the flat pool; there is no gather step and no cross-sequence padding,
because each program walks only its own sequence's true length.

Split-K (flash-decoding): at B=1 this grid puts 14 programs on a 40-SM T4 and
the GPU starves. `NUM_SPLITS` programs share one (sequence, head), each
reducing a chunk into partial `(m, l, acc)`; a second one-launch kernel merges
the partials with the standard log-sum-exp combine.

## Measured (T4, fp16, H=14, H_kv=2, D=64, T=2048)

| B | sdpa+repeat_kv | kernel | splits | speedup |
|---|---|---|---|---|
| 1 | 0.429 ms | 0.085 ms | 8 | 5.1x |
| 4 | 0.735 ms | 0.115 ms | 4 | 6.4x |
| 8 | 1.200 ms | 0.181 ms | 2 | 5.9x |
| 16 | 1.432 ms | 0.234 ms | 1 | 6.7x |
| 32 | 2.905 ms | 0.412 ms | 1 | 7.9x |

Launch parameters were swept on the T4 (`make` target `tune`): 256-token tiles
win nearly everywhere, and modest split counts beat aggressive ones. sm_75 has
no `cp.async`, so software pipelining buys nothing; the whole game is tile
size, warp count, and split count.

End-to-end (the ladder, open-loop Poisson, rate 16): `continuous_fused` 287.8
tok/s = 10.5x naive (vs 271.6 for sdpa continuous), and `paged_fused` 276.3 vs
229.0 for gather-paged. The second number is the interesting one: paging used
to cost 16-19% of throughput against the contiguous engine, all of it the
per-step gather + unfused attention. With the kernel reading the pool directly,
the penalty is gone and paged keeps its 68%->4% fragmentation win for free.

## Two things that went wrong, on purpose kept in the record

**The merge that ate the split.** The first split-K implementation merged
partials with ~6 torch ops on the host. Six kernel launches at decode sizes
cost more than the split saved: B=1 went 0.171 ms -> 0.301 ms. Fusing the merge
into a single Triton launch flipped it to 0.089 ms. Launch overhead is not a
detail at these sizes; it is the budget.

**The fp16 tie.** The model-level oracle initially "failed": one token in 192
differed from the SDPA baseline. Teacher-forced diagnosis showed the flip sat
at a step where SDPA's own top1-top2 logit margin was exactly 0.0 — a true
fp16 tie, which argmax resolves by reduction order, and this kernel reduces in
a different order than SDPA. Token-identity across different reduction orders
is an invalid oracle. The correctness contract is now: op-level allclose
against SDPA (26 cases), plus teacher-forced model-level agreement where every
disagreement must sit at a provable near-tie (margin < 1e-3); a flip with a
real margin still fails the build.

## The knee moved

The crossover study measures engine-level decode throughput vs batch at
S=2048. With SDPA the curve kneed at B~4, ten times below the roofline's
B*=39. Through the kernel:

| B | sdpa tok/s | kernel tok/s |
|---|---|---|
| 1 | 28 | 32 |
| 4 | 109 | 126 |
| 8 | 148 | 246 |
| 16 | 162 | 495 |

SDPA scales 5.8x from B=1 to 16; the kernel scales 15.5x, near-linear, and the
knee detector finds no saturation anywhere in the measurable range (B=32 OOMs
in the sdpa prefill, not the kernel). The old knee was never memory bandwidth;
it was the attention path's own overheads, and removing them puts the
measured curve back on the roofline's shape.

## What the remaining gap is

Decode on a T4 at short contexts is launch- and Python-bound before it is
bandwidth-bound: a step is ~200 kernel launches plus the transformers Python
stack, and the attention kernel is now a small slice of it. The ladder says
the same thing: op-level the kernel wins 5-8x, end-to-end the fused engines
gain ~5-20% because attention was one slice of the step. The next factor is
not a better attention kernel; it is removing per-step launch and dispatch
overhead (CUDA graphs) and the scheduler's Python (the C++ hot path). Both
are in flight.

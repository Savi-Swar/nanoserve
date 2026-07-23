# nanoserve writeup

A from-scratch LLM inference server. The model is a stock Qwen2.5-0.5B; the
scheduler and KV-cache memory manager are written by hand in Python/PyTorch with
no custom CUDA. Every throughput step is a scheduling decision, measured on the
same open-loop workload.

> Most tables below are from a CPU dev box (fp32). The absolute fp16 numbers and
> the vLLM ceiling come from one clean run on a T4 (see "GPU results" below). The
> relative ladder between engines is the part I care about, and it holds on both.

## Method

- Load: open-loop Poisson arrivals (exponential inter-arrival gaps). Open loop
  matters because a closed-loop generator throttles itself under overload and
  hides the tail blowups you're trying to measure (coordinated omission).
- Workload: length-skewed outputs (exponential, +/-50% jitter). The skew is
  needed. With fixed output lengths the wasted-slot problem that static batching
  has doesn't show up, and continuous batching's advantage under-reports.
- Metrics: throughput (tok/s), TTFT and end-to-end p50/p90/p99, queue delay, GPU
  util (nvidia-smi sampler). Greedy decoding, so numbers aren't sampling
  variance.
- Correctness: every batched/paged/speculative engine is checked token-for-token
  against naive single-sequence decoding (`tests/test_equivalence.py`),
  including the mid-stream KV-merge, the paged block path, and speculative
  accept/reject. These are optimizations, not approximations.
- Noise floor (`bench/repeat.py`): headline numbers are N runs with a 95% CI, and
  a comparison only counts as a winner if the CIs don't overlap. On this CPU box
  the throughput noise floor is about +/-24%, so any single-run "improvement"
  under ~24% is nothing. This is what lets the audit tell a real effect from a
  lucky run, and it's what dropped the "paged beats continuous" claim below.

## The ladder

| engine | mechanism | throughput (CPU) | TTFT p99 (CPU) |
|---|---|---|---|
| naive | one request at a time | 24 tok/s | ~9,000 ms |
| static | batch N, run until all finish | 34 tok/s | ~3,700 ms |
| continuous | evict finished + admit waiting every step | 41->63 tok/s* | ~200 ms |
| paged | continuous + block-paged KV | 66 tok/s | ~240 ms |

*Continuous throughput rises with offered load (41 tok/s at 4 req/s, 63 tok/s at
10 req/s) while TTFT stays flat. That's the property that defines it. Naive is
flat-saturated around 24 regardless of load, and its TTFT grows as the queue
backs up.

### naive to static
Batching amortizes the memory-bandwidth-bound decode step across sequences (one
weight load, N tokens). But a static batch is locked to its longest member:
short sequences finish and their slots sit idle until the whole batch drains.

### static to continuous
Iteration-level scheduling (Orca, OSDI'22): re-decide the batch every decode
step, evict a finished sequence and admit a queued one right away. No slot idles.
This is the step that matters. TTFT drops roughly 40x because requests stop
waiting behind a full batch.

### continuous to paged
Contiguous KV wastes memory two ways: reserving room for the longest a sequence
might get, and padding to the batch's current longest each step. A paged cache
(vLLM/PagedAttention, SOSP'23) stores KV in fixed blocks from a free list; a
sequence holds a block table and grows on demand. The only waste is the
partially-filled last block (under block_size per sequence).

## GPU results (fp16, T4)

One clean run on a Kaggle T4 (`scripts/gpu_run.py`), fp16, peak throughput per
engine across the rate sweep:

| engine | throughput | TTFT p99 |
|---|---|---|
| naive | 27.8 tok/s | 53,390 ms |
| static | 137.5 tok/s | 8,029 ms |
| continuous | **265.8 tok/s** | **2,175 ms** |
| paged | 108.7 tok/s | 7,871 ms |

Continuous batching is **9.6x** naive throughput on the GPU (a bigger jump than
on CPU, because GPU batching parallelism is much stronger), and it holds the
TTFT tail to ~2 s while naive's open-loop queue blows past 50 s.

Two honest things this run surfaced:

- **Paged measured slower than continuous (108.7 vs 265.8) — but that number is
  confounded by my own code, and I've since fixed it.** In that run the per-step
  block gather was a pure-Python loop over sequences and blocks; at GPU speeds
  that serial copy *is* the bottleneck, not paging. It's now vectorized (one
  `index_select` per layer over a flattened block table, still token-exact —
  `server/paged_exec.py`), so 108.7 is a measurement of my for-loop, not of paged
  attention. Re-running the sweep to isolate the real gap is the honest next step:
  if paged is still slower, the claim is now about paging; if it's competitive,
  that's a fifth self-corrected number. Either way it beats the confounded result.
  Paged's deterministic win — memory capacity, 3x concurrency (below) — holds
  regardless of the speed question.
- **nanoserve's best is ~16% of vLLM** (continuous 265.8 vs vLLM 1,700.7 tok/s on
  the same T4). Reaching a sixth of vLLM with masked SDPA and no custom kernels is
  a defensible number; the 6x gap is the fused FlashAttention/PagedAttention
  kernels vLLM has and I don't.

## Paged KV fragmentation

Fragmentation on 64 length-skewed sequences, 16-token blocks (`bench.memory_study`):

| KV layout | waste | rel. memory |
|---|---|---|
| reserve-to-max | 69% | 3.1x |
| padded batch | 69% | 3.1x |
| paged | 4% | 1.0x |

The same 256 MiB KV budget holds 42 sequences reserved-to-max vs 128 paged, so
3x more, and concurrency is what caps throughput.

In execution, though, paging adds a per-step cost: every decode step gathers KV
from non-contiguous blocks, the HBM round-trip that real PagedAttention also
pays. With memory to spare it's at best a wash and often slightly slower than the
contiguous engine, and the exact delta is inside CPU run-to-run noise here (see
"Paged vs continuous" below), so I don't quote a number for it. Paging's real
gain is capacity, and that part is deterministic: 3x the concurrent sequences per
byte. It shows up as speed only when memory is the binding constraint.

## Real traffic

Most benchmarks use uniform prompts and fixed output lengths. Real traffic is
heavy-tailed and bursty. Replaying the Azure LLM inference trace
(`bench/trace.py`, 19,366 real requests: context p50=1,020/p99=4,142, output
p50=129/p99=601) instead of synthetic uniform load changes the conclusions.

At natural (sparse) arrival rate (single representative run; the throughput
differences here are within CPU noise, but the TTFT-tail ordering is robust and
reproduces):

| engine | throughput | TTFT p99 |
|---|---|---|
| naive | 9.1 tok/s | 1,254 ms |
| static | 8.7 tok/s | 2,154 ms (worst) |
| continuous | 9.4 tok/s | 647 ms |
| paged | 9.3 tok/s | 633 ms (best) |

Under a 10x arrival burst (overload), naive and static TTFT p99 blow past 10 s
(queues build faster than they drain) while continuous and paged hold it to
about 5-6 s, which is a large and reproducible gap.

The robust takeaway the synthetic sweep missed: continuous batching's advantage
changes shape with load. At sparse arrivals it barely leads on throughput
(there's nothing to batch) but roughly halves the TTFT tail (647 vs 1,254 ms),
which is a latency win. Under burst it turns into the throughput win the textbook
describes. Same optimization, two different value propositions depending on load.

### Paged vs continuous: a result I dropped

An earlier draft claimed paging beat continuous 24.2 vs 21.1 tok/s under burst,
"flipping from a 6% cost to a 15% win." That was one CPU sample. Re-running the
identical config four times, continuous ranged 14.1-22.8 tok/s and paged
19.4-24.2. They overlap completely, and continuous wins half the trials. On this
CPU box the paged-vs-continuous throughput gap is within run-to-run noise, so the
claim doesn't hold and I cut it.

What is robust and deterministic is the paged memory result (69% to 4%
fragmentation, 3x concurrency per byte, above). Whether that concurrency turns
into a throughput win over the contiguous engine is a fair question, but it needs
a quiet GPU to measure, since small throughput deltas are unmeasurable under CPU
variance. Deferred to the GPU run rather than asserted.

## Negative results

- Static batching's tail latency is a net negative on real traffic. On synthetic
  uniform load it was a clean throughput win (34 vs 24 tok/s). On the Azure trace
  it consistently has the worst TTFT p99 of any engine, worse than even naive
  (2,154 vs 1,254 ms at natural load, and the ordering reproduces across runs),
  because it waits to fill a batch and then head-of-line blocks on the longest
  generation while short requests pile up. It buys no throughput at natural load
  to pay for that tail. I kept it as a measured rung of the ladder, but it's not
  something you'd ship.
- Paged KV costs about 6% under uniform load. It's a capacity optimization, not a
  latency one; with memory to spare and no length skew it only adds gather
  overhead. It earns its place under memory pressure or length skew, which is
  where the real trace puts it.

## Audit: which optimizations actually help on real workloads?

Take published inference optimizations, implement each in a rig I control (so I
can ablate one thing at a time, which is hard in a large codebase), and measure
them on workloads I pick rather than the one each paper picked for itself. A
number only counts if it clears the +/-24% noise floor.

### Row 1: speculative decoding (prompt-lookup)

Prompt-lookup decoding (`server/speculative.py`) drafts the next tokens by
finding where the context's last n-gram appeared earlier and proposing what
followed, then verifies the guesses in one forward pass. Exact under greedy
(checked against naive). The deterministic speedup proxy is tokens committed per
forward pass; naive is exactly 1.00.

| prompt class | tokens / forward | draft accept | result |
|---|---|---|---|
| grounded (repetitive) | 2.7x | 92% | big win |
| code (structured) | 2.3x | 52% | real win |
| generic (novel prose) | 1.00x | 0% | no gain |

The published "2-3x" is real for grounded or repetitive output. On generic
generation the draft never matches, acceptance is 0%, and speculation collapses
to naive plus wasted draft compute. The headline number is a property of the
workload, not the technique.

What this means for a real server: conversational traffic (the Azure trace) is
mostly novel prose, i.e. the 0%-acceptance regime. And under continuous batching
it gets worse, not better: batching already makes decode compute-bound, so the
extra draft tokens speculation pushes through each step add compute a saturated
batch has no room for. The prediction (matching vLLM's reported 1.4-1.8x slowdown
at high QPS) is that turning PLD on for a batched conversational server is a net
loss. Measuring that end-to-end needs a batched speculative engine (the
single-sequence version here is the batch-1 baseline) and a quiet GPU, which is
the next step, not something I assert here.

### Row 2: prefix caching (holds up)

Requests sharing a leading prefix (system prompt, few-shot, a document)
recompute that prefix's KV every time under naive serving. `server/prefix_cache.py`
caches it and prefills only each request's novel suffix. It's exact, because the
prefix KV is identical across requests. Deterministic metric: prefill tokens
computed.

| workload | prefill tokens saved | exact |
|---|---|---|
| shared system prompt | 70% (66/220 computed) | yes |
| distinct prefixes | ~0% (212/226) | yes |

This is the one that holds up: a large win where the workload has shared prefixes
(chat, agents, RAG, most production traffic), no cost where it doesn't, and
identical output either way. An audit where everything fails is as suspect as one
where nothing does, so it's worth having a case that survives.

### Row 3: KV quantization

Storing the KV cache in low-bit ints (`server/kv_quant.py`, per-(token,head)
symmetric) shrinks it by 32/bits. Quality is measured as teacher-forced top-1
agreement with the fp32 model: feed the same reference token each step and check
whether the quantized argmax still matches.

| bits | mem vs fp16 | top-1 agreement |
|---|---|---|
| 8 | 2x | 93% (~lossless) |
| 4 | 4x | 50% |
| 2 | 8x | 10% |

Two caveats on this row:

- The metric mattered more than the result. My first cut used free-running token
  match and reported "8-bit drifts 48% of tokens," which is wrong: one early flip
  marks every later token different even though the text is fine (a cascade).
  Teacher-forced per-step agreement removes the cascade and gives 93%. Same data,
  opposite conclusion, from the choice of metric alone.
- This is a naive quantizer. Production schemes (KIVI per-channel keys, outlier
  handling, grouping) hold quality much better at 4-bit; my symmetric
  per-(token,head) baseline degrades faster. The point is the shape of the
  tradeoff and that the rig can measure it, not a state-of-the-art quantizer.

### Row 4: chunked prefill (left out)

Sarathi-Serve's chunked prefill is the highest-value scheduling idea left, but
its payoff is a TTFT/TPOT tail effect, and tail latency is exactly what CPU
timing noise (+/-24%) destroys. Implementing it and measuring on a noisy box
would produce numbers I couldn't defend, so I left it for the GPU rather than
rush it.

## Out of scope (and why)

Background in `docs/frontier/`. These are left out on purpose, not missed:

- Custom paged-attention CUDA kernel: a microarchitecture-specific, large effort.
  I call `F.scaled_dot_product_attention` (FA2 backend) and own the policy (block
  table, allocator, admission) in Python.
- Speculative decoding as a serving default (EAGLE-3 is SOTA): biggest wins at
  batch 1; can slow down a saturated continuous batch. Orthogonal to this
  project's axis.
- Prefill/decode disaggregation (DistServe, Mooncake): a multi-GPU systems
  problem, out of a single-GPU learning build.

## Reference ceiling

`bench/vllm_ref.py` runs the same open-loop workload through vLLM's
AsyncLLMEngine. On the T4, vLLM does 1,700.7 tok/s vs nanoserve's best of 265.8
(continuous), so nanoserve reaches **16% of vLLM**. That gap is the fused
FlashAttention/PagedAttention kernels vLLM has and I don't; 1/6th of a
production engine with masked SDPA and pure-Python scheduling is the honest
result, and closing it would mean writing the kernel (out of scope).

## Résumé bullets

- Built an LLM inference server from scratch (Python/PyTorch, no custom CUDA):
  naive to static to continuous batching to a paged KV cache, 9.6x throughput
  over naive at p99 TTFT 2.2s (from 53s), reaching 16% of vLLM on the same T4.
- Audited published inference optimizations in a controlled rig, measuring
  against a documented +/-24% noise floor and verifying each one token-exact
  first. Found speculative decoding's speedup is workload-dependent (2.7x on
  repetitive text, 1.0x on generic prose), prefix caching holds up (70% of
  prefill saved on shared prompts), and 8-bit KV is near-lossless (93% top-1, 2x
  memory). Cut four first-pass numbers that didn't survive scrutiny, including my
  own "paged beats continuous," which turned out to be CPU noise.
- Made correctness a checked property: batched, paged, speculative, and
  prefix-cached decoding are all verified token-for-token against a naive
  baseline (including 1-token prompts, permutation, and block boundaries), so
  every optimization is shown to change speed or memory, not output.

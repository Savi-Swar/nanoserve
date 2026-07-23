# Designing a High-Performance LLM Inference Server — Research Report

**Purpose.** This report informs a from-scratch inference server in Python/PyTorch built to *learn systems*, using a small model (Qwen2.5-0.5B). The intended build arc is **naive → static batching → continuous batching → paged KV cache**, with ablations that measure which optimization actually mattered. Throughout, the report gives (a) the ordering of optimizations by impact, (b) the specific knobs that matter, and (c) realistic speedup magnitudes so you can *predict* your ablation results and know whether your measurements are sane.

Citations use `[N]` markers linking to the **Sources** section.

> **TL;DR for a 0.5B model on one GPU.** The two changes that dominate real serving throughput — *continuous batching* and *paged KV cache* — were designed for the regime where KV cache is the capacity bottleneck. For a 0.5B model that regime is hard to reach on a big GPU (weights ≈ 1 GB, KV ≈ 12 KiB/token). To make your ablations *show* the textbook speedups you must deliberately create memory/queue pressure: long sequences, many concurrent requests, and a `gpu_memory_utilization` cap. Otherwise continuous batching will still help a lot (it fixes GPU *idle time*, not just memory), but PagedAttention may look like noise. This is itself a finding worth reproducing. See §11.

---

## 1. The throughput ladder: naive → static → continuous batching

### The mechanism at each rung

Autoregressive decoding runs one forward pass per output token. A single request during decode processes **one token at a time**, so the GPU's matrix units are almost idle — the step is dominated by *reading weights and KV cache from HBM*, not by compute (§4). The only way to get utilization up is to run **many requests' tokens through the same weight-load** — i.e., batch across requests. The ladder is about *how* you batch.

**Rung 0 — Naive / single-request.** Process one request end-to-end, then the next. GPU utilization during decode is a few percent for a small model. This is your baseline; everything is measured against it.

**Rung 1 — Static (request-level) batching.** Collect *B* requests, pad them to a common length, run them as one batch until **all** finish, then take the next batch. Problem: generation lengths vary widely. A batch of 8 where one request emits 500 tokens and the rest emit 20 keeps 7 slots idle (padded/finished) for the tail of the run. This is *head-of-line blocking inside the batch* and it wastes most of the batch's potential. Static batching also forces you to *wait* to fill a batch, inflating latency at low load. FasterTransformer is the canonical optimized static-batching baseline; Anyscale measured it at **~4×** naive throughput. [3]

**Rung 2 — Continuous / iteration-level batching (the Orca contribution).** Instead of scheduling whole requests, Orca (OSDI '22) schedules at the granularity of a **single iteration** (one decode step): "the scheduler invokes the execution engine to run only a single iteration of the model on the batch," and the batch composition is re-decided *every* step. [1][2] The instant one sequence emits its EOS, it leaves the batch and a **queued request is admitted in its place at the next iteration** — no waiting for the slowest sequence, no idle slots. This keeps the batch continuously full, which is exactly what a memory-bandwidth-bound decode needs.

Orca also introduced **selective batching**: attention cannot be batched naively across sequences of different lengths (each has its own KV history), so Orca batches the *shared* dense ops (QKV projection, MLP, output projection) across all tokens in the batch while handling attention per-sequence. This is the structural insight your implementation will have to reproduce: **everything except attention batches trivially; attention is the part that needs care** (§7).

### Realistic throughput multipliers (all relative to a naive/static baseline)

From the widely-cited Anyscale benchmark (1,000 requests, 512 input tokens, output lengths ~Exp(mean 128), max 2,048), measured on the *same* hardware: [3]

| System | Technique | Throughput vs naive static |
|---|---|---|
| Naive static batching | rung 0/1 | 1× (baseline) |
| FasterTransformer | optimized static batching | ~4× |
| Ray Serve / HF TGI | continuous batching | ~8× |
| vLLM | continuous batching + PagedAttention | ~23× |

Anyscale's headline is **up to 23× throughput and lower p50 latency across the whole CDF** for continuous batching + paging over naive batching. [3] The **continuous-batching step alone (~4×→~8×, i.e. ~2× over optimized static; often 2–5× over static in production)** is the single largest "unlock" for a decode-heavy workload, because it directly removes GPU idle time. [1][3] Orca's own paper reports a much larger **36.9×** headline vs FasterTransformer, but that is at a fixed latency target and on large models — treat vendor/paper headline numbers as *upper bounds under favorable conditions*, and the Anyscale table as the more transferable guide. [1][3]

**What to expect in your ablation.** The gap between static and continuous batching *widens with output-length variance*. If you benchmark with fixed-length outputs, static batching looks fine and continuous batching gives little — because the wasted-slot problem it solves doesn't exist. To make the win visible, use a **skewed output-length distribution** (e.g., exponential) and a stream of arrivals, not a fixed batch. This is the most important experimental-design point in the whole report.

---

## 2. PagedAttention / paged KV cache (vLLM)

### The problem it solves: KV fragmentation

The KV cache for one sequence grows by one token's worth of K and V per decode step, up to an *a priori unknown* final length. The pre-vLLM approach reserved a **contiguous** buffer sized for the maximum possible length (e.g., 2,048) for every active sequence. This wastes memory three ways: [4]

- **Internal fragmentation** — the slots reserved for tokens the sequence never generates (a request that stops at 30 tokens but reserved 2,048).
- **Reservation waste** — memory reserved for not-yet-generated tokens that could have served *other* requests right now.
- **External fragmentation** — the buffer allocator leaves unusable gaps between different requests' contiguous regions.

The vLLM paper measured that **existing systems waste 60–80% of KV memory** to fragmentation and over-reservation, so only 20–40% of KV memory does useful work. [4] Because KV cache is the capacity bottleneck (§3), wasting it directly caps the batch size and therefore throughput.

### The mechanism: paging, borrowed from OS virtual memory

PagedAttention partitions each sequence's KV cache into fixed-size **blocks** (default **16 tokens/block**), each block holding the K and V for that many contiguous positions of *one* sequence. [4][5] Blocks need **not** be contiguous in GPU memory. Three data structures make this work — directly analogous to OS paging:

- **Free list** — a pool of all physical KV blocks; allocation pops a block, freeing pushes one back. O(1).
- **Block table** — per sequence, a small array mapping *logical* block index → *physical* block address (the "page table"). The attention kernel consults this to find where each chunk of KV actually lives.
- **On-demand allocation** — a new physical block is allocated only when the current one fills (every 16 tokens). A sequence never holds more than `block_size − 1 = 15` tokens of slack.

**Result:** internal fragmentation is bounded to *at most one block per sequence* (≤15 tokens), and external fragmentation is *eliminated* because all blocks are the same size and interchangeable. vLLM reports waste **under ~4%**, versus the 60–80% above. [3][4] That reclaimed memory becomes larger batches, which is where the extra throughput over plain continuous batching (the ~8×→~23× jump in the Anyscale table) comes from. [3][4]

### Block-size choice (a real knob)

- **Smaller blocks** (e.g., 8) → less internal waste, but longer block tables, more kernel indexing overhead, and more allocator traffic.
- **Larger blocks** (e.g., 32) → cheaper indexing and fewer allocations, but more internal fragmentation and coarser sharing.
- **16 is vLLM's default** as the empirical sweet spot. [4][5] For your from-scratch build, treat block size as an ablation axis; expect it to be a *second-order* knob (single-digit % effects) compared to whether you page at all.

### Copy-on-write: sharing prefixes and beams

Because blocks are reference-counted, multiple sequences can **share** the physical blocks of a common prefix. When two sequences share a prefix and one needs to write into the last shared block, vLLM does **copy-on-write**: copy that one block, then let the writer diverge — exactly like OS `fork()`. [4] This is what makes **parallel sampling** (n samples from one prompt), **beam search** (many hypotheses share history), and **shared system prompts** cheap: the shared context is stored once and referenced by many. The paper reports large memory savings on beam search / parallel sampling from this alone. [4] (Automatic prefix caching, §6, is the persistent-across-requests generalization of the same idea.)

---

## 3. KV cache memory math — the capacity bottleneck

### The formula

Per token, the KV cache stores one K vector and one V vector *per KV head, per layer*:

```
kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
```

The leading `2` is for K and V. For a full sequence and a batch:

```
kv_bytes_total = kv_bytes_per_token * seq_len * batch_size
```

Note **`n_kv_heads`, not `n_heads`** — this is the whole point of GQA/MQA (§ below).

### Why KV cache, not weights, is the bottleneck

Weights are a **fixed** cost paid once, independent of load. KV cache scales with `batch_size × seq_len` — it grows with *every concurrent request and every token*. On a production serving GPU, weights occupy a fixed slice and **KV cache consumes essentially all remaining memory**; the number of tokens of KV you can hold *is* your maximum concurrency, which *is* your throughput ceiling for a memory-bandwidth-bound decode. [4] During decode the step time is dominated by *reading* weights + KV from HBM (§4), so both the *capacity* and the *speed* of the system are governed by KV cache, not by FLOPs.

### GQA's effect

Grouped-Query Attention shares each KV head across a group of query heads, so `n_kv_heads < n_heads`. The KV cache shrinks by exactly the ratio `n_heads / n_kv_heads`. For Llama-3-70B (64 query heads, 8 KV heads, 80 layers, head_dim 128, bf16): `2·80·8·128·2 = 327,680 bytes ≈ 320 KiB/token` — an **8× reduction** vs the multi-head equivalent (2.5 MB/token → 0.31 MB/token). [6][7] MQA (`n_kv_heads = 1`) is the extreme case.

### Worked numbers for Qwen2.5-0.5B (your model)

Config: `n_layers = 24`, `n_heads = 14`, `n_kv_heads = 2`, `head_dim = 64`, bf16 (`dtype_bytes = 2`). [8]

```
kv_bytes_per_token = 2 * 24 * 2 * 64 * 2 = 12,288 bytes = 12 KiB / token
```

- **Per 1,000-token sequence:** ≈ 12.3 MB. **Per 2,048-token context:** ≈ 25.2 MB.
- **GQA saving on this model:** with MHA it would be `2·24·14·64·2 = 86,016 B ≈ 84 KiB/token`; GQA (2 vs 14 KV heads) is a **7× reduction** (`14/2`). [6][8]
- **Weights:** ≈ 0.49 B params × 2 bytes ≈ **~1.0 GB** in bf16. [8]

**The load-bearing implication for your ablations.** On, say, a 24 GB GPU: weights ~1 GB leaves ~20 GB for KV ⇒ ~20 GB / 12 KiB ≈ **1.6 million tokens** of KV headroom. You will *not* naturally hit the KV-capacity wall that PagedAttention was built to demolish. To reproduce the paper's paging win you must **shrink the KV budget on purpose** (cap `gpu_memory_utilization`, or reserve only a few hundred MB for KV) and/or use long contexts, so that fragmentation actually costs you batch slots. Without that pressure, expect PagedAttention to be near-noise on a 0.5B model — a legitimate result to report (§11).

---

## 4. Prefill vs decode — the two regimes

**Prefill** processes *all* prompt tokens in **one** forward pass. That's a big matmul with **high arithmetic intensity** → **compute-bound**, and it saturates the GPU. It produces the KV cache for the prompt and the first output token; its latency is essentially **TTFT**. [9]

**Decode** processes **one token per request** per step. The matmuls are tall-skinny (batch×1), so almost no FLOPs are done per byte of weights+KV read from HBM → **memory-bandwidth-bound**, low Model-FLOPs-Utilization. [9] Batching many requests is what raises decode arithmetic intensity toward the compute-bound regime — which is *why* continuous batching (§1) matters so much: it is the lever that pushes decode off the memory-bound floor.

**Prefill–decode interference.** In a naive continuous batcher, when a long prompt arrives it runs a big prefill in the same iteration as everyone else's decodes. The decodes **stall** behind the prefill, causing a spike in inter-token latency (a "generation stall"). So you face a tension: admit prefills eagerly (good TTFT, bad TPOT jitter) vs. protect decodes (smooth TPOT, worse TTFT). [9]

**Chunked prefill (Sarathi / Sarathi-Serve).** Split a long prefill into fixed-size **chunks** and spread it over several iterations, and in each iteration **piggyback ongoing decodes with a prefill chunk** ("stall-free batching"): pack all running decodes first, then fill the remaining token budget with a prefill chunk, chunking the last prefill if it doesn't fit. [9] This caps how much a prefill can delay any single decode step, so TPOT stays bounded while the GPU stays busy. Reported result: **up to 2.6× (Mistral-7B, 1×A100), 3.7–4.3× (vs vLLM) and higher vs Orca** more serving capacity under strict latency SLOs; chunk overhead is modest (**≤~25%** even at the smallest 512-token budget). [9] Token budgets used: **512 (strict SLO), 2,048 (relaxed)**. [9]

**TTFT/TPOT tradeoff and "goodput."** The two user-facing latencies pull against each other: bigger batches and eager prefills raise throughput but hurt one of TTFT/TPOT. The metric that captures "throughput that's actually usable" is **goodput** — requests/sec that *meet* their latency SLOs (e.g., p99 TTFT < X *and* p99 TPOT/TBT < Y). Raw throughput can look great while goodput collapses because tails blew the SLO. [9][10][12] Design and tune to goodput, not raw tok/s.

---

## 5. Scheduling policies, admission control, preemption

Every iteration, a continuous batcher runs an **admission decision**: given the running set and the waiting queue, which requests are in the batch this step? Two budgets bound it: [11]

- **`max_num_seqs`** — max number of sequences (requests) in one iteration's batch. Caps concurrency directly.
- **`max_num_batched_tokens`** — max total tokens processed in one iteration (prefill tokens + one per decode). This is what chunked prefill spends against; with chunked prefill on, vLLM has historically defaulted this to a small value like **512**, and prioritizes decodes: "batch all pending decode requests before scheduling any prefill," then spend remaining budget on prefill, chunking the last one if it won't fit. [9][11]

Other knobs and policies:

- **`gpu_memory_utilization`** (default ~0.9) — fraction of GPU memory vLLM claims; everything not held by weights becomes the KV block pool. *This is your primary lever for creating KV pressure in ablations.* [11]
- **Ordering policy** — default **FCFS**; vLLM also supports **priority** scheduling. FCFS is simplest and fine for a learning build; priority matters when you have SLO classes. [11]
- **Backpressure / admission control** — if you admit more than KV can hold, you must either queue (backpressure) or preempt. A real server refuses/queues new work when the KV pool is full rather than OOM-ing.

**Preemption / eviction — recompute vs swap.** When KV runs out mid-flight, the scheduler **preempts** a running sequence to free its blocks for others, and resumes it later. Two recovery modes: [4][11]

- **Recompute** — drop the victim's KV, and when readmitted, re-run prefill over its already-generated tokens to rebuild KV. Cheap in memory, costs compute; usually the default.
- **Swap** — copy the victim's KV blocks out to CPU RAM and back later. Saves the recompute FLOPs but costs PCIe bandwidth and CPU memory. vLLM's block structure makes both clean (whole blocks move/rebuild). [4]

**Tuning intuition:** frequent preemption is a symptom of over-admission — *lower* `max_num_seqs` / `max_num_batched_tokens` (or raise the KV budget) to reduce it. [11] For your build, a simple FCFS + recompute preemptor is enough to demonstrate the mechanism.

---

## 6. Prefix caching (automatic prefix caching, APC)

APC persists KV blocks across *requests*: hash each block over `(parent_block_hash, token_ids, extra_keys)` so a cache hit requires an **exact token-by-token prefix match**; on a new request, reuse the physical blocks for the matched prefix and **skip prefill for those tokens**. [13][14] It's the cross-request generalization of PagedAttention's copy-on-write prefix sharing (§2).

**When it helps a lot:** many requests share a long common prefix — a 2K-token system prompt across all chat users, shared few-shot exemplars, multi-turn conversations where the history is re-sent each turn. Reported TTFT reductions of **60–80%** for chat apps with long shared system prompts, and SGLang's RadixAttention (a tree-structured prefix cache) reports **75–95% cache hit rates** on multi-turn workloads. [13][15] The win is almost entirely **TTFT/prefill compute**, not decode throughput.

**When it's negligible:** unique prompts with no shared prefix (the hash never hits); or a workload already dominated by decode where prefill was a small fraction of the time. For a single-model learning benchmark with random prompts, expect ~0 benefit — only worth building/measuring if you deliberately construct a shared-prefix workload. Note the privacy angle: cache sharing can leak "was this prefix seen before" via timing, which vLLM mitigates with an optional per-request **cache salt**. [13]

---

## 7. Attention kernels — the crux of a from-scratch build

Everything except attention (QKV/MLP/output projections) batches trivially: concatenate all tokens in the step and do one big matmul. **Attention is the hard part** because each sequence attends over its *own* KV history of a *different* length. Three kernel ideas matter:

- **FlashAttention** — computes attention in tiles with an **online softmax** (running max + running denominator), never materializing the full `S×S` score matrix. This makes it IO-aware (fewer HBM round-trips) and memory-linear in sequence length. It's the standard prefill kernel. [16]
- **FlashDecoding** — FlashAttention specialized for decode (query length 1, long KV): it **parallelizes across the KV sequence dimension** (splits KV into chunks processed in parallel, then combines partial softmaxes), exposing parallelism that a single query token otherwise lacks and keeping the GPU busy at small batch. [16]
- **PagedAttention kernel** — the attention kernel must read K/V from **non-contiguous blocks**. It takes the **block table** as input and, for each logical position, gathers the right physical block before computing scores. This block-table-driven gather is the single piece of machinery that makes paged KV work; FlashAttention-2, FlashInfer, and others now support paged KV directly (FlashInfer uses a compressed block-table format). [4][5][16][17]

**What pure-PyTorch implementers actually do.** Without custom CUDA, the common path is **masked batched attention with padding**: pad all sequences in the batch to the max length, stack KV into a dense `(B, H, S_max, d)` tensor, compute `softmax(QKᵀ/√d + mask)·V` with `torch.nn.functional.scaled_dot_product_attention` (which dispatches to a FlashAttention backend when eligible), and use an **additive −inf mask** to (a) enforce causality and (b) null out padding positions. [16]

- **Cost of padding:** you compute (and store) attention for padded positions you throw away; waste ≈ `(S_max − S_mean)/S_max`. With skewed lengths this is large — exactly the memory/compute waste PagedAttention eliminates. This is the honest reason a naive PyTorch server can't match vLLM, and it's the perfect thing to *measure* in your ablation.
- **Practical from-scratch recipe:** (1) naive = loop over requests, per-request SDPA; (2) static batch = pad + masked batched SDPA; (3) continuous batch = keep a persistent per-sequence KV list, re-pad each iteration, admit/evict between steps; (4) paged = replace the padded dense KV with a block pool + block-table gather (you can prototype the gather in pure PyTorch with `index_select`/`gather` before worrying about a fused kernel). Steps 1–3 need no custom CUDA; step 4's kernel is where "real" systems spend their effort.

---

## 8. Beyond scope (know they exist)

- **Speculative decoding.** A cheap **draft** proposes *k* tokens; the target model **verifies** them in one batched forward pass and accepts the longest correct prefix. Output is *identical* to the target (lossless). Speedups scale with acceptance rate: draft-model specdec ~2×; **Medusa** (extra decoding heads on the target, no separate draft) ~2× with 60–80% acceptance; **EAGLE/EAGLE-3** (predict the target's penultimate-layer *features*, not tokens) reach **0.8–0.9 acceptance → ~3–4× on H100**. [18] Helps **latency** at low-to-moderate batch; the benefit shrinks as batch grows (the GPU is already compute-saturated). Out of scope for the core arc but a natural later ablation on a small model.
- **Disaggregated prefill/decode (DistServe, Mooncake).** Run prefill and decode on **separate** GPU pools so compute-bound prefill can't interfere with memory-bound decode, transferring KV between them. DistServe optimizes goodput under per-phase SLOs; Mooncake (Moonshot/Kimi) adds a **KVCache-centric** scheduler using CPU/DRAM/SSD as a distributed KV store, reporting **59–498%** higher effective capacity under SLOs on real traces. [19][20] Strictly a multi-GPU/cluster technique — irrelevant to a single-GPU learning build, but the *reason* it exists (prefill/decode interference, §4) is the same tension you'll manage with chunked prefill.

---

## 9. Metrics and load generation

**Latency metrics (per request):**
- **TTFT (Time To First Token)** — arrival → first output token. Dominated by queue wait + prefill. [9][10]
- **TPOT / ITL / TBT (Time Per Output Token / Inter-Token Latency / Time Between Tokens)** — average interval between consecutive output tokens during decode. Governs "typing speed." [9][10]
- **End-to-end latency** ≈ `TTFT + (n_output − 1) × TPOT`.
- **Tails:** report **p50/p90/p99**, not just mean — serving quality lives in the tail, and SLOs are usually stated on p99. [10][12]

**Throughput metrics (system):** output **tokens/sec** and **requests/sec** (and total tok/s incl. prompt). **GPU utilization / MFU** — how close decode gets to compute-bound (usually low, §4).

**Goodput** — requests/sec that *satisfy* the SLO (e.g., p99 TTFT and p99 TPOT both under threshold). The metric that actually matters; raw throughput without an SLO is misleading because a saturated system posts high tok/s while its tails explode. [9][10][12]

**How load is generated — open-loop Poisson.** Serving benchmarks (including vLLM's harness) drive requests as an **open-loop** process with **Poisson arrivals** at a target rate λ (req/s), independent of whether the server has kept up. [12] This is essential: a **closed-loop** generator (send next request only after the previous finishes) *self-throttles* under overload, so an overloaded system automatically receives less load and its tail latency looks deceptively healthy. [12] Sweep λ upward and plot p50/p99 TTFT and TPOT vs load; the "knee" where tails blow up is your capacity, and goodput-at-SLO is read off that curve. Standardize on a fixed prompt/output-length distribution (e.g., 512 in, Exp(128) out) so runs are comparable. [3][12]

---

## 10. How the major systems differ (state of the art)

**vLLM.** The reference open-source engine and origin of **PagedAttention** (§2). Broad model/hardware coverage, continuous batching, APC, chunked prefill, speculative decoding, tensor/pipeline parallelism; the de-facto baseline everyone benchmarks against and the codebase to read when building your own. [3][4][21]

**HuggingFace TGI (Text Generation Inference).** Early production continuous-batching server (Rust router + Python/CUDA workers), tightly integrated with the HF ecosystem. As of 2025 it is effectively in **maintenance mode**; HF itself points users toward vLLM or SGLang. Historically a solid ~8×-class continuous-batching engine in the Anyscale comparison. [3][21]

**NVIDIA TensorRT-LLM.** Compiles models into optimized **TensorRT engines** with fused kernels, in-flight (continuous) batching, paged KV, FP8/INT4 quantization, and tight Hopper/Blackwell tuning. Typically the **lowest p50/p95 latency** and best single-GPU efficiency on NVIDIA hardware, at the cost of an ahead-of-time compile step and less flexibility. [21][22]

**SGLang.** Adds **RadixAttention** — a radix-tree prefix cache that maximizes cross-request KV reuse — plus a front-end language for structured/multi-step programs. Wins hardest on workloads with heavy prefix sharing (agents, multi-turn, RAG), reporting **up to ~6.4× throughput / ~3.7× lower latency** on structured workloads and ~**29%** over vLLM on shared-context small-model benchmarks. [15][21][22] (LMDeploy and DeepSpeed-MII/FastGen are further alternatives; LMDeploy's TurboMind is another high-throughput paged engine, and DeepSpeed-FastGen's "dynamic SplitFuse" is a chunked-prefill-style scheduler.)

**"State of the art" as of 2026** = continuous batching + paged KV as *table stakes*, plus chunked prefill, automatic prefix caching, quantization (FP8/INT4), speculative decoding, and — at cluster scale — prefill/decode disaggregation with a KV-cache-aware scheduler. vLLM and SGLang are the two most-recommended open engines; TensorRT-LLM leads latency on NVIDIA silicon; TGI is legacy. [21][22]

---

## 11. What actually makes an inference server fast — ranked

Ordered by typical impact, with a note on how much each is likely to matter **on a 0.5B model / single-GPU (or CPU) learning setup**.

1. **Continuous (iteration-level) batching.** *The* highest-leverage change. Removes GPU idle time from finished/short sequences; **~2–5× over static batching**, more with high output-length variance. [1][3] **On your setup:** big win *if* you drive it with variable-length outputs and real arrivals. Build this first and second (static → continuous) and measure the delta — this is your headline ablation.
2. **Paged KV cache (PagedAttention).** Reclaims the **60–80%** KV memory that fragmentation wastes → larger batches → the ~8×→~23× jump over plain continuous batching in the Anyscale table. [3][4] **On your setup:** only visible under deliberate KV pressure (cap `gpu_memory_utilization`, long contexts, high concurrency); otherwise near-noise for a 0.5B model (§3). Reporting *"paging didn't help until I constrained memory"* is a correct and instructive result.
3. **Chunked prefill + decode-priority scheduling.** Kills prefill–decode interference, protecting TPOT tails and thus **goodput** (up to ~2.6×+ capacity under strict SLOs). [9] **On your setup:** matters once you send long prompts mixed with ongoing decodes; the effect shows up in *tail* TPOT, not mean throughput.
4. **Quantization (weights/KV, FP8/INT4).** Not in your core arc, but often a large real-world lever (more KV headroom + faster memory-bound decode). Mentioned for completeness.
5. **Automatic prefix caching.** Large **TTFT** win *only* with shared prefixes (system prompts, multi-turn); ~0 otherwise. [13][15] **On your setup:** build a shared-prefix workload to see it, else skip.
6. **Attention kernel quality (FlashAttention/FlashDecoding, paged gather).** Determines the constant factor and how much padding waste you carry. [16] **On your setup:** in pure PyTorch, use `scaled_dot_product_attention` (Flash backend) and an additive mask; the padding waste you measure *is* the motivation for step 4 (paging).
7. **Block size, preemption mode, FCFS-vs-priority, swap-vs-recompute.** Real but **second-order** knobs (single-digit %). [4][11] Tune last; don't expect them to move the headline.
8. **Speculative decoding / disaggregation.** Latency and cluster-scale levers respectively; **out of scope** for the core build, valuable as "know they exist." [18][19][20]

**Bottom line for the build.** Implement `naive → static → continuous → paged`, benchmark **open-loop with Poisson arrivals and skewed output lengths**, and report **goodput at an SLO** alongside raw tok/s and p50/p99 TTFT/TPOT. Expect the **continuous-batching step to dominate** your measured speedup, **paging to require engineered memory pressure** to shine on a 0.5B model, and the small knobs to be noise. Designing the *workload* to expose each optimization is as important as implementing the optimization.

---

## Sources

1. Orca: A Distributed Serving System for Transformer-Based Generative Models (Yu et al., OSDI 2022) — USENIX: https://www.usenix.org/conference/osdi22/presentation/yu
2. Iteration Batching (Continuous Batching) explainer — FriendliAI: https://friendli.ai/blog/llm-iteration-batching
3. How Continuous Batching Enables 23x Throughput... (benchmark table: static/FT/TGI/vLLM) — Anyscale: https://www.anyscale.com/blog/continuous-batching-llm-inference
4. Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP 2023) — arXiv: https://arxiv.org/pdf/2309.06180
5. PagedAttention / block table / block size — vLLM automatic prefix caching design & implementation docs: https://docs.vllm.ai/en/stable/design/prefix_caching/ and https://docs.vllm.ai/en/v0.6.1/automatic_prefix_caching/details.html
6. KV cache size formula and GQA reduction (Llama-3-70B worked example) — Spheron KV-cache optimization guide: https://www.spheron.network/blog/kv-cache-optimization-guide/
7. From Tokens to Throughput: KV Caching in LLMs — Medium (R. Ekstein): https://medium.com/@rjekstein/from-tokens-to-throughput-the-power-of-key-value-caching-in-llms-8b84d97e6b17
8. Qwen2.5-0.5B model config (n_layers=24, n_heads=14, n_kv_heads=2, head_dim=64, bf16) — HuggingFace: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
9. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (Agrawal et al., OSDI 2024) — arXiv HTML: https://arxiv.org/html/2403.02310 and USENIX PDF: https://www.usenix.org/system/files/osdi24-agrawal.pdf
10. LLM Inference SLO Engineering: TTFT, ITL, P99 budgets — Spheron: https://www.spheron.network/blog/llm-inference-slo-ttft-itl-latency-budget-guide-2026/
11. vLLM scheduler knobs (max_num_seqs, max_num_batched_tokens, gpu_memory_utilization, preemption recompute/swap, chunked-prefill decode-priority policy) — vLLM Optimization & Scheduler docs: https://docs.vllm.ai/en/stable/configuration/optimization/ and https://audreywongkg.medium.com/understanding-vllm-scheduling-token-budgets-chunked-prefill-and-policies-2c879e3980e3
12. Open-loop vs closed-loop load generation, Poisson arrivals, goodput methodology — "Human-less LLM Serving" (arXiv 2606.20577): https://arxiv.org/html/2606.20577v1
13. Automatic Prefix Caching (hash-based block reuse, cache salt, TTFT impact) — vLLM design docs: https://docs.vllm.ai/en/stable/design/prefix_caching/
14. Automatic Prefix Caching implementation details — vLLM: https://docs.vllm.ai/en/v0.6.1/automatic_prefix_caching/details.html
15. SGLang / RadixAttention throughput & cache-hit numbers — Spheron engine comparison: https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/
16. FlashAttention, FlashDecoding, paged KV kernels — "Attention kernels for LLM inference" (fergus): https://fergus.hashnode.dev/attention-kernels-for-llm-inference
17. FlashInfer attention kernels / compressed block tables: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
18. Speculative decoding — EAGLE/EAGLE-3 & Medusa acceptance rates and speedups — Spheron EAGLE-3 guide: https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/ and NVIDIA intro: https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/
19. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving (OSDI 2024) — USENIX: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
20. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving — arXiv: https://arxiv.org/abs/2407.00079
21. Comparing the Top Inference Runtimes for LLM Serving (2025), incl. TGI maintenance status — MarkTechPost: https://www.marktechpost.com/2025/11/07/comparing-the-top-6-inference-runtimes-for-llm-serving-in-2025/
22. vLLM vs TensorRT-LLM vs SGLang H100 benchmarks (latency/throughput) — Spheron: https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/

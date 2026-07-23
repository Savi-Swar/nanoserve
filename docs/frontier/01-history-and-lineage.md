# The History and Lineage of LLM Inference Serving

*A narrative of ideas — how the field got from "just call `model.generate()` in a loop" to the systems that run production LLMs today.*

> **Who this is for.** You are building an inference server from scratch. Before you write a scheduler or a KV-cache allocator, it helps to know *why* the canonical designs look the way they do — because almost every one is a direct response to a specific, physical property of autoregressive Transformer decode. This document traces the intellectual lineage: the core problem, the systems that attacked it, the key insight of each, and the mental models practitioners actually reason with.

---

## 1. The core problem: why naive serving wastes the GPU

An autoregressive LLM generates one token at a time. To produce a sequence of length *N*, you run the model forward *N* times, feeding each output token back in as the next input. This single fact drives everything downstream.

Decomposed, a request has two phases with opposite physical character:

- **Prefill** processes the entire prompt in one forward pass. All prompt tokens are known up front, so they go through the network *in parallel* as one big matrix-multiply. Each weight matrix is loaded from memory once and reused across many tokens, so the ratio of arithmetic to memory traffic — the **arithmetic intensity** — is high. Prefill is **compute-bound**: it saturates the GPU's tensor cores.
- **Decode** generates one token per forward pass. To produce that single token you must stream the *entire* model's weights (and the growing KV cache) from GPU memory, then do a tiny amount of arithmetic on them. Arithmetic intensity is roughly **1**. Decode is **memory-bandwidth-bound**: the tensor cores sit mostly idle while the GPU waits on HBM (High-Bandwidth Memory).

This is the crux. On a modern GPU the tensor cores can do hundreds of TFLOP/s, but HBM bandwidth is a few TB/s. The **roofline model** says an operation is limited by whichever ceiling it hits first, and the crossover ("ridge point") sits at an arithmetic intensity of a few hundred FLOP/byte. Prefill lives above the ridge (compute-bound); decode lives far below it (bandwidth-bound). Empirically, dense single-request decode can leave the *vast majority* of both compute and memory bandwidth idle — one 2025 study measured 96–97% of HBM bandwidth sitting idle during long-context prefill on an A100, and the symmetric waste story holds for compute during decode.

**Why naive serving wastes the GPU:** if you serve one request at a time, every decode step drags the full weight matrix across the memory bus to compute one token's worth of math. The tensor cores are starved. The only way to amortize that weight-loading cost is to **batch** many requests so that each loaded weight is reused across many sequences' tokens in the same pass. Batching is not a nice-to-have; for decode it is the *entire* efficiency story. The history of LLM serving is largely the history of learning to batch decode well despite two obstacles: (1) requests arrive and finish at different times and have wildly different lengths, and (2) the KV cache — the per-request memory that makes decode cheap — is huge, grows unpredictably, and is hard to pack into fixed GPU memory.

### The KV cache: the real bottleneck

Naively, generating token *t* requires attending to all previous tokens, which would mean re-encoding the whole sequence every step — O(*N*²) work. The **KV cache** avoids this: the key and value vectors for every past token are computed once and stored, so each new token only computes *its own* K/V and attends against the cache. This turns decode from quadratic to linear, and it is why decode is fast per step.

But the cache is enormous and dynamic. Its size is `2 × layers × heads × head_dim × seq_len × batch × dtype_bytes`, and it *grows by one token's worth every single step*, for every concurrent request, until that request finishes at an unknown length. This is why practitioners say **"the KV cache is the real bottleneck."** Available GPU memory for KV cache — not FLOPs — usually caps how many requests you can batch, and therefore caps throughput. Managing this memory well is the central systems problem, and it is exactly what vLLM would later attack.

---

## 2. Pre-LLM batching: FasterTransformer and Triton dynamic batching (≈2019–2022)

Before LLMs dominated, the serving world already knew batching mattered — but for the *wrong shape* of workload.

**NVIDIA Triton Inference Server** (formerly TensorRT Inference Server, ~2019 onward) introduced **dynamic batching**: the server transparently groups individual client requests that arrive close in time into one larger batch, runs them together, and demultiplexes the responses. The client sends one request and gets one response, unaware batching happened. For classic models — a ResNet image classifier, a BERT encoder — this is nearly optimal: every request does the *same fixed amount of work* (one forward pass), so requests in a batch start and finish together.

**NVIDIA FasterTransformer** (open-sourced ~2019, matured through 2021–2022) supplied the highly optimized GPU kernels — kernel fusion, FP16, multi-GPU tensor/pipeline parallelism — that made Transformer forward passes fast. It became the de facto high-performance backend and the standard *baseline* that every later LLM-serving paper measured against.

**Why this was insufficient for generative LLMs.** Dynamic batching batches at the granularity of a *whole request*. That assumption breaks catastrophically for autoregressive generation:

- **Requests finish at different times.** In a batch, one request may want 20 output tokens and another 500. With request-level batching the whole batch is locked together for the length of the *longest* member. The short request finished long ago but its slot can't be released and no new request can join — the GPU runs a shrinking, increasingly empty batch. This is called **static batching**, and its GPU utilization collapses when output lengths vary.
- **Prefill and decode have different shapes.** A request in prefill processes many tokens at once; a request in decode processes one. You cannot naively stack a prefill request and a decode request into a single batched matrix-multiply because their tensor shapes don't line up.

The measured cost of these limitations was stark: static batching versus continuous batching on Llama-13B (A100) differed by **~23×** in throughput in Anyscale's widely cited benchmark (2023). Request-level batching left most of that on the table.

---

## 3. Orca (OSDI 2022): the foundational idea

**Paper:** *Orca: A Distributed Serving System for Transformer-Based Generative Models.* Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. USENIX OSDI 2022 (published July 2022). Seoul National University; the work seeded the company FriendliAI.

Orca is the origin point of modern LLM serving. It introduced two techniques that, together, break the request-level-batching straitjacket.

### 3.1 Iteration-level scheduling (a.k.a. continuous batching / iteration batching)

**The insight:** stop scheduling at the granularity of a *request*; schedule at the granularity of an *iteration* (one forward pass = one generated token for the batch).

Mechanically: instead of picking a batch, running it to completion, and only then touching the queue again, Orca's scheduler invokes the execution engine to run **exactly one iteration** of the model on the current batch, then **returns control to the scheduler**. At that boundary the scheduler can:

- **Evict** any request that just emitted its end-of-sequence token (return it to the client, free its slot immediately), and
- **Admit** a newly arrived request into the freed slot for the *very next* iteration.

The batch is thus *continuously* refreshed. A short request leaves the moment it's done; a waiting request enters the moment there's room. No sequence waits for the longest member of its batch. This is why the technique is variously called **iteration-level scheduling**, **continuous batching**, or **iteration batching** — all the same idea. It attacks obstacle (1) from §1: heterogeneous arrival and completion times.

### 3.2 Selective batching

Iteration-level scheduling creates a new problem: at any given iteration the batch is a *mix* — some requests are in prefill (many tokens), some in decode (one token), and they have different sequence lengths and cached states. A single fused batched operation can't handle that heterogeneity uniformly.

**The insight:** don't batch every operation the same way — batch *selectively*.

- Operations that are token-wise independent and shape-uniform — the big **matrix multiplications** in the feed-forward and QKV projections — *are* batched: flatten all tokens from all requests in the batch into one tall matrix and do one big GEMM. This is where the GPU efficiency comes from.
- The **attention** operation, which must respect each request's own sequence boundaries and its own KV cache, is *not* naively batched across requests. Orca splits the batch and processes attention per-request (or per-group), then re-merges for the next batched operation.

Selective batching is what makes iteration-level scheduling *implementable* on a real Transformer. It attacks obstacle (2)'s cousin: mixing prefill and decode, and mixing sequences of different lengths, in one batch.

**Why it's foundational:** Orca established the scheduling architecture that essentially every subsequent system inherited. When you read "continuous batching" in any 2023+ system's docs, you are reading Orca's iteration-level scheduling. It reported large throughput and latency gains over FasterTransformer, but its lasting contribution is conceptual, not a number.

---

## 4. vLLM + PagedAttention (SOSP 2023): the watershed

**Paper:** *Efficient Memory Management for Large Language Model Serving with PagedAttention.* Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. ACM SOSP 2023. arXiv:2309.06180 (submitted September 12, 2023). UC Berkeley (Sky Computing Lab) and collaborators.

Orca solved *scheduling*. But it left the other half of §1 open: **KV-cache memory management**. Pre-vLLM systems (including Orca) stored each request's KV cache in a **single contiguous** chunk of GPU memory, pre-reserved for the request's *maximum possible* length. This wastes memory three ways:

1. **Internal fragmentation** — you reserve for max length (say 2048) but the request only generates 30 tokens; the rest is reserved-but-unused.
2. **External fragmentation** — variable-size contiguous reservations leave unusable gaps between allocations.
3. **Over-reservation** — memory reserved for future tokens that don't exist yet can't be used by anyone else *now*.

The vLLM authors measured that existing systems wasted **60–80%** of KV-cache memory to these effects. And since KV-cache memory caps batch size (§1), wasted memory means small batches means low throughput. The bottleneck was memory management, not the model.

### 4.1 PagedAttention: the OS-paging analogy

**The insight — and it is one of the great analogies in ML systems:** the KV-cache problem is *exactly* the problem operating systems solved decades ago with **virtual memory and paging**.

The mapping:

| Operating system | PagedAttention |
|---|---|
| Process address space | A request's KV cache |
| Page (fixed-size) | KV **block** (fixed number of tokens, e.g. 16) |
| Physical memory frame | A physical GPU-memory block |
| Page table | Per-request **block table** mapping logical → physical blocks |
| Demand paging (allocate on use) | Allocate a new block only when the current one fills |

Concretely: split each request's KV cache into **fixed-size blocks** holding a handful of tokens each. Blocks need **not** be contiguous in physical memory — a per-request block table records where each logical block physically lives. As a request generates tokens, vLLM hands it a new physical block only when needed (demand allocation), from a shared pool. When a request finishes, its blocks return to the pool for anyone.

The consequences fall out of the analogy for free:

- **Near-zero fragmentation.** Because blocks are small and fixed-size, the only waste is at most one partially-filled block per request (internal fragmentation bounded by block size). External fragmentation vanishes.
- **Bigger batches → higher throughput.** Reclaimed memory becomes more concurrent requests.
- **Copy-on-write sharing.** Two requests with a shared prefix (a common system prompt, or parallel samples from one prompt) can *point their block tables at the same physical blocks*, and only fork a block when one diverges — just like COW pages after `fork()`. This makes parallel sampling and beam search cheap.

### 4.2 Why it was a watershed

vLLM reported **2–4× throughput** over the state of the art — FasterTransformer *and* Orca — at the same latency, purely from better memory management. But the deeper reasons it became the reference implementation:

- It combined Orca's continuous batching **and** paged memory into one coherent, **open-source** system that anyone could run.
- PagedAttention became an **industry norm** within roughly a year: adopted or reimplemented in Hugging Face TGI, NVIDIA TensorRT-LLM (as "in-flight batching"), and beyond.
- It gave the field a shared *vocabulary and mental model* — "treat the KV cache like paged virtual memory" — that reframed how everyone thought about the problem.

If Orca defined the scheduler, vLLM defined the memory allocator. Together they are the two load-bearing pillars every from-scratch server rebuilds.

---

## 5. The orthogonal thread that matters: FlashAttention (2022)

**Paper:** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. arXiv:2205.14135 (May 2022); NeurIPS 2022.

FlashAttention is not a *serving* system — it's a *kernel* — but it is inseparable from this history because it repaired the attention operation itself, and it taught the field to think in terms of the **memory hierarchy**.

**The insight:** attention was being computed the obvious, wrong way. The standard implementation materializes the full *N×N* attention-score matrix in HBM, then reads it back to apply softmax and multiply by V. For long sequences that matrix is huge, and writing/reading it dominates the runtime — attention was **IO-bound on HBM traffic**, not compute-bound, even though everyone treated it as a math problem.

FlashAttention makes attention **IO-aware**. Using **tiling**, it loads blocks of Q, K, V from slow HBM into fast on-chip **SRAM**, computes the attention for that tile entirely in SRAM using an **online softmax** (running max/sum so you never need the whole row at once), and writes only the final output back — **never materializing the N×N matrix in HBM at all**. Same exact result, dramatically less memory traffic: ~7.6× speedup on the attention op, and it made *long* contexts practical.

**Why it belongs in this lineage:**

- **The mental model it cemented.** FlashAttention made "HBM vs SRAM, count the bytes moved, not the FLOPs" the default way systems people reason about GPU kernels — the same roofline/IO-awareness thinking that explains why decode is bandwidth-bound (§1). This lens is now foundational.
- **It's in the stack.** Production servers layer FlashAttention (and its successors) *underneath* PagedAttention — vLLM and TGI use flash-style attention kernels for the batched attention step. (Follow-ups: **FlashAttention-2**, July 2023, better GPU work-partitioning; **FlashAttention-3**, July 2024, tuned for Hopper.)

Prefill efficiency, long-context feasibility, and the whole field's kernel-level vocabulary trace back here.

---

## 6. The shift from throughput to goodput: SLO thinking (2024)

Through 2023 the implicit objective was **throughput** — tokens per second, requests per second. But throughput is the wrong target for interactive serving, and 2024 was the year the field said so out loud.

A user of a chat product experiences two distinct latencies:

- **TTFT (Time To First Token)** — how long until the answer *starts*. Dominated by prefill. A slow TTFT feels like a hang.
- **TPOT / ITL (Time Per Output Token / Inter-Token Latency)** — how *smoothly* the answer streams. Dominated by decode. A slow TPOT feels like typing that stutters.

A server can have magnificent aggregate throughput while violating both — e.g. by cramming huge batches that make every individual request slow. Hence **goodput**: the request rate a system sustains *while meeting its latency SLOs* (say, "TTFT < 500 ms and TPOT < 50 ms for 90% of requests"). Goodput, not raw throughput, is what actually matters, and optimizing for it exposed a tension that Orca-style co-located batching had buried: **prefill and decode interfere with each other**.

### 6.1 The prefill–decode interference problem

In a vanilla continuous-batching system, when a big prefill lands in a batch alongside ongoing decodes, the prefill's heavy compute **stalls** every decode in that iteration — every streaming user hitches while one new user's prompt is processed. Throughput-optimal, SLO-terrible. Two families of solutions emerged, both in 2024:

### 6.2 Sarathi-Serve — chunked prefill + stall-free batching (OSDI 2024)

**Paper:** *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.* Amey Agrawal et al. arXiv:2403.02310; USENIX OSDI 2024. Microsoft Research India / Georgia Tech.

**The insight:** don't let a big prefill monopolize an iteration — **chop it up**. **Chunked prefill** splits a long prompt into several near-equal-size chunks processed across successive iterations. Then, using **stall-free batching**, each iteration co-schedules a prefill *chunk* with the ongoing decodes, sized so the iteration's total compute stays bounded. Crucially, it exploits the **arithmetic-intensity slack** in decode: decode iterations barely use the tensor cores (§1), so you can "hide" a prefill chunk's compute in that idle capacity *without slowing the decodes down*. Decodes never stall; prefills still make progress. Reported up to **2.6×** (Mistral-7B, one A100) and **6.9×** (Falcon-180B, 8×A100) throughput within SLO versus Orca/vLLM. This keeps prefill and decode *on the same GPUs* but interleaves them intelligently.

### 6.3 DistServe & Splitwise — disaggregation (OSDI/ISCA 2024)

**Papers:** *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (Yinmin Zhong et al., arXiv:2401.09670, OSDI 2024); *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (Pratyush Patel, Esha Choukse, et al., Microsoft, ISCA 2024).

**The opposite insight:** if prefill and decode interfere and have *different hardware appetites*, stop running them on the same machines at all. **Disaggregate** them onto separate GPU pools — a prefill cluster and a decode cluster — and ship the KV cache from prefill nodes to decode nodes over fast interconnect. Now each phase is scheduled, batched, and even *hardware-matched* independently: decode (bandwidth-bound, low compute) can run on cheaper/lower-power GPUs while prefill (compute-bound) gets the beefy ones. DistServe reported serving **7.4× more requests** or meeting **12.6× tighter SLO**; Splitwise showed up to **1.4× throughput at 20% lower cost**, and framed it explicitly as a cost/power co-design problem.

Chunked-prefill (share the GPU, interleave finely) vs. disaggregation (split the GPUs, specialize each) is a genuine, still-live design fork — and which wins depends on your workload and cluster. Both exist *because* the field switched its objective from throughput to goodput.

---

## 7. The other 2024 thread: reuse the KV cache across requests

vLLM freed *unused* KV memory; a parallel idea is to *reuse already-computed* KV across requests that share a prefix.

**SGLang / RadixAttention.** Lianmin Zheng, Ying Sheng, et al. (LMSYS/Stanford/Berkeley). arXiv:2312.07104 (Dec 2023); popularized in the LMSYS blog, January 17, 2024. **The insight:** shared prefixes are everywhere — a common system prompt, few-shot examples, a multi-turn chat history, tree-of-thought branches. Instead of recomputing that prefix's KV for every request, keep an **LRU cache of KV blocks indexed by a radix tree** over token sequences. When a request arrives, walk the tree to find the **longest matching cached prefix**, reuse its KV directly, and only compute the novel suffix. This cuts prefill work and TTFT dramatically on prefix-heavy workloads. It generalizes vLLM's copy-on-write prefix sharing from "within a request family" to "automatically, across all requests over time." Prefix caching is now standard in vLLM, SGLang, TensorRT-LLM, and others.

---

## 8. Timeline

| Year | System / Paper | Venue | Key contribution | Why it mattered |
|---|---|---|---|---|
| ~2019–21 | Triton Inference Server; FasterTransformer | NVIDIA | Dynamic (request-level) batching; optimized fused Transformer kernels | The standard pre-LLM serving stack and the baseline everyone measured against — but request-level batching breaks on variable-length generation |
| May 2022 | **FlashAttention** (Dao et al.) | arXiv / NeurIPS 2022 | IO-aware, tiled, exact attention; never materializes N×N in HBM | Fixed attention's memory-traffic bottleneck; cemented HBM/SRAM roofline thinking; now a layer under every server |
| Jul 2022 | **Orca** (Yu et al.) | OSDI 2022 | **Iteration-level scheduling** (continuous batching) + **selective batching** | The foundational scheduler: refresh the batch every token; mix prefill/decode. All later systems inherit it |
| Jun 2023 | Anyscale continuous-batching benchmark | Blog | Quantified static vs continuous batching: ~23× throughput | Made the case for continuous batching concrete and public |
| Jul 2023 | FlashAttention-2 (Dao) | arXiv | Better GPU work partitioning | ~2× over FlashAttention-1 |
| Sep 2023 | **vLLM / PagedAttention** (Kwon et al.) | SOSP 2023 | KV cache as **paged virtual memory**: fixed blocks, block tables, COW sharing | Watershed: killed 60–80% KV memory waste; 2–4× over FT/Orca; open-source reference; industry norm |
| Oct 2023 | TensorRT-LLM (NVIDIA) | Release | "In-flight batching" (= continuous batching) + paged KV in a vendor stack | Continuous batching + paging go mainstream/production |
| Dec 2023–Jan 2024 | **SGLang / RadixAttention** (Zheng, Sheng et al.) | arXiv / LMSYS blog | Automatic cross-request KV reuse via a radix tree of prefixes | Generalized prefix sharing; big TTFT wins on prefix-heavy workloads |
| Jan 2024 | **DistServe** (Zhong et al.) | OSDI 2024 | **Prefill/decode disaggregation** onto separate GPU pools, optimizing **goodput** | Eliminated phase interference; independent scaling; 7.4× more requests / 12.6× tighter SLO |
| Mar 2024 | **Sarathi-Serve** (Agrawal et al.) | OSDI 2024 | **Chunked prefill** + **stall-free batching** | Hides prefill in decode's compute slack; tames throughput–latency tradeoff without disaggregating |
| Jun 2024 | **Splitwise** (Patel, Choukse et al.) | ISCA 2024 | Phase splitting as a cost/power hardware co-design | Framed disaggregation as an economics problem; phase-matched hardware |
| Jul 2024 | FlashAttention-3 (Shah, Dao et al.) | arXiv | Hopper-tuned attention (async, FP8) | Kept the kernel layer current with new GPUs |

*(Dates are first public appearance — arXiv submission or conference/blog — not necessarily journal-of-record publication.)*

---

## 9. The mental models the field actually uses

If you internalize nothing else, internalize these. Practitioners reason in this vocabulary:

1. **Prefill vs. decode are two different machines.** Prefill = parallel, compute-bound, sets TTFT. Decode = sequential, memory-bandwidth-bound, sets TPOT. Almost every design choice is really a statement about how to treat these two phases (interleave them? chunk one? split them onto different GPUs?).
2. **The KV cache is the real bottleneck.** Not FLOPs — memory. KV-cache capacity caps batch size, and batch size caps decode throughput. Manage that memory well and everything else follows; manage it badly and no kernel saves you.
3. **Batching is the only way to feed the tensor cores during decode.** Because decode's arithmetic intensity ≈ 1, a single request wastes the GPU. Batch size *is* the amortization factor for weight-loading. This is *why* continuous batching exists.
4. **Roofline / arithmetic-intensity thinking.** For any operation, compare FLOPs-to-bytes against the GPU's compute-to-bandwidth ratio (the ridge point). Below the ridge you're bandwidth-bound (decode); above it you're compute-bound (prefill). This one model predicts almost every performance behavior you'll see.
5. **The memory hierarchy: HBM vs. SRAM.** GPU on-chip SRAM is tiny and blindingly fast; HBM is large and (relatively) slow. Winning kernels (FlashAttention) minimize HBM traffic by doing as much as possible in SRAM. "Count the bytes moved across the slow boundary" is the habit of mind.
6. **Goodput > throughput.** The objective is the request rate you sustain *while meeting latency SLOs* (TTFT and TPOT percentiles), not raw tokens/sec. A design that maximizes throughput while blowing your p99 latency is a failure. This reframing (2024) is what motivated chunked prefill and disaggregation.
7. **Borrow from operating systems.** Paging (PagedAttention), LRU caches and radix trees (RadixAttention), copy-on-write, demand allocation, scheduling granularity — LLM serving keeps rediscovering that its problems are OS problems in a new costume. When stuck, ask "what did OS designers do about this?"

---

## 10. The through-line: the 4–5 ideas everything is a variation of

Strip away the system names and the whole field reduces to a handful of moves, each a direct response to a physical fact from §1:

1. **Batch to beat memory bandwidth.** Decode is bandwidth-bound, so amortize weight-loading across many requests. *Everything* about serving throughput descends from this. (The reason continuous batching, disaggregation, and prefix caching all exist is to keep batches full and useful.)

2. **Schedule at the iteration, not the request** (Orca). Because sequences finish at unpredictable, different times, the only way to keep the batch full is to make admit/evict decisions every single token. Continuous batching is this idea; it is non-negotiable in any modern server.

3. **Treat KV-cache memory like an operating system treats RAM** (vLLM/PagedAttention). Fixed-size blocks, indirection through a block table, demand allocation, copy-on-write sharing. This turns "how many requests fit" from a fragmentation lottery into near-optimal packing — and packing is throughput.

4. **Minimize traffic across the slow memory boundary** (FlashAttention, and the roofline mindset generally). Do the math where the data already is (SRAM); never move bytes you don't have to (never materialize N×N in HBM). This is the kernel-level expression of the same bandwidth obsession as idea #1.

5. **Separate prefill from decode and optimize for goodput, not throughput** (Sarathi-Serve, DistServe, Splitwise). The two phases have opposite hardware appetites and interfere when mixed carelessly. Whether you interleave finely (chunked prefill) or split the hardware (disaggregation), the move is the same: stop pretending prefill and decode are one workload, and measure success by SLO attainment.

Everything newer — speculative decoding, quantized KV caches, KV offloading to CPU/NVMe, MoE-aware scheduling, cross-region caching — is a variation, combination, or refinement of these five. Build your server around them and you are building on the field's actual load-bearing ideas rather than reinventing its dead ends.

---

## Sources

Primary sources (papers and official pages) preferred; each key claim was cross-checked against at least two sources. Access dates: all retrieved **July 23, 2026**.

**Orca (iteration-level scheduling, selective batching)**
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," USENIX OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu (PDF: https://www.usenix.org/system/files/osdi22-yu.pdf) — published July 2022.
- FriendliAI research overview of Orca / continuous (iteration) batching. https://friendli.ai/research/orca

**vLLM / PagedAttention**
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," ACM SOSP 2023. arXiv:2309.06180, submitted Sep 12, 2023. https://arxiv.org/abs/2309.06180 (PDF: https://arxiv.org/pdf/2309.06180)
- ACM DL record, SOSP '23: https://dl.acm.org/doi/10.1145/3600006.3613165
- vLLM PagedAttention design docs: https://docs.vllm.ai/en/latest/design/paged_attention/
- Wikipedia, "PagedAttention" (for adoption/industry-norm framing): https://en.wikipedia.org/wiki/PagedAttention

**FlashAttention**
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135 (May 2022); NeurIPS 2022. https://arxiv.org/abs/2205.14135 (OpenReview: https://openreview.net/pdf?id=H4DqfPSibmx)
- Dao-AILab implementation and successor versions (FA-2, FA-3): https://github.com/Dao-AILab/flash-attention

**Pre-LLM batching (Triton, FasterTransformer)**
- NVIDIA Triton Inference Server, dynamic batching docs: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
- Triton concurrency + dynamic batching tutorial: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/examples/jetson/concurrency_and_dynamic_batching/README.html

**Continuous batching, quantified**
- Anyscale, "How continuous batching enables 23x throughput in LLM inference while reducing p50 latency" (2023). https://www.anyscale.com/blog/continuous-batching-llm-inference
- Baseten, "Continuous vs dynamic batching for AI inference": https://www.baseten.co/blog/continuous-vs-dynamic-batching-for-ai-inference/

**Chunked prefill / stall-free (Sarathi-Serve)**
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," USENIX OSDI 2024. arXiv:2403.02310. https://arxiv.org/abs/2403.02310 (USENIX: https://www.usenix.org/conference/osdi24/presentation/agrawal)

**Disaggregation / goodput (DistServe, Splitwise)**
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," USENIX OSDI 2024. arXiv:2401.09670. https://arxiv.org/abs/2401.09670 (USENIX: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- Patel, Choukse et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024. https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/ (ACM DL: https://dl.acm.org/doi/10.1109/ISCA59077.2024.00019)

**Prefix reuse (SGLang / RadixAttention)**
- Zheng, Sheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," arXiv:2312.07104. https://arxiv.org/abs/2312.07104
- LMSYS blog, "Fast and Expressive LLM Inference with RadixAttention and SGLang," Jan 17, 2024. https://www.lmsys.org/blog/2024-01-17-sglang/

**Roofline / arithmetic intensity / prefill-decode physics**
- "Prefill Is Compute-Bound. Decode Is Memory-Bound. Why Your GPU Shouldn't Do Both," Towards Data Science. https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/
- "Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference," arXiv:2503.08311. https://arxiv.org/html/2503.08311v2

**Production stacks (TensorRT-LLM, TGI)**
- NVIDIA TensorRT-LLM in-flight batching (technical blog): https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/
- Hugging Face Text Generation Inference (continuous batching + FlashAttention/PagedAttention): referenced via Baseten/TGI comparisons above.

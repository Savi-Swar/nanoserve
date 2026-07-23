# The Attention-Kernel and Prefill-Scheduling Frontier

*A survey for a from-scratch, pure-PyTorch inference server.*
*Last updated: 2026-07-23.*

## Why this document exists

If you are building an inference server in pure PyTorch, there is a hard line running through the system. **Below the line** is the attention kernel — a fused, hand-tuned CUDA/CUTLASS artifact that decides how fast a single forward pass runs and how much memory the KV cache wastes. You are not writing this. Companies with dozens of GPU engineers (Dao-AILab, NVIDIA, the FlashInfer team) write it, and re-write it for every new GPU generation. **Above the line** is scheduling, batching, memory management, and request orchestration — the layer that decides *which tokens get fed to the kernel, in what shape, in what order*. This layer is pure Python/PyTorch, it is where 2023-2025 systems research made most of its wall-clock gains, and it is entirely yours to implement.

The purpose of this document is to let you (a) describe the kernel layer accurately so you can position your project honestly ("nanoserve calls `F.scaled_dot_product_attention` / FlashInfer; it does not reimplement FlashAttention"), and (b) identify the frontier ideas that are *algorithmic and schedulable* rather than kernel-level, so you can actually implement them and claim them.

The one-sentence summary: **FlashAttention is the substrate everything sits on; PagedAttention and FlashInfer are the memory-layout kernels serving frameworks standardized on; and chunked prefill / stall-free scheduling (Sarathi-Serve) plus attention sinks (StreamingLLM) are the ideas you can build in a scheduler without touching CUDA.**

---

## Part 1 — The kernel substrate: FlashAttention 1 / 2 / 3

### The core problem attention has

Naive attention computes `S = QKᵀ` (an N×N matrix for sequence length N), applies softmax rowwise to get `P`, then `O = PV`. The N×N score matrix `S` and probability matrix `P` are the killers: for N = 8K, that is a 64M-entry matrix *per head per layer*, and the standard implementation **materializes it in HBM** (GPU high-bandwidth memory). Attention is not compute-bound — it is **memory-bound**. Runtime is dominated by reads/writes of `S`/`P` between slow HBM (~2 TB/s on A100) and the arithmetic units, not by the matmuls themselves.

### FlashAttention (v1) — May 2022

**Paper:** Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135, NeurIPS 2022.

**The insight — IO-awareness via tiling + online softmax.** FlashAttention computes *exact* attention (not an approximation) but reorganizes it so the N×N matrices are **never written to HBM**. It tiles Q, K, V into blocks small enough to fit in on-chip SRAM (~100 KB, ~19 TB/s), and streams through K/V blocks while maintaining a running softmax using the **online softmax** trick (Milakov & Gimelshein 2018): keep a running max `m` and running normalizer `ℓ`, and rescale the accumulated output each time a new block shifts the max. Softmax is computed incrementally without ever seeing the full row at once. The N×N intermediate lives and dies in SRAM.

**Why it wins (the asymptotics).** Standard attention needs Θ(Nd + N²) HBM accesses; FlashAttention needs Θ(N²d²/M) where M is SRAM size and d is head dim. For d ≈ 64–128 and M ≈ 100 KB, d²/M ≪ 1, so it does *many times* fewer HBM accesses. It also uses **recomputation**: rather than storing `P` for the backward pass, it recomputes it from the stored softmax statistics — trading extra FLOPs for far less memory, which is a net win because the op is memory-bound. **Memory drops from quadratic to linear in N.**

**Measured impact (v1):** 15% end-to-end speedup on BERT-large (seq 512) over the MLPerf 1.1 record; **3× on GPT-2** (seq 1K); 2.4× on long-range-arena (seq 1K–4K). It enabled the first Transformers to beat chance on **Path-X (16K)** and **Path-256 (64K)**. This is why longer context windows became practical at all.

### FlashAttention-2 — July 2023

**Paper:** Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691, ICLR 2024.

FA1 was still only ~25–40% of theoretical FLOPs on A100 because of **suboptimal work partitioning** between thread blocks and warps. FA2 is an engineering rewrite, not a new algorithm:

1. **Reduced non-matmul FLOPs.** Non-matmul ops (the rescaling in online softmax) run on much lower-throughput units than Tensor Cores, so FA2 tweaks the algorithm to move work onto matmuls and defer/minimize rescaling.
2. **Parallelize over the sequence-length dimension**, not just batch×heads. FA1 parallelized over batch and heads only; when batch×heads is small (long sequences, small batch), the GPU sat idle. FA2 adds a parallelization axis over query blocks, so more SMs stay busy.
3. **Better warp partitioning ("split-Q").** FA1 split K across warps ("split-K"), which forced warps to communicate through shared memory. FA2 splits **Q** across warps and keeps K/V shared, eliminating that inter-warp synchronization.

**Measured impact:** ~2× over FA1, reaching **50–73% of theoretical max FLOPs** on A100 (up to ~230 TFLOPs/s FP16). FA2 became the default attention kernel in essentially every training and inference stack from late 2023 onward.

### FlashAttention-3 — July 2024

**Paper:** Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," arXiv:2407.08608, NeurIPS 2024. (Blog: tridao.me/blog/2024/flash3.)

FA2 was designed for Ampere. On **Hopper (H100)** it hit only **~35% utilization** because it didn't use Hopper's new hardware: the **Tensor Memory Accelerator (TMA)** for async bulk copies, warpgroup-wide **WGMMA** async matmuls, and native **FP8** Tensor Cores. FA3 is a Hopper-specific rewrite around three ideas:

1. **Asynchrony via warp specialization + TMA.** Producer warps issue async TMA loads while consumer warps run WGMMA matmuls, overlapping data movement with compute so the pipeline never stalls waiting on memory.
2. **Interleave (overlap) GEMM and softmax.** The `QKᵀ` matmul (Tensor Cores) and the exp/softmax (multifunction units) are on different execution units. FA3 uses **ping-pong scheduling** across warpgroups and intra-warpgroup pipelining so that while one warpgroup does softmax, another does the next matmul. Reported internal progression: inter-warpgroup overlap 570→620 TFLOPs, intra-warpgroup 620→640-660 TFLOPs (FP16).
3. **FP8 with incoherent processing.** For FP8, block quantization plus **incoherent processing** (multiply Q and K by a random orthogonal / Hadamard matrix to spread outliers before quantizing) reduces quantization error. **2.6× smaller numerical error** than baseline FP8 attention on outlier-heavy data.

**Measured impact:** **1.5–2.0× faster than FA2** with FP16, reaching **740 TFLOPs/s (75% of H100's theoretical max)**; FP8 reaches **~1.2 PFLOPs/s**. This is the current high-water mark for dense attention on Hopper.

### Why this is the substrate

Every serving system — vLLM, SGLang, TensorRT-LLM, TGI — calls a FlashAttention-lineage kernel (or FlashInfer, which is itself FA-derived) for the actual `softmax(QKᵀ)V`. **You will too**, via `torch.nn.functional.scaled_dot_product_attention` (which dispatches to a FlashAttention-2 backend on suitable GPUs) or by calling FlashInfer directly. Re-deriving these kernels is a multi-engineer-year CUDA/CUTLASS effort tied to a specific GPU microarchitecture; a from-scratch PyTorch server has no business reimplementing them and should say so plainly.

---

## Part 2 — Decode-phase kernels: Flash-Decoding & FlashDecoding++

FA1/FA2 were tuned for **training and prefill**, where you process many query tokens at once (batch×heads×query_len is large, so the GPU is saturated). The **decode phase** is different: you generate one token at a time, so the query is a *single* token per sequence. With small batch and long context, batch×heads is tiny — the GPU is starved even though the KV cache is huge.

### Flash-Decoding — October 2023

**Source:** Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context inference," Together AI / PyTorch blog, Oct 12 2023.

**Mechanism — add a parallelization axis over KV length.** During decode, a single query attends over the entire KV cache. Flash-Decoding splits the K/V along the **sequence dimension** into chunks and, in a first pass, computes partial attention (a partial output plus a **log-sum-exp scalar per split**) for each chunk **in parallel using FlashAttention**. A second pass reduces across splits, rescaling each partial output by its log-sum-exp — the same online-softmax combine, but now across chunks instead of within a kernel. This keeps *all* SMs busy even when batch=1, at the cost of a small final reduction.

**Measured impact:** up to **8× faster decoding** for very long sequences; the attention component itself up to **50× faster than FlashAttention** in microbenchmarks; attention time stays **roughly constant out to 32K–64K tokens** (CodeLLaMa-34B, A100, batch 1–256, seq 512–64K).

This "split-KV" / "split-K decoding" idea is now standard and is what makes long-context decode tolerable. It is a **kernel-level** technique, but the underlying idea — decode is memory-bound on the KV cache, parallelize over KV — is worth understanding because it explains why *batching more sequences together* (a scheduler concern) is the main lever you actually control.

### FlashDecoding++ — November 2023 / MLSys 2024

**Paper:** Hong et al., "FlashDecoding++: Faster Large Language Model Inference on GPUs," arXiv:2311.01282, MLSys 2024.

Three additions aimed at both prefill and decode:

1. **Asynchronous softmax with a unified max.** Partial-softmax computations normally need to synchronize to agree on the running max. FlashDecoding++ observes attention scores are statistically well-bounded, so it uses a **pre-set unified maximum** for all partial softmaxes, removing the synchronization between splits (with a rare recompute fallback if the guess is exceeded). → ~1.05× prefill, ~1.14× decode.
2. **Flat-GEMM optimization with double buffering.** Decode matmuls are "flat" (one dimension is tiny, e.g. M=1). FlashDecoding++ pads to a better shape and double-buffers → up to **52%** faster flat GEMM.
3. **Heuristic dataflow with hardware adaptation** — pick GEMM implementations based on input shape/GPU → up to **29%** over static dataflow.

**Measured impact:** up to **4.86× vs HuggingFace on NVIDIA**, **2.18× on AMD**. Takeaway for you: decode is dominated by memory movement and flat, awkward matmul shapes — this is exactly why single-request decode is inefficient and why **continuous batching** (packing many sequences' decode steps into one kernel launch) is the highest-leverage thing your scheduler does.

---

## Part 3 — The memory-layout kernel you must emulate: PagedAttention

This is the most important section for a from-scratch implementer, because PagedAttention is not really an *attention algorithm* — it is a **KV-cache memory-layout scheme plus a kernel that reads from that layout**. You have to solve the same problem it solves, and how you solve it determines your throughput ceiling.

### The problem: KV-cache fragmentation

**Paper:** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," arXiv:2309.06180, SOSP 2023 (this is the vLLM paper).

The KV cache grows one token at a time and you don't know the final length in advance. The naive approach reserves a **contiguous** buffer sized to `max_seq_len` per request. This causes three kinds of waste: **internal fragmentation** (reserved-but-unused tail), **external fragmentation** (contiguous free blocks of the wrong size), and no sharing across requests. vLLM measured that naive systems waste **60–80%** of KV memory. Since KV memory caps your batch size, and batch size caps your throughput, this waste directly throttles the server.

### The solution: paging, borrowed from OS virtual memory

PagedAttention partitions each sequence's KV cache into fixed-size **blocks** (e.g. 16 tokens/block). Blocks need **not be contiguous** in physical GPU memory. A per-request **block table** maps *logical* block index → *physical* block address (exactly like an OS page table maps virtual → physical pages). Blocks are allocated on demand as generation proceeds. Result: waste only occurs in the last partial block of each sequence — **under 4%** internal fragmentation, near-zero external fragmentation. It also enables **copy-on-write block sharing** (e.g. a shared prompt prefix, or parallel samples from one prompt) via reference-counted blocks.

### The kernel: attention over non-contiguous blocks

The PagedAttention **CUDA kernel** is the part that makes this work at speed. During `softmax(QKᵀ)V`, instead of striding through one contiguous K/V tensor, the kernel:

1. reads the request's **block table**,
2. for each logical position, **translates** to the physical block and offset,
3. **gathers** the K/V for that block (often directly into registers/SRAM), and
4. runs the FlashAttention-style online-softmax accumulation over the gathered blocks.

The indirection (a block-table lookup + non-contiguous fetch per block) costs vLLM a measured **~5–10% slowdown in raw kernel latency** vs a fully-contiguous FlashAttention read — a trade they gladly make because the memory savings let them run **much larger batches** (often 2–4× throughput overall). **This gather-from-block-table is the exact thing a from-scratch implementer must emulate.**

### What pure-PyTorch alternatives look like (and their costs)

You cannot write the fused paged kernel in Python. Here are your realistic options, cheapest-quality to best, with honest costs:

| Approach | How | Cost / limitation |
|---|---|---|
| **Contiguous per-request buffers + `F.sdpa`** | Give each sequence its own preallocated `[max_len, n_kv_heads, d]` KV tensor; slice `[:cur_len]` each step. | Simplest, and `sdpa` gives you FA2-speed attention. But you pay the **60–80% memory waste** PagedAttention was invented to fix → small batches → low throughput. Fine for a demo/single-user server. |
| **Gather / `index_select` into a contiguous scratch buffer** | Keep a global paged KV pool; each step, use the block table to `torch.index_select`/`gather` the active blocks into a **temporary contiguous tensor**, then call `sdpa`. | This is "paged storage, contiguous compute." You get paged memory efficiency, but the **gather is a separate HBM round-trip** (materialize the whole active KV contiguously every step) — exactly the extra memory traffic the fused kernel avoids by gathering *inside* the kernel. Extra latency grows with context length. |
| **Ragged batch + masked/padded `sdpa`** | Pad all sequences in a batch to the max length, build an additive attention mask, one big `sdpa` call. | Simple and vectorized, but you **compute (and pay memory for) the padding**: a batch with one 8K sequence and seven 200-token sequences does ~8K×8 work. Wasteful when lengths are skewed; the mask itself can be O(batch·N²) if materialized. |
| **PyTorch `FlexAttention` + paged block mask** | `torch.nn.attention.flex_attention` compiles a fused kernel from a Python `score_mod`/`BlockMask`; it has a paged-attention path that maps a page table into the block mask. | This is the closest a "PyTorch-native" project gets to a real paged kernel — it *compiles* down via `torch.compile`/Triton, so it's not pure eager PyTorch, but it's not hand-written CUDA either. Caveats (as of 2025-2026): backend restrictions (e.g. SGLang's FlexAttention path required `page_size=1`, no FP8 KV, no sliding window). Up to ~5× over naive masked `sdpa` for structured sparsity. |
| **Call FlashInfer / vLLM's kernel directly** | Import the prebuilt kernel. | Best performance, but now you're *depending on* the custom kernel, not writing PyTorch — which is a legitimate architecture, just be honest that the kernel isn't yours. |

**The honest positioning:** a pure-PyTorch server can implement the **block table, the allocator, copy-on-write sharing, and prefix caching** — all the *policy* — in Python, and that is genuinely most of the intellectual content of PagedAttention. What it cannot do in eager PyTorch is the **fused gather-inside-the-kernel**; it must fall back to `index_select`-then-`sdpa` (extra HBM traffic) or padded `sdpa` (wasted compute) or `FlexAttention` (compiled, not eager). Say exactly that.

---

## Part 4 — The standardized serving kernel: FlashInfer

**Paper:** Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving," arXiv:2501.01005, MLSys 2025 (**Best Paper Award**). Repo: github.com/flashinfer-ai/flashinfer.

By 2024, every serving framework had its own slightly-different attention kernels (paged, prefix-shared, speculative, sliding-window). FlashInfer's thesis: these are **all the same computation over different sparsity patterns**, so unify them. It is now the *de facto* standard attention backend for serving.

**Three contributions:**

1. **Block-sparse composable KV format.** FlashInfer represents the KV cache as **block-sparse** (BSR-like) matrices with configurable block sizes. Crucially, **PagedAttention is just block-sparse attention where each page is a sparse block**; **prefix sharing** is block-sparse with shared blocks; **speculative/tree decoding** is block-sparse over a token tree. One formulation, many serving features. This is the conceptual unification worth citing: *paged + prefix + speculative attention are special cases of block-sparse attention.*
2. **JIT-compiled customizable templates.** A CUDA/CUTLASS template lets users define attention *variants* (custom masks, `score_mod`-style logits transforms, different KV layouts) and JIT-compiles a fused kernel, instead of shipping a combinatorial explosion of hand-written kernels.
3. **Load-balanced scheduling compatible with CUDAGraph.** Requests have wildly different lengths, causing GPU load imbalance. FlashInfer plans a load-balanced work assignment **while keeping shapes static enough for CUDAGraph capture** (CUDAGraph needs fixed configuration; dynamic batching fights that). This is the hard systems trick.

**Measured impact:** **29–69% inter-token-latency reduction** vs compiler backends; **28–30% latency reduction** for long-context; **13–17% speedup** for parallel generation. Runs on **Turing (2018 T4) through Blackwell (B200)**. Adopted by **vLLM, SGLang, MLC-Engine/MLC-LLM, TensorRT-LLM, TGI**.

**Relevance to you:** FlashInfer is the thing your project is "not writing." It is also the cleanest mental model for *why* paged/prefix/speculative are one problem — if you ever want block sharing or tree/speculative decode, the block-sparse lens is how the frontier thinks about it. In practice a PyTorch server that wants production kernels imports FlashInfer rather than reimplementing it.

---

## Part 5 — The schedulable frontier: chunked prefill & stall-free batching (Sarathi / Sarathi-Serve)

**This is the part you can actually build.** It is a pure scheduling technique — no custom kernels — and it produced large measured gains.

**Papers:** Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," arXiv:2308.16369 (Aug 2023); and Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," arXiv:2403.02310, **OSDI 2024**.

### The problem: prefill-decode interference

Prefill and decode have opposite hardware profiles. **Prefill** processes the whole prompt at once — it is **compute-bound** and saturates the GPU. **Decode** does one token per sequence — it is **memory-bound** and *underutilizes* compute (low arithmetic intensity). When a scheduler naively mixes them, two bad things happen:

- **Decode stalls behind prefill.** In vLLM's original scheduler, when a long prefill enters a batch, all ongoing decodes **pause** while the prefill runs → **TPOT (time-per-output-token) tail latency spikes**. Users see generation "freeze."
- **You can't have both throughput and low latency.** Prefill-prioritizing schedules get throughput but wreck decode latency; decode-prioritizing schedules protect latency but waste the GPU during memory-bound decode steps.

### The fix, part 1 — Chunked prefill

Split a long prompt's prefill into **fixed-size token chunks** (e.g. 512 tokens) and process one chunk per iteration across several iterations, instead of one giant prefill step. A 4K prompt becomes 8 chunks of 512. Each iteration is now bounded in size.

### The fix, part 2 — Stall-free batching (piggybacking decodes)

Because decode batches are memory-bound with **spare compute (spare arithmetic intensity)**, you can **piggyback** a prefill chunk into the same batch as the ongoing decodes "for free" — the prefill chunk uses the compute the decodes weren't using. The scheduler builds **hybrid batches** = (all active decodes) + (one prefill chunk sized to fill a **token budget**). Because every iteration always includes the decodes, **decodes never stall** — hence "stall-free." New requests join without pausing anyone. The **token budget** (max tokens per iteration) is the single tuning knob: it caps iteration time (protecting TPOT), and the prefill chunk fills whatever budget the decodes leave.

### How to implement it (no kernels required)

You already have a continuous-batching loop. To add Sarathi-style scheduling:

1. **Maintain two request states:** `PREFILLING` (with a cursor into the prompt) and `DECODING`.
2. **Per iteration, set a `token_budget`** (e.g. 512). First, add **all** `DECODING` requests (each contributes 1 query token). Then take a `PREFILLING` request and add a **chunk** of its remaining prompt tokens up to the remaining budget; advance its cursor. If a prefill finishes its last chunk, flip it to `DECODING`.
3. **Build one hybrid batch** and run a **single forward pass** — a mix of 1-token decode queries and a contiguous prefill chunk. Your attention call handles variable query lengths per sequence (a ragged/varlen `sdpa` or FlashInfer's batch-prefill wrapper; even padded `sdpa` works at small scale).
4. **Tune `token_budget`** down for tighter TPOT SLOs, up for more throughput.

That's it — this is a Python scheduler change. The KV cache and attention kernel are unchanged.

**Measured impact (vs vLLM baseline, A100s):** **2.6×** higher serving capacity on **Mistral-7B (1×A100)**; up to **3.7×** on **Yi-34B (2×A100)**; up to **5.6×** end-to-end capacity on **Falcon-180B** with pipeline parallelism — all *under the same tail-latency SLO*. Chunked prefill is now a first-class option in vLLM (`--enable-chunked-prefill`) and SGLang, largely because it also smooths the pipeline-parallel bubble. **If your project implements one frontier idea from scratch, this is the highest-ROI one.**

---

## Part 6 — StreamingLLM / attention sinks

**Paper:** Xiao, Tian, Chen, Han, Lewis, "Efficient Streaming Language Models with Attention Sinks," arXiv:2309.17453 (Sep 2023), ICLR 2024. Repo: mit-han-lab/streaming-llm.

**The problem.** For infinite/streaming generation you'd like to keep only a **sliding window** of recent KV to bound memory. But naive **window attention** (drop the oldest tokens) causes perplexity to **explode** the moment the very first tokens fall out of the window — the model breaks.

**The observation — "attention sinks."** LLMs dump a large, near-constant fraction of their attention weight onto the **first few tokens** regardless of semantic relevance. Softmax must sum to 1, so when no "real" token deserves attention, the model parks the excess on the initial tokens — they act as an **attention sink** / no-op bias. When the window evicts those initial tokens, the softmax distribution is destabilized and the model collapses.

**The fix.** Keep the KV of the **first ~4 tokens ("sink tokens") permanently**, plus a sliding window of recent tokens. Evict only the middle. This is a **KV-cache eviction policy**, not a kernel.

**Measured impact.** Enables stable modeling on up to **4M+ tokens** with a fixed cache; up to **22.2× speedup** over the sliding-window-with-recomputation baseline. Note: it *does not* extend the model's effective context — it enables **stable, bounded-memory streaming**, not recall of the evicted middle. (A dedicated learned sink token added during pretraining improves this further; some modern models ship one.)

**Relevance to you.** This is fully schedulable: it's a rule in your KV-cache manager — "pin blocks 0..k, evict from the middle of the window." Pairs naturally with a paged KV cache (pin the sink pages). Cheap to implement, good story for long-running / streaming endpoints.

---

## Part 7 — Quantized / FP8 attention, and where it matters

Two distinct places quantization touches attention:

- **FP8 KV cache (storage).** Store K/V in FP8 (E4M3/E5M2) or INT8 instead of FP16 → **halve KV memory**, roughly doubling the batch/context you can hold. This is a **serving-side** win and is schedulable-ish: you pick the KV dtype; the attention kernel must support reading FP8 KV (FlashInfer, vLLM, TensorRT-LLM do). Main risk is accuracy on outlier-heavy heads; per-channel/per-token scales mitigate it.
- **FP8 attention compute.** Do the `QKᵀ` and `PV` matmuls in FP8 on Hopper/Blackwell Tensor Cores → the ~1.2 PFLOPs/s FA3-FP8 regime. This is **kernel-level** and matters most where attention is a large share of runtime: **very long context** and **high-throughput serving**.

**SageAttention** (Zhang et al., arXiv:2410.02367, ICLR 2025; and SageAttention2/2++ 2025) is the notable line here: an **8-bit** quantized attention (e.g. INT8 for `QKᵀ`, FP8 for `PV`, with smoothing to handle outliers) reporting **2–3× over FlashAttention-2** and, in later versions, matching **FA3-FP8 speed with better accuracy** — "plug-and-play," no retraining, near-zero end-to-end metric loss across language/image/video models. Heavily adopted in **diffusion/video** inference (where attention dominates), increasingly relevant to long-context LLM serving.

**For your project:** FP8 *compute* attention is kernel work you won't write. FP8/INT8 **KV cache** is a memory-layout choice you *can* expose if your attention backend supports the dtype — a cheap way to grow effective batch size. Frame FP8-compute as "kernel-level, out of scope"; frame FP8 KV as "a config knob gated on backend support."

---

## Part 8 — Honest framing: what a pure-PyTorch server can and cannot do

**Cannot do (kernel-level, not your project):**

- Write FlashAttention 1/2/3 or match their SRAM tiling, warp specialization, TMA/WGMMA async, or FP8 Tensor-Core paths. These are CUDA/CUTLASS, microarchitecture-specific, multi-engineer-year artifacts.
- Write the **fused** PagedAttention kernel that gathers non-contiguous KV blocks *inside* the attention loop. In eager PyTorch you gather *first* (`index_select` → contiguous scratch, extra HBM traffic) or pad (`masked sdpa`, wasted compute) or compile (`FlexAttention`/Triton, no longer eager).
- Write FP8/INT8 attention compute kernels (SageAttention, FA3-FP8).
- Get CUDAGraph-compatible load-balanced kernel scheduling (FlashInfer's trick).

**Can do (algorithmic / schedulable — genuinely most of the systems content):**

- **Continuous batching** — pack many sequences' decode steps into one forward pass. The single biggest throughput lever, and pure Python.
- **PagedAttention *policy*** — block table, block allocator, on-demand allocation, copy-on-write prefix sharing, prefix caching. The kernel gather is the only part you must approximate; the memory *management* is all yours.
- **Chunked prefill + stall-free hybrid batching (Sarathi-Serve)** — a scheduler change, 2.6–5.6× measured serving-capacity gains. Highest-ROI frontier idea to implement.
- **Attention sinks / StreamingLLM eviction** — a KV-cache policy rule for bounded-memory streaming.
- **FP8/INT8 KV cache** as a config knob (gated on backend dtype support).
- **Split-KV intuition** — even if you don't write the kernel, knowing decode is KV-memory-bound tells you batching is your lever.

**The one-line positioning for nanoserve:** *"nanoserve implements the scheduling and memory-management frontier — continuous batching, a paged KV cache with a block table and prefix sharing, Sarathi-style chunked-prefill stall-free batching, and attention-sink eviction — all in pure PyTorch. It calls `F.scaled_dot_product_attention` (FA2 backend) / FlashInfer for the fused `softmax(QKᵀ)V` itself, because that kernel layer is a GPU-microarchitecture-specific CUDA artifact that no honest pure-PyTorch project reimplements."* That is an accurate, defensible framing: you own the algorithmic frontier and cite the kernel frontier.

---

## Sources

**FlashAttention 1/2/3**
- FlashAttention (v1): Dao et al., arXiv:2205.14135 — https://arxiv.org/abs/2205.14135 (submitted May 27 2022; NeurIPS 2022)
- FlashAttention-2: Dao, arXiv:2307.08691 — https://arxiv.org/abs/2307.08691 (Jul 17 2023; ICLR 2024)
- FlashAttention-3: Shah et al., arXiv:2407.08608 — https://arxiv.org/abs/2407.08608 (Jul 2024; NeurIPS 2024)
- FA3 blog (Tri Dao): https://tridao.me/blog/2024/flash3/ (Jul 2024)
- FA3 NeurIPS proceedings PDF: https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf

**Flash-Decoding / FlashDecoding++**
- Flash-Decoding blog (Princeton NLP): https://princeton-nlp.github.io/flash-decoding/ (Oct 12 2023)
- Flash-Decoding blog (Together AI): https://www.together.ai/blog/flash-decoding-for-long-context-inference (Oct 2023)
- FlashDecoding++: Hong et al., arXiv:2311.01282 — https://arxiv.org/abs/2311.01282 (Nov 2023; MLSys 2024). Proceedings: https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf

**PagedAttention / vLLM**
- Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention," arXiv:2309.06180 — https://arxiv.org/abs/2309.06180 (SOSP 2023)
- vLLM blog: https://vllm.ai/blog/2023-06-20-vllm (Jun 20 2023)
- vLLM v1 flash_attn backend docs: https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/flash_attn/ (accessed Jul 2026)

**FlashInfer**
- Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving," arXiv:2501.01005 — https://arxiv.org/abs/2501.01005 (Jan 2025; MLSys 2025 Best Paper). Repo: https://github.com/flashinfer-ai/flashinfer
- NVIDIA blog: https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/ (2025)

**Chunked prefill / Sarathi**
- SARATHI: Agrawal et al., arXiv:2308.16369 — https://arxiv.org/abs/2308.16369 (Aug 2023)
- Sarathi-Serve: Agrawal et al., arXiv:2403.02310 — https://arxiv.org/abs/2403.02310 (OSDI 2024). Proceedings: https://dl.acm.org/doi/10.5555/3691938.3691945

**StreamingLLM / attention sinks**
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks," arXiv:2309.17453 — https://arxiv.org/abs/2309.17453 (Sep 29 2023; ICLR 2024). Repo: https://github.com/mit-han-lab/streaming-llm

**Quantized / FP8 attention**
- SageAttention: Zhang et al., arXiv:2410.02367 — https://arxiv.org/abs/2410.02367 (Oct 2024; ICLR 2025). Repo: https://github.com/thu-ml/SageAttention
- SageAttention2++: arXiv:2505.21136 — https://arxiv.org/abs/2505.21136 (2025)
- INT-FlashAttention: arXiv:2409.16997 — https://arxiv.org/abs/2409.16997 (2024)

**PyTorch-native attention**
- FlexAttention docs: https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html (accessed Jul 2026)
- FlexAttention paper (MLSys 2025): https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf
- SDPA tutorial: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

*Cross-verification: every headline number above (FA3 740 TFLOPs/75% util, FA2 35% Hopper util, Flash-Decoding 8×/50×, PagedAttention <4% waste & 5–10% kernel slowdown, FlashInfer 29–69% ITL, Sarathi-Serve 2.6×/3.7×/5.6×, StreamingLLM 4M tokens/22.2×) was confirmed against at least the primary source plus one secondary (proceedings, official blog, or independent summary).*

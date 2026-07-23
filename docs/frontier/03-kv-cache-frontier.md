# The KV-Cache-Centric Frontier of LLM Serving (2024–2026)

*A research briefing for building a from-scratch inference server with a paged KV cache.*
*Compiled 2026-07-23. All claims cross-checked against ≥2 sources where possible; see Sources.*

---

## 0. Framing: why the KV cache is the object the stack is organized around

Autoregressive transformers cache the key/value projections of every past token so
that each new decode step is `O(1)` in attention work instead of `O(n)`. That cache —
the **KV cache** — is the single largest, most dynamic, and most reusable piece of
state in the serving loop, and over 2024–2026 essentially every serving-systems
advance has been a way to *store it more cheaply, move it around, share it, shrink it,
or route to where it already lives*.

Three structural facts drive this:

1. **It is the memory bottleneck, not compute.** On an A100-40GB serving a 13B FP16
   model, weights take ~26 GB, leaving only ~12 GB for KV cache; KV costs roughly
   **~0.8 MB per token**, so a single long sequence can consume gigabytes and the
   *number of concurrent sequences you can batch* is capped by KV memory, not FLOPs.
   ([Kwon et al., PagedAttention, SOSP 2023]; corroborated by Introl/runpod writeups.)

2. **Naive allocation wastes most of it.** Pre-vLLM systems that reserved a contiguous
   max-length buffer per request wasted an estimated **60–80% of KV memory** to
   internal/external fragmentation and reservation. PagedAttention cut waste to
   **under 4%**, and that memory-efficiency win alone yielded **2–4× throughput** over
   FasterTransformer and Orca. Throughput here is almost entirely a function of how
   many requests you can fit in the KV budget. ([SOSP 2023]; Red Hat / vLLM docs.)

3. **It is highly reusable.** The same KV blocks recur *across* requests: shared system
   prompts, few-shot exemplars, agent tool definitions, RAG document chunks, and
   multi-turn conversation history. Recomputing them is pure waste. This is what turns
   the KV cache from a per-request scratchpad into a **cross-request, cross-node
   asset** — the premise behind prefix caching, offloading, and "KV cache as a service."

The rest of this document walks the frontier from single-node internals outward to
cluster-scale KV pools.

---

## 1. PagedAttention internals worth knowing for a from-scratch build

PagedAttention (Kwon et al., vLLM, **SOSP 2023, Best Paper**) borrows OS virtual-memory
paging: the KV cache of a sequence is stored in **fixed-size blocks** that need not be
physically contiguous, mapped through a per-sequence **block table** (logical block →
physical block).

Design points that matter when you implement one:

- **Block size.** vLLM's default is **16 tokens/block**. Small blocks minimize internal
  fragmentation (you waste at most `block_size − 1` token slots in the last block) and
  make prefix sharing fine-grained; large blocks (32–64) amortize block-table lookups
  and kernel overhead and are sometimes preferred for very long (100K+) sequences. The
  block is the unit of allocation, sharing, and eviction, so it sets the granularity of
  everything downstream. (vLLM docs; Spheron/Red Hat explainers, 2025.)

- **Block tables** are the indirection layer; the attention kernel gathers K/V from
  scattered physical blocks via the table. This is exactly what later critiques target
  (see §6).

- **Copy-on-write (CoW).** When two sequences share a prefix (e.g., beam search
  branches, or forked agent states), they share the same physical blocks read-only. On
  the first *write* that diverges, the shared block is copied and the writer gets a
  private copy — the classic OS CoW pattern, applied to KV blocks. This makes parallel
  sampling and beam search cheap in memory.

- **Reference counting + eviction.** Blocks are ref-counted; a block is freed only when
  no sequence references it. Free blocks come from a shared pool, allocated on demand as
  sequences grow.

**vLLM V1 rewrite (2025).** vLLM's V1 engine (see "Inside vLLM: Anatomy of a
High-Throughput LLM Inference System," vLLM blog, **2025-09-05**) reorganizes KV
management around a cleaner **KV cache manager**:

- **Prefix caching is on by default** and folded into the block manager: shared-prefix
  blocks are referenced by multiple requests from a free-block pool rather than
  duplicated, giving on-demand allocation *and* automatic prefix reuse in one structure.
- **Per-request salting** optionally isolates prefix reuse — only requests carrying the
  same salt can hit each other's cached blocks — closing a cross-tenant leakage/privacy
  hole in shared deployments.
- The **V1 scheduler mixes prefill and decode tokens in the same step** (V0 could only do
  one or the other), which interacts with chunked prefill (§3) to smooth latency.

---

## 2. Prefix caching / Automatic Prefix Caching (APC) and RadixAttention

**The idea.** Instead of discarding a request's KV cache when it finishes, keep it and
let the *next* request that shares a leading prefix skip recomputing it. This is
Automatic Prefix Caching (APC). vLLM implements exact-match prefix caching
(`enable_prefix_caching=True`, default in V1): hash the token blocks and reuse blocks
whose prefix hash matches.

**RadixAttention (SGLang / LMSYS, blog 2024-01-17; SGLang system).** SGLang generalizes
prefix caching with a **radix tree** (a space-efficient trie whose edges carry
variable-length token sequences) keyed by token IDs, whose leaves point at KV blocks in
paged GPU memory (SGLang uses one-token pages). Mechanism:

- On a new request, walk the tree to the deepest matching node and **reuse the KV of the
  entire matched path**, computing only from the branch point onward.
- **LRU eviction that recursively evicts leaf nodes** reclaims GPU memory under pressure.
- A **cache-aware scheduling** policy orders/groups requests to *raise the hit rate*
  (e.g., batch requests that share a subtree together).
- It is compatible with continuous batching and paged attention, and extends to image
  tokens for multimodal.

The key generalization vs. vLLM's basic APC: a radix tree naturally shares **any common
sub-sequence reachable as a tree path**, not just one fixed leading prefix, and supports
efficient insert/search/evict.

**Measured impact.**
- SGLang reports **up to 5× higher throughput** than vLLM (v0.2.5), Guidance, and TGI
  across nine workloads on Llama-7B/A10G — MMLU (5-shot), HellaSwag (20-shot), ReAct
  agent, Tree-of-Thought, JSON decode, multi-turn chat, DSPy RAG, LLaVA-bench — winning
  on **all**, with the largest gains in **time-to-first-token** where a prefix hit
  eliminates prefill. (LMSYS blog, 2024-01-17.)
- Workloads with layered shared structure (system prompt + retrieved docs + history)
  see **2–4× higher cache hit rates**; agents sharing a fixed system prompt + tool
  definitions report **75–95% hit rates** on multi-turn sessions. (Secondary syntheses;
  directionally consistent with the primary blog's TTFT emphasis.)

**Why a from-scratch builder cares:** prefix caching is the single highest-leverage
feature for real workloads (chat, agents, RAG, few-shot) because prefixes are enormous
and repeated. If you already have paged blocks + hashing, exact-match APC is a small
addition; a radix tree is the richer version.

---

## 3. Prefill–Decode disaggregation

**The problem.** Prefill (process the whole prompt, one big parallel matmul, compute-
bound) and decode (generate one token at a time, memory-bandwidth-bound) have opposite
resource profiles. Colocating them on the same GPU causes **interference**: a long
prefill stalls in-flight decodes, blowing their **TPOT** (time-per-output-token) SLO,
while decodes fragment prefill batching. They also want *different* parallelism and
*different* batch sizes, and they scale independently.

### DistServe (Zhong et al., OSDI 2024; arXiv 2401.09670, Jan 2024)
- **Assigns prefill and decode to different GPUs**, eliminating prefill↔decode
  interference.
- **Co-optimizes resource allocation and parallelism per phase**, given the app's TTFT
  (prefill) and TPOT (decode) targets, and **places phases by cluster bandwidth** to
  minimize the cost of shipping KV cache between them.
- Result: serves **7.4× more requests** or meets **12.6× tighter SLOs** vs. state-of-the-
  art colocated systems, while keeping >90% of requests within latency limits.

### Mooncake (Moonshot AI / Kimi; arXiv 2407.00079, Jul 2024; FAST '25 Best Paper)
Mooncake is the production platform behind **Kimi**, and it is explicitly **KVCache-
centric**:
- **Disaggregated prefill and decode clusters**, plus a **disaggregated KVCache pool**
  built from the *underutilized CPU, DRAM, and SSD* of the GPU cluster — i.e., the KV
  cache spans a storage hierarchy, not just HBM.
- **Conductor**, a global scheduler, dispatches each request based on the *current
  distribution of KV cache and load* (route to where the prefix already lives), balancing
  throughput against latency SLOs.
- A **prediction-based early-rejection** policy sheds load proactively under overload.
- Impact: **up to 525% throughput** in simulated long-context scenarios under SLO; in
  production it lets Kimi handle substantially more requests (the abstract states ~75%
  more; project/blog materials cite **115% and 107% more requests on A800 and H800
  clusters** respectively vs. the prior system). Operates across **thousands of nodes,
  >100 billion tokens/day**. Long-context is where it shines.

### The KV-cache transfer problem
Disaggregation's defining cost: once prefill finishes, the request's KV cache must be
**moved to the decode GPU/pool** over NVLink / InfiniBand / RDMA / network. This transfer
sits on the critical path to first token, and its cost scales with context length. Much
subsequent work (Mooncake's transfer engine, LMCache connectors, ShadowServe's
interference-free fetching, layer-wise/streamed transfer) is about hiding or compressing
this movement.

### The alternative: chunked prefill (colocation)
Not everyone disaggregates. **Sarathi-Serve** (Agrawal et al., OSDI 2024) keeps prefill
and decode on the *same* GPU but slices each prompt into **bounded-size chunks** and uses
**stall-free batching**: every step prioritizes all active decodes and fills the
remaining token budget with a prefill chunk, so a long prompt never stalls decodes.
Chunked prefill (colocation) vs. PD-disaggregation is now the central architectural fork;
vLLM V1's mixed prefill+decode scheduler is the chunked-prefill lineage. A from-scratch
builder should pick deliberately: disaggregation buys clean SLOs and independent scaling
at the cost of a KV-transfer path; chunked prefill is simpler and single-node but couples
the two phases.

---

## 4. KV cache offloading / tiering — "KV cache as a service"

Once you accept that KV cache is a reusable asset, GPU HBM is too small and too precious
to be its only home. The offloading/tiering line treats KV as a **first-class,
cross-request, cross-node object** living in a hierarchy: **HBM → CPU DRAM → local
NVMe/SSD → remote/networked store**.

### LMCache (arXiv 2510.09665, 2025) + CacheGen
- **LMCache** is a dedicated KV-cache layer that moves KV out of GPU memory into tiered
  storage (CPU RAM, local disk, remote backends) and **reuses it across requests,
  sessions, and even separate engine instances** to avoid recomputing prefill. It ships
  batched data movement, compute/IO pipelining, a modular **KV connector**, and a control
  API for orchestrating KV across GPU/CPU/storage/network. It is integrated with vLLM.
- Reported: at low QPS, **1.9–8.1× smaller TTFT** and **2.3–14× higher query throughput
  at equal TTFT** vs. the strongest baseline across five models.
- **CacheGen** (now part of LMCache) tackles the *transmission* problem: it **compresses
  KV into a compact bitstream** for transfer (custom quantization + arithmetic/entropy
  coding), exploiting that transmitted KV needn't stay in a GPU-usable tensor layout, so
  it can compress far more aggressively than in-GPU formats. This makes distributed
  prefix caching viable even on modest network bandwidth.

### Why this is "KV cache as a service"
The endpoint of this line is a **shared KV store** that any prefill/decode node can read
and write — decoupling *where a prefix's KV was computed* from *where the next request
runs*. That is precisely what Mooncake's pool, ShadowServe, and the cluster routing work
in §7 assume. The trade you are managing: recompute-on-GPU vs. fetch-from-tier — worth it
only when `transfer_cost < prefill_cost`, which is why compression (CacheGen) and fast
interconnects (RDMA) are load-bearing.

---

## 5. KV cache compression — trading quality for capacity

Three families, each shrinking KV along a different axis. All trade some accuracy for
capacity/bandwidth; the practical question is *how much quality per byte saved*.

### (a) Quantization — fewer bits per KV element
- **KVQuant** (Hooper et al., arXiv 2401.18079, **NeurIPS 2024**): non-uniform,
  **outlier-aware** quantization. Keys quantized **per-channel**, and ~**1% of
  high-magnitude channels kept in fp16**. Achieves **2-bit** KV with **<0.5 perplexity
  degradation on Wikitext-2** vs fp16 and **~6.9× memory savings**, enabling contexts
  toward **10M tokens**.
- **KIVI** (Liu et al., 2024): tuning-free, plug-and-play **2-bit** asymmetric
  quantization — **keys per-channel, values per-token** (the KV distributions differ in
  which axis has outliers) — with only **~3% GSM8K accuracy drop**.
- **FP8 KV cache** (production, e.g. TensorRT-LLM / vLLM): store K/V in FP8 (E4M3/E5M2),
  ~2× the KV capacity of FP16 with typically negligible quality loss; the pragmatic
  default in industry.
- **Caveats (2025 work).** 4-bit KV is generally near-lossless; **2-bit degrades sharply
  on reasoning/long-generation** without extra tricks (channel-wise precision boosts,
  variance normalization, calibration). Treat aggressive KV quant as accuracy-sensitive,
  especially for chain-of-thought.

### (b) Token eviction / sparsity — keep fewer tokens
- **StreamingLLM** (Xiao et al., 2023): discovered **attention sinks** — the first few
  tokens absorb disproportionate attention. Keeping the sink tokens **+ a sliding window
  of recent tokens** lets a model stream **effectively unbounded** length with bounded KV,
  at some loss of true long-range recall. The "attention sink" insight is now standard.
- **H2O — Heavy-Hitter Oracle** (Zhang et al., NeurIPS 2023): keep a small set of
  **heavy-hitter** tokens (high accumulated attention) plus recent tokens; with **~20%**
  of tokens retained, up to **29× throughput** over HF Accelerate on OPT-6.7B/30B.
- **Scissorhands** (2023): exploits **persistence of importance** — tokens important once
  tend to stay important — and evicts by a pivotal-token metric with per-layer rates.
- **SnapKV** (Li et al., arXiv 2404.14469, 2024): at the end of prefill, uses an
  **observation window** of the last prompt tokens to vote on which earlier tokens matter,
  **clusters + pools** them, and keeps only those — compressing the *prompt* KV before
  decoding even starts. Strong for long-context prompts.

### (c) Structural / low-rank (architectural, adjacent)
Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and **Multi-head Latent
Attention (MLA, DeepSeek)** shrink KV at the *model* level by sharing/compressing KV heads
— a different lever than the runtime methods above, but the biggest single reason modern
models have manageable KV footprints. Worth knowing because it changes your per-token KV
size (and thus every budget in §0).

**Practical guidance:** quantization (esp. FP8, then 4-bit) is the safest capacity win;
eviction/sparsity risks silent quality loss on recall- and reasoning-heavy tasks
("The Pitfalls of KV Cache Compression," 2025, documents failure modes). Layer these
under prefix caching + paging, not instead of them.

---

## 6. The PagedAttention critique: vAttention and contiguous virtual memory

PagedAttention won by fixing fragmentation — but it did so by making the KV cache
**physically *and* virtually non-contiguous**, and that has costs.

**vAttention** (Prabhu et al., Microsoft Research; arXiv 2405.04437, May 2024; **ASPLOS
2025**) is the sharpest critique (the "PagedAttention considered harmful" argument):

- **Kernel rewrites / portability.** Because KV blocks are scattered, *every* attention
  kernel must be rewritten to dereference the block table and gather non-contiguous
  blocks. You cannot just drop in a new FlashAttention/FlashInfer kernel — each needs a
  paged variant, which lags and fragments the ecosystem.
- **Performance overhead.** The extra indirection in the kernel slows attention by
  **>10%** in their measurements, and **block-table management adds CPU overhead** on the
  scheduling path.
- **The fix.** Keep the KV cache **virtually contiguous** (so unmodified kernels work
  out-of-the-box) but **decouple virtual from physical allocation** using **CUDA
  low-level virtual-memory management APIs** (`cuMemCreate`/`cuMemMap`-style): reserve a
  big contiguous *virtual* range per request, back it with physical pages on demand. You
  get paging's anti-fragmentation *and* contiguity.
- **Result:** **up to 1.23× throughput** vs. PagedAttention-based FlashAttention-2 /
  FlashInfer kernels, with existing kernels running unmodified.

Related lines: **vTensor** and **vAttention**-style CUDA-VMM management, and vLLM's own
tracking issue to bring VMM-based allocation into the engine (GitHub vllm #17612). The
takeaway for a from-scratch build: **paging is not the only way to beat fragmentation.**
If you can target CUDA VMM APIs, virtual-contiguity gets you kernel simplicity and
portability. But paging remains the more portable, hardware-agnostic choice today
(non-CUDA backends, TPUs via "Ragged Paged Attention," etc.), and it is what prefix
sharing/CoW are naturally expressed in. Know the trade before you commit your KV layout.

---

## 7. Where it's heading (2025–2026): cluster-scale KV

The clear trajectory: KV cache graduates from a per-GPU data structure to a
**cluster-wide, tiered, routable resource**, and the scheduler's job becomes *put the
request where its KV already is.*

- **KV-cache-aware routing.** Route each request to the replica most likely to already
  hold its prefix's KV, instead of naive round-robin. **Google's GKE Inference Gateway**
  (announced Google Cloud Next, **April 2025**) does cache-aware routing via the
  **llm-d Endpoint Picker (EPP)**. **CacheRoute** jointly scores prefix-overlap, live GPU
  memory, and predicted TTFT to pick a vLLM instance. The tension is **cache affinity vs.
  load balancing** — always routing to the cache-owner overloads it — which **DualMap**
  (2026) explicitly balances.

- **Global KV pools.** A **shared KV store spanning prefill and decode nodes** so that
  reuse is decoupled from placement (Mooncake's pool is the production existence proof;
  session-centric schedulers over a global store report ~**10–16% higher TPS**). "KV as a
  service" becomes literal infrastructure.

- **Cross-node / cross-region reuse at scale.** Work like **GORGO** (network-aware
  cross-region KV reuse), **ShadowServe** (interference-free KV fetching for distributed
  prefix caching), **Harvest** (opportunistic P2P GPU KV caching), and **KV stores scaling
  KV cache** point to KV caches shared across data centers, with the network — not HBM —
  as the new bottleneck, and compression (CacheGen) as the enabling trick.

- **Agent/conversation-aware scheduling.** As agentic workloads dominate, schedulers are
  becoming **session-** and **conversation-level** aware (SMetric, "Observation, Not
  Prediction"), keeping a conversation's growing KV pinned near its serving node across
  turns.

**Net thesis for the reader:** a competitive 2026 inference server is organized *around*
the KV cache. Start with paged blocks; add automatic prefix caching (radix tree if you
can); decide colocation-with-chunked-prefill vs. prefill-decode disaggregation
deliberately; treat KV as tierable (CPU/NVMe/remote) and compressible (FP8/4-bit first);
and, at multi-replica scale, make routing KV-aware. Every one of those is a lever on the
same object.

---

## Sources

Primary sources (papers, official blogs/repos) first; dates are publication/last-major-update.

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," **SOSP 2023** (Best Paper). https://dl.acm.org/doi/10.1145/3600006.3613165 · arXiv: https://arxiv.org/pdf/2309.06180 — 60–80%→<4% waste; 2–4× throughput; block-size 16; ~0.8MB KV/token/13B.
- "Fast and Expressive LLM Inference with RadixAttention and SGLang," LMSYS Org blog, **2024-01-17**. https://www.lmsys.org/blog/2024-01-17-sglang/ — radix tree, LRU recursive leaf eviction, cache-aware scheduling, up to 5× throughput, 9 benchmarks on Llama-7B/A10G.
- vLLM, "Automatic Prefix Caching" (design docs), accessed **2026-07**. https://docs.vllm.ai/en/stable/design/prefix_caching/
- "Inside vLLM: Anatomy of a High-Throughput LLM Inference System," vLLM blog, **2025-09-05**. https://vllm.ai/blog/2025-09-05-anatomy-of-vllm — V1 KV cache manager, default prefix caching, per-request salting, mixed prefill+decode scheduler.
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," **OSDI 2024** / arXiv 2401.09670 (**Jan 2024**). https://arxiv.org/abs/2401.09670 · PDF: https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf — 7.4× more requests / 12.6× tighter SLO.
- Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving," arXiv 2407.00079 (**Jul 2024**); **FAST '25** ("Trading More Storage for Less Computation," Best Paper). https://arxiv.org/abs/2407.00079 · https://www.usenix.org/conference/fast25/presentation/qin · Project: https://kvcache-ai.github.io/Mooncake/ — Conductor scheduler, disaggregated KV pool (CPU/DRAM/SSD), early rejection, up to 525% sim throughput, 115%/107% on A800/H800, >100B tokens/day.
- Agrawal et al., "Taming the Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," **OSDI 2024**. https://www.usenix.org/system/files/osdi24-agrawal.pdf — chunked prefill, stall-free batching (colocation alternative to disaggregation).
- LMCache team, "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference," arXiv 2510.09665 (**2025**). https://arxiv.org/pdf/2510.09665 · Repo: https://github.com/lmcache/lmcache · Docs: https://docs.lmcache.ai/developer_guide/architecture.html — tiered KV (GPU/CPU/disk/remote), 1.9–8.1× smaller TTFT, 2.3–14× throughput at equal TTFT; CacheGen compressed KV transmission.
- Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," **NeurIPS 2024** / arXiv 2401.18079. https://arxiv.org/pdf/2401.18079 · https://people.eecs.berkeley.edu/~ysshao/assets/papers/kvquant-neurips2024.pdf — 2-bit, <0.5 PPL degradation, 6.9× memory savings, ~1% outlier channels fp16.
- Liu et al., "KIVI: Plug-and-play 2bit KV Cache Quantization with Streaming Asymmetric Quantization," **2024**. https://www.researchgate.net/publication/376831635 — per-channel K, per-token V, ~3% GSM8K drop.
- Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs," **NeurIPS 2023**. https://www.researchgate.net/publication/401463261 — 20% heavy hitters, up to 29× throughput vs HF Accelerate on OPT-6.7B/30B.
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks (StreamingLLM)," **2023**. (attention sinks + sliding window.)
- Li et al., "SnapKV: LLM Knows What You are Looking For Before Generation," arXiv 2404.14469 (**2024**). https://arxiv.org/pdf/2404.14469 — observation-window voting, cluster+pool prompt-KV compression.
- "Scissorhands" (persistence-of-importance eviction), **2023**; summarized in MarkTechPost, "Top 10 KV Cache Compression Techniques," **2026-04-29**. https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/
- "The Pitfalls of KV Cache Compression," arXiv 2510.00231 (**2025**). https://arxiv.org/pdf/2510.00231 — documents eviction/quant failure modes.
- Prabhu et al., "vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention," arXiv 2405.04437 (**May 2024**); **ASPLOS 2025**. https://arxiv.org/abs/2405.04437 · MSR PDF: https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/vattention_arxiv24.pdf — critique of non-contiguous KV (>10% kernel slowdown, kernel-rewrite/portability, CPU block-table overhead), CUDA VMM APIs, up to 1.23× vs paged FA2/FlashInfer.
- vLLM issue #17612, "Implement vAttention: Virtual Memory Management for KV Cache." https://github.com/vllm-project/vllm/issues/17612
- Google GKE Inference Gateway / llm-d EPP cache-aware routing (Google Cloud Next, **April 2025**), explainer: https://www.spheron.network/blog/gke-inference-gateway-kv-cache-aware-llm-routing/
- "CacheRoute: KV-Cache-Aware Routing for LLM Backend Services on Kubernetes," **2025/26**. https://www.researchgate.net/publication/406925855
- "DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving," arXiv 2602.06502 (**2026**). https://arxiv.org/pdf/2602.06502
- "ShadowServe: Interference-Free KV Cache Fetching for Distributed Prefix Caching," arXiv 2509.16857 (**2025**). https://arxiv.org/pdf/2509.16857
- "GORGO: Maximizing KV-Cache Reuse While Minimizing Network Latency in Cross-Region LLM Load Balancing," arXiv 2602.11688 (**2026**). https://arxiv.org/html/2602.11688v1
- Supporting explainers (secondary, used for cross-checking numbers): Red Hat Developer, "How PagedAttention resolves memory waste," **2025-07-24** (https://developers.redhat.com/articles/2025/07/24/how-pagedattention-resolves-memory-waste-llm-systems); Introl blog, "KV Cache Optimization" (https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide); Spheron blogs on vLLM/NVMe offloading (2026).

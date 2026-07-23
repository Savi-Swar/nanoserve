# Systems & Scheduling: The Continuous-Batching Serving Frontier (2025–2026)

*Reference brief for building a from-scratch continuous-batching LLM server. Two goals: (1) know the real production systems to benchmark against and cite, (2) understand the scheduling ideas worth implementing. Every claim below is cross-checked against at least two sources; see the Sources section for URLs and dates.*

---

## Part A — The Production Systems Landscape

The serving landscape in 2025–2026 has converged on a shared core — **iteration-level (continuous) batching** over a **paged/blocked KV cache** — first popularized by Orca (OSDI '22) and vLLM's PagedAttention (SOSP '23). Systems now differentiate on three axes: the **scheduler** (how prefill and decode are interleaved, whether CPU overhead is hidden), the **execution backend** (eager PyTorch vs. `torch.compile`/CUDA graphs vs. a fully compiled engine), and **ease of use / model coverage** vs. peak performance. The six systems below span that space.

### vLLM (V1 engine)

vLLM is the de-facto open-source baseline and the one your server will most often be compared against. The **V1 engine rewrite**, released as alpha on **2025-01-27** and made the default in the 0.8.x series, is the single most important recent change. V1 was motivated by V0 accumulating so many features (chunked prefill, prefix caching, spec decode, disaggregation) that the scheduler and worker code became a bottleneck: input tensors and Python metadata were rebuilt every step, adding significant per-iteration CPU overhead that starved the GPU on fast models. V1's answer has four pillars. (1) A **unified scheduler** that abolishes the prefill/decode phase distinction — scheduling decisions are just a dictionary `{request_id: num_tokens}` and a fixed per-step token budget, so chunked prefill, prefix caching, and speculative decoding all fall out of one mechanism rather than special cases. (2) An **isolated `EngineCore` process** running the scheduler + model executor in a tight loop, separated by IPC from the API server, so CPU-heavy work (tokenization, multimodal preprocessing, de-tokenization, response streaming) overlaps with GPU execution. A **persistent batch** caches input tensors and applies only diffs each step, with heavy use of NumPy vectorization instead of Python loops. (3) **Near-zero-cost prefix caching** via a constant-time eviction data structure — V1 reports <1% throughput loss even at 0% hit rate, so it is on by default (V0 kept it off). (4) **`torch.compile` + piecewise CUDA graphs** (led by Kaichao You), which capture the model into replayable graphs while "cutting out" the attention op so dynamic-shape attention (varying sequence lengths, prefix cache) still works. Net reported result: **up to 1.7× higher throughput vs. V0**, larger for vision-language models. Position on the tradeoff triangle: excellent ease-of-use and model coverage, very strong throughput, competitive latency — the sensible default and the primary system to benchmark against.

### SGLang

SGLang (LMSYS/xAI, paper **arXiv:2312.07104**, Dec 2023; blog 2024-01-17) is a co-designed frontend language + runtime whose defining backend idea is **RadixAttention**: the KV cache of all in-flight and recent requests is organized as a **radix tree** keyed by token sequences, with an LRU eviction policy, so *any* shared prefix — system prompts, few-shot exemplars, multi-turn history, tree-of-thought branches — is automatically detected and reused across requests without the programmer marking it. This makes SGLang the throughput leader on prefix-heavy and agentic workloads; the paper reports up to **6.4× higher throughput** vs. prior systems. For **structured generation**, SGLang compiles JSON/regex constraints into a **compressed finite-state machine** that can emit multiple tokens per decode step when the grammar forces them, rather than one-token-at-a-time masking. The **v0.4** release (**2024-12-04**) added a **zero-overhead batch scheduler** that runs one batch ahead — it prepares the next batch's metadata (including radix-tree operations) on the CPU while the GPU computes the current batch, using "future tokens" to resolve data dependencies — reporting ~1.1× over its own prior version and ~1.3× over other baselines. v0.4 also shipped a **cache-aware load balancer** that routes across workers using an approximate per-worker radix tree to maximize cache hits (see Part B). Position: top-tier throughput especially for structured/prefix-heavy/agentic workloads; ease-of-use is good but the programming model is more opinionated than vLLM's drop-in OpenAI server.

### NVIDIA TensorRT-LLM

TensorRT-LLM is NVIDIA's performance-ceiling engine for NVIDIA GPUs. Unlike the PyTorch-based engines, it **ahead-of-time compiles** the model — weights plus architecture — through the TensorRT deep-learning compiler into an optimized engine specialized to a specific GPU SKU, batch-size range, and sequence-length range, performing kernel selection, aggressive **kernel/layer fusion**, and CUDA-graph capture. It ships hand-optimized attention kernels and first-class low-precision support (FP8, FP4, INT4 AWQ, INT8 SmoothQuant). Its continuous-batching implementation is called **in-flight batching**: new requests in the context (prefill) phase are admitted into a running batch alongside sequences in the generation (decode) phase at each iteration, over a paged KV cache. The payoff is the best raw latency/throughput on NVIDIA hardware when configured well; the cost is ease of use — the compile step, engine-per-configuration rigidity, and slower support for brand-new architectures. It is typically served behind **Triton Inference Server** (or increasingly **NVIDIA Dynamo** for multi-node/disaggregated setups). Position: highest peak performance on NVIDIA GPUs, lowest ease-of-use, NVIDIA-only.

### HuggingFace Text Generation Inference (TGI)

TGI is HuggingFace's production toolkit, notable architecturally for its **three-tier split**: a **Launcher** (process orchestration), a **Rust Router** (HTTP/gRPC front end that validates, queues, and forms continuous batches), and a **Python model Server** (inference). Putting the router and request-batching logic in **Rust** gives high-concurrency request handling with low overhead while keeping the ML code in Python; the router dynamically merges new requests into the running batch when KV memory allows and filters out completed ones. TGI leans on backend kernels (FlashAttention/PagedAttention, and optionally a TensorRT-LLM or vLLM-derived backend) rather than inventing its own. Position: solid, easy-to-deploy, tightly integrated with the HF Hub ecosystem; historically a step behind vLLM/SGLang on bleeding-edge throughput but very production-friendly.

### LMDeploy (TurboMind) and DeepSpeed-FastGen / MII

Two more worth citing. **LMDeploy** (InternLM) centers on the **TurboMind** engine, a C++/CUDA runtime whose **persistent batch** pre-configures *N* fixed batch slots that requests enter and leave as they finish, over a **blocked KV cache** managed as an LRU memory pool (indirect buffer pointers in the FMHA kernels handle the non-contiguity). It also implements dynamic split-fuse and high-performance quantized (e.g., 4-bit) inference, and is frequently the throughput leader for InternLM/Llama-family models. **DeepSpeed-FastGen** (Microsoft, paper **arXiv:2401.08671**, Jan 2024) is served via **DeepSpeed-MII** and introduces **Dynamic SplitFuse** — the same core idea as Sarathi's chunked prefill: split long prompts into chunks and *fuse* prefill chunks with ongoing decodes into uniform-sized composite batches, so no single iteration is dominated by a huge prefill. Reported: up to **2.3× higher effective throughput, 2× lower average latency, up to 3.7× lower token-level tail latency** vs. vLLM at the time. Position for both: strong throughput, more niche adoption than vLLM/SGLang.

### Comparison table

| System | Core scheduler idea | Execution backend | Signature feature | Peak throughput | Latency | Ease of use | HW |
|---|---|---|---|---|---|---|---|
| **vLLM V1** | Unified `{id:tokens}` budget; chunked prefill default; isolated EngineCore | PyTorch + `torch.compile` + piecewise CUDA graphs | PagedAttention; ~free prefix cache | Very high | Very good | **Excellent** (OpenAI-compatible) | Broad (NVIDIA/AMD/TPU/CPU) |
| **SGLang** | Zero-overhead scheduler (1 batch ahead) | PyTorch + CUDA graphs | **RadixAttention** prefix reuse; FSM structured output | **Very high** (prefix/agentic) | Very good | Good | Broad |
| **TensorRT-LLM** | In-flight batching | **AOT-compiled** TensorRT engine | Fused compiled kernels; FP8/FP4 | **Highest** (NVIDIA) | **Best** (NVIDIA) | Low (compile step) | NVIDIA only |
| **TGI** | Rust router forms batches; dynamic merge/filter | Python server + backend kernels | Rust router 3-tier arch | High | Good | Very good (HF Hub) | Broad |
| **LMDeploy** | Persistent batch, N fixed slots | **TurboMind** C++/CUDA | Blocked KV LRU pool; 4-bit | Very high | Very good | Good | NVIDIA-centric |
| **DeepSpeed-FastGen** | **Dynamic SplitFuse** (chunk + fuse) | DeepSpeed-Inference kernels | Uniform composite batches | High | Good (low tail) | Moderate (via MII) | NVIDIA-centric |

*Throughput/latency rankings are workload- and configuration-dependent; treat them as "which corner of the triangle each system optimizes," not a fixed leaderboard. Always re-benchmark on your own model + traffic.*

---

## Part B — The Scheduling & SLO Frontier

### Goodput vs. throughput, and SLO-aware serving

Raw **throughput** (tokens/sec or requests/sec) is the wrong north star for interactive serving because it rewards huge batches that wreck latency. The frontier metric is **goodput**: the rate of requests that satisfy their latency SLOs. Interactive LLM SLOs are split across the two phases: **TTFT** (time-to-first-token, dominated by prefill + queueing) and **TPOT/ITL** (time-per-output-token / inter-token latency, dominated by decode). A useful goodput definition (DistServe, OSDI '24, **arXiv:2401.09670**) is: requests/sec such that both `TTFT < t1` **and** `TPOT < t2` hold for the SLO-target fraction (e.g., P90). This matters because the two phases have opposite hardware profiles — **prefill is compute-bound and saturates the GPU; decode is memory-bandwidth-bound and under-utilizes compute** — so a batch that is great for one is bad for the other. DistServe's radical response is **prefill/decode disaggregation** (separate GPU pools), reporting up to **4.48× higher goodput** and far more stringent sustainable SLOs than vLLM on summarization; but disaggregation needs KV-cache transfer infrastructure and is a Part-C "infra" item, not a pure-Python win.

**Tuning `max_num_seqs` and `max_num_batched_tokens`.** These are the two knobs that set your latency/throughput operating point on a single replica (vLLM optimization docs, 2025). `max_num_seqs` caps concurrent running sequences; `max_num_batched_tokens` caps tokens processed per iteration (the prefill/chunk budget). Rules of thumb, verified across the vLLM docs and Red Hat's tuning guide: **raise `max_num_batched_tokens` (>8192) for throughput and better TTFT** (more prefill work per step); **lower it (e.g., 2048) for better ITL** because fewer/smaller prefills stall ongoing decodes. **Lower `max_num_seqs` to cut latency** and per-request KV pressure; raise it for throughput until the GPU saturates and latency degrades. Constraint: `max_num_batched_tokens >= max_num_seqs`. The practical method: sweep these under an open-loop load generator (below) and pick the highest arrival rate that still meets your TTFT/TPOT SLOs — that is your goodput-maximizing config.

### FCFS vs. priority; preemption strategies

The default policy in nearly every engine is **FCFS** (a FIFO queue by arrival time) — simple and starvation-free. vLLM V1 adds an optional **priority policy**: a heap ordered by `(priority, arrival_time)`, where a higher-priority waiting request can **forcibly preempt** a lower-priority running one back to the waiting queue (vLLM PR #5958 / #19057). Measured cost was **<4% slowdown** for Llama-8B with priority enabled, and zero when disabled — cheap enough to offer, useful for mixed latency-critical vs. best-effort traffic. Beware **starvation**: strict priority can indefinitely stall low-priority work, so production systems add aging or reserved capacity.

Preemption happens whenever admitted requests collectively need more KV blocks than exist. There are two recovery strategies, and knowing when each wins is a real design decision:

- **Recompute** — drop the preempted sequence's KV blocks and, on re-admission, **re-run prefill from scratch**. Cost is GPU compute; zero PCIe traffic. This is vLLM **V1's default** because with fast prefill and prefix caching, recompute overhead is low.
- **Swap** — serialize the preempted sequence's KV blocks to **CPU DRAM**, then copy them back over PCIe on re-admission. Cost is PCIe bandwidth (~32 GB/s on PCIe 4.0 x16); restoring a long-context sequence can take hundreds of ms.

The crossover follows sequence length: recompute is **O(s²)** in tokens (attention over the whole prompt) while swap is **O(s)** (a linear memory copy), so **short sequences favor recompute, long sequences favor swap** (multiple sources put the crossover in the low-thousands of tokens, e.g., ~4k). Build your scheduler so the recovery mode is a policy choice, not a hardwired assumption.

### Fairness: VTC and multi-tenancy

When many clients share one endpoint, FCFS lets a heavy client monopolize the batch. **VTC — Virtual Token Counter** (Sheng et al., UC Berkeley/Stanford/Duke; **OSDI '24**, **arXiv:2401.00588**) is the first fairness scheduler for continuous batching. It defines fairness over a **token-based cost function** (counting input and output tokens, typically with different weights since a decode token costs more GPU-time than a prefill token) and keeps a **virtual counter** per client of service received so far. At each step it **admits requests from the client(s) with the lowest counter**, incrementing counters by tokens actually processed. It is **work-conserving** — it only reorders dispatch, never rejects a fitting request or idles the GPU — and the paper proves a **2×-tight bound** on the service gap between any two continuously-backlogged clients. VTC is a strong, implementable model for multi-tenant fairness; follow-ups (e.g., locality-aware fair scheduling, arXiv:2501.14312) reconcile fairness with prefix-cache locality.

### Chunked-prefill scheduling (Sarathi-Serve)

**Sarathi-Serve** (Agrawal et al., **OSDI '24**, **arXiv:2403.02310**) is the paper to cite for the single most impactful, implementable scheduling idea. The problem: batching prefills with decodes creates **generation stalls** — a long prefill iteration blocks every decode in the batch, spiking ITL/tail latency. Sarathi's two mechanisms: **chunked-prefills** split a long prompt into near-equal token chunks processed over several iterations, and **stall-free batching** admits a new request's prefill chunk into a batch *without pausing* ongoing decodes, packing each iteration up to a fixed token budget (the chunk size). The exposed knob is exactly `max_num_batched_tokens`: a **smaller chunk → tighter ITL/TPOT bound but more prefill iterations (lower throughput)**; a **larger chunk → higher throughput but longer decode stalls**. This is the fundamental throughput/latency dial of a unified scheduler. Reported: up to **2.6× higher serving capacity** (Mistral-7B, 1×A100), **3.7×** (Yi-34B, 2×A100), **5.6×** with pipeline parallelism on Falcon-180B. vLLM V1's unified scheduler and DeepSpeed-FastGen's Dynamic SplitFuse are the same family of idea — **this is the scheduler you should build.**

### Load balancing / request routing across replicas

Above one replica, routing policy is the lever. **Random / round-robin** is the naive baseline; it ignores per-replica load and cache state. **Power-of-two-choices** (sample two replicas, send to the less loaded) is the classic cheap improvement that avoids the herd behavior of "always pick the global least-loaded" while getting most of the benefit — a good default when you lack cache visibility. But for LLMs the big win is **cache-aware routing**: because prefill is expensive and prefix caching is nearly free, route each request to the replica whose KV/prefix cache **already holds its prefix**. Implementations in 2024–2026: **SGLang's cache-aware load balancer** (approximate per-worker radix tree, lazily updated), **NVIDIA Dynamo's KV-aware router** (routes to instances holding matching KV blocks), and **llm-d's prefix-cache-aware scheduler** on the Kubernetes Gateway API Inference Extension, which **scores each replica by prefix-match length weighted against current load** — pure round-robin actively *breaks* prefix caching by scattering related requests. Reported gains reach **~2× throughput** on the same hardware. Standard load balancers are prefix-cache-blind, which is the core reason LLM serving needs its own router. The general recipe for your server: `score(replica) = α·prefix_overlap − β·current_load`, route to the argmax (add power-of-two sampling to bound router cost at scale).

### Autoscaling and its LLM-specific pitfalls

Autoscaling LLM replicas is uniquely hard for two reasons, both well-documented (Spheron KEDA/Knative guide 2026; CloudNativeNow; ScaleOps). **(1) Long cold starts.** A new pod needs **3–10 minutes** to pull a multi-GB image, load weights into VRAM, **capture CUDA graphs**, and warm caches — far longer than the traffic spike that triggered scaling, so reactive scaling arrives too late. Mitigations: over-provision headroom, pre-pull/pre-warm images, scale on a **leading indicator** (queue depth) not a lagging one. **(2) The metrics that Kubernetes ships are the wrong metrics.** Serving engines **intentionally hold GPU memory at 90%+** to maximize KV cache, so a memory-threshold HPA fires constantly and falsely; and a vLLM pod at 50 concurrent requests may show **5–8% CPU while the GPU is at 95%** and users see 30-second latency — CPU/memory HPAs are blind to the real bottleneck. The fix is to scale on **engine-native signals** exported to Prometheus: **`vllm:num_requests_waiting`** (queue depth) and **`vllm:gpu_cache_usage_perc`** (KV pressure), driven by KEDA, with node autoscaling (Karpenter/Cluster Autoscaler) underneath. **(3) KV state is ephemeral** — scaling a pod down or restarting it discards its prefix-cache/KV state, so aggressive scale-to-zero trades cost for cold prefill misery. This is infra-tier work, but your server should **export queue-depth and KV-usage metrics** so it is autoscalable at all.

### Open-loop vs. closed-loop load generation (validates your harness)

This is the most important benchmarking correctness point, and it validates building a proper load harness. A **closed-loop** generator keeps a fixed number of virtual clients, each sending its next request only *after* the previous response completes. This creates a hidden feedback loop: when the server slows down, clients automatically send *less* load, so the offered rate silently drops and **tail latency looks artificially healthy** — the system never actually gets pushed past its knee. This is a form of **coordinated omission** (the slow requests that would have arrived during a stall are simply never issued). An **open-loop** generator issues requests on an **independent arrival process** — for LLM serving, a **Poisson process** at a target rate λ — regardless of whether prior requests have finished. Poisson arrivals are the correct default because they model independent users, expose queueing behavior, and let you measure **true P95/P99 tails** and locate the goodput cliff. Practical requirements: sweep λ and report goodput vs. arrival rate (find the SLO knee); use **≥1000 samples** to estimate P99 with ~10% margin at 95% confidence; report the arrival process explicitly. Tools that get this right include **vLLM's `benchmark_serving`**, **GuideLLM** (offers a `poisson` profile), and **sglang.bench_serving**. **Design implication for nanoserve: your benchmark harness must be open-loop Poisson** — a closed-loop harness cannot measure the tails or the capacity claims you will want to publish. (Report both: closed-loop for max sustainable throughput, open-loop Poisson for SLO/tail behavior.)

---

## Part C — What to Implement in a Pure-Python Scheduler (ranked)

Ranked by learning-value-per-effort for a from-scratch server with **no custom CUDA kernels**.

**Tier 1 — Pure-Python, high learning value, do these first:**
1. **Continuous (iteration-level) batching** with a unified `{request_id: num_tokens}` token budget per step — the foundation; V1's core abstraction, all in scheduler logic.
2. **Chunked-prefill + stall-free batching (Sarathi-Serve)** with `max_num_batched_tokens` as the throughput/latency knob — highest-payoff scheduling idea, entirely a scheduling-loop concern.
3. **FCFS and priority policies** (FIFO queue vs. `(priority, arrival_time)` heap) plus **preemption with a selectable recompute-vs-swap policy** — pure bookkeeping over your block table; teaches the whole admission/eviction lifecycle.
4. **VTC fairness** (per-client virtual token counters, admit lowest-counter, work-conserving) — a self-contained, provably-bounded algorithm; ~100 lines.
5. **An open-loop Poisson benchmark harness** measuring TTFT/TPOT/ITL and P95/P99, sweeping λ to plot goodput — measurement infrastructure, not kernels; prerequisite for trusting everything else.
6. **Prometheus-style metrics** (`num_requests_waiting`, `gpu_cache_usage_perc`) — trivial to export, unlocks autoscaling and observability.

**Tier 2 — Pure-Python but needs multi-replica scaffolding:**
7. **Router with power-of-two-choices** load balancing — simple, no cache visibility needed.
8. **Prefix/cache-aware routing** (`score = α·prefix_overlap − β·load`) — needs a shared or approximate radix tree of per-replica cache state, but the routing logic is Python.
9. **Prefix caching** via a radix tree over the block table (RadixAttention-style) — the *tree bookkeeping and eviction* are pure Python; it only pays off if your attention kernel can consume non-contiguous cached blocks.

**Tier 3 — Needs kernel / infra work (cite, don't build from scratch first):**
10. **Paged/blocked KV attention** — the block table is Python, but the **attention kernel that reads paged KV** is custom CUDA (use FlashAttention/xFormers/PagedAttention rather than writing your own).
11. **`torch.compile` + piecewise CUDA graphs** — compiler/graph-capture engineering; large CPU-overhead win but not a scheduler concept.
12. **Prefill/decode disaggregation (DistServe)** and **KV-cache-transfer routing** — needs cross-GPU KV movement infrastructure.
13. **Quantized / fused compiled kernels (TensorRT-LLM-style)** and **speculative decoding kernels** — deep kernel work.

**One-line takeaway:** every idea in Part B's scheduling sections (batching, chunked prefill, priority, preemption policy, VTC fairness, open-loop benchmarking, cache-aware routing scores) is implementable in pure Python; the CUDA line is drawn at the *attention kernel* and *graph compilation*, which you should borrow, not rebuild.

---

## Executive Summary (~10 bullets)

- **Benchmark against vLLM V1 first.** Its 2025-01-27 rewrite (unified `{id:tokens}` scheduler, isolated EngineCore to hide CPU overhead, ~free prefix caching, `torch.compile` + piecewise CUDA graphs) delivers up to **1.7× over V0** and is the default open-source baseline.
- **The serving field has converged on one core** — iteration-level continuous batching over a paged KV cache — and differentiates on scheduler design, execution backend (eager vs. compiled), and ease-of-use vs. peak performance.
- **Systems to cite:** vLLM (default), **SGLang** (RadixAttention prefix reuse + zero-overhead scheduler, best for structured/agentic), **TensorRT-LLM** (AOT-compiled, highest NVIDIA ceiling, lowest ease-of-use), **TGI** (Rust-router 3-tier, HF ecosystem), **LMDeploy/TurboMind** and **DeepSpeed-FastGen** (Dynamic SplitFuse).
- **Optimize goodput, not throughput:** the sustainable request rate meeting both `TTFT` and `TPOT/ITL` SLOs. Prefill is compute-bound, decode is memory-bound — their opposite profiles are why one batch config can't win both.
- **`max_num_batched_tokens` and `max_num_seqs` are your operating-point knobs:** bigger token budget → throughput + TTFT; smaller → tighter ITL; lower `max_num_seqs` → lower latency. Tune by sweeping under load against SLOs.
- **Chunked-prefill / stall-free batching (Sarathi-Serve, OSDI '24) is the highest-value scheduling idea** — it removes decode stalls and exposes the throughput/latency dial directly; up to 2.6–5.6× serving capacity. Build this.
- **Preemption has two modes with a length-based crossover:** recompute (O(s²) compute, no PCIe — V1 default, wins short) vs. swap (O(s) PCIe copy — wins long, ~4k+ tokens). Make it a policy, not a hardwire.
- **For multi-tenant fairness use VTC (OSDI '24):** per-client virtual token counters, admit the least-served client, work-conserving, with a proven 2×-tight service-gap bound — ~100 lines of pure Python.
- **Cross-replica routing must be cache-aware:** round-robin *breaks* prefix caching; score replicas by `prefix_overlap − load` (SGLang/Dynamo/llm-d), add power-of-two-choices to bound router cost — up to ~2× throughput.
- **Autoscaling LLMs is uniquely hard:** 3–10 min cold starts and CPU/memory HPAs that are blind (GPU 95% while CPU 5–8%; engines hold memory at 90%+ by design). Scale on **queue depth** and **KV usage**, and export those metrics from your server.
- **Your benchmark harness must be open-loop Poisson.** Closed-loop generators create coordinated omission — the server slows, clients back off, tails look fake-healthy. Only independent Poisson arrivals reveal true P95/P99 and the goodput cliff.

---

## Sources

*(Accessed 2026-07-23. Dates are publication/release dates where known.)*

**vLLM V1**
- vLLM Blog, "vLLM V1: A Major Upgrade to vLLM's Core Architecture," 2025-01-27 — https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- Red Hat Developer, "vLLM V1 Alpha: A major upgrade to vLLM's core architecture," 2025-01-28 — https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture
- vLLM Docs, "vLLM V1 User Guide" — https://docs.vllm.ai/en/stable/usage/v1_guide/
- vLLM, "CUDA Graphs design doc" — https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md
- vLLM Docs, "Optimization and Tuning" (max_num_seqs / max_num_batched_tokens) — https://docs.vllm.ai/en/stable/configuration/optimization/
- Red Hat Developer, "Practical strategies for vLLM performance tuning," 2026-03-03 — https://developers.redhat.com/articles/2026/03/03/practical-strategies-vllm-performance-tuning
- vLLM PR #5958 (apatke), "Adding Priority Scheduling"; PR #19057, "Priority Scheduling in V1 Engine" — https://github.com/vllm-project/vllm/pull/5958 ; https://github.com/vllm-project/vllm/pull/19057
- DeepWiki, "Request Scheduling (vLLM)" — https://deepwiki.com/vllm-project/vllm/2.5-request-scheduling

**SGLang**
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," arXiv:2312.07104, Dec 2023 — https://arxiv.org/abs/2312.07104
- LMSYS Org, "Fast and Expressive LLM Inference with RadixAttention and SGLang," 2024-01-17 — https://lmsys.org/blog/2024-01-17-sglang/
- LMSYS Org, "SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs," 2024-12-04 — https://lmsys.org/blog/2024-12-04-sglang-v0-4/

**TensorRT-LLM**
- NVIDIA, "TensorRT-LLM Documentation / Overview" — https://nvidia.github.io/TensorRT-LLM/
- NVIDIA, "GPT Attention / in-flight batching (legacy advanced docs)" — https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/gpt-attention.md
- NVIDIA Developer, "TensorRT-LLM Now Accelerates Encoder-Decoder Models with In-Flight Batching" — https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/

**TGI**
- HuggingFace, "Text Generation Inference Architecture" — https://huggingface.co/docs/text-generation-inference/en/architecture
- HuggingFace TGI, architecture.md — https://github.com/huggingface/text-generation-inference/blob/main/docs/source/architecture.md
- DeepWiki, "TGI Router Component / Three-Tier Architecture" — https://deepwiki.com/huggingface/text-generation-inference/2.1-router-component

**LMDeploy / DeepSpeed-FastGen**
- InternLM LMDeploy, "Architecture of TurboMind" — https://lmdeploy.readthedocs.io/en/latest/inference/turbomind.html
- Holmes et al., "DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference," arXiv:2401.08671, Jan 2024 — https://arxiv.org/abs/2401.08671

**Scheduling / SLO frontier**
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI '24, arXiv:2403.02310 — https://arxiv.org/abs/2403.02310 ; USENIX — https://www.usenix.org/conference/osdi24/presentation/agrawal
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving," OSDI '24, arXiv:2401.09670 — https://arxiv.org/abs/2401.09670 ; https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf
- Sheng et al., "Fairness in Serving Large Language Models" (VTC), OSDI '24, arXiv:2401.00588 — https://arxiv.org/abs/2401.00588
- "Locality-aware Fair Scheduling in LLM Serving," arXiv:2501.14312, Jan 2025 — https://arxiv.org/abs/2501.14312
- vLLM Docs / research on preemption recompute vs. swap; FastSwitch, arXiv:2411.18424 — https://arxiv.org/pdf/2411.18424

**Routing / load balancing**
- NVIDIA Dynamo, "KV-Cache-Aware Routing (Router Guide)" — https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing
- llm-d, "KV-Cache Wins You Can See: From Prefix Caching in vLLM to Distributed Scheduling with llm-d" — https://llm-d.ai/blog/kvcache-wins-you-can-see
- llm-d, "v0.5: Sustaining Performance at Scale" — https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale
- TrueFoundry, "KV Cache Routing: Why Standard Load Balancers Break Prefix Caching" — https://www.truefoundry.com/blog/kv-cache-routing-why-standard-load-balancers-break-prefix-caching-and-how-to-fix-it

**Autoscaling**
- Spheron Blog, "GPU Inference Autoscaling with KEDA and Knative on Kubernetes: Cold-Start and Scale-to-Zero for LLM Serving (2026)" — https://www.spheron.network/blog/keda-knative-gpu-autoscaling-kubernetes-llm-cold-start/
- Cloud Native Now, "The Inference Bottleneck: Architecting Kubernetes Autoscaling for Production LLMs" — https://cloudnativenow.com/contributed-content/the-inference-bottleneck-architecting-kubernetes-autoscaling-for-production-llms/
- ScaleOps, "vLLM on Kubernetes: Deploy, Scale, and Monitor LLM Inference" — https://scaleops.com/blog/vllm-kubernetes/

**Benchmarking / load generation**
- GuideLLM (load profiles incl. Poisson) — https://medium.com/@jajodia.nirjhar/exploring-guidellm-benchmarking-a-live-llm-on-openshift-ccc2d0841794
- BentoML, "LLM performance benchmarks" (open-loop vs closed-loop, TTFT/ITL/tails) — https://bentoml.com/llm/inference-optimization/llm-performance-benchmarks

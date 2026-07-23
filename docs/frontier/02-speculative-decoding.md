# Speculative Decoding and Its Frontier

*A deep dive for building a from-scratch inference server. Last updated: 2026-07-23.*

Speculative decoding is the dominant technique for cutting **decode latency without changing the output distribution**. This document explains the mechanism, the exact-distribution guarantee, the major families of methods with measured speedups, the tree-verification machinery, and — critically for anyone building a serving system — how it interacts with continuous batching and when it helps versus hurts.

---

## 1. The core problem

Autoregressive decoding is sequential: token *t+1* needs token *t*. Each step requires streaming the **entire model's weights** from HBM into the compute units to produce **one** token. At batch size 1 (and small batches), this makes decoding overwhelmingly **memory-bandwidth-bound**: the GPU's arithmetic units sit mostly idle while weights are shuffled. A 70B model in fp16 is ~140 GB of weights; on an A100 (~2 TB/s) that is a hard floor of ~70 ms per token *from bandwidth alone*, regardless of how few FLOPs a single-token forward actually needs.

The key insight: **a forward pass that scores N tokens in parallel costs almost the same wall-clock time as a forward pass that scores 1 token**, because both are dominated by the weight-load, not the arithmetic. If you can cheaply *guess* the next several tokens and then verify all of them in a single parallel forward pass, you amortize one expensive weight-load across many tokens. That is speculative decoding.

---

## 2. The original idea: draft-then-verify (2022–2023)

Two groups published essentially the same algorithm within months:

- **Leviathan, Kalman, Matias (Google)** — *"Fast Inference from Transformers via Speculative Decoding."* arXiv:2211.17192, first posted **Nov 2022**, ICML 2023. Measured **2×–3× speedup** on T5-XXL (11B) versus the standard T5X implementation, **with identical outputs**.
- **Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper (DeepMind)** — *"Accelerating Large Language Model Decoding with Speculative Sampling."* arXiv:2302.01318, posted **Feb 2, 2023**. Measured **2×–2.5× speedup** decoding **Chinchilla 70B** in a distributed setup, "without compromising the sample quality or making modifications to the model itself."

### 2.1 The mechanism

You have two models:
- **Target model** *M_p* (large, the one whose distribution you want), producing distribution `p(x)`.
- **Draft/approximation model** *M_q* (small and cheap), producing `q(x)`.

Each speculative iteration:
1. **Draft.** Run *M_q* autoregressively for **γ** steps (γ typically 3–7), producing candidate tokens `x_1 … x_γ` and their draft probabilities `q(x_i | prefix)`.
2. **Verify (one parallel forward pass).** Run *M_p* **once** over the whole block `prefix, x_1 … x_γ`. Because attention over a fixed prefix can be batched, this single pass yields the target probabilities `p(x_i | prefix, x_{<i})` for **all** γ+1 positions at once — for roughly the cost of one normal decode step.
3. **Accept / reject** each drafted token left-to-right using the rule below.
4. **Correct.** On the first rejection, resample that one position from an adjusted distribution; if *all* γ were accepted, sample one *bonus* token from `p` at position γ+1 (that distribution is already available from the same forward pass).

So one target forward pass yields **between 1 and γ+1 tokens**. If the draft is good, you routinely emit 3–5 tokens per expensive pass.

### 2.2 Why the output distribution is preserved exactly (speculative sampling / modified rejection sampling)

This is the load-bearing guarantee. For each drafted token `x` at a position:

- Accept `x` with probability `min(1, p(x) / q(x))`.
- If rejected, sample a replacement from the **residual distribution** `p'(x) = norm(max(0, p(x) − q(x)))`.

Standard rejection-sampling algebra shows the token finally emitted at that position is distributed **exactly** as `p(x)` — the target's true distribution — for *any* draft `q`, even a bad one. Intuition:
- Where the draft over-samples a token (`q(x) > p(x)`), it gets probabilistically rejected down to `p(x)`.
- Where the draft under-covers (`p(x) > q(x)`), the residual distribution adds back exactly the missing mass.

Two consequences a server-builder must internalize:
1. **Draft quality affects only speed, never correctness.** A worse draft model → more rejections → smaller speedup, but the emitted text is distributed identically to plain target-model sampling (within floating-point numerics — DeepMind's paper is careful to say "within hardware numerics").
2. **Greedy (temperature 0) is the special case** where accept-if-`argmax`-matches; the same framework covers all temperatures and top-p/top-k, because those just reshape `p` and `q`.

### 2.3 The speedup math

Let **α** = the expected per-token acceptance probability (how often the target accepts a drafted token). The expected number of tokens produced per target forward pass is a geometric-series expression; for the accept-until-first-reject scheme it is:

```
E[tokens per target pass] = (1 − α^(γ+1)) / (1 − α)
```

The **wall-clock speedup** also depends on **c** = cost ratio (draft-step time / target-step time). Roughly:

```
speedup ≈ E[tokens per pass] / (1 + c·γ)
```

Takeaways:
- Higher **α** (better draft alignment) → more accepted tokens → more speedup. This is why the entire research frontier is a race to raise acceptance.
- The draft must be **much cheaper** than the target (small **c**), or its serial cost eats the gains. A draft that is 1/20th the target's cost is a common target.
- There is an **optimal γ**: too small wastes verification headroom; too large wastes draft compute on tokens that will be rejected once an early token misses (everything after the first rejection is thrown away). Leviathan derives the optimal γ from α and c; production systems increasingly tune γ **dynamically**.

**Acceptance rate / "acceptance length" (τ)** is the headline metric in every paper: τ = average number of tokens accepted (including the bonus token) per target forward pass. Plain draft-model SD gets τ ≈ 2–3; the strongest 2025 methods reach τ ≈ 4–7.

---

## 3. Draft-model speculative decoding: choosing the drafter

The classic setup uses a **separate small model from the same family/tokenizer** (e.g., Llama-68M/1B drafting for Llama-70B). Design tradeoffs:

- **Alignment vs. cost.** A bigger draft aligns better (higher α) but costs more per draft step (higher c). The sweet spot is a draft ~10–20× cheaper than the target that still tracks it well. Same tokenizer/vocabulary is effectively mandatory for the accept/reject math to be simple.
- **Memory.** The draft's weights and its **own KV cache** must be co-resident with the target's. vLLM's implementation runs a separate "draft runner" and had to extend the KV-cache/memory manager to hold KV for **both** models simultaneously — real memory pressure that competes with the KV cache you'd otherwise spend on more concurrent requests.
- **Distribution shift.** A draft trained on different data drifts from the target on domain-specific inputs, tanking α. Distillation of the draft on the target's outputs (online or offline) is a common fix.
- **Availability.** Often no good small model exists in the family, which is exactly what motivated the *self-speculative* methods below (Medusa, EAGLE) that graft a cheap drafter onto the target itself.

Typical measured range for draft-model SD: **~1.5×–2.5×** end-to-end at batch 1. vLLM's own blog reports **up to 1.5×** with a draft model for Llama3-70B on ShareGPT at QPS=1.

---

## 4. Self-speculative decoding

The idea: avoid a second model entirely. Reuse the target's own representations to produce drafts cheaply.

### 4.1 Medusa (Jan 2024)

*"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads."* Cai, Li, et al., arXiv:2401.10774 (**Jan 19, 2024**), ICML 2024. Repo: FasterDecoding/Medusa.

**Mechanism.** Freeze the target. Bolt on **K extra lightweight "Medusa heads"** (each a small feed-forward layer on top of the target's last hidden state) that predict tokens at offsets +1, +2, …, +K **in parallel** from a single forward pass. Because the heads fire simultaneously (not autoregressively), each is independently uncertain, so instead of committing to one continuation Medusa takes the **top-k candidates per head** and forms their **Cartesian product into a token tree**, then verifies all branches in one pass via **tree attention** (Section 6). A **"typical acceptance" scheme** (accept tokens above a probability threshold rather than exact rejection sampling) boosts acceptance at nonzero temperature.

**Two training regimes:**
- **Medusa-1**: train only the heads, frozen backbone → **~2.2×** speedup, distribution preserved.
- **Medusa-2**: jointly fine-tune heads + backbone → **~2.3×–3.6×** (paper headline; ~2.8× typical on Vicuna/Zephyr, MT-Bench/AlpacaEval). Note Medusa-2 changes the backbone, so it's not strictly lossless vs. the original model.

Medusa's limitation: independent heads ignore correlations between successive drafted tokens (head +2 doesn't see head +1's choice), capping acceptance. This is precisely what EAGLE fixes. (Hydra, arXiv:2402.05109, makes the heads sequentially dependent as an intermediate step.)

### 4.2 EAGLE / EAGLE-2 / EAGLE-3 — currently among the strongest

The EAGLE line (SafeAILab / Peking Univ. + Microsoft) is the state of the art for open-source serving as of 2025–2026, and is the default recommendation in SGLang.

**EAGLE-1** — *"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty."* arXiv:2401.15077 (**Jan 26, 2024**), ICML 2024.
- **Key idea: autoregress at the *feature* level, not the token level.** The drafter is a single lightweight transformer decoder layer that predicts the target's **second-to-top-layer hidden feature** for the next position, then reuses the *target's own LM head* to turn features into tokens. Feature-space is smoother/more predictable than token-space.
- **Resolving feature uncertainty.** A feature alone can't determine the next feature (the next *token* sampled matters). EAGLE feeds the **already-sampled token embedding, shifted one step**, into the drafter alongside the feature — collapsing the uncertainty and enabling accurate feature prediction with tiny overhead.
- Results (MT-bench): **~3× faster** than vanilla, **2× faster than Lookahead, 1.6× faster than Medusa**; provably preserves the distribution (uses exact speculative sampling). Average acceptance length τ ≈ 3.8–4.

**EAGLE-2** — *"EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees."* arXiv:2406.16858 (**Jun 24, 2024**), EMNLP 2024.
- **Insight:** acceptance is **context-dependent**, and the drafter's confidence is well-calibrated (confidence ≈ acceptance probability). So instead of EAGLE-1's *static* draft tree, build a **context-aware dynamic draft tree**: expand tree nodes greedily by confidence, spending draft budget where acceptance is likely.
- Results: **3.05×–4.26×** over vanilla; **20%–40% faster than EAGLE-1**. Lossless. τ climbs to ~4.5–5.

**EAGLE-3** — *"EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test."* arXiv:2503.01840 (**Mar 3, 2025**, rev. Apr 2025), NeurIPS 2025.
- **Two changes that break EAGLE's scaling ceiling:**
  1. **Drop feature-prediction; predict tokens directly.** EAGLE-1/2 were constrained by having to reconstruct top-layer features (an implicit distillation loss that stopped improving as training data grew). EAGLE-3 removes it.
  2. **Multi-layer feature fusion.** Instead of only the second-to-top feature, fuse representations from **low (syntax/morphology), middle (semantics), and high (output distribution) layers** of the target as the drafter's input.
  3. **"Training-time test":** simulate the multi-step draft-and-accept process *during training* so the drafter is trained on its own multi-step rollouts, eliminating train/inference mismatch that otherwise degrades acceptance as draft length grows.
- Results: **4.1×–6.5× speedup at temperature 0** on academic benchmarks; up to **6.47×** single-task (HumanEval, Vicuna-13B); ~**5.5×** five-task mean (Vicuna-13B greedy) vs. ~1.93× for plain speculative sampling; ~**1.4× better than EAGLE-2**. Average acceptance length τ ≈ **4.25–6.5** depending on model/temperature (e.g., LLaMA-3.1-8B τ ≈ 4.5 at temp 0). Critically, **τ stays roughly flat as prompt grows 1K→32K tokens** (≈2.63→2.64 in one reported long-context setting), so the speedup doesn't decay on long context.
- **Batch behavior (the important caveat):** the per-request speedup is huge at batch 1, but the *throughput* gain shrinks as batch grows. In SGLang at **batch 64, EAGLE-3 gives ~1.38× throughput** over baseline; ~1.42× at batch 24; ~1.01× (break-even) by batch 56 on some vLLM configs. This is the general SD-vs-batching tension (Section 7), not an EAGLE-specific flaw.

**Why EAGLE is strong:** feature-level (or fused-feature) autoregression makes a *tiny* drafter that tracks the target far better than an independent-head scheme or a generic small model, so α (and thus τ) is high while c stays low.

### 4.3 Layer-skip self-speculation

Another self-speculative family drafts by running the target model **partially** (skip late layers / early-exit) and verifies with the full model. Examples: **Draft & Verify** (Zhang et al., 2023) and **LayerSkip** (Meta, 2024). No extra weights, but acceptance depends on how well the shallow sub-network mimics the full model; speedups are typically more modest (~1.2×–2×).

---

## 5. Model-free and retrieval-based drafters

When you don't want to train anything, generate drafts from **text you already have**:

- **Prompt Lookup Decoding (PLD) / prompt-lookup** — Apoorv Saxena, **Nov 2023** (repo apoorvumang/prompt-lookup-decoding; upstreamed into HF Transformers and vLLM). No model at all: match the **last few generated tokens against n-grams in the prompt**, and propose the continuation that followed that n-gram in the prompt. Astonishingly effective when output copies input — **2×–4×** on summarization, QA, RAG, code-editing, agentic "repeat the context" tasks. Useless when output doesn't overlap input. This is the single cheapest thing to implement and often the best first feature to ship.

- **Lookahead Decoding (Jacobi)** — Fu, Bailis, Stoica, Zhang; LMSYS blog **Nov 2023**, arXiv:2402.02057, ICML 2024. Repo hao-ai-lab/LookaheadDecoding. Uses **Jacobi iteration** (solve all future positions in parallel as a fixed-point problem) to generate n-grams, caches good n-grams from the Jacobi trajectory, and verifies them. **No draft model, no extra training, lossless.** Speedup **1.5×–2.3×**; it trades extra FLOPs per step for fewer steps (decoding steps drop ~linearly with log(FLOPs/step)). Plain Jacobi alone barely helps because tokens land in wrong positions; the n-gram caching is what makes it work.

- **REST: Retrieval-Based Speculative Decoding** — He et al., arXiv:2311.08252 (**Nov 2023**), NAACL 2024. Repo FasterDecoding/REST. Replace the draft model with a **datastore**: retrieve documents whose text matches the **longest suffix** of the current context, build a **Trie** of their continuations, prune low-frequency branches, and verify the candidate subtree with tree attention. Plug-and-play on any model, no training. **1.62×–2.36×** on 7B/13B for code/text. Quality of speedup scales with datastore relevance.

- **N-gram / Suffix decoding** — generalizations of the above (match against generated history + prompt). Shipped in vLLM as "N-Gram Matching" and "Suffix Decoding," among the best-supported methods in vLLM 0.12 (Dec 2025).

---

## 6. Tree attention / token-tree verification — the enabling machinery

Everything above (Medusa, EAGLE-2/3, SpecInfer, REST, Lookahead) verifies **many candidate continuations at once**, not a single linear draft. The mechanism is **tree attention** with a custom attention mask.

**SpecInfer** — Miao, Oliaro, et al., arXiv:2305.09781 (**May 2023**), ASPLOS 2024 — formalized **token-tree verification**. Instead of one draft sequence, the drafter(s) produce a **token tree**: each node is a candidate token, each root-to-node path is a candidate continuation. Multiple small "boosted" draft models cover more of the target's distribution. SpecInfer reported **1.5×–2.8×** (single/multi-GPU) and up to **2.6×–3.5×** for offloading-based inference.

**How verification of a whole tree happens in one forward pass:**
1. **Flatten** the tree's nodes into a single sequence fed to the target.
2. **Build a tree/causal attention mask** so each node attends **only to its ancestors** (its own path back to the root), not to siblings or cousins in other branches. This makes the single forward pass compute, for every node simultaneously, `p(node | its own path)` — exactly what you'd get if you ran each branch separately, but batched.
3. **Position IDs** are set by tree depth (not flat index) so RoPE/positional encodings are correct per branch.
4. **Accept the best verified path**: walk from the root, accept the longest prefix whose tokens pass the accept rule; the tree lets you "win" whichever branch happens to match the target, dramatically raising the expected accepted length versus a single linear guess.

Why it matters for a server: one target forward pass now amortizes over an entire *fan-out* of guesses. Cost grows with the number of tree nodes (more FLOPs, more KV), so **tree size is a tunable**: bigger trees raise α-per-step but cost more compute — great at batch 1 (compute is free), bad at large batch (compute is the bottleneck). Dynamic trees (EAGLE-2) size the tree to context.

---

## 7. Interaction with continuous batching (the non-trivial part)

This is where naive intuition fails and why big serving systems were **slow to adopt** SD.

**The regime flip.** SD's entire benefit rests on decoding being **memory-bound**, so that the extra verification FLOPs are "free." But **continuous batching already fixes the memory-bound problem** by a different route: batching many requests together reuses each weight-load across many sequences, pushing the GPU toward **compute-bound**. In a compute-bound regime, there is no idle arithmetic to soak up — the extra draft passes and the (γ+1)× or tree-sized verification tokens now **compete for the compute you're already saturating**. SD can then *slow you down*.

Concretely, from the vLLM spec-decode blog (**Oct 17, 2024**):
- **Low QPS (batch ~1):** up to **1.5×** (draft model, Llama3-70B/ShareGPT) and up to **2.8×** (prompt-lookup, CNN/DailyMail summarization).
- **High QPS (large batch):** **1.4× *slowdown*** on ShareGPT and **1.8× slowdown** on CNN/DailyMail. Speculation actively hurts.

**Why systems were slow to adopt it:**
- It complicates the scheduler: the engine must handle **variable numbers of tokens accepted per request per step** (0…γ+1), so sequences advance by different amounts each iteration — awkward inside a continuous-batching loop that assumed one token per sequence per step.
- It needs **KV management for two models** (draft + target) or for **tree-shaped** KV (branches that get pruned).
- Verification with a **tree mask** requires custom attention kernels that also have to work under paged KV cache (vLLM's PagedAttention).
- For a heavily-loaded server, the operator's objective is usually **throughput** (tokens/sec/GPU), and naive SD trades throughput for latency — the wrong trade under load.

**How vLLM / TensorRT-LLM / SGLang handle it now:**
- **vLLM** integrated SD into continuous batching via a **draft runner + target runner** and an extended scheduler/memory manager (2024). It supports draft-model, prompt-lookup/n-gram, EAGLE/EAGLE-2/EAGLE-3, and Medusa/MLP-speculator style heads. By vLLM 0.12 (Dec 2025), **n-gram, suffix, and EAGLE** paths are the best-supported. The roadmap (and increasingly the practice) is **dynamic speculative decoding**: automatically **shrink proposal length (γ) / tree size as batch/QPS rises**, and grow it when acceptance is high and load is low — so SD gates itself off precisely when it would hurt.
- **TensorRT-LLM** compiles the whole loop — logit prediction, draft-token acceptance, and next-draft generation — **inside the TRT engine**. Supports draft-model, **Medusa, EAGLE-1/EAGLE-2** (in-engine), **EAGLE-3** (PyTorch backend, incl. disaggregated two-model serving), and **Lookahead**. Baseten and others report production 2–3× at low concurrency.
- **SGLang** offers EAGLE-2/EAGLE-3, MTP, draft-model, and n-gram; **EAGLE-3 is the recommended default** for speed/quality. SpecForge (LMSYS, Jul 2025) is a training framework specifically for producing EAGLE-3 drafters. EAGLE-3 was merged into vLLM, SGLang, and TensorRT-LLM main by early 2026.

**Practical rule for a serving system:** treat γ/tree-size as a **load-adaptive knob**. Use aggressive speculation for latency-sensitive, low-concurrency traffic (chat with few users, single-user local, agentic step-by-step); back off toward γ=0 as the batch saturates compute. Several 2025 papers (AdaSpec, AdaServe, SmartSpec/dynamic SD, "Batch Speculative Decoding Done Right" arXiv:2510.22876) formalize exactly this SLO-aware auto-tuning.

---

## 8. When it helps vs. when it hurts — the decision table

| Regime | Memory- vs compute-bound | Speculative decoding? |
|---|---|---|
| **Batch 1 / single user / local** | Strongly memory-bound | **Big win** (2×–6× with EAGLE-3). Ship it. |
| **Low concurrency, latency-SLO serving** | Memory-bound | **Win** (1.5×–3×). Use dynamic γ. |
| **High concurrency / throughput-max serving** | Compute-bound | **Often a loss** unless γ/tree shrinks adaptively; may need to disable. |
| **Long-context prefill-heavy** | Prefill already compute-bound; decode still memory-bound | Helps the **decode** phase; EAGLE-3's τ is roughly context-length-stable. See MagicDec / OWL for long-context-specific gains. |
| **Input-grounded (summarize/RAG/code-edit)** | Memory-bound decode | **Prompt-lookup shines (2×–4×)** for near-zero cost. |
| **High acceptance domain (structured/code)** | — | Higher α → bigger win; SD loves predictable text. |
| **High-temperature / very diverse sampling** | — | Lower α → smaller win (draft agrees less often). |

The governing variables: **(a) batch size / GPU compute utilization** (the single biggest factor — determines whether verification FLOPs are free), **(b) acceptance rate α** (draft alignment × domain predictability), and **(c) draft cost ratio c**.

---

## 9. The frontier (2025–2026)

Active research directions, with representative work:

1. **Native multi-token prediction (MTP) as a first-class drafter.** DeepSeek-V3 (Dec 2024) trained with an MTP objective; those auxiliary heads double as **built-in draft heads** — no separate model. NVIDIA **Nemotron-3** (2025–2026) shares parameters across MTP heads to remove the "fixed offset" limitation, so one head can draft variable lengths without train/inference mismatch. Trend: models **ship pre-trained to be their own drafter**.

2. **Scaling the drafter without a scaling ceiling.** EAGLE-3's "training-time test" + direct-token prediction showed drafter quality keeps improving with more training data. Follow-ons (SpecForge training framework; JetSpec parallel tree drafting; GRIFFIN token alignment; LK-Losses / direct acceptance-rate optimization, 2026) push acceptance further.

3. **Batch/serving-aware speculation.** The central unsolved systems problem: **keep speedups at high concurrency.** MagicDec and SSSD ("Simply-Scalable Speculative Decoding") argue SD *can* help even large batch/long context if you pick draft length by a proper latency model; AdaSpec/AdaServe/SmartSpec do SLO-aware dynamic γ; "Batch Speculative Decoding Done Right" (Oct 2025) fixes correctness/efficiency bugs in batched SD. Interpretable latency models for SD in serving (2025–2026) aim to *predict* the optimal γ per step.

4. **Long-context speculative decoding.** OWL (arXiv:2510.07535) tackles window-length dependence; LongSpec (arXiv:2502.17421) does lossless long-context drafting/verification. Goal: keep α high when the KV cache is huge and drafting a token is relatively cheaper than reading the cache.

5. **Reasoning models & RL rollouts.** Long chain-of-thought generations are decode-dominated and thus ideal SD targets. Reported: EAGLE-3 cuts **RL rollout generation latency ~1.5×–1.8×** and overall RL step time up to ~1.41× on 8B reasoning workloads (system-integrated SD for RL post-training, 2026). "Lookahead Reasoning" (arXiv:2506.19830) speculates at the *reasoning-step* level, not just token level.

6. **MoE-specific and hardware-specific SD.** MoESD explores SD for sparse MoE (where only some experts load, changing the memory/compute balance). Vendor work (AMD Instinct EAGLE-3 training/serving via vLLM + Quark, vLLM blog Jul 2026) shows SD moving onto non-NVIDIA accelerators.

7. **Retrieval + speculation hybrids.** LogitSpec (next-next-token retrieval speculation), RACER, SENSE (semantic-embedding retrieval), CREST (compacted REST datastore) — combining retrieval drafts with model drafts to raise α cheaply.

8. **Semi-autoregressive / diffusion drafters.** Block-diffusion draft trees and dual-diffusion draft models (2025–2026) explore non-autoregressive drafting to propose many tokens at once.

The unifying theme: the field has largely **won the correctness/latency battle at batch 1** (EAGLE-3-class methods give 4–6×) and has moved to the harder problem — **preserving those gains under production-scale batching, long context, and reasoning-length outputs**, ideally with the drafter baked into the base model at pretraining time.

---

## 10. If you implement a basic version (minimal path)

1. **Start with prompt-lookup decoding.** No model, ~50 lines: match the last k generated tokens against prompt n-grams, propose the follow-on, verify. Immediate 2×–4× on input-grounded tasks, and it exercises your whole draft-then-verify plumbing.
2. **Then draft-model SD** with the exact rejection-sampling rule (Section 2.2) to get the lossless guarantee right and testable (assert output distribution matches plain sampling under a fixed seed / KL check).
3. **Add tree attention** (Section 6) once linear SD works — this is the biggest single lever on α.
4. **Make γ/tree-size load-adaptive** before you ever run it under real concurrency, or SD will regress your throughput (Section 7).
5. **Graduate to an EAGLE-style feature drafter** if you control training — it's the current best acceptance-per-cost.

---

## Sources

Primary papers:
- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, arXiv:2211.17192 (Nov 2022; ICML 2023). https://arxiv.org/abs/2211.17192 — HF page: https://huggingface.co/papers/2211.17192
- Chen et al. (DeepMind), *Accelerating LLM Decoding with Speculative Sampling*, arXiv:2302.01318 (Feb 2, 2023). https://arxiv.org/pdf/2302.01318 — coverage: https://syncedreview.com/2023/02/09/deepminds-speculative-sampling-achieves-2-2-5x-decoding-speedups-in-large-language-models/
- Cai, Li, et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, arXiv:2401.10774 (Jan 19, 2024; ICML 2024). https://arxiv.org/pdf/2401.10774 — repo: https://github.com/FasterDecoding/Medusa
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, arXiv:2401.15077 (Jan 26, 2024; ICML 2024). https://arxiv.org/abs/2401.15077
- Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*, arXiv:2406.16858 (Jun 24, 2024; EMNLP 2024). https://arxiv.org/pdf/2406.16858
- Li et al., *EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test*, arXiv:2503.01840 (Mar 3, 2025; NeurIPS 2025). https://arxiv.org/abs/2503.01840 — topic page: https://www.emergentmind.com/topics/eagle-3
- Miao, Oliaro, et al., *SpecInfer: Accelerating Generative LLM Serving with Tree-based Speculative Inference and Verification*, arXiv:2305.09781 (May 2023; ASPLOS 2024). https://arxiv.org/abs/2305.09781
- Fu, Bailis, Stoica, Zhang, *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding*, arXiv:2402.02057 (ICML 2024); LMSYS blog (Nov 21, 2023). https://arxiv.org/pdf/2402.02057 — https://lmsys.org/blog/2023-11-21-lookahead-decoding/ — repo: https://github.com/hao-ai-lab/LookaheadDecoding
- He et al., *REST: Retrieval-Based Speculative Decoding*, arXiv:2311.08252 (Nov 2023; NAACL 2024). https://arxiv.org/abs/2311.08252 — repo: https://github.com/FasterDecoding/REST
- Saxena, *Prompt Lookup Decoding* (Nov 2023). https://github.com/apoorvumang/prompt-lookup-decoding
- Hydra (sequentially-dependent Medusa heads), arXiv:2402.05109. https://arxiv.org/pdf/2402.05109

Systems / serving:
- vLLM, *How Speculative Decoding Boosts vLLM Performance by up to 2.8x* (Oct 17, 2024). https://blog.vllm.ai/2024/10/17/spec-decode.html (redirects to https://vllm.ai/blog/2024-10-17-spec-decode)
- vLLM, *2024 Retrospective and 2025 Vision* (Jan 10, 2025). https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision
- vLLM Blog, *EAGLE-3 Speculative Decoding on AMD Instinct GPUs* (Jul 13, 2026). https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct
- Red Hat Developer, *Fly EAGLE(3) fly: faster inference with vLLM & speculative decoding* (Jul 1, 2025). https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding
- TensorRT-LLM, *Speculative Sampling* docs. https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html — Lookahead example: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/lookahead/README.md
- Baseten, *How we built production-ready speculative decoding with TensorRT-LLM*. https://www.baseten.co/blog/how-we-built-production-ready-speculative-decoding-with-tensorrt-llm/
- SGLang, *Speculative Decoding* docs. https://docs.sglang.ai/advanced_features/speculative_decoding.html
- LMSYS, *SpecForge: Accelerating Speculative Decoding Training for SGLang* (Jul 25, 2025). https://www.lmsys.org/blog/2025-07-25-spec-forge/
- NVIDIA, *An Introduction to Speculative Decoding for Reducing Latency in AI Inference*. https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/

Frontier (2025–2026):
- DeepSeek-V3 Technical Report (MTP), arXiv:2412.19437. https://arxiv.org/html/2412.19437v1
- NVIDIA Nemotron-3 (shared-parameter MTP heads), arXiv:2512.20856. https://arxiv.org/pdf/2512.20856
- *Batch Speculative Decoding Done Right*, arXiv:2510.22876. https://arxiv.org/pdf/2510.22876
- *SSSD: Simply-Scalable Speculative Decoding*, arXiv:2411.05894. https://arxiv.org/html/2411.05894v3
- AdaSpec, arXiv:2503.05096; AdaServe, arXiv:2501.12162; SPIRe, arXiv:2504.06419
- OWL (long-context SD), arXiv:2510.07535; LongSpec, arXiv:2502.17421
- *Scaling Speculative Decoding with Lookahead Reasoning*, arXiv:2506.19830. https://arxiv.org/pdf/2506.19830
- MoESD (SD for MoE), arXiv:2505.19645
- *A Survey of Speculative Decoding*, arXiv:2411.13157. https://arxiv.org/pdf/2411.13157

*Dates reflect first arXiv posting / publication where known; several 2026-dated blog and arXiv items were surfaced via search in July 2026 and are noted as such.*

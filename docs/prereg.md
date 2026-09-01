# PREREGISTRATION: a $30 audit of nanoserve vs vLLM (draft)

Status: DRAFT. This file becomes PREREGISTRATION.md in the nanoserve repo. It is
committed and its hash made public (tag + GitHub issue timestamp) BEFORE any paid
GPU is rented. After that commit, nothing in sections 1 through 6 changes; every
correction goes in the amendment log (section 7) with a date and a reason.
Items marked TODO-measure are constants that require a calibration run on the
actual rented card; the calibration protocol is fixed here, only the measured
value gets filled in, and it is logged as amendment A1 when it lands.
Items marked TODO-pin are version hashes frozen at the pre-rental commit.

Order of this document is the order of the work: predictions first, measurement
plan second, analysis rules third.

## 1. Study question and scope

Question: how much of a production inference server's performance does a
from-scratch scheduler-plus-Triton-kernel server capture, and can a cost model
committed before measurement predict both servers' behavior within stated
tolerance?

Systems:
- nanoserve at commit TODO-pin (continuous_fused engine, paged_fused where the
  workload is memory-pressured).
- vLLM at one pinned version, TODO-pin (wheel hash + docker digest), run at
  three configs (the flag ladder, section 4.5). Optional: SGLang default config
  as a third engine, 4090 only, one config.

Hardware:
- Kaggle T4 16GB (free): dev, client validation, one headline continuity point
  with Qwen2.5-0.5B against the existing repo numbers.
- RTX 4090 24GB (Vast.ai or Salad): the main study card.
- A100 40GB (Vast.ai or RunPod): the scale check, reduced grid.

Models (one family so architecture constants reuse):
- Qwen2.5-0.5B fp16 (T4 continuity point).
- Qwen2.5-7B fp16 (4090 main study and A100 scale check). 15.2 GB weights fits
  24GB with about 8 GB KV headroom, roughly 140K tokens of KV at 57 KB/token.
  No quantization anywhere; it would break the cost model.

Workloads (frozen with seeds at the prereg commit):
1. ShareGPT, 500 prompts, sampling seed 20270301, output lengths replayed with
   ignore_eos (avg roughly 202 in / 179 out, matching the vLLM v0.6.0 blog shape).
2. Synthetic prefill-heavy (about 462 in / 16 out) and decode-heavy (about 462
   in / 256 out), same shapes as the vLLM blog so readers can calibrate.
3. A 30-minute BurstGPT slice, trace arrival timestamps rescaled to 3 load
   levels (0.5x, 1.0x, 1.5x of predicted saturation), slice offset frozen at
   the prereg commit.

## 2. Cost model

The model is bench/roofline.py's, with one change learned from Phase 1: the
repo's T4 crossover test showed the roofline misses absolutes by up to 10x when
it assumes spec bandwidth and free launches (predicted B*=39, measured knee B=4
on the SDPA path), while ratio predictions (spec-decode crossover) hit to the
batch. So every absolute prediction here uses MEASURED bandwidth and a measured
overhead term, calibrated per card before the study runs, and the
nanoserve-vs-vLLM comparison is predicted as a ratio.

### 2.1 Architecture constants (from model config, exact, no TODO)

KV bytes per token = 2 (K and V) x n_layers x n_kv_heads x head_dim x dtype_bytes

| model | layers | kv_heads | head_dim | kv bytes/token | fp16 weights |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 2 | 64 | 12,288 B | 0.99 GB |
| Qwen2.5-7B | 28 | 4 | 128 | 57,344 B | 15.2 GB |

Param counts computed exactly by bench/roofline.py estimate_params (GQA,
SwiGLU, tied/untied embeddings); 7B verified against the published 7.62B at the
prereg commit.

### 2.2 Hardware constants

Spec sheet values (verify against vendor sheets at commit):

| GPU | spec BW | peak fp16 dense | VRAM |
|---|---|---|---|
| T4 | 320 GB/s | 65 TFLOPS | 16 GB |
| RTX 4090 | 1008 GB/s | 165 TFLOPS | 24 GB |
| A100 40GB | 1555 GB/s | 312 TFLOPS | 40 GB |

Calibrated values, one calibration run per rented card before any benchmark:
- BW_eff = c_bw x spec_BW. c_bw: TODO-measure via a CUDA copy/triad
  microbenchmark on the rented card (protocol: 1 GB buffers, 50 reps, report
  median). Prior expectation 0.75 to 0.85; a value outside 0.6 to 0.95 means a
  broken host and the card is rejected, not the model.
- MFU for prefill: TODO-measure from one timed prefill of a 2048-token prompt.
  Prior expectation 0.5 to 0.7 for the cuBLAS-dominated prefill path, 0.3 to
  0.5 if the unfused path binds.
- t_overhead: per-decode-step fixed overhead (scheduler + launch), TODO-measure
  as (measured B=1 step time) minus (roofline B=1 step time) on the calibration
  run. This is the term whose omission produced the 10x T4 miss.

### 2.3 Formulas

Decode step time, batch B, mean context S:

    t_step = (W_bytes + B x S x kv_per_tok) / BW_eff + t_overhead

Decode throughput ceiling:

    decode_tok_s(GPU, model, B) = B / t_step

Weight/KV crossover batch (where KV traffic equals weight traffic):

    B* = W_bytes / (S x kv_per_tok)

Prefill time for a P-token prompt:

    t_prefill = 2 x N_params x P / (peak_FLOPS x MFU)

Max KV-bounded concurrency:

    C_max = (VRAM - W_bytes - activation_reserve) / (S_avg x kv_per_tok)
    activation_reserve = 1 GB placeholder, TODO-measure at calibration.

Saturation QPS (decode-side, per workload):

    QPS_sat = decode_tok_s(B_eff) / mean_output_tokens

where B_eff = min(C_max, B at the measured knee). TTFT-SLO breaking QPS: the
offered rate at which arrival_rate x t_prefill(mean_P) exceeds the compute
fraction left after decode; the closed form and the number are filled from the
calibration constants before rental (amendment A1) and frozen then.

### 2.4 Tolerance bands and scoring rule

- Roofline quantities with measured BW_eff (B=1 tok/s, batch curve, B*, C_max):
  plus or minus 15 percent. Vidur reports 5 to 9 percent error for a fitted
  simulator; 15 percent is the honest band for a closed-form model.
- Scheduling-dependent quantities (saturation QPS, SLO-break QPS, goodput
  ratios): plus or minus 30 percent.
- Every prediction is scored hit or miss against the band. Misses are reported
  with a found or conjectured cause. No band is widened after data exists.

## 3. Preregistered predictions

Rows marked TODO-calibrate get numeric values from the calibration constants,
entered via amendment A1 BEFORE the paid benchmark runs; the formulas are
frozen now. Provisional values below use c_bw = 0.80, t_overhead = 0.

| # | quantity | prediction | band | a miss means |
|---|---|---|---|---|
| P1 | 7B B=1 decode tok/s, 4090 | BW_eff / (15.23e9 + S x 57344); ~52 tok/s at S=2048, c_bw=0.80; TODO-calibrate | 15% | bandwidth model wrong or overhead term mismeasured |
| P2 | 7B B=1 decode tok/s, A100 40GB | same formula, 1555 GB/s; ~81 tok/s provisional; TODO-calibrate | 15% | model fails to transfer across GPU class |
| P3 | 7B decode tok/s vs B curve, 4090, S=2048 | B / t_step per section 2.3 at B in {1,2,4,8,16,32,64}; near-linear to the knee | 15% per point | kernel or scheduler overhead not captured by t_overhead |
| P4 | weight/KV crossover batch B*, 7B, S=2048 | B* = 15.23e9 / (2048 x 57344) = ~130 | 15% | KV traffic accounting wrong |
| P5 | max concurrent seqs, 7B on 4090, ShareGPT S_avg~381 | C_max = (24e9 - 15.23e9 - 1e9) / (381 x 57344) = ~355 | 15% | activation reserve or allocator overhead underestimated |
| P6 | saturation QPS, ShareGPT, 7B 4090, nanoserve | QPS_sat formula; TODO-calibrate | 30% | scheduler loses throughput the roofline cannot see |
| P7 | TTFT-SLO breaking QPS, ShareGPT, 7B 4090, nanoserve | utilization formula per 2.3; TODO-calibrate | 30% | prefill/decode interference worse than the utilization argument (no chunked prefill) |
| P8 | nanoserve/vLLM-eager goodput ratio, ShareGPT 4090 | itemized overhead budget, section 3.1; predicted ratio TODO-calibrate from budget | 30% | at least one budget line item is wrong; identify which |
| P9 | nanoserve/vLLM-default goodput ratio, ShareGPT 4090 | P8 ratio x (eager/default vLLM ratio predicted at 0.85 to 1.0) | 30% | CUDA-graphs and default scheduling worth more than budgeted |
| P10 | 0.5B continuity, T4 | continuous_fused ~287 tok/s reproduces within band | 15% | environment drift; halts the study until explained |

### 3.1 Itemized overhead budget for P8 (the ratio prediction)

Where nanoserve loses to eager vLLM, each line preregistered, each checked
independently in the report:
- Python step orchestration: measured in Phase 2 at 0.2 to 3.2 percent of a
  decode step on the fused path. Budget: 3 percent.
- Kernel efficiency gap (Triton decode attention vs FlashAttention/
  PagedAttention, plus unfused elementwise ops): budget TODO-calibrate from one
  op-level A/B at calibration; prior 10 to 25 percent.
- No CUDA graphs (launch overhead per step): budget 5 to 15 percent at B=1,
  shrinking with batch; TODO-calibrate.
- Scheduler quality (batch shaping, no chunked prefill): the residual; whatever
  the total gap leaves is attributed here and that attribution is itself a
  falsifiable claim checked against the TTFT decomposition.

## 4. Measurement protocol

### 4.1 Load generation
Open loop only. Arrival times drawn in advance and sent on schedule regardless
of completions. Poisson sweeps at 6 offered rates spanning 0.2x to 1.5x of the
predicted saturation QPS (P6), plus one bursty condition (Gamma inter-arrivals,
burstiness 0.5), plus the BurstGPT replay at its 3 rescaled levels. Closed-loop
numbers are never reported.

### 4.2 Client discipline
Multi-process client, or a logged client-CPU audit on every run; any window
where client CPU exceeds 80 percent is excluded (section 5.4). Same client
code, same machine, for every server.

### 4.3 Warmup, windows, repetitions
2-minute warmup discarded per condition. 5-minute measured window. 3
repetitions on the 4090 grid, 2 on the A100 grid, ABBA interleaved between
servers within the same rental session so host noise hits both. Percentile
floors: no p99 quoted under 1,000 samples in the window, no p99.9 under
100,000 pooled samples (the repo's existing rule).

### 4.4 Token accounting and output control
One tokenizer (the Qwen2.5 tokenizer) counts tokens for every system.
Pre-tokenized prompts fed to both servers, token counts verified equal.
Output lengths forced with ignore_eos to the dataset's replayed lengths.
Greedy sampling, temperature 0. Prefix caching off everywhere.

### 4.5 The flag ladder (every flag disclosed for every run)
1. vLLM matched: enforce_eager, prefix caching off, chunked prefill off (or the
   nearest equivalents in the pinned version).
2. vLLM default: out-of-the-box flags.
3. vLLM tuned: max_num_batched_tokens and max_num_seqs swept once, best kept.
nanoserve runs one config per engine, disclosed identically. Disclosure list
per run: engine version and commit, model revision hash, dtype, TP degree,
gpu_memory_utilization, max_num_seqs, max_num_batched_tokens, chunked prefill
setting, prefix caching setting, eager vs graphs, scheduler steps, sampling,
ignore_eos, tokenizer, client spec and CPU utilization, warmup protocol, seed.

### 4.6 Metrics
p50/p99 TTFT, p50/p99 TPOT, ITL distribution, tokens/s, and goodput: completed
requests per second meeting BOTH TTFT p99 <= 1 s AND TPOT p99 <= 100 ms
(DistServe framing), attainment threshold 90 percent, measured from client send
time, no request dropping counted as success, one tokenizer. Cost proxy:
tokens per dollar-hour at the actual rented price.

### 4.7 Versions and repro
TODO-pin at the prereg commit: vLLM wheel hash, docker image digests, model
revision hashes, nanoserve commit, CUDA and driver versions recorded per host.
One command regenerates every figure from committed raw per-request CSVs; the
same script rerun on a fresh GPU regenerates the raw data. Nothing debugs on
the meter: the full harness runs end to end on Kaggle T4 before any rental.

## 5. Analysis rules (written before data)

### 5.1 When a comparison counts
A difference between two systems or configs is claimed only when per-run ranges
across repetitions do not overlap (the repo's existing noise rule). Overlapping
ranges are reported as indistinguishable, never as a small win.

### 5.2 Statistics
Median across repetitions is the headline; min-max spread shown on every
figure. No mean-of-percentiles; percentiles computed on pooled raw samples
within a window, then the median of window values across reps.

### 5.3 Negative results clause
Everything in the prediction table is reported, hit or miss, in the main body,
not an appendix. The abstract states the miss count. Losing configurations for
nanoserve are reported with the same prominence as winning ones. If the study
produces no wins for nanoserve, it ships anyway; the deliverable is the scored
cost model, not a horse race result.

### 5.4 Exclusion rules (decided now)
A run is excluded only for: host preemption mid-window, client CPU over 80
percent, GPU thermal throttling flagged by nvidia-smi, or a server crash. Every
exclusion is reported with its reason. No exclusion for "the number looks
wrong."

### 5.5 Figures (fixed set)
(1) p99 TTFT and p99 TPOT vs offered QPS, one line per server config.
(2) Goodput vs QPS with the SLO on the figure.
(3) Predicted vs measured roofline with tolerance bands (the signature figure).
(4) Itemized overhead waterfall for the nanoserve/vLLM gap.
(5) The prediction table, scored.
No other result figures are added after data; exploratory findings go in a
clearly labeled post-hoc section.

## 6. Budget plan (about $30 cash)

| line | resource | hours | cost |
|---|---|---|---|
| dev + T4 point | Kaggle T4 | free tier (30 h/wk) | $0 |
| shakeout | Modal free credits, A100 40GB | ~14 h | $0 (monthly $30 credit) |
| main study | Vast.ai RTX 4090 at ~$0.25/h | ~30 h | ~$7.50 |
| scale check | Vast.ai or RunPod A100 40GB at ~$0.70/h | ~14 h | ~$10 |
| reserve | re-runs and mistakes | | ~$10 |
| total | | | under $30 |

4090 grid sizing check: 2 servers x 3 configs x 3 workloads x 6 QPS points x 5
min x 3 reps is about 27 hours; trim to 2 reps or prune the tuned config if it
does not fit. A100 grid: 1 workload x 6 QPS x 2 servers x 2 reps. Prices are
Aug 2026 directional; re-verify the day of rental and record the actual price
paid (it feeds the tokens-per-dollar metric).

## 7. Amendment log

Format, one entry per change, appended only:

    A<n> | date | section | what changed | why | data collected yet? (yes/no)

Pre-registered planned amendments:
- A1: calibration constants (c_bw, MFU, t_overhead, activation_reserve) and the
  numeric values they imply for P1, P2, P6, P7, P8. Must land before any paid
  benchmark run. Collected-data flag: no.
No other amendment may alter a formula, band, SLO, exclusion rule, or the
figure list after A1. Amendments after data collection may only add
clarifications or post-hoc analyses labeled as such.

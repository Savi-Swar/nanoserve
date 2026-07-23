# nanoserve

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Savi-Swar/nanoserve/blob/main/notebooks/nanoserve_gpu.ipynb)

A from-scratch LLM inference server in Python + PyTorch, no custom CUDA. The
model is a stock Qwen2.5-0.5B. The part I actually built is the scheduler: the
layer that decides how requests share the GPU.

To reproduce the fp16 numbers on a free GPU, open the Colab notebook above and
run all cells (details in [docs/gpu_run.md](docs/gpu_run.md)).

![throughput vs load](results/throughput_vs_rate.png)

Under continuous batching, throughput goes up as offered load increases while
TTFT stays roughly flat. The numbers below are from a CPU dev box (fp16 on a
real GPU is a lot faster); the interesting part is the relative gap between
engines, not the absolute values.

| engine | throughput | TTFT p99 |
|---|---|---|
| naive | 24 tok/s | ~9,000 ms |
| static | 34 tok/s | ~3,700 ms |
| continuous | 41→63 tok/s (scales with load) | ~200 ms |
| paged | 66 tok/s | ~240 ms |

Replaying the real
[Azure inference trace](docs/writeup.md#real-traffic-the-synthetic-benchmark-lied)
instead of uniform synthetic prompts changes the conclusions. Continuous
batching's advantage moves from throughput to tail latency, and static batching
ends up with the worst TTFT tail of any engine (worse than naive) because it
head-of-line blocks on long generations. One early "paged beats continuous"
result didn't hold up across repeated runs (it sat inside the noise floor), so I
dropped it. More in [docs/writeup.md](docs/writeup.md).

## Audit: which optimizations actually help on real workloads?

This is the main point of the project. Implement published optimizations one at
a time in a rig I control, ablate each in isolation, and measure them on
workloads I pick rather than the one each paper picked for itself. Everything is
measured against a ±24% noise floor (`make noise`), and each optimization is
checked to produce identical output tokens to naive before I measure speed.

| optimization | result | `make` |
|---|---|---|
| speculative decoding | 2.7x on repetitive text, 1.0x (no gain) on generic prose. The speedup is a property of the workload, not the method. | `spec` |
| prefix caching | holds up: 70% of prefill saved with a shared system prompt, ~0 without, output identical | `prefix` |
| KV quantization | 8-bit is near-lossless (93% top-1 agreement, 2x memory); 4/2-bit degrade. Also a metric trap: naive token-match said 8-bit "drifts 48%", teacher-forced agreement says 93%. | `kvquant` |
| chunked prefill | left out: its payoff is tail latency, which the CPU noise floor would bury | — |

Method and the numbers that didn't survive:
[docs/writeup.md](docs/writeup.md#audit-which-optimizations-survive-contact-with-real-workloads).

## Reproduce

```bash
make all      # memory ablation + engine sweep + graphs -> results/

# or in Docker, with the PNGs written back out through the bind mount:
docker build -t nanoserve . && docker run --rm -v $(pwd)/results:/app/results nanoserve
```

`make` with no target lists everything. `make bench DEVICE=cuda` runs the sweep
on a GPU. Individual commands are under [Quickstart](#quickstart).

## The ladder

Four engines, each a step up from the last:

```
naive           one request at a time              baseline
static batching wait for N, run together           GPU idles when short reqs finish
continuous      evict finished, admit waiting      slot reused right away
paged KV cache  block allocator + free list        ~3x more concurrency per byte
```

Paged trades about 6% raw speed (the per-step block gather) for roughly 3x the
concurrency per byte of memory. On 64 length-skewed sequences it drops KV
fragmentation from 69% to 4%, so a fixed budget holds 3x more sequences. The
gain is capacity under memory pressure, not latency. Details in
[docs/writeup.md](docs/writeup.md).

## Layout

```
server/
  request.py       Request + SamplingParams
  model.py         ModelRunner: prefill / decode primitives, sampling
  batched.py       BatchState: left-padded batched KV, admit/step/evict
  paged_cache.py   BlockAllocator: free list, block tables, fragmentation metrics
  paged_exec.py    PagedKVStore + PagedBatchState: KV in blocks, gather/scatter
  speculative.py   prompt-lookup speculative decoding
  prefix_cache.py  prefix KV reuse
  kv_quant.py      low-bit KV cache quantization
  engine.py        naive / static / continuous / paged engines
bench/
  workload.py      open-loop Poisson load generator + prompt bank
  trace.py         real Azure trace replay
  metrics.py       throughput, TTFT/e2e/queue p50/p90/p99, JSON out
  gpu.py           nvidia-smi utilization sampler (no-op off CUDA)
  run_bench.py     one engine, one workload -> report
  sweep.py         engine x rate grid -> results/sweep.json
  repeat.py        N-run stats: mean, 95% CI, noise floor
  memory_study.py  fragmentation ablation: reserve vs padded vs paged
  spec_study.py    speculative decoding tokens/forward
  prefix_study.py  prefix caching prefill savings
  kv_quant_study.py  KV quant memory/quality
  trace_compare.py all engines on the real trace
  roofline.py      analytical throughput ceiling
  plot.py          the grid -> PNGs
  vllm_ref.py      vLLM reference, run on a CUDA box
docs/
  research.md      background reading, mapped to this build
  frontier/        deeper notes: history, spec decoding, KV, kernels, systems
  writeup.md       method, the ladder, the ablations, the audit
tests/             fast tests + a guarded equivalence oracle
```

## Quickstart

```bash
pip install -r requirements.txt

# one run
python -m bench.run_bench --engine paged --rate 8 --n 48 --max-tokens 64

# the full comparison + graphs
python -m bench.sweep --engines naive static continuous paged --rates 2 4 8 16 --n 64
python -m bench.plot

# the fragmentation ablation (no GPU needed)
python -m bench.memory_study --n 64 --block-size 16

# tests; the equivalence oracle loads the model so it's gated:
python -m pytest -q
RUN_SLOW=1 python -m pytest tests/test_equivalence.py -q
```

On Apple Silicon, MPS matmul is broken for this model, so dev happens on CPU
(`--device cpu`). Real numbers come from a CUDA GPU (`--device cuda`).

## Correctness

Batched, paged, speculative, and prefix-cached decoding are all checked
token-for-token against single-sequence naive decoding under greedy, including
the mid-stream KV-merge path. These are scheduling and memory optimizations, not
approximations, so the output shouldn't change.

## CI / perf regression

GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on every push and PR:

- **tests**: Python 3.12, `pip install -r requirements.txt`, `python -m pytest -q`
  (fast tests only; `RUN_SLOW` is unset so the model-loading equivalence oracle
  skips). No GPU, no model download.
- **perf**: the regression gate, `python -m bench.regression --check`.

The gate runs a deterministic, no-model proxy (CI runners have no GPU, and
pulling a ~1 GB model per PR is wasteful) built from two signals:

- fragmentation and paged capacity from the `bench.memory_study` ablation (pure
  functions of a seeded workload and the real `BlockAllocator`, so
  reproducible) for the correctness of paged packing;
- allocator throughput (ops/sec) from a micro-benchmark of the pure-Python
  `add_seq`/`append_token`/`free_seq` hot path (median of repeats) as a timing
  signal.

It fails the build if a metric regresses past its threshold: fragmentation up
>3%, paged capacity down >3% (tight, since these are deterministic), or
allocator ops/sec down >15% (loose, since timing is noisy on shared runners).

```bash
python -m bench.regression --update-baseline   # refresh results/baseline.json
python -m bench.regression --check             # gate against the baseline
```

Commit `results/baseline.json` and regenerate it when a change is meant to move
the numbers.

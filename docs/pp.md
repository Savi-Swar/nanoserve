# Serving a model bigger than the GPU

Qwen2.5-7B is 15.2 GB of fp16 weights; a T4 has 14.6 GB usable. Serving it
at all requires splitting the model, so this chapter runs on a two-T4 pair:
layers 0-11 with the embeddings on gpu0 (6.2 GiB), layers 12-27 with the
head on gpu1 (8.0 GiB), an accelerate device map moving activations across
the cut.

The paged machinery follows the layers. Each layer's KV pool lives on that
layer's device, the cache mirrors its index tensors per device on first
touch, and the Triton launchers re-enter under the tensor's own GPU when the
current one differs. None of it costs anything single-GPU: every mirror is a
no-op copy.

## What it measures

| configuration | tok/s | TTFT p99 | GPU util (mean) |
|---|---|---|---|
| hf generate, single stream | 14.6 | - | - |
| continuous batching | 116.3 | 3.5 s | 37% |
| paged_fused | 116.7 | 3.6 s | 36% |
| interleaved halves (pp2) | 98.5 | 0.54 s | 37% |
| interleaved + issue threads | 93.9 | 0.59 s | 35% |

Correctness first: the sharded fused path is token-exact against stock hf
generation through the cut, checked every run. Continuous batching buys
8x over the naive single stream. And then the number that owns the rest of
the chapter: 36% mean utilization. The two stages run one after the other,
and each GPU waits out the other's turn.

## The bubble that would not fill

The textbook fix is microbatch interleaving: keep two half-batches at
independent decode steps, issue both forwards back to back, and let CUDA's
async semantics run half B's first stage on gpu0 while half A's second
stage runs on gpu1. vLLM sizes microbatches-in-flight to the stage count
for exactly this.

It did not work, twice, and the second failure explains the first.

Back-to-back async issue (pp2): throughput fell to 98.5 and utilization did
not move. Per-half issue threads (pp2t): 93.9, still 35%. The diagnostic
settles it: at B=8 the sharded step takes 73.6 ms end to end, and 52.3 ms
of that is python issue time before the sync (overlap-limit 0.71). Walking
28 hooked layers through transformers on this host costs more CPU than the
GPUs spend computing, so the second half is issued when the first is nearly
done. Threads cannot fix that: the issue cost is GIL-holding python, so two
issue threads interleave on the lock and the total python per round is
unchanged, plus overhead.

One real improvement survived: TTFT p99 fell from 3.5 s to 0.54 s, because
two halves double admission capacity and the queue drains. Interleaving here
is a latency feature, not a throughput one.

## What would actually work

The bubble is the same enemy this project already beat once: python and
launch overhead dominating the step. The single-GPU cure was capturing the
whole step in a CUDA graph (34 -> 9.4 ms). A recording cannot span devices,
so the sharded version needs one graph per stage per device, with the
cross-device activation copy managed between replays. That removes the
52 ms of python entirely and is the planned v3; the measured prediction it
must beat is 116.7 tok/s at 36%, with the interleave layered back on top
once the step is no longer issue-bound.

A note on why not tensor parallelism: the T4 pair is PCIe-only, and TP
needs an all-reduce every layer. Pipeline parallelism crosses the
interconnect once per token per cut, which is why it is the right shape for
this hardware even with the bubble.

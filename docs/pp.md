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

## v3: one graph per stage

The bubble's enemy was the same one this project beat once before: python
dominating the step. The cure was the same too, with a twist: a CUDA graph
cannot span devices, so the sharded step became two graphs and a
single-copy handoff (replay stage0 on gpu0, copy the hidden states across
the cut, replay stage1 on gpu1), with capture-time python calling the
decoder layers directly.

Getting the capture correct took four fixes, each a general lesson:

1. accelerate's device hooks poison capture (their send_to_device creates a
   dependency on uncaptured work); detached during capture, restored after.
2. The rotary inv_freq buffer lived on the cpu (accelerate's dispatch skips
   some non-persistent buffers), so the eager path had been paying a silent
   host-to-device copy every step, and capture turned it into a hard error.
3. cuBLAS allocates its workspace lazily per stream; warm up on the same
   stream the graph records on or the allocation lands inside the capture.
4. The nastiest: replayed rotary output ignored the position static
   entirely (cos identical for position 3 and position 9). Under a device
   map, module forwards route through a dynamo-compiled wrapper, and
   capturing the compiled rotary baked the positions. Fixed by computing
   rotary eagerly each step into statics both graphs read; a bake probe in
   the diagnostic now guards it.

Result: token-exact against hf, decode step 73 -> 23 ms (3.2x). And the
serving numbers moved barely at all: 119 vs 116 tok/s short-output, 144 vs
140 on 192-token outputs, utilization still 37%. The step win is real and
the workload swallowed it, because at 7B the wall clock belongs to
PREFILL: every admission runs the prompt through the eager hooked stack at
a few hundred tokens per second, and sixteen of those cost about as much
as all the decoding put together. The graphs' fingerprint shows exactly
where prefill does not mask it: decode-heavy TTFT p99 fell 1101 -> 204 ms.

So the chapter ends the way the whole project keeps ending: fix the
bottleneck, measure, and meet the next one. Attention, then launch
overhead, then the python issue cost, now prefill. Chunked or
direct-stage prefill is the named next wall.

A note on why not tensor parallelism: the T4 pair is PCIe-only, and TP
needs an all-reduce every layer. Pipeline parallelism crosses the
interconnect once per token per cut, which is why it is the right shape for
this hardware even with the bubble.

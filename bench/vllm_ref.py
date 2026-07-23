"""Reference-ceiling harness: run the benchmark workload through vLLM.

This is the *reference ceiling* for the from-scratch engine in this repo. It
drives the EXACT same open-loop Poisson workload (bench/workload.py) through
vLLM's production continuous-batching engine and reports the same metrics
(bench/metrics.py), so a plot can put "our engine" next to "what a mature
system does on this box." If our engine lands within a sensible fraction of
this line, the from-scratch implementation is doing its job.

Why AsyncLLMEngine and not LLM.generate:
    The offline `LLM.generate(prompts)` batch API takes a *list* up front and
    schedules it however it likes. That erases arrival timing entirely, so it
    cannot model an open-loop Poisson load and cannot measure a real TTFT tail
    under queueing. We instead use the async, streaming AsyncLLMEngine: one
    asyncio task per request that sleeps until the request's arrival offset,
    then streams tokens, recording the wall-clock time of the first streamed
    token (true TTFT) and of completion. This mirrors real clients hitting a
    live server, which is the whole point of the comparison.

Run it on a GPU box (vLLM is Linux/CUDA only):
    pip install vllm
    python -m bench.vllm_ref --model Qwen/Qwen2.5-0.5B --rate 20 --n 200 \
        --max-tokens 128 --out results/vllm_r20.json

    # then compare against the from-scratch engine's run at the same rate/n.

API note:
    The vLLM API used here (AsyncEngineArgs, AsyncLLMEngine.from_engine_args,
    and the streaming `async for out in engine.generate(prompt, sampling_params,
    request_id=...)` interface where each yielded RequestOutput carries
    cumulative `outputs[0].token_ids` and a `.finished` flag) matches vLLM
    ~0.4.x through ~0.6.x. On the newer V1 engine (vLLM >= 0.7) AsyncLLMEngine
    still exists but is being superseded by `vllm.v1` / `AsyncLLM`; the
    from_engine_args + streaming-generate contract used below is stable across
    the 0.4-0.6 line. `RequestOutput.metrics.first_scheduled_time` (used to
    recover queue delay) is best-effort and may be None on some builds; we fall
    back gracefully.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from .gpu import GpuSampler
from .metrics import Report, build_report
from .workload import build_requests


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="vLLM reference-ceiling harness (same workload as the "
        "from-scratch engine, run through vLLM's continuous batching)."
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--rate", type=float, default=20.0,
                   help="mean arrival rate (requests/sec, Poisson)")
    p.add_argument("--n", type=int, default=200, help="number of requests")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    # vLLM engine knobs
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--out", default=None, help="path to write results JSON")
    return p.parse_args(argv)


def report_to_dict(report: Report, model: str, rate: float, max_tokens: int) -> dict:
    """Serialize a metrics.Report into the exact schema the sweep + plots expect."""
    gpu = None
    if report.gpu:
        gpu = {"mean": report.gpu["mean"], "peak": report.gpu["peak"]}
    return {
        "engine": "vllm",
        "device": "cuda",
        "model": model,
        "rate": rate,
        "n": report.n,
        "max_tokens": max_tokens,
        "wall": report.wall,
        "out_tokens": report.out_tokens,
        "throughput": report.throughput,
        "per_req_decode_tps": report.per_req_decode_tps,
        "ttft": report.ttft,
        "e2e": report.e2e,
        "queue": report.queue,
        "gpu": gpu,
    }


async def _drive_one(engine, req, offset, run_start, VllmSamplingParams):
    """Sleep until this request's arrival offset, then stream it through vLLM,
    populating the project's Request timing fields in place."""
    # Open-loop: release at the scheduled wall-clock offset regardless of
    # whether earlier requests have finished (that's what builds a real queue).
    target = run_start + offset
    delay = target - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    req.arrival_time = time.perf_counter()

    # Per-request sampling params. ignore_eos=True keeps token counts stable
    # across engines so throughput is an apples-to-apples comparison.
    sp = VllmSamplingParams(
        max_tokens=req.sampling.max_tokens,
        temperature=req.sampling.temperature,
        ignore_eos=True,
    )

    request_id = str(req.id)
    final = None
    async for out in engine.generate(req.prompt, sp, request_id=request_id):
        # First streamed token => true TTFT anchor.
        if req.first_token_time is None and out.outputs and out.outputs[0].token_ids:
            req.first_token_time = time.perf_counter()
        final = out

    finish = time.perf_counter()
    req.finish_time = finish
    if req.first_token_time is None:
        # Degenerate case (e.g. max_tokens produced nothing streamed): anchor
        # TTFT at completion so the metrics props stay well-defined.
        req.first_token_time = finish

    if final is not None:
        comp = final.outputs[0] if final.outputs else None
        if comp is not None:
            # num_output is len(output_tokens); copy the real ids so counts and
            # decode_tps are exact.
            req.output_tokens = list(comp.token_ids)
        if final.prompt_token_ids is not None:
            req.prompt_len = len(final.prompt_token_ids)

        # Recover queue delay from vLLM's own metrics when available:
        # schedule_time = arrival_time + (first_scheduled - engine_arrival),
        # keeping arrival on our clock while using vLLM's measured queue wait.
        metrics = getattr(final, "metrics", None)
        if (
            metrics is not None
            and getattr(metrics, "first_scheduled_time", None) is not None
            and getattr(metrics, "arrival_time", None) is not None
        ):
            queue_wait = metrics.first_scheduled_time - metrics.arrival_time
            req.schedule_time = req.arrival_time + max(0.0, queue_wait)
        else:
            # No metrics hook on this build: best proxy is the first-token time
            # (queue delay is then folded into TTFT, reported as ~decode-free).
            req.schedule_time = req.first_token_time


async def run_workload(args) -> tuple[list, float]:
    """Build the shared workload, stream it all through vLLM, return (reqs, wall)."""
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm import SamplingParams as VllmSamplingParams

    reqs, offsets = build_requests(
        n=args.n,
        rate=args.rate,
        max_tokens=args.max_tokens,
        seed=args.seed,
        temperature=args.temperature,
    )

    engine_args = AsyncEngineArgs(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    run_start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _drive_one(engine, req, off, run_start, VllmSamplingParams)
        )
        for req, off in zip(reqs, offsets)
    ]
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - run_start
    return reqs, wall


def main(argv=None) -> int:
    args = parse_args(argv)

    # Guard the import so this file is safe to at least parse/inspect on a Mac
    # dev box; it only actually runs on a CUDA machine with vLLM installed.
    try:
        import vllm  # noqa: F401
    except ImportError:
        print(
            "vLLM not installed; run on a CUDA box: pip install vllm\n"
            "(vLLM is Linux/CUDA only and is not expected on the Mac dev box.)",
            file=sys.stderr,
        )
        return 1

    with GpuSampler() as gpu:
        reqs, wall = asyncio.run(run_workload(args))

    report = build_report("vllm", "cuda", reqs, wall, gpu.summary())
    print(report.render())

    results = report_to_dict(report, args.model, args.rate, args.max_tokens)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

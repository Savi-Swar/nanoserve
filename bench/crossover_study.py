"""Does the roofline model predict the machine? The decode-batch crossover test.

bench/roofline.py makes a falsifiable claim: decode throughput scales ~linearly
with batch size B up to a crossover

    B* = W / (S * kv_per_tok)          (weight bytes / per-batch KV bytes)

and flattens above it. Below B* a decode step is weight-bound (the fixed weight
read W dominates HBM traffic, so amortizing it over B tokens buys ~linear
throughput); above B* it's KV-bound (KV traffic B*S*kv_per_tok grows with B and
cancels the B in the numerator, so tok/s saturates). See
roofline.crossover_batch / decode_throughput_tok_s.

This script measures that crossover. For each B it runs a real batched decode
(server.batched.BatchState over B identical sequences), times `--steps` decode
steps, and computes decode throughput = (B * steps) / elapsed. It finds the B
where measured throughput stops scaling ~linearly and compares it to the
predicted B*: analytical prediction vs. what the hardware actually produces.

    python -m bench.crossover_study                       # full sweep (slow on CPU)
    python -m bench.crossover_study --batches 1 2 4 8 --steps 8 --seq-len 64

Clean only on GPU. The roofline assumes a single saturable HBM read bandwidth:
a decode step is a memory-copy of the model out of VRAM, and B* is the batch at
which the KV copy equals the weight copy. A CPU has caches, prefetchers,
multiple memory channels, and per-op Python/kernel-launch overhead that
dominates at these tiny sizes; it isn't bandwidth-bound the same way, so the CPU
crossover need not match B* and the absolute tok/s are overhead-limited. On CPU
this is a smoke test that the harness works; run on a T4/L4/A10 for the real
predicted-vs-measured comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from server.batched import BatchState
from server.model import ModelRunner
from server.paged_cache import kv_bytes_per_token
from server.request import Request, SamplingParams

from bench import roofline


def make_prompt_ids(m: ModelRunner, seq_len: int) -> list[int]:
    """Filler prompt of exactly `seq_len` real tokens, so the KV cache holds ~S
    tokens during decode and the measured S matches the S fed to the roofline."""
    base = m.encode(
        "The transformer decodes one token at a time, streaming every weight "
        "out of memory for each step, which is why batching amortizes the read."
    )
    if not base:
        base = [m.eos_id or 0]
    ids = (base * (seq_len // len(base) + 1))[:seq_len]
    return ids


def time_batch(m: ModelRunner, prompt_ids: list[int], B: int, steps: int) -> dict:
    """Prefill B identical sequences into one BatchState, then time `steps`
    batched decode .step() calls. m.sync() brackets the timing to flush queued
    device work before start and before stop; step() also sync()s internally, so
    every measured step is fully materialized.

    step() runs the full [B,1] batched forward over all rows every call,
    regardless of whether a row hit max_tokens, so even if rows finish partway
    through the window (no eviction) the timed compute is still the full B-wide
    decode. max_tokens is set to `steps`."""
    reqs = [
        Request(
            id=i,
            prompt="",  # unused: prompt_ids drives tokenization for exact length
            sampling=SamplingParams(max_tokens=steps, temperature=0.0, ignore_eos=True),
            prompt_ids=list(prompt_ids),
        )
        for i in range(B)
    ]
    batch = BatchState(m)
    batch.add(reqs)  # prefill; kernels/weights warm after this

    m.sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        batch.step()
    m.sync()
    elapsed = time.perf_counter() - t0

    tokens = B * steps
    return {
        "batch": B,
        "seq_len_ctx": batch.T,        # actual context width the KV holds
        "steps": steps,
        "elapsed_s": elapsed,
        "decode_tok_s": tokens / elapsed if elapsed > 0 else 0.0,
        "per_step_ms": 1e3 * elapsed / steps,
    }


def measured_crossover(sweep: list[dict]) -> tuple[float | None, str]:
    """Find the batch where throughput stops scaling ~linearly with B.

    Heuristic: for each consecutive pair (B_i -> B_{i+1}) compute the scaling
    efficiency of that jump,

        eff = (T_{i+1}/T_i - 1) / (B_{i+1}/B_i - 1)

    realized marginal throughput gain over the ideal linear gain. eff = 1.0 is
    perfect linear scaling (doubling B doubles tok/s); eff = 0 is full saturation
    (more batch buys nothing). Crossover is the first B_i whose next jump falls
    below eff = 0.5, i.e. marginal gain per doubling under 50% of linear, the
    knee the roofline predicts at B*. If every jump stays above 0.5 the sweep
    never saturated (crossover beyond the largest B) and we return None."""
    thr = 0.5
    for a, b in zip(sweep, sweep[1:]):
        b_ratio = b["batch"] / a["batch"]
        if b_ratio <= 1:
            continue
        t_ratio = b["decode_tok_s"] / a["decode_tok_s"] if a["decode_tok_s"] > 0 else 0.0
        eff = (t_ratio - 1) / (b_ratio - 1)
        b["scaling_eff_to_next"] = eff
        if eff < thr:
            return a["batch"], (
                f"first B whose next jump scales at {eff:.2f}x of linear "
                f"(< {thr:.1f}): throughput knee at B~={a['batch']}"
            )
    return None, (
        f"no jump fell below {thr:.1f} scaling efficiency; throughput still "
        f"scaled ~linearly across the whole sweep; crossover is beyond the "
        f"largest B tested"
    )


def main():
    p = argparse.ArgumentParser(
        description="Measure the decode-throughput batch crossover and compare "
                    "it to the roofline model's predicted B*.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64],
                   help="batch sizes to sweep (default doubles 1..64).")
    p.add_argument("--seq-len", type=int, default=128,
                   help="context length S the KV cache holds during decode.")
    p.add_argument("--steps", type=int, default=20,
                   help="decode steps to time per batch size.")
    p.add_argument("--mem-bandwidth-gbps", type=float, default=320.0,
                   help="HBM read bandwidth for the roofline prediction "
                        "(default 320 ~= NVIDIA T4).")
    p.add_argument("--out", default="results/crossover.json")
    a = p.parse_args()

    S = a.seq_len
    batches = sorted(set(a.batches))

    print(f"### decode-crossover study   model={a.model}  device={a.device}")
    print(f"loading model...")
    m = ModelRunner(a.model, device=a.device)
    m.warmup()

    on_cpu = m.device == "cpu"
    if on_cpu:
        print("\n[!] device=cpu: this is a SMOKE TEST. A CPU is not saturably")
        print("    bandwidth-bound (caches/prefetch/multi-channel + Python & kernel")
        print("    overhead dominate at these sizes), so the roofline's B* need not")
        print("    match here. Run on a GPU (T4/L4/A10) for the clean test.")

    prompt_ids = make_prompt_ids(m, S)

    # ---- MEASURE: time a batched decode at each B --------------------------
    print(f"\nmeasuring decode throughput  (S={S} ctx tokens, {a.steps} timed "
          f"steps/batch)...")
    sweep = []
    oom_at = None
    for B in batches:
        try:
            row = time_batch(m, prompt_ids, B, a.steps)
        except torch.cuda.OutOfMemoryError:
            # B*S*S attention at large B outgrows a small GPU; keep the curve we
            # have (the knee is usually well below the largest B) instead of
            # losing the whole run.
            oom_at = B
            torch.cuda.empty_cache()
            print(f"  B={B:>4}  OOM, stopping the sweep here, "
                  f"reporting the {len(sweep)} batches that fit")
            break
        sweep.append(row)
        print(f"  B={B:>4}  {row['decode_tok_s']:>9.1f} tok/s  "
              f"{row['per_step_ms']:>8.2f} ms/step")

    if not sweep:
        print("[!] no batch size fit in memory, nothing to report")
        return

    meas_cross, meas_reason = measured_crossover(sweep)

    # ---- PREDICT: the roofline's crossover B* for this model/S -------------
    cfg = roofline.config_from_runner(m)
    params = roofline.estimate_params(cfg)["total"]
    dtype_bytes = m.dtype.itemsize
    kv_per_tok = kv_bytes_per_token(m)          # bytes/token at the model's dtype
    pred_cross = roofline.crossover_batch(params, kv_per_tok, S, dtype_bytes)
    mem_bw = a.mem_bandwidth_gbps * roofline.GB

    # Overlay the analytical throughput ceiling on each measured B.
    for row in sweep:
        row["predicted_ceiling_tok_s"] = roofline.decode_throughput_tok_s(
            params, kv_per_tok, row["batch"], S, mem_bw, dtype_bytes)

    # ---- REPORT -------------------------------------------------------------
    print(f"\n== crossover: predicted vs measured ==")
    print(f"  {'B':>5} {'decode tok/s':>13} {'per-step ms':>12} "
          f"{'roofline tok/s':>15}")
    for row in sweep:
        print(f"  {row['batch']:>5} {row['decode_tok_s']:>13.1f} "
              f"{row['per_step_ms']:>12.2f} {row['predicted_ceiling_tok_s']:>15.0f}")

    meas_str = f"{meas_cross:g}" if meas_cross is not None else "not reached (> max B)"
    print(f"\n  predicted crossover B* = {pred_cross:.1f}   "
          f"(W={params*dtype_bytes/roofline.GB:.3f} GB / "
          f"(S={S} * kv_per_tok={kv_per_tok} B))")
    print(f"  measured crossover  ~= {meas_str}")
    print(f"    ({meas_reason})")
    if on_cpu:
        print("  [smoke test on CPU: don't read agreement/disagreement here as a "
              "verdict on the model; the clean test is on GPU.]")

    # ---- PERSIST ------------------------------------------------------------
    out = {
        "model": a.model,
        "device": m.device,
        "dtype_bytes": dtype_bytes,
        "seq_len": S,
        "steps": a.steps,
        "mem_bandwidth_gbps": a.mem_bandwidth_gbps,
        "params": params,
        "kv_bytes_per_token": kv_per_tok,
        "weights_bytes": params * dtype_bytes,
        "predicted_crossover_batch": pred_cross,
        "measured_crossover_batch": meas_cross,
        "measured_crossover_reason": meas_reason,
        "oom_at_batch": oom_at,
        "cpu_smoke_test": on_cpu,
        "note": (
            "CPU is not saturably bandwidth-bound; measured crossover need not "
            "match B* here. Clean predicted-vs-measured test is on GPU."
        ) if on_cpu else "",
        "sweep": sweep,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

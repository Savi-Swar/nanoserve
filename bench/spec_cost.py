"""Analytical cost model for SPECULATIVE DECODING vs batch size — predict, on
the back of an envelope, the batch size at which drafting flips from a win to a
net loss, so a measured batched-spec engine has a yardstick to beat.

The physics in one paragraph
----------------------------
Speculative decoding trades extra COMPUTE for fewer sequential model FORWARDS.
A plain decode step advances every one of the B rows by exactly one token, so
it processes B tokens. A speculative step first drafts `g` extra candidate
tokens per row (cheaply, with a small drafter or a prompt-lookup table), then
verifies all of them in a single big forward that processes B*(1+g) tokens.
Verification accepts a prefix of the draft: with per-token acceptance rate
`a` (0..1), each row commits, in expectation,

    tokens_committed_per_step = 1 + a*g          (the +1 is the always-correct
                                                  "bonus" token from the verify
                                                  pass; a*g are accepted drafts)

So the *benefit* of a spec step is 1 + a*g committed tokens instead of 1.

The *cost* of that step — relative to a plain decode — depends on which side of
the roofline the batch sits on (see bench/roofline.py). A decode step is a
memory-copy of the model weights out of HBM; arithmetic is nearly free until
the batch is large enough to saturate the compute units.

  * WEIGHT-BOUND regime (small B, below the roofline crossover B*): the step is
    dominated by streaming the weights from HBM once. The B*(1+g) tokens all
    ride along on that single weight read essentially for free, so processing
    (1+g)x more tokens barely changes the step time:   cost_factor ~= 1.

  * COMPUTE-BOUND regime (large B, past B*): the ALUs are the bottleneck and
    step time scales with the number of tokens processed, so drafting g extra
    tokens per row genuinely costs (1+g)x the FLOPs:    cost_factor ~= (1+g).

We interpolate linearly between the two roofs using the same crossover batch
B* the roofline model predicts (B* = W / (S * kv_bytes_per_token); see
bench.roofline.crossover_batch and server.paged_cache.kv_bytes_per_token):

    cost_factor(B) = 1 + g * min(1, B / Bstar)

Putting benefit over cost:

    speedup(B) = (1 + a*g) / cost_factor(B)
               = (1 + a*g) / (1 + g * min(1, B/Bstar))

Speculation is a WIN when speedup(B) > 1 and a net LOSS when speedup(B) < 1.

The win->loss crossover has a clean closed form. Below B* the denominator is
1 + g*B/Bstar, and speedup = 1 exactly when

    1 + a*g = 1 + g*(B/Bstar)   =>   B_crossover = a * Bstar.

So spec pays off only for batches below `a * Bstar`, and turns into a tax above
it. (Past B* itself the step is fully compute-bound, cost_factor = 1+g, and
speedup = (1+a*g)/(1+g) < 1 for any a<1 — always a loss — consistent with the
crossover always landing at a*Bstar < Bstar.)

Why this matters
----------------
The crossover scales with acceptance `a`. Using the project's own measured
per-token acceptance numbers (bench/spec_study.py, results/spec.json):

  * generic prose  — PLD acceptance ~0  (tokens/forward ~1.0): a*Bstar ~= 0, so
    spec is a net loss at essentially any batch > 1.
  * code           — acceptance ~0.52  (tokens/forward ~2.3): wins to mid batch.
  * grounded/RAG   — acceptance ~0.92  (tokens/forward ~2.7): wins far longer,
    almost to B* itself.

That single number — "spec becomes a net loss for batch >= Y" — is what pairs
with a measured batched-spec engine. This module needs no model download: it
pulls Qwen2.5-0.5B geometry and the crossover predictor straight from
bench.roofline.
"""
from __future__ import annotations

import argparse
import json
import math
import os

from bench.roofline import (
    QWEN_0_5B,
    crossover_batch,
    estimate_params,
    kv_bytes_per_token,
)

# Measured per-token acceptance rates from this project (results/spec.json via
# bench/spec_study.py). Used only to LABEL a given `a` with the regime it most
# resembles, so the report reads in plain English.
MEASURED_REGIMES = [
    (0.00, "generic prose (PLD accept ~0, tok/forward ~1.0)"),
    (0.52, "code (accept ~0.52, tok/forward ~2.3)"),
    (0.92, "grounded/RAG (accept ~0.92, tok/forward ~2.7)"),
]


def default_bstar(seq_len: int = 2048, dtype_bytes: int = 2) -> float:
    """Roofline crossover batch B* for Qwen2.5-0.5B at the given context, fp16.
    B* = W / (S * kv_bytes_per_token). Comes out ~39 at S=2048."""
    params = estimate_params(QWEN_0_5B)["total"]
    kv_per_tok = kv_bytes_per_token(QWEN_0_5B, dtype_bytes)
    return crossover_batch(params, kv_per_tok, seq_len, dtype_bytes)


def cost_factor(B: float, g: int, bstar: float) -> float:
    """Step cost relative to a plain decode: 1 in the weight-bound regime,
    ramping to (1+g) once the batch saturates compute at B*."""
    return 1.0 + g * min(1.0, B / bstar)


def speedup(B: float, a: float, g: int, bstar: float) -> float:
    """Predicted speculative speedup at batch B: committed-tokens/step over the
    relative step cost."""
    return (1.0 + a * g) / cost_factor(B, g, bstar)


def crossover_batch_spec(a: float, g: int, bstar: float) -> float:
    """The batch where speedup crosses 1 (win -> loss): closed form a * Bstar.
    Below it spec wins, above it spec is a net loss."""
    return a * bstar


def regime_label(a: float) -> str:
    """Name the closest measured acceptance regime for readable reporting."""
    best = min(MEASURED_REGIMES, key=lambda kv: abs(kv[0] - a))
    return best[1]


def analyze(a: float, g: int, bstar: float, batches: list[int]) -> dict:
    """Full prediction for one acceptance rate: per-batch speedup, the closed-
    form crossover, and the first batch in the sweep that is a net loss."""
    xover = crossover_batch_spec(a, g, bstar)
    rows = []
    first_loss = None
    for B in batches:
        s = speedup(B, a, g, bstar)
        is_win = s > 1.0
        rows.append({
            "batch": B,
            "speedup": s,
            "cost_factor": cost_factor(B, g, bstar),
            "committed_per_step": 1.0 + a * g,
            "win": is_win,
        })
        if not is_win and first_loss is None:
            first_loss = B

    # Smallest integer batch that is a net loss, from the closed form:
    # speedup < 1  <=>  B > a*Bstar. So loss starts at floor(a*Bstar)+1.
    loss_from_batch = math.floor(xover) + 1  # >=1 always (xover>=0)
    if xover <= 0.0:
        verdict = "always loss"           # a=0: no accepted drafts, pure tax
    elif crossover_batch_spec(a, g, bstar) >= max(batches):
        verdict = "always win"            # within the swept batch range
    else:
        verdict = "flips"

    return {
        "acceptance": a,
        "draft_len": g,
        "bstar": bstar,
        "regime": regime_label(a),
        "crossover_batch": xover,          # closed form a*Bstar (fractional)
        "loss_from_batch": loss_from_batch,  # integer: net loss for batch >= this
        "first_loss_in_sweep": first_loss,
        "verdict": verdict,
        "rows": rows,
    }


def print_report(result: dict, batches: list[int]) -> None:
    a = result["acceptance"]
    g = result["draft_len"]
    print(f"\n== acceptance a={a:.2f}  draft g={g}  [{result['regime']}] ==")
    print(f"  committed tokens/step = 1 + a*g = {1.0 + a*g:.3f}   "
          f"(vs 1.0 for plain decode)")
    print(f"  {'B':>6} {'cost_x':>8} {'speedup':>9} {'verdict':>8}")
    for row in result["rows"]:
        tag = "WIN" if row["win"] else "loss"
        mark = ""
        # Flag the batch straddling the win->loss crossover.
        xb = result["crossover_batch"]
        if row["batch"] <= xb < row["batch"] * 2 and 0 < xb < max(batches):
            mark = "  <- crossover"
        print(f"  {row['batch']:>6} {row['cost_factor']:>8.3f} "
              f"{row['speedup']:>9.3f} {tag:>8}{mark}")

    if result["verdict"] == "always loss":
        print(f"  => a={a:.2f}: net LOSS at every batch >= 1 "
              f"(no accepted drafts to pay for the extra compute).")
    elif result["verdict"] == "always win":
        print(f"  => a={a:.2f}: still a WIN across the whole swept range "
              f"(crossover a*Bstar = {result['crossover_batch']:.1f} "
              f">= max batch {max(batches)}).")
    else:
        print(f"  => a={a:.2f}: crossover a*Bstar = "
              f"{result['crossover_batch']:.1f}  ->  net LOSS for batch >= "
              f"{result['loss_from_batch']}.")


def main():
    p = argparse.ArgumentParser(
        description="Analytical model of where speculative decoding flips from "
                    "a win to a net loss as batch size grows.")
    p.add_argument("--accept", type=float, nargs="+",
                   default=[0.05, 0.52, 0.92],
                   help="per-token draft acceptance rate(s) a in [0,1]. "
                        "Defaults span the project's measured regimes: "
                        "0.05 generic, 0.52 code, 0.92 grounded.")
    p.add_argument("--draft", type=int, default=8,
                   help="draft length g (extra tokens drafted per row).")
    p.add_argument("--bstar", type=float, default=None,
                   help="roofline crossover batch B*. Default: computed from "
                        "bench.roofline for Qwen2.5-0.5B at S=2048 fp16 (~39).")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="context length S used to derive B* (default 2048).")
    p.add_argument("--dtype-bytes", type=int, default=2,
                   help="bytes per weight/KV element used to derive B* (2=fp16).")
    p.add_argument("--batches", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128],
                   help="batch sizes to evaluate.")
    p.add_argument("--out", default="results/spec_cost.json",
                   help="path to write the JSON prediction.")
    a = p.parse_args()

    bstar = a.bstar if a.bstar is not None else default_bstar(a.seq_len, a.dtype_bytes)

    print(f"### nanoserve speculative-decoding cost model ###")
    print(f"  draft g={a.draft}   roofline crossover B*={bstar:.1f}  "
          f"(Qwen2.5-0.5B, S={a.seq_len}, fp16)")
    print(f"  model: speedup(B) = (1 + a*g) / (1 + g*min(1, B/B*))")

    results = []
    for acc in a.accept:
        res = analyze(acc, a.draft, bstar, a.batches)
        print_report(res, a.batches)
        results.append(res)

    # Headline: the generic-prose prediction is the number that pairs with a
    # measured batched-spec engine (the regime nearest a=0 / lowest acceptance).
    generic = min(results, key=lambda r: r["acceptance"])
    print(f"\n### HEADLINE ###")
    if generic["verdict"] == "always loss":
        print(f"  at acceptance a={generic['acceptance']:.2f} (generic prose), "
              f"speculative decoding is a net LOSS for batch >= 1.")
    elif generic["verdict"] == "always win":
        print(f"  at acceptance a={generic['acceptance']:.2f} (generic prose), "
              f"speculative decoding still WINS across batch <= "
              f"{max(a.batches)}.")
    else:
        print(f"  at acceptance a={generic['acceptance']:.2f} (generic prose), "
              f"speculative decoding is a net LOSS for batch >= "
              f"{generic['loss_from_batch']}.")

    grounded = max(results, key=lambda r: r["acceptance"])
    if grounded is not generic:
        gv = grounded["verdict"]
        if gv == "always win":
            print(f"  by contrast, at a={grounded['acceptance']:.2f} "
                  f"(grounded/RAG) it still wins across the swept range.")
        elif gv == "always loss":
            print(f"  even at a={grounded['acceptance']:.2f} "
                  f"(grounded/RAG) it is a net loss for batch >= 1.")
        else:
            print(f"  by contrast, at a={grounded['acceptance']:.2f} "
                  f"(grounded/RAG) it stays a win until batch >= "
                  f"{grounded['loss_from_batch']}.")

    out = {
        "model": "speedup(B) = (1 + a*g) / (1 + g*min(1, B/Bstar))",
        "draft_len": a.draft,
        "bstar": bstar,
        "seq_len": a.seq_len,
        "dtype_bytes": a.dtype_bytes,
        "batches": a.batches,
        "predictions": results,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

"""Predict the graphed engine's behavior at 1.5B and 3B before measuring it.

The prereg discipline (docs/prereg.md) exercised on free hardware: every
constant below is either a model-config fact or a number already measured at
0.5B, the predictions are committed to git before the 1.5B/3B runs exist,
and each prediction states the band that falsifies it.

Cost model, decode step through the CUDA-graph engine:

    t_step(m; B, ctx) = W_m / BW_w  +  t_o  +  B * ctx * kv_m / BW_kv

W_m = weight bytes read per step (all params, fp16), BW_w = effective weight
bandwidth, t_o = fixed per-step overhead the graph can't remove (replay
launch, buffer copies, sampling), BW_kv = effective bandwidth of the paged
KV reads (scattered 16-token blocks; far below streamed-weight bandwidth).

Calibration, all from 0.5B on the T4 (kernel-ci v15 graph_bench, full-ci v3
ladder):
  t(B=1,  ctx=128)  = 9.39 ms
  t(B=16, ctx=128)  = 9.35 ms   (flat in B at short ctx: weights dominate)
  t(B=16, ctx=1024) = 12.65 ms
  ladder peak       = 578.3 tok/s (rate 16, n 32, max_tokens 48)

BW_kv falls straight out of the ctx sweep. (BW_w, t_o) cannot be separated
with one model size, so P1 brackets t_o in [0, 4] ms and predicts an
interval; P2 is the sharper conditional prediction that only exists after
the 1.5B measurement pins the pair.

    python -m bench.scale_serve_predict
"""
from __future__ import annotations

import json
import os

from bench.roofline import estimate_params, kv_bytes_per_token
from bench.scale_predict import MODELS

DT = 2  # fp16

# measured 0.5B constants (provenance in the docstring)
T_B1_C128 = 9.39e-3
T_B16_C128 = 9.35e-3
T_B16_C1024 = 12.65e-3
PEAK_05 = 578.3
T_O_LO, T_O_HI = 0.0, 4.0e-3   # the bracket P1 lives with until 1.5B lands


def weight_bytes(tag):
    return estimate_params(MODELS[tag])["total"] * DT


def kv_tok(tag):
    return kv_bytes_per_token(MODELS[tag], DT)


def main():
    w05, kv05 = weight_bytes("0.5B"), kv_tok("0.5B")
    # KV bandwidth from the 0.5B ctx sweep at B=16
    dkv = 16 * (1024 - 128) * kv05
    bw_kv = dkv / (T_B16_C1024 - T_B16_C128)

    # weight-bandwidth bracket: each t_o hypothesis implies a BW_w
    def bw_w(t_o):
        kv_term = 1 * 128 * kv05 / bw_kv
        return w05 / (T_B1_C128 - t_o - kv_term)

    def t_step(tag, B, ctx, t_o):
        return (weight_bytes(tag) / bw_w(t_o) + t_o
                + B * ctx * kv_tok(tag) / bw_kv)

    # ladder-utilization factor from the 0.5B peak (arrivals, prefills, and
    # partial batches keep peak below B_max / t_step)
    u = PEAK_05 * T_B16_C128 / 16

    preds = {"calibration": {
        "bw_kv_gbs": bw_kv / 1e9, "bw_w_gbs_range":
        sorted([bw_w(T_O_HI) / 1e9, bw_w(T_O_LO) / 1e9]),
        "t_o_bracket_ms": [T_O_LO * 1e3, T_O_HI * 1e3],
        "ladder_utilization": u},
        "models": {}}

    print(f"calibration: BW_kv {bw_kv/1e9:.1f} GB/s; "
          f"BW_w in [{min(bw_w(T_O_LO), bw_w(T_O_HI))/1e9:.0f}, "
          f"{max(bw_w(T_O_LO), bw_w(T_O_HI))/1e9:.0f}] GB/s "
          f"for t_o in [{T_O_LO*1e3:.0f}, {T_O_HI*1e3:.0f}] ms; "
          f"ladder utilization {u:.2f}")
    print(f"\n{'model':>6} {'W GB':>6} {'kv B/tok':>9} "
          f"{'P1 t(B1,c128) ms':>17} {'t(B16,c1024) ms':>16} "
          f"{'P3 peak tok/s':>15}")
    for tag in ("1.5B", "3B"):
        # each t_o hypothesis is a coupled (t_o, BW_w) pair; the interval is
        # the envelope over the bracket
        cands1 = [t_step(tag, 1, 128, o) for o in (T_O_LO, T_O_HI)]
        cands2 = [t_step(tag, 16, 1024, o) for o in (T_O_LO, T_O_HI)]
        peak = [u * 16 / t_step(tag, 16, 150, o) for o in (T_O_LO, T_O_HI)]
        lo1, hi1 = min(cands1), max(cands1)
        lo2, hi2 = min(cands2), max(cands2)
        # scheduling band on the ladder prediction, per the prereg
        plo, phi = min(peak) * 0.7, max(peak) * 1.3
        preds["models"][tag] = {
            "weight_gb": weight_bytes(tag) / 1e9,
            "kv_per_token": kv_tok(tag),
            "P1_t_b1_c128_ms": [lo1 * 1e3, hi1 * 1e3],
            "t_b16_c1024_ms": [lo2 * 1e3, hi2 * 1e3],
            "P3_ladder_peak_toks": [plo, phi],
        }
        print(f"{tag:>6} {weight_bytes(tag)/1e9:>6.2f} {kv_tok(tag):>9} "
              f"{f'[{lo1*1e3:.1f}, {hi1*1e3:.1f}]':>17} "
              f"{f'[{lo2*1e3:.1f}, {hi2*1e3:.1f}]':>16} "
              f"{f'[{plo:.0f}, {phi:.0f}]':>15}")

    preds["P2"] = ("after 1.5B lands, solve (BW_w, t_o) exactly from the two "
                   "B=1 ctx=128 points (0.5B, 1.5B); the 3B B=1 ctx=128 "
                   "prediction from that pair must land within 15%")
    preds["P4"] = "graph peak >= 1.5x eager fused peak at every model size"
    preds["P5"] = ("direction only: nanoserve's percent of vLLM at 3B exceeds "
                   "the 34 percent measured at 0.5B (the remaining gap is "
                   "per-step overhead, which amortizes as the GEMMs grow)")
    print(f"\nP2: {preds['P2']}\nP4: {preds['P4']}\nP5: {preds['P5']}")

    os.makedirs("results", exist_ok=True)
    with open("results/scale_serve_predictions.json", "w") as f:
        json.dump(preds, f, indent=2)
    print("\nwrote results/scale_serve_predictions.json")


if __name__ == "__main__":
    main()

"""Judge the scale-serve predictions against what the T4 measured.

Reads the committed predictions (results/scale_serve_predictions.json) and
the per-model measurement files the scaleserve run produced, prints one
verdict per prediction, and writes results/scale_serve.json. The rules are
the ones in the predictions file; nothing here re-derives or re-fits except
P2, whose recipe (solve the pair on 0.5B+1.5B, then check 3B at 15%) was
itself preregistered.

    python -m bench.scale_serve_compare
"""
from __future__ import annotations

import json
import os

from bench.scale_serve_predict import (T_B1_C128, kv_tok, weight_bytes)

TAGS = {"0.5B": "Qwen/Qwen2.5-0.5B", "1.5B": "Qwen/Qwen2.5-1.5B",
        "3B": "Qwen/Qwen2.5-3B"}


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def graph_cell(tag, B, ctx):
    d = _load(f"results/graph_bench_{tag}.json")
    if not d:
        return None
    for r in d["rows"]:
        if r["batch"] == B and r["ctx"] == ctx:
            return r["graph_ms"]
    return None


def peaks(tag):
    d = _load(f"results/sweep_{tag}.json")
    if not d:
        return {}
    best = {}
    for r in d["runs"]:
        e = r["engine"]
        if e not in best or r["throughput"] > best[e]:
            best[e] = r["throughput"]
    return best


def main():
    preds = _load("results/scale_serve_predictions.json")
    out = {"verdicts": {}, "measured": {}}
    v = out["verdicts"]

    def judge(name, value, lo, hi):
        if value is None:
            verdict = "MISSING"      # a step failed; not evidence either way
        else:
            verdict = "HELD" if lo <= value <= hi else "FALSIFIED"
        v[name] = {"measured": value, "band": [lo, hi], "verdict": verdict}
        print(f"{name}: measured {value if value is None else round(value,1)} "
              f"vs [{lo:.1f}, {hi:.1f}] -> {verdict}")

    # P1 / t-tables
    for tag in ("1.5B", "3B"):
        m = preds["models"][tag]
        judge(f"P1 {tag} t(B1,c128) ms", graph_cell(tag, 1, 128),
              *m["P1_t_b1_c128_ms"])
        judge(f"   {tag} t(B16,c1024) ms", graph_cell(tag, 16, 1024),
              *m["t_b16_c1024_ms"])

    # P2: pin (BW_w, t_o) on 0.5B + 1.5B, check 3B at 15%
    t15 = graph_cell("1.5B", 1, 128)
    t3 = graph_cell("3B", 1, 128)
    if t15 and t3:
        bw_kv = preds["calibration"]["bw_kv_gbs"] * 1e9
        kvt = lambda tag: 1 * 128 * kv_tok(tag) / bw_kv
        # two equations: t = W/BW_w + t_o + kv_term
        a05 = T_B1_C128 - kvt("0.5B")
        a15 = t15 * 1e-3 - kvt("1.5B")
        bw_w = (weight_bytes("1.5B") - weight_bytes("0.5B")) / (a15 - a05)
        t_o = a05 - weight_bytes("0.5B") / bw_w
        pred3 = (weight_bytes("3B") / bw_w + t_o + kvt("3B")) * 1e3
        out["P2_fit"] = {"bw_w_gbs": bw_w / 1e9, "t_o_ms": t_o * 1e3,
                         "pred_3B_ms": pred3}
        print(f"\nP2 fit: BW_w {bw_w/1e9:.0f} GB/s, t_o {t_o*1e3:.2f} ms "
              f"-> 3B predicted {pred3:.1f} ms")
        judge("P2 3B t(B1,c128) ms (15% band)", t3,
              pred3 * 0.85, pred3 * 1.15)

    # P3 / P4 from the ladder sweeps
    print()
    for tag in ("1.5B", "3B"):
        p = peaks(tag)
        out["measured"][tag] = p
        g = p.get("paged_fused_graph")
        f_ = p.get("paged_fused")
        judge(f"P3 {tag} ladder peak tok/s", g,
              *preds["models"][tag]["P3_ladder_peak_toks"])
        if g and f_:
            ok = g >= 1.5 * f_
            v[f"P4 {tag} graph>=1.5x fused"] = {
                "measured": g / f_, "verdict": "HELD" if ok else "FALSIFIED"}
            print(f"P4 {tag}: graph/fused = {g/f_:.2f}x -> "
                  f"{'HELD' if ok else 'FALSIFIED'}")

    # P5: percent of vllm across scale, direction only
    print()
    ratios = {}
    for tag in TAGS:
        vl = _load(f"results/vllm_{tag}.json")
        p = peaks(tag) if tag != "0.5B" else peaks("0.5B")
        g = p.get("paged_fused_graph")
        if vl and vl.get("throughput") and g:
            ratios[tag] = g / vl["throughput"]
            print(f"   {tag}: nanoserve {g:.1f} / vllm "
                  f"{vl['throughput']:.1f} = {ratios[tag]*100:.0f}%")
    out["vllm_ratios"] = ratios
    if "3B" in ratios:
        base = ratios.get("0.5B", 0.34)
        ok = ratios["3B"] > base
        v["P5 ratio rises with scale"] = {
            "measured": ratios["3B"], "baseline": base,
            "verdict": "HELD" if ok else "FALSIFIED"}
        print(f"P5: {ratios['3B']*100:.0f}% at 3B vs {base*100:.0f}% at 0.5B "
              f"-> {'HELD' if ok else 'FALSIFIED'}")

    held = sum(1 for x in v.values() if x["verdict"] == "HELD")
    falsified = sum(1 for x in v.values() if x["verdict"] == "FALSIFIED")
    print(f"\n{held} held, {falsified} falsified, "
          f"{len(v) - held - falsified} missing of {len(v)}")
    os.makedirs("results", exist_ok=True)
    with open("results/scale_serve.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/scale_serve.json")


if __name__ == "__main__":
    main()

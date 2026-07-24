"""Roofline / analytical cost model. Predict throughput before measuring it.

A ceiling makes a measured number mean something: 40% of the roof is a
scheduling/overhead problem worth chasing, 95% is done.

Every kernel is bounded by either compute (peak FLOP/s) or memory bandwidth
(GB/s moving bytes from HBM into the compute units). Which one binds is set by
arithmetic intensity I = FLOPs / bytes-moved, against the ridge point
I* = peak_flops / bandwidth (A10: ~125 TFLOP/s / 0.6 TB/s ~= 208 FLOP/byte).
Below I* you're memory-bound (ALUs starve waiting for bytes), above it
compute-bound. That's the roof: a flat bandwidth ceiling ramping into a flat
compute ceiling at the ridge.

Decode is memory-bound. Decoding one token is a full forward pass at sequence
length 1: every weight is read from HBM for a single GEMV, a couple FLOPs per
weight byte. Intensity ~1-2 FLOP/byte, far below the ridge, so decode latency
is set by how fast you stream the weights (and KV cache) out of HBM, not by the
math. A decode step is a memory-copy of the model; batching amortizes that copy
across sequences (weights read once, reused for all B tokens).

Prefill is compute-bound. A P-token prompt multiplies the weights by a P-row
matrix, so each weight byte is reused P times: intensity ~ P, which for a few
hundred tokens sits above the ridge. Prefill is a dense GEMM on the compute
roof. That asymmetry is why serving stacks schedule the two phases differently.

No model download needed: give it param count + KV bytes/token (or let it
estimate params from config dims) and it prints the ceilings, a batch sweep,
and with --measured a predicted-vs-measured comparison.
"""
from __future__ import annotations

import argparse
import json

# ---------------------------------------------------------------------------
# Hardware presets. Bandwidth is HBM read (GB/s, 1 GB = 1e9 bytes). Peak FLOP/s
# is dense fp16/bf16 tensor-core (TFLOP/s). These two set the roofs; the rest is
# model geometry.
# ---------------------------------------------------------------------------
MEM_BW_GBPS = {
    "H100": 3350.0,   # SXM HBM3
    "A100": 2039.0,   # 80GB HBM2e
    "A10": 600.0,     # GDDR6
    "L4": 300.0,      # GDDR6, Ada
    "CPU": 100.0,     # a fast dual-socket DDR5 box, order-of-magnitude
}
PEAK_TFLOPS = {       # dense fp16 tensor-core, no sparsity
    "H100": 989.0,
    "A100": 312.0,
    "A10": 125.0,
    "L4": 121.0,
    "CPU": 2.0,
}

# Qwen2.5-0.5B geometry (from its config.json). Default so `python -m
# bench.roofline` runs with no args and no model load.
QWEN_0_5B = dict(
    name="Qwen2.5-0.5B",
    num_hidden_layers=24,
    hidden_size=896,
    num_attention_heads=14,
    num_key_value_heads=2,     # GQA: 2 KV heads shared across 14 query heads
    head_dim=64,               # 896 / 14
    intermediate_size=4864,
    vocab_size=151936,
    tie_word_embeddings=True,  # 0.5B ties input/output embeddings
    hf_param_count=494_032_768,  # published ~0.494B, for the estimate check
)

GB = 1e9   # bandwidth uses decimal GB (vendor convention)


# ---------------------------------------------------------------------------
# Model geometry -> parameter count and KV bytes/token
# ---------------------------------------------------------------------------
def estimate_params(cfg: dict) -> dict:
    """Count parameters exactly from config dims: GQA (K/V project to
    num_kv_heads*head_dim, not hidden), the 3-matrix SwiGLU MLP, tied-vs-untied
    embeddings. Returns a breakdown so the estimate is auditable.

    The folklore "params ~= 12 * layers * hidden^2" assumes intermediate_size =
    4*hidden, MHA, and no embedding; printed alongside for sanity.
    """
    L = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    n_q = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_q)
    hd = cfg.get("head_dim", H // n_q)
    inter = cfg["intermediate_size"]
    vocab = cfg["vocab_size"]
    tied = cfg.get("tie_word_embeddings", True)

    q_dim = n_q * hd            # query projection output width
    kv_dim = n_kv * hd          # key/value projection width (smaller under GQA)

    # Per-layer attention: q, k, v, o projections. Biases/LayerNorms are a
    # rounding error at this scale, omitted to keep the count transparent.
    attn = H * q_dim + H * kv_dim + H * kv_dim + q_dim * H
    # Per-layer SwiGLU MLP: gate + up (H->inter) and down (inter->H).
    mlp = H * inter + H * inter + inter * H
    per_layer = attn + mlp

    embed = vocab * H                       # input embedding table
    lm_head = 0 if tied else vocab * H      # untied output projection, if any

    total = L * per_layer + embed + lm_head
    folklore = 12 * L * H * H               # the 12*L*H^2 rule of thumb

    return {
        "per_layer": per_layer,
        "attn_per_layer": attn,
        "mlp_per_layer": mlp,
        "layers_total": L * per_layer,
        "embedding": embed,
        "lm_head": lm_head,
        "total": total,
        "folklore_12LH2": folklore,
    }


def kv_bytes_per_token(cfg: dict, dtype_bytes: int = 2) -> int:
    """KV cache bytes for one token, summed over layers:
    2 (K and V) * layers * kv_heads * head_dim * dtype_bytes.
    Mirrors server.paged_cache.kv_bytes_per_token but from a plain config dict,
    no model load."""
    L = cfg["num_hidden_layers"]
    n_kv = cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    hd = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    return 2 * L * n_kv * hd * dtype_bytes


def config_from_runner(runner) -> dict:
    """Pull the same geometry out of a loaded server.model.ModelRunner, for when
    you already have one in hand."""
    c = runner.model.config
    n_q = c.num_attention_heads
    return dict(
        name=getattr(runner, "model_name", "loaded-model"),
        num_hidden_layers=c.num_hidden_layers,
        hidden_size=c.hidden_size,
        num_attention_heads=n_q,
        num_key_value_heads=getattr(c, "num_key_value_heads", n_q),
        head_dim=getattr(c, "head_dim", c.hidden_size // n_q),
        intermediate_size=c.intermediate_size,
        vocab_size=c.vocab_size,
        tie_word_embeddings=getattr(c, "tie_word_embeddings", True),
    )


# ---------------------------------------------------------------------------
# The roofline predictions
# ---------------------------------------------------------------------------
def decode_step_latency_s(params, kv_per_tok, B, S, mem_bw_bytes_s, dtype_bytes=2):
    """Time for one decode step (all B sequences advance one token).

    A decode step streams out of HBM, once:
      (a) every weight             W = params * dtype_bytes
      (b) the KV cache it attends over  = B * S * kv_per_tok
    Compute is negligible (GEMV, intensity ~1), so latency = bytes / bandwidth.
    Weights are read once regardless of batch size, which is why batching helps.
    """
    W = params * dtype_bytes
    bytes_moved = W + B * S * kv_per_tok
    return bytes_moved / mem_bw_bytes_s


def decode_throughput_tok_s(params, kv_per_tok, B, S, mem_bw_bytes_s, dtype_bytes=2):
    """Decode throughput ceiling = tokens produced per step / step latency.
    Each step produces B tokens (one per sequence)."""
    lat = decode_step_latency_s(params, kv_per_tok, B, S, mem_bw_bytes_s, dtype_bytes)
    return B / lat


def crossover_batch(params, kv_per_tok, S, dtype_bytes=2):
    """Batch size where KV-cache traffic equals weight traffic:
        B * S * kv_per_tok = W = params * dtype_bytes
    Below it decode is weight-bound: the weight read dominates, so throughput
    grows ~linearly with B (amortizing a fixed W over more tokens). Above it
    decode is KV-bound: KV traffic grows with B*S and cancels the B in the
    numerator, so throughput saturates and more batch stops helping.
    """
    W = params * dtype_bytes
    return W / (S * kv_per_tok)


def prefill_latency_s(params, P, peak_flops):
    """Prefill a P-token prompt ~= one forward pass over a P-row activation:
    ~2 * params * P FLOPs (the 2 is multiply+add). Dense GEMM on the compute
    roof, so latency = FLOPs / peak_flops."""
    return 2 * params * P / peak_flops


def arithmetic_intensity(params, kv_per_tok, B, S, dtype_bytes=2):
    """FLOP per byte for a decode step.
    decode FLOPs ~= 2 * params * B (B GEMVs); bytes = W + B*S*kv_per_tok.
    Tiny intensity (<< ridge) means memory-bound."""
    W = params * dtype_bytes
    flops = 2 * params * B
    bytes_moved = W + B * S * kv_per_tok
    return flops / bytes_moved


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_int(n: float) -> str:
    return f"{n:,.0f}"


def print_param_report(cfg: dict):
    est = estimate_params(cfg)
    print(f"== Parameter estimate: {cfg.get('name', 'model')} ==")
    print(f"  layers={cfg['num_hidden_layers']}  hidden={cfg['hidden_size']}  "
          f"heads={cfg['num_attention_heads']}  kv_heads="
          f"{cfg.get('num_key_value_heads', cfg['num_attention_heads'])}  "
          f"head_dim={cfg.get('head_dim', cfg['hidden_size']//cfg['num_attention_heads'])}  "
          f"inter={cfg['intermediate_size']}  vocab={cfg['vocab_size']}  "
          f"tied={cfg.get('tie_word_embeddings', True)}")
    print(f"  per-layer params   : {_fmt_int(est['per_layer'])}  "
          f"(attn {_fmt_int(est['attn_per_layer'])} + mlp {_fmt_int(est['mlp_per_layer'])})")
    print(f"  all layers         : {_fmt_int(est['layers_total'])}")
    print(f"  embedding          : {_fmt_int(est['embedding'])}"
          f"{'  (tied, counted once)' if not est['lm_head'] else ''}")
    if est["lm_head"]:
        print(f"  lm_head (untied)   : {_fmt_int(est['lm_head'])}")
    print(f"  TOTAL params       : {_fmt_int(est['total'])}  "
          f"({est['total']/1e9:.3f} B)")
    print(f"  folklore 12*L*H^2  : {_fmt_int(est['folklore_12LH2'])}  "
          f"({est['folklore_12LH2']/1e9:.3f} B)  [rule-of-thumb, ignores embed/GQA/MLP width]")
    hf = cfg.get("hf_param_count")
    if hf:
        gap = 100.0 * (est["total"] - hf) / hf
        print(f"  published count    : {_fmt_int(hf)}  "
              f"({hf/1e9:.3f} B)  -> estimate off by {gap:+.1f}%")
    return est["total"]


def print_ceilings(params, kv_per_tok, mem_bw_gbps, peak_tflops, S, dtype_bytes=2):
    W = params * dtype_bytes
    mem_bw = mem_bw_gbps * GB
    peak_flops = peak_tflops * 1e12
    ridge = peak_flops / mem_bw
    print(f"\n== Roofs ==")
    print(f"  weights W          : {W/GB:.3f} GB  ({_fmt_int(params)} params x {dtype_bytes}B)")
    print(f"  KV per token       : {_fmt_int(kv_per_tok)} B  ({kv_per_tok/1024:.1f} KiB)")
    print(f"  mem bandwidth      : {mem_bw_gbps:.0f} GB/s")
    print(f"  peak compute       : {peak_tflops:.0f} TFLOP/s (fp16 dense)")
    print(f"  ridge point I*     : {ridge:.0f} FLOP/byte  "
          f"(above=compute-bound, below=memory-bound)")
    I1 = arithmetic_intensity(params, kv_per_tok, 1, S, dtype_bytes)
    print(f"  decode intensity   : {I1:.2f} FLOP/byte at B=1,S={S}  "
          f"-> {I1/ridge*100:.2f}% of ridge => MEMORY-BOUND")


def print_batch_sweep(params, kv_per_tok, mem_bw_gbps, S, batches, dtype_bytes=2):
    mem_bw = mem_bw_gbps * GB
    W = params * dtype_bytes
    b_cross = crossover_batch(params, kv_per_tok, S, dtype_bytes)
    print(f"\n== Decode throughput ceiling vs batch  (S={S} ctx, "
          f"{mem_bw_gbps:.0f} GB/s) ==")
    print(f"  crossover batch B* = W/(S*kv_per_tok) = {b_cross:.1f}  "
          f"(weight-bound below, KV-bound above)")
    print(f"  {'B':>5} {'step ms':>9} {'tok/s':>10} {'KV/W':>7} {'regime':>13}")
    for B in batches:
        lat = decode_step_latency_s(params, kv_per_tok, B, S, mem_bw, dtype_bytes)
        tps = B / lat
        kv_over_w = (B * S * kv_per_tok) / W
        regime = "weight-bound" if B < b_cross else "KV-bound"
        mark = "  <- crossover" if (B < b_cross <= B * 2 or
                                    (batches and B == min(batches, key=lambda x: abs(x - b_cross)))) else ""
        print(f"  {B:>5} {lat*1e3:>9.2f} {tps:>10.0f} {kv_over_w:>7.2f} {regime:>13}{mark}")
    return b_cross


def print_prefill(params, peak_tflops, prompt_lens):
    peak_flops = peak_tflops * 1e12
    print(f"\n== Prefill latency (compute-bound, {peak_tflops:.0f} TFLOP/s) ==")
    print(f"  {'P tokens':>9} {'GFLOPs':>10} {'latency ms':>12} {'tok/s':>10}")
    for P in prompt_lens:
        lat = prefill_latency_s(params, P, peak_flops)
        gflops = 2 * params * P / 1e9
        print(f"  {P:>9} {gflops:>10.1f} {lat*1e3:>12.2f} {P/lat:>10.0f}")


def overlay_measured(params, kv_per_tok, mem_bw_gbps, S, B, path, dtype_bytes=2):
    """Compare predicted decode ceiling to measured throughput from a bench
    sweep JSON (schema: {"runs":[{engine,device,rate,throughput,...}]}).

    Measured 'throughput' is whole-run output tok/s; the prediction is the
    decode ceiling at batch B, context S. The gap is headroom the
    scheduler/overheads leave on the table. Runs here are CPU, so a large gap is
    expected; the model just gives the yardstick."""
    with open(path) as f:
        data = json.load(f)
    runs = data.get("runs", [])
    mem_bw = mem_bw_gbps * GB
    predicted = decode_throughput_tok_s(params, kv_per_tok, B, S, mem_bw, dtype_bytes)
    print(f"\n== Predicted vs measured  ({path}) ==")
    print(f"  prediction: decode ceiling at B={B}, S={S}, "
          f"{mem_bw_gbps:.0f} GB/s = {predicted:,.0f} tok/s")
    print(f"  {'engine':>11} {'device':>7} {'rate':>6} {'measured':>10} "
          f"{'predicted':>10} {'gap%':>8}")
    for r in runs:
        meas = r.get("throughput", 0.0)
        gap = 100.0 * (predicted - meas) / predicted if predicted else 0.0
        print(f"  {r.get('engine',''):>11} {r.get('device',''):>7} "
              f"{r.get('rate',0):>6.1f} {meas:>10.1f} {predicted:>10.0f} {gap:>7.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Roofline cost model for LLM prefill/decode. Predicts "
                    "throughput ceilings without loading a model.")
    p.add_argument("--params", type=float, default=None,
                   help="param count (e.g. 0.494e9). Default: estimated from the "
                        "Qwen2.5-0.5B preset config.")
    p.add_argument("--kv-per-token", type=int, default=None,
                   help="KV cache bytes/token. Default: computed from preset config.")
    p.add_argument("--device", default="A10", choices=list(MEM_BW_GBPS.keys()),
                   help="hardware preset for bandwidth+compute (default A10).")
    p.add_argument("--mem-bandwidth-gbps", type=float, default=None,
                   help="override HBM bandwidth in GB/s.")
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="override peak fp16 compute in TFLOP/s.")
    p.add_argument("--dtype-bytes", type=int, default=2,
                   help="bytes per weight/KV element (2=fp16, 4=fp32).")
    p.add_argument("--batch", type=int, default=None,
                   help="batch size for the single-point prediction / overlay.")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="context length S the KV cache is sized for (default 2048).")
    p.add_argument("--measured", default=None,
                   help="path to a sweep JSON to overlay predicted-vs-measured.")
    a = p.parse_args()

    cfg = dict(QWEN_0_5B)

    # Resolve hardware roofs (explicit override wins over the device preset).
    mem_bw_gbps = a.mem_bandwidth_gbps or MEM_BW_GBPS[a.device]
    peak_tflops = a.peak_tflops or PEAK_TFLOPS[a.device]

    print(f"### nanoserve roofline model   device={a.device}  "
          f"dtype={a.dtype_bytes}B\n")

    # Params: use the estimate unless overridden.
    if a.params is not None:
        params = int(a.params)
        print(f"== Parameters (given) ==\n  {_fmt_int(params)}  "
              f"({params/1e9:.3f} B)")
    else:
        params = print_param_report(cfg)

    # KV bytes/token: compute from config unless overridden.
    if a.kv_per_token is not None:
        kv_per_tok = a.kv_per_token
    else:
        kv_per_tok = kv_bytes_per_token(cfg, a.dtype_bytes)

    S = a.seq_len
    print_ceilings(params, kv_per_tok, mem_bw_gbps, peak_tflops, S, a.dtype_bytes)

    batches = [1, 4, 8, 16, 32, 64]
    b_cross = print_batch_sweep(params, kv_per_tok, mem_bw_gbps, S, batches, a.dtype_bytes)

    print_prefill(params, peak_tflops, [128, 512, 2048])

    if a.measured:
        B = a.batch or 16
        overlay_measured(params, kv_per_tok, mem_bw_gbps, S, B, a.measured, a.dtype_bytes)
    elif a.batch:
        mem_bw = mem_bw_gbps * GB
        tps = decode_throughput_tok_s(params, kv_per_tok, a.batch, S, mem_bw, a.dtype_bytes)
        print(f"\n== Single-point prediction ==")
        print(f"  B={a.batch}, S={S}, {mem_bw_gbps:.0f} GB/s "
              f"-> {tps:,.0f} tok/s decode ceiling")

    print(f"\nTakeaway: decode is memory-bound; throughput scales ~linearly with "
          f"batch up to B*={b_cross:.0f}, then the KV cache read saturates HBM "
          f"and it flattens. Batch past B* only if you also cut S (context) or "
          f"KV bytes (quantize KV / fewer KV heads).")


if __name__ == "__main__":
    main()

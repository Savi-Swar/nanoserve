"""Scale axis: do the audit findings hold beyond 0.5B?

Every other study here pins one model (Qwen2.5-0.5B) and varies the serving
knob. The reviewer objection is that 0.5B is a toy: its tiny KV cache, aggressive
GQA, and habit of looping on repetitive text could all be artifacts of size.
This reruns a small comparable slice of the audit at 0.5B / 1.5B / 3B so each
finding gets a trend line instead of a single point.

No new physics; reuses the existing studies' logic verbatim:

  (a) spec decoding tokens/forward, generic vs grounded  -> server.speculative
  (b) prefix-cache prefill_saved on a shared system prompt -> server.prefix_cache
  (c) 8-bit KV perplexity delta vs fp16                   -> bench.kv_quant_study
  (d) predicted weight->KV crossover batch B*             -> bench.roofline

Kept light (short prompts, few generated tokens) so three sizes finish in a few
minutes on a T4.

    python -m bench.scale_study --device cuda \
        --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B
"""
from __future__ import annotations

import argparse
import json
import os

from server.model import ModelRunner
from server.prefix_cache import PrefixCache
from server.speculative import SpeculativeEngine
from server.paged_cache import kv_bytes_per_token
from server.request import Request, SamplingParams

from bench.kv_quant_study import perplexity
from bench.roofline import estimate_params, config_from_runner, crossover_batch

# --- fixed, comparable workloads (small on purpose) -------------------------

# (a) speculative decoding. GENERIC = novel prose with no repeated n-grams for
# prompt-lookup to latch onto; GROUNDED = a repetitive list to continue, where
# the last n-gram already appeared verbatim.
SPEC_GENERIC = ("Explain the theory of general relativity in a detailed "
                "paragraph, describing how mass curves spacetime.")
SPEC_GROUNDED = ("Repeat this list exactly: alpha beta gamma delta alpha beta "
                 "gamma delta alpha beta gamma delta alpha beta gamma")
SPEC_N = 32  # tokens generated per prompt

# (b) prefix caching. A shared system prompt, then a second request reusing it
# verbatim; prefill_saved is the fraction of prompt tokens skipped.
SYSTEM = ("You are a helpful, precise assistant. Read the user's question "
          "carefully, think step by step, and answer concisely and correctly. ")
PREFIX_PROMPTS = [
    SYSTEM + "Question: What is the capital of France?",
    SYSTEM + "Question: How many legs does a spider have?",
]

# (c) KV quantization. A fixed held-out passage; perplexity delta at 8-bit.
PPL_TEXT = (
    "The transformer processes a sequence in parallel with self-attention, where "
    "each token attends to every earlier token to build a contextual representation. "
    "During generation the model runs one token at a time, and the keys and values "
    "of all previous tokens are cached so they are not recomputed. This key-value "
    "cache is what dominates memory at inference time, and it grows with both the "
    "batch size and the sequence length."
)

# (d) roofline preset: T4-class HBM bandwidth and the context length the KV
# cache is sized for. B* = W / (S * kv_per_tok) is bandwidth-independent (dtype
# cancels in the ratio); the bandwidth only names the deployment described.
T4_MEM_BW_GBPS = 320.0
ROOFLINE_SEQ_LEN = 2048


def spec_tokens_per_forward(m, prompt, n):
    """Run one prompt through the speculative engine synchronously and return
    its tokens-per-forward-pass (naive baseline is always 1.0)."""
    eng = SpeculativeEngine(m)
    req = Request(0, prompt, SamplingParams(max_tokens=n, temperature=0.0,
                                            ignore_eos=True))
    eng._process(req)  # synchronous, fills req.output_tokens
    return eng.stats()["tokens_per_forward"]


def prefix_saved(m, prompts):
    """Prime a PrefixCache with the first prompt, then a second sharing its
    prefix; return the fraction of prompt tokens skipped across both."""
    cache = PrefixCache(m)
    for p in prompts:
        cache.prefill(m.encode(p))
    return cache.stats()["prefill_saved"]


def kv8_ppl_delta(m):
    """fp16-baseline perplexity, 8-bit-KV perplexity, and their gap on a fixed
    passage. 8-bit KV should barely move it; this checks that holds up-scale."""
    ids = m.encode(PPL_TEXT)
    ppl_fp16 = perplexity(m, ids, None)
    ppl_8 = perplexity(m, ids, 8)
    return ppl_fp16, ppl_8, ppl_8 - ppl_fp16


def predicted_crossover(m):
    """Roofline B* where KV traffic overtakes the weight read: the batch past
    which decode stops scaling. Params from config estimate, KV bytes/token from
    paged-cache accounting, both at the model's dtype (which cancels)."""
    cfg = config_from_runner(m)
    params = estimate_params(cfg)["total"]
    kv_per_tok = kv_bytes_per_token(m)
    dtype_bytes = m.dtype.itemsize
    b_star = crossover_batch(params, kv_per_tok, ROOFLINE_SEQ_LEN, dtype_bytes)
    return params, kv_per_tok, b_star


def audit_model(model_name, device):
    print(f"\n=== {model_name} (device={device}) ===")
    m = ModelRunner(model_name, device=device)
    m.warmup()

    sg = spec_tokens_per_forward(m, SPEC_GENERIC, SPEC_N)
    sr = spec_tokens_per_forward(m, SPEC_GROUNDED, SPEC_N)
    print(f"  spec tokens/forward : generic {sg:.2f}   grounded {sr:.2f}")

    ps = prefix_saved(m, PREFIX_PROMPTS)
    print(f"  prefix prefill_saved: {ps*100:.0f}%")

    ppl_fp16, ppl_8, ppl_delta = kv8_ppl_delta(m)
    print(f"  8-bit KV perplexity : fp16 {ppl_fp16:.2f} -> 8bit {ppl_8:.2f} "
          f"(delta {ppl_delta:+.3f})")

    params, kv_per_tok, b_star = predicted_crossover(m)
    print(f"  roofline B*         : {b_star:.1f}  "
          f"(params {params/1e9:.3f}B, KV {kv_per_tok/1024:.1f} KiB/tok, "
          f"S={ROOFLINE_SEQ_LEN}, {T4_MEM_BW_GBPS:.0f} GB/s)")

    return {
        "model": model_name,
        "device": device,
        "spec_generic_tpf": sg,
        "spec_grounded_tpf": sr,
        "prefix_saved": ps,
        "ppl_fp16": ppl_fp16,
        "ppl_8bit": ppl_8,
        "ppl_delta_8bit": ppl_delta,
        "params": params,
        "kv_bytes_per_token": kv_per_tok,
        "roofline_seq_len": ROOFLINE_SEQ_LEN,
        "roofline_mem_bw_gbps": T4_MEM_BW_GBPS,
        "predicted_crossover_batch": b_star,
    }


def short_name(model_name):
    return model_name.split("/")[-1]


def print_table(results):
    """Rows = audit metrics, columns = model sizes."""
    cols = [short_name(r["model"]) for r in results]
    w0 = 24
    wc = max(14, max((len(c) for c in cols), default=14) + 2)

    def row(label, cells):
        print(f"{label:<{w0}}" + "".join(f"{c:>{wc}}" for c in cells))

    print("\n" + "=" * (w0 + wc * len(cols)))
    print("SCALE AXIS: audit metrics across model sizes")
    print("=" * (w0 + wc * len(cols)))
    row("metric", cols)
    print("-" * (w0 + wc * len(cols)))
    row("spec tok/fwd (generic)", [f"{r['spec_generic_tpf']:.2f}" for r in results])
    row("spec tok/fwd (grounded)", [f"{r['spec_grounded_tpf']:.2f}" for r in results])
    row("prefix prefill_saved", [f"{r['prefix_saved']*100:.0f}%" for r in results])
    row("8bit KV ppl delta", [f"{r['ppl_delta_8bit']:+.3f}" for r in results])
    row("predicted crossover B*", [f"{r['predicted_crossover_batch']:.1f}" for r in results])
    print("-" * (w0 + wc * len(cols)))
    row("params (B)", [f"{r['params']/1e9:.3f}" for r in results])
    row("KV KiB/token", [f"{r['kv_bytes_per_token']/1024:.1f}" for r in results])
    print("=" * (w0 + wc * len(cols)))
    print("Read across a row: does the 0.5B finding hold as the model grows? "
          "Spec stays workload-bound (grounded >> generic at every size); "
          "prefix caching keeps saving; 8-bit KV stays cheap; B* shifts with "
          "the params/KV-bytes ratio.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+",
                   default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
                            "Qwen/Qwen2.5-3B"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/scale.json")
    a = p.parse_args()

    results = [audit_model(name, a.device) for name in a.models]
    print_table(results)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

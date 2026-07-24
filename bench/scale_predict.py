"""Predict the moving crossover across model scale — GPU-free.

The speculative-decoding win->loss crossover is B = a * B*, where B* = W /
(S * kv_per_token) is the roofline weight->KV crossover (bench/roofline.py) and
`a` is draft acceptance. Both terms move with model size, but not together:
weights grow ~H^2 (the attention/MLP projections), while KV per token grows ~H
(kv_heads * head_dim, held fixed under GQA). So W / kv_per_token ~ H, and **B*
rises with scale** — which drags the spec crossover along with it.

That matters because "speculative decoding inverts under batching on generic
traffic" is the most scale-sensitive claim in this repo: draft cost vs target
cost is exactly the ratio people expect to move with model size. The cost model
predicts where the crossover lands at *every* size, before a GPU runs. This is
the prediction; bench/spec_batched_study.py measures it per model.

    python -m bench.scale_predict
"""
from __future__ import annotations

from bench.roofline import crossover_batch, estimate_params, kv_bytes_per_token

# Real Qwen2.5 configs (head_dim = hidden_size / num_attention_heads when null).
MODELS = {
    "0.5B": dict(num_hidden_layers=24, hidden_size=896, num_attention_heads=14,
                 num_key_value_heads=2, head_dim=64, intermediate_size=4864,
                 vocab_size=151936, tie_word_embeddings=True),
    "1.5B": dict(num_hidden_layers=28, hidden_size=1536, num_attention_heads=12,
                 num_key_value_heads=2, head_dim=128, intermediate_size=8960,
                 vocab_size=151936, tie_word_embeddings=True),
    "3B": dict(num_hidden_layers=36, hidden_size=2048, num_attention_heads=16,
               num_key_value_heads=2, head_dim=128, intermediate_size=11008,
               vocab_size=151936, tie_word_embeddings=True),
}

# Acceptance measured at 0.5B (bench/spec_study.py). Prompt-lookup acceptance is
# a property of the *text* (does an n-gram repeat?), not the model, so it's held
# across sizes -- and whether that holds is itself a prediction the GPU checks.
WORKLOADS = {"generic (a=0.05)": 0.05, "code (a=0.52)": 0.52, "grounded (a=0.92)": 0.92}
S = 2048    # decode context length
DT = 2      # fp16 bytes


def main():
    print(f"### scale axis: predicted moving crossover   (S={S}, fp16)\n")
    print(f"  {'model':>5}  {'params':>8}  {'kv/tok':>8}  {'W (GB)':>7}  {'B* roofline':>12}")
    b = {}
    for name, cfg in MODELS.items():
        p = estimate_params(cfg)["total"]
        kv = kv_bytes_per_token(cfg, DT)
        bstar = crossover_batch(p, kv, S, DT)
        b[name] = bstar
        print(f"  {name:>5}  {p / 1e9:>6.3f}B  {kv:>7}B  {p * DT / 1e9:>6.2f}  {bstar:>11.1f}")

    print(f"\n  spec win->loss crossover  B = a * B*  (net loss for batches above):")
    print(f"  {'workload':>17}   " + "   ".join(f"{n:>5}" for n in MODELS))
    for wname, a in WORKLOADS.items():
        cells = "   ".join(f"{a * b[n]:>5.1f}" for n in MODELS)
        print(f"  {wname:>17}   {cells}")

    gen = ", ".join(f"{0.05 * b[n]:.0f}" for n in MODELS)
    grd = ", ".join(f"{0.92 * b[n]:.0f}" for n in MODELS)
    print(f"""
  Reading: B* rises with scale (weights ~H^2, KV ~H, so W/KV ~ H), so every
  crossover moves right. But the generic-prose crossover stays in the single
  digits ({gen}), far below any real serving batch (16-64) -- so "spec decoding
  is a net loss on batched generic traffic" holds at 1.5B and 3B, not just 0.5B.
  The grounded win extends further ({grd}).
  Measure per model:  python -m bench.spec_batched_study --device cuda
""")


if __name__ == "__main__":
    main()

"""Audit row #1 — speculative decoding (prompt-lookup).

Measures tokens-per-forward-pass (the deterministic speedup proxy: naive is
always 1.0) across prompt classes, and asserts spec output is token-identical
to naive. The finding: the headline speedup is entirely a property of the
*workload*, not the method — grounded/repetitive text flies, generic
generation gets nothing.

    python -m bench.spec_study
"""
from __future__ import annotations

import argparse
import json
import os

from server.model import ModelRunner, sample
from server.request import Request, SamplingParams
from server.speculative import SpeculativeEngine

PROMPTS = {
    "grounded": [
        "Repeat this list exactly: alpha beta gamma delta alpha beta gamma delta alpha beta gamma",
        "Copy: red red blue blue green green red red blue blue green green red red",
    ],
    "code": [
        "def add(a, b):\n    return a + b\ndef sub(a, b):\n    return a - b\ndef mul(a, b):\n    return",
        "x = 1\ny = 2\nz = 3\nx = 1\ny = 2\nz = 3\nx = 1\ny = 2\nz =",
    ],
    "generic": [
        "Explain the theory of general relativity in a detailed paragraph.",
        "Describe the process of photosynthesis and why it matters for life.",
    ],
}


def naive(m, prompt, n):
    sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    ids = m.encode(prompt)
    logits, kv, cur = m.prefill(ids)
    toks = [sample(logits, sp)]
    while len(toks) < n:
        logits, kv, cur = m.decode(toks[-1], kv, cur)
        toks.append(sample(logits, sp))
    return toks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--ngram", type=int, default=3)
    p.add_argument("--draft", type=int, default=8)
    p.add_argument("--out", default="results/spec.json")
    a = p.parse_args()

    m = ModelRunner(a.model, device=a.device)
    m.warmup()

    out = {}
    all_exact = True
    print(f"{'class':<10}{'exact':<8}{'tokens/fwd':<12}{'accept_rate':<12}(naive=1.00 t/f)")
    for cls, prompts in PROMPTS.items():
        eng = SpeculativeEngine(m, ngram=a.ngram, draft=a.draft)  # shared -> aggregate stats
        exact = True
        for prompt in prompts:
            ref = naive(m, prompt, a.n)
            req = Request(0, prompt, SamplingParams(max_tokens=a.n, temperature=0.0, ignore_eos=True))
            eng._process(req)  # synchronous, no thread
            if req.output_tokens != ref:
                exact = False
        s = eng.stats()
        all_exact = all_exact and exact
        out[cls] = {"exact": exact, **s}
        print(f"{cls:<10}{'OK' if exact else 'FAIL':<8}{s['tokens_per_forward']:<12.2f}{s['draft_accept_rate']:<12.2f}")

    print("-" * 52)
    g = out["grounded"]["tokens_per_forward"]
    gen = out["generic"]["tokens_per_forward"]
    print(f"grounded {g:.1f}x fewer forward passes; generic {gen:.2f}x (i.e. no win).")
    print(f"exactness: {'PASS' if all_exact else 'FAIL'}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

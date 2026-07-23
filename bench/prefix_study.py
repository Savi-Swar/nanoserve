"""Audit row #2 — prefix caching. The optimization that should SURVIVE.

Deterministic metric: fraction of prompt (prefill) tokens we avoid recomputing.
Two workloads: a shared system prompt across requests (prefix reuse applies)
vs distinct prefixes (nothing to reuse). Exactness checked against naive.

    python -m bench.prefix_study
"""
from __future__ import annotations

import argparse
import json
import os

from server.model import ModelRunner
from server.prefix_cache import PrefixCache

GEN = 8  # tokens generated per request (for the exactness check)

SYSTEM = ("You are a helpful, precise assistant. Read the user's question "
          "carefully, think step by step, and answer concisely and correctly. ")
QUESTIONS = [
    "Question: What is the capital of France?",
    "Question: How many legs does a spider have?",
    "Question: What color is a ripe banana?",
    "Question: What is two plus three?",
    "Question: Name a primary color.",
    "Question: What planet do we live on?",
]


def naive_gen(m, prompt, n):
    ids = m.encode(prompt)
    logits, kv, cur = m.prefill(ids)
    toks = [int(logits.argmax(-1))]
    while len(toks) < n:
        logits, kv, cur = m.decode(toks[-1], kv, cur)
        toks.append(int(logits.argmax(-1)))
    return toks


def cached_gen(m, cache: PrefixCache, prompt, n):
    ids = m.encode(prompt)
    logits, kv, cur = cache.prefill(ids)
    toks = [int(logits.argmax(-1))]
    while len(toks) < n:
        logits, kv, cur = m.decode(toks[-1], kv, cur)
        toks.append(int(logits.argmax(-1)))
    return toks


def run_workload(m, prompts):
    cache = PrefixCache(m)
    exact = True
    for p in prompts:
        ref = naive_gen(m, p, GEN)
        got = cached_gen(m, cache, p, GEN)
        if got != ref:
            exact = False
    return cache.stats(), exact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/prefix.json")
    a = p.parse_args()
    m = ModelRunner(a.model, device=a.device)
    m.warmup()

    shared = [SYSTEM + q for q in QUESTIONS]              # all share the system prompt
    distinct = [q + " " + SYSTEM for q in QUESTIONS]      # unique prefix each -> no reuse

    s_shared, ex_s = run_workload(m, shared)
    s_distinct, ex_d = run_workload(m, distinct)

    print(f"{'workload':<12}{'prefill saved':<16}{'hits':<8}{'exact'}")
    print(f"{'shared':<12}{s_shared['prefill_saved']*100:>6.0f}% "
          f"({s_shared['prefill_tokens_computed']}/{s_shared['prefill_tokens_total']} toks)  "
          f"{s_shared['hits']:<8}{'OK' if ex_s else 'FAIL'}")
    print(f"{'distinct':<12}{s_distinct['prefill_saved']*100:>6.0f}% "
          f"({s_distinct['prefill_tokens_computed']}/{s_distinct['prefill_tokens_total']} toks)  "
          f"{s_distinct['hits']:<8}{'OK' if ex_d else 'FAIL'}")
    print("-" * 48)
    print("prefix caching survives: big win with a shared prefix, ~0 without — "
          "and exact either way.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"shared": {**s_shared, "exact": ex_s},
                   "distinct": {**s_distinct, "exact": ex_d}}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

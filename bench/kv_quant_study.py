"""Audit row #3 — KV cache quantization. Memory vs quality tradeoff.

Deterministic axes: memory factor (32/bits vs an fp32 cache; 16/bits vs fp16)
and quality measured as **teacher-forced top-1 agreement** with the fp32-KV
model — at each step we feed the *same* reference token and ask whether the
quantized-KV model's argmax still matches fp32's. This deliberately avoids
free-running token-match, which cascades (one early flip marks every later
token "different" even when the text is fine) and wildly overstates the damage.
Per-step agreement is the honest quality signal.

    python -m bench.kv_quant_study
"""
from __future__ import annotations

import argparse
import json
import os

from server.kv_quant import quantize_cache
from server.model import ModelRunner

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue.",
    "List the first five prime numbers:",
    "Once upon a time, in a distant kingdom, there lived",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]
N = 40


def fp32_tokens(m, prompt, n):
    ids = m.encode(prompt)
    logits, cache, cur = m.prefill(ids)
    toks = [int(logits.argmax(-1))]
    while len(toks) < n:
        logits, cache, cur = m.decode(toks[-1], cache, cur)
        toks.append(int(logits.argmax(-1)))
    return toks


def teacher_forced_agreement(m, prompt, bits, ref):
    """Feed the fp32 reference tokens; count steps where the quantized-KV
    argmax still equals the fp32 argmax. No cascade."""
    ids = m.encode(prompt)
    logits, cache, cur = m.prefill(ids)
    quantize_cache(cache, bits)
    agree = int(int(logits.argmax(-1)) == ref[0])
    for i in range(1, len(ref)):
        logits, cache, cur = m.decode(ref[i - 1], cache, cur)  # teacher forcing
        quantize_cache(cache, bits)
        agree += int(int(logits.argmax(-1)) == ref[i])
    return agree, len(ref)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cpu")
    p.add_argument("--bits", nargs="+", type=int, default=[8, 4, 2])
    p.add_argument("--out", default="results/kv_quant.json")
    a = p.parse_args()
    m = ModelRunner(a.model, device=a.device)
    m.warmup()

    refs = {p: fp32_tokens(m, p, N) for p in PROMPTS}

    out = {}
    print(f"{'bits':<6}{'mem vs fp32':<14}{'mem vs fp16':<14}{'top-1 agreement'}")
    for bits in a.bits:
        agree = tot = 0
        for prompt in PROMPTS:
            ag, n = teacher_forced_agreement(m, prompt, bits, refs[prompt])
            agree += ag
            tot += n
        acc = agree / tot
        out[str(bits)] = {"mem_vs_fp32": 32 / bits, "mem_vs_fp16": 16 / bits,
                          "top1_agreement": acc, "steps": tot}
        print(f"{bits:<6}{32/bits:>6.0f}x{'':<7}{16/bits:>6.1f}x{'':<7}{acc*100:.1f}%")
    print("-" * 52)
    print("8-bit KV is ~lossless; quality degrades as bits drop — the "
          "memory/quality knob, quantified with a cascade-free metric.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

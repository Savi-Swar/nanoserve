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
import math
import os

import torch

from server.kv_quant import quantize_cache
from server.model import ModelRunner

# A held-out English passage for perplexity (a real quality metric, unlike
# top-1 agreement which is a proxy a reviewer will poke at).
PPL_TEXT = (
    "The transformer processes a sequence in parallel with self-attention, where "
    "each token attends to every earlier token to build a contextual representation. "
    "During generation the model runs one token at a time, and the keys and values "
    "of all previous tokens are cached so they are not recomputed. This key-value "
    "cache is what dominates memory at inference time, and it grows with both the "
    "batch size and the sequence length. Quantizing the cache to fewer bits trades a "
    "little numerical precision for a large cut in that memory footprint, which is "
    "worth it only if the loss in output quality stays small."
)

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


def perplexity(m, ids, bits):
    """Teacher-forced perplexity of `ids` with the KV cache quantized to `bits`
    (bits=None = fp16 baseline). Lower is better; the fp16-vs-quantized gap is
    the real quality cost of the quantizer."""
    logits, cache, cur = m.prefill(ids[:1])
    if bits:
        quantize_cache(cache, bits)
    nll = 0.0
    for i in range(1, len(ids)):
        logp = torch.log_softmax(logits[0].float(), dim=-1)
        nll += -logp[ids[i]].item()
        logits, cache, cur = m.decode(ids[i], cache, cur)
        if bits:
            quantize_cache(cache, bits)
    return math.exp(nll / (len(ids) - 1))


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
    ppl_ids = m.encode(PPL_TEXT)
    ppl_fp16 = perplexity(m, ppl_ids, None)

    out = {"fp16_perplexity": ppl_fp16}
    print(f"fp16 baseline perplexity: {ppl_fp16:.2f}  ({len(ppl_ids)} tokens)\n")
    print(f"{'bits':<6}{'mem vs fp16':<13}{'top-1 agree':<13}{'perplexity':<13}{'ppl delta'}")
    for bits in a.bits:
        agree = tot = 0
        for prompt in PROMPTS:
            ag, n = teacher_forced_agreement(m, prompt, bits, refs[prompt])
            agree += ag
            tot += n
        acc = agree / tot
        ppl = perplexity(m, ppl_ids, bits)
        out[str(bits)] = {"mem_vs_fp32": 32 / bits, "mem_vs_fp16": 16 / bits,
                          "top1_agreement": acc, "steps": tot, "perplexity": ppl}
        print(f"{bits:<6}{16/bits:>5.1f}x{'':<7}{acc*100:>7.1f}%{'':<5}"
              f"{ppl:>8.2f}{'':<5}{ppl - ppl_fp16:>+7.2f}")
    print("-" * 58)
    print("perplexity is the metric a reviewer trusts; top-1 agreement is the fast "
          "proxy. 8-bit KV should barely move perplexity, low bits should blow it up.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

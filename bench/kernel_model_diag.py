"""Diagnose kernel-vs-sdpa divergence through the real model.

The model-level greedy test failed on the T4 while every op-level test passed.
Two very different causes look identical from a failed assert:

  (a) a real bug in the kernel's model integration (wrong mask, wrong scale,
      wrong layer wiring): divergence early, with a large logit gap
  (b) argmax tie-flips: the kernel's reduction order differs from sdpa's, so
      fp16 logits differ in the last bits and a near-tie flips; divergence
      late, with a tiny top-1/top-2 margin at the flip

This script runs greedy decode twice per prompt (sdpa, then kernel), teacher-
forcing the SDPA tokens through the kernel run so a single flip can't cascade,
and reports per step: argmax agreement, the top1-top2 margin at any flip, and
the max |logit delta| between implementations.

    python -m bench.kernel_model_diag --device cuda
"""
from __future__ import annotations

import argparse

import torch

from server.model import ModelRunner
from server.kernels.paged_attention_triton import (
    use_triton_attention, restore_attention, KERNEL_CALLS)

PROMPTS = [
    "The capital of France is",
    "In a distant galaxy, a small robot",
    "def fibonacci(n):",
    "The three laws of thermodynamics state",
]
N = 48


@torch.no_grad()
def greedy_logits(runner, prompt, n, force_tokens=None):
    """Greedy continuation; returns (tokens, per-step last-logits list).
    If force_tokens is given, feed those instead of own argmax (teacher
    forcing), so both runs see identical inputs at every step."""
    ids = runner.encode(prompt)
    logits, kv, cur = runner.prefill(ids)
    steps = [logits.float().cpu()]
    toks = [int(logits.argmax(-1))]
    for i in range(n - 1):
        feed = force_tokens[i] if force_tokens is not None else toks[-1]
        logits, kv, cur = runner.decode(feed, kv, cur)
        steps.append(logits.float().cpu())
        toks.append(int(logits.argmax(-1)))
    return toks, steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    m = ModelRunner(a.model, device=a.device)
    import server.kernels.paged_attention_triton as pat

    for prompt in PROMPTS:
        base_toks, base_steps = greedy_logits(m, prompt, N)

        calls0 = pat.KERNEL_CALLS
        prev = use_triton_attention(m.model)
        try:
            # teacher-forced on the sdpa tokens: divergences can't cascade
            kern_toks, kern_steps = greedy_logits(m, prompt, N,
                                                  force_tokens=base_toks)
        finally:
            restore_attention(m.model, prev)
        used_kernel = pat.KERNEL_CALLS > calls0

        flips = [i for i, (x, y) in enumerate(zip(base_toks, kern_toks)) if x != y]
        max_dl = max((kern_steps[i] - base_steps[i]).abs().max().item()
                     for i in range(len(base_steps)))
        print(f"\nprompt: {prompt!r}  (kernel path ran: {used_kernel})")
        print(f"  agreement: {N - len(flips)}/{N} steps   max |logit delta|: {max_dl:.4f}")
        for i in flips[:6]:
            top2 = base_steps[i].topk(2).values[0]
            margin = (top2[0] - top2[1]).item()
            kd = (kern_steps[i] - base_steps[i]).abs().max().item()
            print(f"  flip @ step {i:>3}: sdpa top1-top2 margin {margin:.5f}, "
                  f"|logit delta| at this step {kd:.4f}")
        if not flips:
            print("  no flips: token-identical under teacher forcing")


if __name__ == "__main__":
    main()

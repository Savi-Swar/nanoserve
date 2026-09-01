"""Feasibility probe for pipeline-parallel serving on 2x T4.

Qwen2.5-7B fp16 is 15.2 GB of weights; one T4 has ~14.6 GB usable, so the
model literally does not fit a single card. This probe answers the questions
the pillar hangs on, cheaply, before any engine work:

  1. does the 7B load split across both T4s (accelerate device_map)?
  2. what is the actual layer split and per-GPU memory after load?
  3. does greedy generation work through the split, and at what tok/s?
  4. where is the cut (which module boundary), for the KV-pool design?

    python scripts/pp_probe.py
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-7B"


def main():
    n = torch.cuda.device_count()
    print(f"cuda devices: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  {i}: {p.name} {p.total_memory/2**30:.1f} GiB")

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="balanced")
    print(f"loaded in {time.perf_counter()-t0:.0f}s")

    # the split, module by module
    seen = {}
    for name, dev in model.hf_device_map.items():
        seen.setdefault(dev, []).append(name)
    for dev, names in sorted(seen.items(), key=lambda x: str(x[0])):
        head = names[0]
        tail = names[-1]
        print(f"  device {dev}: {len(names)} modules ({head} .. {tail})")

    for i in range(n):
        print(f"  gpu{i} allocated {torch.cuda.memory_allocated(i)/2**30:.2f} GiB")

    ids = tok("The three laws of thermodynamics state", return_tensors="pt")
    ids = {k: v.to(model.device) for k, v in ids.items()}
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"64 tokens in {dt:.2f}s = {64/dt:.1f} tok/s (hf generate, naive pp)")
    print(f"output: {text[:120]!r}")
    for i in range(n):
        print(f"  gpu{i} peak {torch.cuda.max_memory_allocated(i)/2**30:.2f} GiB")


if __name__ == "__main__":
    main()

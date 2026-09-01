"""Why didn't async issue overlap the pipeline stages?

The interleaved engine issued two half-batch forwards back to back and got
zero overlap (37% GPU util unchanged, throughput down). The suspect is that
the step is python-issue-bound: on this host, walking 28 transformer layers
through accelerate's hooks costs as much CPU as the GPUs spend computing,
so the second half is issued only when the first is nearly done.

Measures, for the sharded 7B fused path at B=8:
  t_issue  time for forward_async to RETURN (pure python+launch cost)
  t_step   time for issue plus synchronize (what the GPUs need end to end)
  overlap-limit = t_issue / t_step: at ~1.0 async issue cannot overlap
  anything and threads are required; at ~0.0 issue order alone should work.

    python scripts/pp_diag.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from server.kernels.paged_attention_triton import (restore_attention,
                                                   use_triton_attention)
from server.paged_exec import PagedBatchState
from server.request import Request, SamplingParams

MODEL = "Qwen/Qwen2.5-7B"


def main():
    from server.model import ModelRunner
    m = ModelRunner(MODEL, device="pipeline")
    prev = use_triton_attention(m.model)
    try:
        state = PagedBatchState(m, num_blocks=2048, block_size=16, fused=True)
        ids = m.encode("The quick brown fox jumps over the lazy dog. " * 20)[:128]
        reqs = [Request(i, "", SamplingParams(max_tokens=512, temperature=0.0,
                                              ignore_eos=True),
                        prompt_ids=list(ids)) for i in range(8)]
        state.add(reqs)
        for _ in range(3):
            state.step()

        n = 20
        issue, total = 0.0, 0.0
        for _ in range(n):
            m.sync()
            t0 = time.perf_counter()
            logits = state.forward_async()
            t1 = time.perf_counter()
            m.sync()
            t2 = time.perf_counter()
            state.finish_async(logits)
            issue += t1 - t0
            total += t2 - t0
        issue /= n
        total /= n
        print(f"B=8 sharded fused step: issue {issue*1e3:.1f} ms, "
              f"issue+sync {total*1e3:.1f} ms")
        print(f"overlap-limit (issue/step): {issue/total:.2f}")
        if issue / total > 0.7:
            print("python-issue-bound: back-to-back async issue cannot "
                  "overlap; the second half must be issued from a second "
                  "thread so its python runs during the first half's GPU time")
        else:
            print("not issue-bound: overlap failure lies elsewhere "
                  "(look for a hidden sync in the forward)")
    finally:
        restore_attention(m.model, prev)


if __name__ == "__main__":
    main()

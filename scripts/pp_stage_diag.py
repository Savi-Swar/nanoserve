"""Localize the stage-graph divergence: eager vs replay, stage by stage.

The captured pipeline runs 3.3x faster and emits wrong tokens. This drives
one decode step through both paths from identical state and diffs at every
seam: stage0 hidden, the handoff, stage1 logits. Whichever seam first
disagrees owns the bug.

    python scripts/pp_stage_diag.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from server.kernels.paged_attention_triton import (restore_attention,
                                                   use_triton_attention)
from server.paged_exec import PagedBatchState
from server.request import Request, SamplingParams

MODEL = "Qwen/Qwen2.5-7B"
PROMPTS = ["The capital of France is", "def fibonacci(n):",
           "The three laws of thermodynamics state"]


def mad(a, b):
    return float((a.float() - b.float().to(a.device)).abs().max())


def main():
    from server.model import ModelRunner
    m = ModelRunner(MODEL, device="pipeline")
    prev = use_triton_attention(m.model)
    try:
        state = PagedBatchState(m, num_blocks=2048, block_size=16, fused=True,
                                graphs=True, graph_buckets=[4])
        spg = state._graphed
        reqs = [Request(i, p, SamplingParams(max_tokens=8, temperature=0.0,
                                             ignore_eos=True))
                for i, p in enumerate(PROMPTS)]
        state.add(reqs)
        m.sync()

        B = state.size
        Bp = spg.bucket_for(B)
        tables_t, lens_t = state._tables_t, state._lens_t
        last = torch.tensor(state.last_tok, device=m.device).unsqueeze(1)
        pos = lens_t.unsqueeze(1)

        # fill the statics exactly as step() would
        w = tables_t.shape[1]
        for s in (spg.s0, spg.s1):
            s["tok"][:B].copy_(last)
            s["pos"][:B].copy_(pos)
            s["tables"][:B, :w].copy_(tables_t)
            s["lens"][:B].copy_(lens_t)
            if Bp > B:
                s["lens"][B:Bp].fill_(1)
        m.sync()

        # eager stage0 on the SAME statics (hook-free, like capture time).
        # KV writes go to the same slots the replay writes, which is fine:
        # both write identical values if the compute matches.
        saved = spg._strip_hooks()
        try:
            with torch.no_grad():
                he, cose, sine = spg._fwd0(Bp)
                he, cose, sine = he.clone(), cose.clone(), sine.clone()
            m.sync()
            spg.g0[Bp].replay()
            m.sync()
            hg, cosg, sing = spg.out0[Bp]
            print(f"stage0 hidden  max|eager-replay| = {mad(he, hg):.6f}")
            print(f"stage0 cos     max|eager-replay| = {mad(cose, cosg):.6f}")
            print(f"stage0 sin     max|eager-replay| = {mad(sine, sing):.6f}")

            # handoff + eager stage1 on the replayed stage0 output
            spg.hid1[:Bp].copy_(hg)
            spg.cos1[:Bp].copy_(cosg)
            spg.sin1[:Bp].copy_(sing)
            m.sync()
            with torch.no_grad():
                lg_e = spg._fwd1(Bp).clone()
            m.sync()
            spg.g1[Bp].replay()
            m.sync()
            lg_g = spg.out1[Bp]
            print(f"stage1 logits  max|eager-replay| = {mad(lg_e, lg_g):.6f}")
            print(f"  eager argmax  {lg_e[:B, -1].argmax(-1).tolist()}")
            print(f"  replay argmax {lg_g[:B, -1].argmax(-1).tolist()}")

            # second replay with identical statics must be bit-identical
            spg.g0[Bp].replay()
            m.sync()
            hg2 = spg.out0[Bp][0]
            print(f"stage0 replay-vs-replay          = {mad(hg, hg2):.6f}")

            # bake test: if replayed cos ignores the pos static, the capture
            # froze the position input instead of re-reading it
            spg.s0["pos"][:Bp].fill_(3)
            spg.g0[Bp].replay(); m.sync()
            c3 = spg.out0[Bp][1].clone()
            spg.s0["pos"][:Bp].fill_(9)
            spg.g0[Bp].replay(); m.sync()
            c9 = spg.out0[Bp][1]
            print(f"cos(pos=3) vs cos(pos=9) diff    = {mad(c3, c9):.6f} "
                  f"(0 means position is BAKED)")
            spg.s0["tok"][:Bp].fill_(11)
            spg.g0[Bp].replay(); m.sync()
            h11 = spg.out0[Bp][0].clone()
            spg.s0["tok"][:Bp].fill_(99)
            spg.g0[Bp].replay(); m.sync()
            h99 = spg.out0[Bp][0]
            print(f"h(tok=11) vs h(tok=99) diff      = {mad(h11, h99):.6f} "
                  f"(0 means token is BAKED)")
            spg.s0["lens"][:Bp].fill_(2)
            spg.g0[Bp].replay(); m.sync()
            l2 = spg.out0[Bp][0].clone()
            spg.s0["lens"][:Bp].fill_(5)
            spg.g0[Bp].replay(); m.sync()
            l5 = spg.out0[Bp][0]
            print(f"h(lens=2) vs h(lens=5) diff      = {mad(l2, l5):.6f} "
                  f"(0 means lens is BAKED)")

            # which side is the truth: run the EAGER stage0 output through
            # eager stage1 and compare with the known-good reference argmax
            spg.hid1[:Bp].copy_(he)
            spg.cos1[:Bp].copy_(cose)
            spg.sin1[:Bp].copy_(sine)
            m.sync()
            with torch.no_grad():
                lg_from_eager = spg._fwd1(Bp).clone()
            m.sync()
            print(f"  eager-chain argmax {lg_from_eager[:B, -1].argmax(-1).tolist()}")
        finally:
            spg._restore_hooks(saved)

        # reference: what the plain eager fused step says the logits are
        # (fresh identical state so pool contents are untouched by the above)
        state2 = PagedBatchState(m, num_blocks=2048, block_size=16, fused=True)
        reqs2 = [Request(10 + i, p, SamplingParams(max_tokens=8,
                                                   temperature=0.0,
                                                   ignore_eos=True))
                 for i, p in enumerate(PROMPTS)]
        state2.add(reqs2)
        m.sync()
        with torch.no_grad():
            ref = state2.forward_async()
        m.sync()
        print(f"reference eager-step argmax {ref[:B].argmax(-1).tolist()}")
    finally:
        restore_attention(m.model, prev)


if __name__ == "__main__":
    main()

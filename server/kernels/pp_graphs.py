"""Per-stage CUDA graphs for a pipeline-sharded model.

The sharded decode step measured 52.3 ms of python issue against 73.6 ms
end to end: walking the hooked layers through transformers costs more CPU
than the GPUs spend computing, which is why no interleaving schedule could
fill the pipeline bubble. This removes the python the same way the
single-GPU step did, except a CUDA graph cannot span devices, so the step
becomes two graphs and one handoff:

    replay stage0 (gpu0: embeddings, rotary, layers up to the cut)
    copy hidden states + rotary cos/sin across the cut     (3 small copies)
    replay stage1 (gpu1: remaining layers, norm, lm_head)

Capture-time python calls the decoder layers directly, bypassing the
transformers model forward entirely (no mask builder, no hook bookkeeping in
the hot path afterward). Each stage owns static buffers on its device and
its own NanoPagedCache view over the shared pool, so nothing inside either
capture touches the other GPU. Slot arithmetic is recorded per stage from
that device's own static tables/lens, exactly like the single-GPU capture.

Same external contract as GraphedDecode (bucket_for / cap / step), so the
fused state's graphed branch works unchanged.
"""
from __future__ import annotations

import inspect

import torch

from . import paged_attention_triton as pat
from .paged_runtime import NanoPagedCache


def _layer_call(layer, h, pos, cache, cpos, cos_sin):
    # transformers has renamed the cache kwarg across 5.x minors; resolve
    # against the layer's real signature so the cache is never silently
    # dropped (capture-time python only, costs nothing at serve time)
    params = inspect.signature(type(layer).forward).parameters
    kw = {"attention_mask": None, "position_ids": pos, "use_cache": True,
          "cache_position": cpos, "position_embeddings": cos_sin}
    kw = {k: v for k, v in kw.items() if k in params
          or any(p.kind == p.VAR_KEYWORD for p in params.values())}
    if "past_key_value" in params:
        kw["past_key_value"] = cache
    elif "past_key_values" in params:
        kw["past_key_values"] = cache
    else:
        assert any(p.kind == p.VAR_KEYWORD for p in params.values()), \
            "decoder layer accepts no cache kwarg"
        kw["past_key_values"] = cache
    out = layer(h, **kw)
    return out[0] if isinstance(out, tuple) else out


class StagePipelineGraphs:
    """One captured graph per (stage, batch bucket) for a two-stage split."""

    def __init__(self, runner, store, block_size: int, buckets: list[int],
                 table_cap: int, scratch_block: int):
        model = runner.model
        self.store = store
        self.bs = block_size
        self.buckets = sorted(buckets)
        self.cap = table_cap

        core = model.model
        self.embed = core.embed_tokens
        self.rotary = core.rotary_emb
        self.norm = core.norm
        self.head = model.lm_head
        layers = list(core.layers)
        devs = [runner.layer_device(i) for i in range(len(layers))]
        cut = next(i for i, d in enumerate(devs) if str(d) != str(devs[0]))
        self.d0, self.d1 = devs[0], devs[cut]
        self.layers0, self.layers1 = layers[:cut], layers[cut:]

        Bm = self.buckets[-1]
        H = model.config.hidden_size
        rd = getattr(model.config, "head_dim",
                     H // model.config.num_attention_heads)

        def statics(dev):
            return {
                "tok": torch.zeros(Bm, 1, dtype=torch.long, device=dev),
                "pos": torch.zeros(Bm, 1, dtype=torch.long, device=dev),
                "tables": torch.full((Bm, table_cap), scratch_block,
                                     dtype=torch.long, device=dev),
                "lens": torch.ones(Bm, dtype=torch.long, device=dev),
                "slots": torch.zeros(Bm, dtype=torch.long, device=dev),
                "cpos": torch.zeros(1, dtype=torch.long, device=dev),
            }

        self.s0 = statics(self.d0)
        self.s1 = statics(self.d1)
        dt = next(model.parameters()).dtype
        # the handoff buffers: hidden states and rotary tables on stage 1
        self.hid1 = torch.zeros(Bm, 1, H, dtype=dt, device=self.d1)
        self.cos1 = torch.zeros(Bm, 1, rd, dtype=dt, device=self.d1)
        self.sin1 = torch.zeros(Bm, 1, rd, dtype=dt, device=self.d1)

        self.g0: dict[int, torch.cuda.CUDAGraph] = {}
        self.g1: dict[int, torch.cuda.CUDAGraph] = {}
        self.out0: dict[int, tuple] = {}
        self.out1: dict[int, torch.Tensor] = {}
        self._pool0 = self._pool1 = None

        prev_force = pat.FORCE_NUM_SPLITS
        pat.FORCE_NUM_SPLITS = 1
        try:
            for Bp in self.buckets:
                self._capture(Bp)
        finally:
            pat.FORCE_NUM_SPLITS = prev_force

    # --- capture-time forwards (python here, never at serve time) --------
    def _slots_of(self, s, Bp):
        lens = s["lens"][:Bp]
        blk = s["tables"][:Bp].gather(1, (lens // self.bs).unsqueeze(1)).squeeze(1)
        s["slots"][:Bp].copy_(blk * self.bs + lens % self.bs)
        return lens

    def _cache_of(self, s, Bp, lens):
        return NanoPagedCache(self.store, s["tables"][:Bp], lens,
                              s["slots"][:Bp], self.bs,
                              max_len=self.cap * self.bs - 1)

    def _fwd0(self, Bp):
        s = self.s0
        lens = self._slots_of(s, Bp)
        cache = self._cache_of(s, Bp, lens)
        h = self.embed(s["tok"][:Bp])
        cos, sin = self.rotary(h, s["pos"][:Bp])
        for layer in self.layers0:
            h = _layer_call(layer, h, s["pos"][:Bp], cache, s["cpos"], (cos, sin))
        return h, cos.to(h.dtype), sin.to(h.dtype)

    def _fwd1(self, Bp):
        s = self.s1
        lens = self._slots_of(s, Bp)
        cache = self._cache_of(s, Bp, lens)
        h = self.hid1[:Bp]
        cs = (self.cos1[:Bp], self.sin1[:Bp])
        for layer in self.layers1:
            h = _layer_call(layer, h, s["pos"][:Bp], cache, s["cpos"], cs)
        return self.head(self.norm(h))

    @torch.no_grad()
    def _capture(self, Bp):
        for dev, fwd, graphs, outs, pool_attr in (
                (self.d0, self._fwd0, self.g0, self.out0, "_pool0"),
                (self.d1, self._fwd1, self.g1, self.out1, "_pool1")):
            with torch.cuda.device(dev):
                st = torch.cuda.Stream(dev)
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    for _ in range(2):
                        fwd(Bp)
                torch.cuda.current_stream().wait_stream(st)
                g = torch.cuda.CUDAGraph()
                pool = getattr(self, pool_attr)
                kw = {"pool": pool} if pool is not None else {}
                with torch.cuda.graph(g, **kw):
                    outs[Bp] = fwd(Bp)
                if pool is None:
                    setattr(self, pool_attr, g.pool())
                graphs[Bp] = g

    # --- the GraphedDecode contract --------------------------------------
    def bucket_for(self, B: int) -> int | None:
        for b in self.buckets:
            if b >= B:
                return b
        return None

    @torch.no_grad()
    def step(self, B: int, tok, pos, tables, lens) -> torch.Tensor:
        Bp = self.bucket_for(B)
        w = tables.shape[1]
        for s in (self.s0, self.s1):
            s["tok"][:B].copy_(tok)
            s["pos"][:B].copy_(pos)
            s["tables"][:B, :w].copy_(tables)
            s["lens"][:B].copy_(lens)
            if Bp > B:
                s["lens"][B:Bp].fill_(1)
        self.g0[Bp].replay()
        h, cos, sin = self.out0[Bp]
        self.hid1[:Bp].copy_(h)          # the cut, three copies, no python loop
        self.cos1[:Bp].copy_(cos)
        self.sin1[:Bp].copy_(sin)
        self.g1[Bp].replay()
        return self.out1[Bp][:B]

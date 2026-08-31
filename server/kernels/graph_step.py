"""CUDA-graph capture of the fused decode step.

At small batches the step is dominated not by GPU work but by launching it:
a 24-layer decode is ~200 kernel launches, each a few microseconds of CPU
that the GPU spends idle. A CUDA graph records the whole step once and
replays it as one launch.

What makes the fused path capturable:

- The paged kernel's loop bound is loaded from the lens tensor on device, so
  a replayed graph scans each row's true current length. Nothing about
  sequence growth is baked into the recording.
- All inputs live in static buffers (tokens, positions, tables, lens,
  slots); each step copies the batch's state in, replays, and reads logits
  from the static output the capture produced.
- Buckets by batch size only (never by length): one graph per bucket, rows
  padded up to the bucket with a scratch sequence whose writes land in a
  reserved scratch block and whose outputs are ignored.
- Split-K is pinned to 1 during capture: the split variant sizes its chunks
  from a host value at launch, which a recording would fossilize. The
  non-split kernel is exact at every length; splits only ever bought
  occupancy, and at graphed batch sizes the launch overhead they fought is
  gone anyway.

Capture happens at engine construction (after the attention switch and the
JIT warmup), never inside a request.
"""
from __future__ import annotations

import torch

from . import paged_attention_triton as pat
from .paged_runtime import NanoPagedCache


class GraphedDecode:
    """Owns the static buffers and one captured graph per batch bucket."""

    def __init__(self, model, store, block_size: int, buckets: list[int],
                 table_cap: int, scratch_block: int):
        self.m = model
        self.store = store
        self.bs = block_size
        self.buckets = sorted(buckets)
        self.cap = table_cap
        dev = next(model.parameters()).device
        Bm = self.buckets[-1]

        # static inputs, shared across buckets (bucket graphs use [:Bp] views)
        self.tok = torch.zeros(Bm, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(Bm, 1, dtype=torch.long, device=dev)
        self.tables = torch.full((Bm, table_cap), scratch_block,
                                 dtype=torch.long, device=dev)
        self.lens = torch.ones(Bm, dtype=torch.long, device=dev)
        self.slots = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.cpos = torch.zeros(1, dtype=torch.long, device=dev)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.out: dict[int, torch.Tensor] = {}
        self._pool = None

        prev_force = pat.FORCE_NUM_SPLITS
        pat.FORCE_NUM_SPLITS = 1
        prev_mask = self._register_no_mask()
        try:
            for Bp in self.buckets:
                self._capture(Bp)
        finally:
            pat.FORCE_NUM_SPLITS = prev_force
            self._register_mask(prev_mask)

    @staticmethod
    def _register_no_mask():
        """The transformers mask builder allocates a device scalar with
        torch.tensor(0.0, device=...), a host-to-device copy that CUDA
        forbids mid-capture. The fused decode ignores the mask anyway (the
        kernel bounds itself by each row's length), so during capture the
        mask interface for our attention name returns None. Restored after,
        because the prefill fallback path does use masks."""
        try:
            from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        except ImportError:
            return None
        prev = ALL_MASK_ATTENTION_FUNCTIONS[pat.ATTN_NAME]
        GraphedDecode._register_mask(lambda *a, **k: None)
        return prev

    @staticmethod
    def _register_mask(fn):
        if fn is None:
            return
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        try:
            ALL_MASK_ATTENTION_FUNCTIONS.register(pat.ATTN_NAME, fn)
        except Exception:
            try:
                ALL_MASK_ATTENTION_FUNCTIONS[pat.ATTN_NAME] = fn
            except Exception:
                ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[pat.ATTN_NAME] = fn

    def _forward(self, Bp: int):
        # slot arithmetic is recorded too, so replays recompute it from the
        # current tables/lens contents
        lens = self.lens[:Bp]
        blk = self.tables[:Bp].gather(1, (lens // self.bs).unsqueeze(1)).squeeze(1)
        self.slots[:Bp].copy_(blk * self.bs + lens % self.bs)
        cache = NanoPagedCache(self.store, self.tables[:Bp], lens,
                               self.slots[:Bp], self.bs,
                               max_len=self.cap * self.bs - 1)
        out = self.m(input_ids=self.tok[:Bp], position_ids=self.pos[:Bp],
                     past_key_values=cache, use_cache=True,
                     cache_position=self.cpos)
        return out.logits

    @torch.no_grad()
    def _capture(self, Bp: int):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._forward(Bp)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        kw = {"pool": self._pool} if self._pool is not None else {}
        with torch.cuda.graph(g, **kw):
            self.out[Bp] = self._forward(Bp)
        if self._pool is None:
            self._pool = g.pool()
        self.graphs[Bp] = g

    def bucket_for(self, B: int) -> int | None:
        for b in self.buckets:
            if b >= B:
                return b
        return None

    @torch.no_grad()
    def step(self, B: int, tok, pos, tables, lens) -> torch.Tensor:
        """Copy the live batch into the static buffers, replay, return
        logits[:B]. tables is [B, w] with w <= table_cap; pad rows were preset
        to the scratch sequence at init and self-heal after every step."""
        Bp = self.bucket_for(B)
        self.tok[:B].copy_(tok)
        self.pos[:B].copy_(pos)
        self.tables[:B, :tables.shape[1]].copy_(tables)
        self.lens[:B].copy_(lens)
        if Bp > B:   # reset pad rows (previous step may have grown them)
            self.lens[B:Bp].fill_(1)
        self.graphs[Bp].replay()
        return self.out[Bp][:B]

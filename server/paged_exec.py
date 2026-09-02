"""Paged execution: a batch state whose KV lives in a block pool, gathered into
a contiguous cache each decode step and scattered back after.

KV bytes are stored in fixed blocks drawn from a free list
(server/paged_cache.py); attention runs over KV reassembled from a per-sequence
block table.

Gather/scatter is vectorized: a sequence's block table + position maps to a
linear slot (block * block_size + offset), so gathering the whole batch is one
`index_select` per layer, not a Python loop over sequences and blocks. With the
naive per-block loop the serial Python copy dominates at GPU speeds and makes
paging look artificially slow; the slot-table gather keeps the paged-vs-
contiguous comparison about paging itself.

To keep decode OOM-free without a preemption/recompute loop, a sequence reserves
its whole potential span (prompt + max_tokens) at block granularity on
admission; admission control refuses a request the pool can't hold
(backpressure). Scheduling (continuous admit/evict) matches the contiguous
engine; only the memory substrate changes.

Verified token-for-token against naive decoding by the equivalence oracle.
"""
from __future__ import annotations

import time

import torch
from transformers import DynamicCache

from .batched import _sample_batch
from .model import ModelRunner
from .paged_cache import BlockAllocator
from .request import Request


class PagedKVStore:
    """Global block pool: per layer, key/value tensors of shape
    [num_blocks, block_size, n_kv_heads, head_dim]. A token at (block b,
    offset o) lives at linear slot b*block_size + o in the flattened pool."""

    def __init__(self, model: ModelRunner, num_blocks: int, block_size: int):
        cfg = model.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.n_slots = num_blocks * block_size
        dt = model.dtype
        shape = (num_blocks, block_size, self.n_kv, self.head_dim)
        # each layer's pool lives where that layer's weights live, so a
        # pipeline-sharded model writes and reads KV without cross-GPU hops
        self.layer_dev = [model.layer_device(li) for li in range(self.n_layers)]
        self.key = [torch.zeros(shape, device=self.layer_dev[li], dtype=dt)
                    for li in range(self.n_layers)]
        self.val = [torch.zeros(shape, device=self.layer_dev[li], dtype=dt)
                    for li in range(self.n_layers)]

    def _flat(self, t):  # [num_blocks, block_size, n_kv, head_dim] -> [n_slots, n_kv, head_dim]
        return t.view(self.n_slots, self.n_kv, self.head_dim)

    def _slots(self, table: list[int], start: int, length: int, dev) -> torch.Tensor:
        """Linear slot index for positions [start, start+length) of one seq."""
        pos = torch.arange(start, start + length, device=dev)
        tbl = torch.tensor(table, device=dev, dtype=torch.long)
        return tbl[pos // self.block_size] * self.block_size + (pos % self.block_size)

    def _per_dev(self, t: torch.Tensor) -> dict:
        """One copy of an index tensor per pool device (no-op single-GPU)."""
        out = {}
        for dev in self.layer_dev:
            k = str(dev)
            if k not in out:
                out[k] = t.to(dev)
        return out

    def write_range(self, table: list[int], start: int, k_layers, v_layers, length: int):
        """Write `length` contiguous tokens (k_layers[li]/v_layers[li] are
        [n_kv, length, head_dim]) starting at position `start`."""
        slots = self._per_dev(
            self._slots(table, start, length, self.layer_dev[0]))
        for li in range(self.n_layers):
            s = slots[str(self.layer_dev[li])]
            k = k_layers[li].permute(1, 0, 2).to(s.device).contiguous()
            v = v_layers[li].permute(1, 0, 2).to(s.device).contiguous()
            self._flat(self.key[li]).index_copy_(0, s, k)
            self._flat(self.val[li]).index_copy_(0, s, v)

    def write_tokens(self, slots: torch.Tensor, k_last, v_last):
        """Scatter one new token per row. k_last[li]/v_last[li] are
        [B, n_kv, head_dim]; slots is [B] (one linear slot per row)."""
        by = self._per_dev(slots)
        for li in range(self.n_layers):
            s = by[str(self.layer_dev[li])]
            self._flat(self.key[li]).index_copy_(0, s, k_last[li].to(s.device))
            self._flat(self.val[li]).index_copy_(0, s, v_last[li].to(s.device))

    def gather_batch(self, tables, lengths, T_max):
        """Left-padded KV for the whole batch. Returns (keys, values, mask)
        where keys[li]/values[li] are [B, n_kv, T_max, head_dim]. One
        index_select per layer, no Python loop over blocks."""
        dev = self.key[0].device
        B = len(tables)
        slot_idx = torch.zeros(B, T_max, dtype=torch.long, device=dev)
        mask = torch.zeros(B, T_max, dtype=torch.long, device=dev)
        for i, (table, L) in enumerate(zip(tables, lengths)):
            pad = T_max - L
            slot_idx[i, pad:] = self._slots(table, 0, L, dev)
            mask[i, pad:] = 1
        flat_by = self._per_dev(slot_idx.reshape(-1))
        keys, vals = [], []
        for li in range(self.n_layers):
            flat_idx = flat_by[str(self.layer_dev[li])]
            k = self._flat(self.key[li]).index_select(0, flat_idx)
            v = self._flat(self.val[li]).index_select(0, flat_idx)
            keys.append(k.view(B, T_max, self.n_kv, self.head_dim).permute(0, 2, 1, 3).contiguous())
            vals.append(v.view(B, T_max, self.n_kv, self.head_dim).permute(0, 2, 1, 3).contiguous())
        return keys, vals, mask


class CppAllocator:
    """The Python BlockAllocator's surface over the C++ one, so the paged
    state can run its block bookkeeping in nanoserve_core unchanged. On the
    fused path the allocator is touched only at admission and eviction (blocks
    are reserved up front, steps grow device-side counters), so these
    crossings are rare by design and the per-op boundary tax never applies."""

    class _Tables:
        def __init__(self, core):
            self._core = core

        def __getitem__(self, sid):
            return self._core.table(sid)

    def __init__(self, num_blocks: int, block_size: int):
        import nanoserve_core as nc
        self._core = nc.BlockAllocator(num_blocks, block_size)
        self.tables = self._Tables(self._core)

    @property
    def num_free(self) -> int:
        return self._core.num_free

    def can_admit(self, n_tokens: int) -> bool:
        return self._core.can_admit(n_tokens)

    def add_seq(self, sid: int, n_tokens: int):
        from .paged_cache import OutOfBlocks
        if not self._core.can_admit(n_tokens):
            raise OutOfBlocks()
        self._core.add_seq(sid, n_tokens)
        return self._core.table(sid)

    def free_seq(self, sid: int):
        self._core.free_seq(sid)


class PagedBatchState:
    """Same public interface as BatchState (add / step / evict / size /
    any_active / reqs), but KV lives in the paged store."""

    def __init__(self, model: ModelRunner, num_blocks: int = 4096, block_size: int = 16,
                 fused: bool = False, alloc_backend: str = "py",
                 graphs: bool = False, graph_buckets=(1, 2, 4, 8, 16),
                 graph_max_ctx: int = 4096):
        self.m = model
        self.store = PagedKVStore(model, num_blocks, block_size)
        if alloc_backend == "cpp":
            self.alloc = CppAllocator(num_blocks, block_size)
        else:
            self.alloc = BlockAllocator(num_blocks, block_size)
        # CUDA-graph decode: capture one graph per batch bucket at init, then
        # each step is a buffer copy plus one replay. Requires the fused path
        # and a CUDA device; silently stays un-graphed otherwise.
        self._graphed = None
        # blocks the state holds for itself (graph mode's scratch block);
        # accounting checks should expect num_free == num_blocks - this
        self.reserved_blocks = 0
        if graphs and fused and str(model.device).startswith("cuda"):
            self.alloc.add_seq(-7, 1)      # scratch block for pad rows
            self.reserved_blocks = 1
            scratch = self.alloc.tables[-7][0]
            for li in range(self.store.n_layers):
                self.store._flat(self.store.key[li])[
                    scratch * block_size:(scratch + 1) * block_size] = 0
                self.store._flat(self.store.val[li])[
                    scratch * block_size:(scratch + 1) * block_size] = 0
            cap = min(graph_max_ctx // block_size + 1, num_blocks)
            if getattr(model, "pipeline", False):
                # a recording cannot span devices: one graph per stage, the
                # activation handoff done eagerly between replays
                from .kernels.pp_graphs import StagePipelineGraphs
                self._graphed = StagePipelineGraphs(model, self.store,
                                                    block_size,
                                                    list(graph_buckets), cap,
                                                    scratch)
            else:
                from .kernels.graph_step import GraphedDecode
                self._graphed = GraphedDecode(model.model, self.store,
                                              block_size, list(graph_buckets),
                                              cap, scratch)
        self.block_size = block_size
        # fused: decode attention reads the pool directly (Triton kernel on
        # CUDA, gather fallback elsewhere) instead of materializing a
        # contiguous copy each step. Requires the model to be switched to the
        # nanoserve attention implementation (kernels.use_triton_attention).
        self.fused = fused
        self.reqs: list[Request] = []
        self.sids: list[int] = []
        self.true_len: list[int] = []
        self.last_tok: list[int] = []
        self.active: list[bool] = []
        self._next_sid = 0
        # fused-path device mirrors of tables/lengths; rebuilt on add/evict
        # (rare), incremented in place per step so the hot loop stays free of
        # per-row python
        self._tables_t: torch.Tensor | None = None
        self._lens_t: torch.Tensor | None = None

    def _rebuild_fused(self):
        dev = self.m.device
        B = self.size
        if B == 0:
            self._tables_t = self._lens_t = None
            return
        tables = [self.alloc.tables[self.sids[i]] for i in range(B)]
        maxb = max(len(t) for t in tables)
        tt = torch.zeros(B, maxb, dtype=torch.long)
        for i, t in enumerate(tables):
            tt[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        self._tables_t = tt.to(dev)
        self._lens_t = torch.tensor(self.true_len, dtype=torch.long, device=dev)

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def any_active(self) -> bool:
        return any(self.active)

    def can_admit(self, req: Request) -> bool:
        need = len(req.input_ids(self.m)) + req.sampling.max_tokens
        return self.alloc.can_admit(need)

    # --- admission -----------------------------------------------------
    @torch.no_grad()
    def add(self, reqs: list[Request]):
        dev = self.m.device
        enc = [r.input_ids(self.m) for r in reqs]
        for r, ids in zip(reqs, enc):
            r.prompt_len = len(ids)
        Lp = max(len(ids) for ids in enc)

        pad_id = self.m.tokenizer.pad_token_id or self.m.eos_id or 0
        input_ids, gmask, gpos = [], [], []
        for ids in enc:
            padn = Lp - len(ids)
            input_ids.append([pad_id] * padn + ids)
            gmask.append([0] * padn + [1] * len(ids))
            gpos.append([0] * padn + list(range(len(ids))))
        if self._graphed is not None and hasattr(self._graphed, "prefill"):
            # direct-stage prefill: hook-free layer calls, last-position
            # lm_head only. Same padded batch, same additive semantics the
            # stock path would build (causal, pad keys masked for everyone).
            km = torch.tensor(gmask, device=dev, dtype=torch.bool)
            q = torch.arange(Lp, device=dev)
            allow = (q[None, :] <= q[:, None])[None, :, :] & km[:, None, :]
            neg = torch.finfo(self.m.dtype).min
            mask_add = torch.where(
                allow, 0.0, neg).to(self.m.dtype)[:, None, :, :]
            last_logits, gcache = self._graphed.prefill(
                torch.tensor(input_ids, device=dev), mask_add,
                torch.tensor(gpos, device=dev))
            first = _sample_batch(last_logits, reqs)
        else:
            out = self.m.model(
                input_ids=torch.tensor(input_ids, device=dev),
                attention_mask=torch.tensor(gmask, device=dev),
                position_ids=torch.tensor(gpos, device=dev),
                past_key_values=DynamicCache(),
                use_cache=True,
            )
            gcache = out.past_key_values
            first = _sample_batch(out.logits[:, -1, :], reqs)

        self.m.sync()
        t = time.perf_counter()
        for j, (r, ids) in enumerate(zip(reqs, enc)):
            L = len(ids)
            padn = Lp - L
            sid = self._next_sid
            self._next_sid += 1
            self.alloc.add_seq(sid, r.prompt_len + r.sampling.max_tokens)
            table = self.alloc.tables[sid]
            k_layers = [gcache.layers[li].keys[j, :, padn:, :] for li in range(self.store.n_layers)]
            v_layers = [gcache.layers[li].values[j, :, padn:, :] for li in range(self.store.n_layers)]
            self.store.write_range(table, 0, k_layers, v_layers, L)

            tok = int(first[j])
            r.output_tokens.append(tok)
            r.first_token_time = t
            self.reqs.append(r)
            self.sids.append(sid)
            self.true_len.append(L)
            self.last_tok.append(tok)
            self.active.append(True)
        if self.fused:
            self._rebuild_fused()

    # --- one decode iteration ------------------------------------------
    @torch.no_grad()
    def step(self) -> list[int]:
        if self.fused:
            return self._step_fused()
        return self._step_gather()

    @torch.no_grad()
    def _step_fused(self) -> list[int]:
        return self._finish_step(self.forward_async())

    @torch.no_grad()
    def forward_async(self) -> torch.Tensor:
        """Issue the fused decode forward and return this step's last-token
        logits WITHOUT synchronizing. CUDA work is queued, not finished; the
        sync happens when the caller reads the logits (sampling). Splitting
        the step here is what lets a pipeline-sharded model interleave two
        half-batches: issue A, issue B (B's first stage overlaps A's second),
        then sample both."""
        from .kernels.paged_runtime import NanoPagedCache

        dev = self.m.device
        B = self.size
        bs = self.block_size
        T_max = max(self.true_len)
        if self._tables_t is None:
            self._rebuild_fused()
        tables_t, lens_t = self._tables_t, self._lens_t

        g = self._graphed
        if (g is not None and g.bucket_for(B) is not None
                and tables_t.shape[1] <= g.cap and T_max + 1 < g.cap * bs):
            last = torch.tensor(self.last_tok, device=dev).unsqueeze(1)
            logits = g.step(B, last, lens_t.unsqueeze(1), tables_t, lens_t)
            lens_t += 1
            for i in range(B):
                self.true_len[i] += 1
            return logits[:, -1, :]

        # each row's write slot for this step's token, no per-row python
        blk = tables_t.gather(1, (lens_t // bs).unsqueeze(1)).squeeze(1)
        slots = blk * bs + lens_t % bs

        cache = NanoPagedCache(self.store, tables_t, lens_t, slots, bs, max_len=T_max)
        last = torch.tensor(self.last_tok, device=dev).unsqueeze(1)
        pos = lens_t.unsqueeze(1)
        out = self.m.model(
            input_ids=last,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([T_max], device=dev),
        )
        lens_t += 1
        for i in range(B):
            self.true_len[i] += 1
        return out.logits[:, -1, :]

    @torch.no_grad()
    def finish_async(self, logits: torch.Tensor) -> list[int]:
        """Sample and retire for a forward_async() issued earlier."""
        return self._finish_step(logits)

    @torch.no_grad()
    def _step_gather(self) -> list[int]:
        dev = self.m.device
        B = self.size
        T_max = max(self.true_len)
        bs = self.block_size

        tables = [self.alloc.tables[self.sids[i]] for i in range(B)]
        keys, vals, mask = self.store.gather_batch(tables, self.true_len, T_max)
        cache = DynamicCache()
        for li in range(self.store.n_layers):
            cache.update(keys[li], vals[li], li)

        last = torch.tensor(self.last_tok, device=dev).unsqueeze(1)
        pos = torch.tensor(self.true_len, device=dev).unsqueeze(1)
        full_mask = torch.cat([mask, torch.ones(B, 1, device=dev, dtype=torch.long)], dim=1)
        out = self.m.model(
            input_ids=last,
            attention_mask=full_mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([T_max], device=dev),
        )
        # scatter the just-appended token's KV back to each row's next slot
        new_cache = out.past_key_values
        slots = torch.tensor(
            [tables[i][self.true_len[i] // bs] * bs + self.true_len[i] % bs for i in range(B)],
            device=dev, dtype=torch.long,
        )
        k_last = [new_cache.layers[li].keys[:, :, -1, :] for li in range(self.store.n_layers)]
        v_last = [new_cache.layers[li].values[:, :, -1, :] for li in range(self.store.n_layers)]
        self.store.write_tokens(slots, k_last, v_last)
        for i in range(B):
            self.true_len[i] += 1

        return self._finish_step(out.logits[:, -1, :])

    def _finish_step(self, logits) -> list[int]:
        """Sample the next token per row and retire whatever finished."""
        nxt = _sample_batch(logits, self.reqs)
        self.m.sync()
        t = time.perf_counter()

        finished = []
        for i, (r, tok) in enumerate(zip(self.reqs, nxt.tolist())):
            if not self.active[i]:
                continue
            self.last_tok[i] = tok
            r.output_tokens.append(tok)
            done = r.num_output >= r.sampling.max_tokens or (
                not r.sampling.ignore_eos and tok == self.m.eos_id
            )
            if done:
                r.finish_time = t
                self.active[i] = False
                finished.append(i)
        return finished

    # --- eviction ------------------------------------------------------
    def evict(self, rows: list[int]):
        drop = set(rows)
        for i in rows:
            self.alloc.free_seq(self.sids[i])
        keep = [i for i in range(self.size) if i not in drop]
        self.reqs = [self.reqs[i] for i in keep]
        self.sids = [self.sids[i] for i in keep]
        self.true_len = [self.true_len[i] for i in keep]
        self.last_tok = [self.last_tok[i] for i in keep]
        self.active = [self.active[i] for i in keep]
        if self.fused:
            self._rebuild_fused()

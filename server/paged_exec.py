"""Paged execution: a batch state whose KV actually lives in a block pool,
gathered into a contiguous cache each decode step and scattered back after.

This is the real thing, not accounting: KV bytes are stored in fixed blocks
drawn from a free list (server/paged_cache.py), and attention runs over KV
reassembled from a per-sequence block table. The per-step gather is the extra
HBM round-trip that real PagedAttention pays (~5-10%); here it buys demand-
paged memory and admission control instead of reserving contiguous space.

To keep decode OOM-free without a preemption/recompute loop, a sequence
reserves its whole potential span (prompt + max_tokens) in *block* granularity
at admission — far tighter than reserving the model's max context, and the
engine's admission control refuses a request the pool can't currently hold
(backpressure). The scheduling story (continuous admit/evict) is identical to
the contiguous engine; only the memory substrate changes.

Verified token-for-token against naive decoding by the equivalence oracle.
"""
from __future__ import annotations

import time

import torch
from transformers import DynamicCache

from .batched import _lpad_T, _lpad_1, _sample_batch
from .model import ModelRunner
from .paged_cache import BlockAllocator, OutOfBlocks
from .request import Request


class PagedKVStore:
    """Global block pool: per layer, key/value tensors of shape
    [num_blocks, n_kv_heads, block_size, head_dim]."""

    def __init__(self, model: ModelRunner, num_blocks: int, block_size: int):
        cfg = model.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.block_size = block_size
        self.num_blocks = num_blocks
        dev, dt = model.device, model.dtype
        shape = (num_blocks, self.n_kv, block_size, self.head_dim)
        self.key = [torch.zeros(shape, device=dev, dtype=dt) for _ in range(self.n_layers)]
        self.val = [torch.zeros(shape, device=dev, dtype=dt) for _ in range(self.n_layers)]

    def write_range(self, table: list[int], start: int, k_layers, v_layers, length: int):
        """Write `length` contiguous tokens starting at position `start` for one
        sequence. k_layers[li]/v_layers[li] are [n_kv, length, head_dim]."""
        bs = self.block_size
        pos = 0
        while pos < length:
            tok = start + pos
            blk = table[tok // bs]
            off = tok % bs
            cnt = min(bs - off, length - pos)
            for li in range(self.n_layers):
                self.key[li][blk, :, off:off + cnt, :] = k_layers[li][:, pos:pos + cnt, :]
                self.val[li][blk, :, off:off + cnt, :] = v_layers[li][:, pos:pos + cnt, :]
            pos += cnt

    def gather(self, table: list[int], length: int):
        """Read tokens [0, length) for one sequence -> (k_layers, v_layers),
        each [n_kv, length, head_dim]."""
        bs = self.block_size
        kout = [torch.empty(self.n_kv, length, self.head_dim, device=self.key[0].device,
                            dtype=self.key[0].dtype) for _ in range(self.n_layers)]
        vout = [torch.empty_like(kout[li]) for li in range(self.n_layers)]
        pos = 0
        while pos < length:
            blk = table[pos // bs]
            off = pos % bs
            cnt = min(bs - off, length - pos)
            for li in range(self.n_layers):
                kout[li][:, pos:pos + cnt, :] = self.key[li][blk, :, off:off + cnt, :]
                vout[li][:, pos:pos + cnt, :] = self.val[li][blk, :, off:off + cnt, :]
            pos += cnt
        return kout, vout


class PagedBatchState:
    """Same public interface as BatchState (add / step / evict / size /
    any_active / reqs), but KV lives in the paged store."""

    def __init__(self, model: ModelRunner, num_blocks: int = 4096, block_size: int = 16):
        self.m = model
        self.store = PagedKVStore(model, num_blocks, block_size)
        self.alloc = BlockAllocator(num_blocks, block_size)
        self.block_size = block_size
        self.reqs: list[Request] = []
        self.sids: list[int] = []          # allocator seq-id per row
        self.true_len: list[int] = []      # tokens written per row
        self.last_tok: list[int] = []      # next token to feed per row
        self.active: list[bool] = []
        self._next_sid = 0

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
            # reserve the whole potential span in blocks; write the prompt now
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
            self.true_len.append(L)          # prompt written; the sampled token
            self.last_tok.append(tok)        # will be written on its decode step
            self.active.append(True)

    # --- one decode iteration ------------------------------------------
    @torch.no_grad()
    def step(self) -> list[int]:
        dev = self.m.device
        B = self.size
        T_max = max(self.true_len)

        # gather every row's KV from blocks into a left-padded contiguous cache
        keys_by_layer = [[] for _ in range(self.store.n_layers)]
        vals_by_layer = [[] for _ in range(self.store.n_layers)]
        mask = torch.zeros(B, T_max, device=dev, dtype=torch.long)
        for i in range(B):
            L = self.true_len[i]
            k_layers, v_layers = self.store.gather(self.alloc.tables[self.sids[i]], L)
            pad = T_max - L
            mask[i, pad:] = 1
            for li in range(self.store.n_layers):
                keys_by_layer[li].append(_lpad_T(k_layers[li].unsqueeze(0), pad))
                vals_by_layer[li].append(_lpad_T(v_layers[li].unsqueeze(0), pad))
        # populate a fresh cache via the stable update() API (no internal classes)
        cache = DynamicCache()
        for li in range(self.store.n_layers):
            cache.update(torch.cat(keys_by_layer[li], dim=0),
                         torch.cat(vals_by_layer[li], dim=0), li)

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
        # extract the just-appended token's KV (last column) and scatter to blocks
        new_cache = out.past_key_values
        for i in range(B):
            L = self.true_len[i]
            table = self.alloc.tables[self.sids[i]]
            k_layers = [new_cache.layers[li].keys[i, :, -1:, :] for li in range(self.store.n_layers)]
            v_layers = [new_cache.layers[li].values[i, :, -1:, :] for li in range(self.store.n_layers)]
            self.store.write_range(table, L, k_layers, v_layers, 1)
            self.true_len[i] = L + 1

        nxt = _sample_batch(out.logits[:, -1, :], self.reqs)
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

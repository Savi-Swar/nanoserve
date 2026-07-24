"""Batched execution over a set of active sequences: the layer static and
continuous batching share.

Sequences in a batch have different lengths. The KV cache is left-padded so
every active sequence aligns at the right edge (newest token always at column
T-1), making per-step decode a single [B, 1] forward. RoPE stays correct
because we pass true per-row position_ids; padded columns are masked out of
attention, so the pad zeros never touch the softmax.

The gap between the shortest and longest sequence is exactly the padding
fragmentation the paged KV cache (next tier) removes.

Targets transformers 5.x, where a DynamicCache holds tensors in
`cache.layers[i].keys / .values` ([B, n_kv_heads, T, head_dim]).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .model import ModelRunner
from .request import Request


class BatchState:
    """The running batch: row i of every tensor belongs to reqs[i]."""

    def __init__(self, model: ModelRunner):
        self.m = model
        self.reqs: list[Request] = []
        self.cache = None                          # a persistent DynamicCache
        self.mask: torch.Tensor | None = None      # [B, T] 1=real 0=pad
        self.true_len: torch.Tensor | None = None  # [B] real tokens so far
        self.last_tok: torch.Tensor | None = None  # [B, 1] token to feed next
        self.active: list[bool] = []               # False once a row is done
                                                    # (static keeps it; continuous evicts)

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def T(self) -> int:
        if self.cache is None or not self.cache.layers or self.cache.layers[0].keys is None:
            return 0
        return self.cache.layers[0].keys.shape[2]

    @property
    def n_layers(self) -> int:
        return len(self.cache.layers)

    @property
    def any_active(self) -> bool:
        return any(self.active)

    # --- admission -----------------------------------------------------
    @torch.no_grad()
    def add(self, reqs: list[Request]):
        """Prefill `reqs` as one padded group and merge into the batch."""
        dev = self.m.device
        enc = [r.input_ids(self.m) for r in reqs]
        for r, ids in zip(reqs, enc):
            r.prompt_len = len(ids)
        Lp = max(len(ids) for ids in enc)

        pad_id = self.m.tokenizer.pad_token_id or self.m.eos_id or 0
        input_ids, gmask, gpos = [], [], []
        for ids in enc:
            padn = Lp - len(ids)
            input_ids.append([pad_id] * padn + ids)          # left-pad prompts
            gmask.append([0] * padn + [1] * len(ids))
            gpos.append([0] * padn + list(range(len(ids))))  # true RoPE positions
        input_ids = torch.tensor(input_ids, device=dev)
        gmask_t = torch.tensor(gmask, device=dev)
        gpos_t = torch.tensor(gpos, device=dev)

        out = self.m.model(
            input_ids=input_ids,
            attention_mask=gmask_t,
            position_ids=gpos_t,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        gcache = out.past_key_values
        first = _sample_batch(out.logits[:, -1, :], reqs)
        true_len = torch.tensor([len(ids) for ids in enc], device=dev)

        self.m.sync()
        t = time.perf_counter()
        for r, tok in zip(reqs, first.tolist()):
            r.output_tokens.append(tok)
            r.first_token_time = t

        if self.size == 0:
            self.reqs = list(reqs)
            self.cache = gcache
            self.mask = gmask_t
            self.true_len = true_len
            self.last_tok = first.unsqueeze(1)
            self.active = [True] * len(reqs)
            return

        self._merge(reqs, gcache, gmask_t, true_len, first)
        self.active += [True] * len(reqs)

    def _merge(self, reqs, gcache, gmask, true_len, first):
        """Left-pad whichever group is shorter in T so both align at the right
        edge, then concatenate on the batch dimension."""
        Ta, Tb = self.T, gcache.layers[0].keys.shape[2]
        T = max(Ta, Tb)
        for li in range(self.n_layers):
            a, g = self.cache.layers[li], gcache.layers[li]
            a.keys = torch.cat([_lpad_T(a.keys, T - Ta), _lpad_T(g.keys, T - Tb)], dim=0)
            a.values = torch.cat([_lpad_T(a.values, T - Ta), _lpad_T(g.values, T - Tb)], dim=0)
        self.mask = torch.cat(
            [_lpad_1(self.mask, T - Ta), _lpad_1(gmask, T - Tb)], dim=0
        )
        self.true_len = torch.cat([self.true_len, true_len], dim=0)
        self.last_tok = torch.cat([self.last_tok, first.unsqueeze(1)], dim=0)
        self.reqs = self.reqs + list(reqs)

    # --- one decode iteration over the whole batch ---------------------
    @torch.no_grad()
    def step(self) -> list[int]:
        """Advance every active row by one token. Returns row indices that
        just finished (for the engine to evict)."""
        dev = self.m.device
        T = self.T
        pos = self.true_len.unsqueeze(1)  # [B,1] true next position per row
        mask = torch.cat(
            [self.mask, torch.ones(self.size, 1, device=dev, dtype=self.mask.dtype)], dim=1
        )
        out = self.m.model(
            input_ids=self.last_tok,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=torch.tensor([T], device=dev),
        )
        self.cache = out.past_key_values
        self.mask = mask
        self.true_len = self.true_len + 1

        nxt = _sample_batch(out.logits[:, -1, :], self.reqs)
        self.last_tok = nxt.unsqueeze(1)
        self.m.sync()
        t = time.perf_counter()

        finished = []
        for i, (r, tok) in enumerate(zip(self.reqs, nxt.tolist())):
            if not self.active[i]:
                continue  # row already done, occupies a slot but isn't extended
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
        """Drop finished rows and reclaim any now-all-pad leading columns."""
        drop = set(rows)
        keep = [i for i in range(self.size) if i not in drop]
        if not keep:
            self.reqs, self.cache, self.active = [], None, []
            self.mask = self.true_len = self.last_tok = None
            return
        idx = torch.tensor(keep, device=self.m.device)
        for l in self.cache.layers:
            l.keys = l.keys.index_select(0, idx)
            l.values = l.values.index_select(0, idx)
        self.mask = self.mask.index_select(0, idx)
        self.true_len = self.true_len.index_select(0, idx)
        self.last_tok = self.last_tok.index_select(0, idx)
        self.reqs = [self.reqs[i] for i in keep]
        self.active = [self.active[i] for i in keep]
        self._trim()

    def _trim(self):
        """Remove leading columns that are padding for every remaining row.
        The contiguous cache's version of freeing memory."""
        if self.mask is None or self.T == 0:
            return
        col_used = self.mask.sum(dim=0)  # [T]
        lead = int((col_used == 0).to(torch.int).cumprod(0).sum())
        if lead == 0:
            return
        self.mask = self.mask[:, lead:]
        for l in self.cache.layers:
            l.keys = l.keys[:, :, lead:, :]
            l.values = l.values[:, :, lead:, :]


def _lpad_T(t: torch.Tensor, n: int) -> torch.Tensor:
    return t if n <= 0 else F.pad(t, (0, 0, n, 0))  # pad dim -2 (T) on the left


def _lpad_1(t: torch.Tensor, n: int) -> torch.Tensor:
    return t if n <= 0 else F.pad(t, (n, 0))         # pad dim -1 on the left


def _sample_batch(logits: torch.Tensor, reqs: list[Request]) -> torch.Tensor:
    """Vectorized greedy; per-row temperature falls back to a small loop only
    when a request actually samples (benchmarks default to greedy)."""
    if all(r.sampling.greedy for r in reqs):
        return logits.argmax(dim=-1)
    from .model import sample as _s
    return torch.tensor(
        [_s(logits[i : i + 1], r.sampling) for i, r in enumerate(reqs)],
        device=logits.device,
    )

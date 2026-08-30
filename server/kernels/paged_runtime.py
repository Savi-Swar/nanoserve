"""No-gather decode plumbing: run the HF model with KV read straight from the
paged pool.

The trick: transformers plumbs whatever `Cache.update()` returns into the
attention implementation untouched. NanoPagedCache.update() writes the new
token's K/V into the pool at each row's next slot and returns a PagedKV
handle instead of tensors; the registered attention function recognizes the
handle and runs the paged kernel over the pool (CUDA) or a vectorized gather +
SDPA (CPU fallback, used by the token-exactness tests). Deleting the per-step
gather is the point: the old path materialized a contiguous copy of every
sequence's whole KV every decode step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass
class PagedKV:
    """Duck-typed stand-in for the K/V tensors transformers hands to the
    attention fn. Carries everything the paged kernel needs; proxies the few
    tensor attributes generic code might poke."""
    k_flat: torch.Tensor        # [n_slots, H_kv, D]
    v_flat: torch.Tensor
    tables: torch.Tensor        # [B, max_blocks] int
    lens: torch.Tensor          # [B] int, INCLUDING the token written this step
    block_size: int

    @property
    def shape(self):
        B = self.tables.shape[0]
        return (B, self.k_flat.shape[1], int(self.lens.max()), self.k_flat.shape[2])

    @property
    def dtype(self):
        return self.k_flat.dtype

    @property
    def device(self):
        return self.k_flat.device

    @property
    def is_cuda(self):
        return self.k_flat.is_cuda


def gather_sdpa_fallback(query, handle: PagedKV, scale=None):
    """Reference path for the handle when triton is unavailable (CPU tests):
    vectorized slot gather into a left-padded batch, then SDPA. Mirrors the
    old gather_batch, but per-layer from the flat pool."""
    B, H, _, D = query.shape
    Hkv = handle.k_flat.shape[1]
    dev = query.device
    lens = handle.lens.tolist()
    T = max(lens)
    bs = handle.block_size
    slot_idx = torch.zeros(B, T, dtype=torch.long, device=dev)
    mask = torch.full((B, T), float("-inf"), device=dev)
    for i, L in enumerate(lens):
        pos = torch.arange(L, device=dev)
        tbl = handle.tables[i].long()
        pad = T - L
        slot_idx[i, pad:] = tbl[pos // bs] * bs + pos % bs
        mask[i, pad:] = 0.0
    flat = slot_idx.reshape(-1)
    k = handle.k_flat.index_select(0, flat).view(B, T, Hkv, D).permute(0, 2, 1, 3)
    v = handle.v_flat.index_select(0, flat).view(B, T, Hkv, D).permute(0, 2, 1, 3)
    group = H // Hkv
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    am = mask[:, None, None, :].to(query.dtype)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, k, v, attn_mask=am, scale=scale)
    return out  # [B, H, 1, D]


class NanoPagedCache(DynamicCache):
    """One decode step's cache view over the paged pool. Constructed fresh per
    step by PagedBatchState._step_fused with that step's tables/lengths/write
    slots; update() scatters the new token's K/V into the pool and hands the
    attention fn a PagedKV handle."""

    def __init__(self, store, tables: torch.Tensor, lens: torch.Tensor,
                 slots: torch.Tensor, block_size: int, max_len: int | None = None):
        super().__init__()
        self._store = store
        self._tables = tables
        self._lens = lens              # true lengths BEFORE this step's token
        self._slots = slots            # where each row's new token lands
        self._bs = block_size
        # caller usually knows max(lens) already; computing it here would sync
        self._max_len = max_len if max_len is not None else (
            int(lens.max()) if lens.numel() else 0)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states/value_states: [B, H_kv, 1, D], rope already applied
        kf = self._store._flat(self._store.key[layer_idx])
        vf = self._store._flat(self._store.val[layer_idx])
        kf.index_copy_(0, self._slots, key_states[:, :, 0, :])
        vf.index_copy_(0, self._slots, value_states[:, :, 0, :])
        handle = PagedKV(kf, vf, self._tables, self._lens + 1, self._bs)
        return handle, handle

    # transformers queries these while building masks / positions; keep them
    # consistent with a cache of length max_len that q_len tokens will extend.
    def get_seq_length(self, layer_idx: int = 0):
        return self._max_len

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        # transformers passes a cache_position tensor in some versions and a
        # plain int (query length) in others
        q_len = cache_position.shape[0] if hasattr(cache_position, "shape") else int(cache_position)
        return self._max_len + q_len, 0

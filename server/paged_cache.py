"""Paged KV cache — the memory manager, modeled on OS virtual-memory paging
(vLLM/PagedAttention, SOSP 2023).

The KV cache is the capacity bottleneck of an inference server: how many
sequences you can hold at once caps batch size, which caps throughput. The
naive layouts waste most of it:

  reserve-to-max   give every sequence room for the longest it *could* get
                   -> 60-80% wasted on sequences that finish early
  padded batch     pad every sequence to the batch's current longest
                   -> wastes the gap between shortest and longest each step

Paging fixes both: KV lives in fixed-size blocks drawn from a shared pool via
a free list. A sequence holds a block table (its list of block ids) and grows
one block at a time on demand. The only waste is *internal* fragmentation in
each sequence's last, partially-filled block — bounded by block_size-1 tokens,
independent of how long the sequence could have grown.

This module is the allocator (the systems artifact) plus the byte accounting
used by the fragmentation ablation. The block *tensors* live in PagedKVStore.
"""
from __future__ import annotations

import math


class OutOfBlocks(Exception):
    """Raised when the pool is exhausted — the server's OOM signal, which the
    scheduler answers with admission control / preemption."""


class BlockAllocator:
    """Fixed-size block pool with a free list and per-sequence block tables.

    All units are *blocks* (block_size tokens each). O(1) alloc/free."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free: list[int] = list(range(num_blocks - 1, -1, -1))  # used as a stack
        self.tables: dict[int, list[int]] = {}   # seq_id -> [block_id, ...]
        self.length: dict[int, int] = {}         # seq_id -> tokens stored
        self.fill: dict[int, int] = {}           # seq_id -> tokens in the last block

    # --- capacity ------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self.free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free)

    def blocks_for(self, n_tokens: int) -> int:
        return max(1, math.ceil(n_tokens / self.block_size))

    def can_admit(self, n_tokens: int) -> bool:
        return self.blocks_for(n_tokens) <= self.num_free

    def _grab(self) -> int:
        if not self.free:
            raise OutOfBlocks()
        return self.free.pop()

    # --- lifecycle -----------------------------------------------------
    def add_seq(self, seq_id: int, n_tokens: int) -> list[int]:
        """Allocate enough blocks to hold a fresh sequence's prompt."""
        nb = self.blocks_for(n_tokens)
        if nb > self.num_free:
            raise OutOfBlocks()
        table = [self._grab() for _ in range(nb)]
        self.tables[seq_id] = table
        self.length[seq_id] = n_tokens
        self.fill[seq_id] = n_tokens - (nb - 1) * self.block_size
        return table

    def append_token(self, seq_id: int) -> int | None:
        """Grow a sequence by one token. Returns a new block id if one was
        allocated this step, else None. Raises OutOfBlocks under pressure."""
        new_block = None
        if self.fill[seq_id] == self.block_size:  # last block full -> need another
            b = self._grab()
            self.tables[seq_id].append(b)
            self.fill[seq_id] = 0
            new_block = b
        self.fill[seq_id] += 1
        self.length[seq_id] += 1
        return new_block

    def free_seq(self, seq_id: int):
        for b in self.tables.pop(seq_id):
            self.free.append(b)
        self.length.pop(seq_id, None)
        self.fill.pop(seq_id, None)

    def slot(self, seq_id: int, pos: int) -> tuple[int, int]:
        """Map a token position to (block_id, offset_in_block)."""
        return self.tables[seq_id][pos // self.block_size], pos % self.block_size

    # --- metrics -------------------------------------------------------
    def internal_frag_tokens(self) -> int:
        """Total wasted token-slots = capacity of allocated blocks minus tokens
        actually stored. This is the *only* waste paging incurs."""
        return sum(
            len(self.tables[s]) * self.block_size - self.length[s] for s in self.tables
        )

    def utilization(self) -> float:
        cap = self.num_used * self.block_size
        stored = sum(self.length.values())
        return stored / cap if cap else 1.0


def kv_bytes_per_token(model) -> int:
    """KV cache bytes for one token: 2 (K and V) * layers * kv_heads * head_dim
    * dtype_bytes. For Qwen2.5-0.5B fp16 this is 12,288 B = 12 KiB/token."""
    cfg = model.model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    dtype_bytes = model.dtype.itemsize
    return 2 * n_layers * n_kv * head_dim * dtype_bytes

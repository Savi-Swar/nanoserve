"""Prefix caching (RadixAttention's core idea, simplified).

Requests that share a leading prefix (a system prompt, few-shot examples, a
shared document) recompute that prefix's KV every time under naive serving.
Since a token's KV depends only on the tokens before it (and absolute RoPE
positions), a shared prefix's KV is identical across requests and can be
computed once and reused. This engine caches prefix KV and prefills only each
request's novel suffix.

Deterministic metric: prefill tokens actually computed vs total prompt tokens.
That ratio doesn't depend on CPU timing, so it's a clean audit number.
Exactness: reused prefix KV must yield output token-identical to a full prefill
(checked against naive).
"""
from __future__ import annotations

from transformers import DynamicCache


class PrefixCache:
    def __init__(self, model):
        self.m = model
        self.entries: list[tuple[tuple, DynamicCache, int]] = []  # (ids, kv, len)
        self.tokens_total = 0      # prompt tokens if we prefilled everything
        self.tokens_computed = 0   # prompt tokens we actually ran through the model
        self.hits = 0

    def _clone(self, cache, L: int) -> DynamicCache:
        c = DynamicCache()
        for i, layer in enumerate(cache.layers):
            c.update(layer.keys[:, :, :L, :].clone(), layer.values[:, :, :L, :].clone(), i)
        return c

    def _longest_prefix(self, ids: list[int]):
        """Longest KV run reusable from any cached entry = the longest common
        token prefix (radix behavior). Capped at len-1 so at least one token is
        always computed (we need fresh logits for the next token)."""
        cap = len(ids) - 1
        best_len, best = 0, None
        for eids, kv, L in self.entries:
            c = 0
            lim = min(len(eids), cap)
            while c < lim and ids[c] == eids[c]:
                c += 1
            if c > best_len:
                best_len, best = c, kv
        return best_len, best

    def prefill(self, ids: list[int]):
        """Prefill with prefix reuse. Returns (last_logits, kv, cur)."""
        self.tokens_total += len(ids)
        L, kv = self._longest_prefix(ids)
        if L == 0:
            logits, cache, cur = self.m.prefill(ids)
            self.tokens_computed += len(ids)
        else:
            self.hits += 1
            cache = self._clone(kv, L)
            suffix = ids[L:]
            all_logits, cache, cur = self.m.decode_many(suffix, cache, L)
            logits = all_logits[-1:]  # logits after the final suffix token
            self.tokens_computed += len(suffix)
        self.entries.append((tuple(ids), self._clone(cache, len(ids)), len(ids)))
        return logits, cache, cur

    def stats(self) -> dict:
        saved = 1 - self.tokens_computed / self.tokens_total if self.tokens_total else 0.0
        return {"prefill_tokens_total": self.tokens_total,
                "prefill_tokens_computed": self.tokens_computed,
                "prefill_saved": saved, "hits": self.hits}

"""KV cache quantization — store the cache in low-bit ints to fit more of it.

Per-token symmetric quantization: one scale per (position), shared across heads
and head_dim, mapping to signed `bits`-bit ints and back. This simulates
storing the KV cache at `bits` bits instead of fp32 — the memory the cache
occupies drops by 32/bits, and the question the audit asks is what that costs in
output quality (how many generated tokens drift from the fp32 result).
"""
from __future__ import annotations

import torch


def _quantize(x: torch.Tensor, qmax: int) -> torch.Tensor:
    # per-(token, head) scale: amax over head_dim only -> [., kv_heads, T, 1].
    # Sharing one scale across heads (a coarser scheme) loses too much precision
    # because heads differ in magnitude; per-head is the honest baseline.
    scale = x.abs().amax(dim=3, keepdim=True).clamp(min=1e-8)
    q = torch.round(x / scale * qmax).clamp(-qmax, qmax)
    return q / qmax * scale  # dequantized (what attention will read)


def quantize_cache(cache, bits: int):
    """Replace the cache's KV with its `bits`-bit quantized-then-dequantized
    version, in place — simulating low-bit KV storage."""
    if not bits:
        return
    qmax = 2 ** (bits - 1) - 1
    for layer in cache.layers:
        layer.keys = _quantize(layer.keys, qmax)
        layer.values = _quantize(layer.values, qmax)

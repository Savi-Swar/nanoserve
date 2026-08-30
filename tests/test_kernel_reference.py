"""The tiled online-softmax algorithm vs direct attention math, on CPU.

reference_decode_attention() is a line-for-line torch mirror of the Triton
kernel's algorithm. Testing it against plain softmax attention validates the
tiling/rescaling math without a GPU; tests/test_kernel_equivalence.py then
holds the Triton code itself to the same answers on CUDA.
"""
import math

import pytest
import torch

from server.kernels.paged_attention_triton import reference_decode_attention


def direct_attention(q, k, v, mask_add=None, scale=None):
    """Unbatched-softmax ground truth: repeat_kv + explicit softmax(QK^T)V."""
    B, H, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    group = H // Hkv
    if scale is None:
        scale = D ** -0.5
    k = k.repeat_interleave(group, dim=1).float()   # [B, H, T, D]
    v = v.repeat_interleave(group, dim=1).float()
    s = torch.einsum("bhd,bhtd->bht", q.float(), k) * scale
    if mask_add is not None:
        s = s + mask_add[:, None, :].float()
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bht,bhtd->bhd", p, v).to(q.dtype)


CASES = [
    # (B, H, Hkv, T, D)  -- includes GQA, MHA, tiny T, T not a tile multiple
    (1, 2, 2, 8, 16),
    (2, 14, 2, 200, 64),     # Qwen2.5-0.5B decode shape (GQA 7:1)
    (3, 4, 1, 300, 32),
    (1, 8, 8, 129, 64),      # one past a tile boundary
    (2, 6, 2, 1, 64),        # single-token context
]


@pytest.mark.parametrize("B,H,Hkv,T,D", CASES)
def test_reference_matches_direct(B, H, Hkv, T, D):
    torch.manual_seed(B * 1000 + T)
    q = torch.randn(B, H, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    ref = reference_decode_attention(q, k, v, block_l=64)
    want = direct_attention(q, k, v)
    torch.testing.assert_close(ref, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("B,H,Hkv,T,D", CASES[:3])
def test_reference_with_padding_mask(B, H, Hkv, T, D):
    """Left-padded batches: additive mask kills a per-sequence prefix."""
    torch.manual_seed(7)
    q = torch.randn(B, H, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)
    mask = torch.zeros(B, T)
    for b in range(B):
        pad = (b * 37) % max(1, T - 1)   # varied pad lengths, always >=1 valid
        mask[b, :pad] = float("-inf")
    ref = reference_decode_attention(q, k, v, mask_add=mask, block_l=64)
    want = direct_attention(q, k, v, mask_add=mask)
    torch.testing.assert_close(ref, want, atol=1e-4, rtol=1e-4)


def test_reference_dtype_min_mask_acts_as_neg_inf():
    """transformers masks use finfo.min, not -inf; same softmax result."""
    torch.manual_seed(3)
    q = torch.randn(1, 4, 32)
    k = torch.randn(1, 2, 50, 32)
    v = torch.randn(1, 2, 50, 32)
    m_inf = torch.zeros(1, 50)
    m_inf[0, :10] = float("-inf")
    m_min = torch.zeros(1, 50)
    m_min[0, :10] = torch.finfo(torch.float16).min
    a = reference_decode_attention(q, k, v, mask_add=m_inf)
    b = reference_decode_attention(q, k, v, mask_add=m_min)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_reference_custom_scale():
    torch.manual_seed(5)
    q = torch.randn(2, 4, 16)
    k = torch.randn(2, 2, 40, 16)
    v = torch.randn(2, 2, 40, 16)
    scale = 0.5 / math.sqrt(16)
    ref = reference_decode_attention(q, k, v, scale=scale)
    want = direct_attention(q, k, v, scale=scale)
    torch.testing.assert_close(ref, want, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# paged variant: the block-table slot math
# --------------------------------------------------------------------------
def build_paged_case(B, Hkv, D, lens, block_size, seed=0):
    """Random pool + block tables whose physical blocks are deliberately
    scattered and interleaved across sequences (the case that catches slot
    math bugs). Returns (pool_k, pool_v, tables, contiguous_k, contiguous_v)."""
    torch.manual_seed(seed)
    import math as _m
    blocks_needed = [max(1, _m.ceil(l / block_size)) for l in lens]
    total = sum(blocks_needed)
    perm = torch.randperm(total * 2)[:total]      # scattered physical blocks
    num_blocks = total * 2
    pool_k = torch.randn(num_blocks * block_size, Hkv, D)
    pool_v = torch.randn(num_blocks * block_size, Hkv, D)
    tables = torch.zeros(B, max(blocks_needed), dtype=torch.long)
    i = 0
    for b, nb in enumerate(blocks_needed):
        tables[b, :nb] = perm[i:i + nb]
        i += nb
    # gather ground-truth contiguous KV per sequence (padded to max len)
    T = max(lens)
    ck = torch.zeros(B, Hkv, T, D)
    cv = torch.zeros(B, Hkv, T, D)
    for b, L in enumerate(lens):
        pos = torch.arange(L)
        slots = tables[b][pos // block_size] * block_size + pos % block_size
        ck[b, :, :L] = pool_k[slots].permute(1, 0, 2)
        cv[b, :, :L] = pool_v[slots].permute(1, 0, 2)
    return pool_k, pool_v, tables, ck, cv


def test_paged_reference_matches_direct():
    from server.kernels.paged_attention_triton import reference_paged_decode_attention
    B, H, Hkv, D, bs = 3, 14, 2, 64, 16
    lens = [1, 37, 200]                    # 1 token, partial block, many blocks
    pool_k, pool_v, tables, ck, cv = build_paged_case(B, Hkv, D, lens, bs, seed=9)
    torch.manual_seed(10)
    q = torch.randn(B, H, D)
    got = reference_paged_decode_attention(q, pool_k, pool_v, tables,
                                           torch.tensor(lens), bs)
    # ground truth: per-sequence direct attention on the gathered KV
    for b, L in enumerate(lens):
        want = direct_attention(q[b:b + 1], ck[b:b + 1, :, :L], cv[b:b + 1, :, :L])
        torch.testing.assert_close(got[b:b + 1], want, atol=1e-4, rtol=1e-4)


def test_paged_reference_block_boundary_lens():
    """Lengths exactly at and one past block boundaries."""
    from server.kernels.paged_attention_triton import reference_paged_decode_attention
    B, H, Hkv, D, bs = 4, 4, 2, 32, 16
    lens = [16, 17, 32, 33]
    pool_k, pool_v, tables, ck, cv = build_paged_case(B, Hkv, D, lens, bs, seed=13)
    torch.manual_seed(14)
    q = torch.randn(B, H, D)
    got = reference_paged_decode_attention(q, pool_k, pool_v, tables,
                                           torch.tensor(lens), bs)
    for b, L in enumerate(lens):
        want = direct_attention(q[b:b + 1], ck[b:b + 1, :, :L], cv[b:b + 1, :, :L])
        torch.testing.assert_close(got[b:b + 1], want, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# split-K merge math
# --------------------------------------------------------------------------
@pytest.mark.parametrize("splits", [1, 2, 4, 7, 64])
def test_split_reference_matches_direct(splits):
    from server.kernels.paged_attention_triton import reference_split_decode_attention
    torch.manual_seed(31 + splits)
    q = torch.randn(2, 14, 64)
    k = torch.randn(2, 2, 200, 64)
    v = torch.randn(2, 2, 200, 64)
    got = reference_split_decode_attention(q, k, v, num_splits=splits)
    want = direct_attention(q, k, v)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_split_reference_with_mask_and_empty_chunks():
    """Masked prefixes can make whole chunks contribute nothing; the merge
    must drop them cleanly (weight exp(-inf - m*) = 0)."""
    from server.kernels.paged_attention_triton import reference_split_decode_attention
    torch.manual_seed(41)
    q = torch.randn(2, 4, 32)
    k = torch.randn(2, 2, 96, 32)
    v = torch.randn(2, 2, 96, 32)
    mask = torch.zeros(2, 96)
    mask[0, :50] = float("-inf")   # first several chunks fully masked
    mask[1, :3] = float("-inf")
    got = reference_split_decode_attention(q, k, v, mask_add=mask, num_splits=8)
    want = direct_attention(q, k, v, mask_add=mask)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)

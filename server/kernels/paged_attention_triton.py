"""Fused decode attention in Triton (M1: contiguous KV).

One kernel program per (sequence, head). The program holds a single query row,
streams K/V tiles of BLOCK_L tokens, and keeps a running online softmax
(max m, denominator l, fp32 accumulator) so the full score row never
materializes. GQA is handled natively: head h reads kv-head h // (H // H_kv),
no repeat_kv copies.

This replaces, for decode steps (q_len == 1), the transformers SDPA path. It
plugs into any HF model through the attention-interface registry, so the
engines pick it up without code changes:

    from server.kernels.paged_attention_triton import use_triton_attention
    prev = use_triton_attention(model)     # model = ModelRunner().model

Correctness contract: token-identical greedy output vs the SDPA path
(tests/test_kernel_equivalence.py). The pure-PyTorch mirror of the same tiled
algorithm lives in reference_decode_attention() and is tested against SDPA on
CPU, so the algorithm and the Triton code are validated separately.

M2 replaces the contiguous K/V with block-table indexing into the paged pool.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # macOS dev box: no triton wheels. GPU boxes get it via torch.
    HAS_TRITON = False

# incremented every time the Triton path actually runs, so the equivalence
# test can assert it wasn't silently falling back to SDPA the whole time
KERNEL_CALLS = 0

_NEG_INF = float("-inf")


if HAS_TRITON:

    @triton.jit
    def _decode_attn_kernel(
        q_ptr, k_ptr, v_ptr, mask_ptr, out_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_mb, stride_mt,
        stride_ob, stride_oh, stride_od,
        T, D, scale,
        H: tl.constexpr, GROUP: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        kvh = h // GROUP

        d_offs = tl.arange(0, BLOCK_D)
        d_valid = d_offs < D
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d_offs * stride_qd,
                    mask=d_valid, other=0.0).to(tl.float32)

        m_i = tl.full([1], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for start in range(0, T, BLOCK_L):
            t_offs = start + tl.arange(0, BLOCK_L)
            t_valid = t_offs < T
            ld_mask = t_valid[:, None] & d_valid[None, :]

            k = tl.load(k_ptr + b * stride_kb + kvh * stride_kh
                        + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            s = tl.sum(q[None, :] * k, axis=1) * scale
            if HAS_MASK:
                mrow = tl.load(mask_ptr + b * stride_mb + t_offs * stride_mt,
                               mask=t_valid, other=float("-inf")).to(tl.float32)
                s = s + mrow
            s = tl.where(t_valid, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            # if no valid position has been seen yet m_new is -inf; shift by 0
            # instead so exp() sees finite arguments (exp(-inf - 0) = 0, no nan)
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe)
            alpha = tl.exp(m_i - m_safe)

            v = tl.load(v_ptr + b * stride_vb + kvh * stride_vh
                        + t_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        out = acc / l_safe
        tl.store(out_ptr + b * stride_ob + h * stride_oh + d_offs * stride_od,
                 out.to(out_ptr.dtype.element_ty), mask=d_valid)


def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     mask_add: torch.Tensor | None = None,
                     scale: float | None = None,
                     block_l: int = 128, num_warps: int = 4,
                     num_splits: int | None = None) -> torch.Tensor:
    """Fused decode attention over contiguous KV.

    q [B, H, D]; k, v [B, H_kv, T, D]; mask_add [B, T] additive float
    (0 = attend, -inf / dtype-min = masked) or None. Returns [B, H, D] in
    q's dtype. H must be a multiple of H_kv (GQA)."""
    if not HAS_TRITON:
        raise RuntimeError("triton is not available on this machine")
    B, H, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    assert H % Hkv == 0, "H must be a multiple of H_kv"
    if scale is None:
        scale = D ** -0.5
    out = torch.empty_like(q)
    has_mask = mask_add is not None
    if has_mask:
        assert mask_add.shape == (B, T), f"mask {tuple(mask_add.shape)} != {(B, T)}"
        m, smb, smt = mask_add, mask_add.stride(0), mask_add.stride(1)
    else:
        m, smb, smt = q, 0, 0  # dummy pointer, never read

    S = num_splits if num_splits is not None else _auto_splits(B, H, T, block_l)
    if S > 1:
        BD = triton.next_power_of_2(D)
        m_part = torch.empty(B * H * S, device=q.device, dtype=torch.float32)
        l_part = torch.empty(B * H * S, device=q.device, dtype=torch.float32)
        acc_part = torch.empty(B * H * S, BD, device=q.device, dtype=torch.float32)
        _decode_attn_split_kernel[(B * H * S,)](
            q, k, v, m, m_part, l_part, acc_part,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            smb, smt,
            T, D, scale,
            H=H, GROUP=H // Hkv, NUM_SPLITS=S,
            HAS_MASK=has_mask,
            BLOCK_L=block_l, BLOCK_D=BD,
            num_warps=num_warps,
        )
        return _merge_partials(m_part.view(B * H, S), l_part.view(B * H, S),
                               acc_part.view(B * H, S, BD)[:, :, :D], B, H, D, q.dtype)

    grid = (B * H,)
    _decode_attn_kernel[grid](
        q, k, v, m, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        smb, smt,
        out.stride(0), out.stride(1), out.stride(2),
        T, D, scale,
        H=H, GROUP=H // Hkv,
        HAS_MASK=has_mask,
        BLOCK_L=block_l, BLOCK_D=triton.next_power_of_2(D),
        num_warps=num_warps,
    )
    return out


if HAS_TRITON:

    @triton.jit
    def _paged_decode_attn_kernel(
        q_ptr, kp_ptr, vp_ptr, tbl_ptr, lens_ptr, out_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_ks, stride_kh, stride_kd,
        stride_vs, stride_vh, stride_vd,
        stride_tb, stride_tk,
        stride_ob, stride_oh, stride_od,
        D, scale,
        H: tl.constexpr, GROUP: tl.constexpr, BS: tl.constexpr,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """M2: same online softmax as the contiguous kernel, but K/V rows come
        straight out of the block pool via the block table. One program per
        (sequence, head); each program walks only its own sequence's true
        length, so there is no cross-sequence padding at all."""
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        kvh = h // GROUP

        d_offs = tl.arange(0, BLOCK_D)
        d_valid = d_offs < D
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d_offs * stride_qd,
                    mask=d_valid, other=0.0).to(tl.float32)
        seq_len = tl.load(lens_ptr + b)

        m_i = tl.full([1], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for start in range(0, seq_len, BLOCK_L):
            t_offs = start + tl.arange(0, BLOCK_L)
            t_valid = t_offs < seq_len
            # logical position -> physical slot through the block table
            blk = tl.load(tbl_ptr + b * stride_tb + (t_offs // BS) * stride_tk,
                          mask=t_valid, other=0)
            slot = blk * BS + t_offs % BS
            ld_mask = t_valid[:, None] & d_valid[None, :]

            k = tl.load(kp_ptr + slot[:, None] * stride_ks + kvh * stride_kh
                        + d_offs[None, :] * stride_kd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            s = tl.sum(q[None, :] * k, axis=1) * scale
            s = tl.where(t_valid, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s - m_safe)
            alpha = tl.exp(m_i - m_safe)

            v = tl.load(vp_ptr + slot[:, None] * stride_vs + kvh * stride_vh
                        + d_offs[None, :] * stride_vd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        out = acc / l_safe
        tl.store(out_ptr + b * stride_ob + h * stride_oh + d_offs * stride_od,
                 out.to(out_ptr.dtype.element_ty), mask=d_valid)


def paged_decode_attention(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                           block_tables: torch.Tensor, lens: torch.Tensor,
                           block_size: int, scale: float | None = None,
                           block_l: int = 128, num_warps: int = 4,
                           num_splits: int | None = None,
                           max_len: int | None = None) -> torch.Tensor:
    """Fused decode attention straight over the paged pool (no gather).

    q [B, H, D]; k_pool, v_pool flat [n_slots, H_kv, D] (PagedKVStore._flat
    layout); block_tables [B, max_blocks] int (rows padded with anything, only
    the first ceil(len/bs) entries of each row are read); lens [B] int true
    lengths. BLOCK_L must be a multiple of block_size so a tile never
    straddles a partial block boundary mid-token."""
    if not HAS_TRITON:
        raise RuntimeError("triton is not available on this machine")
    B, H, D = q.shape
    Hkv = k_pool.shape[1]
    assert H % Hkv == 0
    assert block_l % block_size == 0, "BLOCK_L must be a multiple of block_size"
    if scale is None:
        scale = D ** -0.5
    if block_tables.dtype not in (torch.int32, torch.int64):
        block_tables = block_tables.to(torch.int32)
    if lens.dtype not in (torch.int32, torch.int64):
        lens = lens.to(torch.int32)
    out = torch.empty_like(q)

    if num_splits is not None:
        S = num_splits
    else:
        # per-seq lengths vary; size the split policy on the longest
        T_est = max_len if max_len is not None else int(lens.max())
        S = _auto_splits(B, H, T_est, block_l)
    if S > 1:
        BD = triton.next_power_of_2(D)
        m_part = torch.empty(B * H * S, device=q.device, dtype=torch.float32)
        l_part = torch.empty(B * H * S, device=q.device, dtype=torch.float32)
        acc_part = torch.empty(B * H * S, BD, device=q.device, dtype=torch.float32)
        _paged_decode_attn_split_kernel[(B * H * S,)](
            q, k_pool, v_pool, block_tables, lens, m_part, l_part, acc_part,
            q.stride(0), q.stride(1), q.stride(2),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
            block_tables.stride(0), block_tables.stride(1),
            D, scale,
            H=H, GROUP=H // Hkv, BS=block_size, NUM_SPLITS=S,
            BLOCK_L=block_l, BLOCK_D=BD,
            num_warps=num_warps,
        )
        return _merge_partials(m_part.view(B * H, S), l_part.view(B * H, S),
                               acc_part.view(B * H, S, BD)[:, :, :D], B, H, D, q.dtype)

    grid = (B * H,)
    _paged_decode_attn_kernel[grid](
        q, k_pool, v_pool, block_tables, lens, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        D, scale,
        H=H, GROUP=H // Hkv, BS=block_size,
        BLOCK_L=block_l, BLOCK_D=triton.next_power_of_2(D),
        num_warps=num_warps,
    )
    return out



if HAS_TRITON:

    @triton.jit
    def _decode_attn_split_kernel(
        q_ptr, k_ptr, v_ptr, mask_ptr, m_ptr, l_ptr, acc_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_mb, stride_mt,
        T, D, scale,
        H: tl.constexpr, GROUP: tl.constexpr, NUM_SPLITS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Split-K variant: NUM_SPLITS programs share one (sequence, head),
        each reducing a chunk of the context into partial (m, l, acc). At
        small batches one program per (b, h) starves the GPU; this is the
        flash-decoding fix. Partials merge on the host (log-sum-exp)."""
        pid = tl.program_id(0)
        s = pid % NUM_SPLITS
        bh = pid // NUM_SPLITS
        b = bh // H
        h = bh % H
        kvh = h // GROUP

        chunk = tl.cdiv(T, NUM_SPLITS)
        t_start = s * chunk
        t_end = tl.minimum(t_start + chunk, T)

        d_offs = tl.arange(0, BLOCK_D)
        d_valid = d_offs < D
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d_offs * stride_qd,
                    mask=d_valid, other=0.0).to(tl.float32)

        m_i = tl.full([1], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for start in range(t_start, t_end, BLOCK_L):
            t_offs = start + tl.arange(0, BLOCK_L)
            t_valid = t_offs < t_end
            ld_mask = t_valid[:, None] & d_valid[None, :]
            k = tl.load(k_ptr + b * stride_kb + kvh * stride_kh
                        + t_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            sc = tl.sum(q[None, :] * k, axis=1) * scale
            if HAS_MASK:
                mrow = tl.load(mask_ptr + b * stride_mb + t_offs * stride_mt,
                               mask=t_valid, other=float("-inf")).to(tl.float32)
                sc = sc + mrow
            sc = tl.where(t_valid, sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, axis=0))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(sc - m_safe)
            alpha = tl.exp(m_i - m_safe)
            v = tl.load(v_ptr + b * stride_vb + kvh * stride_vh
                        + t_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        part = bh * NUM_SPLITS + s
        one = tl.arange(0, 1)
        tl.store(m_ptr + part + one, m_i)
        tl.store(l_ptr + part + one, l_i)
        tl.store(acc_ptr + part * BLOCK_D + d_offs, acc, mask=d_valid)

    @triton.jit
    def _paged_decode_attn_split_kernel(
        q_ptr, kp_ptr, vp_ptr, tbl_ptr, lens_ptr, m_ptr, l_ptr, acc_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_ks, stride_kh, stride_kd,
        stride_vs, stride_vh, stride_vd,
        stride_tb, stride_tk,
        D, scale,
        H: tl.constexpr, GROUP: tl.constexpr, BS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        s = pid % NUM_SPLITS
        bh = pid // NUM_SPLITS
        b = bh // H
        h = bh % H
        kvh = h // GROUP

        seq_len = tl.load(lens_ptr + b)
        chunk = tl.cdiv(seq_len, NUM_SPLITS)
        t_start = s * chunk
        t_end = tl.minimum(t_start + chunk, seq_len)

        d_offs = tl.arange(0, BLOCK_D)
        d_valid = d_offs < D
        q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d_offs * stride_qd,
                    mask=d_valid, other=0.0).to(tl.float32)

        m_i = tl.full([1], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for start in range(t_start, t_end, BLOCK_L):
            t_offs = start + tl.arange(0, BLOCK_L)
            t_valid = t_offs < t_end
            blk = tl.load(tbl_ptr + b * stride_tb + (t_offs // BS) * stride_tk,
                          mask=t_valid, other=0)
            slot = blk * BS + t_offs % BS
            ld_mask = t_valid[:, None] & d_valid[None, :]
            k = tl.load(kp_ptr + slot[:, None] * stride_ks + kvh * stride_kh
                        + d_offs[None, :] * stride_kd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            sc = tl.sum(q[None, :] * k, axis=1) * scale
            sc = tl.where(t_valid, sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, axis=0))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(sc - m_safe)
            alpha = tl.exp(m_i - m_safe)
            v = tl.load(vp_ptr + slot[:, None] * stride_vs + kvh * stride_vh
                        + d_offs[None, :] * stride_vd,
                        mask=ld_mask, other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new

        part = bh * NUM_SPLITS + s
        one = tl.arange(0, 1)
        tl.store(m_ptr + part + one, m_i)
        tl.store(l_ptr + part + one, l_i)
        tl.store(acc_ptr + part * BLOCK_D + d_offs, acc, mask=d_valid)


def _auto_splits(B: int, H: int, T: int, block_l: int) -> int:
    """More programs until the GPU has work: target ~128 concurrent programs
    (a few waves on a 40-SM T4). Never split below one tile per chunk."""
    progs = B * H
    if progs >= 128 or T <= 2 * block_l:
        return 1
    by_occupancy = max(1, 128 // progs)
    by_length = max(1, (T + block_l - 1) // block_l)
    return int(min(by_occupancy, by_length, 32))


def _merge_partials(m_part, l_part, acc_part, B, H, D, dtype):
    """Log-sum-exp merge of split partials. m/l: [B*H, S]; acc: [B*H, S, D]."""
    m_star = m_part.max(dim=1).values                      # [BH]
    w = torch.exp(m_part - m_star[:, None])                # -inf chunks -> 0
    l_star = (l_part * w).sum(dim=1)                       # [BH]
    out = (acc_part * w[:, :, None]).sum(dim=1) / l_star[:, None]
    return out.view(B, H, D).to(dtype)


# ---------------------------------------------------------------------------
# pure-PyTorch mirror of the exact tiled algorithm, for CPU-side validation
# ---------------------------------------------------------------------------
def reference_decode_attention(q, k, v, mask_add=None, scale=None, block_l=128):
    """Same online-softmax tiling as the kernel, in plain torch. Slow; exists
    so the algorithm can be tested against SDPA without a GPU."""
    B, H, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    group = H // Hkv
    if scale is None:
        scale = D ** -0.5
    out = torch.empty(B, H, D, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            qq = q[b, h].float()
            kk = k[b, h // group].float()
            vv = v[b, h // group].float()
            m = _NEG_INF
            l = 0.0
            acc = torch.zeros(D)
            for s0 in range(0, T, block_l):
                kt, vt = kk[s0:s0 + block_l], vv[s0:s0 + block_l]
                s = (qq[None, :] * kt).sum(-1) * scale
                if mask_add is not None:
                    s = s + mask_add[b, s0:s0 + block_l].float()
                m_new = max(m, s.max().item())
                m_safe = 0.0 if m_new == _NEG_INF else m_new
                p = torch.exp(s - m_safe)
                alpha = math.exp(m - m_safe) if m != _NEG_INF else 0.0
                acc = acc * alpha + p @ vt
                l = l * alpha + p.sum().item()
                m = m_new
            out[b, h] = acc / (l if l else 1.0)
    return out.to(q.dtype)


def reference_paged_decode_attention(q, k_pool, v_pool, block_tables, lens,
                                     block_size, scale=None):
    """Torch mirror of the paged kernel's indexing: gather each sequence's
    slots through its block table, then run the same tiled algorithm. Exists
    so the slot math is testable on CPU."""
    B = q.shape[0]
    outs = []
    for b in range(B):
        L = int(lens[b])
        pos = torch.arange(L)
        tbl = block_tables[b].long()
        slots = tbl[pos // block_size] * block_size + pos % block_size
        k = k_pool[slots].permute(1, 0, 2)[None]   # [1, Hkv, L, D]
        v = v_pool[slots].permute(1, 0, 2)[None]
        outs.append(reference_decode_attention(q[b:b + 1], k, v, scale=scale))
    return torch.cat(outs, dim=0)




def reference_split_decode_attention(q, k, v, mask_add=None, scale=None,
                                     num_splits=4):
    """Torch mirror of the split-K scheme: per-chunk partial (m, l, acc)
    computed directly, merged with the same log-sum-exp combine the GPU path
    uses. Validates the merge math on CPU."""
    B, H, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    group = H // Hkv
    if scale is None:
        scale = D ** -0.5
    chunk = -(-T // num_splits)
    out = torch.empty(B, H, D, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            qq = q[b, h].float()
            kk = k[b, h // group].float()
            vv = v[b, h // group].float()
            ms, ls, accs = [], [], []
            for s in range(num_splits):
                t0, t1 = s * chunk, min((s + 1) * chunk, T)
                if t0 >= t1:
                    ms.append(_NEG_INF); ls.append(0.0); accs.append(torch.zeros(D))
                    continue
                sc = (qq[None, :] * kk[t0:t1]).sum(-1) * scale
                if mask_add is not None:
                    sc = sc + mask_add[b, t0:t1].float()
                m = sc.max().item()
                if m == _NEG_INF:            # fully masked chunk
                    ms.append(_NEG_INF); ls.append(0.0); accs.append(torch.zeros(D))
                    continue
                p = torch.exp(sc - m)
                ms.append(m); ls.append(p.sum().item()); accs.append(p @ vv[t0:t1])
            m_star = max(ms)
            w = [math.exp(m - m_star) if m != _NEG_INF else 0.0 for m in ms]
            l_star = sum(l * wi for l, wi in zip(ls, w))
            acc = sum((a * wi for a, wi in zip(accs, w)), torch.zeros(D))
            out[b, h] = acc / l_star
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# transformers integration: register as a custom attention implementation
# ---------------------------------------------------------------------------
ATTN_NAME = "nanoserve_triton"


def _attention_forward(module, query, key, value, attention_mask,
                       dropout=0.0, scaling=None, sliding_window=None, **kwargs):
    """transformers attention-interface entry point. Kernel path for CUDA
    decode steps (q_len == 1); everything else falls back to SDPA, so prefill
    and CPU runs are untouched. A PagedKV handle (from NanoPagedCache) means
    the fused no-gather path: kernel straight over the pool on CUDA, gather +
    SDPA fallback elsewhere."""
    global KERNEL_CALLS
    from .paged_runtime import PagedKV, gather_sdpa_fallback
    if isinstance(key, PagedKV):
        h = key
        if HAS_TRITON and query.is_cuda:
            out = paged_decode_attention(query[:, :, 0, :], h.k_flat, h.v_flat,
                                         h.tables, h.lens, h.block_size,
                                         scale=scaling, max_len=h.max_len)
            KERNEL_CALLS += 1
            return out[:, None, :, :], None
        return gather_sdpa_fallback(query, h, scale=scaling).transpose(1, 2), None

    q_len = query.shape[2]
    use_kernel = (
        HAS_TRITON and query.is_cuda and q_len == 1
        and dropout == 0.0 and sliding_window is None
    )
    if not use_kernel:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling,
                                      sliding_window=sliding_window, **kwargs)

    T = key.shape[2]
    mask_row = None
    if attention_mask is not None:
        am = attention_mask
        if am.dtype == torch.bool:
            am = torch.where(am, 0.0, torch.finfo(query.dtype).min).to(query.dtype)
        # [B, 1, q_len, T_total] -> the last query row, trimmed to kv length
        mask_row = am[:, 0, -1, :T].contiguous()

    out = decode_attention(query[:, :, 0, :], key, value, mask_row, scale=scaling)
    KERNEL_CALLS += 1
    return out[:, None, :, :], None  # [B, q_len=1, H, D], matching sdpa's layout


def use_triton_attention(model) -> str:
    """Register the kernel as attention implementation ATTN_NAME and switch
    `model` to it. Returns the previous implementation name (for restore)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    try:
        ALL_ATTENTION_FUNCTIONS.register(ATTN_NAME, _attention_forward)
    except Exception:
        try:
            ALL_ATTENTION_FUNCTIONS[ATTN_NAME] = _attention_forward
        except Exception:
            ALL_ATTENTION_FUNCTIONS._global_mapping[ATTN_NAME] = _attention_forward
    # custom implementations need a mask flavor too; eager's float mask is what
    # the kernel expects. Registries differ across 5.x minors, hence the guards.
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        eager_mask = ALL_MASK_ATTENTION_FUNCTIONS["eager"]
        try:
            ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_NAME, eager_mask)
        except Exception:
            try:
                ALL_MASK_ATTENTION_FUNCTIONS[ATTN_NAME] = eager_mask
            except Exception:
                ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[ATTN_NAME] = eager_mask
    except ImportError:
        pass
    prev = model.config._attn_implementation
    model.config._attn_implementation = ATTN_NAME
    return prev


def restore_attention(model, prev: str):
    model.config._attn_implementation = prev

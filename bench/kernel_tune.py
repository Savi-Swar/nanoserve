"""Tune the decode kernel's launch parameters on the actual GPU.

T4 (sm_75) has no cp.async, so pipelining stages buy nothing; the knobs that
matter are BLOCK_L (tile length), num_warps, and the split count. Rather than
triton's runtime autotuner (which re-benchmarks silently on new shapes), this
sweeps the grid once per (B, T) and writes the winners to
results/kernel_tune.json; the defaults get baked from that.

Two stages per (B, T): pick (BLOCK_L, num_warps) at S=1, then sweep S with the
winner. Correctness is asserted against SDPA for every timed config.

    python -m bench.kernel_tune
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def time_fn(fn, warmup=10, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    p.add_argument("--seq-lens", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--heads", type=int, default=14)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--out", default="results/kernel_tune.json")
    a = p.parse_args()

    if not torch.cuda.is_available():
        print("[!] no CUDA; tuner is a no-op here")
        return

    from server.kernels.paged_attention_triton import decode_attention

    H, Hkv, D = a.heads, a.kv_heads, a.head_dim
    group = H // Hkv
    print(f"tuning decode kernel, H={H} Hkv={Hkv} D={D} on "
          f"{torch.cuda.get_device_name(0)}")

    results = []
    for T in a.seq_lens:
        for B in a.batches:
            torch.manual_seed(B * 7 + T)
            q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
            k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
            v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.float16)
            kk = k.repeat_interleave(group, dim=1)
            vv = v.repeat_interleave(group, dim=1)
            want = torch.nn.functional.scaled_dot_product_attention(
                q[:, :, None, :], kk, vv)[:, :, 0, :]

            # stage 1: (BLOCK_L, warps) at S=1
            best = None
            for bl in (64, 128, 256):
                for w in (2, 4, 8):
                    got = decode_attention(q, k, v, block_l=bl, num_warps=w,
                                           num_splits=1)
                    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
                    ms = time_fn(lambda: decode_attention(
                        q, k, v, block_l=bl, num_warps=w, num_splits=1))
                    if best is None or ms < best[0]:
                        best = (ms, bl, w)
            ms1, bl, w = best

            # stage 2: split sweep with the stage-1 winner
            best_s = (ms1, 1)
            for S in (2, 4, 8, 16, 32):
                if S * 1 > max(1, T // 64):
                    continue
                got = decode_attention(q, k, v, block_l=bl, num_warps=w,
                                       num_splits=S)
                torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
                ms = time_fn(lambda: decode_attention(
                    q, k, v, block_l=bl, num_warps=w, num_splits=S))
                if ms < best_s[0]:
                    best_s = (ms, S)
            ms_best, S = best_s

            results.append({"batch": B, "seq_len": T, "block_l": bl,
                            "num_warps": w, "splits": S,
                            "ms_nosplit": ms1, "ms_best": ms_best})
            print(f"  B={B:>3} T={T:>5}: BLOCK_L={bl:>3} warps={w} S={S:>2}  "
                  f"{ms_best:.3f}ms  (S=1 best {ms1:.3f}ms)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "heads": H,
                   "kv_heads": Hkv, "head_dim": D, "rows": results}, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

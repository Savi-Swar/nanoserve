"""Op-level decode-attention timing: Triton kernel vs the SDPA path it
replaces (repeat_kv + scaled_dot_product_attention), at Qwen2.5-0.5B shapes.

This is the per-iteration perf signal for kernel work; the end-to-end number
(engine throughput vs vLLM) comes from the full pipeline. CUDA only.

    python -m bench.kernel_bench --batches 1 8 16 32 --seq-len 2048
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def time_fn(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--heads", type=int, default=14)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--out", default="results/kernel_bench.json")
    a = p.parse_args()

    if not torch.cuda.is_available():
        print("[!] no CUDA; kernel bench is a no-op here")
        return

    from server.kernels.paged_attention_triton import decode_attention

    H, Hkv, D, T = a.heads, a.kv_heads, a.head_dim, a.seq_len
    group = H // Hkv
    dev = "cuda"
    print(f"decode attention, H={H} Hkv={Hkv} D={D} T={T} fp16 "
          f"({torch.cuda.get_device_name(0)})")
    print(f"{'B':>4} {'sdpa ms':>10} {'triton ms':>10} {'speedup':>9}")

    rows = []
    for B in a.batches:
        torch.manual_seed(B)
        q = torch.randn(B, H, D, device=dev, dtype=torch.float16)
        k = torch.randn(B, Hkv, T, D, device=dev, dtype=torch.float16)
        v = torch.randn(B, Hkv, T, D, device=dev, dtype=torch.float16)
        q4 = q[:, :, None, :]

        def run_sdpa():
            kk = k.repeat_interleave(group, dim=1)
            vv = v.repeat_interleave(group, dim=1)
            return torch.nn.functional.scaled_dot_product_attention(q4, kk, vv)

        def run_triton():
            return decode_attention(q, k, v)

        # sanity before timing: same answer
        torch.testing.assert_close(run_triton(), run_sdpa()[:, :, 0, :],
                                   atol=2e-2, rtol=2e-2)
        ms_sdpa = time_fn(run_sdpa)
        ms_tri = time_fn(run_triton)
        rows.append({"batch": B, "sdpa_ms": ms_sdpa, "triton_ms": ms_tri,
                     "speedup": ms_sdpa / ms_tri})
        print(f"{B:>4} {ms_sdpa:>10.3f} {ms_tri:>10.3f} {ms_sdpa / ms_tri:>8.2f}x")

    out = {"heads": H, "kv_heads": Hkv, "head_dim": D, "seq_len": T,
           "device": torch.cuda.get_device_name(0), "rows": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")
    print("note: sdpa baseline here includes the repeat_kv copies the kernel "
          "avoids; the fair end-to-end comparison is the engine sweep.")


if __name__ == "__main__":
    main()

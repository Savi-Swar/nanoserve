"""Can this host overlap two GPUs from one thread at all?

The graphed interleave shows zero cross-device overlap (four stage replays
run strictly serially). Before blaming the engine, test the substrate:

  1. raw kernels: big matmuls issued on both devices back to back
  2. graph replays: the same matmuls captured per device, replayed
  3. the pipeline shape: matmul d0, copy d0->d1, matmul d1, two chains

Each reports serial wall vs concurrent wall; a ratio near 1.0 means the
host or driver serializes and no engine change can help.

    python scripts/overlap_probe.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

N = 4096
ITERS = 30


def work(dev, x, w):
    y = x
    for _ in range(ITERS):
        y = y @ w
    return y


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def main():
    assert torch.cuda.device_count() >= 2, "needs two GPUs"
    d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
    x0 = torch.randn(N, N, device=d0, dtype=torch.float16)
    w0 = torch.randn(N, N, device=d0, dtype=torch.float16)
    x1 = torch.randn(N, N, device=d1, dtype=torch.float16)
    w1 = torch.randn(N, N, device=d1, dtype=torch.float16)

    # warm
    work(d0, x0, w0); work(d1, x1, w1); sync_all()

    # 1. raw kernels
    t0 = time.perf_counter()
    work(d0, x0, w0); sync_all()
    work(d1, x1, w1); sync_all()
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    work(d0, x0, w0)
    work(d1, x1, w1)
    sync_all()
    conc = time.perf_counter() - t0
    print(f"raw kernels : serial {serial*1e3:7.1f}ms  concurrent "
          f"{conc*1e3:7.1f}ms  ratio {serial/conc:.2f}x")

    # 2. graph replays
    graphs = []
    for dev, x, w in ((d0, x0, w0), (d1, x1, w1)):
        with torch.cuda.device(dev):
            torch.cuda.synchronize(dev)
            st = torch.cuda.Stream(dev)
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                work(dev, x, w)
            torch.cuda.synchronize(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st):
                work(dev, x, w)
            graphs.append(g)
    sync_all()
    t0 = time.perf_counter()
    graphs[0].replay(); sync_all()
    graphs[1].replay(); sync_all()
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    graphs[0].replay()
    graphs[1].replay()
    sync_all()
    conc = time.perf_counter() - t0
    print(f"graph replay: serial {serial*1e3:7.1f}ms  concurrent "
          f"{conc*1e3:7.1f}ms  ratio {serial/conc:.2f}x")

    # 3. two pipeline-shaped chains (m0 -> copy -> m1), interleaved
    h1a = torch.empty(N, N, device=d1, dtype=torch.float16)
    h1b = torch.empty(N, N, device=d1, dtype=torch.float16)

    def chain(hbuf):
        y = work(d0, x0, w0)
        hbuf.copy_(y)
        return work(d1, hbuf, w1)

    chain(h1a); sync_all()
    t0 = time.perf_counter()
    chain(h1a); sync_all()
    chain(h1b); sync_all()
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    chain(h1a)
    chain(h1b)
    sync_all()
    conc = time.perf_counter() - t0
    print(f"2 chains    : serial {serial*1e3:7.1f}ms  concurrent "
          f"{conc*1e3:7.1f}ms  ratio {serial/conc:.2f}x "
          f"(ideal ~1.5x: chain B's d0 half overlaps chain A's d1 half)")

    p2p = torch.cuda.can_device_access_peer(0, 1)
    print(f"p2p access 0->1: {p2p}")


if __name__ == "__main__":
    main()

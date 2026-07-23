"""Replay a real Azure LLM inference trace instead of synthetic uniform load.

Real traffic is nothing like `poisson arrivals + 512-token prompts`: prompt
lengths are heavy-tailed, output lengths are unpredictable, arrivals are
bursty. The trace carries per-request arrival time, context (prompt) token
count, and generated token count. Prompt *content* doesn't affect serving
performance — only lengths and arrival timing do — so we synthesize a prompt of
the exact context length from a filler token.

Trace: Azure/AzurePublicDataset -> data/azure_llm_conv.csv
columns: TIMESTAMP, ContextTokens, GeneratedTokens.

`len_scale` divides both lengths so the heavy-tailed *shape* is preserved but
the run is CPU-feasible; use len_scale=1 on a GPU for the real thing.
"""
from __future__ import annotations

import csv
from datetime import datetime

from server.request import Request, SamplingParams

_FMT = "%Y-%m-%d %H:%M:%S.%f"
DEFAULT_PATH = "data/azure_llm_conv.csv"
FILLER_TOKEN = 1000  # any benign in-vocab id; content is irrelevant to timing


def load_rows(path: str = DEFAULT_PATH):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["TIMESTAMP"], int(r["ContextTokens"]), int(r["GeneratedTokens"])))
    return rows


def build_trace_requests(path: str, n: int, start: int = 0, len_scale: float = 1.0,
                         time_scale: float = 1.0):
    """Return (requests, arrival_offsets) for `n` real requests starting at
    row `start`, with lengths divided by `len_scale` and inter-arrival gaps
    multiplied by `time_scale`."""
    rows = load_rows(path)
    window = rows[start:start + n]
    if not window:
        raise ValueError(f"trace window empty (start={start}, n={n}, rows={len(rows)})")
    t0 = datetime.strptime(window[0][0][:26], _FMT)
    reqs, offsets = [], []
    for i, (ts, ctx, gen) in enumerate(window):
        off = (datetime.strptime(ts[:26], _FMT) - t0).total_seconds() * time_scale
        c = max(1, round(ctx / len_scale))
        g = max(1, round(gen / len_scale))
        reqs.append(Request(
            id=i, prompt="",
            sampling=SamplingParams(max_tokens=g, temperature=0.0, ignore_eos=True),
            prompt_ids=[FILLER_TOKEN] * c,
        ))
        offsets.append(off)
    base = offsets[0]
    return reqs, [o - base for o in offsets]


def effective_rate(offsets: list[float]) -> float:
    span = offsets[-1] - offsets[0]
    return len(offsets) / span if span > 0 else float(len(offsets))

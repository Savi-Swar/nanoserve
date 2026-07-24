"""Open-loop load generation. Requests arrive on a wall clock following a
Poisson process (exponential inter-arrival gaps), the standard model for
independent clients hitting a service. Open-loop means arrivals don't wait for
prior requests to finish, so an overloaded engine builds a real queue and TTFT
blows up. That queue is what continuous batching kills.
"""
from __future__ import annotations

import random
import time

from server.request import Request, SamplingParams

# Varied-length prompts, so prefill cost and sequence length differ across
# requests. Uneven finish times are what make static batching waste the GPU and
# continuous batching win.
PROMPT_BANK = [
    "Hi.",
    "What is 17 times 24?",
    "Write one sentence about the ocean.",
    "List three prime numbers.",
    "Explain what a hash map is in two sentences.",
    "Summarize why the sky is blue.",
    "Give me a haiku about winter.",
    "Translate 'good morning' into French and Spanish.",
    "Name a fruit, a color, and a country.",
    "Describe the water cycle briefly.",
    "What are the first five Fibonacci numbers?",
    "Write a short function signature for binary search in Python.",
    "In one line, what does a CPU cache do?",
    "Explain recursion to a five year old in two sentences.",
    "What is the capital of Japan and roughly how many people live there?",
]


def sample_prompt(rng: random.Random) -> str:
    return rng.choice(PROMPT_BANK)


def build_requests(
    n: int,
    rate: float,
    max_tokens: int,
    seed: int = 0,
    temperature: float = 0.0,
    jitter_tokens: bool = True,
) -> tuple[list[Request], list[float]]:
    """Return (requests, arrival_offsets_seconds). Offsets are relative to t0."""
    rng = random.Random(seed)
    reqs, offsets = [], []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(rate)  # exponential gap => Poisson arrivals
        # vary output length ±50% so finish times are uneven
        mt = max_tokens
        if jitter_tokens:
            mt = max(8, int(max_tokens * rng.uniform(0.5, 1.5)))
        reqs.append(
            Request(
                id=i,
                prompt=sample_prompt(rng),
                sampling=SamplingParams(
                    max_tokens=mt, temperature=temperature, ignore_eos=True
                ),
            )
        )
        offsets.append(t)
    return reqs, offsets


def replay(engine, reqs: list[Request], offsets: list[float]):
    """Submit each request when its arrival offset elapses on the wall clock."""
    t0 = time.perf_counter()
    for req, off in zip(reqs, offsets):
        now = time.perf_counter() - t0
        if off > now:
            time.sleep(off - now)
        engine.submit(req)

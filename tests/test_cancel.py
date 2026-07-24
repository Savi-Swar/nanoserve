"""Cancellation / block-reclamation guarantees.

The fast test hammers the allocator directly (no model) and is the real leak
guard in CI. The RUN_SLOW test drives the actual paged engine: cancel a random
subset of a batch mid-generation and assert the pool is fully reclaimed.
"""
import os

import pytest

from bench.cancel_chaos import alloc_stress, engine_chaos


def test_allocator_never_leaks_under_abort_chaos():
    # thousands of admit/grow/abort/free cycles; a single leaked block fails.
    r = alloc_stress(num_blocks=256, block_size=16, cycles=1500, seed=1)
    assert r["leaked"] == 0
    assert r["aborts"] > 0  # we actually exercised the abort path


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1",
                    reason="set RUN_SLOW=1 (loads the model)")
def test_engine_reclaims_blocks_on_cancel():
    r = engine_chaos(cycles=6, per_cycle=8, num_blocks=512, kill_frac=0.6,
                     max_tokens=10, device="cpu")
    assert r["killed_midstream"] > 0
    assert r["leaked_cycles"] == 0

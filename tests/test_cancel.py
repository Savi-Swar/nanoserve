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


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1",
                    reason="set RUN_SLOW=1 (loads the model)")
def test_impossible_request_rejected_not_wedged():
    """A request whose reservation exceeds the whole pool must be rejected at
    admission, and requests behind it must still be served. Before the fix
    the empty-batch force-admit raised OutOfBlocks inside the engine thread
    and every queued request hung forever (the vLLM #39734 failure class)."""
    import time

    from server.engine import ENGINES
    from server.model import ModelRunner
    from server.request import Request, SamplingParams

    m = ModelRunner("Qwen/Qwen2.5-0.5B", device="cpu")
    done = []
    eng = ENGINES["paged"](m, on_finish=lambda r: done.append(r.id),
                           max_batch=4, num_blocks=32, block_size=16)
    eng.start()
    poison = Request(0, "hello", SamplingParams(max_tokens=2000,
                                                temperature=0.0,
                                                ignore_eos=True))
    normal = Request(1, "hi", SamplingParams(max_tokens=5, temperature=0.0,
                                             ignore_eos=True))
    eng.submit(poison)
    eng.submit(normal)
    t0 = time.time()
    while len(done) < 2 and time.time() - t0 < 120:
        time.sleep(0.2)
    eng.stop()
    assert poison.status == "rejected"
    assert normal.status == "done"
    assert len(normal.output_tokens) == 5
    assert eng.state.alloc.num_free == 32

"""C++ scheduler vs the Python bookkeeping: identical decisions, proven by
replaying the same trace through both and hashing every decision. Skipped when
the extension isn't built (make cpp)."""
import pytest

nc = pytest.importorskip("nanoserve_core")

from bench.sched_replay import (BLOCK_SIZE, NUM_BLOCKS, PyScheduler,
                                build_trace, run_trace)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpp_decisions_match_python(seed):
    trace = build_trace(300, seed)
    h_py, steps_py, _ = run_trace(PyScheduler(NUM_BLOCKS, BLOCK_SIZE), trace)
    h_cc, steps_cc, _ = run_trace(nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE), trace)
    assert steps_py == steps_cc
    assert h_py == h_cc


def test_both_deterministic():
    trace = build_trace(300, 7)
    assert run_trace(PyScheduler(NUM_BLOCKS, BLOCK_SIZE), trace)[0] \
        == run_trace(PyScheduler(NUM_BLOCKS, BLOCK_SIZE), trace)[0]
    assert run_trace(nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE), trace)[0] \
        == run_trace(nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE), trace)[0]


def test_pool_drains_clean():
    # every block freed once the trace fully drains, in both impls
    trace = build_trace(200, 3)
    for sched in (PyScheduler(NUM_BLOCKS, BLOCK_SIZE),
                  nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE)):
        run_trace(sched, trace)
        assert sched.size == 0
        assert sched.num_free_blocks == NUM_BLOCKS


def test_eos_cuts_early():
    s = nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE)
    s.admit(0, 10, 50)
    s.admit(1, 10, 50)
    s.plan_step()
    fin = s.commit_step([0, 1])   # row 1 hits eos
    assert fin == [1]
    assert s.size == 1
    assert s.row_ids() == [0]


def test_lockfree_variant_matches():
    trace = build_trace(200, 5)
    h_mutex, _, _ = run_trace(nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE, False), trace)
    h_lf, _, _ = run_trace(nc.Scheduler(NUM_BLOCKS, BLOCK_SIZE, True), trace)
    assert h_mutex == h_lf

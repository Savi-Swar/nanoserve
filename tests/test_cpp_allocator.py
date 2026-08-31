"""C++ allocator vs Python allocator, differentially.

Same seeded stream of admit/grow/abort ops runs against both; block tables,
free counts, and OutOfBlocks behavior must match exactly at every step. LIFO
order is part of the contract (both pop the same physical blocks in the same
order), which is what makes the C++ port drop-in.

Skipped unless the extension is built: `make cpp`.
"""
import random

import pytest

nc = pytest.importorskip("nanoserve_core")

from server.paged_cache import BlockAllocator as PyAlloc, OutOfBlocks


@pytest.mark.parametrize("lockfree", [False, True])
def test_differential_lockstep(lockfree):
    rng = random.Random(7)
    NB, BS = 128, 16
    py = PyAlloc(NB, BS)
    cc = nc.BlockAllocator(NB, BS, lockfree=lockfree)

    live = []
    sid = 0
    for cycle in range(300):
        # admit a few
        for _ in range(rng.randint(0, 4)):
            need = rng.randint(1, 6) * BS
            can_py = py.can_admit(need)
            assert can_py == cc.can_admit(need)
            if not can_py:
                continue
            t_py = py.add_seq(sid, need)
            t_cc = cc.add_seq(sid, need)
            assert t_py == list(t_cc), f"table mismatch on add sid={sid}"
            live.append(sid)
            sid += 1
        # grow / abort
        for s in list(live):
            r = rng.random()
            if r < 0.15:
                py.free_seq(s)
                cc.free_seq(s)
                live.remove(s)
                continue
            try:
                b_py = py.append_token(s)
                oob_py = False
            except OutOfBlocks:
                b_py, oob_py = None, True
            try:
                b_cc = cc.append_token(s)
                oob_cc = False
            except RuntimeError:
                b_cc, oob_cc = None, True
            assert oob_py == oob_cc
            if oob_py:                       # python leaves state; drop the seq
                py.free_seq(s)
                cc.free_seq(s)
                live.remove(s)
                continue
            assert b_py == b_cc, f"grow block mismatch sid={s}"
        assert py.num_free == cc.num_free
        # spot-check full tables
        for s in live[:5]:
            assert py.tables[s] == list(cc.table(s))

    for s in live:
        py.free_seq(s)
        cc.free_seq(s)
    assert py.num_free == cc.num_free == NB


def test_cpp_rollback_on_exhaustion():
    """A partially-satisfiable add_seq must roll its blocks back."""
    cc = nc.BlockAllocator(4, 16)
    cc.add_seq(0, 3 * 16)          # 3 of 4 blocks
    with pytest.raises(RuntimeError):
        cc.add_seq(1, 2 * 16)      # needs 2, only 1 free
    assert cc.num_free == 1        # nothing leaked by the failed admit
    cc.free_seq(0)
    assert cc.num_free == 4

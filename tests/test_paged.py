"""Unit tests for the paged KV block allocator."""
import math

import pytest

from server.paged_cache import BlockAllocator, OutOfBlocks


def test_add_seq_block_count():
    a = BlockAllocator(num_blocks=100, block_size=16)
    a.add_seq(0, 40)                       # 40 tokens -> ceil(40/16)=3 blocks
    assert len(a.tables[0]) == 3
    assert a.num_used == 3
    assert a.fill[0] == 40 - 2 * 16        # 8 tokens in the last block


def test_append_allocates_only_on_boundary():
    a = BlockAllocator(num_blocks=100, block_size=4)
    a.add_seq(0, 4)                        # exactly one full block
    assert len(a.tables[0]) == 1 and a.fill[0] == 4
    nb = a.append_token(0)                 # crosses boundary -> new block
    assert nb is not None and len(a.tables[0]) == 2 and a.fill[0] == 1
    nb = a.append_token(0)                 # room in current block -> no alloc
    assert nb is None and a.fill[0] == 2


def test_free_returns_and_reuses_blocks():
    a = BlockAllocator(num_blocks=8, block_size=4)
    a.add_seq(0, 16)                       # uses all 4... wait 16/4=4 blocks
    assert a.num_used == 4
    a.free_seq(0)
    assert a.num_used == 0 and a.num_free == 8
    a.add_seq(1, 8)                        # reuse freed blocks
    assert a.num_used == 2


def test_out_of_blocks_on_add_and_append():
    a = BlockAllocator(num_blocks=2, block_size=4)
    a.add_seq(0, 8)                        # consumes both blocks exactly
    assert a.num_free == 0
    with pytest.raises(OutOfBlocks):
        a.add_seq(1, 1)                    # nothing left
    with pytest.raises(OutOfBlocks):
        a.append_token(0)                  # last block full, none free


def test_internal_fragmentation():
    a = BlockAllocator(num_blocks=100, block_size=16)
    a.add_seq(0, 17)                       # 2 blocks (32 slots), 17 stored -> 15 wasted
    a.add_seq(1, 16)                       # 1 block, 0 wasted
    assert a.internal_frag_tokens() == 15
    # waste is always < block_size per sequence
    assert a.internal_frag_tokens() < a.block_size * 2


def test_can_admit_and_blocks_for():
    a = BlockAllocator(num_blocks=3, block_size=16)
    assert a.blocks_for(1) == 1
    assert a.blocks_for(16) == 1
    assert a.blocks_for(17) == 2
    assert a.can_admit(48) is True        # 3 blocks, exactly fits
    assert a.can_admit(49) is False       # needs 4


def test_slot_mapping():
    a = BlockAllocator(num_blocks=100, block_size=16)
    table = a.add_seq(0, 40)
    assert a.slot(0, 0) == (table[0], 0)
    assert a.slot(0, 16) == (table[1], 0)
    assert a.slot(0, 33) == (table[2], 1)

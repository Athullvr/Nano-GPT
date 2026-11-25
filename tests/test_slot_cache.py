"""
Unit and Integration Tests for SlotAllocatedKVCache (Phase 3C Stage 1)

Covers:
1. Initial allocation and state.
2. Multiple simultaneous allocations.
3. Capacity exhaustion error handling.
4. Independent sequence lengths across slots.
5. Correct key/value writes and reads (single-token and multi-token prefill slices).
6. Multi-layer cache isolation.
7. Rejection of invalid slot IDs (negative, out of bounds, wrong type).
8. Rejection of writes to unallocated slots.
9. Out-of-bounds physical position writes.
10. Release and reuse lifecycle.
11. No stale KV data exposure after slot reuse (verified via valid-prefix slicing).
12. Batched active views and attention mask correctness with heterogeneous lengths.
13. Batched single-token write step and automatic length advancement.
14. Memory size calculation accuracy (bytes, MiB, MB).
"""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.inference.slot_cache import SlotAllocatedKVCache


@pytest.fixture
def tiny_slot_cache() -> SlotAllocatedKVCache:
    """Fixture providing a 4-slot, 2-layer, 4-head cache with context 64 on CPU."""
    return SlotAllocatedKVCache(
        n_layers=2,
        num_slots=4,
        n_heads=4,
        max_seq_len=64,
        head_dim=32,
        device="cpu",
        dtype=torch.float32,
    )


@pytest.fixture
def touchline_slot_cache() -> SlotAllocatedKVCache:
    """Fixture providing a full-dimension TouchlineLM cache (4 layers, 8 slots, 4 heads, 1024 context, dim 64)."""
    return SlotAllocatedKVCache(
        n_layers=4,
        num_slots=8,
        n_heads=4,
        max_seq_len=1024,
        head_dim=64,
        device="cpu",
        dtype=torch.float32,
    )


def test_initial_allocation_state(tiny_slot_cache: SlotAllocatedKVCache):
    """Test initial allocation state, sequence lengths, and free slots."""
    cache = tiny_slot_cache
    assert cache.num_slots == 4
    assert cache.get_active_slots() == []
    assert cache.get_num_free_slots() == 4

    # Allocate first slot
    s0 = cache.allocate()
    assert s0 == 0
    assert cache.is_allocated(0) is True
    assert cache.is_allocated(1) is False
    assert cache.get_sequence_length(0) == 0
    assert cache.get_active_slots() == [0]
    assert cache.get_num_free_slots() == 3


def test_multiple_allocations_and_capacity_exhaustion(tiny_slot_cache: SlotAllocatedKVCache):
    """Test allocating all slots up to capacity and throwing RuntimeError on exhaustion."""
    cache = tiny_slot_cache
    slots = [cache.allocate() for _ in range(4)]
    assert slots == [0, 1, 2, 3]
    assert cache.get_active_slots() == [0, 1, 2, 3]
    assert cache.get_num_free_slots() == 0

    # Next allocation must raise RuntimeError
    with pytest.raises(RuntimeError, match="Slot capacity exhausted"):
        cache.allocate()


def test_invalid_slot_id_and_unallocated_access(tiny_slot_cache: SlotAllocatedKVCache):
    """Test rejection of out-of-bounds, negative, or unallocated slot accesses."""
    cache = tiny_slot_cache
    with pytest.raises(IndexError, match="out of bounds"):
        cache.is_allocated(4)
    with pytest.raises(IndexError, match="out of bounds"):
        cache.is_allocated(-1)

    # Slot 0 is not yet allocated
    with pytest.raises(ValueError, match="not currently allocated"):
        cache.get_sequence_length(0)

    dummy_k = torch.randn(4, 1, 32)
    dummy_v = torch.randn(4, 1, 32)
    with pytest.raises(ValueError, match="not currently allocated"):
        cache.write(slot_id=0, layer_idx=0, position=0, key=dummy_k, value=dummy_v)


def test_independent_sequence_lengths_and_writes(tiny_slot_cache: SlotAllocatedKVCache):
    """Test writing different lengths into different slots and reading them back independently."""
    cache = tiny_slot_cache
    s0 = cache.allocate()
    s1 = cache.allocate()

    # Prefill 5 tokens into slot 0
    k0 = torch.randn(4, 5, 32)
    v0 = torch.randn(4, 5, 32)
    cache.write(slot_id=s0, layer_idx=0, position=0, key=k0, value=v0)
    assert cache.get_sequence_length(s0) == 5

    # Slot 1 must remain 0 length
    assert cache.get_sequence_length(s1) == 0

    # Prefill 12 tokens into slot 1
    k1 = torch.randn(4, 12, 32)
    v1 = torch.randn(4, 12, 32)
    cache.write(slot_id=s1, layer_idx=0, position=0, key=k1, value=v1)
    assert cache.get_sequence_length(s1) == 12

    # Verify reads match exact written values
    k_read0, v_read0 = cache.get_valid_cache(slot_id=s0, layer_idx=0)
    assert k_read0.shape == (1, 4, 5, 32)
    assert torch.equal(k_read0[0], k0)
    assert torch.equal(v_read0[0], v0)

    k_read1, v_read1 = cache.get_valid_cache(slot_id=s1, layer_idx=0)
    assert k_read1.shape == (1, 4, 12, 32)
    assert torch.equal(k_read1[0], k1)
    assert torch.equal(v_read1[0], v1)


def test_multi_layer_isolation(tiny_slot_cache: SlotAllocatedKVCache):
    """Verify that writing to layer 0 does not alter layer 1."""
    cache = tiny_slot_cache
    s = cache.allocate()

    k_l0 = torch.randn(4, 3, 32)
    v_l0 = torch.randn(4, 3, 32)
    k_l1 = torch.randn(4, 3, 32)
    v_l1 = torch.randn(4, 3, 32)

    cache.write(slot_id=s, layer_idx=0, position=0, key=k_l0, value=v_l0)
    cache.write(slot_id=s, layer_idx=1, position=0, key=k_l1, value=v_l1)

    r_k0, r_v0 = cache.get_valid_cache(slot_id=s, layer_idx=0)
    r_k1, r_v1 = cache.get_valid_cache(slot_id=s, layer_idx=1)

    assert torch.equal(r_k0[0], k_l0)
    assert torch.equal(r_k1[0], k_l1)
    assert not torch.equal(r_k0[0], r_k1[0])


def test_out_of_bounds_positions_rejected(tiny_slot_cache: SlotAllocatedKVCache):
    """Writing beyond max_seq_len or negative positions must be rejected."""
    cache = tiny_slot_cache
    s = cache.allocate()

    k_bad = torch.randn(4, 10, 32)
    v_bad = torch.randn(4, 10, 32)

    # max_seq_len is 64; position 60 + 10 = 70 > 64
    with pytest.raises(ValueError, match="exceeds slot capacity"):
        cache.write(slot_id=s, layer_idx=0, position=60, key=k_bad, value=v_bad)

    with pytest.raises(IndexError, match="negative"):
        cache.write(slot_id=s, layer_idx=0, position=-1, key=k_bad, value=v_bad)


def test_slot_release_and_reuse_stale_data_isolation(tiny_slot_cache: SlotAllocatedKVCache):
    """
    Critical correctness test:
    When Slot 0 generates 20 tokens, releases, and is reallocated for a shorter 5-token request,
    the cache read must strictly expose only the 5 new tokens. Stale tokens 5..19 must never leak.
    """
    cache = tiny_slot_cache
    s0 = cache.allocate()
    assert s0 == 0

    # Request A writes 20 tokens
    k_reqA = torch.randn(4, 20, 32)
    v_reqA = torch.randn(4, 20, 32)
    cache.write(slot_id=s0, layer_idx=0, position=0, key=k_reqA, value=v_reqA)
    assert cache.get_sequence_length(s0) == 20

    # Release slot 0
    cache.release(s0)
    assert cache.is_allocated(0) is False
    assert cache.get_num_free_slots() == 4

    # Reallocate slot 0 for Request B
    s0_reused = cache.allocate()
    assert s0_reused == 0
    assert cache.get_sequence_length(s0_reused) == 0

    # An immediate read on freshly allocated slot returns 0 length
    k_empty, v_empty = cache.get_valid_cache(slot_id=s0_reused, layer_idx=0)
    assert k_empty.shape == (1, 4, 0, 32)

    # Request B writes 5 tokens
    k_reqB = torch.randn(4, 5, 32)
    v_reqB = torch.randn(4, 5, 32)
    cache.write(slot_id=s0_reused, layer_idx=0, position=0, key=k_reqB, value=v_reqB)
    assert cache.get_sequence_length(s0_reused) == 5

    # Valid read must expose exactly 5 tokens, perfectly matching Request B
    k_read, v_read = cache.get_valid_cache(slot_id=s0_reused, layer_idx=0)
    assert k_read.shape == (1, 4, 5, 32)
    assert torch.equal(k_read[0], k_reqB)
    assert torch.equal(v_read[0], v_reqB)

    # Underlying physical memory at indices 5..19 still contains Request A's vectors,
    # but get_valid_cache strictly isolates Request B from them!
    assert torch.equal(cache.k_cache[0][0, :, 5:20, :], k_reqA[:, 5:20, :])


def test_batched_active_views_and_attention_mask(tiny_slot_cache: SlotAllocatedKVCache):
    """
    Test batched slicing across active slots with heterogeneous lengths.
    Verifies that unwritten / stale positions are masked out with False in attn_mask.
    """
    cache = tiny_slot_cache
    s0 = cache.allocate()  # Will hold 3 tokens
    s1 = cache.allocate()  # Will hold 7 tokens
    s2 = cache.allocate()  # Will hold 5 tokens

    k0, v0 = torch.randn(4, 3, 32), torch.randn(4, 3, 32)
    k1, v1 = torch.randn(4, 7, 32), torch.randn(4, 7, 32)
    k2, v2 = torch.randn(4, 5, 32), torch.randn(4, 5, 32)

    cache.write(slot_id=s0, layer_idx=0, position=0, key=k0, value=v0)
    cache.write(slot_id=s1, layer_idx=0, position=0, key=k1, value=v1)
    cache.write(slot_id=s2, layer_idx=0, position=0, key=k2, value=v2)

    active_slots = [s0, s1, s2]
    k_b, v_b, mask_b = cache.get_batched_active_views(active_slots=active_slots, layer_idx=0)

    # T_max_active = max(3, 7, 5) = 7
    assert k_b.shape == (3, 4, 7, 32)
    assert v_b.shape == (3, 4, 7, 32)
    assert mask_b.shape == (3, 1, 1, 7)

    # Row 0 (slot 0, length 3): tokens 0..2 are True, 3..6 are False
    assert mask_b[0, 0, 0, :3].all().item() is True
    assert (~mask_b[0, 0, 0, 3:]).all().item() is True
    assert torch.equal(k_b[0, :, :3, :], k0)

    # Row 1 (slot 1, length 7): all 7 tokens are True
    assert mask_b[1, 0, 0, :].all().item() is True
    assert torch.equal(k_b[1, :, :7, :], k1)

    # Row 2 (slot 2, length 5): tokens 0..4 are True, 5..6 are False
    assert mask_b[2, 0, 0, :5].all().item() is True
    assert (~mask_b[2, 0, 0, 5:]).all().item() is True
    assert torch.equal(k_b[2, :, :5, :], k2)


def test_write_batched_step_automatic_advance(tiny_slot_cache: SlotAllocatedKVCache):
    """
    Test write_batched_step during a single decode iteration.
    Writes new step token into each active slot at pos L_i and advances valid length.
    """
    cache = tiny_slot_cache
    s0 = cache.allocate()
    s1 = cache.allocate()

    # Prefill: slot 0 has 2 tokens, slot 1 has 4 tokens
    cache.write(slot_id=s0, layer_idx=0, position=0, key=torch.randn(4, 2, 32), value=torch.randn(4, 2, 32))
    cache.write(slot_id=s1, layer_idx=0, position=0, key=torch.randn(4, 4, 32), value=torch.randn(4, 4, 32))

    # Single-token step input [B_act=2, n_heads=4, 1, head_dim=32]
    k_step = torch.randn(2, 4, 1, 32)
    v_step = torch.randn(2, 4, 1, 32)

    k_b, v_b, mask = cache.write_batched_step(
        active_slots=[s0, s1],
        layer_idx=0,
        k_step=k_step,
        v_step=v_step,
    )

    # Slot 0 length advanced to 3, slot 1 advanced to 5
    assert cache.get_sequence_length(s0) == 3
    assert cache.get_sequence_length(s1) == 5

    # T_max_active = max(3, 5) = 5
    assert k_b.shape == (2, 4, 5, 32)
    assert mask.shape == (2, 1, 1, 5)

    # Newly written token at pos L_i matches step input
    assert torch.equal(k_b[0, :, 2:3, :], k_step[0])
    assert torch.equal(k_b[1, :, 4:5, :], k_step[1])

    # Mask reflects newly written positions
    assert mask[0, 0, 0, :3].all().item() is True
    assert (~mask[0, 0, 0, 3:]).all().item() is True
    assert mask[1, 0, 0, :5].all().item() is True


def test_memory_accounting_touchline_dimensions(touchline_slot_cache: SlotAllocatedKVCache):
    """
    Test memory calculation matching the Phase 3C specification:
    4 layers, 8 slots, 4 heads, 1024 context, 64 head_dim, FP32.
    Expected: exactly 64 MiB (67.11 MB).
    """
    cache = touchline_slot_cache
    expected_bytes = 2 * 4 * 8 * 4 * 1024 * 64 * 4  # 67,108,864 bytes
    assert cache.memory_bytes() == expected_bytes
    assert cache.memory_mib() == 64.0
    assert abs(cache.memory_mb() - 67.108864) < 1e-5


def test_from_model_constructor():
    """Test from_model factory constructor with real TouchlineLM config."""
    config = ModelConfig(
        vocab_size=4096,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=1024,
    )
    model = TouchlineNanoGPT(config).eval()
    cache = SlotAllocatedKVCache.from_model(model, num_slots=8, device="cpu")

    assert cache.n_layers == 4
    assert cache.num_slots == 8
    assert cache.n_heads == 4
    assert cache.max_seq_len == 1024
    assert cache.head_dim == 64
    assert cache.memory_mib() == 64.0

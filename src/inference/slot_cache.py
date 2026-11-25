"""
TouchlineLM Slot-Allocated KV Cache for Continuous Batching (Phase 3C Stage 1)

Provides a fixed-slot preallocated Key-Value cache manager:
- Preallocates contiguous buffers for a maximum number of concurrent request slots (num_slots).
- Tracks allocation state, valid sequence lengths, and metadata independently per slot.
- Exposes single-slot and batched write interfaces at physical sequence positions.
- Exposes valid-prefix slicing (get_valid_cache) and batched slice views with attention masks.
- Enforces strict validation: rejects invalid slot IDs, unallocated slots, and out-of-bounds positions.
- Guarantees isolation on slot reuse via explicit sequence-length metadata and valid-prefix slicing.
"""

from typing import List, Tuple, Optional, Union, Dict
import torch
import torch.nn as nn

from src.model.config import ModelConfig


class SlotAllocatedKVCache:
    """
    Fixed-slot preallocated Key-Value cache for continuous batching.

    Storage layout:
        k_cache: List of n_layers tensors, each [num_slots, n_heads, max_seq_len, head_dim]
        v_cache: List of n_layers tensors, each [num_slots, n_heads, max_seq_len, head_dim]

    Each slot maintains:
        - allocated: bool indicating whether the slot is currently active.
        - seq_len: int representing the count of valid cached tokens (0 initially).
    """

    def __init__(
        self,
        n_layers: int,
        num_slots: int,
        n_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.n_layers = n_layers
        self.num_slots = num_slots
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        # Preallocated contiguous buffers per layer:
        # [num_slots, n_heads, max_seq_len, head_dim]
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(num_slots, n_heads, max_seq_len, head_dim, device=self.device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(num_slots, n_heads, max_seq_len, head_dim, device=self.device, dtype=dtype)
            for _ in range(n_layers)
        ]

        # Per-slot allocation metadata
        self._allocated: List[bool] = [False] * num_slots
        self._seq_lengths: List[int] = [0] * num_slots

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        num_slots: int = 8,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SlotAllocatedKVCache":
        """Convenience factory constructing cache sized to a TouchlineNanoGPT model."""
        config: ModelConfig = getattr(model, "config")
        target_device = device if device is not None else next(model.parameters()).device
        return cls(
            n_layers=config.n_layers,
            num_slots=num_slots,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            device=target_device,
            dtype=dtype,
        )

    # -------------------------------------------------------------------------
    # Slot Lifecycle Management
    # -------------------------------------------------------------------------

    def allocate(self) -> int:
        """
        Allocates the first available free slot.

        Returns:
            slot_id (int) in [0, num_slots - 1].

        Raises:
            RuntimeError: If all slots are currently allocated (capacity exhausted).
        """
        for slot_id in range(self.num_slots):
            if not self._allocated[slot_id]:
                self._allocated[slot_id] = True
                self._seq_lengths[slot_id] = 0
                return slot_id
        raise RuntimeError(
            f"Slot capacity exhausted: all {self.num_slots} slots are currently allocated."
        )

    def release(self, slot_id: int) -> None:
        """
        Releases an allocated slot, making it available for subsequent reuse.
        Resets valid sequence length to zero.

        Args:
            slot_id: The slot index to release.

        Raises:
            IndexError: If slot_id is out of range.
            ValueError: If slot_id is not currently allocated.
        """
        self._validate_slot_id(slot_id)
        if not self._allocated[slot_id]:
            raise ValueError(f"Slot {slot_id} is not currently allocated.")
        self._allocated[slot_id] = False
        self._seq_lengths[slot_id] = 0

    def is_allocated(self, slot_id: int) -> bool:
        """Returns True if the given slot_id is currently allocated."""
        self._validate_slot_id(slot_id)
        return self._allocated[slot_id]

    def get_active_slots(self) -> List[int]:
        """Returns a list of currently allocated slot IDs."""
        return [i for i, alloc in enumerate(self._allocated) if alloc]

    def get_num_free_slots(self) -> int:
        """Returns the number of unallocated slots available."""
        return sum(1 for alloc in self._allocated if not alloc)

    def get_sequence_length(self, slot_id: int) -> int:
        """
        Returns the number of valid tokens stored in the specified slot.

        Args:
            slot_id: The slot index.

        Raises:
            IndexError: If slot_id is out of range.
            ValueError: If slot_id is not allocated.
        """
        self._validate_slot_allocated(slot_id)
        return self._seq_lengths[slot_id]

    # -------------------------------------------------------------------------
    # Single-Slot Write and Read Operations
    # -------------------------------------------------------------------------

    def write(
        self,
        slot_id: int,
        layer_idx: int,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Writes key and value tokens for a single slot at physical sequence index `position`.

        Supports writing either a single token (T_new = 1) or a contiguous slice of tokens
        (e.g., prompt prefill of length P).

        Args:
            slot_id: Target allocated slot index.
            layer_idx: Target transformer layer index in [0, n_layers - 1].
            position: Starting physical sequence index in [0, max_seq_len - 1].
            key: Key tensor shaped [n_heads, T_new, head_dim] or [1, n_heads, T_new, head_dim].
            value: Value tensor shaped [n_heads, T_new, head_dim] or [1, n_heads, T_new, head_dim].

        Raises:
            IndexError: If slot_id, layer_idx, or position is invalid.
            ValueError: If slot_id is unallocated or position + T_new > max_seq_len.
        """
        self._validate_slot_allocated(slot_id)
        self._validate_layer_idx(layer_idx)

        # Normalize 4D [1, n_heads, T_new, head_dim] to 3D [n_heads, T_new, head_dim]
        if key.dim() == 4:
            if key.size(0) != 1:
                raise ValueError(f"Expected batch dimension 1 for single-slot write, got {key.size(0)}")
            key = key.squeeze(0)
            value = value.squeeze(0)

        if key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                f"Expected key/value tensors of shape [n_heads, T_new, head_dim], "
                f"got key: {list(key.shape)}, value: {list(value.shape)}"
            )

        n_heads, T_new, head_dim = key.shape
        if n_heads != self.n_heads or head_dim != self.head_dim:
            raise ValueError(
                f"Shape mismatch: expected (n_heads={self.n_heads}, head_dim={self.head_dim}), "
                f"got ({n_heads}, {head_dim})"
            )

        if position < 0:
            raise IndexError(f"Position cannot be negative, got {position}")
        if position + T_new > self.max_seq_len:
            raise ValueError(
                f"Write exceeds slot capacity: position {position} + tokens {T_new} = "
                f"{position + T_new} > max_seq_len {self.max_seq_len}"
            )

        # Physical in-place write into preallocated tensor:
        # self.k_cache[layer_idx]: [num_slots, n_heads, max_seq_len, head_dim]
        self.k_cache[layer_idx][slot_id, :, position : position + T_new, :].copy_(key)
        self.v_cache[layer_idx][slot_id, :, position : position + T_new, :].copy_(value)

        # Update valid sequence length if this write extends beyond the previous length
        new_valid_len = position + T_new
        if new_valid_len > self._seq_lengths[slot_id]:
            self._seq_lengths[slot_id] = new_valid_len

    def get_valid_cache(
        self,
        slot_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sliced view of valid key and value tensors for a single allocated slot.
        The slice strictly spans physical indices [0, valid_seq_len).

        Guarantees that unwritten positions and stale past data beyond valid_seq_len
        are never exposed to the caller.

        Args:
            slot_id: Allocated slot index.
            layer_idx: Transformer layer index.

        Returns:
            Tuple of (k_valid, v_valid) each shaped [1, n_heads, valid_seq_len, head_dim].
        """
        self._validate_slot_allocated(slot_id)
        self._validate_layer_idx(layer_idx)

        valid_len = self._seq_lengths[slot_id]
        if valid_len == 0:
            # Return empty slice with 0 sequence dimension
            k_empty = self.k_cache[layer_idx][slot_id : slot_id + 1, :, :0, :]
            v_empty = self.v_cache[layer_idx][slot_id : slot_id + 1, :, :0, :]
            return k_empty, v_empty

        # Return sliced view [1, n_heads, valid_len, head_dim]
        k_view = self.k_cache[layer_idx][slot_id : slot_id + 1, :, :valid_len, :]
        v_view = self.v_cache[layer_idx][slot_id : slot_id + 1, :, :valid_len, :]
        return k_view, v_view

    # -------------------------------------------------------------------------
    # Batched Active Slicing and Writing (Continuous Batching Decode Interface)
    # -------------------------------------------------------------------------

    def get_batched_active_views(
        self,
        active_slots: List[int],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds batched key, value, and attention mask tensors across multiple active slots
        having heterogeneous sequence lengths.

        Args:
            active_slots: List of currently allocated slot IDs [B_act].
            layer_idx: Transformer layer index.

        Returns:
            k_batch: [B_act, n_heads, T_max_active, head_dim]
            v_batch: [B_act, n_heads, T_max_active, head_dim]
            attn_mask: [B_act, 1, 1, T_max_active] boolean mask (True = valid token, False = padding/stale)
        """
        if not active_slots:
            raise ValueError("active_slots list cannot be empty")
        for s in active_slots:
            self._validate_slot_allocated(s)
        self._validate_layer_idx(layer_idx)

        B_act = len(active_slots)
        lengths = [self._seq_lengths[s] for s in active_slots]
        T_max_active = max(lengths)

        if T_max_active == 0:
            # All active slots are empty
            k_batch = torch.empty((B_act, self.n_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
            v_batch = torch.empty((B_act, self.n_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
            attn_mask = torch.empty((B_act, 1, 1, 0), device=self.device, dtype=torch.bool)
            return k_batch, v_batch, attn_mask

        # Index select active slots and slice up to T_max_active
        slot_tensor = torch.tensor(active_slots, dtype=torch.long, device=self.device)
        k_batch = torch.index_select(self.k_cache[layer_idx], 0, slot_tensor)[:, :, :T_max_active, :]
        v_batch = torch.index_select(self.v_cache[layer_idx], 0, slot_tensor)[:, :, :T_max_active, :]

        # Construct boolean attention mask [B_act, 1, 1, T_max_active]
        # True for positions 0 <= j < lengths[b], False for j >= lengths[b]
        attn_mask = torch.zeros((B_act, 1, 1, T_max_active), dtype=torch.bool, device=self.device)
        for b, seq_len in enumerate(lengths):
            if seq_len > 0:
                attn_mask[b, 0, 0, :seq_len] = True

        return k_batch, v_batch, attn_mask

    def write_batched_step(
        self,
        active_slots: List[int],
        layer_idx: int,
        k_step: torch.Tensor,
        v_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Writes newly projected single-token key and value tensors for all active slots
        at their respective physical indices L_i, increments valid lengths L_i <- L_i + 1,
        and returns the updated batched views and boolean attention mask.

        Args:
            active_slots: List of active slot IDs of length B_act.
            layer_idx: Transformer layer index.
            k_step: [B_act, n_heads, 1, head_dim] newly projected keys.
            v_step: [B_act, n_heads, 1, head_dim] newly projected values.

        Returns:
            Tuple of (k_batch, v_batch, attn_mask):
                k_batch: [B_act, n_heads, T_max_active, head_dim]
                v_batch: [B_act, n_heads, T_max_active, head_dim]
                attn_mask: [B_act, 1, 1, T_max_active] boolean mask (True for 0 <= j <= L_i)
        """
        B_act = len(active_slots)
        if k_step.shape != (B_act, self.n_heads, 1, self.head_dim):
            raise ValueError(
                f"Expected k_step shape ({B_act}, {self.n_heads}, 1, {self.head_dim}), got {list(k_step.shape)}"
            )

        # Write each slot at its current length L_i
        for b, slot_id in enumerate(active_slots):
            pos = self._seq_lengths[slot_id]
            self.write(
                slot_id=slot_id,
                layer_idx=layer_idx,
                position=pos,
                key=k_step[b : b + 1],
                value=v_step[b : b + 1],
            )

        # Return updated views including the newly written token
        return self.get_batched_active_views(active_slots, layer_idx)

    # -------------------------------------------------------------------------
    # Telemetry and Sizing
    # -------------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Returns total preallocated memory in bytes across all layers."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return (
            2
            * self.n_layers
            * self.num_slots
            * self.n_heads
            * self.max_seq_len
            * self.head_dim
            * element_size
        )

    def memory_mib(self) -> float:
        """Returns total preallocated memory in mebibytes (MiB)."""
        return self.memory_bytes() / (1024 ** 2)

    def memory_mb(self) -> float:
        """Returns total preallocated memory in decimal megabytes (MB)."""
        return self.memory_bytes() / (1000 ** 2)

    # -------------------------------------------------------------------------
    # Private Validation Helpers
    # -------------------------------------------------------------------------

    def _validate_slot_id(self, slot_id: int) -> None:
        if not isinstance(slot_id, int):
            raise TypeError(f"slot_id must be an int, got {type(slot_id)}")
        if slot_id < 0 or slot_id >= self.num_slots:
            raise IndexError(f"slot_id {slot_id} out of bounds for cache with {self.num_slots} slots")

    def _validate_slot_allocated(self, slot_id: int) -> None:
        self._validate_slot_id(slot_id)
        if not self._allocated[slot_id]:
            raise ValueError(f"Slot {slot_id} is not currently allocated")

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if not isinstance(layer_idx, int):
            raise TypeError(f"layer_idx must be an int, got {type(layer_idx)}")
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of bounds for {self.n_layers} layers")

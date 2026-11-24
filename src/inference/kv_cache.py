"""
TouchlineLM KV Cache Management

Provides:
1. DynamicKVCache: Standard list-of-tuples dynamic cache using torch.cat (preserves full backward compatibility).
2. StaticKVCache: Preallocated, zero-allocation cache with in-place narrow().copy_() updates for bounded memory and zero fragmentation.
"""

from typing import List, Tuple, Union, Optional
import torch
import torch.nn as nn

from src.model.config import ModelConfig


class StaticLayerCache:
    """
    Lightweight layer-level view into a parent StaticKVCache.
    """

    def __init__(self, parent: "StaticKVCache", layer_idx: int) -> None:
        self.parent = parent
        self.layer_idx = layer_idx

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the preallocated buffer in-place for this layer and returns
        the active slice [B, n_heads, current_len + T_new, head_dim].
        """
        return self.parent.update(self.layer_idx, k, v)


class StaticKVCache:
    """
    Preallocated, zero-allocation Key-Value cache for TouchlineLM.

    Preallocates maximum context length tensors:
        k_cache: [B, n_heads, max_seq_len, head_dim]
        v_cache: [B, n_heads, max_seq_len, head_dim]
    and writes incoming tokens in-place using narrow().copy_().

    Eliminates dynamic reallocations, memory fragmentation, and copying
    overhead of past tokens during autoregressive decoding.
    """

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Preallocated contiguous buffers
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(batch_size, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(batch_size, n_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.current_seq_len = 0

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        batch_size: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "StaticKVCache":
        """Convenience constructor from a TouchlineNanoGPT model instance."""
        config: ModelConfig = model.config
        target_device = device if device is not None else next(model.parameters()).device
        return cls(
            n_layers=config.n_layers,
            batch_size=batch_size,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            device=target_device,
            dtype=dtype,
        )

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Writes newly projected key and value tokens in-place into the preallocated buffer.
        Returns contiguous views up to the active sequence length.
        """
        T_new = k.shape[2]
        pos = self.current_seq_len
        if pos + T_new > self.max_seq_len:
            raise ValueError(
                f"Sequence length {pos + T_new} (past {pos} + new {T_new}) "
                f"exceeds preallocated cache capacity {self.max_seq_len}"
            )

        self.k_cache[layer_idx].narrow(2, pos, T_new).copy_(k)
        self.v_cache[layer_idx].narrow(2, pos, T_new).copy_(v)

        # Return view of all tokens up to current position + new tokens
        return (
            self.k_cache[layer_idx].narrow(2, 0, pos + T_new),
            self.v_cache[layer_idx].narrow(2, 0, pos + T_new),
        )

    def advance(self, T_new: int) -> None:
        """Advances the internal sequence counter after all layers complete a forward pass."""
        self.current_seq_len += T_new

    def reset(self) -> None:
        """Resets the sequence pointer for a new generation run without reallocating buffers."""
        self.current_seq_len = 0

    def get_seq_length(self) -> int:
        """Returns the current number of cached tokens."""
        return self.current_seq_len

    def memory_bytes(self) -> int:
        """Calculates total preallocated memory in bytes."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.n_layers * self.batch_size * self.n_heads * self.max_seq_len * self.head_dim * element_size

    def memory_mb(self) -> float:
        """Calculates total preallocated memory in Megabytes (MB)."""
        return self.memory_bytes() / (1024 ** 2)

    def __getitem__(self, idx: int) -> StaticLayerCache:
        """Returns a layer-specific proxy for block-level consumption."""
        if idx < 0 or idx >= self.n_layers:
            raise IndexError(f"Layer index {idx} out of bounds for {self.n_layers} layers")
        return StaticLayerCache(self, idx)

    def __len__(self) -> int:
        return self.n_layers

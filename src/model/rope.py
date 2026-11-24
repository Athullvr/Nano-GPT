"""
Rotary Position Embedding (RoPE) implementation for TouchlineLM.

RoPE (Su et al., 2021 / 2024, "RoFormer: Enhanced Transformer with Rotary Position Embedding")
encodes relative positional information directly into query and key representations via
complex-plane rotations.

Mathematical Formulation:
For an attention vector x in R^d (where d = head_dim is even):
1. Frequencies: theta_i = base^(-2i/d) for i in [0, d/2 - 1]
2. Positional angles: m * theta_i for position m in [0, max_seq_len - 1]
3. Rotation is applied to coordinate pairs (x_{2i}, x_{2i+1}) or split halves (x_{:d/2}, x_{d/2:}):
   R(x, m) = x * cos(m * theta) + rotate_half(x) * sin(m * theta)
   where rotate_half(x) = [-x_{d/2:}, x_{:d/2}]

Properties:
- <R(q, m), R(k, n)> = q^T R_{n-m} k (strictly depends on relative distance m - n).
- Decays naturally with relative distance |m - n|.
- Preserves vector norm: ||R(x, m)||^2 = ||x||^2.
"""

from typing import Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor.

    Transforms [x_1, x_2] to [-x_2, x_1] across the final dimension.

    Args:
        x: Tensor with shape [..., head_dim]

    Returns:
        Tensor with rotated halves: [-x[..., head_dim // 2:], x[..., :head_dim // 2]]
    """
    dim = x.shape[-1]
    half_dim = dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor shaped [batch_size, n_heads, seq_len, head_dim]
        k: Key tensor shaped [batch_size, n_heads, seq_len, head_dim]
        cos: Cosine tensor shaped [1, 1, seq_len, head_dim] or [seq_len, head_dim]
        sin: Sine tensor shaped [1, 1, seq_len, head_dim] or [seq_len, head_dim]

    Returns:
        Rotated (q_rot, k_rot) with matching shapes.
    """
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    if sin.ndim == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)

    seq_len = q.shape[2]
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(nn.Module):
    """
    Precomputes and caches rotary position embeddings up to max_seq_len.

    Parameters:
        dim: Dimensionality of each attention head (head_dim, must be even).
        max_seq_len: Maximum context window length (default: 1024).
        base: Base frequency for geometric progression (default: 10000.0).
    """

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies: theta_i = base^(-2i/dim) for i in [0, dim/2 - 1]
        half_dim = dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin cached tables up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        """Precomputes cos and sin tables up to max_seq_len."""
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: [max_seq_len, half_dim]
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate frequencies to cover full head_dim: [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        # Register as buffers (not trainable parameters)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves cos and sin tensors sliced for the given sequence length.

        Args:
            x: Input tensor (used to match device and dtype)
            seq_len: Number of tokens in sequence (seq_len <= max_seq_len)

        Returns:
            cos: Tensor [1, 1, seq_len, dim]
            sin: Tensor [1, 1, seq_len, dim]
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Requested seq_len ({seq_len}) exceeds precomputed max_seq_len ({self.max_seq_len})"
            )
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
        return cos, sin

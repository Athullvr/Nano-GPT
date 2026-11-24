"""
Multi-Head Causal Self-Attention module for TouchlineLM.

Implements causal multi-head self-attention using PyTorch Scaled Dot-Product Attention (SDPA).
Integrates Rotary Position Embeddings (RoPE) applied to Query and Key projections,
enforcing autoregressive masking so position i can attend only to positions <= i.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.rope import apply_rotary_emb


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE positional encoding and PyTorch SDPA.

    Parameters:
        config: ModelConfig containing d_model, n_heads, head_dim, dropout, bias.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # Independent linear projections for Query, Key, Value
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Output projection back to residual stream
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for causal multi-head self-attention with optional per-layer KV caching.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            cos: RoPE cosine tensor [1, 1, seq_len, head_dim]
            sin: RoPE sine tensor [1, 1, seq_len, head_dim]
            attn_mask: Optional attention mask (if None, causality determined by seq_len)
            layer_past: Optional tuple (k_past, v_past) each shaped [batch_size, n_heads, past_len, head_dim]
            use_cache: If True, returns (output, present_kv_tuple)

        Returns:
            Output tensor [batch_size, seq_len, d_model] if use_cache=False,
            else tuple (output, (k_present, v_present))
        """
        B, T, C = x.shape

        # Step 1: Linear projections [B, T, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Reshape to multi-head format: [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 3: Apply Rotary Position Embeddings (RoPE) to newly projected Q and K
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Step 4: KV Cache concatenation (keys are stored already-rotated)
        if layer_past is not None:
            k_past, v_past = layer_past
            k = torch.cat((k_past, k), dim=2)
            v = torch.cat((v_past, v), dim=2)

        present = (k, v) if use_cache else None

        # Step 5: Scaled Dot-Product Attention
        dropout_p = self.dropout_p if self.training else 0.0
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            )
        elif T == 1:
            # Single-token decoding: Query has length 1 at position m and attends
            # to all past and present positions (0 .. m). Non-causal attention is applied.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False
            )
        else:
            # Full sequence prefill or uncached sequence: Query position i attends only to positions j <= i
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )

        # Step 6: Recombine heads: [B, n_heads, T, head_dim] -> [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Step 7: Output projection & residual dropout
        out = self.resid_dropout(self.out_proj(out))

        if use_cache:
            return out, present
        return out

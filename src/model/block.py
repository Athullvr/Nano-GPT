"""
Transformer Block module for TouchlineLM.

Implements a Pre-LayerNorm Transformer decoder block:
1. x = x + SelfAttention(LayerNorm_1(x))
2. x = x + FeedForward(LayerNorm_2(x))

Pre-LayerNorm ensures an unobstructed residual path across layers, improving
gradient propagation and numerical stability during deep Transformer training.
"""

from typing import Optional
import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.config import ModelConfig
from src.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    A single decoder-only Transformer block with Pre-LayerNorm.

    Parameters:
        config: ModelConfig instance defining layer dimensions and hyperparameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            cos: RoPE cosine tensor [1, 1, seq_len, head_dim]
            sin: RoPE sine tensor [1, 1, seq_len, head_dim]
            attn_mask: Optional attention mask tensor

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-LN -> Attention -> Residual connection
        x = x + self.attn(self.ln_1(x), cos=cos, sin=sin, attn_mask=attn_mask)
        # Pre-LN -> MLP -> Residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

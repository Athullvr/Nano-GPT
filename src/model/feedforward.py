"""
Feed-Forward Network (MLP) module for TouchlineLM.

Two-layer position-wise feed-forward network with GELU non-linearity:
FFN(x) = Dropout(W_2 * GELU(W_1 * x + b_1) + b_2)
"""

import torch
import torch.nn as nn

from src.model.config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Parameters:
        config: ModelConfig specifying d_model, d_ff, dropout, bias.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Linear expansion: d_model -> d_ff
        h = self.fc1(x)
        # Non-linear activation
        h = self.act(h)
        # Linear projection: d_ff -> d_model
        h = self.fc2(h)
        # Regularization dropout
        return self.dropout(h)

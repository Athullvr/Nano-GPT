"""
Token Embeddings module for TouchlineLM.

Maps discrete token indices to continuous dense representations of dimension d_model.
Note: Positional embeddings are handled dynamically by Rotary Position Embeddings (RoPE)
in the attention mechanism, so absolute positional embeddings are not added here.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Embedding layer mapping discrete vocabulary token IDs to continuous vectors.

    Parameters:
        vocab_size: Size of the vocabulary (default: 4096).
        d_model: Dimensionality of the embedding space (default: 256).
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Embed input token indices.

        Args:
            idx: Tensor of token indices with shape [batch_size, sequence_length].

        Returns:
            Tensor of embeddings with shape [batch_size, sequence_length, d_model].
        """
        return self.embedding(idx)

    @property
    def weight(self) -> nn.Parameter:
        """Direct access to the underlying embedding weight tensor."""
        return self.embedding.weight

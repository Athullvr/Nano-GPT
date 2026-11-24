"""
TouchlineLM Model Package.

Exports the core decoder-only Transformer architecture and configuration:
- ModelConfig: Architectural hyperparameters.
- TouchlineNanoGPT: Complete educational Transformer.
- TokenEmbedding: Embedding module.
- RotaryEmbedding: Rotary Position Embeddings (RoPE).
- CausalSelfAttention: Multi-head attention with causal mask.
- FeedForward: Two-layer MLP with GELU.
- TransformerBlock: Pre-LayerNorm Transformer block.
"""

from src.model.config import ModelConfig
from src.model.embeddings import TokenEmbedding
from src.model.rope import RotaryEmbedding, apply_rotary_emb, rotate_half
from src.model.attention import CausalSelfAttention
from src.model.feedforward import FeedForward
from src.model.block import TransformerBlock
from src.model.model import TouchlineNanoGPT

__all__ = [
    "ModelConfig",
    "TokenEmbedding",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "rotate_half",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "TouchlineNanoGPT",
]

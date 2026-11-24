"""
TouchlineNanoGPT: Decoder-Only Transformer for Football Intelligence.

Complete model implementation integrating:
- Token Embeddings (with optional tied LM Head projection weights)
- Rotary Position Embeddings (RoPE)
- Pre-LayerNorm Transformer Blocks
- PyTorch Scaled Dot-Product Attention (SDPA) with causal masking
- Final LayerNorm and Next-Token Prediction Head
- Configurable loss computation and parameter counting
"""

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.block import TransformerBlock
from src.model.config import ModelConfig
from src.model.embeddings import TokenEmbedding
from src.model.rope import RotaryEmbedding


class TouchlineNanoGPT(nn.Module):
    """
    TouchlineNanoGPT decoder-only Transformer model.

    Parameters:
        config: ModelConfig specifying architectural parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding lookup
        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)

        # Rotary position embeddings (cached up to max_seq_len)
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        # Input dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer decoder blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final Pre-LayerNorm layer before LM head
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Language Model Head (token projection)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embedding and output projection
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following standard Transformer practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Forward pass for next-token prediction with optional per-layer KV caching.

        Args:
            idx: Tensor of token IDs [batch_size, seq_len]
            targets: Optional ground-truth token IDs [batch_size, seq_len] for cross-entropy loss
            past_key_values: Optional list of (k_past, v_past) per layer
            use_cache: If True, returns KV cache for future incremental decode steps

        Returns:
            If use_cache is False: (logits, loss)
            If use_cache is True and targets is None: (logits, presents)
            If use_cache is True and targets is not None: (logits, loss, presents)

        Raises:
            ValueError: If total sequence length exceeds configured max_seq_len.
        """
        B, T = idx.shape

        # Determine past cached length if past_key_values is provided
        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

        total_length = past_length + T
        # Enforce context length boundary
        if total_length > self.config.max_seq_len:
            raise ValueError(
                f"Total sequence length {total_length} (past {past_length} + current {T}) "
                f"exceeds maximum configured context length {self.config.max_seq_len}"
            )

        # Token embedding lookup: [B, T] -> [B, T, d_model]
        x = self.token_embedding(idx)
        x = self.dropout(x)

        # Compute RoPE cos, sin tensors for length T with position offset: [1, 1, T, head_dim]
        cos, sin = self.rope(x, seq_len=T, offset=past_length)

        # Pass through Transformer blocks
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, present = block(x, cos=cos, sin=sin, layer_past=layer_past, use_cache=True)
                if presents is not None:
                    presents.append(present)
            else:
                x = block(x, cos=cos, sin=sin)

        # Advance preallocated static cache sequence counter if applicable
        if use_cache and hasattr(past_key_values, "advance"):
            past_key_values.advance(T)
            presents = past_key_values

        # Final LayerNorm
        x = self.ln_f(x)

        # Project to vocabulary logits: [B, T, vocab_size]
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten B and T dimensions for cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        if use_cache:
            if targets is not None:
                return logits, loss, presents
            return logits, presents
        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Calculates the total number of unique trainable parameters.
        Correctly accounts for weight tying so tied weights are not counted twice.

        Args:
            non_embedding: If True, excludes token embedding parameters.

        Returns:
            Total count of unique trainable parameters.
        """
        unique_params = set()
        for p in self.parameters():
            if p.requires_grad:
                unique_params.add(p)

        if non_embedding:
            unique_params.discard(self.token_embedding.embedding.weight)

        return sum(p.numel() for p in unique_params)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive generation for inference and inspection.

        Args:
            idx: Conditioning prefix token IDs [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = standard, <1.0 = more confident)
            top_k: Optional top-k filtering threshold
            use_cache: If True, uses per-layer KV cache for single-token incremental decode

        Returns:
            idx: Tensor of generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        if max_new_tokens <= 0:
            return idx

        if not use_cache:
            for _ in range(max_new_tokens):
                # Crop context if it exceeds max_seq_len
                idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
                logits, _ = self(idx_cond)
                # Take logits at the final position
                logits = logits[:, -1, :] / max(temperature, 1e-5)

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)

            return idx

        # KV cached generation path
        # 1. Prefill step
        idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
        logits, past_key_values = self(idx_cond, use_cache=True)
        logits_step = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
            logits_step[logits_step < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits_step, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # 2. Incremental single-token decode steps
        for _ in range(max_new_tokens - 1):
            if idx.size(1) >= self.config.max_seq_len:
                break
            logits, past_key_values = self(idx_next, past_key_values=past_key_values, use_cache=True)
            logits_step = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits_step, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

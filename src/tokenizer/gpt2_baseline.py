"""
TouchlineLM - GPT-2 Baseline Tokenizer Wrapper
Provides encoding, decoding, token counting, and special token preservation
using tiktoken's GPT-2 BPE model as a benchmark.
"""

import os
from pathlib import Path
from typing import List, Optional

# Ensure offline cache directory is set before loading tiktoken
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".tiktoken_cache"
if "TIKTOKEN_CACHE_DIR" not in os.environ and DEFAULT_CACHE_DIR.exists():
    os.environ["TIKTOKEN_CACHE_DIR"] = str(DEFAULT_CACHE_DIR)

import tiktoken


class GPT2BaselineTokenizer:
    """Baseline tokenizer utilizing tiktoken GPT-2 BPE encoding."""

    SPECIAL_TOKENS = {
        "<|endoftext|>": 50256,
        "<|narrative|>": 50257,
        "<|player_dossier|>": 50258,
        "<|tactical_qa|>": 50259,
    }

    def __init__(self):
        self._base_enc = tiktoken.get_encoding("gpt2")
        # Extended encoding allowing custom football special tokens
        self.enc = tiktoken.Encoding(
            name="gpt2_football_baseline",
            pat_str=self._base_enc._pat_str,
            mergeable_ranks=self._base_enc._mergeable_ranks,
            special_tokens=self.SPECIAL_TOKENS,
        )

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        """Encode text into token IDs, supporting special tokens."""
        if allowed_special is None:
            allowed_special = set(self.SPECIAL_TOKENS.keys())
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into string."""
        return self.enc.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """Count tokens in string."""
        return len(self.encode(text))

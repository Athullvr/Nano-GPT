"""
TouchlineLM - Custom Byte-Level BPE Tokenizer (Phase 3B.1)
Implements a domain-specific Byte-Pair Encoding (BPE) tokenizer with:
- 256 raw byte base vocabulary (zero OOV)
- Deterministic BPE merges up to target vocabulary size (4,096 tokens)
- Dedicated special token handling (<|endoftext|>, <|narrative|>, <|player_dossier|>, <|tactical_qa|>)
- High-efficiency pre-tokenization regex splitting
- Full serialization and reproducible loading
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import regex

logger = logging.getLogger("bpe_tokenizer")

# Standard GPT-4 regex pattern for natural language pre-tokenization
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

DEFAULT_SPECIAL_TOKENS = {
    "<|endoftext|>": 4092,
    "<|narrative|>": 4093,
    "<|player_dossier|>": 4094,
    "<|tactical_qa|>": 4095,
}


class TouchlineBPETokenizer:
    """Domain-specific Byte-Level BPE Tokenizer for TouchlineLM."""

    def __init__(
        self,
        vocab_size: int = 4096,
        special_tokens: Optional[Dict[str, int]] = None,
        pat_str: str = GPT4_SPLIT_PATTERN
    ):
        self.target_vocab_size = vocab_size
        self.special_tokens = dict(special_tokens or DEFAULT_SPECIAL_TOKENS)
        self.special_token_by_id = {v: k for k, v in self.special_tokens.items()}
        self.pat_str = pat_str
        self.compiled_pat = regex.compile(self.pat_str)

        # Build regex for matching special tokens
        escaped_specials = [regex.escape(k) for k in self.special_tokens.keys()]
        self.special_pat = regex.compile("(" + "|".join(escaped_specials) + ")") if escaped_specials else None

        # Merges: (int, int) -> int (new token id)
        self.merges: Dict[Tuple[int, int], int] = {}
        # Vocab: int -> bytes
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def _get_pair_counts(self, words_and_counts: Dict[Tuple[int, ...], int]) -> Counter:
        """Count adjacent token pairs across all chunk tuples weighted by occurrence."""
        pairs = Counter()
        for word, count in words_and_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += count
        return pairs

    def _merge_word(self, word: Tuple[int, ...], pair: Tuple[int, int], new_id: int) -> Tuple[int, ...]:
        """Replace all occurrences of pair with new_id in word."""
        new_word = []
        i = 0
        p0, p1 = pair
        while i < len(word):
            if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train(self, text: str, verbose: bool = True) -> None:
        """
        Train BPE tokenizer on raw text.
        Must be trained exclusively on the training split!
        """
        # Strip any special tokens before training BPE merges
        if self.special_pat:
            text = self.special_pat.sub(" ", text)

        # 1. Pre-tokenize text into word chunks
        raw_words = self.compiled_pat.findall(text)
        word_counts = Counter(raw_words)
        logger.info(f"Pre-tokenization yielded {len(raw_words):,} tokens ({len(word_counts):,} unique words)")

        # Convert each unique word into tuple of initial byte token IDs
        words_and_counts = {
            tuple(w.encode("utf-8")): count
            for w, count in word_counts.items()
        }

        # Calculate exact number of merges needed
        max_merges = self.target_vocab_size - 256 - len(self.special_tokens)
        logger.info(f"Target vocab: {self.target_vocab_size}. Performing up to {max_merges} merges...")

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for merge_idx in range(max_merges):
            pair_counts = self._get_pair_counts(words_and_counts)
            if not pair_counts:
                logger.info(f"No more pairs to merge at step {merge_idx}.")
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count <= 1:
                logger.info(f"Most frequent pair count is {best_count} at step {merge_idx}; stopping.")
                break

            new_token_id = 256 + merge_idx
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Update words with new merged token
            new_words_and_counts = {}
            for word, count in words_and_counts.items():
                if best_pair[0] in word:
                    new_word = self._merge_word(word, best_pair, new_token_id)
                    new_words_and_counts[new_word] = count
                else:
                    new_words_and_counts[word] = count
            words_and_counts = new_words_and_counts

            if verbose and (merge_idx + 1) % 500 == 0:
                logger.info(f"Completed {merge_idx + 1}/{max_merges} merges. Current vocab: {256 + merge_idx + 1}")

        logger.info(f"BPE training complete: {len(self.merges)} merges learned. Total vocab: {self.vocab_size}")

    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """Encode a single regex chunk of bytes using learned merges."""
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # Find the pair in ids with the lowest merge rank
            best_pair = None
            best_rank = float("inf")
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            ids = list(self._merge_word(tuple(ids), best_pair, self.merges[best_pair]))
        return ids

    def encode(self, text: str, allowed_special: Optional[Union[Set[str], str]] = "all") -> List[int]:
        """
        Encode text into a list of token IDs.
        Handles special tokens cleanly without confusion.
        """
        if not text:
            return []

        # Determine which special tokens to handle
        if allowed_special == "all":
            active_specials = set(self.special_tokens.keys())
        elif isinstance(allowed_special, (set, list)):
            active_specials = set(allowed_special)
        else:
            active_specials = set()

        tokens = []

        if active_specials and self.special_pat:
            # Split text on special tokens
            parts = self.special_pat.split(text)
            for part in parts:
                if not part:
                    continue
                if part in active_specials:
                    tokens.append(self.special_tokens[part])
                else:
                    chunks = self.compiled_pat.findall(part)
                    for chunk in chunks:
                        tokens.extend(self._encode_chunk(chunk.encode("utf-8")))
        else:
            chunks = self.compiled_pat.findall(text)
            for chunk in chunks:
                tokens.extend(self._encode_chunk(chunk.encode("utf-8")))

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into string."""
        byte_parts = []
        for tid in token_ids:
            if tid in self.special_token_by_id:
                byte_parts.append(self.special_token_by_id[tid].encode("utf-8"))
            elif tid in self.vocab:
                byte_parts.append(self.vocab[tid])
            else:
                # Fallback for unrecognized token ID
                byte_parts.append(b"\xef\xbf\xbd")

        full_bytes = b"".join(byte_parts)
        return full_bytes.decode("utf-8", errors="replace")

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save vocabulary, merges, and configuration files reproducibly."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "name": "touchline_bpe_4k",
            "target_vocab_size": self.target_vocab_size,
            "actual_vocab_size": self.vocab_size,
            "n_merges": len(self.merges),
            "pat_str": self.pat_str,
            "special_tokens": self.special_tokens
        }
        with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save vocab as hex-encoded mapping
        vocab_hex = {str(k): v.hex() for k, v in self.vocab.items()}
        with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_hex, f, indent=2)

        # Save merges line by line: "p0 p1 new_id"
        with open(out_dir / "merges.txt", "w", encoding="utf-8") as f:
            for (p0, p1), new_id in self.merges.items():
                f.write(f"{p0} {p1} {new_id}\n")

        logger.info(f"Saved tokenizer artifacts to {out_dir}")

    @classmethod
    def load(cls, input_dir: Union[str, Path]) -> "TouchlineBPETokenizer":
        """Load a saved tokenizer from disk."""
        in_dir = Path(input_dir)
        config_path = in_dir / "tokenizer_config.json"
        vocab_path = in_dir / "vocab.json"
        merges_path = in_dir / "merges.txt"

        if not (config_path.exists() and vocab_path.exists() and merges_path.exists()):
            raise FileNotFoundError(f"Missing tokenizer files in {in_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        tok = cls(
            vocab_size=config["target_vocab_size"],
            special_tokens=config["special_tokens"],
            pat_str=config["pat_str"]
        )

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_hex = json.load(f)
        tok.vocab = {int(k): bytes.fromhex(v) for k, v in vocab_hex.items()}

        tok.merges = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    p0, p1, new_id = int(parts[0]), int(parts[1]), int(parts[2])
                    tok.merges[(p0, p1)] = new_id

        logger.info(f"Loaded tokenizer from {in_dir} (Vocab size: {tok.vocab_size})")
        return tok

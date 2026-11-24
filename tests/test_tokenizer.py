"""
TouchlineLM - Tokenizer Test Suite (Phase 3B.1)
Tests:
1. Exact vocabulary size (4,096 tokens)
2. Special token handling & non-confusion
3. Encode/decode lossless roundtrips
4. Reproducible save and load
5. Split isolation (strict training on train.txt only)
6. Edge cases: empty strings, whitespace, long documents, and unicode
"""

import json
from pathlib import Path
import pytest
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer, DEFAULT_SPECIAL_TOKENS
from src.tokenizer.gpt2_baseline import GPT2BaselineTokenizer


@pytest.fixture
def trained_tokenizer():
    tok_dir = Path("configs/tokenizer")
    assert tok_dir.exists(), "configs/tokenizer directory must exist (run train_tokenizer.py first)"
    return TouchlineBPETokenizer.load(tok_dir)


@pytest.fixture
def gpt2_tokenizer():
    return GPT2BaselineTokenizer()


def test_vocab_size_is_exact_4096(trained_tokenizer):
    assert trained_tokenizer.vocab_size == 4096, f"Expected 4096, got {trained_tokenizer.vocab_size}"
    assert len(trained_tokenizer.special_tokens) == 4
    assert len(trained_tokenizer.merges) == 4096 - 256 - 4  # exactly 3836 merges


def test_special_token_handling(trained_tokenizer):
    for token_str, token_id in DEFAULT_SPECIAL_TOKENS.items():
        assert trained_tokenizer.special_tokens[token_str] == token_id
        encoded = trained_tokenizer.encode(token_str)
        assert encoded == [token_id], f"Special token {token_str} must encode to {[token_id]}, got {encoded}"
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == token_str, f"Decoded mismatch for {token_str}"

    # Embedded special tokens
    compound = "<|narrative|>\n# Match Report\nPlayer: Lionel Messi\n<|endoftext|>"
    tokens = trained_tokenizer.encode(compound)
    assert tokens[0] == DEFAULT_SPECIAL_TOKENS["<|narrative|>"]
    assert tokens[-1] == DEFAULT_SPECIAL_TOKENS["<|endoftext|>"]
    assert trained_tokenizer.decode(tokens) == compound


def test_special_tokens_not_confused_when_disallowed(trained_tokenizer):
    text = "<|narrative|> test"
    # When special tokens are disallowed ("none")
    tokens = trained_tokenizer.encode(text, allowed_special="none")
    assert DEFAULT_SPECIAL_TOKENS["<|narrative|>"] not in tokens
    # But decode still recovers the exact original text losslessly!
    assert trained_tokenizer.decode(tokens) == text


def test_roundtrip_fidelity(trained_tokenizer):
    sample_texts = [
        "In this Final encounter, Argentina and France contested an intense match.",
        "Lionel Andrés Messi Cuccittini completed 49 of 60 attempted passes (81.7% accuracy).",
        "Kylian Mbappé Lottin fashioned 6 shots (4 on target) accumulating 2.44 xG.",
        "Defensive actions: won 8 of 26 tackles, 13 interceptions, 59 ball recoveries."
    ]
    for text in sample_texts:
        tokens = trained_tokenizer.encode(text)
        assert len(tokens) > 0
        decoded = trained_tokenizer.decode(tokens)
        assert decoded == text, f"Fidelity mismatch on: '{text}'"


def test_reproducibility(trained_tokenizer, tmp_path):
    # Save to temp path
    trained_tokenizer.save(tmp_path)
    # Reload
    reloaded = TouchlineBPETokenizer.load(tmp_path)
    assert reloaded.vocab_size == trained_tokenizer.vocab_size
    assert reloaded.merges == trained_tokenizer.merges

    text = "Tactical Q&A: Why did Argentina generate an xG of 3.42 in open play?"
    assert trained_tokenizer.encode(text) == reloaded.encode(text)


def test_split_isolation():
    from scripts.train_tokenizer import main as train_main
    import sys

    # Assert training script raises ValueError if passed val or test files
    for bad_file in ["data/processed/splits/val.txt", "data/processed/splits/test.txt"]:
        sys.argv = ["train_tokenizer.py", "--train-file", bad_file]
        with pytest.raises(ValueError, match="NEVER be trained on validation or test"):
            train_main()


def test_empty_and_unusual_documents(trained_tokenizer):
    # Empty string
    assert trained_tokenizer.encode("") == []
    assert trained_tokenizer.decode([]) == ""

    # Pure whitespace & newlines
    ws = "   \n\n\t  \r\n   "
    assert trained_tokenizer.decode(trained_tokenizer.encode(ws)) == ws

    # Long document (repeated 200 times)
    long_doc = "Argentina defeated France in the 2022 World Cup Final after a 3-3 draw. " * 200
    assert len(long_doc) > 10000
    long_tokens = trained_tokenizer.encode(long_doc)
    assert trained_tokenizer.decode(long_tokens) == long_doc

    # Non-ASCII and accents (critical for international player names)
    accents = "Luka Modrić, Aurélien Tchouaméni, سالم الدosari ⚽🏆"
    assert trained_tokenizer.decode(trained_tokenizer.encode(accents)) == accents


def test_gpt2_baseline_wrapper(gpt2_tokenizer):
    text = "<|narrative|>\nEngland 6 - 2 Iran\n<|endoftext|>"
    tokens = gpt2_tokenizer.encode(text)
    assert len(tokens) > 0
    assert gpt2_tokenizer.decode(tokens) == text
    assert gpt2_tokenizer.count_tokens(text) == len(tokens)

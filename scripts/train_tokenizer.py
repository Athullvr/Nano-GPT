"""
TouchlineLM - Tokenizer Training Script (Phase 3B.1)
Trains the custom 4,096-vocabulary Byte-Pair Encoding (BPE) tokenizer
strictly and exclusively on the training split (data/processed/splits/train.txt).
Ensures zero data leakage from validation or test splits.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer, DEFAULT_SPECIAL_TOKENS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_tokenizer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom TouchlineLM BPE tokenizer.")
    parser.add_argument("--train-file", type=str, default="data/processed/splits/train.txt",
                        help="Path to training corpus file (MUST be train split)")
    parser.add_argument("--output-dir", type=str, default="configs/tokenizer",
                        help="Output directory for tokenizer artifacts")
    parser.add_argument("--vocab-size", type=int, default=4096,
                        help="Target vocabulary size (default: 4096)")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    output_dir = Path(args.output_dir)

    # 1. Split Isolation Guardrail
    forbidden_tokens = ["val.txt", "val.jsonl", "test.txt", "test.jsonl"]
    for fb in forbidden_tokens:
        if fb in str(train_path).lower():
            raise ValueError(f"FATAL: Tokenizer must NEVER be trained on validation or test data! Found: {train_path}")

    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")

    logger.info(f"Reading training corpus from {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    file_size_kb = len(train_text.encode("utf-8")) / 1024
    logger.info(f"Training corpus size: {len(train_text):,} characters ({file_size_kb:.1f} KB)")

    # 2. Initialize and Train Tokenizer
    logger.info(f"Initializing TouchlineBPETokenizer with target vocab size: {args.vocab_size}...")
    tokenizer = TouchlineBPETokenizer(
        vocab_size=args.vocab_size,
        special_tokens=DEFAULT_SPECIAL_TOKENS
    )

    t0 = time.time()
    tokenizer.train(train_text, verbose=True)
    elapsed = time.time() - t0
    logger.info(f"Training completed in {elapsed:.2f} seconds.")

    # 3. Verify Round-trip Correctness on Training Sample
    logger.info("Verifying encode/decode roundtrip fidelity...")
    sample_text = train_text[:2000]
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == sample_text, "FATAL: Encode/decode roundtrip failed on sample text!"
    logger.info(f"Roundtrip fidelity verified on sample ({len(encoded)} tokens)!")

    # Verify special tokens
    for st in DEFAULT_SPECIAL_TOKENS.keys():
        st_enc = tokenizer.encode(st)
        assert len(st_enc) == 1, f"Special token {st} was not encoded as single token! Got: {st_enc}"
        assert tokenizer.decode(st_enc) == st, f"Special token {st} decode mismatch!"
    logger.info("Special tokens verified successfully!")

    # 4. Save Artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir)
    logger.info(f"Tokenizer artifacts successfully saved to {output_dir}")

    # 5. Print Summary
    print("\n" + "=" * 60)
    print("TOUCHLINE-LM BPE TOKENIZER TRAINING SUMMARY")
    print("=" * 60)
    print(f"Training Source:       {train_path}")
    print(f"Corpus Characters:     {len(train_text):,}")
    print(f"Target Vocabulary:     {args.vocab_size}")
    print(f"Actual Vocabulary:     {tokenizer.vocab_size}")
    print(f"Learned BPE Merges:    {len(tokenizer.merges)}")
    print(f"Special Tokens Count:  {len(tokenizer.special_tokens)}")
    print(f"Artifacts Directory:   {output_dir}")
    print(f"Training Duration:     {elapsed:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

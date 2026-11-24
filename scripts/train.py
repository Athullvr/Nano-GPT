#!/usr/bin/env python3
"""
TouchlineLM — Training Script (Phase 4A)

Configurable training entrypoint designed for both local verification (dry-run)
and high-throughput GPU training in Google Colab.

Usage Examples:
  # Local 5-step dry run verification:
  python scripts/train.py --dry-run --device cpu

  # Full training on Google Colab (with CUDA GPU):
  python scripts/train.py --device cuda --batch-size 16 --seq-len 512 --max-steps 3000
"""

import argparse
import logging
from pathlib import Path
import sys
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer
from src.training.dataset import prepare_tokenized_dataset
from src.training.trainer import TouchlineTrainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TouchlineLM Training Pipeline")

    # Data & Artifact paths (all paths relative to repo root by default)
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/splits/train.txt",
        help="Path to training text corpus",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/processed/splits/val.txt",
        help="Path to validation text corpus",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="configs/tokenizer",
        help="Directory containing trained tokenizer artifacts",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.json",
        help="Path to ModelConfig JSON file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/intermediate/token_cache",
        help="Directory to cache pre-tokenized binary files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint to resume training from",
    )

    # Optimization hyperparameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per training step")
    parser.add_argument("--seq-len", type=int, default=256, help="Context sequence length (<= 1024)")
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate floor")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay for 2D weights")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient norm clipping")
    parser.add_argument("--max-steps", type=int, default=3000, help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--eval-interval", type=int, default=250, help="Steps between validation evaluations")
    parser.add_argument("--eval-iters", type=int, default=40, help="Number of batches per evaluation")
    parser.add_argument("--save-interval", type=int, default=500, help="Steps between saving checkpoints")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between console logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device ('auto', 'cuda', 'cpu')",
    )

    # Smoke test flag
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute a 3-step dry run with 2 evaluation iterations and exit",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths relative to current working directory or repo root
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)
    tokenizer_dir = Path(args.tokenizer_dir)
    model_config_path = Path(args.model_config)
    cache_dir = Path(args.cache_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = TouchlineBPETokenizer.load(tokenizer_dir)
    logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")

    # 2. Pre-tokenize and cache datasets
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_bin = cache_dir / "train.bin"
    val_bin = cache_dir / "val.bin"

    if train_bin.exists():
        logger.info(f"Loading cached training tokens from {train_bin}...")
        import numpy as np
        train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    else:
        logger.info(f"Tokenizing training split ({train_file}) -> {train_bin}...")
        train_data = prepare_tokenized_dataset(train_file, tokenizer, bin_path=train_bin)

    if val_bin.exists():
        logger.info(f"Loading cached validation tokens from {val_bin}...")
        import numpy as np
        val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    else:
        logger.info(f"Tokenizing validation split ({val_file}) -> {val_bin}...")
        val_data = prepare_tokenized_dataset(val_file, tokenizer, bin_path=val_bin)

    logger.info(f"Corpus ready: Train tokens = {len(train_data):,}, Val tokens = {len(val_data):,}")

    # 3. Load or build Model
    logger.info(f"Loading model configuration from {model_config_path}...")
    config = ModelConfig.from_json(model_config_path)
    model = TouchlineNanoGPT(config)
    total_params = model.get_num_params(non_embedding=False)
    logger.info(f"TouchlineNanoGPT initialized: {total_params:,} parameters (~{total_params/1e6:.2f}M)")

    # 4. Configure Training
    if args.dry_run:
        logger.info("Dry-run mode activated: Overriding max_steps=3, eval_interval=2, eval_iters=2")
        max_steps = 3
        eval_interval = 2
        eval_iters = 2
        save_interval = 2
    else:
        max_steps = args.max_steps
        eval_interval = args.eval_interval
        eval_iters = args.eval_iters
        save_interval = args.save_interval

    train_config = TrainingConfig(
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=max_steps,
        warmup_steps=args.warmup_steps if not args.dry_run else 1,
        grad_clip=args.grad_clip,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        save_interval=save_interval,
        log_interval=args.log_interval if not args.dry_run else 1,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        seed=args.seed,
    )

    trainer = TouchlineTrainer(
        model=model,
        config=train_config,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
    )

    # 5. Optionally resume from checkpoint
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 6. Execute Training
    results = trainer.train()
    logger.info(f"Training successfully completed. Best validation loss: {results['best_val_loss']:.4f}")

    # 7. Generate sample inference demonstration
    logger.info("Generating post-run sample completions...")
    samples = trainer.generate_samples(max_new_tokens=30)
    for i, s in enumerate(samples, 1):
        logger.info(f"Sample {i}:\nPrompt: {s['prompt']}\nGeneration:\n{s['completion']}\n" + "-" * 50)


if __name__ == "__main__":
    main()

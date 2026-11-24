"""
Automated Unit and Integration Tests for TouchlineLM Training Pipeline (Phase 4A).

Validates:
1. Dataset preparation, pre-tokenization caching, and batch extraction.
2. Next-token causal prediction target alignment (y[b, t] == x[b, t + 1]).
3. AdamW parameter grouping (2D weights receive decay, 1D receive zero decay).
4. Cosine warmup learning rate scheduler behavior across warmup and decay phases.
5. TouchlineTrainer training step, gradient clipping, and weight updates.
6. Periodic evaluation, validation loss, and perplexity computation.
7. Atomic checkpoint saving and full resumption fidelity (weights, step, scheduler).
8. Non-finite loss safety (detection and safe termination on NaN/Inf).
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer
from src.training.dataset import TouchlineDataset, get_batch, prepare_tokenized_dataset
from src.training.optim import CosineWarmupScheduler, configure_optimizers
from src.training.trainer import TouchlineTrainer, TrainingConfig


@pytest.fixture
def small_model_config() -> ModelConfig:
    """Fast, lightweight model configuration for unit test execution."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        bias=False,
        rope_base=10000.0,
        tie_weights=True,
    )


@pytest.fixture
def synthetic_tokens() -> np.ndarray:
    """Synthetic sequence of 1,000 token IDs within vocabulary bounds [0, 255]."""
    np.random.seed(42)
    return np.random.randint(0, 256, size=1000, dtype=np.uint16)


# =========================================================================
# 1. Dataset & Batching Tests
# =========================================================================

def test_get_batch_shapes_and_alignment(synthetic_tokens: np.ndarray):
    """Verify batch dimensions and causal next-token alignment (y[b, t] == x[b, t+1])."""
    batch_size = 4
    seq_len = 32

    x, y = get_batch(synthetic_tokens, batch_size=batch_size, seq_len=seq_len, device="cpu")

    # Shape verification
    assert x.shape == (batch_size, seq_len)
    assert y.shape == (batch_size, seq_len)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # Strict causal alignment check: y at position t must equal x at position t+1
    assert torch.equal(x[:, 1:], y[:, :-1]), "Next-token target alignment mismatch!"


def test_touchline_dataset_len_and_slicing(tmp_path: Path, synthetic_tokens: np.ndarray):
    """Test standard PyTorch Dataset wrapper with both in-memory and binary memmap."""
    seq_len = 50
    dataset_mem = TouchlineDataset(synthetic_tokens, seq_len=seq_len)
    expected_len = (len(synthetic_tokens) - 1) // seq_len
    assert len(dataset_mem) == expected_len

    x0, y0 = dataset_mem[0]
    assert x0.shape == (seq_len,)
    assert y0.shape == (seq_len,)
    assert torch.equal(x0[1:], y0[:-1])

    # Test binary file memmap backing
    bin_file = tmp_path / "test_corpus.bin"
    synthetic_tokens.tofile(bin_file)
    dataset_mmap = TouchlineDataset(bin_file, seq_len=seq_len)
    assert len(dataset_mmap) == expected_len

    x1, y1 = dataset_mmap[0]
    assert torch.equal(x0, x1)
    assert torch.equal(y0, y1)


def test_prepare_tokenized_dataset(tmp_path: Path):
    """Test text reading, tokenization, and uint16 binary file creation."""
    text_file = tmp_path / "sample.txt"
    bin_file = tmp_path / "sample.bin"
    text_file.write_text("Denmark vs Tunisia 0-0 Group Stage", encoding="utf-8")

    # Load actual repo tokenizer
    tokenizer = TouchlineBPETokenizer.load("configs/tokenizer")
    tokens = prepare_tokenized_dataset(text_file, tokenizer, bin_path=bin_file)

    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.uint16
    assert len(tokens) > 0
    assert bin_file.exists()
    assert bin_file.stat().st_size == len(tokens) * 2  # 2 bytes per uint16


# =========================================================================
# 2. Optimizer & Parameter Grouping Tests
# =========================================================================

def test_configure_optimizers_parameter_grouping(small_model_config: ModelConfig):
    """Verify 2D weights get decay, 1D weights get 0 decay, and tied weights are not duplicated."""
    model = TouchlineNanoGPT(small_model_config)
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4)

    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # Ensure all decay params are 2D or higher (Linear.weight)
    for p in decay_group["params"]:
        assert p.dim() >= 2

    # Ensure all no-decay params are 1D (LayerNorm weights/biases)
    for p in no_decay_group["params"]:
        assert p.dim() < 2

    # Verify tied weights: lm_head.weight and token_embedding.embedding.weight appear only once
    all_opt_params = set(id(p) for group in optimizer.param_groups for p in group["params"])
    total_model_trainable = set(id(p) for p in model.parameters() if p.requires_grad)
    assert all_opt_params == total_model_trainable


# =========================================================================
# 3. Learning Rate Scheduler Tests
# =========================================================================

def test_cosine_warmup_scheduler():
    """Verify linear warmup and cosine decay schedule."""
    dummy_param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([dummy_param], lr=1e-3)

    warmup_steps = 100
    max_steps = 1000
    min_lr = 1e-4
    base_lr = 1e-3

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )

    # 1. Warmup phase: linearly increasing
    prev_lr = 0.0
    for step in range(warmup_steps):
        lrs = scheduler.step()
        lr = lrs[0]
        assert lr > prev_lr
        prev_lr = lr

    # Exactly at end of warmup, learning rate reaches peak base_lr
    assert math.isclose(prev_lr, base_lr, rel_tol=1e-3)

    # 2. Decay phase: decreasing down to min_lr
    for step in range(warmup_steps, max_steps):
        lrs = scheduler.step()
        lr = lrs[0]
        assert lr <= prev_lr
        prev_lr = lr

    # At max_steps, learning rate reaches min_lr
    assert math.isclose(prev_lr, min_lr, rel_tol=1e-3)

    # 3. Beyond max_steps: stays at min_lr floor
    lrs = scheduler.step()
    assert math.isclose(lrs[0], min_lr, rel_tol=1e-5)


# =========================================================================
# 4. Trainer Step, Evaluation & Checkpointing Tests
# =========================================================================

def test_trainer_step_updates_weights(
    small_model_config: ModelConfig,
    synthetic_tokens: np.ndarray,
    tmp_path: Path,
):
    """Verify that train_step updates model parameters and increments step."""
    model = TouchlineNanoGPT(small_model_config)
    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=2,
        seq_len=16,
        max_steps=10,
        warmup_steps=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )

    trainer = TouchlineTrainer(
        model=model,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )

    # Record parameter clone before step
    initial_param = model.blocks[0].attn.q_proj.weight.clone()

    loss = trainer.train_step()
    assert isinstance(loss, float)
    assert loss > 0.0
    assert trainer.step == 1

    # Verify weights changed after backward and step
    updated_param = model.blocks[0].attn.q_proj.weight
    assert not torch.equal(initial_param, updated_param), "Weights did not update after train_step!"


def test_trainer_gradient_accumulation(
    small_model_config: ModelConfig,
    synthetic_tokens: np.ndarray,
    tmp_path: Path,
):
    """Verify that train_step executes cleanly with gradient accumulation steps > 1."""
    model = TouchlineNanoGPT(small_model_config)
    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=2,
        seq_len=16,
        grad_accum_steps=3,
        max_steps=5,
        warmup_steps=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )

    trainer = TouchlineTrainer(
        model=model,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )

    initial_param = model.blocks[0].attn.q_proj.weight.clone()
    loss = trainer.train_step()

    assert isinstance(loss, float)
    assert loss > 0.0
    assert trainer.step == 1
    assert not torch.equal(initial_param, model.blocks[0].attn.q_proj.weight)


def test_trainer_evaluate_and_perplexity(
    small_model_config: ModelConfig,
    synthetic_tokens: np.ndarray,
    tmp_path: Path,
):
    """Verify evaluation calculation and finite validation perplexity."""
    model = TouchlineNanoGPT(small_model_config)
    config = TrainingConfig(
        batch_size=2,
        seq_len=16,
        eval_iters=3,
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )

    trainer = TouchlineTrainer(
        model=model,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )

    metrics = trainer.evaluate()
    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "val_perplexity" in metrics
    assert metrics["val_loss"] > 0.0
    assert metrics["val_perplexity"] >= 1.0
    assert math.isclose(metrics["val_perplexity"], math.exp(metrics["val_loss"]), rel_tol=1e-4)


def test_checkpoint_save_and_resume(
    small_model_config: ModelConfig,
    synthetic_tokens: np.ndarray,
    tmp_path: Path,
):
    """Test saving checkpoint atomically, restoring state, and step resumption."""
    checkpoint_dir = tmp_path / "ckpts"
    config = TrainingConfig(
        learning_rate=5e-4,
        batch_size=2,
        seq_len=16,
        max_steps=5,
        warmup_steps=1,
        checkpoint_dir=str(checkpoint_dir),
        device="cpu",
        seed=123,
    )

    # 1. Run 3 steps with Trainer A
    model_a = TouchlineNanoGPT(small_model_config)
    trainer_a = TouchlineTrainer(
        model=model_a,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )
    for _ in range(3):
        trainer_a.train_step()

    ckpt_path = checkpoint_dir / "step_3.pt"
    trainer_a.save_checkpoint(ckpt_path, val_loss=4.5)
    assert ckpt_path.exists()

    # 2. Initialize Trainer B with fresh model and load checkpoint
    model_b = TouchlineNanoGPT(small_model_config)
    trainer_b = TouchlineTrainer(
        model=model_b,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )
    restored_step = trainer_b.load_checkpoint(ckpt_path)

    assert restored_step == 3
    assert trainer_b.step == 3
    assert trainer_b.scheduler.current_step == trainer_a.scheduler.current_step

    # Verify identical model weights
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


# =========================================================================
# 5. Non-Finite Loss Safety Test
# =========================================================================

def test_non_finite_loss_detection(
    small_model_config: ModelConfig,
    synthetic_tokens: np.ndarray,
    tmp_path: Path,
):
    """Verify that encountering non-finite loss (NaN/Inf) raises FloatingPointError cleanly."""
    model = TouchlineNanoGPT(small_model_config)
    config = TrainingConfig(
        batch_size=2,
        seq_len=16,
        checkpoint_dir=str(tmp_path),
        device="cpu",
    )

    trainer = TouchlineTrainer(
        model=model,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )

    # Intentionally corrupt model weights to produce NaN loss
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="Non-finite loss detected"):
        trainer.train_step()


# =========================================================================
# 6. Pre-Training Audit Regression Tests (Phase 4A.1)
# =========================================================================

def test_scheduler_exact_warmup_and_decay_steps():
    """
    Verify exact learning rate at steps 1, 2, 10, 100, and 200 with 200 warmup steps:
    - Step 1: 3.00e-6
    - Step 2: 6.00e-6
    - Step 10: 3.00e-5
    - Step 100: 3.00e-4
    - Step 200: 6.00e-4 (exact peak)
    """
    dummy = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([dummy], lr=6e-4)
    sched = CosineWarmupScheduler(opt, warmup_steps=200, max_steps=3000, min_lr=6e-5)

    recorded_lrs = []
    for step_num in range(1, 205):
        lr_active = opt.param_groups[0]["lr"]
        recorded_lrs.append((step_num, lr_active))
        sched.step()

    expected = {
        1: 3.0e-6,
        2: 6.0e-6,
        10: 3.0e-5,
        100: 3.0e-4,
        200: 6.0e-4,
    }
    for step_target, expected_lr in expected.items():
        actual_lr = recorded_lrs[step_target - 1][1]
        assert math.isclose(actual_lr, expected_lr, rel_tol=1e-5), (
            f"Step {step_target} LR mismatch: expected {expected_lr:.6e}, got {actual_lr:.6e}"
        )


def test_checkpoint_rng_restoration_and_no_warmup_restart(
    small_model_config: ModelConfig,
    synthetic_tokens: np.ndarray,
    tmp_path: Path,
):
    """
    Verify checkpoint resumption:
    1. Resumes at step 250 with cosine decay active (does NOT restart warmup).
    2. Restores Python, NumPy, and PyTorch RNG states identically.
    """
    import random
    ckpt_dir = tmp_path / "resume_test_ckpts"
    config = TrainingConfig(
        learning_rate=6e-4,
        min_lr=6e-5,
        batch_size=2,
        seq_len=16,
        max_steps=1000,
        warmup_steps=100,
        checkpoint_dir=str(ckpt_dir),
        device="cpu",
        seed=777,
    )

    # 1. Trainer A runs past warmup (120 steps)
    model_a = TouchlineNanoGPT(small_model_config)
    trainer_a = TouchlineTrainer(
        model=model_a,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )
    for _ in range(120):
        trainer_a.train_step()

    ckpt_file = ckpt_dir / "step_120.pt"
    trainer_a.save_checkpoint(ckpt_file)

    # Note the LR for the upcoming step 121
    expected_lr_121 = trainer_a.optimizer.param_groups[0]["lr"]
    # Verify we are in decay phase (LR < 6e-4 and LR > 6e-5)
    assert 6e-5 < expected_lr_121 < 6e-4

    # Run step 121 on Trainer A to capture ground truth continuation loss
    loss_a_121 = trainer_a.train_step()

    # 2. Advance Python, NumPy, and PyTorch RNGs arbitrarily to simulate external execution
    random.random()
    np.random.rand(10)
    torch.randn(10)

    # 3. Trainer B loads checkpoint
    model_b = TouchlineNanoGPT(small_model_config)
    trainer_b = TouchlineTrainer(
        model=model_b,
        config=config,
        train_data=synthetic_tokens,
        val_data=synthetic_tokens,
    )
    trainer_b.load_checkpoint(ckpt_file)

    assert trainer_b.step == 120
    # Crucial check: optimizer LR is set to step 121 decay value, NOT step 1 warmup (which was 6e-6)!
    resumed_lr = trainer_b.optimizer.param_groups[0]["lr"]
    assert math.isclose(resumed_lr, expected_lr_121, rel_tol=1e-5)
    assert resumed_lr > 1e-4  # Far above warmup start (6e-6)

    # Run step 121 on Trainer B and verify exact reproduction
    loss_b_121 = trainer_b.train_step()
    assert math.isclose(loss_a_121, loss_b_121, rel_tol=1e-5)


def test_cuda_error_handling_when_cuda_unavailable(
    small_model_config: ModelConfig,
    synthetic_tokens: np.ndarray,
    monkeypatch,
):
    """Verify that requesting CUDA when unavailable raises a descriptive RuntimeError."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    config = TrainingConfig(device="cuda")
    model = TouchlineNanoGPT(small_model_config)

    with pytest.raises(RuntimeError, match="CUDA device requested"):
        TouchlineTrainer(
            model=model,
            config=config,
            train_data=synthetic_tokens,
            val_data=synthetic_tokens,
        )


def test_dataset_token_ids_strictly_within_vocabulary():
    """Verify that all tokens in train and val splits fall strictly within [0, 4095]."""
    tok = TouchlineBPETokenizer.load("configs/tokenizer")
    for split in ["train", "val"]:
        with open(f"data/processed/splits/{split}.txt", "r", encoding="utf-8") as f:
            text = f.read()
        tokens = np.array(tok.encode(text, allowed_special="all"))
        assert tokens.min() >= 0
        assert tokens.max() < 4096
        # Verify <|endoftext|> is present
        assert (tokens == 4092).sum() > 0


"""
TouchlineLM Training Engine.

Implements the complete autoregressive Transformer training loop:
- Batch extraction and device management.
- Training loss calculation and gradient clipping.
- Periodic validation loss and perplexity evaluation.
- Domain sample text generation at evaluation checkpoints.
- Atomic checkpoint saving and state restoration.
- Safe termination on non-finite loss (NaN/Inf).
"""

from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer
from src.training.dataset import get_batch
from src.training.optim import CosineWarmupScheduler, configure_optimizers

logger = logging.getLogger("trainer")


@dataclass
class TrainingConfig:
    """Configuration hyperparameters for TouchlineLM pre-training."""

    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    batch_size: int = 16
    seq_len: int = 256
    max_steps: int = 3000
    warmup_steps: int = 200
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 40
    save_interval: int = 500
    log_interval: int = 10
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    sample_prompts: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Instantiate configuration from dictionary."""
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        if "betas" in filtered and isinstance(filtered["betas"], list):
            filtered["betas"] = tuple(filtered["betas"])
        return cls(**filtered)


class TouchlineTrainer:
    """
    Orchestrates the pre-training loop for TouchlineNanoGPT.
    """

    def __init__(
        self,
        model: TouchlineNanoGPT,
        config: TrainingConfig,
        train_data: Union[np.ndarray, torch.Tensor],
        val_data: Union[np.ndarray, torch.Tensor],
        tokenizer: Optional[TouchlineBPETokenizer] = None,
    ):
        self.model = model
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer

        # Configure compute device
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model.to(self.device)

        # Set deterministic seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(config.seed)

        # Configure optimizer and scheduler
        self.optimizer = configure_optimizers(
            self.model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=config.betas,
        )
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr=config.min_lr,
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Default football prompts for evaluation text generation
        if self.config.sample_prompts is None:
            self.config.sample_prompts = [
                "<|narrative|> # Match Report: Argentina vs France",
                "<|player_dossier|> # Analytical Player Dossier: Lionel Andrés Messi Cuccittini",
                "<|tactical_qa|> In modern football analytics, expected goals (xG)",
            ]

    def train_step(self) -> float:
        """
        Executes a single forward/backward optimization step.

        Returns:
            Scalar training loss for the current step.

        Raises:
            FloatingPointError: If loss is NaN or Infinite.
        """
        self.model.train()
        x, y = get_batch(self.train_data, self.config.batch_size, self.config.seq_len, self.device)

        self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(x, targets=y)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            raise FloatingPointError(
                f"Non-finite loss detected at step {self.step}: {loss_val}. Terminating training safely."
            )

        loss.backward()

        # Gradient clipping prevents exploding gradients in deep Transformers
        if self.config.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.optimizer.step()
        self.scheduler.step()

        self.step += 1
        return loss_val

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the model over multiple validation and training batches.

        Returns:
            Dictionary containing train_loss, val_loss, and val_perplexity.
        """
        self.model.eval()
        results = {}

        for split, data in [("train", self.train_data), ("val", self.val_data)]:
            losses = []
            for _ in range(self.config.eval_iters):
                x, y = get_batch(data, self.config.batch_size, self.config.seq_len, self.device)
                _, loss = self.model(x, targets=y)
                losses.append(loss.item())
            mean_loss = float(np.mean(losses))
            results[f"{split}_loss"] = mean_loss

        val_loss = results["val_loss"]
        # Perplexity = exp(cross_entropy)
        results["val_perplexity"] = float(math.exp(min(val_loss, 20.0)))
        return results

    @torch.no_grad()
    def generate_samples(self, max_new_tokens: int = 40) -> List[Dict[str, str]]:
        """
        Generates autoregressive completions for sample prompts.

        Args:
            max_new_tokens: Tokens to generate per prompt.

        Returns:
            List of prompt/completion dicts.
        """
        if self.tokenizer is None:
            return []

        self.model.eval()
        generations = []

        for prompt in self.config.sample_prompts or []:
            tokens = self.tokenizer.encode(prompt, allowed_special="all")
            idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
            out_idx = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=20)
            completed_text = self.tokenizer.decode(out_idx[0].tolist())
            generations.append({"prompt": prompt, "completion": completed_text})

        return generations

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        val_loss: Optional[float] = None,
        val_perplexity: Optional[float] = None,
    ) -> None:
        """
        Atomically saves training checkpoint containing weights, optimizer, and metadata.

        Args:
            checkpoint_path: Destination path.
            val_loss: Optional validation loss metric.
            val_perplexity: Optional validation perplexity metric.
        """
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")

        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": self.model.config.to_dict(),
            "training_config": self.config.to_dict(),
            "val_loss": val_loss,
            "val_perplexity": val_perplexity,
            "best_val_loss": self.best_val_loss,
            "rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"Saved atomic checkpoint to {path} (Step: {self.step})")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> int:
        """
        Restores model weights, optimizer state, scheduler state, and step from checkpoint.

        Args:
            checkpoint_path: Path to saved checkpoint.

        Returns:
            The restored training step.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())

        logger.info(f"Successfully loaded checkpoint from {path} at step {self.step}")
        return self.step

    def train(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs the training loop up to target max_steps.

        Args:
            max_steps: Optional step override (defaults to config.max_steps).

        Returns:
            Dictionary containing final metrics and training history.
        """
        target_steps = max_steps if max_steps is not None else self.config.max_steps
        logger.info(f"Starting TouchlineLM training from step {self.step} to {target_steps} on {self.device}")

        history: List[Dict[str, Any]] = []

        while self.step < target_steps:
            loss = self.train_step()

            # Logging
            if self.step % self.config.log_interval == 0 or self.step == target_steps:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(f"Step {self.step:5d}/{target_steps} | Train Loss: {loss:.4f} | LR: {lr:.2e}")

            # Evaluation
            if self.step % self.config.eval_interval == 0 or self.step == target_steps:
                metrics = self.evaluate()
                metrics["step"] = self.step
                history.append(metrics)
                logger.info(
                    f"[Eval @ {self.step}] Val Loss: {metrics['val_loss']:.4f} | "
                    f"Val PPL: {metrics['val_perplexity']:.2f}"
                )

                # Check if this is the best checkpoint
                if metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = metrics["val_loss"]
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pt",
                        val_loss=metrics["val_loss"],
                        val_perplexity=metrics["val_perplexity"],
                    )

            # Periodic saving
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
                )

        # Final checkpoint
        final_ckpt = self.checkpoint_dir / "final_model.pt"
        self.save_checkpoint(final_ckpt)
        logger.info(f"Training completed at step {self.step}. Final checkpoint: {final_ckpt}")

        return {
            "final_step": self.step,
            "best_val_loss": self.best_val_loss,
            "history": history,
        }

"""
TouchlineLM Training Package.

Exports dataset handling, optimizer configuration, learning rate scheduling,
and the TouchlineTrainer training loop for TouchlineLM NanoGPT.
"""

from src.training.dataset import TouchlineDataset, get_batch, prepare_tokenized_dataset
from src.training.optim import CosineWarmupScheduler, configure_optimizers
from src.training.trainer import TouchlineTrainer, TrainingConfig

__all__ = [
    "TouchlineDataset",
    "get_batch",
    "prepare_tokenized_dataset",
    "configure_optimizers",
    "CosineWarmupScheduler",
    "TrainingConfig",
    "TouchlineTrainer",
]

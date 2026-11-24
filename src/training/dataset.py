"""
Dataset and Batching pipeline for TouchlineLM.

Provides:
- Efficient memory-mapped token dataset loading.
- Pre-tokenization and uint16 binary caching to prevent redundant memory overhead.
- Next-token prediction (x, y) target alignment.
- Fast random offset batch extraction for Transformer training.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer


def prepare_tokenized_dataset(
    text_path: Union[str, Path],
    tokenizer: TouchlineBPETokenizer,
    bin_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Tokenizes a corpus text file and optionally writes a compact uint16 binary file.

    Args:
        text_path: Path to the input text file.
        tokenizer: TouchlineBPETokenizer instance.
        bin_path: Optional path to save the uint16 binary token array.

    Returns:
        1D numpy array of token IDs (dtype: np.uint16).
    """
    text_path = Path(text_path)
    if not text_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {text_path}")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize with special tokens allowed
    tokens = tokenizer.encode(text, allowed_special="all")
    arr = np.array(tokens, dtype=np.uint16)

    if bin_path is not None:
        bin_path = Path(bin_path)
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        # Write binary file atomically
        tmp_path = bin_path.with_suffix(".tmp")
        arr.tofile(tmp_path)
        tmp_path.replace(bin_path)

    return arr


def get_batch(
    data: Union[np.ndarray, torch.Tensor],
    batch_size: int,
    seq_len: int,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts a batch of next-token prediction sequences from a 1D token array.

    For each sampled offset i:
      x = tokens[i : i + seq_len]
      y = tokens[i + 1 : i + seq_len + 1]
    
    Guarantees:
      - y[b, t] == x[b, t + 1] for all t in [0, seq_len - 2]
      - Zero redundant corpus copies loaded into memory.

    Args:
        data: 1D token array or tensor.
        batch_size: Number of parallel sequences in batch.
        seq_len: Context window length for each sequence.
        device: Target device ("cpu", "cuda", etc.).

    Returns:
        x: Input token IDs [batch_size, seq_len]
        y: Target token IDs [batch_size, seq_len]
    """
    n_tokens = len(data)
    if n_tokens <= seq_len:
        raise ValueError(f"Corpus length ({n_tokens}) must be greater than seq_len ({seq_len})")

    # Generate random starting indices
    ix = torch.randint(0, n_tokens - seq_len, (batch_size,))

    if isinstance(data, np.ndarray):
        x = torch.stack([torch.from_numpy(data[i : i + seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + seq_len + 1].astype(np.int64)) for i in ix])
    else:
        x = torch.stack([data[i : i + seq_len].to(torch.int64) for i in ix])
        y = torch.stack([data[i + 1 : i + seq_len + 1].to(torch.int64) for i in ix])

    if device != "cpu":
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    return x, y


class TouchlineDataset(Dataset):
    """
    Standard PyTorch Dataset interface for contiguous chunked next-token sequences.
    Supports either in-memory arrays or memory-mapped files via np.memmap.
    """

    def __init__(
        self,
        data_or_bin: Union[np.ndarray, str, Path],
        seq_len: int = 256,
    ):
        """
        Initialize the dataset.

        Args:
            data_or_bin: numpy array or path to a uint16 binary file.
            seq_len: Context length for each sequence.
        """
        self.seq_len = seq_len
        if isinstance(data_or_bin, (str, Path)):
            bin_path = Path(data_or_bin)
            if not bin_path.exists():
                raise FileNotFoundError(f"Binary token file not found: {bin_path}")
            # Memory-map the binary file for zero-copy access
            self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        elif isinstance(data_or_bin, np.ndarray):
            self.data = data_or_bin
        else:
            raise TypeError(f"Unsupported data type: {type(data_or_bin)}")

        if len(self.data) <= self.seq_len:
            raise ValueError(
                f"Data length ({len(self.data)}) must be greater than seq_len ({self.seq_len})"
            )

        self.num_chunks = (len(self.data) - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : start + self.seq_len + 1].astype(np.int64))
        return x, y

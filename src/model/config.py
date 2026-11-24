"""
TouchlineLM Model Configuration.

Defines the architectural and operational hyperparameters for TouchlineNanoGPT.
All architectural dimensions are fully configurable, validated upon instantiation,
and serializable to/from JSON.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Union


@dataclass
class ModelConfig:
    """
    Architectural configuration for TouchlineNanoGPT.
    
    Default configuration matches the Phase 3B.2 specifications:
    - vocab_size: 4,096
    - d_model: 256
    - n_layers: 4
    - n_heads: 4
    - head_dim: 64 (d_model // n_heads)
    - d_ff: 1,024 (4 * d_model)
    - max_seq_len: 1,024
    - dropout: 0.1
    - bias: False
    - rope_base: 10,000.0
    - layer_norm_eps: 1e-5
    - tie_weights: True
    """

    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 64
    d_ff: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.1
    bias: bool = False
    rope_base: float = 10000.0
    layer_norm_eps: float = 1e-5
    tie_weights: bool = True

    def __post_init__(self) -> None:
        """Validate architectural invariants."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * head_dim ({self.head_dim})"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.rope_base <= 0:
            raise ValueError(f"rope_base must be positive, got {self.rope_base}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Instantiate configuration from dictionary."""
        known_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

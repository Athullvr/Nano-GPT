"""
TouchlineLM Inference Engine
High-level interface for loading checkpoints and running fast, deterministic text generation.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional, Union, Dict, Any, List
import logging

import torch
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Structured container for model generation results and performance telemetry."""
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    new_tokens: int
    latency_seconds: float
    tokens_per_second: float
    stopped_on_eos: bool
    device: str
    use_cache: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": self.new_tokens,
            "latency_seconds": round(self.latency_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "stopped_on_eos": self.stopped_on_eos,
            "device": self.device,
            "use_cache": self.use_cache,
        }


class TouchlineInferenceEngine:
    """
    Dedicated inference engine for TouchlineNanoGPT models.
    Supports CPU-safe checkpoint deserialization, top-k/top-p nucleus sampling,
    deterministic seed control, and precise latency telemetry.
    """

    def __init__(
        self,
        model: TouchlineNanoGPT,
        tokenizer: TouchlineBPETokenizer,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # Cache default EOS token ID
        self.eos_token_id = self.tokenizer.special_tokens.get("<|endoftext|>", 4092)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        tokenizer_dir: Union[str, Path] = "configs/tokenizer",
        model_config_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ) -> "TouchlineInferenceEngine":
        """
        Loads a trained TouchlineLM model and tokenizer safely into an inference engine.

        Args:
            checkpoint_path: Path to checkpoint .pt file.
            tokenizer_dir: Directory containing trained BPE tokenizer artifacts.
            model_config_path: Optional path to JSON ModelConfig. If omitted, extracted from checkpoint.
            device: Target execution device ('auto', 'cpu', or 'cuda').

        Returns:
            Configured and initialized TouchlineInferenceEngine ready for generation.
        """
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

        # Resolve target device
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device requested ('{device}') but CUDA is not available.")
            target_device = device
        else:
            target_device = "cpu"

        # 1. CPU-safe initial checkpoint loading
        logger.info(f"Loading checkpoint from {ckpt_file} with map_location='cpu'...")
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        # 2. Extract or load model configuration
        if model_config_path is not None and Path(model_config_path).exists():
            model_config = ModelConfig.from_json(model_config_path)
        elif "model_config" in checkpoint:
            model_config = ModelConfig.from_dict(checkpoint["model_config"])
        elif Path("configs/model_config.json").exists():
            model_config = ModelConfig.from_json("configs/model_config.json")
        else:
            raise ValueError("No model configuration found in checkpoint or config paths.")

        # 3. Instantiate model and load state dict
        model = TouchlineNanoGPT(model_config)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # 4. Load tokenizer
        tok_path = Path(tokenizer_dir)
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {tok_path}")
        tokenizer = TouchlineBPETokenizer.load(tok_path)

        logger.info(
            f"Successfully initialized TouchlineInferenceEngine on {target_device} "
            f"({model.get_num_params():,} parameters, step {checkpoint.get('step', 'N/A')})"
        )

        return cls(model=model, tokenizer=tokenizer, device=target_device)

    @staticmethod
    def _sample_next_token(
        logits_step: torch.Tensor,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
    ) -> torch.Tensor:
        """
        Samples the next token ID from next-step logits using greedy or temperature + top-k/top-p.
        """
        if temperature <= 1e-5:
            return torch.argmax(logits_step, dim=-1, keepdim=True)

        logits_step = logits_step / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
            logits_step[logits_step < v[:, [-1]]] = -float("Inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_step, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits_step[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits_step, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        seed: Optional[int] = None,
        stop_at_eos: bool = True,
        stop_token_ids: Optional[Union[int, List[int]]] = None,
        min_tokens: int = 0,
        use_cache: bool = True,
    ) -> GenerationOutput:
        """
        Executes autoregressive text generation from an input prompt.

        Args:
            prompt: Text prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (<= 0.0 or <= 1e-5 for greedy decoding).
            top_k: Optional top-k filtering cutoff.
            top_p: Optional top-p nucleus sampling cumulative threshold.
            seed: Optional random seed for reproducible sampling.
            stop_at_eos: If True, terminates generation upon encountering an EOS/stop token.
            stop_token_ids: Optional additional token ID or list of token IDs to trigger early termination.
            min_tokens: Minimum number of new tokens to generate before allowing EOS termination.
            use_cache: If True (default), uses per-layer KV cache for fast O(1) single-token incremental decode.
                       If False, recomputes entire sequence at each step (legacy / benchmark comparison).

        Returns:
            GenerationOutput containing generated text and performance telemetry.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

        # Build active set of termination tokens
        active_stop_ids = {self.eos_token_id}
        if stop_token_ids is not None:
            if isinstance(stop_token_ids, int):
                active_stop_ids.add(stop_token_ids)
            else:
                active_stop_ids.update(stop_token_ids)

        # Encode input prompt
        prompt_tokens = self.tokenizer.encode(prompt, allowed_special="all")
        if not prompt_tokens:
            # Handle empty prompt by using EOS as default conditioning start
            prompt_tokens = [self.eos_token_id]

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        idx = input_ids.clone()

        stopped_on_eos = False
        new_tokens_count = 0

        # Synchronize GPU for accurate latency measurement
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        if max_new_tokens > 0:
            if not use_cache:
                # Uncached execution path: recompute entire prefix + generated sequence at every step
                for _ in range(max_new_tokens):
                    if idx.size(1) > self.model.config.max_seq_len:
                        idx_cond = idx[:, -self.model.config.max_seq_len:]
                    else:
                        idx_cond = idx

                    logits, _ = self.model(idx_cond)
                    logits_step = logits[:, -1, :].clone()
                    idx_next = self._sample_next_token(logits_step, temperature, top_k, top_p)
                    next_token = idx_next.item()
                    new_tokens_count += 1
                    idx = torch.cat((idx, idx_next), dim=1)

                    if stop_at_eos and next_token in active_stop_ids and new_tokens_count >= min_tokens:
                        stopped_on_eos = True
                        break
            else:
                # Per-layer KV cached execution path:
                # 1. Full-sequence prefill pass
                if idx.size(1) > self.model.config.max_seq_len:
                    idx_cond = idx[:, -self.model.config.max_seq_len:]
                else:
                    idx_cond = idx

                logits, past_key_values = self.model(idx_cond, use_cache=True)
                logits_step = logits[:, -1, :].clone()
                idx_next = self._sample_next_token(logits_step, temperature, top_k, top_p)
                next_token = idx_next.item()
                new_tokens_count += 1
                idx = torch.cat((idx, idx_next), dim=1)

                if stop_at_eos and next_token in active_stop_ids and new_tokens_count >= min_tokens:
                    stopped_on_eos = True
                else:
                    # 2. Single-token incremental decode passes
                    for _ in range(max_new_tokens - 1):
                        if idx.size(1) >= self.model.config.max_seq_len:
                            break

                        logits, past_key_values = self.model(
                            idx_next, past_key_values=past_key_values, use_cache=True
                        )
                        logits_step = logits[:, -1, :].clone()
                        idx_next = self._sample_next_token(logits_step, temperature, top_k, top_p)
                        next_token = idx_next.item()
                        new_tokens_count += 1
                        idx = torch.cat((idx, idx_next), dim=1)

                        if stop_at_eos and next_token in active_stop_ids and new_tokens_count >= min_tokens:
                            stopped_on_eos = True
                            break

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed_time = time.perf_counter() - start_time

        # Decode tokens
        full_tokens = idx[0].tolist()
        gen_token_ids = full_tokens[len(prompt_tokens):]

        full_text = self.tokenizer.decode(full_tokens)
        generated_text = self.tokenizer.decode(gen_token_ids)

        tok_per_sec = new_tokens_count / elapsed_time if elapsed_time > 0 else 0.0

        return GenerationOutput(
            prompt=prompt,
            generated_text=generated_text,
            full_text=full_text,
            prompt_tokens=len(prompt_tokens),
            new_tokens=new_tokens_count,
            latency_seconds=elapsed_time,
            tokens_per_second=tok_per_sec,
            stopped_on_eos=stopped_on_eos,
            device=self.device,
            use_cache=use_cache,
        )

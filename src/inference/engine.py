"""
TouchlineLM Inference Engine
High-level interface for loading checkpoints and running fast, deterministic text generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Optional, Union, Dict, Any, List, Sequence
import logging

import torch
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer
from src.inference.kv_cache import StaticKVCache

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
    cache_type: str = "dynamic"

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
            "cache_type": self.cache_type,
        }


@dataclass
class BatchRequest:
    """Per-request specification for batched text generation."""
    prompt: str
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    min_tokens: int = 0
    stop_at_eos: bool = True
    stop_token_ids: Optional[Union[int, List[int]]] = None
    seed: Optional[int] = None


@dataclass
class BatchGenerationItem:
    """Individual request generation result within a batch."""
    request_index: int
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    new_tokens: int
    ttft_seconds: float
    total_latency_seconds: float
    stopped_on_eos: bool
    status: str = "ok"  # "ok" or "error"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_index": self.request_index,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": self.new_tokens,
            "ttft_seconds": round(self.ttft_seconds, 4),
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "stopped_on_eos": self.stopped_on_eos,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class BatchGenerationOutput:
    """Aggregate result container for batched text generation."""
    items: List[BatchGenerationItem]
    batch_size: int
    prefill_latency_seconds: float
    decode_latency_seconds: float
    total_latency_seconds: float
    aggregate_tokens_per_second: float
    per_request_tokens_per_second: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    successful_requests: int
    failed_requests: int
    cache_type: str
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "batch_size": self.batch_size,
            "prefill_latency_seconds": round(self.prefill_latency_seconds, 4),
            "decode_latency_seconds": round(self.decode_latency_seconds, 4),
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "aggregate_tokens_per_second": round(self.aggregate_tokens_per_second, 2),
            "per_request_tokens_per_second": round(self.per_request_tokens_per_second, 2),
            "peak_allocated_mb": round(self.peak_allocated_mb, 2),
            "peak_reserved_mb": round(self.peak_reserved_mb, 2),
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cache_type": self.cache_type,
            "device": self.device,
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

    @torch.inference_mode()
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
        cache_type: str = "dynamic",
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
            cache_type: Cache implementation strategy ('dynamic' for standard torch.cat, 'static' for preallocated in-place cache).

        Returns:
            GenerationOutput containing generated text and performance telemetry.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

        # Resolve effective cache mode
        if not use_cache or cache_type == "none":
            active_cache_mode = "none"
        elif cache_type == "static":
            active_cache_mode = "static"
        else:
            active_cache_mode = "dynamic"

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
            if active_cache_mode == "none":
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
            elif active_cache_mode == "static":
                # Preallocated zero-allocation KV cache path
                static_cache = StaticKVCache.from_model(self.model, batch_size=1, device=self.device)
                if idx.size(1) > self.model.config.max_seq_len:
                    idx_cond = idx[:, -self.model.config.max_seq_len:]
                else:
                    idx_cond = idx

                logits, _ = self.model(idx_cond, past_key_values=static_cache, use_cache=True)
                logits_step = logits[:, -1, :].clone()
                idx_next = self._sample_next_token(logits_step, temperature, top_k, top_p)
                next_token = idx_next.item()
                new_tokens_count += 1
                idx = torch.cat((idx, idx_next), dim=1)

                if stop_at_eos and next_token in active_stop_ids and new_tokens_count >= min_tokens:
                    stopped_on_eos = True
                else:
                    for _ in range(max_new_tokens - 1):
                        if idx.size(1) >= self.model.config.max_seq_len:
                            break

                        logits, _ = self.model(
                            idx_next, past_key_values=static_cache, use_cache=True
                        )
                        logits_step = logits[:, -1, :].clone()
                        idx_next = self._sample_next_token(logits_step, temperature, top_k, top_p)
                        next_token = idx_next.item()
                        new_tokens_count += 1
                        idx = torch.cat((idx, idx_next), dim=1)

                        if stop_at_eos and next_token in active_stop_ids and new_tokens_count >= min_tokens:
                            stopped_on_eos = True
                            break
            else:
                # Per-layer dynamic KV cached execution path (torch.cat)
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
            use_cache=(active_cache_mode != "none"),
            cache_type=active_cache_mode,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        requests: Sequence[Union[str, BatchRequest]],
        cache_type: str = "dynamic",
        max_batch_size: Optional[int] = None,
    ) -> BatchGenerationOutput:
        """
        Executes batched text generation across multiple heterogeneous requests.

        Supports:
        - Independent prompt lengths via left-padding with token ID 4092 (<|endoftext|>).
        - Per-request position IDs starting from 0 for real prompt tokens.
        - Causal + key padding attention mask with unmasked diagonal on pad rows to prevent NaNs.
        - Per-request sampling parameters (temperature, top_k, top_p, seeds).
        - Independent EOS stopping and per-request token limits.
        - Both dynamic and static KV cache strategies, as well as uncached ('none').
        - OutOfMemoryError recovery and context boundary validation.

        Args:
            requests: Sequence of prompt strings or BatchRequest objects.
            cache_type: 'dynamic', 'static', or 'none'.
            max_batch_size: Optional guard limit on maximum batch size.

        Returns:
            BatchGenerationOutput containing per-request results and aggregate telemetry.
        """
        if not requests:
            return BatchGenerationOutput(
                items=[],
                batch_size=0,
                prefill_latency_seconds=0.0,
                decode_latency_seconds=0.0,
                total_latency_seconds=0.0,
                aggregate_tokens_per_second=0.0,
                per_request_tokens_per_second=0.0,
                peak_allocated_mb=0.0,
                peak_reserved_mb=0.0,
                successful_requests=0,
                failed_requests=0,
                cache_type=cache_type,
                device=self.device,
            )

        if max_batch_size is not None and len(requests) > max_batch_size:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum configured batch size {max_batch_size}"
            )

        # Normalize requests to BatchRequest instances
        norm_requests: List[BatchRequest] = []
        for req in requests:
            if isinstance(req, str):
                norm_requests.append(BatchRequest(prompt=req))
            elif isinstance(req, BatchRequest):
                norm_requests.append(req)
            else:
                raise TypeError(f"Expected str or BatchRequest, got {type(req)}")

        B = len(norm_requests)
        pad_token_id = self.eos_token_id  # 4092

        # Reset CUDA memory stats if on CUDA
        is_cuda = self.device.startswith("cuda")
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        start_total = time.perf_counter()

        # Tokenize and validate all prompts
        tokenized_prompts: List[List[int]] = []
        items: List[Optional[BatchGenerationItem]] = [None] * B
        valid_indices: List[int] = []

        for i, req in enumerate(norm_requests):
            prompt_toks = self.tokenizer.encode(req.prompt, allowed_special="all")
            if not prompt_toks:
                prompt_toks = [self.eos_token_id]
            tokenized_prompts.append(prompt_toks)

            # Check context limit
            total_req_len = len(prompt_toks) + max(req.max_new_tokens, 0)
            if total_req_len > self.model.config.max_seq_len:
                items[i] = BatchGenerationItem(
                    request_index=i,
                    prompt=req.prompt,
                    generated_text="",
                    full_text=req.prompt,
                    prompt_tokens=len(prompt_toks),
                    new_tokens=0,
                    ttft_seconds=0.0,
                    total_latency_seconds=0.0,
                    stopped_on_eos=False,
                    status="error",
                    error=(
                        f"Context limit exceeded: prompt length {len(prompt_toks)} + "
                        f"max_new_tokens {req.max_new_tokens} = {total_req_len} > "
                        f"max_seq_len {self.model.config.max_seq_len}"
                    ),
                )
            else:
                valid_indices.append(i)

        # If all requests failed pre-flight checks
        if not valid_indices:
            total_time = time.perf_counter() - start_total
            return BatchGenerationOutput(
                items=[it for it in items if it is not None],
                batch_size=B,
                prefill_latency_seconds=0.0,
                decode_latency_seconds=0.0,
                total_latency_seconds=total_time,
                aggregate_tokens_per_second=0.0,
                per_request_tokens_per_second=0.0,
                peak_allocated_mb=0.0,
                peak_reserved_mb=0.0,
                successful_requests=0,
                failed_requests=B,
                cache_type=cache_type,
                device=self.device,
            )

        # If any request failed context limits, mark others as ok or fail cleanly
        # In batch lockstep mode, we execute only on valid_indices if valid_indices == range(B),
        # but to keep strict batch indexing, if there are failed requests, we execute on valid ones
        # or we can process valid ones as sub-batch.
        # Notice: mapping valid sub-batch back to original request index.
        active_requests = [norm_requests[i] for i in valid_indices]
        active_tokenized = [tokenized_prompts[i] for i in valid_indices]
        B_act = len(active_requests)

        try:
            # Execute batched generation for active requests
            active_items, t_prefill, t_decode = self._run_batched_generation(
                active_requests=active_requests,
                active_tokenized=active_tokenized,
                valid_indices=valid_indices,
                cache_type=cache_type,
                pad_token_id=pad_token_id,
            )
            for idx_in_valid, orig_idx in enumerate(valid_indices):
                items[orig_idx] = active_items[idx_in_valid]
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OutOfMemoryError in generate_batch: {e}")
            if is_cuda:
                torch.cuda.empty_cache()
            for orig_idx in valid_indices:
                if items[orig_idx] is None:
                    items[orig_idx] = BatchGenerationItem(
                        request_index=orig_idx,
                        prompt=norm_requests[orig_idx].prompt,
                        generated_text="",
                        full_text=norm_requests[orig_idx].prompt,
                        prompt_tokens=len(tokenized_prompts[orig_idx]),
                        new_tokens=0,
                        ttft_seconds=0.0,
                        total_latency_seconds=0.0,
                        stopped_on_eos=False,
                        status="error",
                        error=f"CUDA OutOfMemoryError: {e}",
                    )
            t_prefill = 0.0
            t_decode = 0.0

        if is_cuda:
            torch.cuda.synchronize(self.device)
            peak_alloc = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            peak_res = torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
        else:
            peak_alloc = 0.0
            peak_res = 0.0

        total_latency = time.perf_counter() - start_total

        final_items = [it for it in items if it is not None]
        success_count = sum(1 for it in final_items if it.status == "ok")
        failed_count = sum(1 for it in final_items if it.status != "ok")
        total_gen_tokens = sum(it.new_tokens for it in final_items if it.status == "ok")

        agg_tps = total_gen_tokens / total_latency if total_latency > 0 else 0.0
        avg_req_tps = (
            sum(
                (it.new_tokens / it.total_latency_seconds)
                for it in final_items
                if it.status == "ok" and it.total_latency_seconds > 0
            ) / max(success_count, 1)
        )

        return BatchGenerationOutput(
            items=final_items,
            batch_size=B,
            prefill_latency_seconds=t_prefill,
            decode_latency_seconds=t_decode,
            total_latency_seconds=total_latency,
            aggregate_tokens_per_second=agg_tps,
            per_request_tokens_per_second=avg_req_tps,
            peak_allocated_mb=peak_alloc,
            peak_reserved_mb=peak_res,
            successful_requests=success_count,
            failed_requests=failed_count,
            cache_type=cache_type,
            device=self.device,
        )

    def _run_batched_generation(
        self,
        active_requests: List[BatchRequest],
        active_tokenized: List[List[int]],
        valid_indices: List[int],
        cache_type: str,
        pad_token_id: int,
    ):
        """Internal execution loop for active batch requests."""
        B = len(active_requests)
        device = self.device
        is_cuda = device.startswith("cuda")

        # Determine prompt lengths and P_max
        prompt_lens = [len(toks) for toks in active_tokenized]
        P_max = max(prompt_lens)

        # Build left-padded input_ids [B, P_max], key_padding_mask [B, P_max], position_ids [B, P_max]
        input_ids = torch.full((B, P_max), pad_token_id, dtype=torch.long, device=device)
        key_padding_mask = torch.zeros((B, P_max), dtype=torch.bool, device=device)
        position_ids = torch.zeros((B, P_max), dtype=torch.long, device=device)

        for b in range(B):
            L_b = prompt_lens[b]
            pad_len = P_max - L_b
            input_ids[b, pad_len:] = torch.tensor(active_tokenized[b], dtype=torch.long, device=device)
            key_padding_mask[b, pad_len:] = True
            position_ids[b, pad_len:] = torch.arange(L_b, dtype=torch.long, device=device)

        # Build prefill attention mask [B, 1, P_max, P_max]
        causal = torch.tril(torch.ones((P_max, P_max), device=device, dtype=torch.bool))
        attn_mask = causal.unsqueeze(0).unsqueeze(0) & key_padding_mask.unsqueeze(1).unsqueeze(2)
        diag = torch.arange(P_max, device=device)
        attn_mask[:, :, diag, diag] = True

        # Per-request independent sampling generators
        generators = []
        for req in active_requests:
            if req.seed is not None:
                g = torch.Generator(device=device)
                g.manual_seed(req.seed)
            else:
                g = None
            generators.append(g)

        # Stop token sets per request
        stop_ids_per_req = []
        for req in active_requests:
            stops = {self.eos_token_id} if req.stop_at_eos else set()
            if req.stop_token_ids is not None:
                if isinstance(req.stop_token_ids, int):
                    stops.add(req.stop_token_ids)
                else:
                    stops.update(req.stop_token_ids)
            stop_ids_per_req.append(stops)

        # Initialize KV cache
        if cache_type == "static":
            kv_cache = StaticKVCache.from_model(self.model, batch_size=B, device=device)
            use_cache = True
        elif cache_type == "none":
            kv_cache = None
            use_cache = False
        else:
            kv_cache = None
            use_cache = True

        # Generation tracking
        max_new_tokens_all = max(req.max_new_tokens for req in active_requests)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        stopped_on_eos = [False] * B
        generated_tokens: List[List[int]] = [[] for _ in range(B)]
        gen_counts = torch.zeros(B, dtype=torch.long, device=device)

        # ---------------- PREFILL ----------------
        if is_cuda:
            torch.cuda.synchronize(device)
        t_prefill_start = time.perf_counter()

        if cache_type == "static":
            logits, _ = self.model(
                input_ids,
                past_key_values=kv_cache,
                use_cache=True,
                position_ids=position_ids,
                attn_mask=attn_mask,
            )
        elif cache_type == "none":
            logits, _ = self.model(
                input_ids,
                position_ids=position_ids,
                attn_mask=attn_mask,
            )
        else:
            logits, past_key_values = self.model(
                input_ids,
                use_cache=True,
                position_ids=position_ids,
                attn_mask=attn_mask,
            )

        if is_cuda:
            torch.cuda.synchronize(device)
        t_prefill = time.perf_counter() - t_prefill_start

        # Sample first token for each request
        # Prefill logits at [:, -1, :] correspond to the final prompt token of each row!
        logits_last = logits[:, -1, :]  # [B, vocab_size]
        next_tokens = torch.full((B, 1), pad_token_id, dtype=torch.long, device=device)

        for b in range(B):
            req = active_requests[b]
            if req.max_new_tokens <= 0:
                finished[b] = True
                continue

            tok_id = self._sample_token_for_request(
                logits_last[b : b + 1, :],
                req=req,
                generator=generators[b],
            )
            next_tokens[b, 0] = tok_id
            generated_tokens[b].append(tok_id)
            gen_counts[b] += 1

            if tok_id in stop_ids_per_req[b] and gen_counts[b] >= req.min_tokens:
                finished[b] = True
                stopped_on_eos[b] = True
            elif gen_counts[b] >= req.max_new_tokens:
                finished[b] = True

        # Keep track of full sequence for uncached mode
        if cache_type == "none":
            current_seq = torch.cat([input_ids, next_tokens], dim=1)

        # ---------------- DECODE LOOP ----------------
        t_decode_start = time.perf_counter()
        step = 0

        while not finished.all() and (step < max_new_tokens_all - 1):
            step += 1
            current_seq_len = P_max + step  # Current total key sequence length

            # Build decode position_ids: [B, 1]
            # Real position for row b is L_b + step - 1 (since prompt tokens occupy 0 .. L_b - 1)
            decode_pos = torch.tensor(
                [prompt_lens[b] + step - 1 for b in range(B)],
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)

            # Build decode attention mask: [B, 1, 1, current_seq_len]
            # Key padding mask: True for valid prompt tokens and all newly generated tokens
            decode_key_mask = torch.cat(
                [
                    key_padding_mask,
                    torch.ones((B, step), dtype=torch.bool, device=device),
                ],
                dim=1,
            )
            decode_attn_mask = decode_key_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, current_seq_len]

            # Forward step tokens
            step_input = next_tokens.clone()  # [B, 1]

            if cache_type == "static":
                logits, _ = self.model(
                    step_input,
                    past_key_values=kv_cache,
                    use_cache=True,
                    position_ids=decode_pos,
                    attn_mask=decode_attn_mask,
                )
            elif cache_type == "none":
                # For uncached, input is entire current_seq with full causal + pad mask
                uncached_len = current_seq.size(1)
                uncached_pos = torch.zeros((B, uncached_len), dtype=torch.long, device=device)
                for b in range(B):
                    pad_len = P_max - prompt_lens[b]
                    uncached_pos[b, pad_len:] = torch.arange(
                        prompt_lens[b] + step, dtype=torch.long, device=device
                    )
                u_causal = torch.tril(torch.ones((uncached_len, uncached_len), device=device, dtype=torch.bool))
                u_mask = u_causal.unsqueeze(0).unsqueeze(0) & decode_key_mask.unsqueeze(1).unsqueeze(2)
                u_diag = torch.arange(uncached_len, device=device)
                u_mask[:, :, u_diag, u_diag] = True

                logits, _ = self.model(
                    current_seq,
                    position_ids=uncached_pos,
                    attn_mask=u_mask,
                )
            else:
                logits, past_key_values = self.model(
                    step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=decode_pos,
                    attn_mask=decode_attn_mask,
                )

            logits_step = logits[:, -1, :]  # [B, vocab_size]

            # Sample next token per row
            for b in range(B):
                if finished[b]:
                    next_tokens[b, 0] = pad_token_id
                    continue

                req = active_requests[b]
                tok_id = self._sample_token_for_request(
                    logits_step[b : b + 1, :],
                    req=req,
                    generator=generators[b],
                )
                next_tokens[b, 0] = tok_id
                generated_tokens[b].append(tok_id)
                gen_counts[b] += 1

                if tok_id in stop_ids_per_req[b] and gen_counts[b] >= req.min_tokens:
                    finished[b] = True
                    stopped_on_eos[b] = True
                elif gen_counts[b] >= req.max_new_tokens:
                    finished[b] = True

            if cache_type == "none":
                current_seq = torch.cat([current_seq, next_tokens], dim=1)

        if is_cuda:
            torch.cuda.synchronize(device)
        t_decode = time.perf_counter() - t_decode_start

        # Assemble individual BatchGenerationItems
        items: List[BatchGenerationItem] = []
        for b in range(B):
            req = active_requests[b]
            orig_idx = valid_indices[b]
            gen_toks = generated_tokens[b]
            gen_text = self.tokenizer.decode(gen_toks)
            full_toks = active_tokenized[b] + gen_toks
            full_text = self.tokenizer.decode(full_toks)
            req_latency = t_prefill + t_decode

            items.append(
                BatchGenerationItem(
                    request_index=orig_idx,
                    prompt=req.prompt,
                    generated_text=gen_text,
                    full_text=full_text,
                    prompt_tokens=prompt_lens[b],
                    new_tokens=len(gen_toks),
                    ttft_seconds=t_prefill,
                    total_latency_seconds=req_latency,
                    stopped_on_eos=stopped_on_eos[b],
                    status="ok",
                    error=None,
                )
            )

        return items, t_prefill, t_decode

    def _sample_token_for_request(
        self,
        logits_row: torch.Tensor,
        req: BatchRequest,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """
        Samples a single token for a specific request row with its own sampling config and generator.
        """
        if req.temperature <= 1e-5:
            return torch.argmax(logits_row, dim=-1).item()

        logits_row = logits_row / max(req.temperature, 1e-5)

        if req.top_k is not None and req.top_k > 0:
            v, _ = torch.topk(logits_row, min(req.top_k, logits_row.size(-1)))
            logits_row[logits_row < v[:, [-1]]] = -float("Inf")

        if req.top_p is not None and 0.0 < req.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_row, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > req.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits_row[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits_row, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=generator)
        return sampled.item()

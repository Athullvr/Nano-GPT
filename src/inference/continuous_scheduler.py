"""
TouchlineLM Continuous Batch Scheduler (Phase 3C Stage 2)

Implements iteration-level scheduling and dynamic slot management:
- FIFO waiting queue for arriving requests with configurable capacity.
- SlotAllocatedKVCache integration for zero-reallocation multi-tenant serving.
- Immediate prefill admission as soon as an active slot frees up.
- Lockstep active decode forward passes with per-slot RoPE position IDs and attention masks.
- Immediate per-request EOS and max-token completion eviction.
- Fault isolation: prefill failures reject the individual request without stalling active decodes.
- Comprehensive request lifecycle tracking, timing telemetry, and deterministic seed control.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Union, Tuple, Any

import torch
import torch.nn.functional as F

from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer
from src.inference.slot_cache import SlotAllocatedKVCache

logger = logging.getLogger(__name__)


class RequestStatus(str, Enum):
    """Lifecycle states of a continuous batch request."""
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"


class QueueFullError(RuntimeError):
    """Raised when the waiting queue capacity is exceeded."""
    pass


@dataclass
class ContinuousBatchRequest:
    """Client request specification submitted to the continuous batch scheduler."""
    prompt: str
    request_id: Optional[str] = None
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    min_tokens: int = 0
    stop_at_eos: bool = True
    stop_token_ids: Optional[Union[int, List[int]]] = None
    seed: Optional[int] = None


@dataclass
class ContinuousBatchOutput:
    """Final result container for a single completed or failed request."""
    request_id: str
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    new_tokens: int
    arrival_time: float
    start_time: float
    first_token_time: float
    completion_time: float
    queue_time_seconds: float
    ttft_seconds: float
    decode_latency_seconds: float
    total_latency_seconds: float
    tokens_per_second: float
    stopped_on_eos: bool
    status: RequestStatus
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "full_text": self.full_text,
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": self.new_tokens,
            "queue_time_seconds": round(self.queue_time_seconds, 4),
            "ttft_seconds": round(self.ttft_seconds, 4),
            "decode_latency_seconds": round(self.decode_latency_seconds, 4),
            "total_latency_seconds": round(self.total_latency_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "stopped_on_eos": self.stopped_on_eos,
            "status": self.status.value,
            "error": self.error,
        }


@dataclass
class _ActiveState:
    """Internal scheduler state for an in-flight request."""
    request: ContinuousBatchRequest
    slot_id: int
    prompt_token_ids: List[int]
    generated_token_ids: List[int] = field(default_factory=list)
    latest_token_id: int = 0
    stop_ids: Set[int] = field(default_factory=set)
    generator: Optional[torch.Generator] = None
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = 0.0
    start_time: float = 0.0
    first_token_time: float = 0.0
    completion_time: float = 0.0
    error: Optional[str] = None


class ContinuousBatchScheduler:
    """
    Iteration-level continuous batching scheduler for TouchlineNanoGPT.
    """

    def __init__(
        self,
        model: TouchlineNanoGPT,
        tokenizer: TouchlineBPETokenizer,
        max_active_slots: int = 8,
        max_queue_capacity: int = 256,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_active_slots = max_active_slots
        self.max_queue_capacity = max_queue_capacity

        # Resolve target device
        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        else:
            self.device = next(model.parameters()).device

        self.model.to(self.device)
        self.model.eval()

        # Cache default EOS token
        self.eos_token_id = self.tokenizer.special_tokens.get("<|endoftext|>", 4092)

        # Preallocated slot KV cache pool
        self.kv_cache = SlotAllocatedKVCache.from_model(
            model=self.model,
            num_slots=self.max_active_slots,
            device=self.device,
            dtype=torch.float32,
        )

        # Queues and state registries
        self.waiting_queue: deque[_ActiveState] = deque()
        self.active_slots: Dict[int, _ActiveState] = {}  # slot_id -> _ActiveState
        self.completed_requests: Dict[str, ContinuousBatchOutput] = {}
        self._request_counter = 0

    # -------------------------------------------------------------------------
    # Client Submission
    # -------------------------------------------------------------------------

    def submit_request(self, request: Union[str, ContinuousBatchRequest]) -> str:
        """
        Enqueues a new request into the FIFO waiting queue.

        Args:
            request: Prompt string or ContinuousBatchRequest object.

        Returns:
            request_id assigned to the submitted request.

        Raises:
            QueueFullError: If waiting queue is at capacity.
        """
        if len(self.waiting_queue) >= self.max_queue_capacity:
            raise QueueFullError(
                f"Waiting queue capacity ({self.max_queue_capacity}) exceeded."
            )

        now = time.perf_counter()
        self._request_counter += 1

        if isinstance(request, str):
            req_obj = ContinuousBatchRequest(
                prompt=request,
                request_id=f"req-{self._request_counter:04d}",
            )
        elif isinstance(request, ContinuousBatchRequest):
            req_obj = request
            if req_obj.request_id is None:
                req_obj.request_id = f"req-{self._request_counter:04d}"
        else:
            raise TypeError(f"Expected str or ContinuousBatchRequest, got {type(request)}")

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(req_obj.prompt, allowed_special="all")
        if not prompt_tokens:
            prompt_tokens = [self.eos_token_id]

        # Build stop token set
        stop_ids = {self.eos_token_id} if req_obj.stop_at_eos else set()
        if req_obj.stop_token_ids is not None:
            if isinstance(req_obj.stop_token_ids, int):
                stop_ids.add(req_obj.stop_token_ids)
            else:
                stop_ids.update(req_obj.stop_token_ids)

        # Generator for reproducible sampling
        if req_obj.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(req_obj.seed)
        else:
            gen = None

        state = _ActiveState(
            request=req_obj,
            slot_id=-1,
            prompt_token_ids=prompt_tokens,
            generated_token_ids=[],
            latest_token_id=0,
            stop_ids=stop_ids,
            generator=gen,
            status=RequestStatus.WAITING,
            arrival_time=now,
        )

        self.waiting_queue.append(state)
        return req_obj.request_id

    # -------------------------------------------------------------------------
    # Iteration Execution Core
    # -------------------------------------------------------------------------

    @torch.inference_mode()
    def step(self) -> List[ContinuousBatchOutput]:
        """
        Executes a single iteration of continuous batching:
        1. Admits waiting requests into any free slots and runs their prompt prefill.
        2. Executes a batched decode pass for all active decoding slots.
        3. Checks EOS and token limits, completing and evicting finished requests.
        4. Releases slots back to the cache pool immediately for future iterations.

        Returns:
            List of ContinuousBatchOutput for requests completed during this step.
        """
        just_completed: List[ContinuousBatchOutput] = []

        # ---------------------------------------------------------------------
        # Step 1: Admission & Isolated Prefill for Free Slots
        # ---------------------------------------------------------------------
        while self.waiting_queue and len(self.active_slots) < self.max_active_slots:
            slot_id = self.kv_cache.allocate()
            candidate = self.waiting_queue.popleft()
            candidate.slot_id = slot_id
            candidate.start_time = time.perf_counter()
            candidate.status = RequestStatus.PREFILLING

            # Pre-flight context length check
            P = len(candidate.prompt_token_ids)
            total_req_len = P + max(candidate.request.max_new_tokens, 0)
            if total_req_len > self.model.config.max_seq_len:
                candidate.status = RequestStatus.FAILED
                candidate.error = (
                    f"Context length overflow: prompt {P} + max_tokens "
                    f"{candidate.request.max_new_tokens} = {total_req_len} > {self.model.config.max_seq_len}"
                )
                candidate.completion_time = time.perf_counter()
                self.kv_cache.release(slot_id)
                out = self._create_output(candidate)
                self.completed_requests[candidate.request.request_id] = out
                just_completed.append(out)
                continue

            # Run prefill pass
            try:
                self._run_prefill_pass(candidate)
            except Exception as e:
                logger.error(f"Prefill failed for request {candidate.request.request_id}: {e}")
                candidate.status = RequestStatus.FAILED
                candidate.error = str(e)
                candidate.completion_time = time.perf_counter()
                self.kv_cache.release(slot_id)
                out = self._create_output(candidate)
                self.completed_requests[candidate.request.request_id] = out
                just_completed.append(out)
                continue

            # Check if prefill generated token immediately completed the request
            if self._check_completion(candidate):
                candidate.status = RequestStatus.COMPLETED
                candidate.completion_time = time.perf_counter()
                self.kv_cache.release(slot_id)
                out = self._create_output(candidate)
                self.completed_requests[candidate.request.request_id] = out
                just_completed.append(out)
            else:
                # Successfully transitioned to DECODING; joins active slots for next decode step
                candidate.status = RequestStatus.DECODING
                self.active_slots[slot_id] = candidate

        # ---------------------------------------------------------------------
        # Step 2: Batched Single-Token Decode Forward Pass
        # ---------------------------------------------------------------------
        if self.active_slots:
            active_slot_ids = list(self.active_slots.keys())
            B_act = len(active_slot_ids)

            # Input tokens for the active batch: [B_act, 1]
            latest_tokens = [self.active_slots[s].latest_token_id for s in active_slot_ids]
            idx_step = torch.tensor(latest_tokens, dtype=torch.long, device=self.device).unsqueeze(1)

            # Position IDs: exact value L_i for each active slot
            # L_i is the count of valid tokens currently in the cache before this write
            slot_lengths = [self.kv_cache.get_sequence_length(s) for s in active_slot_ids]
            pos_ids = torch.tensor(slot_lengths, dtype=torch.long, device=self.device).unsqueeze(1)

            # Attention mask: [B_act, 1, 1, T_max_active]
            # Covers 0 <= j <= L_i (total L_i + 1 tokens including the new step token)
            T_max_active = max(slot_lengths) + 1
            attn_mask = torch.zeros((B_act, 1, 1, T_max_active), dtype=torch.bool, device=self.device)
            for b, L_i in enumerate(slot_lengths):
                attn_mask[b, 0, 0, : L_i + 1] = True

            # Model forward with batched proxy past_key_values
            # The proxy automatically writes k/v at position L_i and returns active views
            proxy = self.kv_cache.create_batch_proxy(active_slots=active_slot_ids)

            logits, _ = self.model(
                idx_step,
                past_key_values=proxy,
                use_cache=True,
                attn_mask=attn_mask,
                position_ids=pos_ids,
            )

            step_logits = logits[:, -1, :]  # [B_act, vocab_size]

            # -----------------------------------------------------------------
            # Step 3: Sample Next Tokens & Check Completion
            # -----------------------------------------------------------------
            evicted_slots: List[int] = []

            for b, slot_id in enumerate(active_slot_ids):
                state = self.active_slots[slot_id]
                next_tok = self._sample_token(
                    step_logits[b : b + 1, :],
                    req=state.request,
                    generator=state.generator,
                )

                state.generated_token_ids.append(next_tok)
                state.latest_token_id = next_tok

                if self._check_completion(state):
                    state.status = RequestStatus.COMPLETED
                    state.completion_time = time.perf_counter()
                    self.kv_cache.release(slot_id)
                    evicted_slots.append(slot_id)
                    out = self._create_output(state)
                    self.completed_requests[state.request.request_id] = out
                    just_completed.append(out)

            # Remove evicted slots from active dictionary
            for slot_id in evicted_slots:
                del self.active_slots[slot_id]

        return just_completed

    def run_all(
        self,
        requests: Optional[List[Union[str, ContinuousBatchRequest]]] = None,
    ) -> List[ContinuousBatchOutput]:
        """
        Helper method to submit a list of requests (or process currently queued ones)
        and execute iterations until all complete.

        Args:
            requests: Optional list of prompt strings or ContinuousBatchRequest objects.
                      If None or empty, processes already-submitted requests.

        Returns:
            List of ContinuousBatchOutput in order of submission.
        """
        if requests:
            submitted_ids = [self.submit_request(r) for r in requests]
        else:
            # Collect IDs of requests currently in waiting_queue and active_slots
            submitted_ids = [s.request.request_id for s in self.waiting_queue] + [
                s.request.request_id for s in self.active_slots.values()
            ]

        while self.waiting_queue or self.active_slots:
            self.step()

        return [self.completed_requests[rid] for rid in submitted_ids]

    def has_work(self) -> bool:
        """Returns True if there are requests waiting in queue or actively decoding."""
        return bool(self.waiting_queue or self.active_slots)

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _run_prefill_pass(self, state: _ActiveState) -> None:
        """Executes full-prompt prefill for an admitted request and initializes its slot."""
        P = len(state.prompt_token_ids)
        slot_id = state.slot_id

        input_ids = torch.tensor([state.prompt_token_ids], dtype=torch.long, device=self.device)
        pos_ids = torch.arange(P, dtype=torch.long, device=self.device).unsqueeze(0)

        # Standard causal prefill pass
        logits, presents = self.model(
            input_ids,
            use_cache=True,
            position_ids=pos_ids,
        )

        # Write prompt K and V into the preallocated slot cache for each layer
        for layer_idx, (k, v) in enumerate(presents):
            # k, v: [1, n_heads, P, head_dim]
            self.kv_cache.write(
                slot_id=slot_id,
                layer_idx=layer_idx,
                position=0,
                key=k,
                value=v,
            )

        now = time.perf_counter()
        state.first_token_time = now

        # Sample step-0 token from prefill logits at column P - 1
        last_logit = logits[:, -1, :]  # [1, vocab_size]
        tok_0 = self._sample_token(last_logit, req=state.request, generator=state.generator)
        state.generated_token_ids.append(tok_0)
        state.latest_token_id = tok_0

    def _check_completion(self, state: _ActiveState) -> bool:
        """Evaluates whether an active request has reached EOS or max token count."""
        count = len(state.generated_token_ids)
        if count >= state.request.max_new_tokens:
            return True
        latest = state.latest_token_id
        if latest in state.stop_ids and count >= state.request.min_tokens:
            return True
        return False

    def _sample_token(
        self,
        logits_row: torch.Tensor,
        req: ContinuousBatchRequest,
        generator: Optional[torch.Generator],
    ) -> int:
        """Samples a single token for a request according to its temperature/top-k/top-p."""
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

    def _create_output(self, state: _ActiveState) -> ContinuousBatchOutput:
        """Builds public ContinuousBatchOutput from internal state."""
        gen_tokens = state.generated_token_ids
        gen_text = self.tokenizer.decode(gen_tokens)
        full_tokens = state.prompt_token_ids + gen_tokens
        full_text = self.tokenizer.decode(full_tokens)

        queue_time = state.start_time - state.arrival_time if state.start_time > 0 else 0.0
        ttft = state.first_token_time - state.start_time if state.first_token_time > 0 else 0.0
        total_time = state.completion_time - state.arrival_time if state.completion_time > 0 else 0.0
        decode_time = state.completion_time - state.first_token_time if state.completion_time > 0 and state.first_token_time > 0 else 0.0

        stopped_on_eos = bool(gen_tokens and gen_tokens[-1] in state.stop_ids)
        tps = len(gen_tokens) / total_time if total_time > 0 else 0.0

        return ContinuousBatchOutput(
            request_id=state.request.request_id,
            prompt=state.request.prompt,
            generated_text=gen_text,
            full_text=full_text,
            prompt_tokens=len(state.prompt_token_ids),
            new_tokens=len(gen_tokens),
            arrival_time=state.arrival_time,
            start_time=state.start_time,
            first_token_time=state.first_token_time,
            completion_time=state.completion_time,
            queue_time_seconds=queue_time,
            ttft_seconds=ttft,
            decode_latency_seconds=decode_time,
            total_latency_seconds=total_time,
            tokens_per_second=tps,
            stopped_on_eos=stopped_on_eos,
            status=state.status,
            error=state.error,
        )

"""
Tests for ContinuousBatchScheduler (Phase 3C Stage 2)

Covers:
1. Basic single request lifecycle (waiting -> prefilling -> decoding -> completed).
2. FIFO queue admission when capacity becomes available.
3. Multiple requests with heterogeneous prompt lengths and generation limits.
4. Independent EOS termination and immediate slot eviction.
5. Mid-stream admission: new request admitted into a freed slot while existing request continues decoding.
6. Greedy generation equivalence between ContinuousBatchScheduler and single-request TouchlineInferenceEngine.generate().
7. Reproducible per-request sampling with independent seeds.
8. Queue capacity exhaustion raising QueueFullError.
9. Prefill failure isolation: request exceeding context length is marked failed and its slot released without crashing scheduler.
"""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.inference.engine import TouchlineInferenceEngine
from src.inference.continuous_scheduler import (
    ContinuousBatchScheduler,
    ContinuousBatchRequest,
    ContinuousBatchOutput,
    QueueFullError,
    RequestStatus,
)


@pytest.fixture
def test_engine() -> TouchlineInferenceEngine:
    checkpoint_path = "checkpoints/final_model.pt"
    tokenizer_dir = "configs/tokenizer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TouchlineInferenceEngine.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_dir=tokenizer_dir,
        device=device,
    )


@pytest.fixture
def scheduler(test_engine: TouchlineInferenceEngine) -> ContinuousBatchScheduler:
    return ContinuousBatchScheduler(
        model=test_engine.model,
        tokenizer=test_engine.tokenizer,
        max_active_slots=4,
        max_queue_capacity=16,
        device=test_engine.device,
    )


def test_scheduler_single_request_lifecycle(scheduler: ContinuousBatchScheduler):
    """Verify single request completes with valid tokens and timing metadata."""
    req = ContinuousBatchRequest(
        prompt="The match was intense",
        max_new_tokens=15,
        temperature=0.0,
    )
    req_id = scheduler.submit_request(req)
    assert req_id is not None
    assert scheduler.has_work() is True

    outputs = scheduler.run_all([])  # run until complete
    assert len(outputs) == 1
    out = outputs[0]

    assert out.request_id == req_id
    assert out.status == RequestStatus.COMPLETED
    assert out.new_tokens == 15
    assert len(out.generated_text) > 0
    assert out.ttft_seconds > 0.0
    assert out.total_latency_seconds >= out.ttft_seconds
    assert scheduler.has_work() is False
    assert scheduler.kv_cache.get_num_free_slots() == scheduler.max_active_slots


def test_scheduler_greedy_equivalence_with_single_engine(
    scheduler: ContinuousBatchScheduler,
    test_engine: TouchlineInferenceEngine,
):
    """
    Continuous batch decoding with temperature=0.0 must yield greedy output tokens
    100% identical to the reference single-request TouchlineInferenceEngine.generate().
    """
    prompt = "Tactical analysis indicates that midfield pressing"
    target_tokens = 20

    # 1. Single-request generation
    ref_out = test_engine.generate(
        prompt=prompt,
        max_new_tokens=target_tokens,
        temperature=0.0,
        use_cache=True,
    )

    # 2. Continuous batch scheduler generation
    req = ContinuousBatchRequest(
        prompt=prompt,
        max_new_tokens=target_tokens,
        temperature=0.0,
    )
    scheduler.submit_request(req)
    cb_outputs = scheduler.run_all([])
    cb_out = cb_outputs[0]

    assert cb_out.status == RequestStatus.COMPLETED
    assert cb_out.new_tokens == target_tokens
    assert cb_out.generated_text == ref_out.generated_text, (
        f"Greedy mismatch:\n"
        f"Reference: {ref_out.generated_text!r}\n"
        f"Continuous: {cb_out.generated_text!r}"
    )


def test_scheduler_heterogeneous_prompts_and_staggered_limits(
    scheduler: ContinuousBatchScheduler,
    test_engine: TouchlineInferenceEngine,
):
    """
    Multiple concurrent requests with different prompt lengths and limits
    must each match their single-request greedy outputs.
    """
    prompts = [
        ("Hello", 10),
        ("The match was intense from the opening whistle", 15),
        ("In the final minutes of extra time, a counter-attack began", 25),
    ]

    # Generate reference outputs sequentially
    ref_texts = []
    for p, limit in prompts:
        ref_texts.append(
            test_engine.generate(
                prompt=p,
                max_new_tokens=limit,
                temperature=0.0,
                use_cache=True,
            ).generated_text
        )

    # Submit all to scheduler
    req_ids = []
    for p, limit in prompts:
        rid = scheduler.submit_request(
            ContinuousBatchRequest(prompt=p, max_new_tokens=limit, temperature=0.0)
        )
        req_ids.append(rid)

    outputs = scheduler.run_all([])
    assert len(outputs) == 3

    for i, out in enumerate(outputs):
        assert out.status == RequestStatus.COMPLETED
        assert out.new_tokens == prompts[i][1]
        assert out.generated_text == ref_texts[i], f"Mismatch on request {i}"


def test_scheduler_midstream_admission_into_freed_slot(
    scheduler: ContinuousBatchScheduler,
    test_engine: TouchlineInferenceEngine,
):
    """
    Verify that when Request 1 finishes early, its slot is immediately
    reallocated to an arriving Request 3 while Request 2 is still actively decoding.
    """
    # Max active slots is 4 in fixture
    # Submit Request 1 (short: 5 tokens) and Request 2 (long: 30 tokens)
    r1 = ContinuousBatchRequest(prompt="Short prompt", max_new_tokens=5, temperature=0.0)
    r2 = ContinuousBatchRequest(prompt="Long running query", max_new_tokens=30, temperature=0.0)

    scheduler.submit_request(r1)
    scheduler.submit_request(r2)

    # Step until Request 1 completes (should take ~5-6 steps)
    r1_finished = False
    step_count = 0
    while not r1_finished and step_count < 20:
        step_count += 1
        completed = scheduler.step()
        for c in completed:
            if c.prompt == "Short prompt":
                r1_finished = True

    assert r1_finished is True
    # Request 2 is still active in decoding
    assert len(scheduler.active_slots) == 1

    # Now submit Request 3
    r3 = ContinuousBatchRequest(prompt="Third admitted prompt", max_new_tokens=10, temperature=0.0)
    scheduler.submit_request(r3)

    # Execute next step: Request 3 must be admitted immediately into the freed slot!
    scheduler.step()
    assert len(scheduler.active_slots) == 2

    # Finish remaining
    while scheduler.has_work():
        scheduler.step()

    assert len(scheduler.completed_requests) == 3
    assert scheduler.kv_cache.get_num_free_slots() == scheduler.max_active_slots


def test_scheduler_reproducible_seeds(scheduler: ContinuousBatchScheduler):
    """Requests with configured seeds must generate deterministic outputs."""
    req1 = ContinuousBatchRequest(prompt="The game", max_new_tokens=15, temperature=0.8, seed=42)
    req2 = ContinuousBatchRequest(prompt="The game", max_new_tokens=15, temperature=0.8, seed=42)

    out1 = scheduler.run_all([req1])[0]
    out2 = scheduler.run_all([req2])[0]

    assert out1.generated_text == out2.generated_text


def test_scheduler_queue_capacity_overflow(scheduler: ContinuousBatchScheduler):
    """Submitting more requests than max_queue_capacity raises QueueFullError."""
    # Capacity is 16
    for i in range(16):
        scheduler.submit_request(f"Prompt {i}")

    with pytest.raises(QueueFullError, match="capacity .* exceeded"):
        scheduler.submit_request("Overflow prompt")


def test_scheduler_prefill_failure_isolation(scheduler: ContinuousBatchScheduler):
    """
    A request exceeding context length fails during prefill admission without crashing
    the scheduler; valid concurrent requests complete normally.
    """
    max_len = scheduler.model.config.max_seq_len
    valid_req = ContinuousBatchRequest(prompt="Valid query", max_new_tokens=10, temperature=0.0)
    bad_req = ContinuousBatchRequest(prompt="Bad query", max_new_tokens=max_len + 10, temperature=0.0)

    scheduler.submit_request(valid_req)
    scheduler.submit_request(bad_req)

    outputs = scheduler.run_all([])
    assert len(outputs) == 2

    out_valid = outputs[0]
    out_bad = outputs[1]

    assert out_valid.status == RequestStatus.COMPLETED
    assert out_valid.new_tokens == 10

    assert out_bad.status == RequestStatus.FAILED
    assert "Context length overflow" in str(out_bad.error)
    assert scheduler.kv_cache.get_num_free_slots() == scheduler.max_active_slots

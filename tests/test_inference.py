"""
Tests for TouchlineLM Local Inference Engine and CLI
"""

import math
from pathlib import Path
import pytest
import torch

from src.inference.engine import TouchlineInferenceEngine, GenerationOutput
from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer


@pytest.fixture(scope="module")
def checkpoint_path() -> Path:
    path = Path("checkpoints/final_model.pt")
    if not path.exists():
        pytest.skip("checkpoints/final_model.pt not present; skipping inference checkpoint tests.")
    return path


@pytest.fixture(scope="module")
def cpu_engine(checkpoint_path: Path) -> TouchlineInferenceEngine:
    return TouchlineInferenceEngine.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_dir="configs/tokenizer",
        model_config_path="configs/model_config.json",
        device="cpu",
    )


def test_checkpoint_cpu_safe_loading_and_parameter_count(cpu_engine: TouchlineInferenceEngine):
    """Verify that checkpoint loads safely on CPU and matches exact 4,198,912 parameter count."""
    assert cpu_engine.device == "cpu"
    assert not cpu_engine.model.training  # Model must be in eval mode
    total_params = cpu_engine.model.get_num_params(non_embedding=False)
    assert total_params == 4198912, f"Expected 4,198,912 params, got {total_params}"

    non_emb_params = cpu_engine.model.get_num_params(non_embedding=True)
    assert non_emb_params == 3150336, f"Expected 3,150,336 non-embedding params, got {non_emb_params}"


def test_tokenizer_roundtrip_and_vocabulary(cpu_engine: TouchlineInferenceEngine):
    """Verify tokenizer loaded into engine preserves 4,096 vocab and roundtrip fidelity."""
    assert cpu_engine.tokenizer.vocab_size == 4096
    test_str = "<|narrative|> # Match Report: Argentina vs France (xG: 2.48, xA: 1.15) <|endoftext|>"
    tokens = cpu_engine.tokenizer.encode(test_str, allowed_special="all")
    decoded = cpu_engine.tokenizer.decode(tokens)
    assert decoded == test_str


def test_deterministic_generation_with_seed(cpu_engine: TouchlineInferenceEngine):
    """Verify that setting seed guarantees identical token sequence and text."""
    prompt = "<|narrative|> # Match Report: Argentina vs France\n**Final Scoreline**: Argentina"
    out1 = cpu_engine.generate(prompt, max_new_tokens=25, temperature=0.7, top_k=40, seed=999)
    out2 = cpu_engine.generate(prompt, max_new_tokens=25, temperature=0.7, top_k=40, seed=999)

    assert out1.new_tokens == 25
    assert out2.new_tokens == 25
    assert out1.generated_text == out2.generated_text
    assert out1.full_text == out2.full_text


def test_greedy_decoding_invariance(cpu_engine: TouchlineInferenceEngine):
    """Verify that temperature=0.0 enforces greedy decoding invariant to seed."""
    prompt = "<|player_dossier|> # Analytical Player Dossier: Lionel Andrés Messi Cuccittini"
    out1 = cpu_engine.generate(prompt, max_new_tokens=20, temperature=0.0, seed=1)
    out2 = cpu_engine.generate(prompt, max_new_tokens=20, temperature=0.0, seed=2)

    assert out1.generated_text == out2.generated_text


def test_eos_token_stopping(cpu_engine: TouchlineInferenceEngine):
    """Verify that generation terminates when EOS or stop token is encountered."""
    prompt = "The match began with"
    # Find the next greedy token for this prompt
    prompt_ids = torch.tensor([cpu_engine.tokenizer.encode(prompt, allowed_special="all")])
    logits, _ = cpu_engine.model(prompt_ids)
    next_greedy_token = torch.argmax(logits[:, -1, :], dim=-1).item()

    # If we register next_greedy_token as a stop token, it must terminate after exactly 1 token
    out_stop = cpu_engine.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.0,
        stop_at_eos=True,
        stop_token_ids=next_greedy_token,
    )
    assert out_stop.stopped_on_eos is True
    assert out_stop.new_tokens == 1

    # Also test natural EOS (<|endoftext|>) stopping on benchmark prompt 4
    prompt_qa = "<|tactical_qa|> In modern football analytics, expected goals (xG) measures"
    output = cpu_engine.generate(
        prompt=prompt_qa,
        max_new_tokens=200,
        temperature=0.7,
        top_k=40,
        stop_at_eos=True,
        seed=46,
    )
    assert output.stopped_on_eos is True
    assert output.new_tokens < 200
    assert "<|endoftext|>" in output.full_text


def test_top_p_nucleus_sampling(cpu_engine: TouchlineInferenceEngine):
    """Verify top-p nucleus sampling filters logits and executes without error."""
    prompt = "<|narrative|> # Match Report: Croatia vs Brazil"
    out_p = cpu_engine.generate(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.8,
        top_k=None,  # Pure top-p
        top_p=0.85,
        seed=777,
    )
    assert out_p.new_tokens == 30
    assert len(out_p.generated_text) > 0


def test_empty_prompt_handling(cpu_engine: TouchlineInferenceEngine):
    """Verify that an empty prompt string generates valid output gracefully."""
    output = cpu_engine.generate(prompt="", max_new_tokens=15, temperature=0.7, seed=42)
    assert output.new_tokens == 15
    assert len(output.full_text) > 0


def test_telemetry_metrics(cpu_engine: TouchlineInferenceEngine):
    """Verify generation telemetry outputs valid positive metrics."""
    out = cpu_engine.generate("Football tactics", max_new_tokens=20, seed=12)
    telemetry = out.to_dict()

    assert telemetry["new_tokens"] == 20
    assert telemetry["prompt_tokens"] > 0
    assert telemetry["latency_seconds"] > 0.0
    assert telemetry["tokens_per_second"] > 0.0
    assert telemetry["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available on host")
def test_cuda_inference_execution(checkpoint_path: Path):
    """Verify inference engine initializes and generates on CUDA hardware."""
    cuda_engine = TouchlineInferenceEngine.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_dir="configs/tokenizer",
        device="cuda",
    )
    assert cuda_engine.device.startswith("cuda")
    out = cuda_engine.generate(
        "<|narrative|> # Match Report: Argentina vs France",
        max_new_tokens=25,
        temperature=0.7,
        seed=42,
    )
    assert out.new_tokens == 25
    assert out.device.startswith("cuda")
    assert out.tokens_per_second > 50.0  # Should be very fast on GPU

"""
Tests for TouchlineLM Per-Layer Key-Value (KV) Cache.

Verifies:
1. Cache tensor shapes across all layers during prefill.
2. Incremental sequence length growth during single-token decoding.
3. Position-offset-aware Rotary Position Embedding (RoPE) alignment.
4. Strict numerical logit equivalence between cached and uncached forward passes.
5. Exact generation equivalence for greedy decoding and seeded sampling.
6. Context length boundary enforcement.
7. Empty prompt and single-token prompt handling.
8. CUDA hardware execution and device preservation.
"""

from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.tokenizer.bpe_tokenizer import TouchlineBPETokenizer
from src.inference.engine import TouchlineInferenceEngine, GenerationOutput
from src.inference.kv_cache import StaticKVCache


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=4096,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=64,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> TouchlineNanoGPT:
    torch.manual_seed(42)
    model = TouchlineNanoGPT(tiny_config)
    model.eval()
    return model


@pytest.fixture
def tokenizer() -> TouchlineBPETokenizer:
    tok_dir = Path("configs/tokenizer")
    assert tok_dir.exists(), "Tokenizer directory configs/tokenizer must exist"
    return TouchlineBPETokenizer.load(tok_dir)


@pytest.fixture
def inference_engine(tokenizer: TouchlineBPETokenizer) -> TouchlineInferenceEngine:
    ckpt_path = Path("checkpoints/final_model.pt")
    if ckpt_path.exists():
        return TouchlineInferenceEngine.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            tokenizer_dir=Path("configs/tokenizer"),
            device="cpu",
        )
    # Fallback to initialized model if checkpoint not present
    config = ModelConfig(vocab_size=4096, d_model=256, n_layers=4, n_heads=4, max_seq_len=1024)
    model = TouchlineNanoGPT(config)
    return TouchlineInferenceEngine(model=model, tokenizer=tokenizer, device="cpu")


def test_prefill_cache_shapes(tiny_model: TouchlineNanoGPT, tiny_config: ModelConfig):
    """Prefill pass must return past_key_values for all layers with shape [B, n_heads, seq_len, head_dim]."""
    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))

    with torch.no_grad():
        logits, past_key_values = tiny_model(idx, use_cache=True)

    assert logits.shape == (B, T, tiny_config.vocab_size)
    assert past_key_values is not None
    assert len(past_key_values) == tiny_config.n_layers

    for layer_idx, (k, v) in enumerate(past_key_values):
        assert k.shape == (B, tiny_config.n_heads, T, tiny_config.head_dim), (
            f"Layer {layer_idx} key cache shape mismatch: {k.shape}"
        )
        assert v.shape == (B, tiny_config.n_heads, T, tiny_config.head_dim), (
            f"Layer {layer_idx} value cache shape mismatch: {v.shape}"
        )


def test_incremental_sequence_growth(tiny_model: TouchlineNanoGPT, tiny_config: ModelConfig):
    """Incremental decode steps must increment cache sequence length by exactly 1 per step."""
    B, initial_T = 1, 10
    idx = torch.randint(0, tiny_config.vocab_size, (B, initial_T))

    with torch.no_grad():
        logits, past_key_values = tiny_model(idx, use_cache=True)

    current_len = initial_T
    num_decode_steps = 5

    for step in range(num_decode_steps):
        next_token = torch.randint(0, tiny_config.vocab_size, (B, 1))
        with torch.no_grad():
            step_logits, past_key_values = tiny_model(
                next_token, past_key_values=past_key_values, use_cache=True
            )

        current_len += 1
        assert step_logits.shape == (B, 1, tiny_config.vocab_size)
        assert len(past_key_values) == tiny_config.n_layers

        for layer_idx, (k, v) in enumerate(past_key_values):
            assert k.shape == (B, tiny_config.n_heads, current_len, tiny_config.head_dim), (
                f"Step {step}, Layer {layer_idx}: Key length expected {current_len}, got {k.shape[2]}"
            )
            assert v.shape == (B, tiny_config.n_heads, current_len, tiny_config.head_dim), (
                f"Step {step}, Layer {layer_idx}: Value length expected {current_len}, got {v.shape[2]}"
            )


def test_rope_position_offset_correctness(tiny_model: TouchlineNanoGPT):
    """Verify RoPE precomputed slices at offset m match slicing from position 0."""
    dummy_x = torch.zeros(1, 1, 1, tiny_model.config.head_dim)
    seq_len = 30
    cos_full, sin_full = tiny_model.rope(dummy_x, seq_len=seq_len, offset=0)

    # Test individual offsets
    for offset in [0, 5, 17, 29]:
        cos_offset, sin_offset = tiny_model.rope(dummy_x, seq_len=1, offset=offset)
        assert torch.allclose(cos_offset, cos_full[:, :, offset : offset + 1, :], atol=1e-7)
        assert torch.allclose(sin_offset, sin_full[:, :, offset : offset + 1, :], atol=1e-7)


def test_numerical_logit_equivalence(tiny_model: TouchlineNanoGPT, tiny_config: ModelConfig):
    """
    CRITICAL TEST: Next-token logits from KV cached generation must be strictly
    numerically identical (torch.allclose atol=1e-4) to full-sequence uncached recomputation.
    """
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 12))
    current_seq = prompt.clone()

    # Step 0: Prefill pass vs uncached full pass
    with torch.no_grad():
        logits_uncached_0, _ = tiny_model(current_seq, use_cache=False)
        logits_cached_0, past_key_values = tiny_model(prompt, use_cache=True)

    assert torch.allclose(
        logits_cached_0[:, -1, :], logits_uncached_0[:, -1, :], atol=1e-5, rtol=1e-5
    ), "Prefill next-token logits do not match uncached full forward pass."

    # Multi-step incremental decoding verification
    num_steps = 8
    for step in range(num_steps):
        # Determine next token deterministically (greedy)
        next_tok = torch.argmax(logits_cached_0[:, -1, :], dim=-1, keepdim=True)
        current_seq = torch.cat((current_seq, next_tok), dim=1)

        with torch.no_grad():
            # Uncached: recompute over entire sequence current_seq
            logits_uncached, _ = tiny_model(current_seq, use_cache=False)
            target_logit = logits_uncached[:, -1, :]

            # Cached: forward only next_tok with past_key_values
            logits_cached_step, past_key_values = tiny_model(
                next_tok, past_key_values=past_key_values, use_cache=True
            )
            step_logit = logits_cached_step[:, -1, :]

        # Assert strict numerical equivalence
        max_diff = (step_logit - target_logit).abs().max().item()
        assert torch.allclose(step_logit, target_logit, atol=1e-4, rtol=1e-4), (
            f"Step {step}: Max absolute logit difference {max_diff:.6e} exceeds 1e-4."
        )
        logits_cached_0 = logits_cached_step


def test_greedy_generation_equivalence(inference_engine: TouchlineInferenceEngine):
    """Greedy decoding must produce 100% identical token IDs between cached and uncached paths."""
    prompt = "<|narrative|> # Match Report: Argentina vs France\n**Final Scoreline**: Argentina"

    out_cached = inference_engine.generate(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.0,
        use_cache=True,
    )

    out_uncached = inference_engine.generate(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.0,
        use_cache=False,
    )

    assert out_cached.new_tokens == out_uncached.new_tokens
    assert out_cached.generated_text == out_uncached.generated_text
    assert out_cached.full_text == out_uncached.full_text
    assert out_cached.use_cache is True
    assert out_uncached.use_cache is False


def test_seeded_sampling_generation_equivalence(inference_engine: TouchlineInferenceEngine):
    """Sampling with identical seeds must produce identical generated text between cached and uncached paths."""
    prompt = "<|player_dossier|> # Analytical Player Dossier: Lionel Messi"

    out_cached = inference_engine.generate(
        prompt=prompt,
        max_new_tokens=25,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        seed=1337,
        use_cache=True,
    )

    out_uncached = inference_engine.generate(
        prompt=prompt,
        max_new_tokens=25,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        seed=1337,
        use_cache=False,
    )

    assert out_cached.generated_text == out_uncached.generated_text
    assert out_cached.stopped_on_eos == out_uncached.stopped_on_eos


def test_context_length_boundary(tiny_model: TouchlineNanoGPT, tiny_config: ModelConfig):
    """Model must raise ValueError when past_length + current_seq exceeds max_seq_len."""
    # Pre-fill exactly up to max_seq_len
    full_idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.max_seq_len))
    with torch.no_grad():
        _, past_kv = tiny_model(full_idx, use_cache=True)

    assert past_kv[0][0].shape[2] == tiny_config.max_seq_len

    # Next incremental step should trigger context boundary error
    next_tok = torch.tensor([[1]], dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds maximum configured context length"):
        tiny_model(next_tok, past_key_values=past_kv, use_cache=True)


def test_empty_prompt_and_single_token(inference_engine: TouchlineInferenceEngine):
    """Empty and single-token prompts must generate correctly with KV caching."""
    # Empty prompt
    out_empty = inference_engine.generate(
        prompt="",
        max_new_tokens=10,
        temperature=0.0,
        use_cache=True,
    )
    assert isinstance(out_empty, GenerationOutput)
    assert out_empty.new_tokens > 0
    assert len(out_empty.generated_text) > 0

    # Single-token prompt
    out_single = inference_engine.generate(
        prompt="<|tactical_qa|>",
        max_new_tokens=10,
        temperature=0.0,
        use_cache=True,
    )
    assert isinstance(out_single, GenerationOutput)
    assert out_single.prompt_tokens == 1
    assert out_single.new_tokens > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_execution(tokenizer: TouchlineBPETokenizer):
    """Verify KV cached generation executes on CUDA hardware and preserves tensor devices."""
    ckpt_path = Path("checkpoints/final_model.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint final_model.pt required for CUDA execution test")

    cuda_engine = TouchlineInferenceEngine.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        tokenizer_dir=Path("configs/tokenizer"),
        device="cuda",
    )

    out = cuda_engine.generate(
        prompt="<|narrative|> Match summary",
        max_new_tokens=15,
        temperature=0.0,
        use_cache=True,
    )

    assert out.device == "cuda"
    assert out.new_tokens == 15
    assert out.use_cache is True


def test_static_cache_shapes_and_capacity(tiny_model: TouchlineNanoGPT, tiny_config: ModelConfig):
    """StaticKVCache must preallocate correct tensor buffers and compute exact memory capacity."""
    static_cache = StaticKVCache.from_model(tiny_model, batch_size=2)
    assert len(static_cache) == tiny_config.n_layers
    assert static_cache.get_seq_length() == 0

    for l in range(tiny_config.n_layers):
        assert static_cache.k_cache[l].shape == (2, tiny_config.n_heads, tiny_config.max_seq_len, tiny_config.head_dim)
        assert static_cache.v_cache[l].shape == (2, tiny_config.n_heads, tiny_config.max_seq_len, tiny_config.head_dim)

    # 4 layers * 2 * 2 batch * 4 heads * 64 seq * 64 dim * 4 bytes = 1,048,576 bytes = 1.0 MB
    assert static_cache.memory_bytes() == 1048576
    assert static_cache.memory_mb() == 1.0


def test_static_cache_numerical_equivalence_with_dynamic(tiny_model: TouchlineNanoGPT, tiny_config: ModelConfig):
    """Static preallocated cache must output bit-for-bit identical logits to dynamic cache across multiple steps."""
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 10))

    with torch.inference_mode():
        # Dynamic path
        dyn_logits_0, dyn_past = tiny_model(prompt, use_cache=True)

        # Static path
        static_cache = StaticKVCache.from_model(tiny_model, batch_size=1)
        stat_logits_0, _ = tiny_model(prompt, past_key_values=static_cache, use_cache=True)

    assert torch.allclose(dyn_logits_0, stat_logits_0, atol=1e-5), "Prefill logits mismatch between dynamic and static cache"
    assert static_cache.get_seq_length() == 10

    # Multi-step decode
    for step in range(5):
        next_tok = torch.tensor([[100 + step]])
        with torch.inference_mode():
            dyn_logits_step, dyn_past = tiny_model(next_tok, past_key_values=dyn_past, use_cache=True)
            stat_logits_step, _ = tiny_model(next_tok, past_key_values=static_cache, use_cache=True)

        assert torch.allclose(dyn_logits_step, stat_logits_step, atol=1e-5), f"Step {step} logits mismatch between dynamic and static cache"
        assert static_cache.get_seq_length() == 10 + step + 1


def test_static_cache_greedy_generation_equivalence(inference_engine: TouchlineInferenceEngine):
    """Greedy generation with static cache must produce 100% identical tokens to dynamic cache."""
    prompt = "<|narrative|> # Match Report: Argentina vs France\n**Final Scoreline**: Argentina"

    out_dynamic = inference_engine.generate(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.0,
        cache_type="dynamic",
    )

    out_static = inference_engine.generate(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.0,
        cache_type="static",
    )

    assert out_dynamic.generated_text == out_static.generated_text
    assert out_dynamic.full_text == out_static.full_text
    assert out_static.cache_type == "static"


def test_static_cache_overflow_boundary(tiny_model: TouchlineNanoGPT, tiny_config: ModelConfig):
    """Static cache must raise ValueError when sequence exceeds preallocated capacity."""
    static_cache = StaticKVCache.from_model(tiny_model, batch_size=1)
    full_prompt = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.max_seq_len))

    with torch.inference_mode():
        tiny_model(full_prompt, past_key_values=static_cache, use_cache=True)

    assert static_cache.get_seq_length() == tiny_config.max_seq_len

    next_tok = torch.tensor([[1]])
    with pytest.raises(ValueError, match="exceeds maximum configured context length|exceeds preallocated cache capacity"):
        with torch.inference_mode():
            tiny_model(next_tok, past_key_values=static_cache, use_cache=True)


def test_static_cache_reset(tiny_model: TouchlineNanoGPT):
    """Static cache reset allows re-running generation without memory reallocation."""
    static_cache = StaticKVCache.from_model(tiny_model, batch_size=1)
    p1 = torch.randint(0, 100, (1, 15))
    with torch.inference_mode():
        tiny_model(p1, past_key_values=static_cache, use_cache=True)
    assert static_cache.get_seq_length() == 15

    static_cache.reset()
    assert static_cache.get_seq_length() == 0

    p2 = torch.randint(0, 100, (1, 20))
    with torch.inference_mode():
        tiny_model(p2, past_key_values=static_cache, use_cache=True)
    assert static_cache.get_seq_length() == 20


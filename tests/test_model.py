"""
Comprehensive Unit Tests for TouchlineLM NanoGPT Architecture (Phase 3B.2).

Validates:
1. Model configuration and validation.
2. Tensor output shapes across variable sequence lengths.
3. Loss calculation and finite gradient backpropagation.
4. Causal masking (strictly preventing leakage of future tokens to earlier positions).
5. Rotary Position Embedding (RoPE) properties (norm preservation, relative distance encoding).
6. Weight tying between token embeddings and LM head projection.
7. Exact parameter accounting (~4.2M unique parameters).
8. Context length boundary enforcement (rejection of seq_len > max_seq_len).
9. Autoregressive sequence generation.
"""

import math
import pytest
import torch

from src.model.attention import CausalSelfAttention
from src.model.block import TransformerBlock
from src.model.config import ModelConfig
from src.model.embeddings import TokenEmbedding
from src.model.feedforward import FeedForward
from src.model.model import TouchlineNanoGPT
from src.model.rope import RotaryEmbedding, apply_rotary_emb, rotate_half


@pytest.fixture
def default_config() -> ModelConfig:
    """Returns the baseline 4.2M parameter TouchlineLM configuration."""
    return ModelConfig(
        vocab_size=4096,
        d_model=256,
        n_layers=4,
        n_heads=4,
        head_dim=64,
        d_ff=1024,
        max_seq_len=1024,
        dropout=0.1,
        bias=False,
        rope_base=10000.0,
        layer_norm_eps=1e-5,
        tie_weights=True,
    )


@pytest.fixture
def small_config() -> ModelConfig:
    """Returns a fast, smaller configuration for lightweight unit tests."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        d_ff=128,
        max_seq_len=64,
        dropout=0.0,
        bias=False,
        rope_base=10000.0,
        layer_norm_eps=1e-5,
        tie_weights=True,
    )


# =========================================================================
# 1. Configuration Validation Tests
# =========================================================================

def test_config_validation(default_config: ModelConfig):
    """Test that valid configuration passes and invalid configurations raise ValueError."""
    assert default_config.d_model == default_config.n_heads * default_config.head_dim

    # Mismatched d_model and n_heads * head_dim
    with pytest.raises(ValueError, match="must equal n_heads"):
        ModelConfig(d_model=256, n_heads=4, head_dim=60)

    # Negative or zero dimensions
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        ModelConfig(vocab_size=0)

    with pytest.raises(ValueError, match="dropout must be in"):
        ModelConfig(dropout=1.5)


def test_config_serialization(tmp_path, default_config: ModelConfig):
    """Test JSON serialization and deserialization of configuration."""
    config_path = tmp_path / "test_config.json"
    default_config.save_json(config_path)

    loaded = ModelConfig.from_json(config_path)
    assert loaded == default_config
    assert loaded.to_dict() == default_config.to_dict()


# =========================================================================
# 2. Rotary Position Embedding (RoPE) Tests
# =========================================================================

def test_rope_properties():
    """Test mathematical invariants of RoPE: norm preservation and cache shape."""
    head_dim = 64
    max_seq_len = 128
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len, base=10000.0)

    dummy = torch.empty(1, 1, dtype=torch.float32)
    cos, sin = rope(dummy, seq_len=32)

    assert cos.shape == (1, 1, 32, head_dim)
    assert sin.shape == (1, 1, 32, head_dim)

    # At position 0, cos should be all 1s and sin should be all 0s
    assert torch.allclose(cos[0, 0, 0, :], torch.ones(head_dim), atol=1e-5)
    assert torch.allclose(sin[0, 0, 0, :], torch.zeros(head_dim), atol=1e-5)

    # Test norm preservation: rotation in complex plane must preserve vector L2 norm
    torch.manual_seed(42)
    x = torch.randn(2, 4, 32, head_dim)
    rotated_q, _ = apply_rotary_emb(x, x, cos, sin)

    norm_before = torch.norm(x, dim=-1)
    norm_after = torch.norm(rotated_q, dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


def test_rope_relative_distance_invariance():
    """
    Test that RoPE inner product <R(q, m), R(k, n)> depends purely on relative distance (m - n).
    Specifically, <R(q, m), R(k, n)> == <R(q, m + delta), R(k, n + delta)>
    """
    head_dim = 64
    seq_len = 50
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=seq_len, base=10000.0)

    torch.manual_seed(123)
    # Vectors q and k
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    dummy = torch.empty(1, 1, dtype=torch.float32)
    cos, sin = rope(dummy, seq_len=seq_len)

    # Compare dot product at positions (m=2, n=5) vs (m=12, n=15) -> delta = 10, relative = -3
    m1, n1 = 2, 5
    m2, n2 = 12, 15

    q_m1, _ = apply_rotary_emb(q, q, cos[:, :, m1 : m1 + 1, :], sin[:, :, m1 : m1 + 1, :])
    _, k_n1 = apply_rotary_emb(k, k, cos[:, :, n1 : n1 + 1, :], sin[:, :, n1 : n1 + 1, :])
    dot1 = (q_m1 * k_n1).sum().item()

    q_m2, _ = apply_rotary_emb(q, q, cos[:, :, m2 : m2 + 1, :], sin[:, :, m2 : m2 + 1, :])
    _, k_n2 = apply_rotary_emb(k, k, cos[:, :, n2 : n2 + 1, :], sin[:, :, n2 : n2 + 1, :])
    dot2 = (q_m2 * k_n2).sum().item()

    assert math.isclose(dot1, dot2, rel_tol=1e-4, abs_tol=1e-4)


# =========================================================================
# 3. Model Output Shapes & Variable Sequence Lengths
# =========================================================================

def test_model_output_shapes(small_config: ModelConfig):
    """Test that model returns expected logits shapes [B, T, vocab_size] for various batch sizes."""
    model = TouchlineNanoGPT(small_config)
    model.eval()

    test_shapes = [(1, 1), (2, 8), (4, 32), (1, small_config.max_seq_len)]
    for b, t in test_shapes:
        idx = torch.randint(0, small_config.vocab_size, (b, t))
        with torch.no_grad():
            logits, loss = model(idx)
        assert logits.shape == (b, t, small_config.vocab_size)
        assert loss is None


def test_context_length_overflow(small_config: ModelConfig):
    """Test that sequences exceeding max_seq_len raise a descriptive ValueError."""
    model = TouchlineNanoGPT(small_config)
    overflow_idx = torch.randint(0, small_config.vocab_size, (1, small_config.max_seq_len + 1))
    with pytest.raises(ValueError, match="exceeds maximum configured context length"):
        model(overflow_idx)


# =========================================================================
# 4. Loss Calculation & Backpropagation
# =========================================================================

def test_loss_calculation_and_gradients(small_config: ModelConfig):
    """Test cross-entropy loss computation and backward pass with finite gradients."""
    model = TouchlineNanoGPT(small_config)
    model.train()

    b, t = 2, 16
    idx = torch.randint(0, small_config.vocab_size, (b, t))
    targets = torch.randint(0, small_config.vocab_size, (b, t))

    logits, loss = model(idx, targets=targets)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    # Backward pass
    loss.backward()

    # Verify gradients exist and are finite for all trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} has non-finite gradient"


# =========================================================================
# 5. Causal Masking Invariance (Strict Future Token Isolation)
# =========================================================================

def test_causal_masking_invariance(small_config: ModelConfig):
    """
    Verify strict causal attention: altering tokens at position t > k must NOT change
    the output logits at positions <= k.
    """
    model = TouchlineNanoGPT(small_config)
    model.eval()

    torch.manual_seed(999)
    seq_len = 16
    cutoff = 7  # Earlier positions: 0 .. 7

    # Sequence A
    tokens_a = torch.randint(0, small_config.vocab_size, (1, seq_len))
    # Sequence B: Identical for positions 0..cutoff, completely different for cutoff+1..seq_len-1
    tokens_b = tokens_a.clone()
    tokens_b[0, cutoff + 1 :] = (tokens_b[0, cutoff + 1 :] + 17) % small_config.vocab_size

    with torch.no_grad():
        logits_a, _ = model(tokens_a)
        logits_b, _ = model(tokens_b)

    # Earlier positions must be IDENTICAL within float precision
    diff_prefix = torch.max(torch.abs(logits_a[:, : cutoff + 1, :] - logits_b[:, : cutoff + 1, :])).item()
    assert diff_prefix < 1e-5, f"Causal leakage detected! Max difference at positions <= {cutoff}: {diff_prefix}"

    # Future positions MUST differ
    diff_suffix = torch.max(torch.abs(logits_a[:, cutoff + 1 :, :] - logits_b[:, cutoff + 1 :, :])).item()
    assert diff_suffix > 1e-3, "Perturbed future tokens did not produce expected differences at future positions"


# =========================================================================
# 6. Weight Tying
# =========================================================================

def test_weight_tying(default_config: ModelConfig):
    """Test that input token embeddings and LM head projection share the exact same tensor."""
    model = TouchlineNanoGPT(default_config)

    # 1. Identity of weight tensors
    assert model.lm_head.weight is model.token_embedding.embedding.weight
    assert model.lm_head.weight.data_ptr() == model.token_embedding.embedding.weight.data_ptr()

    # 2. Gradient synchronization after backward pass
    model.train()
    idx = torch.randint(0, default_config.vocab_size, (2, 8))
    targets = torch.randint(0, default_config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    loss.backward()

    # Since weights are tied, they share the exact same grad tensor
    assert model.lm_head.weight.grad is model.token_embedding.embedding.weight.grad


# =========================================================================
# 7. Parameter Accounting Breakdown (~4.2M Parameters)
# =========================================================================

def test_parameter_count_breakdown(default_config: ModelConfig):
    """
    Verify exact parameter accounting for TouchlineLM baseline:
    - Vocab: 4,096
    - d_model: 256
    - n_layers: 4
    - n_heads: 4 (head_dim: 64)
    - d_ff: 1,024
    - bias: False
    - tie_weights: True

    Calculations:
    - Token Embedding: 4096 * 256 = 1,048,576
    - Per Block:
        * ln_1: 2 * 256 = 512
        * Attention (q, k, v, out): 4 * (256 * 256) = 262,144
        * ln_2: 2 * 256 = 512
        * MLP (w1, w2): (256 * 1024) + (1024 * 256) = 524,288
        * Total per block: 512 + 262,144 + 512 + 524,288 = 787,456
    - 4 Blocks: 4 * 787,456 = 3,149,824
    - Final ln_f: 2 * 256 = 512
    - LM Head: Tied with embedding = 0 additional parameters
    - Total Trainable Parameters = 1,048,576 + 3,149,824 + 512 = 4,198,912 (~4.20M)
    - Non-embedding Parameters = 3,149,824 + 512 = 3,150,336 (~3.15M)
    """
    model = TouchlineNanoGPT(default_config)

    total_params = model.get_num_params(non_embedding=False)
    non_emb_params = model.get_num_params(non_embedding=True)

    expected_total = 4_198_912
    expected_non_emb = 3_150_336

    assert total_params == expected_total, f"Expected {expected_total} total parameters, got {total_params}"
    assert non_emb_params == expected_non_emb, f"Expected {expected_non_emb} non-emb parameters, got {non_emb_params}"

    # Check that parameters are approximately 4.2M
    assert 4.19e6 <= total_params <= 4.21e6


# =========================================================================
# 8. Autoregressive Sequence Generation
# =========================================================================

def test_autoregressive_generation(small_config: ModelConfig):
    """Test that model.generate produces correctly shaped autoregressive tokens."""
    model = TouchlineNanoGPT(small_config)
    model.eval()

    prefix = torch.tensor([[10, 20, 30]])  # B=1, T=3
    max_new_tokens = 8

    generated = model.generate(prefix, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10)
    assert generated.shape == (1, 3 + max_new_tokens)
    assert torch.equal(generated[:, :3], prefix)  # Prefix preserved intact

"""
Stage 1 Tests for Batched Model Forward Support.

Covers:
1. Existing scalar-offset RoPE compatibility.
2. Batched position_ids shape, dtype, and bounds validation.
3. Batched forward pass with no padding.
4. Left-padded batch attention (prefill and incremental decode with dynamic & static caches).
5. Numerical equivalence between individual unpadded requests and batched requests
   (valid-token logits only, atol=1e-5, rtol=1e-4) on both CPU and CUDA.
"""

import pytest
import torch

from src.model.config import ModelConfig
from src.model.model import TouchlineNanoGPT
from src.inference.kv_cache import StaticKVCache


def _build_left_padded_prefill_mask(
    key_valid: torch.Tensor,
) -> torch.Tensor:
    """
    Builds a boolean SDPA mask [B, 1, T, T] where True means attention is allowed.
    Combines causal lower-triangular masking with key_valid [B, T], while keeping
    the diagonal i == j unmasked so left-pad query rows never produce NaNs.
    """
    B, T = key_valid.shape
    device = key_valid.device
    causal = torch.tril(torch.ones((T, T), device=device, dtype=torch.bool))
    mask = causal.unsqueeze(0).unsqueeze(0) & key_valid.unsqueeze(1).unsqueeze(2)
    diag = torch.arange(T, device=device)
    mask[:, :, diag, diag] = True
    return mask


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


def _devices_to_test():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def test_rope_scalar_offset_compatibility(tiny_config: ModelConfig):
    """Existing scalar-offset RoPE behavior must remain unchanged."""
    torch.manual_seed(42)
    model = TouchlineNanoGPT(tiny_config).eval()
    dummy_x = torch.zeros(2, 5, tiny_config.d_model)

    cos_0, sin_0 = model.rope(dummy_x, seq_len=5, offset=0)
    assert cos_0.shape == (1, 1, 5, tiny_config.head_dim)
    assert sin_0.shape == (1, 1, 5, tiny_config.head_dim)

    cos_off, sin_off = model.rope(dummy_x, seq_len=1, offset=3)
    assert cos_off.shape == (1, 1, 1, tiny_config.head_dim)
    assert torch.allclose(cos_off, cos_0[:, :, 3:4, :], atol=1e-7)
    assert torch.allclose(sin_off, sin_0[:, :, 3:4, :], atol=1e-7)


def test_rope_batched_position_ids_shape_and_validation(tiny_config: ModelConfig):
    """Batched position_ids must produce [B, 1, T, head_dim] and validate shape/dtype/bounds."""
    torch.manual_seed(42)
    model = TouchlineNanoGPT(tiny_config).eval()
    B, T = 3, 6
    dummy_x = torch.zeros(B, T, tiny_config.d_model)

    pos_ids = torch.tensor(
        [
            [0, 0, 0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    cos, sin = model.rope(dummy_x, seq_len=T, position_ids=pos_ids)
    assert cos.shape == (B, 1, T, tiny_config.head_dim)
    assert sin.shape == (B, 1, T, tiny_config.head_dim)

    # Row 1 matches standard sequential [0..5]
    cos_seq, sin_seq = model.rope(dummy_x, seq_len=T, offset=0)
    assert torch.allclose(cos[1:2], cos_seq, atol=1e-7)
    assert torch.allclose(sin[1:2], sin_seq, atol=1e-7)

    # Invalid ndim
    with pytest.raises(ValueError, match="2D"):
        model.rope(dummy_x, seq_len=T, position_ids=torch.zeros(T, dtype=torch.long))

    # Invalid dtype
    with pytest.raises(TypeError, match="integer dtype"):
        model.rope(dummy_x, seq_len=T, position_ids=torch.zeros((B, T), dtype=torch.float32))

    # Mismatched seq_len
    with pytest.raises(ValueError, match="must match seq_len"):
        model.rope(dummy_x, seq_len=T, position_ids=torch.zeros((B, T + 1), dtype=torch.long))

    # Out-of-bounds position
    bad_pos = pos_ids.clone()
    bad_pos[0, -1] = tiny_config.max_seq_len
    with pytest.raises(ValueError, match="must be in"):
        model.rope(dummy_x, seq_len=T, position_ids=bad_pos)


@pytest.mark.parametrize("device", _devices_to_test())
def test_batched_forward_no_padding(tiny_config: ModelConfig, device: str):
    """Batched forward pass with uniform-length prompts must match individual unpadded runs."""
    torch.manual_seed(42)
    model = TouchlineNanoGPT(tiny_config).to(device).eval()

    p0 = torch.randint(0, 4092, (1, 10), device=device)
    p1 = torch.randint(0, 4092, (1, 10), device=device)
    batch = torch.cat([p0, p1], dim=0)

    with torch.inference_mode():
        logits_0, _ = model(p0)
        logits_1, _ = model(p1)
        logits_batch, _ = model(batch)

    assert logits_batch.shape == (2, 10, tiny_config.vocab_size)
    assert torch.allclose(logits_batch[0], logits_0[0], atol=1e-5, rtol=1e-4)
    assert torch.allclose(logits_batch[1], logits_1[0], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", _devices_to_test())
@pytest.mark.parametrize("cache_mode", ["dynamic", "static"])
def test_left_padded_batch_numerical_equivalence(
    tiny_config: ModelConfig, device: str, cache_mode: str
):
    """
    Left-padded batched prefill + incremental decode must match individual unpadded
    single-request logits within atol=1e-5, rtol=1e-4, with no NaNs on pad rows.
    """
    torch.manual_seed(123)
    model = TouchlineNanoGPT(tiny_config).to(device).eval()
    pad_id = 4092

    p_short = torch.randint(0, 4092, (1, 5), device=device)
    p_long = torch.randint(0, 4092, (1, 12), device=device)
    pad_len = 12 - 5

    idx_batch = torch.cat(
        [
            torch.cat([torch.full((1, pad_len), pad_id, device=device, dtype=torch.long), p_short], dim=1),
            p_long,
        ],
        dim=0,
    )
    pos_ids = torch.tensor(
        [
            [0] * pad_len + list(range(5)),
            list(range(12)),
        ],
        device=device,
        dtype=torch.long,
    )
    key_valid = torch.tensor(
        [
            [False] * pad_len + [True] * 5,
            [True] * 12,
        ],
        device=device,
        dtype=torch.bool,
    )
    prefill_mask = _build_left_padded_prefill_mask(key_valid)

    with torch.inference_mode():
        # Single-request baselines
        s_logits_0, s_past_0 = model(p_short, use_cache=True)
        l_logits_0, l_past_0 = model(p_long, use_cache=True)

        # Batched prefill
        if cache_mode == "static":
            static_cache = StaticKVCache.from_model(model, batch_size=2, device=device)
            b_logits_0, b_past = model(
                idx_batch,
                past_key_values=static_cache,
                use_cache=True,
                attn_mask=prefill_mask,
                position_ids=pos_ids,
            )
        else:
            b_logits_0, b_past = model(
                idx_batch,
                use_cache=True,
                attn_mask=prefill_mask,
                position_ids=pos_ids,
            )

        assert not torch.isnan(b_logits_0).any(), "NaN detected in batched prefill logits"
        assert torch.allclose(b_logits_0[0, pad_len:, :], s_logits_0[0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(b_logits_0[1, :, :], l_logits_0[0], atol=1e-5, rtol=1e-4)

        # Incremental decode for 4 steps
        for step in range(4):
            tok_s = torch.argmax(s_logits_0[:, -1, :], dim=-1, keepdim=True)
            tok_l = torch.argmax(l_logits_0[:, -1, :], dim=-1, keepdim=True)
            tok_b = torch.cat([tok_s, tok_l], dim=0)

            s_logits_0, s_past_0 = model(tok_s, past_key_values=s_past_0, use_cache=True)
            l_logits_0, l_past_0 = model(tok_l, past_key_values=l_past_0, use_cache=True)

            key_valid = torch.cat([key_valid, torch.ones((2, 1), device=device, dtype=torch.bool)], dim=1)
            decode_mask = key_valid.unsqueeze(1).unsqueeze(2)  # [2, 1, 1, 12 + step + 1]
            decode_pos = torch.tensor([[5 + step], [12 + step]], device=device, dtype=torch.long)

            b_logits_0, b_past = model(
                tok_b,
                past_key_values=b_past,
                use_cache=True,
                attn_mask=decode_mask,
                position_ids=decode_pos,
            )

            assert not torch.isnan(b_logits_0).any(), f"NaN detected at decode step {step}"
            assert torch.allclose(b_logits_0[0, -1, :], s_logits_0[0, -1, :], atol=1e-5, rtol=1e-4)
            assert torch.allclose(b_logits_0[1, -1, :], l_logits_0[0, -1, :], atol=1e-5, rtol=1e-4)

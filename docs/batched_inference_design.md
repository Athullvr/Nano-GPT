# TouchlineLM Phase 3B — Batched Inference and Serving Architecture Design

**Status**: Design & Audit Complete (Pending Review — No Code Modified)  
**Baseline Commit**: `f676924` (68 passing tests)  
**Target Model**: `TouchlineNanoGPT` (4,198,912 parameters, FP32, 4 layers, 4 heads, $d_{\text{model}}=256$, $d_{\text{head}}=64$, `max_seq_len=1024`, `vocab_size=4096`)  
**Target Hardware**: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM) and CPU  

---

## 1. Phase 1 — Architecture Audit

An inspection of the model, tokenizer, KV cache, and inference engine at commit `f676924` reveals that while core tensor operations are batch-dimension-aware, several single-request assumptions prevent multi-request execution today.

### 1.1 Model & Attention Batch Dimension Support
- **Embeddings, Pre-LayerNorm, FeedForward, LM Head** ([`src/model/model.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/model.py), [`src/model/block.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/block.py)): All operate over `[B, T, d_model]` and support arbitrary batch sizes $B \ge 1$.
- **Multi-Head Attention** ([`src/model/attention.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/attention.py#L66-L114)): Reshapes projections to `[B, n_heads, T, head_dim]` and passes them to `F.scaled_dot_product_attention` (SDPA). Batch dimension `dim=0` is preserved throughout.
- **Attention Mask Plumb-Through Gap**:
  - `CausalSelfAttention.forward()` (`attention.py:47`) and `TransformerBlock.forward()` (`block.py:41`) already accept `attn_mask: Optional[torch.Tensor] = None`.
  - However, `TouchlineNanoGPT.forward()` (`model.py:83-89`) **does not accept or forward `attn_mask`**.
  - Without `attn_mask`, `CausalSelfAttention` falls back to `is_causal=True` (when $T > 1$) or `is_causal=False` (when $T = 1$), which forces every token in every batch item to attend to all preceding positions—including any padding tokens.

### 1.2 Rotary Position Embeddings (RoPE) Limitations
- **Scalar Offset Assumption** ([`src/model/rope.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/rope.py#L117-L138)):
  `RotaryEmbedding.forward(x, seq_len, offset=0)` accepts only a **single integer `offset`** and returns `cos, sin` shaped `[1, 1, seq_len, head_dim]`, broadcasting identically across all $B$ batch items.
- **Impact on Heterogeneous Prompts**:
  If Request A has 12 tokens and Request B has 40 tokens, left-padding Request A with 28 pad tokens shifts Request A's first real token to index 28. With a scalar `offset=0`, Request A's tokens would receive RoPE angles `[28..39]` instead of `[0..11]`, corrupting its positional representations and producing different logits than single-request execution.
  Supporting `position_ids: Optional[torch.Tensor] = None` of shape `[B, T]` is therefore mandatory for exact numerical equivalence on heterogeneous batches.

### 1.3 KV Cache State Isolation (Dynamic & Static)
- **Dynamic KV Cache** (`List[Tuple[torch.Tensor, torch.Tensor]]`):
  Each layer stores `[B, n_heads, seq_len, head_dim]`. Concatenation (`torch.cat((k_past, k), dim=2)`) keeps batch rows isolated along `dim=0`, but **all $B$ requests must share a uniform `seq_len`**.
- **Static Preallocated KV Cache** ([`src/inference/kv_cache.py`](file:///home/athulvr/Projects/Nano Gpt/src/inference/kv_cache.py#L33-L134)):
  `StaticKVCache` preallocates `[B, n_heads, max_seq_len, head_dim]` and supports `batch_size > 1` in its constructor. However:
  - `self.current_seq_len` is a **single scalar integer** shared across the entire batch.
  - `update()` writes `k` into `narrow(2, pos, T_new)` for all $B$ rows simultaneously, and `advance(T_new)` increments `current_seq_len` globally.
  - This is compatible with **left-padded batched generation** (where all $B$ rows advance in lockstep from `P_max`), but incompatible with unaligned/ragged lengths or continuous batching slot eviction.

### 1.4 Shared Sequence Length & Engine Limitations
- **Hardcoded $B = 1$ in `TouchlineInferenceEngine.generate()`** ([`src/inference/engine.py`](file:///home/athulvr/Projects/Nano Gpt/src/inference/engine.py#L240-L365)):
  - Wraps tokens as `torch.tensor([prompt_tokens])` (line 240).
  - Hardcodes `StaticKVCache.from_model(self.model, batch_size=1)` (line 272).
  - Calls `.item()` on `idx_next` (lines 263, 281, 297, 315, 332), which raises `RuntimeError` if $B > 1$.
  - Decodes only `idx[0].tolist()` (line 345) and tracks a single scalar `stopped_on_eos` and `new_tokens_count`.
- **No Per-Request Completion Tracking**:
  There is no `finished: torch.Tensor` mask (`[B]`), no mechanism to freeze finished rows while unfinished rows continue decoding, and no per-request `max_new_tokens`, `min_tokens`, or sampling settings.

### 1.5 Tokenizer Limitations
- **No Dedicated Pad Token** ([`src/tokenizer/bpe_tokenizer.py`](file:///home/athulvr/Projects/Nano Gpt/src/tokenizer/bpe_tokenizer.py#L23-L28)):
  The 4,096-token vocabulary defines 4 special tokens (`<|endoftext|>: 4092`, `<|narrative|>: 4093`, `<|player_dossier|>: 4094`, `<|tactical_qa|>: 4095`) and **no `<|pad|>` token**.
  Because model weights and the `4096 x 256` embedding table cannot be modified, padding **must reuse `<|endoftext|>` (`4092`)** as `pad_token_id` and rely on `attention_mask` + `position_ids` to neutralize pad positions.
- **No Batch Encoding API**: `encode()` only processes one string at a time.

---

## 2. Phase 2 — Experimental Design: Minimal Batched Inference Interface

### 2.1 Proposed API & Data Structures
Preserve `TouchlineInferenceEngine.generate(...)` unchanged for single-request backward compatibility, and introduce `generate_batch(...)`:

```python
@dataclass
class BatchRequest:
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
    request_index: int
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    new_tokens: int
    ttft_seconds: float
    total_latency_seconds: float
    stopped_on_eos: bool
    status: str  # "ok" or "error"
    error: Optional[str] = None

@dataclass
class BatchGenerationOutput:
    items: List[BatchGenerationItem]
    batch_size: int
    prefill_latency_seconds: float
    decode_latency_seconds: float
    total_latency_seconds: float
    aggregate_tokens_per_second: float
    per_request_tokens_per_second: float
    peak_allocated_mb: Optional[float]
    peak_reserved_mb: Optional[float]
    successful_requests: int
    failed_requests: int
    cache_type: str
    device: str
```

Method signature on `TouchlineInferenceEngine`:
```python
def generate_batch(
    self,
    requests: Sequence[Union[str, BatchRequest]],
    cache_type: str = "dynamic",      # "dynamic", "static", or "none"
    max_batch_size: Optional[int] = None,
) -> BatchGenerationOutput:
```

### 2.2 Left-Padding, `position_ids`, and Attention Masking

1. **Left-Padding Alignment**:
   Given $B$ tokenized prompts with lengths $[L_0, L_1, \dots, L_{B-1}]$ and $P_{\max} = \max_b L_b$:
   - Each prompt $b$ is left-padded with `pad_len_b = P_max - L_b` copies of `pad_token_id` (`4092`).
   - All real prompts end at column `P_max - 1`, so prefill logits `logits[:, -1, :]` correspond directly to each request's true final prompt token.
   - Both `DynamicKVCache` and `StaticKVCache` advance uniformly at `P_max + step` without ragged indexing.

2. **Per-Request `position_ids` (`[B, T]`)**:
   - **Prefill**: For row $b$, `position_ids[b, :pad_len_b] = 0`, and `position_ids[b, pad_len_b:] = [0, 1, ..., L_b - 1]`.
   - **Decode Step $s$ ($s \ge 0$)**: `position_ids[b, 0] = L_b + s` (clamped or halted once finished).
   - `RotaryEmbedding.forward(x, seq_len, offset=0, position_ids=None)`:
     When `position_ids` is provided (`[B, T]`), gather `cos_cached[position_ids]` $\rightarrow$ `[B, 1, T, head_dim]`, preserving exact positional equivalence with single-request execution.

3. **Attention Mask (`[B, 1, T_q, T_k]`)**:
   Maintain `key_padding_mask` (`[B, P_max + s]`, `True` = valid token, `False` = left-pad).
   - **Prefill ($T_q = P_{\max}$)**:
     Combine causal lower-triangular mask (`[1, 1, P_max, P_max]`) with `key_padding_mask[:, None, None, :]`.
     *Numeric safety*: For a left-pad query row $i < \text{pad\_len}_b$, all keys $j \le i$ are pad tokens. To prevent `softmax([-inf, ..., -inf]) = NaN` in pad rows, unmask the diagonal `i == j` for pad positions (their outputs are never read).
   - **Decode ($T_q = 1$)**:
     Pass `attn_mask = key_padding_mask[:, None, None, :]` (`[B, 1, 1, P_max + s]`) with `is_causal=False`. Each active request attends to all of its own real prompt + generated tokens and zero left-padding tokens.

### 2.3 Per-Request Lifecycle, Sampling & Error Handling
- **Independent Completion**: Track `finished = torch.zeros(B, dtype=torch.bool)` and `gen_counts = torch.zeros(B, dtype=torch.long)`. At each step:
  - Record `ttft_seconds` upon completing prefill.
  - Sample per-request next token using each request's `temperature`, `top_k`, `top_p`, and optional per-request `torch.Generator`.
  - If `finished[b]` is already `True`, force `next_token[b] = pad_token_id` and do not increment `gen_counts[b]`.
  - Otherwise, increment `gen_counts[b]`, append token, and set `finished[b] = True` if `(stop_at_eos and token in stop_ids[b] and gen_counts[b] >= req.min_tokens)` or `gen_counts[b] >= req.max_new_tokens`.
  - Exit decode loop immediately when `finished.all()` is `True`.
- **Context-Limit & Memory Validation**:
  - **Pre-flight Validation**: Check `len(prompt_tokens) + req.max_new_tokens <= self.model.config.max_seq_len` (and `len(prompt_tokens) > 0`). Oversized requests are marked `status="error"` (`failed_requests += 1`) with an explicit context-limit message, or rejected cleanly before GPU allocation if all fail.
  - **OOM Guard**: Wrap batch execution in a `try ... except torch.cuda.OutOfMemoryError` handler that releases intermediate tensors, calls `torch.cuda.empty_cache()`, and reports failed requests cleanly.

---

## 3. Identified Correctness Risks

| Risk | Cause | Mitigation |
| :--- | :--- | :--- |
| **RoPE Position Shift** | Scalar `offset` shifts left-padded prompts by `pad_len_b`. | Add `position_ids: [B, T]` support to `RotaryEmbedding` and validate `batched_logits == single_logits` within `atol=1e-5`. |
| **SDPA `NaN` on Pad Queries** | Left-pad rows have all keys masked if causal + key mask both mask `0..pad_len-1`. | Keep diagonal `i == j` unmasked in prefill attention mask so softmax has at least one finite entry per row. |
| **Pad Token vs. EOS Collision** | Reusing `4092` (`<|endoftext|>`) as pad token. | Never rely on token value `4092` to infer padding; always track explicit `key_padding_mask` and `prompt_lengths`. |
| **Straggler Padding Tax** | Short finished requests still occupy batch slots until longest finishes. | Freeze `gen_counts[b]` and mask `key_padding_mask` updates; measure straggler overhead explicitly in benchmarks. |
| **Static Cache Capacity Overrun** | `P_max + max_new_tokens > max_seq_len` for heterogeneous batch. | Validate `P_max + max(req.max_new_tokens for active) <= max_seq_len` before prefill. |

---

## 4. Phase 3 — Benchmark Design

### 4.1 Experiment Matrix
Evaluate batch sizes $B \in \{1, 2, 4, 8\}$ on CUDA (RTX 4050 6 GB) and CPU across:
1. **Homogeneous Workload**: Prompts ~64 tokens, `max_new_tokens = 128`, `temperature = 0.0` (greedy, disabling early EOS or using `min_tokens=128` for controlled length). Measures pure batching compute/memory scaling.
2. **Heterogeneous Workload**: Mixed prompt lengths (e.g., 24, 64, 128, 256 tokens) and mixed `max_new_tokens` (e.g., 50 to 180) with `stop_at_eos=True`. Measures left-padding overhead and straggler inefficiency.
3. **Cache Modes**: Compare `dynamic` vs. `static` cache across all $B \in \{1, 2, 4, 8\}$ (and `none` at $B=1, 4$ for baseline reference).
4. **Error/Boundary Suite**: Test oversized prompt (`> 1024` tokens) and high-batch memory limits to verify `successful_requests` / `failed_requests` reporting.

### 4.2 Measurement Methodology
- Run under `torch.inference_mode()` with 2 warmup iterations and 5 measured repetitions per configuration.
- Call `torch.cuda.empty_cache()` and `torch.cuda.reset_peak_memory_stats()` before each repetition; use `torch.cuda.synchronize()` around prefill and decode timers separately.
- **Metrics Collected**:
  - **Prefill Latency** (ms) and **TTFT** (ms per request).
  - **Decode Latency** (ms per batch step) and **Inter-Token Latency (ITL)** (ms per active request token).
  - **Aggregate Generation Throughput**: $\frac{\sum_{b=1}^B \text{new\_tokens}_b}{T_{\text{total}}}$ (tok/s).
  - **Per-Request Throughput**: $\frac{1}{B} \sum_{b=1}^B \frac{\text{new\_tokens}_b}{T_{\text{req}, b}}$ (tok/s).
  - **Peak Allocated VRAM** (`max_memory_allocated` in MB) and **Peak Reserved VRAM** (`max_memory_reserved` in MB).
  - **Request Outcomes**: `successful_requests` and `failed_requests`.

---

## 5. Proposed Implementation Stages

1. **Stage 1 — Model & RoPE Batch Support + Unit Tests**:
   - Add `position_ids` to `RotaryEmbedding` and `TouchlineNanoGPT.forward()`.
   - Add `attn_mask` passthrough to `TouchlineNanoGPT.forward()`.
   - Write unit tests verifying `B > 1` left-padded logits match `B = 1` unpadded logits (`atol=1e-5`) for both dynamic and static caches.
2. **Stage 2 — Batched Engine & Request Lifecycle**:
   - Add `BatchRequest`, `BatchGenerationItem`, `BatchGenerationOutput`, and `generate_batch()` in `src/inference/engine.py`.
   - Implement left-padding, per-request `finished` tracking, per-request sampling, and context/OOM error handling.
   - Add integration tests for homogeneous batches, heterogeneous prompt lengths, independent EOS stopping, seed reproducibility, and error isolation.
3. **Stage 3 — Batched Benchmark Script & Empirical Evaluation**:
   - Create `scripts/benchmark_batched_inference.py` covering $B \in \{1, 2, 4, 8\}$ for homogeneous and heterogeneous workloads.
   - Document empirical results and scaling bottlenecks (compute vs. memory-bandwidth vs. straggler tax) to inform future continuous batching work.

---

## 6. Files Likely to Require Modification

| File | Role of Change |
| :--- | :--- |
| [`src/model/rope.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/rope.py) | Support optional `position_ids: [B, T]` in `RotaryEmbedding.forward()` and `[B, 1, T, head_dim]` broadcasting in `apply_rotary_emb()`. |
| [`src/model/model.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/model.py) | Accept and forward optional `attn_mask` and `position_ids` in `TouchlineNanoGPT.forward()`. |
| [`src/model/attention.py`](file:///home/athulvr/Projects/Nano Gpt/src/model/attention.py) | Ensure `attn_mask` shape `[B, 1, T_q, T_k]` works cleanly alongside KV cache (already mostly compatible). |
| [`src/tokenizer/bpe_tokenizer.py`](file:///home/athulvr/Projects/Nano Gpt/src/tokenizer/bpe_tokenizer.py) | Optionally add `pad_token_id` property (`4092`) and `encode_batch()` helper. |
| [`src/inference/engine.py`](file:///home/athulvr/Projects/Nano Gpt/src/inference/engine.py) | Add `BatchRequest`, `BatchGenerationItem`, `BatchGenerationOutput`, and `generate_batch()` while preserving `generate()`. |
| `tests/test_batched_inference.py` *(new)* | Pre-implementation and integration test suite for batched logits, padding/masking, EOS independence, and error handling. |
| `scripts/benchmark_batched_inference.py` *(new)* | Reproducible benchmark script for $B \in \{1, 2, 4, 8\}$ across dynamic and static caches. |
